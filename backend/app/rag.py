import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .ingest import chunk_text, doc_hash
from .settings import settings

logger = logging.getLogger(__name__)


def _hash_to_uuid(hex_str: str) -> str:
    """Convert a hex string to a valid UUID for Qdrant point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, hex_str))


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """Hash-based deterministic pseudo-embedder for offline use."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        """Return a deterministic unit-norm vector for *text*."""
        h = hashlib.sha1(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big") % (2**32 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype("float32")
        vec /= np.linalg.norm(vec) + 1e-9
        return vec


# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Simple in-memory cosine-similarity vector store."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict[str, Any]] = []
        self._hashes: set[str] = set()

    def upsert(
        self,
        vectors: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Insert vectors, skipping duplicates. Return count of new items."""
        added = 0
        for vec, meta in zip(vectors, metadatas):
            h = meta.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(vec.astype("float32"))
            self.meta.append(meta)
            if h:
                self._hashes.add(h)
            added += 1
        return added

    def search(
        self, query: np.ndarray, k: int = 4
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top-k (score, metadata) pairs by cosine similarity."""
        if not self.vecs:
            return []
        matrix = np.vstack(self.vecs)
        q = query.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1) * (
            np.linalg.norm(q) + 1e-9
        )
        sims = (matrix @ q.T).ravel() / (norms + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

    @property
    def count(self) -> int:
        return len(self.vecs)


class QdrantStore:
    """Qdrant-backed vector store."""

    def __init__(self, collection: str, dim: int = 384) -> None:
        self.client = QdrantClient(
            url="http://qdrant:6333", timeout=10.0
        )
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=self.dim, distance=qm.Distance.COSINE
                ),
            )

    def upsert(
        self,
        vectors: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
    ) -> int:
        """Upsert vectors into Qdrant. Return count of items sent."""
        points = [
            qm.PointStruct(
                id=_hash_to_uuid(meta.get("id", str(i))),
                vector=vec.tolist(),
                payload=meta,
            )
            for i, (vec, meta) in enumerate(zip(vectors, metadatas))
        ]
        self.client.upsert(
            collection_name=self.collection, points=points
        )
        return len(points)

    def search(
        self, query: np.ndarray, k: int = 4
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top-k (score, payload) pairs."""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True,
        )
        return [
            (float(r.score), dict(r.payload)) for r in results
        ]

    @property
    def count(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.points_count


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a company policy assistant. Answer the user's question using \
ONLY the context chunks provided below.

Rules:
1. Cite every claim using the format: \
"According to <Document Title> (Section: <Section Name>), ..."
2. If the context does not contain enough information, say: \
"I don't have enough information to answer that."
3. Be concise and factual. Do not invent information beyond the context.
4. When multiple documents are relevant, cite all of them.
"""


def _build_context_block(contexts: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM."""
    parts: List[str] = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title", "Unknown")
        section = ctx.get("section", "General")
        text = ctx.get("text", "")
        parts.append(
            f"[{i}] Document: {title}\n"
            f"    Section: {section}\n"
            f"    Content: {text}"
        )
    return "\n\n".join(parts)


class StubLLM:
    """Deterministic stub LLM for offline development and testing."""

    def generate(
        self, query: str, contexts: List[Dict[str, Any]]
    ) -> str:
        lines = ["Answer (stub): Based on the following sources:"]
        for ctx in contexts:
            section = ctx.get("section") or "General"
            title = ctx.get("title", "Unknown")
            lines.append(f"- {title} (Section: {section})")
        lines.append("\nSummary:")
        joined = " ".join(c.get("text", "") for c in contexts)
        lines.append(
            joined[:600] + ("..." if len(joined) > 600 else "")
        )
        return "\n".join(lines)


class OpenRouterLLM:
    """LLM provider using OpenRouter (OpenAI-compatible API)."""

    def __init__(
        self, api_key: str, model: str = "openai/gpt-4o-mini"
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def generate(
        self, query: str, contexts: List[Dict[str, Any]]
    ) -> str:
        context_block = _build_context_block(contexts)
        user_message = (
            f"Context:\n{context_block}\n\nQuestion: {query}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class Metrics:
    """Tracks retrieval and generation latencies."""

    def __init__(self) -> None:
        self.t_retrieval: List[float] = []
        self.t_generation: List[float] = []

    def add_retrieval(self, ms: float) -> None:
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float) -> None:
        self.t_generation.append(ms)

    def summary(self) -> Dict[str, float]:
        avg_r = (
            sum(self.t_retrieval) / len(self.t_retrieval)
            if self.t_retrieval
            else 0.0
        )
        avg_g = (
            sum(self.t_generation) / len(self.t_generation)
            if self.t_generation
            else 0.0
        )
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """Orchestrates ingestion, retrieval, and generation."""

    def __init__(self) -> None:
        self.embedder = LocalEmbedder(dim=384)

        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(
                    collection=settings.collection_name, dim=384
                )
                logger.info("Using Qdrant vector store")
            except Exception:
                logger.warning(
                    "Qdrant unavailable, falling back to in-memory store"
                )
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)
            logger.info("Using in-memory vector store")

        if (
            settings.llm_provider == "openrouter"
            and settings.openrouter_api_key
        ):
            self.llm = OpenRouterLLM(
                api_key=settings.openrouter_api_key,
                model=settings.llm_model,
            )
            self.llm_name = f"openrouter:{settings.llm_model}"
            logger.info("Using OpenRouter LLM: %s", settings.llm_model)
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"
            logger.info("Using stub LLM")

        self.metrics = Metrics()
        self._doc_titles: set[str] = set()

    def ingest_chunks(
        self, chunks: List[Dict[str, str]]
    ) -> Tuple[int, int]:
        """Embed and store chunks. Returns (total_docs, total_chunks)."""
        vectors: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []

        for chunk in chunks:
            text = chunk["text"]
            h = doc_hash(text)
            meta: Dict[str, Any] = {
                "id": h,
                "hash": h,
                "title": chunk["title"],
                "section": chunk.get("section"),
                "text": text,
            }
            vec = self.embedder.embed(text)
            vectors.append(vec)
            metas.append(meta)
            self._doc_titles.add(chunk["title"])

        self.store.upsert(vectors, metas)
        total_chunks = self.store.count
        logger.info(
            "Ingested batch of %d chunks (store total: %d), %d docs",
            len(metas),
            total_chunks,
            len(self._doc_titles),
        )
        return len(self._doc_titles), total_chunks

    def retrieve(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Any]]:
        """Embed query and return top-k matching chunks."""
        t0 = time.time()
        query_vec = self.embedder.embed(query)
        results = self.store.search(query_vec, k=k)
        elapsed_ms = (time.time() - t0) * 1000.0
        self.metrics.add_retrieval(elapsed_ms)
        logger.info(
            "Retrieved %d chunks in %.1f ms for: %s",
            len(results),
            elapsed_ms,
            query[:80],
        )
        return [meta for _score, meta in results]

    def generate(
        self, query: str, contexts: List[Dict[str, Any]]
    ) -> str:
        """Generate an answer from the LLM using retrieved contexts."""
        t0 = time.time()
        answer = self.llm.generate(query, contexts)
        elapsed_ms = (time.time() - t0) * 1000.0
        self.metrics.add_generation(elapsed_ms)
        logger.info("Generated answer in %.1f ms", elapsed_ms)
        return answer

    def stats(self) -> Dict[str, Any]:
        """Return current engine statistics."""
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self.store.count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **self.metrics.summary(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_chunks_from_docs(
    docs: List[Dict[str, str]], chunk_size: int, overlap: int
) -> List[Dict[str, str]]:
    """Chunk each document section and propagate metadata."""
    out: List[Dict[str, str]] = []
    for doc in docs:
        for text in chunk_text(doc["text"], chunk_size, overlap):
            out.append(
                {
                    "title": doc["title"],
                    "section": doc["section"],
                    "text": text,
                }
            )
    return out
