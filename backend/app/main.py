import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .ingest import load_documents
from .models import (
    AskRequest,
    AskResponse,
    Chunk,
    Citation,
    IngestResponse,
    MetricsResponse,
)
from .rag import RAGEngine, build_chunks_from_docs
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Policy & Product Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


def _filter_cited(answer: str, citations: list[Citation]) -> list[Citation]:
    """Keep only citations whose title appears in the LLM answer text.

    The system prompt enforces "According to <Title> (Section: …)" format,
    so checking for the title string in the answer is reliable. We check
    both the raw title (underscores) and display title (spaces) since the
    LLM context now uses human-readable titles.
    Returns empty list when no titles match (e.g. "I don't have enough
    information" with no actual citations in the text).
    """
    cited = [
        c for c in citations if c.title in answer or c.title.replace("_", " ") in answer
    ]
    return cited


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns ok."""
    return {"status": "ok"}


@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return current engine metrics and model info."""
    return MetricsResponse(**engine.stats())


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    """Load documents from data directory, chunk, embed, and store."""
    try:
        t0 = time.time()
        docs = load_documents(settings.data_dir)
        chunks = build_chunks_from_docs(
            docs, settings.chunk_size, settings.chunk_overlap
        )
        total_docs, total_chunks = engine.ingest_chunks(chunks)
        elapsed_ms = (time.time() - t0) * 1000.0
        logger.info(
            "Ingestion completed: %d docs, %d chunks in %.0f ms",
            total_docs,
            total_chunks,
            elapsed_ms,
        )
        return IngestResponse(
            indexed_docs=total_docs,
            indexed_chunks=total_chunks,
        )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Retrieve relevant chunks and generate an answer with citations."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        ctx = engine.retrieve(req.query, k=req.k or 4)
        answer = engine.generate(req.query, ctx)

        # Deduplicate citations by (title, section), then keep only
        # those the LLM actually referenced in its answer.
        seen: set[tuple[str | None, str | None]] = set()
        all_citations: list[Citation] = []
        for chunk in ctx:
            key = (chunk.get("title"), chunk.get("section"))
            if key not in seen:
                seen.add(key)
                all_citations.append(
                    Citation(
                        title=chunk.get("title", "Unknown"),
                        section=chunk.get("section"),
                    )
                )

        citations = _filter_cited(answer, all_citations)

        chunks = [
            Chunk(
                title=c.get("title", "Unknown"),
                section=c.get("section"),
                text=c.get("text", ""),
            )
            for c in ctx
        ]

        stats = engine.stats()
        return AskResponse(
            query=req.query,
            answer=answer,
            citations=citations,
            chunks=chunks,
            metrics={
                "retrieval_ms": stats["avg_retrieval_latency_ms"],
                "generation_ms": stats["avg_generation_latency_ms"],
            },
        )
    except Exception as exc:
        logger.error(
            "Ask failed for query '%s': %s",
            req.query[:80],
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc
