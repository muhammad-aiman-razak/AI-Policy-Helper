# AI Policy & Product Helper

A local-first RAG (Retrieval-Augmented Generation) system that answers company policy questions with citations. Built with **FastAPI**, **Next.js**, and **Qdrant**.

## Quick Start

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Run all services (Qdrant + Backend + Frontend)
docker compose up --build

# 3. Open the UI
# http://localhost:3000
```

Ingest docs from the **Admin** panel, then ask questions in the **Chat**.

## Architecture

```
User ──► Next.js (port 3000)
              │
              ▼
         FastAPI (port 8000)
           ├─ /api/health     → liveness probe
           ├─ /api/metrics    → counters + latencies + model info
           ├─ /api/ingest     → load docs → section split → chunk → embed → store
           └─ /api/ask        → embed query → retrieve top-k → LLM generate → response
              │
              ▼
         Qdrant (port 6333)
           └─ Collection: "policy_helper" (cosine similarity, 384-dim)
```

### Data Flow: Ask Query

1. User submits query via Chat UI
2. Frontend `POST /api/ask { query, k }`
3. Backend embeds query using `LocalEmbedder` (hash-based, 384-dim)
4. Vector search in Qdrant returns top-k chunks by cosine similarity
5. Chunks + query are sent to the LLM with a system prompt requiring citations
6. LLM generates answer grounded in the context, citing sources by title and section
7. Response: `{ answer, citations[], chunks[], metrics }`
8. Frontend renders answer + clickable citation badges + expandable chunk text

### Ingestion Pipeline

```
/data/*.md → load_documents() → _md_sections() → chunk_text() → LocalEmbedder.embed() → Qdrant.upsert()
```

Each chunk carries metadata: `title` (filename stem), `section` (heading text), `text` (content), and `doc_hash` (SHA-256 for deduplication).

## Project Layout

```
├─ backend/
│  └─ app/
│     ├─ main.py        # FastAPI endpoints (thin — delegates to RAGEngine)
│     ├─ rag.py         # RAGEngine, embedder, vector stores, LLM providers
│     ├─ ingest.py      # Document loading, markdown section splitting, chunking
│     ├─ models.py      # Pydantic request/response models
│     ├─ settings.py    # Environment-based configuration
│     └─ tests/
│        ├─ test_api.py   # Integration tests (4 endpoints)
│        └─ test_unit.py  # Unit tests (chunking, embedding, store, metrics)
├─ frontend/
│  ├─ app/              # Next.js App Router (page, layout, globals.css)
│  ├─ components/       # Chat.tsx, AdminPanel.tsx
│  └─ lib/api.ts        # Typed API client (no fetch in components)
├─ data/                # 6 sample policy documents (.md)
├─ docker-compose.yml   # Qdrant + Backend + Frontend (+ optional Ollama)
└─ .env.example         # All configuration variables with defaults
```

## API Reference

### `GET /api/health`
Returns `{ "status": "ok" }`. No dependencies.

### `POST /api/ingest`
Reads all `.md`/`.txt` files from `/app/data`, chunks them, and stores in the vector DB.

**Response:** `{ "indexed_docs": 6, "indexed_chunks": 10 }`

Idempotent — re-ingesting returns the same counts (deduplication via content hashing).

### `POST /api/ask`
**Request:** `{ "query": "string", "k": 8 }`

**Response:**
```json
{
  "query": "Can a customer return a damaged blender after 20 days?",
  "answer": "According to Returns_and_Refunds (Section: Refund Windows), ...",
  "citations": [{ "title": "Returns_and_Refunds", "section": "Refund Windows" }],
  "chunks": [{ "title": "...", "section": "...", "text": "..." }],
  "metrics": { "retrieval_ms": 10.5, "generation_ms": 2400.0 }
}
```

### `GET /api/metrics`
```json
{
  "total_docs": 6,
  "total_chunks": 10,
  "avg_retrieval_latency_ms": 10.07,
  "avg_generation_latency_ms": 2400.0,
  "embedding_model": "local-384",
  "llm_model": "ollama:llama3.2"
}
```

## Switching LLM Providers

| Provider | `.env` setting | Notes |
|----------|---------------|-------|
| **Stub** (default) | `LLM_PROVIDER=stub` | Deterministic, offline. For development and testing. |
| **OpenRouter** | `LLM_PROVIDER=openrouter` | Set `OPENROUTER_API_KEY`. Defaults to GPT-4o-mini. |
| **Ollama** (local) | `LLM_PROVIDER=ollama` | Free, no API key. Requires pulling a model first. |

### Using Ollama (recommended for demo)

```bash
# Start Ollama alongside other services
docker compose --profile ollama up -d

# Pull a model (first time only)
docker compose exec ollama ollama pull llama3.2

# Update .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# Restart backend to pick up the change
docker compose up -d backend
```

If you have Ollama installed locally, set `OLLAMA_HOST=http://host.docker.internal:11434` instead to use your host GPU.

## Tests

```bash
# Run all 36 tests inside the backend container
docker compose run --rm backend pytest -q
```

```
....................................
36 passed in 2.89s
```

**Unit tests** (22): chunking, section extraction, hashing, embedder dimensions/determinism, InMemoryStore operations, metrics averaging.

**Integration tests** (14): all 4 API endpoints, deduplication, empty query validation, citation structure, metrics state reflection.

Tests use `VECTOR_STORE=memory` and `LLM_PROVIDER=stub` so they run without Qdrant or API keys.

## Trade-offs & Design Decisions

### Hash-based embeddings vs. semantic embeddings
The `LocalEmbedder` uses SHA-1 hashing to generate deterministic pseudo-random vectors. This means **retrieval is not semantically meaningful** — it works for this small corpus (10 chunks) by retrieving a high proportion of chunks (k=8 out of 10). For a production system, I would use `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) for real semantic search.

**Why this trade-off:** The starter pack was designed to work fully offline without downloading model files. The hash-based approach keeps the Docker image small and boot time fast. With only 6 documents, high-k retrieval compensates for the lack of semantic understanding.

### High default k (k=8)
With 10 total chunks and non-semantic embeddings, k=8 retrieves 80% of the corpus. This ensures both acceptance queries get the right documents cited. The LLM filters what's relevant from the context.

**Trade-off:** For a larger corpus, this would send too much context to the LLM. The fix would be real embeddings + lower k + optional MMR reranking for diversity.

### Heading-only section filtering
Markdown sections with only a heading and no body text (e.g., `# Delivery & Shipping` with no content underneath) are filtered out during ingestion. This reduces the chunk count from 12 to 10 and avoids wasting retrieval slots on empty chunks.

### Ollama as an LLM alternative
The provided OpenRouter API key was expired. Rather than depending on an external service, I implemented `OllamaLLM` using the OpenAI-compatible `/v1` endpoint. This reuses the same `openai` SDK and system prompt as `OpenRouterLLM` (DRY). Ollama runs as an optional Docker Compose profile so it doesn't affect the default boot.

### Per-citation chunk expansion (AC5)
Each citation badge is individually clickable to expand/collapse its source chunk. This differs from the starter's approach of showing all chunks together in a `<details>` element. The per-citation approach directly maps to AC5: "clicking a citation chip expands to show the underlying chunk text."

## What I'd Ship Next

1. **Semantic embeddings** — Replace `LocalEmbedder` with `sentence-transformers` for real similarity search, then lower k to 4-6.
2. **Streaming responses (SSE)** — Progressive display as the LLM generates, reducing perceived latency.
3. **PDPA masking** — Redact IC numbers and addresses from queries and responses per compliance requirements.
4. **Evaluation script** — Batch test queries with expected citations to catch regressions.
5. **MMR reranking** — Maximal Marginal Relevance to diversify retrieved chunks and avoid returning near-duplicate sections.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `stub` | LLM backend: `stub`, `openrouter`, `ollama` |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Model name for OpenRouter |
| `OPENROUTER_API_KEY` | — | API key for OpenRouter |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `EMBEDDING_MODEL` | `local-384` | Embedding model identifier |
| `VECTOR_STORE` | `qdrant` | Vector store: `qdrant` or `memory` |
| `COLLECTION_NAME` | `policy_helper` | Qdrant collection name |
| `CHUNK_SIZE` | `700` | Words per chunk |
| `CHUNK_OVERLAP` | `80` | Word overlap between chunks |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend API base URL |
