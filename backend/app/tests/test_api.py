"""Integration tests for the 4 API endpoints."""

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /api/ingest
# ---------------------------------------------------------------------------


def test_ingest_returns_doc_and_chunk_counts(client: TestClient) -> None:
    resp = client.post("/api/ingest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["indexed_docs"] > 0
    assert data["indexed_chunks"] > 0


def test_ingest_is_idempotent(client: TestClient) -> None:
    """Re-ingesting should return the same counts (deduplication)."""
    first = client.post("/api/ingest").json()
    second = client.post("/api/ingest").json()
    assert first["indexed_docs"] == second["indexed_docs"]
    assert first["indexed_chunks"] == second["indexed_chunks"]


# ---------------------------------------------------------------------------
# /api/ask
# ---------------------------------------------------------------------------


def test_ask_returns_answer_with_citations(client: TestClient) -> None:
    # Ensure docs are ingested first
    client.post("/api/ingest")
    resp = client.post(
        "/api/ask",
        json={"query": "What is the refund window for small appliances?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "citations" in data and len(data["citations"]) > 0
    assert "chunks" in data and len(data["chunks"]) > 0
    assert "metrics" in data


def test_ask_citations_have_title_and_section(client: TestClient) -> None:
    client.post("/api/ingest")
    resp = client.post(
        "/api/ask",
        json={"query": "What is the warranty policy?"},
    )
    data = resp.json()
    for citation in data["citations"]:
        assert "title" in citation
        assert isinstance(citation["title"], str)


def test_ask_chunks_have_required_fields(client: TestClient) -> None:
    client.post("/api/ingest")
    resp = client.post(
        "/api/ask",
        json={"query": "Tell me about shipping"},
    )
    data = resp.json()
    for chunk in data["chunks"]:
        assert "title" in chunk
        assert "section" in chunk
        assert "text" in chunk
        assert len(chunk["text"]) > 0


def test_ask_empty_query_returns_400(client: TestClient) -> None:
    resp = client.post("/api/ask", json={"query": ""})
    assert resp.status_code == 400


def test_ask_whitespace_query_returns_400(client: TestClient) -> None:
    resp = client.post("/api/ask", json={"query": "   "})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_all_fields(client: TestClient) -> None:
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_docs" in data
    assert "total_chunks" in data
    assert "avg_retrieval_latency_ms" in data
    assert "avg_generation_latency_ms" in data
    assert "embedding_model" in data
    assert "llm_model" in data


def test_metrics_reflect_state_after_ingest(client: TestClient) -> None:
    client.post("/api/ingest")
    data = client.get("/api/metrics").json()
    assert data["total_docs"] > 0
    assert data["total_chunks"] > 0
    assert data["embedding_model"] == "local-384"
    assert data["llm_model"] == "stub"
