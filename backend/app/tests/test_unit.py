"""Unit tests for ingest, rag, and metrics components."""

import numpy as np

from app.ingest import _clean_title, _md_sections, chunk_text, doc_hash
from app.rag import InMemoryStore, LocalEmbedder, Metrics


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_single_chunk_when_text_fits(self) -> None:
        result = chunk_text("one two three", chunk_size=10, overlap=2)
        assert len(result) == 1
        assert result[0] == "one two three"

    def test_splits_with_overlap(self) -> None:
        text = " ".join(f"w{i}" for i in range(10))
        result = chunk_text(text, chunk_size=4, overlap=2)
        assert len(result) == 4
        # First chunk: w0 w1 w2 w3
        assert result[0] == "w0 w1 w2 w3"
        # Second chunk starts at index 2 (overlap=2): w2 w3 w4 w5
        assert result[1] == "w2 w3 w4 w5"

    def test_empty_text_returns_empty(self) -> None:
        assert chunk_text("", chunk_size=10, overlap=2) == []

    def test_whitespace_only_returns_empty(self) -> None:
        assert chunk_text("   ", chunk_size=10, overlap=2) == []


# ---------------------------------------------------------------------------
# _md_sections
# ---------------------------------------------------------------------------


class TestMdSections:
    def test_extracts_heading(self) -> None:
        text = "# Title\nSome body text here."
        sections = _md_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Title"

    def test_multiple_sections(self) -> None:
        text = "# Intro\nHello\n## Details\nMore info"
        sections = _md_sections(text)
        assert len(sections) == 2
        assert sections[0][0] == "Intro"
        assert sections[1][0] == "Details"

    def test_skips_heading_only_sections(self) -> None:
        text = "# Title Only\n## Has Body\nContent here"
        sections = _md_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Has Body"

    def test_no_headings_returns_introduction(self) -> None:
        text = "Just plain text without headings."
        sections = _md_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Introduction"

    def test_heading_extraction_does_not_strip_content_chars(self) -> None:
        """Verify lstrip('#') doesn't eat letters like S from SLA."""
        text = "## SLA\nDelivery details here."
        sections = _md_sections(text)
        assert sections[0][0] == "SLA"


# ---------------------------------------------------------------------------
# _clean_title
# ---------------------------------------------------------------------------


class TestCleanTitle:
    def test_strips_md_extension(self) -> None:
        assert _clean_title("Returns_and_Refunds.md") == "Returns_and_Refunds"

    def test_strips_txt_extension(self) -> None:
        assert _clean_title("Policy.txt") == "Policy"

    def test_no_extension(self) -> None:
        assert _clean_title("README") == "README"


# ---------------------------------------------------------------------------
# doc_hash
# ---------------------------------------------------------------------------


class TestDocHash:
    def test_deterministic(self) -> None:
        assert doc_hash("hello") == doc_hash("hello")

    def test_different_content_different_hash(self) -> None:
        assert doc_hash("hello") != doc_hash("world")

    def test_returns_hex_string(self) -> None:
        h = doc_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------


class TestLocalEmbedder:
    def test_correct_dimensions(self) -> None:
        embedder = LocalEmbedder(dim=384)
        vec = embedder.embed("test")
        assert vec.shape == (384,)

    def test_deterministic_for_same_input(self) -> None:
        embedder = LocalEmbedder(dim=384)
        v1 = embedder.embed("hello world")
        v2 = embedder.embed("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_different_input_different_vector(self) -> None:
        embedder = LocalEmbedder(dim=384)
        v1 = embedder.embed("hello")
        v2 = embedder.embed("world")
        assert not np.array_equal(v1, v2)

    def test_unit_norm(self) -> None:
        embedder = LocalEmbedder(dim=384)
        vec = embedder.embed("test")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    def test_upsert_and_count(self) -> None:
        store = InMemoryStore(dim=3)
        vecs = [np.array([1, 0, 0], dtype="float32")]
        metas = [{"hash": "abc", "text": "hello"}]
        added = store.upsert(vecs, metas)
        assert added == 1
        assert store.count == 1

    def test_dedup_by_hash(self) -> None:
        store = InMemoryStore(dim=3)
        vec = np.array([1, 0, 0], dtype="float32")
        meta = {"hash": "same", "text": "hello"}
        store.upsert([vec], [meta])
        added = store.upsert([vec], [meta])
        assert added == 0
        assert store.count == 1

    def test_search_returns_results(self) -> None:
        store = InMemoryStore(dim=3)
        store.upsert(
            [np.array([1, 0, 0], dtype="float32")],
            [{"hash": "a", "text": "first"}],
        )
        store.upsert(
            [np.array([0, 1, 0], dtype="float32")],
            [{"hash": "b", "text": "second"}],
        )
        results = store.search(np.array([1, 0, 0], dtype="float32"), k=1)
        assert len(results) == 1
        assert results[0][1]["text"] == "first"

    def test_search_empty_store_returns_empty(self) -> None:
        store = InMemoryStore(dim=3)
        results = store.search(np.array([1, 0, 0], dtype="float32"), k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_records_and_averages_retrieval(self) -> None:
        m = Metrics()
        m.add_retrieval(10.0)
        m.add_retrieval(20.0)
        summary = m.summary()
        assert summary["avg_retrieval_latency_ms"] == 15.0

    def test_records_and_averages_generation(self) -> None:
        m = Metrics()
        m.add_generation(100.0)
        m.add_generation(200.0)
        summary = m.summary()
        assert summary["avg_generation_latency_ms"] == 150.0

    def test_empty_metrics_returns_zero(self) -> None:
        m = Metrics()
        summary = m.summary()
        assert summary["avg_retrieval_latency_ms"] == 0.0
        assert summary["avg_generation_latency_ms"] == 0.0
