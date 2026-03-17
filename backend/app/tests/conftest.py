import os

import pytest
from fastapi.testclient import TestClient

# Force in-memory store and stub LLM for tests so they run
# without Qdrant or external API keys.
os.environ["VECTOR_STORE"] = "memory"
os.environ["LLM_PROVIDER"] = "stub"

from app.main import app  # noqa: E402 — must import after env override


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Shared TestClient for integration tests."""
    return TestClient(app)
