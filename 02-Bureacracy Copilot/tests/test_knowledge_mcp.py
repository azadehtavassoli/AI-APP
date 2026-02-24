"""Unit tests for knowledge_mcp adapters."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(BACKEND_DIR))

from knowledge_mcp.adapters import fetch, ingest_text, search  # noqa: E402
from knowledge_mcp.models import IngestTextRequest, SearchFilters, SearchRequest  # noqa: E402
from src.services import vector_store  # noqa: E402


class _FakeEmbeddings:
    """Deterministic embedding model for offline tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each input text deterministically for offline testing.

        Args:
            texts (list[str]): Source texts to embed.

        Returns:
            list[list[float]]: Deterministic dense vectors.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        """Embed one query string deterministically.

        Args:
            query (str): Query text.

        Returns:
            list[float]: Deterministic query vector.
        """
        return self._embed(query)

    @staticmethod
    def _embed(value: str) -> list[float]:
        """Create a tiny deterministic vector from text statistics.

        Args:
            value (str): Input text value.

        Returns:
            list[float]: Vector containing length, checksum, and word count.
        """
        length = float(len(value))
        checksum = float(sum(ord(char) for char in value) % 1000)
        words = float(len(value.split()))
        return [length, checksum, words]


@pytest.fixture()
def isolated_vector_store(tmp_path, monkeypatch):
    """Set up isolated Chroma store and fake embeddings for each test."""

    collection_name = f"knowledge_mcp_test_{uuid.uuid4().hex[:8]}"
    persist_dir = tmp_path / "chroma"

    vector_store.get_client.cache_clear()
    vector_store.settings.chroma_persist_dir = str(persist_dir)
    vector_store.settings.chroma_collection = collection_name

    monkeypatch.setattr("knowledge_mcp.adapters.get_embeddings", lambda: _FakeEmbeddings())
    monkeypatch.setattr("src.services.vector_store.get_embeddings", lambda: _FakeEmbeddings())
    yield
    vector_store.get_client.cache_clear()


def test_ingest_text_inserts_records_with_metadata(isolated_vector_store):
    """Verify ingestion stores source metadata fields on produced chunks.

    Args:
        isolated_vector_store: Auto-configured fixture for isolated Chroma state.

    Returns:
        None.
    """
    request = IngestTextRequest(
        text="Berlin Anmeldung should happen within 14 days after moving.",
        source_type="text",
        source_id="src-berlin-guide",
        source_url="https://example.local/berlin-anmeldung",
        metadata={"city": "Berlin", "procedure": "Anmeldung", "language": "en"},
    )

    result = ingest_text(request)

    assert result.source_id == "src-berlin-guide"
    assert result.chunk_count > 0

    collection = vector_store.get_collection()
    records = collection.get(where={"source_id": "src-berlin-guide"}, include=["metadatas"])
    metadatas = records.get("metadatas") or []

    assert metadatas
    assert all(metadata.get("city") == "Berlin" for metadata in metadatas)
    assert all(metadata.get("procedure") == "Anmeldung" for metadata in metadatas)
    assert all(metadata.get("language") == "en" for metadata in metadatas)
    assert all(metadata.get("chunk_id") for metadata in metadatas)


def test_search_returns_stable_ids(isolated_vector_store):
    """Verify search results include stable chunk IDs for retrieval.

    Args:
        isolated_vector_store: Auto-configured fixture for isolated Chroma state.

    Returns:
        None.
    """
    ingest_text(
        IngestTextRequest(
            text="Residence registration deadline in Berlin is 14 days.",
            source_type="text",
            source_id="source-search-1",
            metadata={"city": "Berlin", "procedure": "Anmeldung", "language": "en"},
        )
    )

    response = search(
        SearchRequest(
            query="What is the registration deadline in Berlin?",
            filters=SearchFilters(city="Berlin", procedure="Anmeldung"),
            top_k=3,
        )
    )

    assert response.total >= 1
    assert all(hit.id for hit in response.hits)


def test_fetch_returns_text_and_metadata_for_returned_id(isolated_vector_store):
    """Verify fetch returns text and metadata for IDs returned by search.

    Args:
        isolated_vector_store: Auto-configured fixture for isolated Chroma state.

    Returns:
        None.
    """
    ingest_text(
        IngestTextRequest(
            text="For Anmeldung, visit your local Bürgeramt in Berlin.",
            source_type="text",
            source_id="source-fetch-1",
            metadata={"city": "Berlin", "procedure": "Anmeldung", "language": "en"},
        )
    )

    search_response = search(
        SearchRequest(query="Where should I register in Berlin?", filters=SearchFilters(city="Berlin"), top_k=2)
    )
    assert search_response.hits

    chunk_id = search_response.hits[0].id
    fetch_response = fetch(chunk_id)

    assert fetch_response.found is True
    assert "Bürgeramt" in fetch_response.text
    assert fetch_response.metadata.get("source_id") == "source-fetch-1"
    assert fetch_response.metadata.get("city") == "Berlin"
