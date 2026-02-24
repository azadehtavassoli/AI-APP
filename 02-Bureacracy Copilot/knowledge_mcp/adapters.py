"""Thin adapters from MCP tools to existing backend ingestion/retrieval modules."""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Optional

from knowledge_mcp.config import ensure_backend_path, get_backend_runtime_info
from knowledge_mcp.models import (
    FetchResponse,
    HealthResponse,
    IngestTextRequest,
    IngestTextResponse,
    SearchFilters,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    UpsertMetadataRequest,
    UpsertMetadataResponse,
)

ensure_backend_path()

from src.core.logging_setup import get_logger
from src.core.security import validate_text_length
from src.services.chunking import chunk_text
from src.services.embeddings import get_embeddings
from src.services.vector_store import (
    add_chunks,
    get_collection,
    list_sources,
    similarity_search_with_score,
)

logger = get_logger(__name__)

SUPPORTED_FILTERS = ("source_type", "source_id", "city", "procedure", "language")
FETCH_METADATA_FIELDS = (
    "source_type",
    "source_id",
    "source_url",
    "page",
    "city",
    "procedure",
    "language",
    "created_at",
    "fetched_at",
)


def _normalize_filters(filters: Optional[SearchFilters]) -> Dict[str, str]:
    """Convert optional typed search filters to a cleaned string dictionary.

    Args:
        filters (Optional[SearchFilters]): Structured filter payload from request.

    Returns:
        Dict[str, str]: Non-empty supported filter values normalized as strings.
    """
    if filters is None:
        return {}

    normalized: Dict[str, str] = {}
    for field_name in SUPPORTED_FILTERS:
        value = getattr(filters, field_name, None)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            normalized[field_name] = value_str
    return normalized


def _to_chroma_where(normalized_filters: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Build Chroma where clause, supporting one or many equality filters."""

    if not normalized_filters:
        return None

    if len(normalized_filters) == 1:
        key, value = next(iter(normalized_filters.items()))
        return {key: value}

    return {
        "$and": [{key: value} for key, value in normalized_filters.items()]
    }


def _build_chunk_id(source_id: str, metadata: Dict[str, Any], text: str) -> str:
    """Build deterministic chunk identifiers from source metadata and text.

    Args:
        source_id (str): Canonical source identifier.
        metadata (Dict[str, Any]): Chunk metadata dictionary.
        text (str): Chunk text content.

    Returns:
        str: Stable SHA-1 digest used as `chunk_id`.
    """
    page_value = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("page_index")
        or "na"
    )
    chunk_index = metadata.get("chunk_index", "na")
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    seed = f"{source_id}:{page_value}:{chunk_index}:{text_hash}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def ingest_text(request: IngestTextRequest) -> IngestTextResponse:
    """Ingest cleaned text/markdown into Chroma using existing pipeline components."""

    text = request.text.strip()
    is_valid, error_message = validate_text_length(text, 100000)
    if not is_valid:
        raise ValueError(error_message)

    source_id = (request.source_id or str(uuid.uuid4())).strip()
    created_at = datetime.utcnow().isoformat()

    source_metadata: Dict[str, Any] = {
        "source_id": source_id,
        "source_type": request.source_type,
        "source_name": source_id,
        "timestamp": created_at,
        "created_at": created_at,
        "char_count": len(text),
    }

    requested_source_name = request.metadata.get("source_name")
    if requested_source_name is not None and str(requested_source_name).strip():
        source_metadata["source_name"] = str(requested_source_name).strip()

    if request.source_url:
        source_metadata["source_url"] = str(request.source_url).strip()

    for filter_name in SUPPORTED_FILTERS:
        if filter_name in request.metadata:
            filter_value = request.metadata.get(filter_name)
            if filter_value is not None and str(filter_value).strip():
                source_metadata[filter_name] = str(filter_value).strip()

    if "fetched_at" in request.metadata:
        source_metadata["fetched_at"] = str(request.metadata["fetched_at"])

    chunks = chunk_text(text=text, source_id=source_id, metadata=source_metadata)

    embeddings_model = get_embeddings()
    embedding_vectors = embeddings_model.embed_documents([doc.page_content for doc in chunks])

    chunks_for_storage = []
    for doc in chunks:
        chunk_metadata = dict(doc.metadata)
        chunk_metadata["chunk_id"] = chunk_metadata.get("chunk_id") or _build_chunk_id(
            source_id=source_id,
            metadata=chunk_metadata,
            text=doc.page_content,
        )
        chunks_for_storage.append({"text": doc.page_content, "metadata": chunk_metadata})

    stored_count = add_chunks(chunks_for_storage, embedding_vectors, source_id)

    return IngestTextResponse(
        source_id=source_id,
        source_type=request.source_type,
        chunk_count=stored_count,
        char_count=len(text),
        created_at=created_at,
        message="ingest_text completed",
    )


def search(request: SearchRequest) -> SearchResponse:
    """Search the local KB and return stable chunk IDs plus snippets."""

    where_filter = _to_chroma_where(_normalize_filters(request.filters))
    results = similarity_search_with_score(
        query=request.query,
        top_k=request.top_k,
        where=where_filter,
    )

    hits = []
    for result in results:
        metadata = result.get("metadata") or {}
        chunk_id = metadata.get("chunk_id") or result.get("id") or ""
        if not chunk_id:
            continue

        document_text = (result.get("document") or "").strip()
        snippet = document_text[:260] + ("..." if len(document_text) > 260 else "")

        hits.append(
            SearchHit(
                id=str(chunk_id),
                snippet=snippet,
                score=result.get("score"),
                metadata=metadata,
            )
        )

    return SearchResponse(total=len(hits), hits=hits)


def fetch(chunk_id: str) -> FetchResponse:
    """Fetch full chunk text + metadata by stable chunk ID."""

    collection = get_collection()
    by_metadata = collection.get(where={"chunk_id": chunk_id}, include=["documents", "metadatas"])

    ids = by_metadata.get("ids") or []
    documents = by_metadata.get("documents") or []
    metadatas = by_metadata.get("metadatas") or []

    if ids:
        metadata = metadatas[0] if metadatas else {}
        raw_metadata = metadata if isinstance(metadata, dict) else {}
        citation_metadata = {
            key: raw_metadata.get(key)
            for key in FETCH_METADATA_FIELDS
            if raw_metadata.get(key) is not None
        }
        citation_metadata["chunk_id"] = raw_metadata.get("chunk_id", chunk_id)

        text_value = documents[0] if documents else ""
        return FetchResponse(found=True, id=chunk_id, text=text_value, metadata=citation_metadata)

    # Fallback for legacy records where only Chroma id exists.
    by_id = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    legacy_docs = by_id.get("documents") or []
    legacy_meta = by_id.get("metadatas") or []
    if legacy_docs:
        metadata = legacy_meta[0] if legacy_meta else {}
        if isinstance(metadata, dict) and "chunk_id" not in metadata:
            metadata["chunk_id"] = chunk_id
        return FetchResponse(found=True, id=chunk_id, text=legacy_docs[0], metadata=metadata or {})

    return FetchResponse(found=False, id=chunk_id, text="", metadata={})


def upsert_metadata(request: UpsertMetadataRequest) -> UpsertMetadataResponse:
    """Patch metadata for all chunks under a source_id if supported by Chroma."""

    collection = get_collection()

    try:
        current = collection.get(where={"source_id": request.source_id}, include=["metadatas"])
        ids = current.get("ids") or []
        metadatas = current.get("metadatas") or []

        if not ids:
            return UpsertMetadataResponse(
                supported=True,
                updated_chunks=0,
                message="No matching chunks for source_id",
            )

        cleaned_patch = {
            key: value
            for key, value in request.patch.items()
            if value is not None and key != "source_id"
        }
        if not cleaned_patch:
            return UpsertMetadataResponse(
                supported=True,
                updated_chunks=0,
                message="No patch fields provided",
            )

        merged_metadatas = []
        for metadata in metadatas:
            current_metadata = dict(metadata or {})
            current_metadata.update(cleaned_patch)
            merged_metadatas.append(current_metadata)

        collection.update(ids=ids, metadatas=merged_metadatas)

        return UpsertMetadataResponse(
            supported=True,
            updated_chunks=len(ids),
            message="Metadata upsert completed",
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("upsert_metadata_not_supported: %s", exc)
        return UpsertMetadataResponse(
            supported=False,
            updated_chunks=0,
            message=f"Metadata update not supported by storage backend: {exc}",
        )


def health() -> HealthResponse:
    """Return basic operational status."""

    runtime = get_backend_runtime_info()
    collection = get_collection()
    count = collection.count()

    return HealthResponse(
        status="ok",
        server="knowledge_mcp",
        collection=runtime["chroma_collection"],
        chroma_persist_dir=runtime["chroma_persist_dir"],
        total_chunks=count,
    )


def stats() -> StatsResponse:
    """Return aggregate knowledge base statistics."""

    sources = list_sources()
    chunks_total = sum(int(source.get("chunk_count", 0)) for source in sources)
    source_type_counter = Counter(str(source.get("source_type", "unknown")) for source in sources)

    return StatsResponse(
        total_sources=len(sources),
        total_chunks=chunks_total,
        by_source_type=dict(source_type_counter),
    )
