"""Pydantic models for knowledge_mcp tool contracts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Supported metadata filters for search."""

    source_type: Optional[str] = Field(default=None)
    source_id: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    procedure: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)


class SearchRequest(BaseModel):
    """Arguments for KB semantic search."""

    query: str = Field(..., min_length=1)
    filters: Optional[SearchFilters] = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    """A lightweight search result item."""

    id: str
    snippet: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response payload."""

    total: int
    hits: List[SearchHit] = Field(default_factory=list)


class FetchRequest(BaseModel):
    """Arguments for chunk fetch by stable id."""

    id: str = Field(..., min_length=1)


class FetchResponse(BaseModel):
    """Fetch response payload with full chunk text and metadata."""

    found: bool
    id: str
    text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestOptions(BaseModel):
    """Optional ingestion knobs."""

    chunk_size: Optional[int] = Field(default=None, ge=100)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)


class IngestTextRequest(BaseModel):
    """Arguments for text/markdown ingestion."""

    text: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    source_id: Optional[str] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    options: Optional[IngestOptions] = Field(default=None)


class IngestTextResponse(BaseModel):
    """Ingestion response payload."""

    source_id: str
    source_type: str
    chunk_count: int
    char_count: int
    created_at: str
    message: str


class UpsertMetadataRequest(BaseModel):
    """Arguments for source-level metadata patching."""

    source_id: str = Field(..., min_length=1)
    patch: Dict[str, Any] = Field(default_factory=dict)


class UpsertMetadataResponse(BaseModel):
    """Metadata patch result payload."""

    supported: bool
    updated_chunks: int
    message: str


class HealthResponse(BaseModel):
    """Operational health status."""

    status: str
    server: str
    collection: str
    chroma_persist_dir: str
    total_chunks: int


class StatsResponse(BaseModel):
    """Knowledge base statistics."""

    total_sources: int
    total_chunks: int
    by_source_type: Dict[str, int] = Field(default_factory=dict)
