"""Agent toolset for ingestion, retrieval, deadlines, and web search.

All tools use Pydantic args_schema for validation and are compatible with
LangChain v1.0+ `create_agent` APIs.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.core.logging_setup import get_logger
from src.core.rate_limit import RateLimitConfig, limiter
from src.core.security import validate_url
from src.services.ingest import (
    extract_text_from_pdf,
    fetch_url_text,
    ingest_source,
    ingest_visual_document,
)
from src.services.vector_store import clear_collection

try:
    from langchain_community.tools import DuckDuckGoSearchResults
except Exception:  # pragma: no cover - optional dependency import guard
    DuckDuckGoSearchResults = None


logger = get_logger(__name__)
WEB_SEARCH_LIMIT = RateLimitConfig(max_requests=15, window_seconds=60)


def _build_retrieval_filters(
    source_type: Optional[str] = None,
    source_id: Optional[str] = None,
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, str]:
    """Build Chroma-compatible metadata filters from optional tool inputs."""

    candidate_filters = {
        "source_type": source_type,
        "source_id": source_id,
        "city": city,
        "procedure": procedure,
        "language": language,
    }
    return {
        key: str(value).strip()
        for key, value in candidate_filters.items()
        if value is not None and str(value).strip()
    }


def _build_ingestion_filter_metadata(
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, str]:
    """Build optional metadata persisted at ingestion time for future filtering."""

    return _build_retrieval_filters(city=city, procedure=procedure, language=language)


def _decode_base64_bytes(value: str) -> bytes:
    """Decode a base64 string payload into raw bytes.

    Args:
        value (str): Base64-encoded file content.

    Returns:
        bytes: Decoded binary payload.
    """
    return base64.b64decode(value)


class IngestFileInput(BaseModel):
    file_base64: str = Field(..., description="Base64-encoded file content")
    filename: str = Field(..., min_length=1, description="Original filename with extension")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


@tool("ingest_file", args_schema=IngestFileInput)
def ingest_file_tool(
    file_base64: str,
    filename: str,
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a PDF/TXT/Markdown file (base64-encoded) with metadata into the vector store."""

    filename_lower = filename.lower()
    raw_bytes = _decode_base64_bytes(file_base64)
    logger.debug("ingest_file_tool_start", extra={"input_filename": filename, "bytes": len(raw_bytes)})

    if filename_lower.endswith(".pdf"):
        text = extract_text_from_pdf(raw_bytes, max_pages=100)
        source_type = "pdf"
    else:
        # Assume UTF-8 text for txt/markdown or similar
        text = raw_bytes.decode("utf-8", errors="ignore")
        source_type = "text"

    ingestion_metadata = _build_ingestion_filter_metadata(
        city=city,
        procedure=procedure,
        language=language,
    )
    result = ingest_source(
        text=text,
        source_name=filename,
        source_type=source_type,
        metadata=ingestion_metadata,
    )
    logger.info(
        "ingest_file_tool_complete",
        extra={"source_id": result.get("source_id"), "source_type": source_type, "chunks": result.get("chunk_count"), "chars": result.get("char_count")},
    )
    return result


class IngestURLInput(BaseModel):
    url: str = Field(..., description="URL to fetch and ingest")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


@tool("ingest_url", args_schema=IngestURLInput)
def ingest_url_tool(
    url: str,
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch a URL, extract text, and ingest with metadata into the vector store."""

    logger.debug("ingest_url_tool_start", extra={"url": url})
    is_valid, error_message = validate_url(url)
    if not is_valid:
        logger.warning("ingest_url_tool_blocked", extra={"url": url, "reason": error_message})
        raise ValueError(error_message)

    text = fetch_url_text(url)
    ingestion_metadata = _build_ingestion_filter_metadata(
        city=city,
        procedure=procedure,
        language=language,
    )
    result = ingest_source(
        text=text,
        source_name=url,
        source_type="url",
        metadata=ingestion_metadata,
    )
    logger.info(
        "ingest_url_tool_complete",
        extra={"source_id": result.get("source_id"), "chunks": result.get("chunk_count"), "chars": result.get("char_count")},
    )
    return result


class IngestVisualInput(BaseModel):
    file_base64: str = Field(..., description="Base64-encoded file content")
    filename: str = Field(..., min_length=1, description="Original filename including extension")
    enable_visual_mode: bool = Field(
        default=True, description="Enable visual/LLM mode in MarkItDown"
    )
    llm_model: Optional[str] = Field(
        default=None, description="Optional LLM model override when visual mode is enabled"
    )
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


@tool("ingest_visual", args_schema=IngestVisualInput)
def ingest_visual_tool(
    file_base64: str,
    filename: str,
    enable_visual_mode: bool = True,
    llm_model: Optional[str] = None,
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a document using MarkItDown with optional visual/LLM mode."""

    raw_bytes = _decode_base64_bytes(file_base64)
    logger.debug(
        "ingest_visual_tool_start",
        extra={
            "input_filename": filename,
            "bytes": len(raw_bytes),
            "visual_mode": enable_visual_mode,
            "llm_model": llm_model,
        },
    )

    result = ingest_visual_document(
        file_bytes=raw_bytes,
        filename=filename,
        enable_visual_mode=enable_visual_mode,
        llm_model=llm_model,
        metadata=_build_ingestion_filter_metadata(
            city=city,
            procedure=procedure,
            language=language,
        ),
    )
    logger.info(
        "ingest_visual_tool_complete",
        extra={
            "source_id": result.get("source_id"),
            "chunks": result.get("chunk_count"),
            "chars": result.get("char_count"),
        },
    )
    return result


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=3, description="Web search query")
    max_results: int = Field(5, ge=1, le=10, description="Maximum results to return")


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web for up-to-date information using DuckDuckGo (no API key required)."""

    if not limiter.allow("tool:web_search", WEB_SEARCH_LIMIT):
        logger.warning(
            "web_search_rate_limit_exceeded",
            extra={"max_requests": WEB_SEARCH_LIMIT.max_requests, "window_seconds": WEB_SEARCH_LIMIT.window_seconds},
        )
        raise ValueError(
            f"Rate limit exceeded for web_search. Allowed {WEB_SEARCH_LIMIT.max_requests} requests per {WEB_SEARCH_LIMIT.window_seconds} seconds."
        )

    if DuckDuckGoSearchResults is None:
        raise RuntimeError(
            "DuckDuckGoSearchResults is unavailable. Ensure langchain-community is installed."
        )

    search = DuckDuckGoSearchResults(max_results=max_results)
    results = search.invoke(query)
    logger.info("web_search_complete", extra={"query": query, "results": len(results) if results else 0})
    return {"results": results}


class ClearKnowledgeBaseInput(BaseModel):
    confirm: bool = Field(..., description="Must be true to clear the knowledge base")


@tool("clear_knowledge_base", args_schema=ClearKnowledgeBaseInput)
def clear_knowledge_base_tool(confirm: bool) -> Dict[str, Any]:
    """Dangerous: delete all stored documents from the knowledge base."""

    if not confirm:
        raise ValueError("Set confirm=true to clear the knowledge base")

    logger.warning("clear_knowledge_base_start")
    deleted_count = clear_collection()
    logger.info("clear_knowledge_base_complete", extra={"deleted_count": deleted_count})
    return {"deleted_count": deleted_count}


def get_default_tools() -> List[Any]:
    """Return the default toolset for the agent."""

    from src.agent.mcp_tools import get_mcp_tools

    return [
        *get_mcp_tools(),
        ingest_file_tool,
        ingest_visual_tool,
        ingest_url_tool,
        web_search_tool,
        clear_knowledge_base_tool,
    ]
