"""FastMCP server exposing local Knowledge Base ingestion and retrieval tools."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from knowledge_mcp import adapters
from knowledge_mcp.config import ensure_backend_path, get_mcp_settings
from knowledge_mcp.models import (
    IngestTextRequest,
    SearchFilters,
    SearchRequest,
    UpsertMetadataRequest,
)

ensure_backend_path()

from src.core.logging_setup import get_logger

logger = get_logger(__name__)
settings = get_mcp_settings()
mcp = FastMCP(
    settings.server_name,
    host=settings.host,
    port=settings.port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


def _duration_ms(start_time: float) -> int:
    """Return elapsed milliseconds from a monotonic timer start point.

    Args:
        start_time (float): Starting value from `time.perf_counter()`.

    Returns:
        int: Elapsed time in whole milliseconds.
    """
    return int((time.perf_counter() - start_time) * 1000)


@mcp.tool()
def search(
    query: str,
    filters: Optional[Dict[str, str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Search the local KB. Returns stable chunk IDs with snippets and metadata."""

    start_time = time.perf_counter()
    parsed_filters = SearchFilters(**(filters or {}))
    request = SearchRequest(query=query, filters=parsed_filters, top_k=top_k)

    response = adapters.search(request)
    duration = _duration_ms(start_time)
    logger.info(
        "knowledge_mcp_tool_call",
        extra={
            "tool": "search",
            "query_chars": len(query),
            "top_k": top_k,
            "hits": response.total,
            "duration_ms": duration,
        },
    )
    return response.model_dump()


@mcp.tool()
def fetch(id: str) -> Dict[str, Any]:
    """Fetch full chunk text + metadata by stable chunk id."""

    start_time = time.perf_counter()
    response = adapters.fetch(id)
    duration = _duration_ms(start_time)
    logger.info(
        "knowledge_mcp_tool_call",
        extra={
            "tool": "fetch",
            "id": id,
            "found": response.found,
            "duration_ms": duration,
        },
    )
    return response.model_dump()


@mcp.tool()
def ingest_text(
    text: str,
    source_type: str,
    source_id: Optional[str] = None,
    source_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ingest cleaned text/markdown into the local KB (no URL/network fetching)."""

    start_time = time.perf_counter()
    request = IngestTextRequest(
        text=text,
        source_type=source_type,
        source_id=source_id,
        source_url=source_url,
        metadata=metadata or {},
        options=options,
    )
    response = adapters.ingest_text(request)
    duration = _duration_ms(start_time)
    logger.info(
        "knowledge_mcp_tool_call",
        extra={
            "tool": "ingest_text",
            "source_type": source_type,
            "source_id": response.source_id,
            "chunks": response.chunk_count,
            "chars": response.char_count,
            "duration_ms": duration,
        },
    )
    return response.model_dump()


@mcp.tool()
def upsert_metadata(source_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Patch chunk metadata for all records belonging to a source_id."""

    start_time = time.perf_counter()
    request = UpsertMetadataRequest(source_id=source_id, patch=patch)
    response = adapters.upsert_metadata(request)
    duration = _duration_ms(start_time)
    logger.info(
        "knowledge_mcp_tool_call",
        extra={
            "tool": "upsert_metadata",
            "source_id": source_id,
            "updated_chunks": response.updated_chunks,
            "supported": response.supported,
            "duration_ms": duration,
        },
    )
    return response.model_dump()


@mcp.tool()
def health() -> Dict[str, Any]:
    """Return basic server and collection health information."""

    start_time = time.perf_counter()
    response = adapters.health()
    duration = _duration_ms(start_time)
    logger.info(
        "knowledge_mcp_tool_call",
        extra={
            "tool": "health",
            "status": response.status,
            "chunks": response.total_chunks,
            "duration_ms": duration,
        },
    )
    return response.model_dump()


@mcp.tool()
def stats() -> Dict[str, Any]:
    """Return aggregate KB statistics."""

    start_time = time.perf_counter()
    response = adapters.stats()
    duration = _duration_ms(start_time)
    logger.info(
        "knowledge_mcp_tool_call",
        extra={
            "tool": "stats",
            "sources": response.total_sources,
            "chunks": response.total_chunks,
            "duration_ms": duration,
        },
    )
    return response.model_dump()


def main() -> None:
    """CLI entrypoint for running knowledge_mcp over stdio or Streamable HTTP."""

    parser = argparse.ArgumentParser(description="knowledge_mcp FastMCP server")
    parser.add_argument(
        "--transport",
        default=settings.transport,
        choices=["stdio", "http", "streamable-http"],
        help="MCP transport mode",
    )
    args = parser.parse_args()

    transport = "streamable-http" if args.transport == "http" else args.transport
    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
