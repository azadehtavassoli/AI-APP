"""MCP-backed LangChain tools for remote Knowledge MCP over Streamable HTTP.

This module provides drop-in tool names used by the agent (`ingest_text`,
`retrieve_context`) while delegating execution to the `knowledge_mcp` server
via the official MCP Python SDK streamable HTTP client.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    ClientSession = importlib.import_module("mcp").ClientSession
    streamable_http_client = importlib.import_module(
        "mcp.client.streamable_http"
    ).streamable_http_client
except Exception:  # pragma: no cover - optional dependency guard
    ClientSession = None
    streamable_http_client = None

from src.core.logging_setup import get_logger
from src.core.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


def _detect_language(question: str) -> str:
    """Detect question language using lightweight script/keyword heuristics.

    Args:
        question (str): User question text.

    Returns:
        str: Detected language code (`en`, `de`, `fa`).
    """
    if not question or not question.strip():
        return "en"

    # Detect Persian script quickly via Unicode block.
    if any("\u0600" <= char <= "\u06FF" for char in question):
        return "fa"

    question_lower = question.lower()
    # Detect German by umlauts and common stopwords.
    if any(char in question_lower for char in ("ä", "ö", "ü", "ß")):
        return "de"

    german_stopwords = {"der", "die", "das", "und", "mit", "für", "ist", "von", "auf"}
    tokenized = question_lower.split()
    german_count = sum(1 for token in tokenized if token in german_stopwords)
    if german_count >= 2:
        return "de"

    return "en"


def _translate_query_with_llm(question: str, target_language: str) -> Optional[str]:
    """Translate question into target language using OpenAI via LangChain.

    Args:
        question (str): Input query text.
        target_language (str): Target language code (`en`, `de`, `fa`).

    Returns:
        Optional[str]: Translated query string, or `None` on failure.
    """
    if not question or not question.strip():
        return None

    language_names = {
        "en": "English",
        "de": "German",
        "fa": "Persian",
    }
    target_name = language_names.get(target_language, target_language)

    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )
        prompt = (
            f"Translate the following query to {target_name}. "
            "Return only the translation without explanations.\n\n"
            f"Query: {question}\n\nTranslation:"
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        translated = str(response.content).strip()
        return translated or None
    except Exception as exc:  # pragma: no cover - failure path is tested via monkeypatch
        logger.warning(
            "mcp_query_translation_failed",
            extra={
                "context": {
                    "target_language": target_language,
                    "error": str(exc),
                }
            },
        )
        return None


def _build_query_variants(question: str) -> List[str]:
    """Build multilingual query variants with non-failing translation fallback.

    Args:
        question (str): Original user question.

    Returns:
        List[str]: Ordered query variants starting with original question.
    """
    detected_language = _detect_language(question)
    variants: List[str] = [question]

    target_languages: List[str] = []
    if detected_language == "en":
        target_languages = ["de"]
    elif detected_language == "de":
        target_languages = ["en"]
    elif detected_language == "fa":
        target_languages = ["de", "en"]

    for target_language in target_languages:
        try:
            translated_query = _translate_query_with_llm(question, target_language)
        except Exception as exc:  # pragma: no cover - covered in tests via monkeypatch
            logger.warning(
                "mcp_query_translation_variant_failed",
                extra={
                    "context": {
                        "target_language": target_language,
                        "error": str(exc),
                    }
                },
            )
            translated_query = None
        if translated_query and translated_query not in variants:
            variants.append(translated_query)

    logger.info(
        "mcp_query_variants_ready",
        extra={
            "context": {
                "detected_language": detected_language,
                "variant_count": len(variants),
            }
        },
    )
    return variants


def _safe_distance(value: Any) -> float:
    """Normalize hit score into sortable distance where lower means better.

    Args:
        value (Any): Raw score value from MCP search hit.

    Returns:
        float: Parsed numeric score, or positive infinity when invalid.
    """
    try:
        return float(value)
    except Exception:
        return float("inf")


def _extract_transport_streams(transport_context: Any) -> tuple[Any, Any]:
    """Return read/write streams from streamable HTTP context payload.

    The MCP Python SDK has returned both 2-item and 3-item tuples across
    versions. We only need the first two values (read/write streams).
    """

    if not isinstance(transport_context, tuple):
        raise TypeError(
            "Unexpected streamable_http_client context payload type: "
            f"{type(transport_context).__name__}"
        )

    if len(transport_context) < 2:
        raise ValueError(
            "streamable_http_client context payload must include read/write streams"
        )

    return transport_context[0], transport_context[1]


def _run_async(coroutine):
    """Run async coroutine from sync context, including active event-loop contexts."""

    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coroutine)
            return future.result()
    except RuntimeError:
        return asyncio.run(coroutine)


def _extract_result_payload(call_result: Any) -> Dict[str, Any]:
    """Parse MCP call result into a dictionary payload."""

    def _unwrap_result_wrapper(payload: Any) -> Any:
        """Unwrap FastMCP payloads shaped as `{"result": {...}}`.

        Args:
            payload (Any): Parsed JSON-like payload from MCP result fields.

        Returns:
            Any: Inner `result` object when present; otherwise original payload.
        """
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            return payload["result"]
        return payload

    result_dict = call_result.model_dump(mode="json", exclude_none=True)

    if result_dict.get("isError"):
        content = result_dict.get("content") or []
        message = "MCP tool call failed"
        if content and isinstance(content, list):
            first = content[0]
            if isinstance(first, dict):
                message = first.get("text") or first.get("json") or message
        raise ValueError(str(message))

    structured = result_dict.get("structuredContent")
    if isinstance(structured, dict):
        unwrapped_structured = _unwrap_result_wrapper(structured)
        if isinstance(unwrapped_structured, dict):
            return unwrapped_structured

    content = result_dict.get("content") or []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("json"), dict):
                unwrapped_json = _unwrap_result_wrapper(item["json"])
                if isinstance(unwrapped_json, dict):
                    return unwrapped_json
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                text_value = item["text"].strip()
                if not text_value:
                    continue
                try:
                    parsed = json.loads(text_value)
                    parsed = _unwrap_result_wrapper(parsed)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue

    return result_dict


async def _call_mcp_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a remote MCP tool over streamable HTTP."""

    if ClientSession is None or streamable_http_client is None:
        raise RuntimeError(
            "MCP dependency is not installed. Install package 'mcp' to enable MCP-backed tools."
        )

    async with streamable_http_client(settings.knowledge_mcp_url) as transport_context:
        read_stream, write_stream = _extract_transport_streams(transport_context)
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _extract_result_payload(result)


def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for MCP tool invocation."""

    logger.debug(
        "mcp_tool_call_start",
        extra={
            "context": {
                "tool_name": tool_name,
                "arguments_keys": sorted(arguments.keys()),
                "arguments": _compact_for_log(arguments),
            }
        },
    )
    result = _run_async(_call_mcp_tool_async(tool_name=tool_name, arguments=arguments))
    logger.debug(
        "mcp_tool_call_complete",
        extra={
            "context": {
                "tool_name": tool_name,
                "result_keys": sorted(result.keys()) if isinstance(result, dict) else None,
                "result_preview": _compact_for_log(result),
            }
        },
    )
    return result


def _compact_for_log(value: Any, max_str: int = 220, max_items: int = 10, max_depth: int = 5) -> Any:
    """Trim nested payloads to keep MCP debug logs concise and readable."""

    if max_depth <= 0:
        return "<max-depth-reached>"

    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return f"{value[:max_str]}...(+{len(value) - max_str} chars)"

    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            output[str(key)] = _compact_for_log(
                item,
                max_str=max_str,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
        if len(items) > max_items:
            output["__truncated_keys__"] = len(items) - max_items
        return output

    if isinstance(value, list):
        compact_list = [
            _compact_for_log(
                item,
                max_str=max_str,
                max_items=max_items,
                max_depth=max_depth - 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            compact_list.append(f"...(+{len(value) - max_items} items)")
        return compact_list

    return value


class IngestTextInput(BaseModel):
    """Arguments for MCP-backed text ingestion tool."""

    text: str = Field(..., min_length=1, description="Plain text to ingest")
    source_name: str = Field(..., min_length=1, description="Human-readable source identifier")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


@tool("ingest_text", args_schema=IngestTextInput)
def ingest_text_tool(
    text: str,
    source_name: str,
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest text into the KB via remote `knowledge_mcp` server."""

    metadata = {
        "source_name": source_name,
        "city": city,
        "procedure": procedure,
        "language": language,
    }
    cleaned_metadata = {
        key: str(value).strip()
        for key, value in metadata.items()
        if value is not None and str(value).strip()
    }

    return _call_mcp_tool(
        "ingest_text",
        {
            "text": text,
            "source_type": "text",
            "source_id": source_name,
            "metadata": cleaned_metadata,
        },
    )


class RetrieveContextInput(BaseModel):
    """Arguments for MCP-backed retrieval context tool."""

    question: str = Field(..., min_length=1, description="Query to retrieve relevant context")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    source_type: Optional[str] = Field(default=None, description="Optional source_type metadata filter")
    source_id: Optional[str] = Field(default=None, description="Optional source_id metadata filter")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter")


@tool("retrieve_context", args_schema=RetrieveContextInput)
def retrieve_context_tool(
    question: str,
    top_k: int = 5,
    source_type: Optional[str] = None,
    source_id: Optional[str] = None,
    city: Optional[str] = None,
    procedure: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve context/citations via `knowledge_mcp.search` + `knowledge_mcp.fetch`."""

    filters = {
        "source_type": source_type,
        "source_id": source_id,
        "city": city,
        "procedure": procedure,
        "language": language,
    }
    cleaned_filters = {
        key: str(value).strip()
        for key, value in filters.items()
        if value is not None and str(value).strip()
    }

    query_variants = _build_query_variants(question)

    all_hits: List[Dict[str, Any]] = []
    per_variant_counts: List[Dict[str, Any]] = []
    for query_variant in query_variants:
        search_result = _call_mcp_tool(
            "search",
            {
                "query": query_variant,
                "filters": cleaned_filters,
                "top_k": top_k,
            },
        )

        hits = search_result.get("hits") if isinstance(search_result, dict) else []
        hits = hits if isinstance(hits, list) else []
        per_variant_counts.append({"query": query_variant[:80], "hits": len(hits)})
        all_hits.extend(hit for hit in hits if isinstance(hit, dict))

    deduped_hits: Dict[str, Dict[str, Any]] = {}
    for hit in all_hits:
        chunk_id = str(hit.get("id", "")).strip()
        if not chunk_id:
            continue

        existing_hit = deduped_hits.get(chunk_id)
        if existing_hit is None:
            deduped_hits[chunk_id] = hit
            continue

        if _safe_distance(hit.get("score")) < _safe_distance(existing_hit.get("score")):
            deduped_hits[chunk_id] = hit

    sorted_hits = sorted(
        deduped_hits.values(),
        key=lambda hit: _safe_distance(hit.get("score")),
    )
    selected_hits = sorted_hits[:top_k]

    logger.info(
        "mcp_retrieve_multilingual_summary",
        extra={
            "context": {
                "variant_count": len(query_variants),
                "variant_hits": per_variant_counts,
                "merged_hits": len(all_hits),
                "deduped_hits": len(deduped_hits),
                "selected_hits": len(selected_hits),
            }
        },
    )

    citations: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    for index, hit in enumerate(selected_hits, start=1):
        if not isinstance(hit, dict):
            continue

        chunk_id = str(hit.get("id", "")).strip()
        if not chunk_id:
            continue

        fetch_result = _call_mcp_tool("fetch", {"id": chunk_id})
        if not isinstance(fetch_result, dict) or not fetch_result.get("found"):
            continue

        metadata = fetch_result.get("metadata") if isinstance(fetch_result.get("metadata"), dict) else {}
        text = str(fetch_result.get("text", "")).strip()
        if text:
            context_parts.append(f"[Source {index}]\n{text}")

        source_id_value = str(metadata.get("source_id") or "unknown")
        source_type_value = str(metadata.get("source_type") or "unknown")
        source_uri_value = str(
            metadata.get("source_url")
            or metadata.get("source_name")
            or source_id_value
        )

        snippet = text[:300].strip()
        if len(text) > 300:
            snippet += "..."

        citations.append(
            {
                "source_id": source_id_value,
                "source_type": source_type_value,
                "source_uri": source_uri_value,
                "chunk_id": chunk_id,
                "snippet": snippet,
                "score": hit.get("score"),
            }
        )

    return {
        "context": "\n\n".join(context_parts),
        "citations": citations,
        "retrieved_count": len(citations),
        "applied_filters": cleaned_filters,
    }


def get_mcp_tools() -> List[Any]:
    """Return MCP-backed KB tool adapters."""

    return [
        ingest_text_tool,
        retrieve_context_tool,
    ]
