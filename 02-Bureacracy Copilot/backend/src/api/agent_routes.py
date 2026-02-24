"""Agent-powered endpoints for ingestion and retrieval testing.

These routes are additive alongside the existing REST API; they do not replace
the legacy ingestion/query endpoints under /api/*.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from tenacity import RetryError

from src.agent import AgentRequest, AgentResponse, RAGAgent
from src.core.logging_setup import get_logger
from src.core.rate_limit import RateLimitConfig, limiter

router = APIRouter(prefix="/api/agent", tags=["agent"])
_agent = RAGAgent()
logger = get_logger(__name__)

AGENT_QUERY_LIMIT = RateLimitConfig(max_requests=20, window_seconds=60)
AGENT_INGEST_LIMIT = RateLimitConfig(max_requests=10, window_seconds=60)
FILTER_METADATA_FIELDS = ("city", "procedure", "language")


def _client_identifier(request: Request) -> str:
    """Build best-effort client identifier for per-client throttling."""

    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request, route_name: str, config: RateLimitConfig) -> None:
    """Reject request with 429 when fixed-window limit is exceeded."""

    client_id = _client_identifier(request)
    key = f"{route_name}:{client_id}"
    if limiter.allow(key, config):
        return

    logger.warning(
        "agent_rate_limit_exceeded",
        extra={
            "route": route_name,
            "client_id": client_id,
            "max_requests": config.max_requests,
            "window_seconds": config.window_seconds,
        },
    )
    raise HTTPException(
        status_code=429,
        detail=(
            f"Rate limit exceeded for {route_name}. "
            f"Allowed {config.max_requests} requests per {config.window_seconds} seconds."
        ),
    )


def _tool_metadata_prompt_lines(metadata: Dict[str, Any]) -> str:
    """Render optional metadata fields for ingest tool invocation prompts."""

    lines: List[str] = []
    for field_name in FILTER_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            lines.append(f"- {field_name}: {value_str}")

    if not lines:
        return ""
    return "\n" + "\n".join(lines)


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or instruction")
    session_id: Optional[str] = Field(
        default=None, description="Conversation/session identifier"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for observability"
    )


class AgentQueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str
    run_id: str


class AgentIngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Plain text to ingest")
    source_name: str = Field(..., min_length=1, description="Human-readable source name")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata to attach"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class AgentIngestFileRequest(BaseModel):
    file_base64: str = Field(..., description="Base64-encoded file content")
    filename: str = Field(..., min_length=1, description="Original filename")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata to attach"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class AgentIngestURLRequest(BaseModel):
    url: str = Field(..., description="URL to fetch and ingest")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata to attach"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class AgentIngestVisualRequest(BaseModel):
    file_base64: str = Field(..., description="Base64-encoded file content")
    filename: str = Field(..., min_length=1, description="Original filename including extension")
    enable_visual_mode: bool = Field(
        default=True, description="Enable MarkItDown visual/LLM mode (set false for deterministic mode)"
    )
    llm_model: Optional[str] = Field(
        default=None, description="Optional LLM model override when visual mode is enabled"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata to attach"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")


def _invoke_agent(prompt: str, session_id: Optional[str], metadata: Dict[str, Any]) -> AgentResponse:
    """Invoke the singleton agent and map domain/runtime errors to HTTP errors.

    Args:
        prompt (str): Prompt passed to the agent runtime.
        session_id (Optional[str]): Caller-provided session identifier.
        metadata (Dict[str, Any]): Observability metadata forwarded to the agent.

    Returns:
        AgentResponse: Normalized structured response from the agent.

    Raises:
        HTTPException: With status 400 for validation/tool input issues and 500 for runtime failures.
    """
    # Ensure every run is scoped to a stable session, generating one when omitted.
    sid = session_id or str(uuid4())
    try:
        return _agent.invoke(AgentRequest(input=prompt, session_id=sid, metadata=metadata))
    except ValueError as exc:  # pragma: no cover - input/domain validation surfaced by tools
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - passthrough
        if isinstance(exc, RetryError):
            root_exc = exc.last_attempt.exception() if exc.last_attempt else exc
            if isinstance(root_exc, ValueError):
                raise HTTPException(status_code=400, detail=str(root_exc))
            raise HTTPException(status_code=500, detail=str(root_exc))

        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest, http_request: Request) -> AgentQueryResponse:
    """Run the agent over a free-form query to exercise retrieval/tools."""

    _enforce_rate_limit(http_request, "agent_query", AGENT_QUERY_LIMIT)

    response = _invoke_agent(request.query, request.session_id, request.metadata)
    return AgentQueryResponse(
        answer=response.answer,
        citations=response.citations,
        tool_calls=response.tool_calls,
        session_id=response.session_id,
        run_id=response.run_id,
    )


@router.post("/ingest/text", response_model=AgentQueryResponse)
async def agent_ingest_text(request: AgentIngestTextRequest, http_request: Request) -> AgentQueryResponse:
    """Ask the agent to ingest raw text using the ingest_text tool."""

    _enforce_rate_limit(http_request, "agent_ingest_text", AGENT_INGEST_LIMIT)

    prompt = (
        "Call the ingest_text tool with exactly these arguments and nothing else.\n"
        f"- source_name: {request.source_name}\n"
        f"- text: {request.text}\n"
        f"{_tool_metadata_prompt_lines(request.metadata)}"
        "After the tool runs, return a short confirmation. Do not call other tools."
    )
    response = _invoke_agent(prompt, request.session_id, request.metadata)
    return AgentQueryResponse(
        answer=response.answer,
        citations=response.citations,
        tool_calls=response.tool_calls,
        session_id=response.session_id,
        run_id=response.run_id,
    )


@router.post("/ingest/file", response_model=AgentQueryResponse)
async def agent_ingest_file(request: AgentIngestFileRequest, http_request: Request) -> AgentQueryResponse:
    """Ask the agent to ingest a base64 file using the ingest_file tool."""

    _enforce_rate_limit(http_request, "agent_ingest_file", AGENT_INGEST_LIMIT)

    prompt = (
        "Call the ingest_file tool with exactly these arguments and nothing else.\n"
        f"- filename: {request.filename}\n"
        f"- file_base64: {request.file_base64}\n"
        f"{_tool_metadata_prompt_lines(request.metadata)}"
        "After the tool runs, return a short confirmation. Do not call other tools."
    )
    response = _invoke_agent(prompt, request.session_id, request.metadata)
    return AgentQueryResponse(
        answer=response.answer,
        citations=response.citations,
        tool_calls=response.tool_calls,
        session_id=response.session_id,
        run_id=response.run_id,
    )


@router.post("/ingest/url", response_model=AgentQueryResponse)
async def agent_ingest_url(request: AgentIngestURLRequest, http_request: Request) -> AgentQueryResponse:
    """Ask the agent to ingest a URL using the ingest_url tool."""

    _enforce_rate_limit(http_request, "agent_ingest_url", AGENT_INGEST_LIMIT)

    prompt = (
        "Call the ingest_url tool with exactly these arguments and nothing else.\n"
        f"- url: {request.url}\n"
        f"{_tool_metadata_prompt_lines(request.metadata)}"
        "After the tool runs, return a short confirmation. Do not call other tools."
    )
    response = _invoke_agent(prompt, request.session_id, request.metadata)
    return AgentQueryResponse(
        answer=response.answer,
        citations=response.citations,
        tool_calls=response.tool_calls,
        session_id=response.session_id,
        run_id=response.run_id,
    )


@router.post("/ingest/visual", response_model=AgentQueryResponse)
async def agent_ingest_visual(request: AgentIngestVisualRequest, http_request: Request) -> AgentQueryResponse:
    """Ask the agent to ingest a document using MarkItDown (visual mode default)."""

    _enforce_rate_limit(http_request, "agent_ingest_visual", AGENT_INGEST_LIMIT)

    prompt = (
        "Call the ingest_visual tool with exactly these arguments and nothing else.\n"
        f"- filename: {request.filename}\n"
        f"- file_base64: {request.file_base64}\n"
        f"- enable_visual_mode: {request.enable_visual_mode}\n"
        f"- llm_model: {request.llm_model}\n"
        f"{_tool_metadata_prompt_lines(request.metadata)}"
        "After the tool runs, return a short confirmation. Do not call other tools."
    )
    response = _invoke_agent(prompt, request.session_id, request.metadata)
    return AgentQueryResponse(
        answer=response.answer,
        citations=response.citations,
        tool_calls=response.tool_calls,
        session_id=response.session_id,
        run_id=response.run_id,
    )
