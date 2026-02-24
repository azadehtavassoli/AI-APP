"""
Pydantic Models for API Request/Response Validation

These models define the structure of data sent to and from the API.
"""

from typing import Optional, List
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


# === Enums ===

class SourceType(str, Enum):
    """Source Type Enumeration"""
    TEXT = "text"
    PDF = "pdf"
    URL = "url"
    VISUAL = "visual_markdown"


# === Request Models ===

class TextIngestRequest(BaseModel):
    """Request Model for Text Ingestion"""
    text: str = Field(..., min_length=1, description="Text content to ingest")
    source_name: str = Field(..., min_length=1, max_length=200, description="Human-readable name for this source")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


class PDFIngestRequest(BaseModel):
    """Request Model for PDF Ingestion"""
    pdf_base64: str = Field(..., description="PDF file content encoded as base64 string")
    filename: str = Field(..., min_length=1, max_length=200, description="Original filename of the PDF")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


class URLIngestRequest(BaseModel):
    """Request Model for URL Ingestion"""
    url: str = Field(..., pattern=r'^https?://.+', description="URL of the web page to ingest")
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


class VisualIngestRequest(BaseModel):
    """Request Model for MarkItDown Visual/LLM Ingestion"""

    file_base64: str = Field(..., description="Base64-encoded file content to convert")
    filename: str = Field(..., min_length=1, max_length=200, description="Original filename including extension")
    enable_visual_mode: bool = Field(
        default=True,
        description="Whether to enable MarkItDown visual mode with LLM support"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Optional LLM model override when visual mode is enabled"
    )
    city: Optional[str] = Field(default=None, description="Optional city metadata filter value")
    procedure: Optional[str] = Field(default=None, description="Optional procedure metadata filter value")
    language: Optional[str] = Field(default=None, description="Optional language metadata filter value")


class ChatRequest(BaseModel):
    """Request Model for Chat/Q&A"""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    language: str = Field(default="en", pattern=r"^(en|de|fa)$", description="Response language")


class QueryRequest(BaseModel):
    """Request Model for RAG Query with Citations"""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    chat_history: Optional[List[dict]] = Field(default=None, description="Previous conversation messages")
    response_language: str = Field(default="auto", pattern=r"^(auto|en|de|fa)$", description="Response language (auto=same as question)")
    model: Optional[str] = Field(default=None, description="Optional OpenAI model override for this query")


# === Response Models ===

class IngestResponse(BaseModel):
    """Response Model for Ingestion Operations"""
    success: bool = Field(description="Whether ingestion succeeded")
    source: dict = Field(description="Source summary information")
    message: str = Field(description="Status message")


class SourceInfo(BaseModel):
    """Information About a Single Source"""
    source_id: str = Field(description="Unique source ID")
    source_type: SourceType = Field(description="Type of source")
    source_name: str = Field(description="Human-readable source name")
    uri: str = Field(description="Source URI (URL for web sources, filename for files)")
    chunk_count: int = Field(description="Number of chunks")
    char_count: int = Field(description="Character count")
    timestamp: str = Field(description="When ingested (ISO format)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ListSourcesResponse(BaseModel):
    """Response Model for Listing Sources"""
    sources: List[SourceInfo] = Field(default_factory=list, description="List of all sources")
    total_count: int = Field(description="Total number of sources")


class ClearResponse(BaseModel):
    """Response Model for Clear Operation"""
    success: bool = Field(description="Whether operation succeeded")
    message: str = Field(description="Status message")


class HealthResponse(BaseModel):
    """Response Model for Health Check"""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    openai_configured: bool = Field(description="Whether OpenAI API key is configured")


class ChatResponse(BaseModel):
    """Response Model for Chat/Q&A"""
    answer: str = Field(description="LLM-generated answer")
    sources: List[str] = Field(default_factory=list, description="Source IDs used")
    confidence: Optional[float] = Field(default=None, description="Confidence score")


class Citation(BaseModel):
    """Citation Model for RAG Responses"""
    source_id: str = Field(description="Unique source identifier")
    source_type: str = Field(description="Type of source (text/pdf/url)")
    source_uri: str = Field(description="Source URI or filename")
    chunk_id: str = Field(description="Chunk identifier")
    snippet: str = Field(description="Relevant text snippet (200-400 chars)")
    score: Optional[float] = Field(default=None, description="Relevance score if available")


class ToolResult(BaseModel):
    """Tool Call Result Model"""
    tool_name: str = Field(description="Name of the tool that was called")
    tool_input: dict = Field(description="Input parameters passed to the tool")
    tool_output: dict = Field(description="Output returned by the tool")
    success: bool = Field(description="Whether the tool call succeeded")
    error: Optional[str] = Field(default=None, description="Error message if tool call failed")


class TokenUsage(BaseModel):
    """Token Usage Metadata for LLM Calls."""

    prompt_tokens: int = Field(default=0, description="Input/prompt token count")
    completion_tokens: int = Field(default=0, description="Output/completion token count")
    total_tokens: int = Field(default=0, description="Total token count")


class QueryResponse(BaseModel):
    """Response Model for RAG Query with Citations"""
    answer: str = Field(description="LLM-generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    tool_results: Optional[List[ToolResult]] = Field(default=None, description="Results from tool calls (if any)")
    model_used: Optional[str] = Field(default=None, description="Model used for answer generation")
    token_usage: Optional[TokenUsage] = Field(default=None, description="LLM token usage for this response")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated LLM cost in USD")


class ErrorResponse(BaseModel):
    """Error Response Model"""
    error: str = Field(description="Error type/message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
