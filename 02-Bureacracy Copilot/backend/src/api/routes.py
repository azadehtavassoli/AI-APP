"""
API Routes and Endpoints

Defines all HTTP endpoints for the backend API.
"""

import base64
from fastapi import APIRouter, HTTPException, status

from src.core.settings import get_settings
from src.core.logging_setup import get_logger
from src.core.security import validate_url, sanitize_filename, PII_WARNING
from src.models.schemas import (
    TextIngestRequest,
    PDFIngestRequest,
    URLIngestRequest,
    VisualIngestRequest,
    IngestResponse,
    ListSourcesResponse,
    SourceInfo,
    ClearResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse
)
from src.services.ingest import (
    ingest_source,
    extract_text_from_pdf,
    fetch_url_text,
    ingest_visual_document
)
from src.services.vector_store import list_sources, clear_collection
from src.services.rag import generate_answer

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health Check Endpoint"""
    logger.info("Health check requested")
    openai_configured = bool(settings.openai_api_key and settings.openai_api_key.strip())
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        openai_configured=openai_configured
    )


@router.post("/ingest/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(request: TextIngestRequest):
    """Ingest Text Content"""
    logger.info(f"Text ingestion requested: {request.source_name}")
    
    try:
        result = ingest_source(
            text=request.text,
            source_name=request.source_name,
            source_type="text",
            metadata={
                "city": request.city,
                "procedure": request.procedure,
                "language": request.language,
            },
        )
        
        return IngestResponse(
            success=True,
            source={
                "id": result["source_id"],
                "type": "text",
                "uri": request.source_name,
                "chunks": result["chunk_count"],
                "characters": result["char_count"],
                "added_at": result["timestamp"],
                "metadata": {
                    "original_length": len(request.text),
                    "used_length": len(request.text),
                    "truncated": False,
                    "extraction_method": None
                }
            },
            message="Successfully ingested text document"
        )
    
    except ValueError as e:
        logger.warning(f"Validation error during text ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during text ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text: {str(e)}"
        )


@router.post("/ingest/pdf", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(request: PDFIngestRequest):
    """Ingest PDF File"""
    logger.info(f"PDF ingestion requested: {request.filename}")
    
    try:
        safe_filename = sanitize_filename(request.filename)
        
        try:
            pdf_bytes = base64.b64decode(request.pdf_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        
        extracted_text = extract_text_from_pdf(pdf_bytes, settings.max_pdf_pages)
        
        result = ingest_source(
            text=extracted_text,
            source_name=safe_filename,
            source_type="pdf",
            metadata={
                "city": request.city,
                "procedure": request.procedure,
                "language": request.language,
            },
        )
        
        return IngestResponse(
            success=True,
            source={
                "id": result["source_id"],
                "type": "pdf",
                "uri": safe_filename,
                "chunks": result["chunk_count"],
                "characters": result["char_count"],
                "added_at": result["timestamp"],
                "metadata": {
                    "original_length": len(extracted_text),
                    "used_length": len(extracted_text),
                    "truncated": False,
                    "extraction_method": "pypdf"
                }
            },
            message=f"Successfully ingested PDF: {safe_filename}"
        )
    
    except ValueError as e:
        logger.warning(f"Validation error during PDF ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest PDF: {str(e)}"
        )


@router.post("/ingest/url", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(request: URLIngestRequest):
    """Ingest Web Page from URL"""
    logger.info(f"URL ingestion requested: {request.url}")
    
    try:
        is_valid, error_msg = validate_url(request.url)
        if not is_valid:
            raise ValueError(error_msg)
        
        extracted_text = fetch_url_text(request.url)
        logger.info(f"Fetched text length: {len(extracted_text)} chars")
        source_name = request.url
        
        # Handle length limits with truncation
        original_length = len(extracted_text)
        truncated = False
        if original_length > settings.max_text_length:
            extracted_text = extracted_text[:settings.max_text_length]
            truncated = True
            logger.warning(f"URL content truncated: {original_length} → {len(extracted_text)} chars")
        
        logger.info(f"After truncation: {len(extracted_text)} chars, truncated: {truncated}")
        
        result = ingest_source(
            text=extracted_text,
            source_name=source_name,
            source_type="url",
            metadata={
                "city": request.city,
                "procedure": request.procedure,
                "language": request.language,
            },
        )
        
        logger.info(f"Ingestion result: source_id={result['source_id']}, chunks={result['chunk_count']}, chars={result['char_count']}")
        
        # Add truncation metadata
        result["original_length"] = original_length
        result["used_length"] = len(extracted_text)
        result["truncated"] = truncated
        
        return IngestResponse(
            success=True,
            source={
                "id": result["source_id"],
                "type": "url",
                "uri": request.url,
                "chunks": result["chunk_count"],
                "characters": result["char_count"],
                "added_at": result["timestamp"],
                "metadata": {
                    "original_length": result.get("original_length", len(extracted_text)),
                    "used_length": result.get("used_length", len(extracted_text)),
                    "truncated": result.get("truncated", False),
                    "extraction_method": "bs4"
                }
            },
            message=f"Successfully ingested URL content"
        )
    
    except ValueError as e:
        logger.warning(f"Validation error during URL ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during URL ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest URL: {str(e)}"
        )


@router.post("/ingest/visual", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_visual(request: VisualIngestRequest):
    """Ingest a document using MarkItDown with optional visual/LLM mode."""

    logger.info(
        "Visual ingestion requested",
        extra={"input_filename": request.filename, "visual_mode": request.enable_visual_mode, "llm_model": request.llm_model},
    )

    try:
        try:
            file_bytes = base64.b64decode(request.file_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")

        result = ingest_visual_document(
            file_bytes=file_bytes,
            filename=request.filename,
            enable_visual_mode=request.enable_visual_mode,
            llm_model=request.llm_model,
            metadata={
                "city": request.city,
                "procedure": request.procedure,
                "language": request.language,
            },
        )

        return IngestResponse(
            success=True,
            source={
                "id": result["source_id"],
                "type": "visual_markdown",
                "uri": request.filename,
                "chunks": result["chunk_count"],
                "characters": result["char_count"],
                "added_at": result["timestamp"],
                "metadata": {
                    "extension": result.get("conversion", {}).get("extension"),
                    "visual_mode": result.get("conversion", {}).get("visual_mode"),
                    "llm_model": result.get("conversion", {}).get("llm_model"),
                    "markdown_length": result.get("markdown_length"),
                    "extraction_method": "markitdown",
                },
            },
            message="Successfully ingested document with markdown + visual descriptions",
        )

    except ValueError as e:
        logger.warning(f"Validation error during visual ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during visual ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest visual document: {str(e)}"
        )


@router.get("/sources", response_model=ListSourcesResponse)
async def get_sources():
    """List All Sources"""
    logger.info("Source listing requested")
    
    try:
        sources = list_sources()
        
        source_infos = [
            SourceInfo(
                source_id=s["source_id"],
                source_type=s["source_type"],
                source_name=s["source_name"],
                uri=s["source_name"],  # For URLs this is the URL, for others the name
                chunk_count=s["chunk_count"],
                char_count=s["char_count"],
                timestamp=s["timestamp"],
                metadata={}
            )
            for s in sources
        ]
        
        return ListSourcesResponse(
            sources=source_infos,
            total_count=len(source_infos)
        )
    
    except Exception as e:
        logger.error(f"Error listing sources: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sources: {str(e)}"
        )


@router.post("/clear", response_model=ClearResponse)
async def clear_knowledge_base():
    """Clear Knowledge Base"""
    logger.warning("Knowledge base clear requested")
    
    try:
        deleted_count = clear_collection()
        
        return ClearResponse(
            success=True,
            message=f"Cleared {deleted_count} documents from knowledge base"
        )
    
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear knowledge base: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(request: QueryRequest):
    """
    Query Documents with RAG and Tool Calling
    
    Retrieves relevant document chunks and generates an answer using LLM.
    Returns answer with citations and tool results (if tools were called).
    
    **Processing Pipeline:**
    1. Validate question
    2. Retrieve similar chunks from vector store
    3. Build context from chunks
    4. Call tools based on question type (NEW in v2.1)
    5. Generate answer via OpenAI LLM (with context + tool results)
    6. Build citations with source details
    
    **Response includes:**
    - Generated answer
    - Citations with source details and snippets
    - Tool results (optional, if tools were called)
    
    **Tool Calling (v2.1):**
    - Deadline calculations
    - Date/entity extraction
    - Appointment link finding
    - Token/cost estimation
    - Structured summaries
    - Email drafting
    
    **Behavior when no documents exist:**
    - Returns helpful message to ingest documents
    - Returns empty citations list
    """
    logger.info(f"Query requested: '{request.question[:100]}...'")
    
    try:
        # Validate input
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")
        
        # Limit top_k to reasonable range
        top_k = max(1, min(request.top_k, 20))
        
        # Generate answer using RAG
        result = generate_answer(
            question=request.question,
            top_k=top_k,
            chat_history=request.chat_history,
            response_language=request.response_language,
            model_override=request.model,
        )
        
        logger.info(f"Query completed: {len(result['answer'])} chars, {len(result['citations'])} citations")
        
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            tool_results=result.get("tool_results"),  # NEW in v2.1
            model_used=result.get("model_used"),
            token_usage=result.get("token_usage"),
            estimated_cost_usd=result.get("estimated_cost_usd")
        )
    
    except ValueError as e:
        logger.warning(f"Validation error during query: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

