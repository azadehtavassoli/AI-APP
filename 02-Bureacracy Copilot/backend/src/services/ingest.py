"""
Document Ingestion Service

Orchestrates the complete document processing pipeline.
"""

import base64
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown
from openai import OpenAI
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.settings import get_settings
from src.core.logging_setup import get_logger
from src.core.security import validate_text_length, validate_url, sanitize_source_name
from src.services.chunking import chunk_text
from src.services.embeddings import get_embeddings
from src.services.vector_store import add_chunks

logger = get_logger(__name__)
settings = get_settings()


SUPPORTED_FILTER_METADATA_FIELDS = ("city", "procedure", "language")


def _extract_filter_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Normalize ingestion metadata fields used for structured retrieval filters."""

    if not metadata:
        return {}

    normalized: Dict[str, str] = {}
    for field_name in SUPPORTED_FILTER_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            normalized[field_name] = value_str

    return normalized


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int) -> str:
    """
    Extract Text from PDF File
    
    Args:
        pdf_bytes: PDF file as bytes
        max_pages: Maximum number of pages to process
    
    Returns:
        str: Extracted text content
    """
    logger.info("Extracting text from PDF")
    
    try:
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        logger.info(f"PDF has {num_pages} pages")
        
        if num_pages > max_pages:
            raise ValueError(f"PDF has {num_pages} pages, maximum allowed is {max_pages}")
        
        text_parts = []
        for page_num, page in enumerate(reader.pages, start=1):
            logger.debug(f"Extracting text from page {page_num}/{num_pages}")
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        
        if not full_text.strip():
            raise ValueError("PDF contains no extractable text")
        
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        
        return full_text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to process PDF: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def fetch_url_text(url: str) -> str:
    """
    Fetch and Extract Text from URL
    
    Args:
        url: URL to fetch
    
    Returns:
        str: Extracted text content
    """
    logger.info(f"Fetching URL: {url}")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('text/html'):
            raise ValueError(f"URL does not return HTML content (got {content_type})")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        if not text.strip() or len(text.strip()) < 100:
            raise ValueError("Extracted text is too short or empty (possibly a non-text page)")
        
        logger.info(f"Extracted {len(text)} characters from URL")
        
        return text
    
    except requests.RequestException as e:
        logger.error(f"Error fetching URL: {e}")
        raise ValueError(f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing URL content: {e}")
        raise ValueError(f"Failed to process URL: {str(e)}")


def _build_markitdown_instance(
    enable_visual_mode: bool,
    llm_model: Optional[str]
) -> Tuple[MarkItDown, Optional[str]]:
    """Create a MarkItDown instance with optional LLM client."""

    if enable_visual_mode:
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required when visual mode is enabled")

        model_name = llm_model or settings.openai_model
        llm_client = OpenAI(api_key=settings.openai_api_key)
        return MarkItDown(llm_client=llm_client), model_name

    return MarkItDown(), None


def convert_bytes_to_markdown(
    content_bytes: bytes,
    filename: str,
    enable_visual_mode: bool = True,
    llm_model: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """Convert binary content to markdown using MarkItDown."""

    if not content_bytes:
        raise ValueError("File content cannot be empty")

    extension = Path(filename).suffix or ""
    if not extension:
        raise ValueError("Filename must include an extension (e.g., .pdf)")

    md, model_used = _build_markitdown_instance(enable_visual_mode, llm_model)

    try:
        markdown_result = md.convert_stream(BytesIO(content_bytes), file_extension=extension)
    except Exception as exc:  # pragma: no cover - runtime conversion errors
        logger.error(f"MarkItDown conversion failed: {exc}", exc_info=True)
        raise ValueError(f"Failed to convert document: {exc}")

    markdown_text = getattr(markdown_result, "text_content", None) or getattr(markdown_result, "markdown", None)
    if not markdown_text:
        markdown_text = str(markdown_result)

    metadata = {
        "extension": extension,
        "visual_mode": enable_visual_mode,
        "llm_model": model_used,
    }

    logger.info(
        "MarkItDown conversion completed",
        extra={"extension": extension, "visual_mode": enable_visual_mode, "markdown_chars": len(markdown_text)},
    )

    return markdown_text, metadata


def ingest_visual_document(
    file_bytes: bytes,
    filename: str,
    enable_visual_mode: bool = True,
    llm_model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a document (with visuals) to markdown and ingest into the vector store."""

    logger.info(
        "Starting visual ingestion",
        extra={"input_filename": filename, "visual_mode": enable_visual_mode, "llm_model": llm_model},
    )

    markdown_text, conversion_metadata = convert_bytes_to_markdown(
        content_bytes=file_bytes,
        filename=filename,
        enable_visual_mode=enable_visual_mode,
        llm_model=llm_model,
    )

    result = ingest_source(
        text=markdown_text,
        source_name=filename,
        source_type="visual_markdown",
        metadata=metadata,
    )

    result.update(
        {
            "markdown_length": len(markdown_text),
            "conversion": conversion_metadata,
        }
    )

    return result


def ingest_source(
    text: str,
    source_name: str,
    source_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ingest a Source into the Knowledge Base
    
    Args:
        text: Text content to ingest
        source_name: Human-readable name for the source
        source_type: Type of source ('text', 'pdf', 'url')
    
    Returns:
        Dict with ingestion results
    """
    logger.info(f"Starting ingestion for {source_type} source: {source_name}")
    
    # Validate input
    is_valid, error_message = validate_text_length(text, settings.max_text_length)
    if not is_valid:
        logger.warning(f"Validation failed: {error_message}")
        raise ValueError(error_message)
    
    # Generate source ID
    source_id = str(uuid.uuid4())
    logger.info(f"Generated source ID: {source_id}")
    
    # Create metadata
    timestamp = datetime.utcnow().isoformat()
    filter_metadata = _extract_filter_metadata(metadata)
    source_metadata = {
        "source_id": source_id,
        "source_type": source_type,
        "source_name": sanitize_source_name(source_name),
        "timestamp": timestamp,
        "char_count": len(text),
        **filter_metadata,
    }
    
    # Chunk text
    document_chunks = chunk_text(text, source_id, source_metadata)
    logger.info(f"Created {len(document_chunks)} chunks")
    
    # Generate embeddings
    embeddings_model = get_embeddings()
    texts_to_embed = [doc.page_content for doc in document_chunks]
    
    logger.info("Generating embeddings...")
    embeddings_list = embeddings_model.embed_documents(texts_to_embed)
    logger.info(f"Generated {len(embeddings_list)} embeddings")
    
    # Store in vector database
    chunks_for_storage = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in document_chunks
    ]
    
    stored_count = add_chunks(chunks_for_storage, embeddings_list, source_id)
    
    logger.info(f"Successfully ingested source {source_id}")
    
    return {
        "source_id": source_id,
        "chunk_count": stored_count,
        "char_count": len(text),
        "timestamp": timestamp,
        "source_type": source_type,
        "source_name": source_metadata["source_name"],
        "metadata": filter_metadata,
    }
