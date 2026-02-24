"""
Text Chunking Service

Splits large documents into smaller chunks for embedding generation.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.settings import get_settings
from src.core.logging_setup import get_logger

logger = get_logger(__name__)
settings = get_settings()


def chunk_text(
    text: str,
    source_id: str,
    metadata: dict
) -> List[Document]:
    """
    Split Text into Chunks
    
    Uses LangChain's RecursiveCharacterTextSplitter which:
    1. Tries to split on paragraph boundaries first
    2. Falls back to sentence boundaries
    3. Falls back to word boundaries
    4. Falls back to character boundaries if needed
    
    This preserves semantic meaning better than naive splitting.
    
    Args:
        text: The text to split into chunks
        source_id: Unique identifier for this source document
        metadata: Metadata to attach to each chunk
    
    Returns:
        List[Document]: List of LangChain Document objects with text and metadata
    """
    logger.info(f"Chunking text for source {source_id}")
    
    # Create text splitter with configured settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the text
    chunks = text_splitter.split_text(text)
    
    # Create Document objects
    documents = []
    for idx, chunk in enumerate(chunks):
        chunk_metadata = {
            **metadata,
            "chunk_index": idx,
            "chunk_total": len(chunks),
            "char_count": len(chunk)  # Character count for this specific chunk
        }
        
        documents.append(Document(
            page_content=chunk,
            metadata=chunk_metadata
        ))
    
    logger.info(f"Created {len(documents)} chunks from {len(text)} characters")
    
    return documents
