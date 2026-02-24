"""
Embeddings Service

Generates vector embeddings for text using OpenAI's API.
"""

from functools import lru_cache
from langchain_openai import OpenAIEmbeddings

from src.core.settings import get_settings
from src.core.logging_setup import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    """
    Get Cached OpenAI Embeddings Instance
    
    Creates and caches an OpenAI embeddings client.
    Using @lru_cache ensures we reuse the same client
    instead of creating new API connections for each request.
    
    Returns:
        OpenAIEmbeddings: Configured embeddings client
    """
    logger.info(f"Initializing OpenAI embeddings with model: {settings.openai_embedding_model}")
    
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_embedding_model
    )
