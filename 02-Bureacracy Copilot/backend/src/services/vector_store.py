"""
Vector Store Service

Manages ChromaDB vector database operations.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from functools import lru_cache
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.core.settings import get_settings
from src.core.logging_setup import get_logger
from src.services.embeddings import get_embeddings

logger = get_logger(__name__)
settings = get_settings()


@lru_cache()
def get_client() -> chromadb.PersistentClient:
    """
    Get Cached ChromaDB Client
    
    Creates a persistent ChromaDB client that stores data on disk.
    
    Returns:
        chromadb.PersistentClient: ChromaDB client instance
    """
    logger.info(f"Initializing ChromaDB client at {settings.chroma_persist_dir}")
    
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(
            anonymized_telemetry=False
        )
    )


def get_collection():
    """
    Get or Create ChromaDB Collection
    
    Returns:
        chromadb.Collection: The documents collection
    """
    client = get_client()
    embeddings = get_embeddings()
    
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"description": "Bureaucracy Copilot document store"}
    )
    
    return collection


def add_chunks(
    chunks: List[Dict[str, Any]],
    embeddings_list: List[List[float]],
    source_id: str
) -> int:
    """
    Add Document Chunks to Vector Store
    
    Args:
        chunks: List of chunks with text and metadata
        embeddings_list: List of embedding vectors
        source_id: Unique identifier for this source
    
    Returns:
        int: Number of chunks added
    """
    collection = get_collection()
    
    ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    logger.info(f"Adding {len(chunks)} chunks to vector store for source {source_id}")
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas
    )
    
    logger.info(f"Successfully added {len(chunks)} chunks")
    
    return len(chunks)


def list_sources() -> List[Dict[str, Any]]:
    """
    List All Sources in Knowledge Base
    
    Returns:
        List[Dict]: List of source information dictionaries
    """
    collection = get_collection()
    
    try:
        results = collection.get(include=["metadatas"])
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        return []
    
    # Group chunks by source_id
    sources_dict = defaultdict(lambda: {
        "chunk_count": 0,
        "char_count": 0
    })
    
    for metadata in results.get("metadatas", []):
        if not metadata:
            continue
        
        source_id = metadata.get("source_id")
        if not source_id:
            continue
        
        if "source_id" not in sources_dict[source_id]:
            sources_dict[source_id]["source_id"] = source_id
            sources_dict[source_id]["source_type"] = metadata.get("source_type", "unknown")
            sources_dict[source_id]["source_name"] = metadata.get("source_name", "Unknown")
            sources_dict[source_id]["timestamp"] = metadata.get("timestamp", "")
        
        sources_dict[source_id]["chunk_count"] += 1
        sources_dict[source_id]["char_count"] += metadata.get("char_count", 0)
    
    sources_list = list(sources_dict.values())
    
    logger.info(f"Found {len(sources_list)} sources in knowledge base")
    
    return sources_list


def clear_collection() -> int:
    """
    Clear All Documents from Knowledge Base
    
    Returns:
        int: Number of items deleted
    """
    client = get_client()
    collection_name = settings.chroma_collection
    
    logger.warning(f"Clearing collection: {collection_name}")
    
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        client.delete_collection(collection_name)
        logger.info(f"Deleted {count} items from knowledge base")
        return count
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        return 0


def similarity_search_with_score(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for Similar Documents with Relevance Scores
    
    Performs vector similarity search using the query text.
    Returns documents with their metadata and relevance scores.
    
    Args:
        query: User's query text
        top_k: Number of results to return (default: 5)
        where: Optional metadata filter for Chroma query
    
    Returns:
        List[Dict]: List of results with keys:
            - document: The chunk text
            - metadata: Chunk metadata (source_id, source_type, etc.)
            - id: Document ID
            - score: Relevance score (distance metric)
    """
    collection = get_collection()
    embeddings = get_embeddings()
    
    # Check if collection is empty
    try:
        count = collection.count()
        if count == 0:
            logger.warning("Collection is empty, returning no results")
            return []
    except Exception as e:
        logger.error(f"Error checking collection count: {e}")
        return []
    
    # Generate query embedding
    try:
        query_vector = embeddings.embed_query(query)
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise ValueError(f"Failed to generate embedding: {str(e)}")
    
    # Perform similarity search
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),  # Don't request more than available
            where=where,
        )
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        raise ValueError(f"Failed to search vector store: {str(e)}")
    
    # Parse results
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    # Build result list
    parsed_results = []
    for i in range(len(documents)):
        parsed_results.append({
            "document": documents[i],
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "id": ids[i] if i < len(ids) else "",
            "score": distances[i] if i < len(distances) else None
        })
    
    logger.info(f"Found {len(parsed_results)} results for query: '{query[:50]}...'")
    
    return parsed_results

