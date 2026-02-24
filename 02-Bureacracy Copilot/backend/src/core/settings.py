"""
Application settings and configuration management

Uses Pydantic Settings for type-safe configuration loaded from environment variables.
All settings can be overridden via .env file or system environment variables.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings
    
    This class defines all configuration for the backend service.
    Values are loaded from environment variables (or .env file).
    
    Why Pydantic Settings?
    - Type validation: Ensures correct data types
    - Default values: Provides sensible defaults
    - Environment variables: Easy configuration without code changes
    - IDE support: Type hints for autocomplete
    """
    
    # Configuration for loading from .env file
    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env file in backend/ directory
        env_file_encoding="utf-8",
        case_sensitive=True,  # Environment variable names are case-sensitive
        extra="ignore"  # Ignore unknown environment variables
    )
    
    # === OpenAI Configuration ===
    # Your OpenAI API key for embeddings and LLM
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key (required)"
    )
    
    # LLM model for chat responses
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_MODEL",
        description="OpenAI chat model"
    )
    
    # Embedding model for vector generation
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
        description="OpenAI embedding model"
    )
    
    # === Document Processing Limits ===
    # Maximum text length for direct text input (characters)
    max_text_length: int = Field(
        default=100000,
        validation_alias="MAX_TEXT_LENGTH",
        description="Max characters for text input"
    )
    
    # Maximum PDF pages to process
    max_pdf_pages: int = Field(
        default=10,
        validation_alias="MAX_PDF_PAGES",
        description="Max pages for PDF processing"
    )
    
    # === Text Chunking Configuration ===
    # Chunk size: How many characters per chunk
    # Smaller chunks: More precise search, more chunks, higher cost
    # Larger chunks: More context, fewer chunks, lower cost
    chunk_size: int = Field(
        default=1000,
        validation_alias="CHUNK_SIZE",
        description="Text chunk size in characters"
    )
    
    # Chunk overlap: How much chunks overlap
    # Overlap helps preserve context across chunk boundaries
    chunk_overlap: int = Field(
        default=200,
        validation_alias="CHUNK_OVERLAP",
        description="Character overlap between chunks"
    )
    
    # === Vector Store Configuration ===
    # Directory where ChromaDB stores its data
    chroma_persist_dir: str = Field(
        default="./src/data/chroma",
        validation_alias="CHROMA_PERSIST_DIR",
        description="ChromaDB data directory"
    )
    
    # Collection name in ChromaDB
    chroma_collection: str = Field(
        default="documents",
        validation_alias="CHROMA_COLLECTION",
        description="ChromaDB collection name"
    )
    
    # === Server Configuration ===
    # Host to bind the server to
    host: str = Field(
        default="0.0.0.0",
        validation_alias="HOST",
        description="Server host"
    )
    
    # Port to listen on
    port: int = Field(
        default=8000,
        validation_alias="PORT",
        description="Server port"
    )
    
    # CORS allowed origins (comma-separated)
    cors_origins: str = Field(
        default="http://localhost:8501,http://localhost:3000",
        validation_alias="CORS_ORIGINS",
        description="Allowed CORS origins"
    )
    
    # === Logging Configuration ===
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    agent_log_file: str = Field(
        default="./logs/agent_runs.log",
        validation_alias="AGENT_LOG_FILE",
        description="Path to the structured agent log file"
    )

    agent_log_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        validation_alias="AGENT_LOG_MAX_BYTES",
        description="Maximum size for agent log before rotation"
    )

    agent_log_backup_count: int = Field(
        default=5,
        validation_alias="AGENT_LOG_BACKUP_COUNT",
        description="Number of rotated agent logs to retain"
    )

    agent_log_retention_days: int = Field(
        default=7,
        validation_alias="AGENT_LOG_RETENTION_DAYS",
        description="Retention window in days for agent log files"
    )

    agent_use_knowledge_mcp: bool = Field(
        default=False,
        validation_alias="AGENT_USE_KNOWLEDGE_MCP",
        description="Enable MCP-backed KB tools for the agent"
    )

    knowledge_mcp_url: str = Field(
        default="http://127.0.0.1:8765/mcp",
        validation_alias="KNOWLEDGE_MCP_URL",
        description="Streamable HTTP endpoint for knowledge_mcp server"
    )
    
    def get_cors_origins_list(self) -> list[str]:
        """
        Get CORS Origins as List
        
        Converts the comma-separated CORS origins string into a list.
        
        Returns:
            list[str]: List of allowed origins
        """
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """
    Get Cached Settings Instance
    
    Why use @lru_cache?
    - Settings are loaded only once (efficient)
    - Same instance used throughout the app (consistency)
    - No repeated file reads or environment lookups
    
    How it works:
    1. First call: Creates Settings instance, caches it
    2. Subsequent calls: Returns cached instance immediately
    
    Testing note:
    - Call get_settings.cache_clear() in tests to reset
    
    Returns:
        Settings: The cached settings instance
    """
    return Settings()


def validate_api_key() -> bool:
    """
    Validate OpenAI API Key Configuration
    
    Checks if the OpenAI API key is configured and non-empty.
    Used during startup to verify configuration before processing requests.
    
    Returns:
        bool: True if API key is configured, False otherwise
    """
    settings = get_settings()
    return bool(settings.openai_api_key and settings.openai_api_key.strip())
