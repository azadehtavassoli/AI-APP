"""Configuration utilities for the knowledge_mcp server."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def ensure_backend_path() -> Path:
    """Ensure the backend directory is importable so `src.*` modules resolve."""

    project_root = Path(__file__).resolve().parent.parent
    backend_dir = project_root / "backend"
    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    return backend_dir


class KnowledgeMCPSettings(BaseSettings):
    """Runtime settings for the Knowledge MCP server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    server_name: str = Field(default="knowledge_mcp", validation_alias="KNOWLEDGE_MCP_NAME")
    transport: str = Field(default="stdio", validation_alias="KNOWLEDGE_MCP_TRANSPORT")
    host: str = Field(default="127.0.0.1", validation_alias="KNOWLEDGE_MCP_HOST")
    port: int = Field(default=8765, validation_alias="KNOWLEDGE_MCP_PORT")
    log_level: str = Field(default="INFO", validation_alias="KNOWLEDGE_MCP_LOG_LEVEL")


@lru_cache()
def get_mcp_settings() -> KnowledgeMCPSettings:
    """Get a cached settings instance for MCP runtime configuration."""

    return KnowledgeMCPSettings()


def get_backend_runtime_info() -> dict[str, str]:
    """Load key backend runtime configuration details for diagnostics."""

    ensure_backend_path()
    from src.core.settings import get_settings  # pylint: disable=import-outside-toplevel

    backend_settings = get_settings()
    chroma_path = os.path.abspath(backend_settings.chroma_persist_dir)
    return {
        "chroma_persist_dir": chroma_path,
        "chroma_collection": backend_settings.chroma_collection,
        "embedding_model": backend_settings.openai_embedding_model,
    }
