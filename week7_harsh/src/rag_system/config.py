"""
Centralised configuration via pydantic-settings.
Values are loaded from environment variables or a .env file.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_dimensions: int = 1536

    # Embedding backend
    embedding_backend: str = "openai"  # "openai" | "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection: str = "rag_documents"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    bm25_weight: float = 0.4

    # Generation
    max_answer_tokens: int = 1024
    llm_temperature: float = 0.0

    # Logging
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = Settings()
