"""
Application configuration using Pydantic Settings.

Loads configuration from environment variables with type validation
and sensible defaults for development.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Server Configuration
    # =========================================================================
    app_name: str = Field(default="mira-memory-engine", description="Application name")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # =========================================================================
    # API Keys
    # =========================================================================
    deepgram_api_key: SecretStr = Field(
        default=SecretStr(""), description="Deepgram API key for STT"
    )
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), description="OpenAI API key for embeddings"
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""), description="Anthropic API key for LLM"
    )

    # =========================================================================
    # Vector Database Configuration (Pinecone only)
    # =========================================================================
    vector_db_provider: Literal["pinecone"] = Field(
        default="pinecone", description="Vector database provider"
    )

    # Pinecone settings
    pinecone_api_key: SecretStr = Field(
        default=SecretStr(""), description="Pinecone API key"
    )
    pinecone_environment: str = Field(
        default="us-east-1", description="Pinecone environment"
    )
    pinecone_index_name: str = Field(
        default="mira-memories", description="Pinecone index name"
    )

    # =========================================================================
    # Memory Pipeline Configuration
    # =========================================================================
    # Chunking
    chunk_strategy: Literal["sliding_window", "sentence", "paragraph", "semantic", "recursive"] = Field(
        default="sliding_window",
        description="Chunking strategy: sliding_window (fast), sentence (conversations), paragraph (docs), semantic (best quality), recursive (mixed)"
    )
    chunk_size: int = Field(
        default=150, ge=50, le=500, description="Target chunk size in words"
    )
    chunk_overlap: int = Field(
        default=30, ge=0, le=100, description="Chunk overlap in words"
    )
    semantic_similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Similarity threshold for semantic chunking"
    )

    # Embedding
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    embedding_dimensions: int = Field(
        default=1536, description="Embedding vector dimensions"
    )
    embedding_batch_size: int = Field(
        default=100, ge=1, le=2048, description="Batch size for embedding requests"
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to retrieve"
    )
    retrieval_score_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Minimum similarity score threshold"
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022", description="Claude model to use"
    )
    llm_max_tokens: int = Field(
        default=2048, ge=100, le=8192, description="Maximum tokens in response"
    )
    llm_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )

    # =========================================================================
    # STT Configuration
    # =========================================================================
    stt_model: str = Field(default="nova-2", description="Deepgram model")
    stt_language: str = Field(default="en", description="Speech recognition language")
    stt_punctuate: bool = Field(default=True, description="Enable auto-punctuation")
    stt_diarize: bool = Field(default=False, description="Enable speaker diarization")
    stt_smart_format: bool = Field(default=True, description="Enable smart formatting")

    # =========================================================================
    # Observability
    # =========================================================================
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP exporter endpoint"
    )
    otel_service_name: str = Field(
        default="mira-memory-engine", description="OpenTelemetry service name"
    )
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")

    # =========================================================================
    # CORS Configuration
    # =========================================================================
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:3001",
        description="Comma-separated list of allowed CORS origins (add your Vercel domain in production)",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
