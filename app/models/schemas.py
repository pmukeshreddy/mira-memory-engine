"""
Pydantic schemas for API request/response validation.

These models define the contract between the API and its clients,
ensuring type safety and automatic documentation.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Health & Metrics
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Service health status"
    )
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Check timestamp"
    )
    services: dict[str, bool] = Field(
        default_factory=dict, description="Individual service health status"
    )


class LatencyMetrics(BaseModel):
    """Latency metrics for a single operation."""

    operation: str = Field(..., description="Operation name")
    p50_ms: float = Field(..., description="50th percentile latency in ms")
    p95_ms: float = Field(..., description="95th percentile latency in ms")
    p99_ms: float = Field(..., description="99th percentile latency in ms")
    count: int = Field(..., description="Total operation count")


class QualityMetrics(BaseModel):
    """Quality metrics for the retrieval system."""
    
    hit_rate: float = Field(..., description="% of queries that found relevant memories")
    avg_retrieval_score: float = Field(..., description="Average similarity score of retrieved memories")
    avg_memories_per_query: float = Field(..., description="Average memories retrieved per query")
    total_words_ingested: int = Field(..., description="Total words processed")
    total_chunks_created: int = Field(..., description="Total memory chunks created")


class MetricsResponse(BaseModel):
    """Application metrics response."""

    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    total_memories: int = Field(..., description="Total memories stored")
    total_queries: int = Field(..., description="Total queries processed")
    total_ingests: int = Field(..., description="Total ingest operations")
    latencies: list[LatencyMetrics] = Field(
        default_factory=list, description="Latency metrics by operation"
    )
    quality: QualityMetrics | None = Field(
        default=None, description="Quality metrics for retrieval"
    )


# =============================================================================
# Memory Ingest
# =============================================================================


class IngestRequest(BaseModel):
    """Request to ingest text into memory."""

    text: str = Field(
        ..., min_length=1, max_length=100000, description="Text content to ingest"
    )
    source: str = Field(default="text", description="Source identifier")
    session_id: str | None = Field(
        default=None, description="Session ID for grouping related memories"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = {"json_schema_extra": {"example": {"text": "The meeting discussed Q4 revenue projections and marketing strategy for the new product launch.", "source": "meeting_notes", "session_id": "meeting-2024-01-15", "metadata": {"department": "sales"}}}}


class IngestResponse(BaseModel):
    """Response after ingesting text."""

    success: bool = Field(..., description="Whether ingest was successful")
    memory_ids: list[str] = Field(
        default_factory=list, description="IDs of created memories"
    )
    chunks_created: int = Field(..., description="Number of chunks created")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    latency_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Latency breakdown by stage"
    )


# =============================================================================
# Memory Query
# =============================================================================


class QueryRequest(BaseModel):
    """Request to query memories with RAG."""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="Query text"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    score_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score (uses config default if not provided)"
    )
    include_context: bool = Field(
        default=True, description="Include retrieved context in response"
    )
    stream: bool = Field(default=False, description="Stream the LLM response")

    model_config = {"json_schema_extra": {"example": {"query": "What were the Q4 revenue projections?", "top_k": 5, "include_context": True, "stream": False}}}


class MemoryResponse(BaseModel):
    """A single memory result."""

    id: str = Field(..., description="Memory ID")
    text: str = Field(..., description="Memory content")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Memory metadata"
    )


class QueryResponse(BaseModel):
    """Response to a memory query."""

    answer: str = Field(..., description="Generated answer from LLM")
    memories: list[MemoryResponse] = Field(
        default_factory=list, description="Retrieved memories"
    )
    query: str = Field(..., description="Original query")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    latency_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Latency breakdown by stage"
    )


# =============================================================================
# Recent Memories
# =============================================================================


class RecentMemoriesResponse(BaseModel):
    """Response containing recent memories."""

    memories: list[MemoryResponse] = Field(
        default_factory=list, description="Recent memories"
    )
    total_count: int = Field(..., description="Total memories in database")
    limit: int = Field(..., description="Limit applied to results")


# =============================================================================
# WebSocket Messages
# =============================================================================


class WSAudioMessage(BaseModel):
    """WebSocket message containing audio data."""

    type: Literal["audio"] = Field(default="audio", description="Message type")
    data: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(default=16000, description="Audio sample rate")
    encoding: str = Field(default="linear16", description="Audio encoding format")


class WSTranscriptMessage(BaseModel):
    """WebSocket message containing transcript data."""

    type: Literal["transcript"] = Field(default="transcript", description="Message type")
    text: str = Field(..., description="Transcribed text")
    is_final: bool = Field(..., description="Whether this is a final result")
    confidence: float = Field(default=1.0, description="Transcription confidence")
    timestamp: float = Field(..., description="Timestamp in seconds")


class WSQueryMessage(BaseModel):
    """WebSocket message for query request."""

    type: Literal["query"] = Field(default="query", description="Message type")
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, description="Number of results")


class WSResponseMessage(BaseModel):
    """WebSocket message for streaming LLM response."""

    type: Literal["response", "context", "done", "error"] = Field(
        ..., description="Message type"
    )
    content: str = Field(default="", description="Message content")
    token_index: int | None = Field(default=None, description="Token index for streaming")
    is_complete: bool = Field(default=False, description="Whether response is complete")
    error: str | None = Field(default=None, description="Error message if applicable")
