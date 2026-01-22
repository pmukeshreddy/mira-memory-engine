"""
REST API routes for the Mira Memory Engine.

Provides endpoints for memory ingestion, querying, and system management.
"""

from datetime import datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.deps import (
    PipelineDep,
    SettingsDep,
    VectorDBDep,
)
from app.utils.uptime import get_uptime
from app.models.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MemoryResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RecentMemoriesResponse,
)
from app.utils.latency import get_tracker

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Memory"])


# =============================================================================
# Health & Status
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its dependencies.",
)
async def health_check(
    vector_db: VectorDBDep,
    settings: SettingsDep,
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and dependency health.
    """
    services: dict[str, bool] = {}

    # Check vector DB
    try:
        count = await vector_db.get_count()
        services["vector_db"] = True
    except Exception:
        services["vector_db"] = False

    # Check API keys are configured
    services["deepgram"] = bool(settings.deepgram_api_key.get_secret_value())
    services["openai"] = bool(settings.openai_api_key.get_secret_value())
    services["anthropic"] = bool(settings.anthropic_api_key.get_secret_value())

    # Determine overall status
    all_healthy = all(services.values())
    critical_healthy = services.get("vector_db", False) and services.get("openai", False)

    if all_healthy:
        status_val = "healthy"
    elif critical_healthy:
        status_val = "degraded"
    else:
        status_val = "unhealthy"

    return HealthResponse(
        status=status_val,
        version=__version__,
        timestamp=datetime.utcnow(),
        services=services,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Application metrics",
    description="Get application metrics including latency percentiles.",
)
async def get_metrics(
    pipeline: PipelineDep,
    vector_db: VectorDBDep,
) -> MetricsResponse:
    """
    Get application metrics.

    Returns uptime, counts, and latency metrics.
    """
    tracker = get_tracker()
    latency_metrics = tracker.get_all_metrics()

    # Convert to response format
    from app.models.schemas import LatencyMetrics, QualityMetrics

    latencies = [
        LatencyMetrics(
            operation=m["operation"],
            p50_ms=m["p50_ms"],
            p95_ms=m["p95_ms"],
            p99_ms=m["p99_ms"],
            count=m["count"],
        )
        for m in latency_metrics
    ]

    memory_count = await vector_db.get_count()

    # Quality metrics
    quality = QualityMetrics(
        hit_rate=round(pipeline.hit_rate * 100, 1),  # As percentage
        avg_retrieval_score=round(pipeline.avg_retrieval_score * 100, 1),  # As percentage
        avg_memories_per_query=round(pipeline.avg_memories_per_query, 1),
        total_words_ingested=pipeline.total_words_ingested,
        total_chunks_created=pipeline.total_chunks_created,
    )

    return MetricsResponse(
        uptime_seconds=get_uptime(),
        total_memories=memory_count,
        total_queries=pipeline.query_count,
        total_ingests=pipeline.ingest_count,
        latencies=latencies,
        quality=quality,
    )


# =============================================================================
# Memory Operations
# =============================================================================


@router.post(
    "/memory/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest text into memory",
    description="Process and store text content in the memory system.",
)
async def ingest_memory(
    request: IngestRequest,
    pipeline: PipelineDep,
) -> IngestResponse:
    """
    Ingest text into the memory store.

    Processes the text through the pipeline:
    1. Chunk into semantic units
    2. Generate embeddings
    3. Store in vector database
    """
    logger.info(
        "ingest_request",
        text_length=len(request.text),
        source=request.source,
    )

    result = await pipeline.ingest(
        text=request.text,
        source=request.source,
        session_id=request.session_id,
        metadata=request.metadata,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Ingest failed",
        )

    return IngestResponse(
        success=result.success,
        memory_ids=result.memory_ids,
        chunks_created=result.chunks_created,
        processing_time_ms=result.processing_time_ms,
        latency_breakdown=result.latency_breakdown,
    )


@router.post(
    "/memory/query",
    response_model=QueryResponse,
    summary="Query memories",
    description="Query the memory store and get an AI-generated response.",
)
async def query_memory(
    request: QueryRequest,
    pipeline: PipelineDep,
) -> QueryResponse:
    """
    Query memories and generate a response.

    Pipeline:
    1. Embed the query
    2. Search for relevant memories
    3. Generate response using LLM
    """
    logger.info(
        "query_request",
        query_preview=request.query[:50],
        top_k=request.top_k,
    )

    # Handle streaming requests separately
    if request.stream:
        # Return streaming response
        async def generate_stream():
            result = await pipeline.query(
                query=request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                stream=False,  # Get memories first
            )

            # First, yield context if requested
            if request.include_context:
                for memory in result.memories:
                    yield f"data: {{'type': 'context', 'memory_id': '{memory.memory_id}', 'score': {memory.score}}}\n\n"

            # Then stream the response
            async for token in pipeline.llm_service.generate_stream(
                request.query, result.memories
            ):
                yield f"data: {{'type': 'token', 'content': '{token}'}}\n\n"

            yield "data: {\"type\": \"done\"}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )

    # Non-streaming query
    result = await pipeline.query(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        stream=False,
    )

    if result.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error,
        )

    # Convert memories to response format
    memories = [
        MemoryResponse(
            id=m.memory_id,
            text=m.text,
            score=m.score,
            metadata=m.metadata,
        )
        for m in result.memories
    ] if request.include_context else []

    return QueryResponse(
        answer=result.answer,
        memories=memories,
        query=result.query,
        processing_time_ms=result.processing_time_ms,
        latency_breakdown=result.latency_breakdown,
    )


@router.get(
    "/memory/recent",
    response_model=RecentMemoriesResponse,
    summary="Get recent memories",
    description="Retrieve the most recently stored memories.",
)
async def get_recent_memories(
    pipeline: PipelineDep,
    vector_db: VectorDBDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> RecentMemoriesResponse:
    """
    Get recently stored memories.

    Returns the most recent memories up to the specified limit.
    """
    memories = await pipeline.get_recent_memories(limit=limit)
    total_count = await vector_db.get_count()
    
    logger.info("recent_memories_fetched", count=len(memories), total=total_count)

    memory_responses = [
        MemoryResponse(
            id=m.memory_id,
            text=m.text,
            score=m.score,
            metadata=m.metadata,
        )
        for m in memories
    ]

    return RecentMemoriesResponse(
        memories=memory_responses,
        total_count=total_count,
        limit=limit,
    )


@router.delete(
    "/memory/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear all memories",
    description="Delete all stored memories from the system.",
)
async def clear_memories(
    pipeline: PipelineDep,
) -> None:
    """
    Clear all stored memories.

    Warning: This action is irreversible.
    """
    logger.warning("clear_memories_requested")

    success = await pipeline.clear_memories()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear memories",
        )

    logger.info("memories_cleared")


# =============================================================================
# Search Endpoint (Direct vector search without LLM)
# =============================================================================


@router.post(
    "/memory/search",
    response_model=list[MemoryResponse],
    summary="Search memories",
    description="Search for relevant memories without generating an LLM response.",
)
async def search_memories(
    query: str,
    pipeline: PipelineDep,
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    score_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> list[MemoryResponse]:
    """
    Search for relevant memories.

    Returns matching memories without LLM generation.
    Useful for direct retrieval operations.
    """
    # Embed the query
    query_embedding = await pipeline.embedding_service.embed_text(query)

    # Search
    memories = await pipeline.vector_db.search(
        embedding=query_embedding,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    return [
        MemoryResponse(
            id=m.memory_id,
            text=m.text,
            score=m.score,
            metadata=m.metadata,
        )
        for m in memories
    ]
