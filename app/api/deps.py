"""
Dependency injection for API endpoints.

Provides reusable dependencies for FastAPI routes including
service instances and configuration.
"""

from typing import Annotated, AsyncGenerator

import structlog
from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.core.memory import MemoryPipeline
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vectordb import VectorDBService

logger = structlog.get_logger(__name__)


# Type aliases for cleaner annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_vector_db(request: Request) -> VectorDBService:
    """
    Get vector database service from app state.

    Args:
        request: FastAPI request object

    Returns:
        VectorDBService instance

    Raises:
        HTTPException: If service not initialized
    """
    if not hasattr(request.app.state, "vector_db"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service not initialized",
        )
    return request.app.state.vector_db


def get_embedding_service(request: Request) -> EmbeddingService:
    """
    Get embedding service from app state.

    Args:
        request: FastAPI request object

    Returns:
        EmbeddingService instance

    Raises:
        HTTPException: If service not initialized
    """
    if not hasattr(request.app.state, "embedding_service"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not initialized",
        )
    return request.app.state.embedding_service


def get_llm_service(request: Request) -> LLMService:
    """
    Get LLM service from app state.

    Args:
        request: FastAPI request object

    Returns:
        LLMService instance

    Raises:
        HTTPException: If service not initialized
    """
    if not hasattr(request.app.state, "llm_service"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service not initialized",
        )
    return request.app.state.llm_service


async def get_memory_pipeline(
    request: Request,
    vector_db: Annotated[VectorDBService, Depends(get_vector_db)],
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> MemoryPipeline:
    """
    Get or create memory pipeline instance.

    Args:
        request: FastAPI request
        vector_db: Vector DB service
        embedding_service: Embedding service
        llm_service: LLM service

    Returns:
        MemoryPipeline instance
    """
    # Check if pipeline exists in app state
    if hasattr(request.app.state, "memory_pipeline"):
        return request.app.state.memory_pipeline

    # Create new pipeline
    pipeline = MemoryPipeline(
        vector_db=vector_db,
        embedding_service=embedding_service,
        llm_service=llm_service,
    )

    # Cache in app state
    request.app.state.memory_pipeline = pipeline

    return pipeline


# Dependency type aliases
VectorDBDep = Annotated[VectorDBService, Depends(get_vector_db)]
EmbeddingDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
LLMDep = Annotated[LLMService, Depends(get_llm_service)]
PipelineDep = Annotated[MemoryPipeline, Depends(get_memory_pipeline)]


async def verify_api_key(
    request: Request,
    settings: SettingsDep,
) -> bool:
    """
    Verify API key from request headers.

    For production use - currently returns True for all requests.
    Implement actual verification as needed.

    Args:
        request: FastAPI request
        settings: Application settings

    Returns:
        True if authorized

    Raises:
        HTTPException: If unauthorized
    """
    # In development, allow all requests
    if settings.app_env == "development":
        return True

    # Check for API key header
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )

    # TODO: Implement actual API key verification
    # For now, accept any non-empty key
    return True


# Auth dependency
AuthDep = Annotated[bool, Depends(verify_api_key)]
