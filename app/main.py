"""
Mira Memory Engine - FastAPI Application Entry Point

A real-time voice-to-memory RAG system with streaming transcription
and intelligent retrieval capabilities.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from app import __version__
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vectordb import VectorDBService
from app.utils.logging import setup_logging
from app.utils.uptime import set_start_time

# Initialize structured logging
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown tasks including:
    - Initializing services
    - Setting up observability
    - Cleaning up resources
    """
    settings = get_settings()

    # Startup
    logger.info(
        "starting_application",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
    )

    set_start_time()

    # Initialize services
    try:
        # Initialize vector database
        vector_db = VectorDBService()
        await vector_db.initialize()
        app.state.vector_db = vector_db

        # Initialize embedding service
        embedding_service = EmbeddingService()
        app.state.embedding_service = embedding_service

        # Initialize LLM service
        llm_service = LLMService()
        app.state.llm_service = llm_service

        logger.info("services_initialized")

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("shutting_down_application")

    # Cleanup services
    if hasattr(app.state, "vector_db"):
        await app.state.vector_db.close()

    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    # Setup logging first
    setup_logging(settings.log_level, settings.app_env)

    # Create FastAPI app
    app = FastAPI(
        title="Mira Memory Engine",
        description="""
## Real-time Voice-to-Memory RAG System

Mira Memory Engine provides intelligent memory storage and retrieval
with streaming speech-to-text capabilities.

### Features

- **Voice Streaming**: Real-time speech-to-text via WebSocket
- **Memory Storage**: Automatic chunking and embedding of transcripts
- **Intelligent Retrieval**: RAG-powered question answering
- **Low Latency**: Optimized for <400ms query response time

### Architecture

```
Audio → STT (Deepgram) → Chunking → Embedding (OpenAI) → Vector DB (Chroma)
                                                              ↓
Query → Embedding → Vector Search → Context Assembly → LLM (Claude) → Response
```
        """,
        version=__version__,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Prometheus metrics endpoint
    if settings.metrics_enabled:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    # Include API routers
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(ws_router)

    return app


# Create the application instance
app = create_app()


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint redirect to docs."""
    return {
        "name": "Mira Memory Engine",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
