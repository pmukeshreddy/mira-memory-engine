"""
Pytest configuration and fixtures.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.config import Settings, get_settings
from app.main import create_app
from app.services.embeddings import EmbeddingService
from app.services.vectordb import VectorDBService


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Create test settings with test-specific configuration."""
    return Settings(
        app_env="development",
        debug=True,
        vector_db_provider="chroma",
        chroma_persist_dir="./data/chroma_test",
        chroma_collection_name="mira_test_memories",
        # Use environment variables for API keys or provide test values
        deepgram_api_key="test_key",
        openai_api_key="test_key",
        anthropic_api_key="test_key",
    )


@pytest.fixture
def app(test_settings: Settings):
    """Create test FastAPI application."""
    # Override settings
    def override_settings():
        return test_settings

    application = create_app()
    application.dependency_overrides[get_settings] = override_settings
    return application


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create test client for synchronous tests."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def vector_db(test_settings: Settings) -> AsyncGenerator[VectorDBService, None]:
    """Create test vector database service."""
    db = VectorDBService()
    await db.initialize()
    yield db
    # Cleanup
    await db.clear_all()
    await db.close()


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return [
        "The quarterly meeting discussed the new product launch timeline. "
        "We agreed to target Q2 for the initial release with a soft launch in March.",
        
        "Key action items from today's standup: Sarah will finish the API documentation, "
        "Mike is investigating the performance issues, and the team will review PRs by EOD.",
        
        "Customer feedback analysis shows 85% satisfaction with the new features. "
        "Main concerns are around mobile responsiveness and loading times.",
        
        "Budget allocation for next quarter: 40% engineering, 25% marketing, "
        "20% operations, and 15% R&D. This represents a 10% increase in R&D spending.",
        
        "The AI integration project is progressing well. We've completed the embedding "
        "pipeline and are now working on the retrieval optimization layer.",
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What were the action items from the standup meeting?"


@pytest.fixture
def mock_embedding() -> list[float]:
    """Create a mock embedding vector."""
    import random
    random.seed(42)
    return [random.uniform(-1, 1) for _ in range(1536)]


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}

    async def embed_text(self, text: str) -> list[float]:
        """Generate deterministic mock embedding."""
        import hashlib
        import random
        
        # Use hash for deterministic results
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(1536)]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed_text(t) for t in texts]


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Create mock embedding service."""
    return MockEmbeddingService()
