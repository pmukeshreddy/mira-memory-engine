"""
End-to-end tests for the Mira Memory Engine.

These tests require API keys and test the full pipeline.
Skip if running in CI without proper configuration.
"""

import os

import pytest

# Skip all tests in this module if API keys aren't configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or 
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="API keys not configured for E2E tests"
)


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_root_endpoint(self, client):
        """Root endpoint should return app info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_check(self, client):
        """Health check should return status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "services" in data

    def test_metrics_endpoint(self, client):
        """Metrics endpoint should return performance data."""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "total_memories" in data
        assert "latencies" in data


class TestMemoryIngest:
    """Test memory ingestion endpoints."""

    def test_ingest_text(self, client):
        """Should ingest text and create memories."""
        response = client.post(
            "/api/v1/memory/ingest",
            json={
                "text": "The quarterly review meeting covered sales performance and marketing initiatives.",
                "source": "test",
                "session_id": "test_session",
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["chunks_created"] >= 1
        assert len(data["memory_ids"]) >= 1

    def test_ingest_empty_text(self, client):
        """Empty text should be rejected."""
        response = client.post(
            "/api/v1/memory/ingest",
            json={"text": ""}
        )
        
        assert response.status_code == 422  # Validation error

    def test_ingest_long_text(self, client):
        """Long text should be chunked appropriately."""
        # Generate long text (~500 words)
        long_text = " ".join(["This is a test sentence."] * 100)
        
        response = client.post(
            "/api/v1/memory/ingest",
            json={"text": long_text}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["chunks_created"] > 1  # Should create multiple chunks


class TestMemoryQuery:
    """Test memory query endpoints."""

    @pytest.fixture(autouse=True)
    def seed_memories(self, client):
        """Seed some memories before query tests."""
        texts = [
            "The product launch is scheduled for March 15th with a soft launch in February.",
            "Customer feedback indicates high satisfaction with the new dashboard features.",
            "Engineering velocity has increased by 25% after implementing the new CI/CD pipeline.",
        ]
        
        for text in texts:
            client.post(
                "/api/v1/memory/ingest",
                json={"text": text, "source": "test"}
            )

    def test_query_memories(self, client):
        """Should query memories and return response."""
        response = client.post(
            "/api/v1/memory/query",
            json={
                "query": "When is the product launch?",
                "top_k": 3,
                "include_context": True,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "memories" in data
        assert "processing_time_ms" in data

    def test_query_with_no_matches(self, client):
        """Query with no relevant memories should still return response."""
        response = client.post(
            "/api/v1/memory/query",
            json={
                "query": "What color is the sky on Mars?",
                "top_k": 3,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_search_without_llm(self, client):
        """Search endpoint should return memories without LLM generation."""
        response = client.post(
            "/api/v1/memory/search",
            params={
                "query": "product launch",
                "top_k": 5,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestMemoryManagement:
    """Test memory management operations."""

    def test_get_recent_memories(self, client):
        """Should retrieve recent memories."""
        # First, ingest some memories
        client.post(
            "/api/v1/memory/ingest",
            json={"text": "Test memory for recent retrieval."}
        )
        
        response = client.get("/api/v1/memory/recent?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total_count" in data

    def test_clear_memories(self, client):
        """Should clear all memories."""
        # Ingest a memory first
        client.post(
            "/api/v1/memory/ingest",
            json={"text": "Memory to be cleared."}
        )
        
        # Clear all memories
        response = client.delete("/api/v1/memory/clear")
        
        assert response.status_code == 204
        
        # Verify memories are cleared
        recent = client.get("/api/v1/memory/recent")
        data = recent.json()
        assert data["total_count"] == 0


class TestLatencyBudget:
    """Test that operations meet latency targets."""

    def test_ingest_latency(self, client):
        """Ingest should complete within target latency."""
        response = client.post(
            "/api/v1/memory/ingest",
            json={
                "text": "Performance test memory for latency measurement.",
                "source": "perf_test",
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        
        # Target: 200ms for ingest pipeline
        # Allow some buffer for test environment
        assert data["processing_time_ms"] < 1000, \
            f"Ingest took {data['processing_time_ms']}ms, expected < 1000ms"

    def test_query_latency(self, client):
        """Query should complete within target latency."""
        # Seed a memory first
        client.post(
            "/api/v1/memory/ingest",
            json={"text": "Query latency test memory content."}
        )
        
        response = client.post(
            "/api/v1/memory/query",
            json={
                "query": "What is the latency test about?",
                "top_k": 3,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Target: 400ms first token, but full response may take longer
        # Allow generous buffer for test environment and LLM generation
        assert data["processing_time_ms"] < 10000, \
            f"Query took {data['processing_time_ms']}ms, expected < 10000ms"
