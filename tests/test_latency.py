"""
Tests for latency tracking utilities.
"""

import asyncio
import time

import pytest

from app.utils.latency import (
    LatencyMetrics,
    LatencyTracker,
    get_tracker,
    latency_tracked,
    track_latency,
    track_latency_sync,
)


class TestLatencyMetrics:
    """Tests for LatencyMetrics class."""

    def test_empty_metrics(self):
        """Empty metrics should return zeros."""
        metrics = LatencyMetrics(operation="test")
        
        assert metrics.count == 0
        assert metrics.p50 == 0.0
        assert metrics.p95 == 0.0
        assert metrics.avg == 0.0

    def test_single_sample(self):
        """Single sample should work correctly."""
        metrics = LatencyMetrics(operation="test")
        metrics.add_sample(0.1)  # 100ms
        
        assert metrics.count == 1
        assert metrics.p50 == pytest.approx(100.0, 1.0)
        assert metrics.avg == pytest.approx(100.0, 1.0)

    def test_multiple_samples(self):
        """Multiple samples should calculate percentiles correctly."""
        metrics = LatencyMetrics(operation="test")
        
        # Add samples (in seconds)
        for i in range(100):
            metrics.add_sample(i / 1000)  # 0-99ms
        
        assert metrics.count == 100
        assert metrics.p50 == pytest.approx(50.0, 5.0)
        assert metrics.p95 == pytest.approx(95.0, 5.0)
        assert metrics.avg == pytest.approx(49.5, 5.0)

    def test_sample_limit(self):
        """Samples should be limited to prevent memory issues."""
        metrics = LatencyMetrics(operation="test")
        
        # Add more than the limit
        for i in range(1500):
            metrics.add_sample(0.01)
        
        # Should only keep last 1000
        assert metrics.count == 1000

    def test_to_dict(self):
        """Should convert to dictionary format."""
        metrics = LatencyMetrics(operation="test_op")
        metrics.add_sample(0.05)
        metrics.add_sample(0.1)
        
        result = metrics.to_dict()
        
        assert result["operation"] == "test_op"
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert result["count"] == 2


class TestLatencyTracker:
    """Tests for LatencyTracker singleton."""

    def test_singleton(self):
        """Should return same instance."""
        tracker1 = LatencyTracker()
        tracker2 = LatencyTracker()
        
        assert tracker1 is tracker2

    def test_record(self):
        """Should record latency samples."""
        tracker = get_tracker()
        tracker.reset()
        
        tracker.record("test_op", 0.1)
        tracker.record("test_op", 0.2)
        
        metrics = tracker.get_metrics("test_op")
        assert metrics is not None
        assert metrics.count == 2

    def test_get_all_metrics(self):
        """Should return all recorded metrics."""
        tracker = get_tracker()
        tracker.reset()
        
        tracker.record("op1", 0.1)
        tracker.record("op2", 0.2)
        
        all_metrics = tracker.get_all_metrics()
        
        assert len(all_metrics) == 2
        ops = {m["operation"] for m in all_metrics}
        assert ops == {"op1", "op2"}


class TestTrackLatency:
    """Tests for latency tracking context managers."""

    def test_sync_context_manager(self):
        """Sync context manager should track time."""
        with track_latency_sync("test_sync") as result:
            time.sleep(0.01)  # 10ms
        
        assert "duration_ms" in result
        assert result["duration_ms"] >= 10

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Async context manager should track time."""
        async with track_latency("test_async") as result:
            await asyncio.sleep(0.01)  # 10ms
        
        assert "duration_ms" in result
        assert result["duration_ms"] >= 10

    def test_sync_decorator(self):
        """Sync decorator should track function latency."""
        tracker = get_tracker()
        tracker.reset()
        
        @latency_tracked("decorated_sync")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        metrics = tracker.get_metrics("decorated_sync")
        assert metrics is not None
        assert metrics.count == 1
        assert metrics.p50 >= 10

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Async decorator should track function latency."""
        tracker = get_tracker()
        tracker.reset()
        
        @latency_tracked("decorated_async")
        async def slow_async_function():
            await asyncio.sleep(0.01)
            return "done"
        
        result = await slow_async_function()
        
        assert result == "done"
        metrics = tracker.get_metrics("decorated_async")
        assert metrics is not None
        assert metrics.count == 1
        assert metrics.p50 >= 10

    def test_default_operation_name(self):
        """Decorator without name should use function name."""
        tracker = get_tracker()
        tracker.reset()
        
        @latency_tracked()
        def my_custom_function():
            return True
        
        my_custom_function()
        
        metrics = tracker.get_metrics("my_custom_function")
        assert metrics is not None
