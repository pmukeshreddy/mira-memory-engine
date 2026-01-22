"""
Latency tracking utilities for performance monitoring.

Provides decorators and context managers for tracking operation
latencies with support for percentile calculations.
"""

import asyncio
import functools
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from statistics import mean, quantiles
from typing import Any, Callable, TypeVar

import structlog
from prometheus_client import Histogram

logger = structlog.get_logger(__name__)

# Type variable for generic function decoration
F = TypeVar("F", bound=Callable[..., Any])


# Prometheus histograms for latency tracking
LATENCY_HISTOGRAM = Histogram(
    "mira_operation_latency_seconds",
    "Operation latency in seconds",
    ["operation"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
)


@dataclass
class LatencyMetrics:
    """Container for latency statistics."""

    operation: str
    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total number of samples."""
        return len(self.samples)

    @property
    def p50(self) -> float:
        """50th percentile (median) latency in ms."""
        if not self.samples:
            return 0.0
        return quantiles(self.samples, n=2)[0] * 1000

    @property
    def p95(self) -> float:
        """95th percentile latency in ms."""
        if len(self.samples) < 2:
            return (self.samples[0] if self.samples else 0.0) * 1000
        return quantiles(self.samples, n=20)[18] * 1000

    @property
    def p99(self) -> float:
        """99th percentile latency in ms."""
        if len(self.samples) < 2:
            return (self.samples[0] if self.samples else 0.0) * 1000
        return quantiles(self.samples, n=100)[98] * 1000

    @property
    def avg(self) -> float:
        """Average latency in ms."""
        if not self.samples:
            return 0.0
        return mean(self.samples) * 1000

    def add_sample(self, duration_seconds: float) -> None:
        """Add a latency sample."""
        self.samples.append(duration_seconds)
        # Keep only last 1000 samples for memory efficiency
        if len(self.samples) > 1000:
            self.samples = self.samples[-1000:]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "operation": self.operation,
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "avg_ms": round(self.avg, 2),
            "count": self.count,
        }


class LatencyTracker:
    """
    Global latency tracker for all operations.

    Thread-safe singleton that collects latency metrics
    across the application.
    """

    _instance: "LatencyTracker | None" = None
    _metrics: dict[str, LatencyMetrics]

    def __new__(cls) -> "LatencyTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = defaultdict(
                lambda: LatencyMetrics(operation="unknown")
            )
        return cls._instance

    def record(self, operation: str, duration_seconds: float) -> None:
        """
        Record a latency measurement.

        Args:
            operation: Name of the operation
            duration_seconds: Duration in seconds
        """
        if operation not in self._metrics:
            self._metrics[operation] = LatencyMetrics(operation=operation)
        self._metrics[operation].add_sample(duration_seconds)

        # Also record to Prometheus
        LATENCY_HISTOGRAM.labels(operation=operation).observe(duration_seconds)

    def get_metrics(self, operation: str) -> LatencyMetrics | None:
        """Get metrics for a specific operation."""
        return self._metrics.get(operation)

    def get_all_metrics(self) -> list[dict[str, Any]]:
        """Get all recorded metrics."""
        return [m.to_dict() for m in self._metrics.values()]

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


# Global tracker instance
_tracker = LatencyTracker()


@contextmanager
def track_latency_sync(operation: str):
    """
    Context manager for tracking synchronous operation latency.

    Args:
        operation: Name of the operation to track

    Yields:
        dict with 'duration_ms' key after completion
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        duration = time.perf_counter() - start
        result["duration_ms"] = duration * 1000
        _tracker.record(operation, duration)
        logger.debug(
            "operation_completed",
            operation=operation,
            duration_ms=round(duration * 1000, 2),
        )


@asynccontextmanager
async def track_latency(operation: str):
    """
    Async context manager for tracking operation latency.

    Args:
        operation: Name of the operation to track

    Yields:
        dict with 'duration_ms' key after completion

    Example:
        async with track_latency("embedding") as timing:
            result = await embed_text(text)
        print(f"Took {timing['duration_ms']}ms")
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        duration = time.perf_counter() - start
        result["duration_ms"] = duration * 1000
        _tracker.record(operation, duration)
        logger.debug(
            "operation_completed",
            operation=operation,
            duration_ms=round(duration * 1000, 2),
        )


def latency_tracked(operation: str | None = None) -> Callable[[F], F]:
    """
    Decorator for tracking function latency.

    Can be used on both sync and async functions.

    Args:
        operation: Operation name (defaults to function name)

    Example:
        @latency_tracked("embedding")
        async def embed_text(text: str) -> list[float]:
            ...
    """

    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with track_latency(op_name):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with track_latency_sync(op_name):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


def get_tracker() -> LatencyTracker:
    """Get the global latency tracker instance."""
    return _tracker
