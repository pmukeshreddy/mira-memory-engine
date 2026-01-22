"""
Embedding service using OpenAI.

Provides text embedding generation using OpenAI's embedding models
with support for batching and caching.
"""

import hashlib
from functools import lru_cache
from typing import Sequence

import structlog
from openai import AsyncOpenAI

from app.config import get_settings
from app.utils.latency import latency_tracked, track_latency

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    OpenAI embedding service for text vectorization.

    Handles embedding generation with batching support and
    optional caching for repeated queries.
    """

    def __init__(self) -> None:
        """Initialize the embedding service."""
        self.settings = get_settings()
        self._client: AsyncOpenAI | None = None
        self._cache: dict[str, list[float]] = {}
        self._cache_enabled = True
        self._max_cache_size = 10000

    def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.openai_api_key.get_secret_value()
            )
        return self._client

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @latency_tracked("embedding_single")
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Check cache
        if self._cache_enabled:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                logger.debug("embedding_cache_hit", text_preview=text[:30])
                return self._cache[cache_key]

        # Generate embedding
        client = self._get_client()

        try:
            response = await client.embeddings.create(
                model=self.settings.embedding_model,
                input=text,
                dimensions=self.settings.embedding_dimensions,
            )

            embedding = response.data[0].embedding

            # Cache result
            if self._cache_enabled:
                self._add_to_cache(cache_key, embedding)

            logger.debug(
                "embedding_generated",
                text_preview=text[:30],
                dimensions=len(embedding),
            )

            return embedding

        except Exception as e:
            logger.error("embedding_failed", error=str(e), text_preview=text[:30])
            raise

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Batches requests for efficiency and handles caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        async with track_latency("embedding_batch") as timing:
            results: list[list[float] | None] = [None] * len(texts)
            texts_to_embed: list[tuple[int, str]] = []

            # Check cache first
            if self._cache_enabled:
                for i, text in enumerate(texts):
                    cache_key = self._cache_key(text)
                    if cache_key in self._cache:
                        results[i] = self._cache[cache_key]
                    else:
                        texts_to_embed.append((i, text))
            else:
                texts_to_embed = list(enumerate(texts))

            # Embed uncached texts in batches
            if texts_to_embed:
                client = self._get_client()
                batch_size = self.settings.embedding_batch_size

                for batch_start in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[batch_start : batch_start + batch_size]
                    batch_texts = [t[1] for t in batch]

                    try:
                        response = await client.embeddings.create(
                            model=self.settings.embedding_model,
                            input=batch_texts,
                            dimensions=self.settings.embedding_dimensions,
                        )

                        # Process results
                        for j, embedding_data in enumerate(response.data):
                            original_idx = batch[j][0]
                            original_text = batch[j][1]
                            embedding = embedding_data.embedding

                            results[original_idx] = embedding

                            # Cache result
                            if self._cache_enabled:
                                cache_key = self._cache_key(original_text)
                                self._add_to_cache(cache_key, embedding)

                    except Exception as e:
                        logger.error(
                            "batch_embedding_failed",
                            error=str(e),
                            batch_size=len(batch),
                        )
                        raise

            logger.info(
                "batch_embedding_completed",
                total=len(texts),
                cached=len(texts) - len(texts_to_embed),
                embedded=len(texts_to_embed),
            )

            # Filter out any None results (shouldn't happen)
            return [r for r in results if r is not None]

    def _add_to_cache(self, key: str, embedding: list[float]) -> None:
        """Add embedding to cache with size limit."""
        if len(self._cache) >= self._max_cache_size:
            # Simple eviction: remove oldest 10%
            keys_to_remove = list(self._cache.keys())[: self._max_cache_size // 10]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[key] = embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("embedding_cache_cleared")

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()


# Utility function for simple embedding
async def get_embedding(text: str) -> list[float]:
    """
    Convenience function to get embedding for a single text.

    Uses a shared service instance.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    service = EmbeddingService()
    return await service.embed_text(text)
