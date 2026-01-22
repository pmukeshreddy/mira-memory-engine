"""
Context assembly for RAG responses.

Handles formatting and ranking of retrieved memories for
optimal LLM context utilization.
"""

from dataclasses import dataclass
from typing import Any

import structlog

from app.models.domain import RetrievalResult
from app.utils.latency import latency_tracked

logger = structlog.get_logger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context assembly."""

    max_context_tokens: int = 4000  # Maximum tokens for context
    max_memories: int = 10  # Maximum memories to include
    include_metadata: bool = True  # Include memory metadata
    include_scores: bool = True  # Include relevance scores
    rerank: bool = True  # Apply reranking
    dedup_threshold: float = 0.9  # Threshold for deduplication


class ContextAssembler:
    """
    Assembles context from retrieved memories for LLM prompts.

    Handles ranking, deduplication, and formatting of memories
    to create optimal context for the LLM.
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        """
        Initialize context assembler.

        Args:
            config: Assembly configuration
        """
        self.config = config or ContextConfig()

        # Rough token estimation (words * 1.3)
        self._tokens_per_word = 1.3

    @latency_tracked("context_assembly")
    def assemble(
        self,
        query: str,
        memories: list[RetrievalResult],
    ) -> str:
        """
        Assemble context from retrieved memories.

        Args:
            query: User's query
            memories: Retrieved memory results

        Returns:
            Formatted context string
        """
        if not memories:
            return self._empty_context()

        # Filter and process memories
        processed = self._process_memories(memories)

        # Deduplicate similar content
        if self.config.rerank:
            processed = self._deduplicate(processed)

        # Rerank by relevance
        if self.config.rerank:
            processed = self._rerank(query, processed)

        # Limit by token count
        processed = self._limit_by_tokens(processed)

        # Format context
        context = self._format_context(processed)

        logger.debug(
            "context_assembled",
            input_memories=len(memories),
            output_memories=len(processed),
            context_length=len(context),
        )

        return context

    def _process_memories(
        self,
        memories: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Filter and validate memories."""
        processed = []

        for memory in memories:
            # Skip empty memories
            if not memory.text or not memory.text.strip():
                continue

            # Skip very low scores
            if memory.score < 0.3:
                continue

            processed.append(memory)

        # Limit to max memories
        return processed[: self.config.max_memories]

    def _deduplicate(
        self,
        memories: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Remove near-duplicate memories.

        Uses simple text overlap detection for efficiency.
        """
        if len(memories) <= 1:
            return memories

        result = [memories[0]]

        for memory in memories[1:]:
            is_duplicate = False

            for existing in result:
                similarity = self._text_similarity(memory.text, existing.text)
                if similarity >= self.config.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(memory)

        return result

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity using word overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _rerank(
        self,
        query: str,
        memories: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Rerank memories - sort by timestamp to preserve conversation flow.
        
        The similarity search already found relevant memories;
        now we restore chronological order for better context.
        """
        # Sort by timestamp to restore conversation flow
        def get_timestamp(memory: RetrievalResult) -> str:
            return memory.metadata.get("timestamp", "") if memory.metadata else ""
        
        sorted_memories = sorted(memories, key=get_timestamp)

        # Update ranks (now chronological)
        for i, memory in enumerate(sorted_memories):
            memory.rank = i + 1

        return sorted_memories

    def _limit_by_tokens(
        self,
        memories: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Limit memories to fit within token budget."""
        result = []
        total_tokens = 0

        for memory in memories:
            # Estimate tokens for this memory
            memory_tokens = self._estimate_tokens(memory)

            if total_tokens + memory_tokens > self.config.max_context_tokens:
                break

            result.append(memory)
            total_tokens += memory_tokens

        return result

    def _estimate_tokens(self, memory: RetrievalResult) -> int:
        """Estimate token count for a memory."""
        text_tokens = len(memory.text.split()) * self._tokens_per_word

        # Add overhead for formatting
        overhead = 50 if self.config.include_metadata else 20

        return int(text_tokens + overhead)

    def _format_context(self, memories: list[RetrievalResult]) -> str:
        """Format memories into context string."""
        if not memories:
            return self._empty_context()

        parts = []

        for i, memory in enumerate(memories, 1):
            memory_parts = []

            # Header with score
            if self.config.include_scores:
                memory_parts.append(f"[Memory {i}] (Relevance: {memory.score:.0%})")
            else:
                memory_parts.append(f"[Memory {i}]")

            # Metadata
            if self.config.include_metadata and memory.metadata:
                meta_items = []

                if memory.metadata.get("timestamp"):
                    meta_items.append(f"Time: {memory.metadata['timestamp']}")
                if memory.metadata.get("source"):
                    meta_items.append(f"Source: {memory.metadata['source']}")
                if memory.metadata.get("speaker"):
                    meta_items.append(f"Speaker: {memory.metadata['speaker']}")

                if meta_items:
                    memory_parts.append(" | ".join(meta_items))

            # Content
            memory_parts.append(memory.text)

            parts.append("\n".join(memory_parts))

        return "\n\n---\n\n".join(parts)

    def _empty_context(self) -> str:
        """Return context for when no memories are found."""
        return "No relevant memories found for this query."

    def get_context_stats(
        self,
        memories: list[RetrievalResult],
    ) -> dict[str, Any]:
        """
        Get statistics about the assembled context.

        Args:
            memories: Memories to analyze

        Returns:
            Statistics dictionary
        """
        if not memories:
            return {
                "memory_count": 0,
                "total_tokens": 0,
                "avg_score": 0.0,
                "score_range": (0.0, 0.0),
            }

        total_tokens = sum(self._estimate_tokens(m) for m in memories)
        scores = [m.score for m in memories]

        return {
            "memory_count": len(memories),
            "total_tokens": total_tokens,
            "avg_score": sum(scores) / len(scores),
            "score_range": (min(scores), max(scores)),
        }


def assemble_context(
    query: str,
    memories: list[RetrievalResult],
    max_tokens: int = 4000,
) -> str:
    """
    Convenience function to assemble context.

    Args:
        query: User's query
        memories: Retrieved memories
        max_tokens: Maximum context tokens

    Returns:
        Formatted context string
    """
    config = ContextConfig(max_context_tokens=max_tokens)
    assembler = ContextAssembler(config)
    return assembler.assemble(query, memories)
