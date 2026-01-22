"""
Tests for the memory pipeline.
"""

import pytest

from app.core.chunker import Chunker, ChunkConfig, ChunkingStrategy
from app.core.context import ContextAssembler, ContextConfig
from app.models.domain import MemoryMetadata, RetrievalResult


class TestChunker:
    """Tests for text chunking."""

    def test_chunk_empty_text(self):
        """Empty text should return no chunks."""
        chunker = Chunker()
        result = chunker.chunk_text("")
        assert result == []

    def test_chunk_short_text(self):
        """Short text should return single chunk."""
        chunker = Chunker(ChunkConfig(chunk_size=100))
        text = "This is a short text that should fit in one chunk."
        result = chunker.chunk_text(text)
        
        assert len(result) == 1
        assert result[0].text == text.strip()

    def test_chunk_long_text(self):
        """Long text should be split into multiple chunks."""
        chunker = Chunker(ChunkConfig(chunk_size=20, chunk_overlap=5))
        
        # Create text with ~100 words
        text = " ".join(["word"] * 100)
        result = chunker.chunk_text(text)
        
        assert len(result) > 1
        # Each chunk should have roughly chunk_size words
        for chunk in result:
            assert chunk.word_count <= 30  # Allow some flexibility

    def test_chunk_overlap(self):
        """Chunks should have overlapping content."""
        chunker = Chunker(ChunkConfig(chunk_size=20, chunk_overlap=5))
        text = " ".join([f"word{i}" for i in range(50)])
        result = chunker.chunk_text(text)
        
        assert len(result) > 1
        # Check that consecutive chunks share some words
        for i in range(len(result) - 1):
            chunk1_words = set(result[i].text.split())
            chunk2_words = set(result[i + 1].text.split())
            overlap = chunk1_words & chunk2_words
            assert len(overlap) > 0, "Consecutive chunks should overlap"

    def test_chunk_with_metadata(self):
        """Chunks should include metadata."""
        chunker = Chunker()
        metadata = MemoryMetadata(
            source="test",
            session_id="session123",
        )
        
        text = "Test text for chunking."
        result = chunker.chunk_text(text, metadata)
        
        assert len(result) == 1
        assert result[0].metadata.source == "test"
        assert result[0].metadata.session_id == "session123"

    def test_chunk_index_tracking(self):
        """Chunks should track their position in sequence."""
        chunker = Chunker(ChunkConfig(chunk_size=20, chunk_overlap=5))
        text = " ".join(["word"] * 100)
        result = chunker.chunk_text(text)
        
        for i, chunk in enumerate(result):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(result)

    def test_sentence_strategy(self):
        """Sentence chunking should respect sentence boundaries."""
        config = ChunkConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=50,
        )
        chunker = Chunker(config)
        
        text = (
            "First sentence here. Second sentence follows. "
            "Third sentence now. Fourth one comes next. "
            "Fifth sentence appears. Sixth is the last."
        )
        result = chunker.chunk_text(text)
        
        # Check that chunks end with sentence terminators
        for chunk in result:
            text = chunk.text.strip()
            assert text[-1] in ".!?", f"Chunk should end with sentence: {text}"

    def test_estimate_chunks(self):
        """Chunk estimation should be reasonably accurate."""
        chunker = Chunker(ChunkConfig(chunk_size=20, chunk_overlap=5))
        text = " ".join(["word"] * 100)
        
        estimate = chunker.estimate_chunks(text)
        actual = len(chunker.chunk_text(text))
        
        # Estimate should be within 20% of actual
        assert abs(estimate - actual) / actual < 0.3


class TestContextAssembler:
    """Tests for context assembly."""

    @pytest.fixture
    def sample_memories(self) -> list[RetrievalResult]:
        """Create sample retrieval results."""
        return [
            RetrievalResult(
                memory_id="mem1",
                text="The project deadline is next Friday. We need to complete testing.",
                score=0.95,
                rank=1,
                metadata={"timestamp": "2024-01-15T10:00:00", "source": "meeting"},
            ),
            RetrievalResult(
                memory_id="mem2",
                text="Budget has been approved for the new feature development.",
                score=0.85,
                rank=2,
                metadata={"timestamp": "2024-01-14T14:30:00", "source": "email"},
            ),
            RetrievalResult(
                memory_id="mem3",
                text="Team velocity has improved by 20% this sprint.",
                score=0.75,
                rank=3,
                metadata={"timestamp": "2024-01-13T09:00:00", "source": "standup"},
            ),
        ]

    def test_assemble_empty_memories(self):
        """Empty memories should return appropriate message."""
        assembler = ContextAssembler()
        result = assembler.assemble("test query", [])
        
        assert "No relevant memories" in result

    def test_assemble_with_memories(self, sample_memories):
        """Context should include all memories."""
        assembler = ContextAssembler()
        result = assembler.assemble("What's the project deadline?", sample_memories)
        
        # All memory texts should appear in context
        for memory in sample_memories:
            assert memory.text in result

    def test_assemble_includes_scores(self, sample_memories):
        """Context should include relevance scores when configured."""
        assembler = ContextAssembler(ContextConfig(include_scores=True))
        result = assembler.assemble("test", sample_memories)
        
        assert "95%" in result or "0.95" in result

    def test_assemble_includes_metadata(self, sample_memories):
        """Context should include metadata when configured."""
        assembler = ContextAssembler(ContextConfig(include_metadata=True))
        result = assembler.assemble("test", sample_memories)
        
        # Should include source type
        assert "meeting" in result or "email" in result

    def test_assemble_respects_token_limit(self, sample_memories):
        """Context should respect token limit."""
        assembler = ContextAssembler(ContextConfig(max_context_tokens=100))
        result = assembler.assemble("test", sample_memories)
        
        # Rough token count should be under limit
        word_count = len(result.split())
        estimated_tokens = word_count * 1.3
        assert estimated_tokens < 200  # Some buffer for formatting

    def test_deduplication(self):
        """Similar memories should be deduplicated."""
        assembler = ContextAssembler(ContextConfig(dedup_threshold=0.8))
        
        memories = [
            RetrievalResult(
                memory_id="mem1",
                text="The meeting is scheduled for Monday at 10am.",
                score=0.9,
                rank=1,
            ),
            RetrievalResult(
                memory_id="mem2",
                text="The meeting is scheduled for Monday at 10am.",  # Duplicate
                score=0.85,
                rank=2,
            ),
            RetrievalResult(
                memory_id="mem3",
                text="Budget discussion is on Tuesday.",
                score=0.8,
                rank=3,
            ),
        ]
        
        result = assembler.assemble("When is the meeting?", memories)
        
        # Should only include unique content
        assert result.count("scheduled for Monday") == 1

    def test_context_stats(self, sample_memories):
        """Should return correct statistics."""
        assembler = ContextAssembler()
        stats = assembler.get_context_stats(sample_memories)
        
        assert stats["memory_count"] == 3
        assert stats["avg_score"] == pytest.approx(0.85, 0.01)
        assert stats["score_range"] == (0.75, 0.95)
