"""
Core domain models for the memory pipeline.

These models represent the internal data structures used throughout
the application for processing and storing memories.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """
    A segment of transcribed speech from the STT service.

    Represents a portion of audio that has been converted to text,
    including timing information and confidence scores.
    """

    text: str = Field(..., description="Transcribed text content")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Transcription confidence score"
    )
    speaker: str | None = Field(
        default=None, description="Speaker identifier (if diarization enabled)"
    )
    is_final: bool = Field(
        default=False, description="Whether this is a final (non-interim) result"
    )
    words: list[dict[str, Any]] = Field(
        default_factory=list, description="Word-level timing information"
    )


class MemoryMetadata(BaseModel):
    """
    Metadata associated with a stored memory.

    Contains contextual information about when, how, and from
    what source the memory was created.
    """

    source: str = Field(
        default="voice", description="Source of the memory (voice, text, import)"
    )
    session_id: str | None = Field(
        default=None, description="Session identifier for grouping related memories"
    )
    speaker: str | None = Field(
        default=None, description="Speaker identifier if available"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the memory was created"
    )
    duration_seconds: float | None = Field(
        default=None, description="Duration of the original audio segment"
    )
    chunk_index: int = Field(default=0, description="Index of this chunk in sequence")
    total_chunks: int = Field(default=1, description="Total chunks in the sequence")
    custom: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata fields"
    )


class Chunk(BaseModel):
    """
    A chunk of text prepared for embedding and storage.

    Represents a semantically meaningful piece of text that has been
    extracted from a larger transcript or document.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique chunk identifier")
    text: str = Field(..., min_length=1, description="Chunk text content")
    word_count: int = Field(..., ge=1, description="Number of words in chunk")
    char_count: int = Field(..., ge=1, description="Number of characters in chunk")
    metadata: MemoryMetadata = Field(
        default_factory=MemoryMetadata, description="Chunk metadata"
    )

    @classmethod
    def from_text(
        cls,
        text: str,
        metadata: MemoryMetadata | None = None,
        chunk_index: int = 0,
        total_chunks: int = 1,
    ) -> "Chunk":
        """Create a chunk from raw text."""
        words = text.split()
        meta = metadata or MemoryMetadata()
        meta.chunk_index = chunk_index
        meta.total_chunks = total_chunks

        return cls(
            text=text.strip(),
            word_count=len(words),
            char_count=len(text),
            metadata=meta,
        )


class Memory(BaseModel):
    """
    A memory stored in the vector database.

    Represents a complete memory unit including the original text,
    its vector embedding, and all associated metadata.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique memory identifier")
    text: str = Field(..., description="Original text content")
    embedding: list[float] = Field(..., description="Vector embedding")
    metadata: MemoryMetadata = Field(
        default_factory=MemoryMetadata, description="Memory metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


class RetrievalResult(BaseModel):
    """
    A result from querying the vector database.

    Includes the matching memory along with relevance scores
    and ranking information.
    """

    memory_id: str = Field(..., description="ID of the matching memory")
    text: str = Field(..., description="Memory text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Result rank (1 = most relevant)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Memory metadata"
    )

    @property
    def is_relevant(self) -> bool:
        """Check if result meets relevance threshold."""
        return self.score >= 0.7
