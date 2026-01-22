"""
Text chunking for the memory pipeline.

Implements multiple chunking strategies for optimal
semantic preservation during embedding and retrieval.

Strategies:
- SLIDING_WINDOW: Fixed size with overlap (default, fast)
- SENTENCE: Accumulates sentences to target size (good for conversations)
- PARAGRAPH: Respects paragraph boundaries (good for documents)
- SEMANTIC: Uses embedding similarity to find break points (best quality, slower)
- RECURSIVE: Hierarchical splitting by structure (good for mixed content)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Callable

import structlog

from app.config import get_settings
from app.models.domain import Chunk, MemoryMetadata
from app.utils.latency import latency_tracked

logger = structlog.get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    SLIDING_WINDOW = "sliding_window"  # Fast, fixed-size chunks with overlap
    SENTENCE = "sentence"              # Accumulate sentences to target size
    PARAGRAPH = "paragraph"            # Respect paragraph boundaries
    SEMANTIC = "semantic"              # Use embeddings to find break points
    RECURSIVE = "recursive"            # Hierarchical structure-aware splitting


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior."""

    strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW
    chunk_size: int = 150  # Target words per chunk
    chunk_overlap: int = 30  # Overlap words
    min_chunk_size: int = 20  # Minimum words to form a chunk
    max_chunk_size: int = 300  # Maximum words per chunk
    sentence_boundary: bool = True  # Try to end at sentence boundaries
    
    # Semantic chunking options
    similarity_threshold: float = 0.5  # Break when similarity drops below this
    embedding_fn: Callable[[str], list[float]] | None = None  # Function to get embeddings
    
    # Recursive chunking separators (in order of priority)
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",      # Paragraphs
        "\n",        # Lines
        ". ",        # Sentences
        "? ",
        "! ",
        "; ",
        ", ",        # Clauses
        " ",         # Words
    ])


class Chunker:
    """
    Text chunker for the memory pipeline.

    Splits text into semantically meaningful chunks suitable for
    embedding and retrieval.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """
        Initialize chunker with configuration.

        Args:
            config: Chunking configuration (uses settings defaults if None)
        """
        settings = get_settings()

        if config is None:
            config = ChunkConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

        self.config = config

        # Sentence boundary pattern
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        )

    @latency_tracked("chunking")
    def chunk_text(
        self,
        text: str,
        metadata: MemoryMetadata | None = None,
    ) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Base metadata for all chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        # Clean and normalize text
        text = self._normalize_text(text)

        # Select chunking strategy
        if self.config.strategy == ChunkingStrategy.SENTENCE:
            chunks = list(self._chunk_by_sentences(text))
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = list(self._chunk_by_paragraphs(text))
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            chunks = list(self._chunk_semantic(text))
        elif self.config.strategy == ChunkingStrategy.RECURSIVE:
            chunks = list(self._chunk_recursive(text))
        else:
            chunks = list(self._chunk_sliding_window(text))

        # Convert to Chunk objects
        result = []
        total_chunks = len(chunks)

        for i, chunk_text in enumerate(chunks):
            chunk = Chunk.from_text(
                text=chunk_text,
                metadata=metadata,
                chunk_index=i,
                total_chunks=total_chunks,
            )
            result.append(chunk)

        logger.debug(
            "text_chunked",
            input_words=len(text.split()),
            chunks=len(result),
            strategy=self.config.strategy.value,
        )

        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _chunk_sliding_window(self, text: str) -> Iterator[str]:
        """
        Chunk using sliding window with overlap.

        Yields:
            Text chunks with specified overlap
        """
        words = text.split()

        if len(words) <= self.config.chunk_size:
            yield text
            return

        start = 0
        while start < len(words):
            # Calculate end position
            end = min(start + self.config.chunk_size, len(words))

            # Extract chunk words
            chunk_words = words[start:end]

            # Try to end at sentence boundary if enabled
            if self.config.sentence_boundary and end < len(words):
                chunk_text = ' '.join(chunk_words)
                adjusted_text = self._adjust_to_sentence_boundary(
                    chunk_text,
                    words[end:end + self.config.chunk_overlap],
                )
                if adjusted_text:
                    yield adjusted_text
                    # Recalculate start based on actual chunk
                    actual_words = len(adjusted_text.split())
                    start = start + actual_words - self.config.chunk_overlap
                    continue

            yield ' '.join(chunk_words)

            # Move window
            if end >= len(words):
                break

            start = end - self.config.chunk_overlap
            if start >= len(words):
                break

    def _adjust_to_sentence_boundary(
        self,
        chunk_text: str,
        lookahead_words: list[str],
    ) -> str | None:
        """
        Adjust chunk to end at a sentence boundary.

        Args:
            chunk_text: Current chunk text
            lookahead_words: Words after current chunk

        Returns:
            Adjusted text or None if no good boundary found
        """
        # Check if chunk already ends with sentence boundary
        if chunk_text.rstrip()[-1] in '.!?':
            return chunk_text

        # Look for sentence boundary in lookahead
        lookahead_text = ' '.join(lookahead_words)
        combined = chunk_text + ' ' + lookahead_text

        # Find sentence boundaries
        sentences = self._sentence_pattern.split(combined)

        if len(sentences) > 1:
            # Take first complete sentence(s) up to max size
            result = sentences[0]
            word_count = len(result.split())

            for sent in sentences[1:]:
                new_count = word_count + len(sent.split())
                if new_count <= self.config.max_chunk_size:
                    result += ' ' + sent
                    word_count = new_count
                else:
                    break

            if len(result.split()) >= self.config.min_chunk_size:
                return result.strip()

        return None

    def _chunk_by_sentences(self, text: str) -> Iterator[str]:
        """
        Chunk by accumulating sentences up to target size.

        Yields:
            Text chunks containing complete sentences
        """
        sentences = self._split_sentences(text)
        current_chunk: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If single sentence exceeds max, split it
            if sentence_words > self.config.max_chunk_size:
                # Yield current chunk first
                if current_chunk:
                    yield ' '.join(current_chunk)
                    current_chunk = []
                    current_words = 0

                # Split long sentence with sliding window
                for chunk in self._chunk_sliding_window(sentence):
                    yield chunk
                continue

            # Check if adding sentence exceeds target
            if current_words + sentence_words > self.config.chunk_size:
                if current_chunk:
                    yield ' '.join(current_chunk)
                current_chunk = [sentence]
                current_words = sentence_words
            else:
                current_chunk.append(sentence)
                current_words += sentence_words

        # Yield remaining
        if current_chunk:
            yield ' '.join(current_chunk)

    def _chunk_by_paragraphs(self, text: str) -> Iterator[str]:
        """
        Chunk by paragraphs, splitting large ones.

        Yields:
            Text chunks respecting paragraph boundaries
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            word_count = len(para.split())

            if word_count <= self.config.chunk_size:
                if word_count >= self.config.min_chunk_size:
                    yield para
            else:
                # Split large paragraph
                for chunk in self._chunk_sliding_window(para):
                    yield chunk

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_semantic(self, text: str) -> Iterator[str]:
        """
        Chunk using semantic similarity to find natural break points.
        
        This method splits text into sentences, then uses embedding similarity
        to determine where to place chunk boundaries. When similarity between
        consecutive sentences drops significantly, a new chunk is started.
        
        Yields:
            Text chunks with semantically coherent content
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            yield text
            return
        
        # If no embedding function provided, fall back to sentence chunking
        if not self.config.embedding_fn:
            logger.warning("semantic_chunking_no_embeddings", fallback="sentence")
            yield from self._chunk_by_sentences(text)
            return
        
        # Get embeddings for all sentences
        try:
            embeddings = [self.config.embedding_fn(s) for s in sentences]
        except Exception as e:
            logger.error("semantic_chunking_embedding_failed", error=str(e))
            yield from self._chunk_by_sentences(text)
            return
        
        # Calculate cosine similarity between consecutive sentences
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0
        
        # Find break points where similarity drops
        current_chunk: list[str] = [sentences[0]]
        current_words = len(sentences[0].split())
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(embeddings[i-1], embeddings[i])
            sentence_words = len(sentences[i].split())
            
            # Break if: low similarity OR exceeds max size
            should_break = (
                similarity < self.config.similarity_threshold or
                current_words + sentence_words > self.config.max_chunk_size
            )
            
            # Also break if chunk is large enough and similarity is dropping
            if current_words >= self.config.chunk_size and similarity < 0.7:
                should_break = True
            
            if should_break and current_words >= self.config.min_chunk_size:
                yield ' '.join(current_chunk)
                current_chunk = [sentences[i]]
                current_words = sentence_words
            else:
                current_chunk.append(sentences[i])
                current_words += sentence_words
        
        if current_chunk:
            yield ' '.join(current_chunk)

    def _chunk_recursive(
        self,
        text: str,
        separators: list[str] | None = None,
    ) -> Iterator[str]:
        """
        Recursively chunk text using a hierarchy of separators.
        
        Tries to split on larger structural elements first (paragraphs),
        then falls back to smaller ones (sentences, words) if chunks are too large.
        
        Yields:
            Text chunks respecting document structure
        """
        if separators is None:
            separators = self.config.separators
        
        word_count = len(text.split())
        
        # Base case: text is small enough
        if word_count <= self.config.chunk_size:
            if word_count >= self.config.min_chunk_size:
                yield text
            return
        
        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep in text:
                splits = text.split(sep)
                splits = [s.strip() for s in splits if s.strip()]
                
                if len(splits) > 1:
                    # Recursively process splits
                    current_chunk: list[str] = []
                    current_words = 0
                    
                    for split in splits:
                        split_words = len(split.split())
                        
                        # If single split is too large, recurse with next separator
                        if split_words > self.config.max_chunk_size:
                            # Yield current chunk first
                            if current_chunk:
                                yield sep.join(current_chunk)
                                current_chunk = []
                                current_words = 0
                            
                            # Recurse with remaining separators
                            remaining_seps = separators[i+1:] if i+1 < len(separators) else [" "]
                            yield from self._chunk_recursive(split, remaining_seps)
                            continue
                        
                        # Check if adding this split exceeds target
                        if current_words + split_words > self.config.chunk_size:
                            if current_chunk and current_words >= self.config.min_chunk_size:
                                yield sep.join(current_chunk)
                            current_chunk = [split]
                            current_words = split_words
                        else:
                            current_chunk.append(split)
                            current_words += split_words
                    
                    # Yield remaining
                    if current_chunk and current_words >= self.config.min_chunk_size:
                        yield sep.join(current_chunk)
                    
                    return
        
        # No separators worked, fall back to sliding window
        yield from self._chunk_sliding_window(text)

    def estimate_chunks(self, text: str) -> int:
        """
        Estimate number of chunks without actually chunking.

        Args:
            text: Text to estimate

        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0

        word_count = len(text.split())

        if word_count <= self.config.chunk_size:
            return 1

        # Estimate based on sliding window
        effective_step = self.config.chunk_size - self.config.chunk_overlap
        return max(1, (word_count - self.config.chunk_overlap) // effective_step + 1)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: MemoryMetadata | None = None,
) -> list[Chunk]:
    """
    Convenience function to chunk text.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
        metadata: Metadata for chunks

    Returns:
        List of Chunk objects
    """
    config = ChunkConfig()
    if chunk_size is not None:
        config.chunk_size = chunk_size
    if overlap is not None:
        config.chunk_overlap = overlap

    chunker = Chunker(config)
    return chunker.chunk_text(text, metadata)
