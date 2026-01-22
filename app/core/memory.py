"""
Memory pipeline orchestrator.

Coordinates the full memory pipeline from text ingestion
through embedding to storage and retrieval.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import uuid4

import structlog

from app.config import get_settings
from app.core.chunker import Chunker, ChunkConfig, ChunkingStrategy
from app.core.context import ContextAssembler
from app.models.domain import Chunk, MemoryMetadata, RetrievalResult
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vectordb import VectorDBService
from app.utils.latency import track_latency

logger = structlog.get_logger(__name__)


@dataclass
class IngestResult:
    """Result of a memory ingest operation."""

    success: bool
    memory_ids: list[str] = field(default_factory=list)
    chunks_created: int = 0
    processing_time_ms: float = 0.0
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class QueryResult:
    """Result of a memory query operation."""

    answer: str
    memories: list[RetrievalResult] = field(default_factory=list)
    query: str = ""
    processing_time_ms: float = 0.0
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    error: str | None = None


class MemoryPipeline:
    """
    Orchestrates the complete memory pipeline.

    Handles:
    - Text ingestion: chunking → embedding → storage
    - Query processing: embedding → retrieval → context → LLM
    """

    def __init__(
        self,
        vector_db: VectorDBService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ) -> None:
        """
        Initialize the memory pipeline.

        Args:
            vector_db: Vector database service
            embedding_service: Embedding service
            llm_service: LLM service
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.llm_service = llm_service

        self.settings = get_settings()
        
        # Configure chunker with selected strategy
        chunk_config = ChunkConfig(
            strategy=ChunkingStrategy(self.settings.chunk_strategy),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            similarity_threshold=self.settings.semantic_similarity_threshold,
        )
        
        # For semantic chunking, provide embedding function
        if chunk_config.strategy == ChunkingStrategy.SEMANTIC:
            # Create sync wrapper for async embedding function
            import asyncio
            def sync_embed(text: str) -> list[float]:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.embedding_service.embed_text(text))
            chunk_config.embedding_fn = sync_embed
        
        self.chunker = Chunker(chunk_config)
        self.context_assembler = ContextAssembler()
        
        logger.info(
            "memory_pipeline_initialized",
            chunk_strategy=self.settings.chunk_strategy,
            chunk_size=self.settings.chunk_size,
        )

        # Track statistics
        self._ingest_count = 0
        self._query_count = 0
        self._total_words_ingested = 0
        self._total_chunks_created = 0
        
        # Quality metrics
        self._total_retrieval_score = 0.0
        self._total_memories_retrieved = 0
        self._queries_with_results = 0  # Hit count

    async def ingest(
        self,
        text: str,
        source: str = "text",
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """
        Ingest text into the memory store.

        Pipeline: text → chunk → embed → store

        Args:
            text: Text content to ingest
            source: Source identifier
            session_id: Optional session ID for grouping
            metadata: Additional metadata

        Returns:
            IngestResult with status and IDs
        """
        start_time = time.perf_counter()
        latency_breakdown: dict[str, float] = {}

        try:
            # Create base metadata
            base_metadata = MemoryMetadata(
                source=source,
                session_id=session_id,
                custom=metadata or {},
            )

            # Step 1: Chunk the text
            async with track_latency("ingest_chunking") as chunk_timing:
                chunks = self.chunker.chunk_text(text, base_metadata)

            latency_breakdown["chunking_ms"] = chunk_timing["duration_ms"]

            if not chunks:
                return IngestResult(
                    success=True,
                    chunks_created=0,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    latency_breakdown=latency_breakdown,
                )

            # Step 2: Generate embeddings
            async with track_latency("ingest_embedding") as embed_timing:
                chunk_texts = [c.text for c in chunks]
                embeddings = await self.embedding_service.embed_texts(chunk_texts)

            latency_breakdown["embedding_ms"] = embed_timing["duration_ms"]

            # Step 3: Store in vector DB
            async with track_latency("ingest_storage") as store_timing:
                metadatas = [c.metadata for c in chunks]
                memory_ids = await self.vector_db.add_memories(
                    texts=chunk_texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

            latency_breakdown["storage_ms"] = store_timing["duration_ms"]

            # Calculate total time
            total_time = (time.perf_counter() - start_time) * 1000

            # Update statistics
            self._ingest_count += 1
            self._total_chunks_created += len(chunks)
            self._total_words_ingested += len(text.split())

            logger.info(
                "memory_ingested",
                chunks=len(chunks),
                total_ms=round(total_time, 2),
                source=source,
            )

            return IngestResult(
                success=True,
                memory_ids=memory_ids,
                chunks_created=len(chunks),
                processing_time_ms=total_time,
                latency_breakdown=latency_breakdown,
            )

        except Exception as e:
            logger.error("ingest_failed", error=str(e))
            return IngestResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                latency_breakdown=latency_breakdown,
            )

    async def query(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        stream: bool = False,
    ) -> QueryResult:
        """
        Query memories and generate a response.

        Pipeline: query → embed → search → context → LLM

        Args:
            query: User's question
            top_k: Number of memories to retrieve
            score_threshold: Minimum similarity score
            stream: Whether to stream the response

        Returns:
            QueryResult with answer and memories
        """
        start_time = time.perf_counter()
        latency_breakdown: dict[str, float] = {}

        try:
            # Step 1: Embed the query
            async with track_latency("query_embedding") as embed_timing:
                query_embedding = await self.embedding_service.embed_text(query)

            latency_breakdown["embedding_ms"] = embed_timing["duration_ms"]

            # Step 2: Search for relevant memories
            async with track_latency("query_search") as search_timing:
                memories = await self.vector_db.search(
                    embedding=query_embedding,
                    top_k=top_k or self.settings.retrieval_top_k,
                    score_threshold=score_threshold or self.settings.retrieval_score_threshold,
                )

            latency_breakdown["search_ms"] = search_timing["duration_ms"]

            # Step 3: Assemble context
            async with track_latency("query_context") as context_timing:
                # Context assembly is synchronous but we track it
                _ = self.context_assembler.assemble(query, memories)

            latency_breakdown["context_ms"] = context_timing["duration_ms"]

            # Step 4: Generate response
            async with track_latency("query_llm") as llm_timing:
                if stream:
                    # For streaming, we return a generator
                    # The actual streaming happens in the API layer
                    answer = ""  # Will be streamed
                else:
                    answer = await self.llm_service.generate(query, memories)

            latency_breakdown["llm_ms"] = llm_timing["duration_ms"]

            # Calculate total time
            total_time = (time.perf_counter() - start_time) * 1000

            # Update statistics
            self._query_count += 1
            self._total_memories_retrieved += len(memories)
            if memories:
                self._queries_with_results += 1
                self._total_retrieval_score += sum(m.score for m in memories)

            logger.info(
                "query_processed",
                memories_found=len(memories),
                total_ms=round(total_time, 2),
            )

            return QueryResult(
                answer=answer,
                memories=memories,
                query=query,
                processing_time_ms=total_time,
                latency_breakdown=latency_breakdown,
            )

        except Exception as e:
            logger.error("query_failed", error=str(e))
            return QueryResult(
                answer="",
                query=query,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                latency_breakdown=latency_breakdown,
            )

    async def get_recent_memories(
        self,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """
        Get most recent memories.

        Uses direct listing from vector store instead of similarity search.

        Args:
            limit: Maximum memories to return

        Returns:
            List of recent memories
        """
        # Use list_recent instead of similarity search
        memories = await self.vector_db.list_recent(limit=limit)

        return memories

    async def clear_memories(self) -> bool:
        """
        Clear all stored memories.

        Returns:
            True if successful
        """
        try:
            await self.vector_db.clear_all()
            logger.info("all_memories_cleared")
            return True
        except Exception as e:
            logger.error("clear_memories_failed", error=str(e))
            return False

    async def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return await self.vector_db.get_count()

    @property
    def ingest_count(self) -> int:
        """Get total ingest operations."""
        return self._ingest_count

    @property
    def query_count(self) -> int:
        """Get total query operations."""
        return self._query_count

    @property
    def total_words_ingested(self) -> int:
        """Get total words ingested."""
        return self._total_words_ingested

    @property
    def total_chunks_created(self) -> int:
        """Get total chunks created."""
        return self._total_chunks_created

    @property
    def avg_retrieval_score(self) -> float:
        """Get average retrieval score across all queries."""
        if self._total_memories_retrieved == 0:
            return 0.0
        return self._total_retrieval_score / self._total_memories_retrieved

    @property
    def hit_rate(self) -> float:
        """Get hit rate (% of queries that found at least 1 memory)."""
        if self._query_count == 0:
            return 0.0
        return self._queries_with_results / self._query_count

    @property
    def avg_memories_per_query(self) -> float:
        """Get average memories retrieved per query."""
        if self._query_count == 0:
            return 0.0
        return self._total_memories_retrieved / self._query_count


class MemoryPipelineFactory:
    """Factory for creating memory pipeline instances."""

    _instance: MemoryPipeline | None = None

    @classmethod
    async def create(
        cls,
        vector_db: VectorDBService | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
    ) -> MemoryPipeline:
        """
        Create a memory pipeline instance.

        Args:
            vector_db: Optional vector DB service
            embedding_service: Optional embedding service
            llm_service: Optional LLM service

        Returns:
            Configured MemoryPipeline
        """
        # Create services if not provided
        if vector_db is None:
            vector_db = VectorDBService()
            await vector_db.initialize()

        if embedding_service is None:
            embedding_service = EmbeddingService()

        if llm_service is None:
            llm_service = LLMService()

        return MemoryPipeline(
            vector_db=vector_db,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )

    @classmethod
    def get_instance(cls) -> MemoryPipeline | None:
        """Get the singleton pipeline instance."""
        return cls._instance

    @classmethod
    def set_instance(cls, pipeline: MemoryPipeline) -> None:
        """Set the singleton pipeline instance."""
        cls._instance = pipeline
