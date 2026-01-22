"""
Vector database service using Pinecone.

Provides vector storage and retrieval using Pinecone serverless.
"""

import os
from datetime import datetime
from typing import Any, Sequence
from uuid import uuid4

import structlog

from app.config import get_settings
from app.models.domain import Memory, MemoryMetadata, RetrievalResult
from app.utils.latency import latency_tracked

logger = structlog.get_logger(__name__)


class PineconeStore:
    """Pinecone vector store implementation."""

    def __init__(self) -> None:
        """Initialize Pinecone store."""
        self.settings = get_settings()
        self._client = None
        self._index = None

    async def initialize(self) -> None:
        """Initialize Pinecone client and index."""
        from pinecone import Pinecone
        from pinecone.exceptions import UnauthorizedException

        api_key = self.settings.pinecone_api_key.get_secret_value()
        if not api_key or api_key == "your_pinecone_api_key_here":
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Please set a valid Pinecone API key in your .env file. "
                "Get your key at: https://app.pinecone.io/"
            )

        try:
            self._client = Pinecone(api_key=api_key)

            # Get or create index
            index_name = self.settings.pinecone_index_name

            # Check if index exists
            existing_indexes = [idx.name for idx in self._client.list_indexes()]

            if index_name not in existing_indexes:
                # Create index
                from pinecone import ServerlessSpec
                
                self._client.create_index(
                    name=index_name,
                    dimension=self.settings.embedding_dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.settings.pinecone_environment,
                    ),
                )
                logger.info("pinecone_index_created", index=index_name)

            self._index = self._client.Index(index_name)

            logger.info(
                "pinecone_initialized",
                index=index_name,
                environment=self.settings.pinecone_environment,
            )
        except UnauthorizedException:
            raise ValueError(
                "Invalid Pinecone API key. "
                "Please check your PINECONE_API_KEY in the .env file. "
                "Get your key at: https://app.pinecone.io/"
            )

    @latency_tracked("vectordb_add")
    async def add(
        self,
        texts: Sequence[str],
        embeddings: Sequence[list[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Add vectors to Pinecone."""
        if not self._index:
            raise RuntimeError("Pinecone not initialized")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(texts))]

        # Prepare vectors
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            meta = metadatas[i] if metadatas else {}
            meta["text"] = text[:40000]  # Pinecone metadata limit

            # Clean metadata
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, datetime):
                    clean_meta[k] = v.isoformat()
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    continue
                else:
                    clean_meta[k] = str(v)

            vectors.append({
                "id": ids[i],
                "values": embedding,
                "metadata": clean_meta,
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch)

        logger.debug("pinecone_vectors_added", count=len(ids))
        return list(ids)

    @latency_tracked("vectordb_query")
    async def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Query Pinecone by vector similarity."""
        if not self._index:
            raise RuntimeError("Pinecone not initialized")

        # Query index
        results = self._index.query(
            vector=embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
        )

        # Convert to RetrievalResult
        retrieval_results = []

        for i, match in enumerate(results.matches):
            if match.score < score_threshold:
                continue

            text = match.metadata.get("text", "") if match.metadata else ""

            retrieval_results.append(
                RetrievalResult(
                    memory_id=match.id,
                    text=text,
                    score=match.score,
                    rank=i + 1,
                    metadata=dict(match.metadata) if match.metadata else {},
                )
            )

        logger.debug(
            "pinecone_query_completed",
            top_k=top_k,
            results=len(retrieval_results),
        )

        return retrieval_results

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors from Pinecone."""
        if not self._index:
            raise RuntimeError("Pinecone not initialized")

        self._index.delete(ids=list(ids))
        logger.debug("pinecone_vectors_deleted", count=len(ids))

    async def delete_all(self) -> None:
        """Delete all vectors from the index."""
        if not self._index:
            raise RuntimeError("Pinecone not initialized")

        self._index.delete(delete_all=True)
        logger.info("pinecone_index_cleared")

    async def count(self) -> int:
        """Get total vector count."""
        if not self._index:
            return 0

        stats = self._index.describe_index_stats()
        return stats.total_vector_count

    async def list_recent(self, limit: int = 50) -> list[RetrievalResult]:
        """List recent vectors from the index."""
        if not self._index:
            return []

        try:
            # List vector IDs (Pinecone lists in order of insertion)
            results = []
            
            # Use list to get vector IDs
            for ids_batch in self._index.list(limit=limit):
                if not ids_batch:
                    break
                    
                # Fetch vectors with metadata
                fetched = self._index.fetch(ids=list(ids_batch))
                
                for vec_id, vec_data in fetched.vectors.items():
                    text = vec_data.metadata.get("text", "") if vec_data.metadata else ""
                    results.append(
                        RetrievalResult(
                            memory_id=vec_id,
                            text=text,
                            score=1.0,  # No similarity score for listing
                            rank=len(results) + 1,
                            metadata=dict(vec_data.metadata) if vec_data.metadata else {},
                        )
                    )
                    
                    if len(results) >= limit:
                        break
                        
                if len(results) >= limit:
                    break
                    
            logger.debug("pinecone_list_recent", count=len(results))
            return results
            
        except Exception as e:
            logger.error("pinecone_list_failed", error=str(e))
            return []

    async def close(self) -> None:
        """Close Pinecone connection."""
        # Pinecone client doesn't require explicit close
        logger.info("pinecone_connection_closed")


class VectorDBService:
    """
    Vector database service using Pinecone.

    Provides a consistent interface for vector storage operations.
    """

    def __init__(self) -> None:
        """Initialize the vector database service."""
        self.settings = get_settings()
        self._store: PineconeStore | None = None

    async def initialize(self) -> None:
        """Initialize the Pinecone store."""
        self._store = PineconeStore()
        await self._store.initialize()

        logger.info(
            "vectordb_service_initialized",
            provider="pinecone",
        )

    async def add_memories(
        self,
        texts: Sequence[str],
        embeddings: Sequence[list[float]],
        metadatas: Sequence[MemoryMetadata] | None = None,
    ) -> list[str]:
        """
        Add memories to the vector store.

        Args:
            texts: Text content of memories
            embeddings: Vector embeddings
            metadatas: Optional metadata for each memory

        Returns:
            List of created memory IDs
        """
        if not self._store:
            raise RuntimeError("Vector store not initialized")

        # Convert metadata to dicts
        meta_dicts = None
        if metadatas:
            meta_dicts = [
                {
                    "source": m.source,
                    "session_id": m.session_id or "",
                    "speaker": m.speaker or "",
                    "timestamp": m.timestamp.isoformat(),
                    "chunk_index": m.chunk_index,
                    "total_chunks": m.total_chunks,
                    **m.custom,
                }
                for m in metadatas
            ]

        return await self._store.add(
            texts=texts,
            embeddings=embeddings,
            metadatas=meta_dicts,
        )

    async def search(
        self,
        embedding: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search for similar memories.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_dict: Metadata filters

        Returns:
            List of retrieval results
        """
        if not self._store:
            raise RuntimeError("Vector store not initialized")

        return await self._store.query(
            embedding=embedding,
            top_k=top_k or self.settings.retrieval_top_k,
            score_threshold=score_threshold or self.settings.retrieval_score_threshold,
            filter_dict=filter_dict,
        )

    async def delete_memories(self, ids: Sequence[str]) -> None:
        """Delete memories by ID."""
        if not self._store:
            raise RuntimeError("Vector store not initialized")

        await self._store.delete(ids)

    async def clear_all(self) -> None:
        """Clear all memories from the store."""
        if not self._store:
            raise RuntimeError("Vector store not initialized")

        await self._store.delete_all()

    async def get_count(self) -> int:
        """Get total memory count."""
        if not self._store:
            return 0

        return await self._store.count()

    async def list_recent(self, limit: int = 50) -> list[RetrievalResult]:
        """List recent memories without requiring a search query."""
        if not self._store:
            return []

        return await self._store.list_recent(limit=limit)

    async def close(self) -> None:
        """Close the vector store connection."""
        if self._store:
            await self._store.close()
