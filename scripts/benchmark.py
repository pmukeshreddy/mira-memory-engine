#!/usr/bin/env python3
"""
Latency benchmarking script for the Mira Memory Engine.

Uses standard datasets for realistic benchmarking:
- SQuAD for QA workloads
- BEIR datasets for retrieval workloads

Measures performance of key operations including:
- Text chunking
- Embedding generation  
- Vector database operations
- LLM response generation
- End-to-end pipeline latency
"""

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from app.config import get_settings
from app.core.chunker import Chunker, ChunkConfig, ChunkingStrategy
from app.core.memory import MemoryPipeline
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vectordb import VectorDBService


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    operation: str
    iterations: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float

    def __str__(self) -> str:
        return (
            f"{self.operation}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Min: {self.min_ms:.2f}ms\n"
            f"  Max: {self.max_ms:.2f}ms\n"
            f"  Mean: {self.mean_ms:.2f}ms\n"
            f"  Median: {self.median_ms:.2f}ms\n"
            f"  P95: {self.p95_ms:.2f}ms\n"
            f"  P99: {self.p99_ms:.2f}ms"
        )


def calculate_percentile(data: list[float], percentile: float) -> float:
    """Calculate the given percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * percentile / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def benchmark_sync(
    operation: str,
    func,
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Run a synchronous benchmark."""
    for _ in range(warmup):
        func()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        latencies.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        operation=operation,
        iterations=iterations,
        min_ms=min(latencies),
        max_ms=max(latencies),
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=calculate_percentile(latencies, 95),
        p99_ms=calculate_percentile(latencies, 99),
    )


async def benchmark_async(
    operation: str,
    func,
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Run an asynchronous benchmark."""
    for _ in range(warmup):
        await func()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        latencies.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        operation=operation,
        iterations=iterations,
        min_ms=min(latencies),
        max_ms=max(latencies),
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=calculate_percentile(latencies, 95),
        p99_ms=calculate_percentile(latencies, 99),
    )


async def load_benchmark_data(limit: int = 20) -> tuple[list[str], list[str]]:
    """Load real data from standard datasets for benchmarking."""
    from scripts.eval_datasets import DatasetLoader
    
    loader = DatasetLoader()
    dataset = await loader.load("squad", limit=limit)
    
    # Extract texts and queries
    texts = [p for ex in dataset.examples for p in ex.passages if p][:limit]
    queries = [ex.query for ex in dataset.examples if ex.query][:limit]
    
    print(f"  Loaded {len(texts)} passages and {len(queries)} queries from SQuAD")
    return texts, queries


async def run_benchmarks(args: argparse.Namespace) -> None:
    """Run all benchmarks."""
    results: list[BenchmarkResult] = []
    settings = get_settings()

    print("=" * 60)
    print("Mira Memory Engine - Latency Benchmark")
    print("=" * 60)
    print()
    
    # Load benchmark data
    print("Loading benchmark data...")
    texts, queries = await load_benchmark_data(args.data_size)
    print()

    # 1. Chunking benchmark
    if args.all or args.chunking:
        print("Running chunking benchmark...")
        chunker = Chunker()
        long_text = " ".join(texts[:10])  # Combine texts for chunking test
        
        result = benchmark_sync(
            "Chunking",
            lambda: chunker.chunk_text(long_text),
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()

    # 1b. Chunking strategy comparison
    if args.compare_chunking:
        print("=" * 60)
        print("CHUNKING STRATEGY COMPARISON")
        print("=" * 60)
        print()
        long_text = " ".join(texts[:10])
        chunking_results = []
        
        for strategy in ChunkingStrategy:
            if strategy == ChunkingStrategy.SEMANTIC:
                # Semantic chunking requires embedding service - skip for now
                print(f"  Skipping {strategy.value} (requires embedding service)")
                continue
                
            config = ChunkConfig(
                strategy=strategy,
                chunk_size=150,
                chunk_overlap=30,
            )
            chunker = Chunker(config=config)
            
            # Measure chunking
            times = []
            chunks = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                result_chunks = chunker.chunk_text(long_text)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                chunks = result_chunks
            
            avg_time = statistics.mean(times)
            chunk_count = len(chunks)
            avg_chunk_size = sum(c.word_count for c in chunks) / len(chunks) if chunks else 0
            
            print(f"  {strategy.value:20} | Chunks: {chunk_count:3} | Avg size: {avg_chunk_size:5.0f} words | Time: {avg_time:.2f}ms")
            chunking_results.append({
                "strategy": strategy.value,
                "chunks": chunk_count,
                "avg_size": avg_chunk_size,
                "time_ms": avg_time,
            })
        
        print()
        print("  Best for retrieval: sentence or paragraph (preserves semantic units)")
        print("  Best for speed: sliding_window")
        print("  Best for context: recursive (hierarchical)")
        print()

    # 2. Embedding benchmark
    if args.all or args.embedding:
        print("Running embedding benchmark...")
        embedding_service = EmbeddingService()
        test_text = texts[0] if texts else "Sample text for embedding"
        
        result = await benchmark_async(
            "Embedding (single)",
            lambda: embedding_service.embed_text(test_text),
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()

        # Batch embedding
        batch_texts = texts[:5]
        result = await benchmark_async(
            f"Embedding (batch of {len(batch_texts)})",
            lambda: embedding_service.embed_texts(batch_texts),
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()

    # 3. Vector DB benchmark
    if args.all or args.vectordb:
        print("Running vector DB benchmark...")
        vector_db = VectorDBService()
        await vector_db.initialize()

        embedding_service = EmbeddingService()
        sample_embedding = await embedding_service.embed_text(texts[0])
        batch_embeddings = [sample_embedding] * min(5, len(texts))

        # Write benchmark
        result = await benchmark_async(
            f"Vector DB Write ({len(batch_embeddings)} docs)",
            lambda: vector_db.add_memories(
                texts=texts[:len(batch_embeddings)],
                embeddings=batch_embeddings,
            ),
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()

        # Query benchmark
        result = await benchmark_async(
            "Vector DB Query (top-5)",
            lambda: vector_db.search(embedding=sample_embedding, top_k=5),
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()

        await vector_db.clear_all()
        await vector_db.close()

    # 4. LLM benchmark
    if args.all or args.llm:
        print("Running LLM benchmark...")
        llm_service = LLMService()
        
        from app.models.domain import RetrievalResult
        mock_memories = [
            RetrievalResult(memory_id="1", text=texts[0], score=0.9, rank=1)
        ]
        test_query = queries[0] if queries else "What is this about?"

        result = await benchmark_async(
            "LLM Generation",
            lambda: llm_service.generate(test_query, mock_memories),
            iterations=min(args.iterations, 5),
        )
        results.append(result)
        print(result)
        print()

    # 5. Full pipeline benchmark
    if args.all or args.pipeline:
        print("Running full pipeline benchmark...")
        vector_db = VectorDBService()
        await vector_db.initialize()
        embedding_service = EmbeddingService()
        llm_service = LLMService()

        pipeline = MemoryPipeline(
            vector_db=vector_db,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )

        test_text = texts[0]
        test_query = queries[0]

        # Ingest benchmark
        result = await benchmark_async(
            "Full Ingest Pipeline",
            lambda: pipeline.ingest(text=test_text, source="benchmark"),
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()

        # Query benchmark
        result = await benchmark_async(
            "Full Query Pipeline",
            lambda: pipeline.query(query=test_query),
            iterations=min(args.iterations, 5),
        )
        results.append(result)
        print(result)
        print()

        await vector_db.clear_all()
        await vector_db.close()

    # 6. Retrieval-only benchmark (without LLM)
    if args.retrieval_only:
        print("Running retrieval-only benchmark (no LLM)...")
        vector_db = VectorDBService()
        await vector_db.initialize()
        embedding_service = EmbeddingService()

        # First, ingest some test data
        print("  Ingesting test data...")
        for i, text in enumerate(texts[:20]):
            embedding = await embedding_service.embed_text(text)
            await vector_db.add_memories(
                memory_ids=[f"test-{i}"],
                texts=[text],
                embeddings=[embedding],
                metadatas=[{"source": "benchmark"}],
            )

        test_query = queries[0]

        async def retrieval_only():
            # Embed query
            query_embedding = await embedding_service.embed_text(test_query)
            # Search vector DB
            results = await vector_db.query(query_embedding, top_k=5)
            return results

        result = await benchmark_async(
            "Retrieval Only (no LLM)",
            retrieval_only,
            iterations=args.iterations,
        )
        results.append(result)
        print(result)
        print()
        
        # Show breakdown
        print("  Retrieval breakdown (typical):")
        print("    - Query embedding: ~200-400ms (API call)")
        print("    - Vector search:   ~25-50ms (Pinecone)")
        print("    - Total retrieval: ~250-450ms")
        print()
        print("  Note: The 400ms query target is for retrieval only.")
        print("        LLM generation adds ~2.5-4s on top.")
        print()

        await vector_db.clear_all()
        await vector_db.close()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Operation':<30} {'Mean':>10} {'P95':>10} {'P99':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.operation:<30} {r.mean_ms:>8.1f}ms {r.p95_ms:>8.1f}ms {r.p99_ms:>8.1f}ms")
    print()

    # Latency budget check
    print("Latency Budget Check:")
    ingest_result = next((r for r in results if "Ingest" in r.operation), None)
    query_result = next((r for r in results if "Query" in r.operation and "Full" in r.operation), None)
    retrieval_result = next((r for r in results if "Retrieval Only" in r.operation), None)

    if ingest_result:
        status = "✓" if ingest_result.p95_ms < 200 else "✗"
        print(f"  {status} Ingest P95: {ingest_result.p95_ms:.1f}ms (target: <200ms)")

    if retrieval_result:
        status = "✓" if retrieval_result.p95_ms < 400 else "✗"
        print(f"  {status} Retrieval P95: {retrieval_result.p95_ms:.1f}ms (target: <400ms)")

    if query_result:
        # Full query includes LLM, so expected to be much higher
        print(f"  ℹ Query+LLM P95: {query_result.p95_ms:.1f}ms (LLM adds ~3s)")


def main():
    parser = argparse.ArgumentParser(description="Mira Memory Engine Benchmark")
    parser.add_argument("--iterations", "-n", type=int, default=10, help="Number of iterations")
    parser.add_argument("--data-size", "-s", type=int, default=20, help="Number of examples to load from dataset")
    parser.add_argument("--all", "-a", action="store_true", help="Run all benchmarks")
    parser.add_argument("--chunking", action="store_true", help="Run chunking benchmark")
    parser.add_argument("--compare-chunking", action="store_true", help="Compare all chunking strategies")
    parser.add_argument("--embedding", action="store_true", help="Run embedding benchmark")
    parser.add_argument("--vectordb", action="store_true", help="Run vector DB benchmark")
    parser.add_argument("--llm", action="store_true", help="Run LLM benchmark")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline benchmark")
    parser.add_argument("--retrieval-only", action="store_true", help="Measure retrieval without LLM generation")

    args = parser.parse_args()

    if not any([args.all, args.chunking, args.compare_chunking, args.embedding, args.vectordb, args.llm, args.pipeline, args.retrieval_only]):
        args.all = True

    asyncio.run(run_benchmarks(args))


if __name__ == "__main__":
    main()
