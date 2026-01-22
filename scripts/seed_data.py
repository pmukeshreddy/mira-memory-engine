#!/usr/bin/env python3
"""
Seed data script for the Mira Memory Engine.

Populates the vector database using standard datasets:
- SQuAD: Reading comprehension passages
- HotpotQA: Multi-hop reasoning contexts  
- SciFact: Scientific fact verification
- FiQA: Financial domain QA

Usage:
    python scripts/seed_data.py --dataset squad --limit 100
    python scripts/seed_data.py --dataset fiqa --test
    python scripts/seed_data.py --all --limit 50
"""

import argparse
import asyncio
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from app.core.memory import MemoryPipeline
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.vectordb import VectorDBService


async def load_seed_data(dataset_name: str, limit: int) -> list[dict]:
    """Load data from standard datasets for seeding."""
    try:
        from scripts.eval_datasets import DatasetLoader
        
        loader = DatasetLoader()
        dataset = await loader.load(dataset_name, limit=limit)
        
        seed_data = []
        for ex in dataset.examples:
            # Use passages if available
            for i, passage in enumerate(ex.passages):
                if passage and len(passage) > 50:
                    seed_data.append({
                        "text": passage,
                        "source": dataset_name,
                        "session_id": f"{dataset_name}-{ex.query_id}",
                        "metadata": {
                            "query": ex.query,
                            "answers": ex.answers[:3] if ex.answers else [],
                            "passage_idx": i,
                        }
                    })
            
            # Also use queries as potential memory content
            if ex.query and ex.answers:
                seed_data.append({
                    "text": f"Question: {ex.query}\nAnswer: {ex.answers[0]}",
                    "source": f"{dataset_name}-qa",
                    "session_id": f"{dataset_name}-{ex.query_id}",
                })
        
        return seed_data[:limit]
        
    except Exception as e:
        print(f"  Warning: Could not load {dataset_name}: {e}")
        return []


async def seed_data(args: argparse.Namespace) -> None:
    """Seed the database with data from standard datasets."""
    print("=" * 60)
    print("Mira Memory Engine - Data Seeding")
    print("=" * 60)
    print()

    # Initialize services
    print("Initializing services...")
    vector_db = VectorDBService()
    await vector_db.initialize()
    
    embedding_service = EmbeddingService()
    llm_service = LLMService()

    pipeline = MemoryPipeline(
        vector_db=vector_db,
        embedding_service=embedding_service,
        llm_service=llm_service,
    )

    if args.clear:
        print("Clearing existing memories...")
        await pipeline.clear_memories()
        print("  Done.")
        print()

    # Determine datasets to load
    if args.all:
        datasets = ["squad", "hotpotqa", "fiqa", "scifact"]
    else:
        datasets = args.datasets if args.datasets else ["squad"]
    
    total_chunks = 0
    total_memories = 0
    
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Loading data from: {ds_name}")
        print("=" * 60)
        
        # Load data
        print(f"Fetching up to {args.limit} examples...")
        seed_items = await load_seed_data(ds_name, args.limit)
        
        if not seed_items:
            print(f"  ⚠ No data loaded for {ds_name}, skipping...")
            continue
        
        print(f"Loaded {len(seed_items)} items to ingest")
        print()
        
        # Ingest data
        for i, item in enumerate(seed_items, 1):
            print(f"  [{i}/{len(seed_items)}] Ingesting: {item['source']} - {item.get('session_id', 'unnamed')[:30]}...")

            result = await pipeline.ingest(
                text=item["text"],
                source=item["source"],
                session_id=item.get("session_id"),
                metadata=item.get("metadata", {}),
            )

            if result.success:
                print(f"    ✓ Created {result.chunks_created} chunks in {result.processing_time_ms:.0f}ms")
                total_chunks += result.chunks_created
                total_memories += 1
            else:
                print(f"    ✗ Failed: {result.error}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Datasets processed: {len(datasets)}")
    print(f"  Memories ingested: {total_memories}")
    print(f"  Total chunks created: {total_chunks}")
    
    memory_count = await vector_db.get_count()
    print(f"  Total memories in database: {memory_count}")
    print()

    # Test query
    if args.test:
        print("Running test queries...")
        
        test_queries = [
            "What is machine learning?",
            "How does natural language processing work?",
            "What are the key concepts in artificial intelligence?",
        ]
        
        for query in test_queries[:2]:
            print(f"\n📝 Query: '{query}'")
            
            result = await pipeline.query(query=query, top_k=3)
            
            print(f"   Retrieved {len(result.memories)} memories:")
            for mem in result.memories[:2]:
                print(f"   - Score: {mem.score:.2f} | {mem.text[:80]}...")
            
            if result.answer:
                print(f"\n   💬 Answer: {result.answer[:200]}...")
            
            print(f"   ⏱ Processing time: {result.processing_time_ms:.0f}ms")

    # Cleanup
    await vector_db.close()
    print("\n✓ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Seed Mira Memory Engine with standard datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/seed_data.py --dataset squad --limit 50
  python scripts/seed_data.py --dataset fiqa hotpotqa --test
  python scripts/seed_data.py --all --limit 20 --clear
        """
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=["squad", "hotpotqa", "fiqa", "scifact", "nq", "triviaqa"],
        help="Datasets to seed from"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Seed from all available datasets"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=50,
        help="Maximum items per dataset (default: 50)"
    )
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear existing data before seeding"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run test queries after seeding"
    )

    args = parser.parse_args()
    asyncio.run(seed_data(args))


if __name__ == "__main__":
    main()
