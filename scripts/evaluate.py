#!/usr/bin/env python3
"""
RAG Evaluation Script using Standard Datasets and Metrics.

Evaluates retrieval and generation quality using industry-standard:
- Datasets: MS MARCO, Natural Questions, HotpotQA, SQuAD, BEIR
- Metrics: MRR, NDCG@k, Recall@k, Hit Rate, F1, Exact Match

Usage:
    python scripts/evaluate.py --dataset squad --limit 100
    python scripts/evaluate.py --dataset fiqa --metrics retrieval
    python scripts/evaluate.py --all --limit 50
"""

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from scripts.eval_datasets import DatasetLoader, EvalDataset, QAExample

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    
    mrr: float = 0.0           # Mean Reciprocal Rank
    ndcg_5: float = 0.0        # NDCG@5
    ndcg_10: float = 0.0       # NDCG@10
    recall_5: float = 0.0      # Recall@5
    recall_10: float = 0.0     # Recall@10
    hit_rate_5: float = 0.0    # Hit Rate@5
    hit_rate_10: float = 0.0   # Hit Rate@10
    avg_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "mrr": round(self.mrr, 4),
            "ndcg@5": round(self.ndcg_5, 4),
            "ndcg@10": round(self.ndcg_10, 4),
            "recall@5": round(self.recall_5, 4),
            "recall@10": round(self.recall_10, 4),
            "hit_rate@5": round(self.hit_rate_5, 4),
            "hit_rate@10": round(self.hit_rate_10, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""
    
    exact_match: float = 0.0   # Exact match accuracy
    f1: float = 0.0            # Token F1 score
    bleu: float = 0.0          # BLEU score
    rouge_l: float = 0.0       # ROUGE-L score
    avg_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "exact_match": round(self.exact_match, 4),
            "f1": round(self.f1, 4),
            "bleu": round(self.bleu, 4),
            "rouge_l": round(self.rouge_l, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class EvalResult:
    """Complete evaluation result."""
    
    dataset: str
    num_examples: int
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "num_examples": self.num_examples,
            "timestamp": self.timestamp,
            "config": self.config,
            "retrieval_metrics": self.retrieval.to_dict(),
            "generation_metrics": self.generation.to_dict(),
        }


class MetricsCalculator:
    """Calculate standard IR and NLP metrics."""
    
    @staticmethod
    def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
        """Calculate Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def dcg(relevances: list[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            dcg += rel / math.log2(i + 2)
        return dcg
    
    @staticmethod
    def ndcg(relevant: set[str], retrieved: list[str], k: int) -> float:
        """Calculate Normalized DCG at k."""
        # Relevance scores (binary)
        relevances = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
        
        dcg = MetricsCalculator.dcg(relevances, k)
        
        # Ideal DCG (all relevant docs first)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = MetricsCalculator.dcg(ideal_relevances, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
        """Calculate Recall at k."""
        if not relevant:
            return 0.0
        retrieved_set = set(retrieved[:k])
        return len(relevant & retrieved_set) / len(relevant)
    
    @staticmethod
    def hit_rate_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
        """Calculate Hit Rate at k (1 if any relevant in top-k)."""
        retrieved_set = set(retrieved[:k])
        return 1.0 if relevant & retrieved_set else 0.0
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer for comparison."""
        import re
        import string
        
        text = text.lower()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def exact_match(prediction: str, ground_truths: list[str]) -> float:
        """Calculate Exact Match score."""
        pred_normalized = MetricsCalculator.normalize_answer(prediction)
        for gt in ground_truths:
            if pred_normalized == MetricsCalculator.normalize_answer(gt):
                return 1.0
        return 0.0
    
    @staticmethod
    def f1_score(prediction: str, ground_truths: list[str]) -> float:
        """Calculate token-level F1 score."""
        pred_tokens = MetricsCalculator.normalize_answer(prediction).split()
        
        best_f1 = 0.0
        for gt in ground_truths:
            gt_tokens = MetricsCalculator.normalize_answer(gt).split()
            
            common = set(pred_tokens) & set(gt_tokens)
            if not common:
                continue
            
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(gt_tokens) if gt_tokens else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        
        return best_f1


class RAGEvaluator:
    """
    Evaluate RAG pipeline on standard datasets.
    
    Measures both retrieval quality and generation quality
    using industry-standard metrics.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = MetricsCalculator()
        self.pipeline = None
        self.vector_db = None
    
    async def setup(self):
        """Initialize the RAG pipeline."""
        from app.core.memory import MemoryPipeline
        from app.services.embeddings import EmbeddingService
        from app.services.llm import LLMService
        from app.services.vectordb import VectorDBService
        
        self.vector_db = VectorDBService()
        await self.vector_db.initialize()
        
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        
        self.pipeline = MemoryPipeline(
            vector_db=self.vector_db,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )
        
        logger.info("evaluator_initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.vector_db:
            await self.vector_db.clear_all()
            await self.vector_db.close()
    
    async def evaluate(
        self,
        dataset: EvalDataset,
        eval_retrieval: bool = True,
        eval_generation: bool = True,
        top_k: int = 10,
    ) -> EvalResult:
        """
        Evaluate the RAG pipeline on a dataset.
        
        Args:
            dataset: Evaluation dataset
            eval_retrieval: Whether to evaluate retrieval
            eval_generation: Whether to evaluate generation
            top_k: Number of results to retrieve
            
        Returns:
            EvalResult with all metrics
        """
        import time
        
        logger.info("starting_evaluation", 
                   dataset=dataset.name,
                   num_examples=len(dataset),
                   eval_retrieval=eval_retrieval,
                   eval_generation=eval_generation)
        
        # Index the corpus if available
        if dataset.corpus:
            await self._index_corpus(dataset.corpus)
        else:
            # Index passages from examples
            await self._index_passages(dataset.examples)
        
        # Collect metrics
        retrieval_metrics = []
        generation_metrics = []
        retrieval_latencies = []
        generation_latencies = []
        
        for i, example in enumerate(dataset.examples):
            if not example.query:
                continue
            
            # Retrieval evaluation
            if eval_retrieval:
                start = time.perf_counter()
                retrieved = await self._retrieve(example.query, top_k)
                retrieval_latencies.append((time.perf_counter() - start) * 1000)
                
                retrieved_ids = [r["id"] for r in retrieved]
                relevant_ids = set(example.relevant_ids) if example.relevant_ids else set()
                
                # If no explicit relevant IDs, use passage matching
                if not relevant_ids and example.passages:
                    relevant_ids = await self._find_relevant_ids(example.passages)
                
                if relevant_ids:
                    retrieval_metrics.append({
                        "rr": self.metrics.reciprocal_rank(relevant_ids, retrieved_ids),
                        "ndcg_5": self.metrics.ndcg(relevant_ids, retrieved_ids, 5),
                        "ndcg_10": self.metrics.ndcg(relevant_ids, retrieved_ids, 10),
                        "recall_5": self.metrics.recall_at_k(relevant_ids, retrieved_ids, 5),
                        "recall_10": self.metrics.recall_at_k(relevant_ids, retrieved_ids, 10),
                        "hit_5": self.metrics.hit_rate_at_k(relevant_ids, retrieved_ids, 5),
                        "hit_10": self.metrics.hit_rate_at_k(relevant_ids, retrieved_ids, 10),
                    })
            
            # Generation evaluation
            if eval_generation and example.answers:
                start = time.perf_counter()
                result = await self.pipeline.query(example.query, top_k=top_k)
                generation_latencies.append((time.perf_counter() - start) * 1000)
                
                prediction = result.answer
                
                generation_metrics.append({
                    "em": self.metrics.exact_match(prediction, example.answers),
                    "f1": self.metrics.f1_score(prediction, example.answers),
                })
            
            # Progress
            if (i + 1) % 10 == 0:
                logger.info("evaluation_progress", completed=i+1, total=len(dataset.examples))
        
        # Aggregate metrics
        ret = RetrievalMetrics()
        if retrieval_metrics:
            ret.mrr = sum(m["rr"] for m in retrieval_metrics) / len(retrieval_metrics)
            ret.ndcg_5 = sum(m["ndcg_5"] for m in retrieval_metrics) / len(retrieval_metrics)
            ret.ndcg_10 = sum(m["ndcg_10"] for m in retrieval_metrics) / len(retrieval_metrics)
            ret.recall_5 = sum(m["recall_5"] for m in retrieval_metrics) / len(retrieval_metrics)
            ret.recall_10 = sum(m["recall_10"] for m in retrieval_metrics) / len(retrieval_metrics)
            ret.hit_rate_5 = sum(m["hit_5"] for m in retrieval_metrics) / len(retrieval_metrics)
            ret.hit_rate_10 = sum(m["hit_10"] for m in retrieval_metrics) / len(retrieval_metrics)
        if retrieval_latencies:
            ret.avg_latency_ms = sum(retrieval_latencies) / len(retrieval_latencies)
        
        gen = GenerationMetrics()
        if generation_metrics:
            gen.exact_match = sum(m["em"] for m in generation_metrics) / len(generation_metrics)
            gen.f1 = sum(m["f1"] for m in generation_metrics) / len(generation_metrics)
        if generation_latencies:
            gen.avg_latency_ms = sum(generation_latencies) / len(generation_latencies)
        
        return EvalResult(
            dataset=dataset.name,
            num_examples=len(dataset.examples),
            retrieval=ret,
            generation=gen,
            config={
                "top_k": top_k,
                "eval_retrieval": eval_retrieval,
                "eval_generation": eval_generation,
            },
        )
    
    async def _index_corpus(self, corpus: dict[str, str]) -> None:
        """Index corpus documents."""
        logger.info("indexing_corpus", num_docs=len(corpus))
        
        batch_size = 50
        items = list(corpus.items())
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for doc_id, text in batch:
                if text:
                    await self.pipeline.ingest(
                        text=text[:5000],  # Limit text size
                        source="corpus",
                        metadata={"doc_id": doc_id},
                    )
    
    async def _index_passages(self, examples: list[QAExample]) -> None:
        """Index passages from examples."""
        logger.info("indexing_passages", num_examples=len(examples))
        
        for i, example in enumerate(examples):
            for j, passage in enumerate(example.passages):
                if passage:
                    await self.pipeline.ingest(
                        text=passage[:5000],
                        source="passage",
                        metadata={"example_id": example.query_id, "passage_idx": j},
                    )
    
    async def _retrieve(self, query: str, top_k: int) -> list[dict]:
        """Retrieve documents for a query."""
        embedding = await self.pipeline.embedding_service.embed_text(query)
        results = await self.vector_db.search(embedding=embedding, top_k=top_k)
        
        return [
            {"id": r.metadata.get("doc_id", r.memory_id), "score": r.score, "text": r.text}
            for r in results
        ]
    
    async def _find_relevant_ids(self, passages: list[str]) -> set[str]:
        """Find IDs of indexed passages that match the ground truth."""
        relevant = set()
        for passage in passages:
            if passage:
                embedding = await self.pipeline.embedding_service.embed_text(passage[:500])
                results = await self.vector_db.search(embedding=embedding, top_k=1, score_threshold=0.9)
                for r in results:
                    relevant.add(r.metadata.get("doc_id", r.memory_id))
        return relevant


async def run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation on specified datasets."""
    loader = DatasetLoader()
    evaluator = RAGEvaluator()
    
    print("=" * 70)
    print("Mira Memory Engine - RAG Evaluation")
    print("=" * 70)
    print()
    
    # Initialize
    await evaluator.setup()
    
    results = []
    datasets_to_eval = args.datasets if args.datasets else ["squad"]
    
    for ds_name in datasets_to_eval:
        try:
            print(f"\n{'='*70}")
            print(f"Evaluating on: {ds_name}")
            print("=" * 70)
            
            # Load dataset
            print(f"Loading dataset (limit={args.limit})...")
            dataset = await loader.load(ds_name, limit=args.limit)
            print(f"Loaded {len(dataset)} examples")
            
            if len(dataset) == 0:
                print(f"  ⚠ No examples loaded for {ds_name}, skipping...")
                continue
            
            # Evaluate
            print("\nRunning evaluation...")
            result = await evaluator.evaluate(
                dataset,
                eval_retrieval=args.metrics in ["all", "retrieval"],
                eval_generation=args.metrics in ["all", "generation"],
                top_k=args.top_k,
            )
            
            results.append(result)
            
            # Print results
            print(f"\n📊 Results for {ds_name}:")
            print("-" * 40)
            
            if args.metrics in ["all", "retrieval"]:
                print("\nRetrieval Metrics:")
                r = result.retrieval
                print(f"  MRR:        {r.mrr:.4f}")
                print(f"  NDCG@5:     {r.ndcg_5:.4f}")
                print(f"  NDCG@10:    {r.ndcg_10:.4f}")
                print(f"  Recall@5:   {r.recall_5:.4f}")
                print(f"  Recall@10:  {r.recall_10:.4f}")
                print(f"  Hit@5:      {r.hit_rate_5:.4f}")
                print(f"  Hit@10:     {r.hit_rate_10:.4f}")
                print(f"  Avg Latency: {r.avg_latency_ms:.1f}ms")
            
            if args.metrics in ["all", "generation"]:
                print("\nGeneration Metrics:")
                g = result.generation
                print(f"  Exact Match: {g.exact_match:.4f}")
                print(f"  F1 Score:    {g.f1:.4f}")
                print(f"  Avg Latency: {g.avg_latency_ms:.1f}ms")
            
            # Clear for next dataset
            await evaluator.vector_db.clear_all()
            
        except Exception as e:
            logger.error("evaluation_failed", dataset=ds_name, error=str(e))
            print(f"  ✗ Evaluation failed: {e}")
    
    # Cleanup
    await evaluator.cleanup()
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<20} {'MRR':>8} {'NDCG@10':>10} {'F1':>8} {'Latency':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r.dataset:<20} {r.retrieval.mrr:>8.4f} {r.retrieval.ndcg_10:>10.4f} "
              f"{r.generation.f1:>8.4f} {r.retrieval.avg_latency_ms:>8.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline on standard datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --dataset squad --limit 50
  python scripts/evaluate.py --dataset fiqa scifact --limit 100
  python scripts/evaluate.py --all --limit 20 --output results.json
        """
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=["squad", "hotpotqa", "nq", "triviaqa", "fiqa", "scifact", "msmarco"],
        help="Datasets to evaluate on"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Evaluate on all available datasets"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=50,
        help="Maximum examples per dataset (default: 50)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)"
    )
    parser.add_argument(
        "--metrics", "-m",
        choices=["all", "retrieval", "generation"],
        default="all",
        help="Which metrics to evaluate (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        args.datasets = ["squad", "hotpotqa", "fiqa", "scifact"]
    elif not args.datasets:
        args.datasets = ["squad"]
    
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
