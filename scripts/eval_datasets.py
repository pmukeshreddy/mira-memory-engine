#!/usr/bin/env python3
"""
Standard dataset loader for RAG evaluation.

Supports industry-standard datasets:
- MS MARCO: Passage retrieval and QA
- Natural Questions: Real Google search questions
- HotpotQA: Multi-hop reasoning questions
- SQuAD: Reading comprehension
- TriviaQA: Trivia-based QA

Usage:
    from scripts.datasets import DatasetLoader
    loader = DatasetLoader()
    dataset = await loader.load("msmarco", split="dev", limit=100)
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Dataset configurations
DATASETS = {
    "msmarco": {
        "name": "MS MARCO Passage Ranking",
        "url": "https://huggingface.co/datasets/ms_marco/resolve/main/v2.1/dev/",
        "files": {
            "queries": "queries.dev.small.tsv",
            "qrels": "qrels.dev.small.tsv",
            "collection": "collection.tsv",
        },
        "format": "tsv",
    },
    "nq": {
        "name": "Natural Questions",
        "hf_dataset": "natural_questions",
        "split": "validation",
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "hf_dataset": "hotpot_qa",
        "config": "distractor",
        "split": "validation",
    },
    "squad": {
        "name": "SQuAD 2.0",
        "hf_dataset": "squad_v2",
        "split": "validation",
    },
    "triviaqa": {
        "name": "TriviaQA",
        "hf_dataset": "trivia_qa",
        "config": "rc",
        "split": "validation",
    },
    "fiqa": {
        "name": "FiQA (Financial QA)",
        "beir": True,
        "dataset_name": "fiqa",
    },
    "scifact": {
        "name": "SciFact",
        "beir": True,
        "dataset_name": "scifact",
    },
}

# BEIR dataset URLs (subset for quick testing)
BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"


@dataclass
class QAExample:
    """A single QA example from a dataset."""
    
    query_id: str
    query: str
    passages: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    relevant_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """A loaded evaluation dataset."""
    
    name: str
    examples: list[QAExample]
    corpus: dict[str, str] = field(default_factory=dict)  # doc_id -> text
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[QAExample]:
        return iter(self.examples)
    
    def sample(self, n: int) -> "EvalDataset":
        """Return a random sample of n examples."""
        import random
        sampled = random.sample(self.examples, min(n, len(self.examples)))
        return EvalDataset(name=self.name, examples=sampled, corpus=self.corpus)


class DatasetLoader:
    """
    Load standard evaluation datasets for RAG benchmarking.
    
    Supports HuggingFace datasets and BEIR benchmark datasets.
    """
    
    def __init__(self, cache_dir: str | None = None):
        """Initialize the dataset loader."""
        self.cache_dir = Path(cache_dir or "./data/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def load(
        self,
        dataset_name: str,
        split: str = "validation",
        limit: int | None = None,
    ) -> EvalDataset:
        """
        Load a standard evaluation dataset.
        
        Args:
            dataset_name: Name of dataset (msmarco, nq, hotpotqa, squad, triviaqa, fiqa, scifact)
            split: Dataset split to load
            limit: Maximum number of examples to load
            
        Returns:
            EvalDataset with examples and corpus
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
        
        config = DATASETS[dataset_name]
        logger.info("loading_dataset", name=dataset_name, split=split, limit=limit)
        
        if config.get("beir"):
            return await self._load_beir_dataset(config, limit)
        elif config.get("hf_dataset"):
            return await self._load_hf_dataset(config, split, limit)
        else:
            return await self._load_custom_dataset(config, split, limit)
    
    async def _load_hf_dataset(
        self,
        config: dict,
        split: str,
        limit: int | None,
    ) -> EvalDataset:
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        hf_name = config["hf_dataset"]
        hf_config = config.get("config")
        
        logger.info("loading_hf_dataset", dataset=hf_name, config=hf_config)
        
        # Load dataset
        if hf_config:
            dataset = load_dataset(hf_name, hf_config, split=split)
        else:
            dataset = load_dataset(hf_name, split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        # Convert to our format
        examples = []
        for i, item in enumerate(dataset):
            example = self._convert_hf_example(hf_name, item, str(i))
            if example:
                examples.append(example)
        
        return EvalDataset(name=config["name"], examples=examples)
    
    def _convert_hf_example(self, dataset_name: str, item: dict, idx: str) -> QAExample | None:
        """Convert HuggingFace example to our format."""
        
        if dataset_name == "natural_questions":
            # Natural Questions format
            question = item.get("question", {}).get("text", "")
            answers = []
            
            # Short answers
            annotations = item.get("annotations", {})
            if annotations:
                short_answers = annotations.get("short_answers", [])
                if short_answers and short_answers[0]:
                    answers = [sa.get("text", "") for sa in short_answers[0] if sa.get("text")]
            
            if not question:
                return None
                
            return QAExample(
                query_id=idx,
                query=question,
                answers=answers,
                passages=[item.get("document", {}).get("text", "")[:2000]],
            )
        
        elif dataset_name == "hotpot_qa":
            return QAExample(
                query_id=item.get("id", idx),
                query=item.get("question", ""),
                answers=[item.get("answer", "")],
                passages=[" ".join(item.get("context", {}).get("sentences", [[]])[0])],
                metadata={"type": item.get("type", ""), "level": item.get("level", "")},
            )
        
        elif dataset_name == "squad_v2":
            answers = item.get("answers", {}).get("text", [])
            context = item.get("context", "")
            
            return QAExample(
                query_id=item.get("id", idx),
                query=item.get("question", ""),
                answers=answers if answers else ["<unanswerable>"],
                passages=[context],
            )
        
        elif dataset_name == "trivia_qa":
            return QAExample(
                query_id=item.get("question_id", idx),
                query=item.get("question", ""),
                answers=item.get("answer", {}).get("aliases", []),
                passages=[item.get("search_context", "")[:2000]],
            )
        
        return None
    
    async def _load_beir_dataset(
        self,
        config: dict,
        limit: int | None,
    ) -> EvalDataset:
        """Load dataset from BEIR benchmark."""
        dataset_name = config["dataset_name"]
        cache_path = self.cache_dir / f"beir_{dataset_name}"
        
        # Download if not cached
        if not cache_path.exists():
            await self._download_beir_dataset(dataset_name, cache_path)
        
        # Load corpus
        corpus = {}
        corpus_file = cache_path / "corpus.jsonl"
        if corpus_file.exists():
            with open(corpus_file) as f:
                for line in f:
                    doc = json.loads(line)
                    corpus[doc["_id"]] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
        
        # Load queries and qrels
        queries = {}
        queries_file = cache_path / "queries.jsonl"
        if queries_file.exists():
            with open(queries_file) as f:
                for line in f:
                    q = json.loads(line)
                    queries[q["_id"]] = q["text"]
        
        qrels = {}
        qrels_file = cache_path / "qrels" / "test.tsv"
        if qrels_file.exists():
            with open(qrels_file) as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qid, did, rel = parts[0], parts[1], int(parts[2])
                        if rel > 0:
                            if qid not in qrels:
                                qrels[qid] = []
                            qrels[qid].append(did)
        
        # Build examples
        examples = []
        for qid, query in queries.items():
            relevant_ids = qrels.get(qid, [])
            relevant_passages = [corpus.get(did, "") for did in relevant_ids if did in corpus]
            
            examples.append(QAExample(
                query_id=qid,
                query=query,
                passages=relevant_passages[:3],  # Limit passages
                relevant_ids=relevant_ids,
            ))
            
            if limit and len(examples) >= limit:
                break
        
        return EvalDataset(name=config["name"], examples=examples, corpus=corpus)
    
    async def _download_beir_dataset(self, dataset_name: str, cache_path: Path) -> None:
        """Download BEIR dataset."""
        url = f"{BEIR_BASE_URL}{dataset_name}.zip"
        zip_path = cache_path.with_suffix(".zip")
        
        logger.info("downloading_beir_dataset", dataset=dataset_name, url=url)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=300)
            response.raise_for_status()
            
            with open(zip_path, "wb") as f:
                f.write(response.content)
        
        # Extract
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_path.parent)
        
        # Rename extracted folder
        extracted = cache_path.parent / dataset_name
        if extracted.exists() and extracted != cache_path:
            extracted.rename(cache_path)
        
        # Cleanup
        zip_path.unlink()
        logger.info("beir_dataset_downloaded", dataset=dataset_name)
    
    async def _load_custom_dataset(
        self,
        config: dict,
        split: str,
        limit: int | None,
    ) -> EvalDataset:
        """Load custom dataset (MS MARCO style)."""
        # For now, create synthetic data based on config
        # In production, download actual MS MARCO files
        logger.warning("custom_dataset_fallback", name=config["name"])
        
        return EvalDataset(
            name=config["name"],
            examples=[],
        )
    
    def list_datasets(self) -> list[dict]:
        """List available datasets."""
        return [
            {"id": k, "name": v["name"]}
            for k, v in DATASETS.items()
        ]


async def main():
    """Test dataset loading."""
    loader = DatasetLoader()
    
    print("Available datasets:")
    for ds in loader.list_datasets():
        print(f"  - {ds['id']}: {ds['name']}")
    
    print("\nLoading SQuAD sample...")
    dataset = await loader.load("squad", limit=5)
    
    print(f"\nLoaded {len(dataset)} examples:")
    for ex in dataset:
        print(f"\n  Query: {ex.query[:80]}...")
        print(f"  Answers: {ex.answers[:2]}")


if __name__ == "__main__":
    asyncio.run(main())
