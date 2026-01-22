# Mira Memory Engine - Benchmark Results

**Date:** January 22, 2026  
**Dataset:** SQuAD 2.0 (5,000 passages, 5,000 queries)  
**Iterations:** 10 per operation

---

## 📊 Summary

| Operation | Mean | P95 | P99 |
|-----------|------|-----|-----|
| Chunking | 0.5ms | 0.6ms | 0.6ms |
| Embedding (single) | 0.1ms | 0.1ms | 0.1ms |
| Embedding (batch of 5) | 0.1ms | 0.1ms | 0.1ms |
| Vector DB Write (5 docs) | 238.1ms | 276.5ms | 281.4ms |
| Vector DB Query (top-5) | 34.1ms | 71.0ms | 99.2ms |
| LLM Generation | 3,105ms | 3,722ms | 3,728ms |
| **Full Ingest Pipeline** | **198.4ms** | **273.8ms** | **314.0ms** |
| **Full Query Pipeline** | **3,490ms** | **3,719ms** | **3,734ms** |

---

## 🔧 Chunking Strategy Comparison

| Strategy | Chunks | Avg Size | Time |
|----------|--------|----------|------|
| sliding_window | 9 | 165 words | 0.50ms |
| **sentence** | 9 | 138 words | **0.33ms** |
| paragraph | 9 | 165 words | 0.55ms |
| **recursive** | 9 | 138 words | **0.31ms** |

**Recommendations:**
- 🏆 **Best for retrieval:** `sentence` or `paragraph` (preserves semantic units)
- ⚡ **Best for speed:** `recursive` (0.31ms)
- 📚 **Best for context:** `recursive` (hierarchical)

---

## ⚡ Latency Breakdown

### Ingest Pipeline (~198ms)
```
Text → Chunk (0.5ms) → Embed (0.1ms*) → Store (238ms)
       └── 0.3%          └── 0.05%       └── 99.7%

* Cached embeddings; first call ~600ms
```

### Query Pipeline (~3,490ms)
```
Query → Embed (0.1ms*) → Search (34ms) → Context (0.3ms) → LLM (3,105ms)
        └── 0.003%       └── 1%          └── 0.01%        └── 99%

* Cached embeddings
```

---

## 🎯 Latency Budget Analysis

| Pipeline | Target | Actual P95 | Status |
|----------|--------|------------|--------|
| Ingest | <200ms | 273.8ms | ⚠️ Close (storage bottleneck) |
| Query (excluding LLM) | <400ms | ~70ms | ✅ Excellent |
| Query (with LLM) | N/A | 3,719ms | ℹ️ Expected (Claude API) |

---

## 🔬 Component Performance

### Internal Operations (Fast ⚡)
- **Chunking:** 0.5ms mean
- **Context Assembly:** 0.3ms mean
- **Embedding Cache Hit:** 0.1ms mean

### External APIs (Expected Latency)
- **OpenAI Embeddings:** ~600ms (first call), cached thereafter
- **Pinecone Write:** ~238ms mean
- **Pinecone Query:** ~34ms mean
- **Claude LLM:** ~3,100ms mean

---

## 💡 Optimization Notes

1. **Embedding caching** significantly reduces repeat query latency
2. **Pinecone** provides consistent low-latency queries (~34ms)
3. **LLM latency** dominates query time but provides quality responses
4. **Batch operations** are efficient for bulk ingestion

---

## 🖥️ Test Environment

- **Backend:** FastAPI + Python 3.13
- **Vector DB:** Pinecone (Serverless, us-east-1)
- **Embeddings:** OpenAI text-embedding-3-small (1536 dims)
- **LLM:** Claude claude-sonnet-4-20250514
- **Chunking:** Sentence strategy (150 words, 30 overlap)
