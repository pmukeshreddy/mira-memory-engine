# Mira Memory Engine - Architecture Document

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [API Design](#api-design)
5. [Latency Optimization](#latency-optimization)
6. [Scalability Considerations](#scalability-considerations)
7. [Security](#security)
8. [Observability](#observability)

---

## System Overview

Mira Memory Engine is a real-time voice-to-memory RAG (Retrieval-Augmented Generation) system designed for low-latency conversational memory experiences.

### Design Principles

1. **Low Latency First**: Every design decision prioritizes response time
2. **Streaming Everything**: Use streaming wherever possible to reduce perceived latency
3. **Separation of Concerns**: Clean boundaries between ingest and query paths
4. **Horizontal Scalability**: Stateless services that scale independently
5. **Observability Built-in**: Metrics and tracing from day one

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                                   │
│                         (React + WebSocket)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌──────────────┐                ┌──────────────┐
            │ AUDIO STREAM │                │  TEXT QUERY  │
            │  (mic input) │                │   (manual)   │
            └──────────────┘                └──────────────┘
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────┐        │
│           STT SERVICE                   │        │
│  ┌─────────────────────────────────┐    │        │
│  │  Deepgram Streaming API         │    │        │
│  │  - nova-2 model                 │    │        │
│  │  - WebSocket connection         │    │        │
│  │  - Real-time word timestamps    │    │        │
│  └─────────────────────────────────┘    │        │
│              ~150ms                      │        │
└─────────────────────────────────────────┘        │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────┐        │
│           TRANSCRIPT BUFFER             │        │
│  - Accumulates until pause detected     │        │
│  - Sentence boundary detection          │        │
│  - Speaker diarization (optional)       │        │
└─────────────────────────────────────────┘        │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MEMORY PIPELINE                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │   CHUNKER   │───▶│  EMBEDDER   │───▶│  VECTOR DB  │                     │
│  │             │    │             │    │             │                     │
│  │ - sliding   │    │ - OpenAI    │    │ - Chroma    │                     │
│  │   window    │    │   text-emb  │    │ - HNSW idx  │                     │
│  │ - 150 words │    │   -3-small  │    │ - cosine    │                     │
│  │ - 30 overlap│    │ - batch API │    │   distance  │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│       ~5ms              ~80ms               ~20ms                           │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                               │
                    │         QUERY PATH            │
                    │◀──────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RETRIEVAL + GENERATION                              │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ QUERY EMBED │───▶│   VECTOR    │───▶│   CONTEXT   │───▶│     LLM     │  │
│  │             │    │   SEARCH    │    │  ASSEMBLY   │    │  STREAMING  │  │
│  │ - same model│    │             │    │             │    │             │  │
│  │   as ingest │    │ - top-k=5   │    │ - rerank    │    │ - Claude    │  │
│  │             │    │ - threshold │    │ - format    │    │ - stream    │  │
│  │             │    │   filter    │    │ - prompt    │    │   tokens    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│       ~80ms              ~30ms              ~10ms             ~250ms        │
│                                                           (first token)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RESPONSE STREAM                                  │
│                     (WebSocket → UI, token by token)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Speech-to-Text Service

**Technology**: Deepgram Streaming API (nova-2)

**Responsibilities**:
- Real-time audio transcription via WebSocket
- Word-level timestamps for precise timing
- Interim results for live feedback
- Optional speaker diarization

**Key Design Decisions**:
- Use streaming over batch for lower latency
- Buffer transcripts until pause detection
- Sentence boundary detection for natural chunking

```python
class STTService:
    async def create_streaming_session(
        on_transcript: Callable[[TranscriptSegment], None],
        on_error: Callable[[str], None],
    ) -> StreamingSession
```

### 2. Text Chunker

**Technology**: Custom sliding window implementation

**Responsibilities**:
- Split text into semantic chunks
- Maintain overlap for context continuity
- Respect sentence boundaries when possible

**Configuration**:
- `chunk_size`: 150 words (target)
- `chunk_overlap`: 30 words
- `min_chunk_size`: 20 words

**Strategies**:
1. **Sliding Window**: Default, works for any text
2. **Sentence-based**: Preserves sentence boundaries
3. **Paragraph-based**: Respects document structure

```python
class Chunker:
    def chunk_text(
        text: str,
        metadata: MemoryMetadata | None = None,
    ) -> list[Chunk]
```

### 3. Embedding Service

**Technology**: OpenAI text-embedding-3-small

**Responsibilities**:
- Generate vector embeddings for text
- Batch processing for efficiency
- Caching to reduce API calls

**Key Design Decisions**:
- 1536-dimension vectors for good accuracy/speed balance
- Batch API calls (up to 100 texts)
- LRU cache for repeated queries

```python
class EmbeddingService:
    async def embed_text(text: str) -> list[float]
    async def embed_texts(texts: list[str]) -> list[list[float]]
```

### 4. Vector Database

**Technology**: ChromaDB (dev) / Pinecone (prod)

**Responsibilities**:
- Store and index vector embeddings
- Fast similarity search (HNSW algorithm)
- Metadata filtering

**Key Design Decisions**:
- Cosine similarity for semantic matching
- HNSW index for O(log n) search
- Abstraction layer for swappable backends

```python
class VectorDBService:
    async def add_memories(texts, embeddings, metadatas) -> list[str]
    async def search(embedding, top_k, score_threshold) -> list[RetrievalResult]
```

### 5. Context Assembler

**Responsibilities**:
- Format retrieved memories for LLM
- Deduplicate similar content
- Rerank by relevance
- Respect token limits

**Key Design Decisions**:
- Include metadata for provenance
- Truncate to fit context window
- Order by relevance score

```python
class ContextAssembler:
    def assemble(query: str, memories: list[RetrievalResult]) -> str
```

### 6. LLM Service

**Technology**: Anthropic Claude (claude-3-5-sonnet)

**Responsibilities**:
- Generate responses from context
- Stream tokens for low perceived latency
- Handle conversation context

**Key Design Decisions**:
- Streaming for first-token latency
- System prompt for consistent behavior
- Temperature 0.7 for balanced responses

```python
class LLMService:
    async def generate(query: str, memories: list[RetrievalResult]) -> str
    async def generate_stream(query, memories) -> AsyncGenerator[str, None]
```

---

## Data Flow

### Ingest Pipeline

```
1. Audio Input
   └── WebSocket receives audio chunks
       └── Forward to Deepgram
   
2. Transcription
   └── Deepgram returns transcript segments
       └── Buffer until pause/sentence boundary
       
3. Chunking
   └── Split transcript into chunks
       └── ~150 words with 30-word overlap
       
4. Embedding
   └── Batch embed chunks via OpenAI
       └── Cache results for efficiency
       
5. Storage
   └── Store vectors + metadata in ChromaDB
       └── Index for fast retrieval
```

### Query Pipeline

```
1. Query Input
   └── User submits question
   
2. Embedding
   └── Embed query using same model
       └── Check cache first
       
3. Vector Search
   └── Find top-k similar memories
       └── Filter by score threshold
       
4. Context Assembly
   └── Format memories for LLM
       └── Deduplicate and rerank
       
5. Generation
   └── Stream response from Claude
       └── Include source attribution
```

---

## API Design

### REST Endpoints

```
POST   /api/v1/memory/ingest      # Ingest text/transcript
POST   /api/v1/memory/query       # Query with RAG response
GET    /api/v1/memory/recent      # Get recent memories
DELETE /api/v1/memory/clear       # Clear all memories
POST   /api/v1/memory/search      # Search without LLM
GET    /api/v1/metrics            # Latency & usage stats
GET    /api/v1/health             # Health check
```

### WebSocket Endpoints

```
WS /ws/audio    # Stream audio → transcript → memory
WS /ws/query    # Stream query → response
```

### Request/Response Models

```python
# Ingest
class IngestRequest:
    text: str
    source: str = "text"
    session_id: str | None
    metadata: dict[str, Any]

class IngestResponse:
    success: bool
    memory_ids: list[str]
    chunks_created: int
    processing_time_ms: float

# Query
class QueryRequest:
    query: str
    top_k: int = 5
    score_threshold: float = 0.7
    include_context: bool = True
    stream: bool = False

class QueryResponse:
    answer: str
    memories: list[MemoryResponse]
    processing_time_ms: float
```

---

## Latency Optimization

### Target Budgets

| Pipeline | Target | Breakdown |
|----------|--------|-----------|
| Ingest | ~200ms | STT streaming (150ms) + Chunk (5ms) + Embed (80ms) + Store (20ms) |
| Query | ~400ms | Embed (80ms) + Search (30ms) + Context (10ms) + LLM first token (250ms) |

### Optimization Strategies

1. **Streaming Everywhere**
   - Audio → STT via WebSocket
   - LLM → Client via token streaming
   - Reduces perceived latency by 60%

2. **Caching**
   - Embedding cache for repeated queries
   - LRU eviction policy
   - 10,000 entry limit

3. **Batching**
   - Batch embedding requests (up to 100)
   - Reduces per-item overhead

4. **Async I/O**
   - All external calls are async
   - Concurrent execution where possible

5. **Connection Pooling**
   - Persistent HTTP connections
   - WebSocket keep-alive

### Monitoring

```python
@latency_tracked("embedding")
async def embed_text(text: str) -> list[float]:
    ...

# Metrics available at /api/v1/metrics
{
    "latencies": [
        {"operation": "embedding", "p50_ms": 75, "p95_ms": 120, "p99_ms": 180},
        {"operation": "vectordb_query", "p50_ms": 25, "p95_ms": 45, "p99_ms": 80},
        ...
    ]
}
```

---

## Scalability Considerations

### Horizontal Scaling

- **API Servers**: Stateless, scale with load balancer
- **Vector DB**: 
  - ChromaDB: Single-node, suitable for <1M vectors
  - Pinecone: Managed, scales to billions
- **LLM Calls**: Rate-limited by provider, queue if needed

### Vertical Scaling

- Increase API server resources for concurrent requests
- GPU acceleration for local embedding (future)

### Caching Layers

```
Request → API Cache → Embedding Cache → Vector DB
            │               │               │
         (future)      (in-memory)     (persistent)
```

### Production Recommendations

| Component | Development | Production |
|-----------|-------------|------------|
| Vector DB | ChromaDB (local) | Pinecone (managed) |
| API Replicas | 1 | 2-10 (auto-scale) |
| LLM | Claude Sonnet | Claude Sonnet |
| Caching | In-memory | Redis (future) |

---

## Security

### API Key Management

- Stored in environment variables
- Never logged or exposed in responses
- Secrets Manager for production (AWS/GCP)

### Data Privacy

- All data in transit encrypted (TLS)
- Vector DB encrypted at rest (Pinecone)
- No PII in logs

### Rate Limiting

- Implement at API gateway level
- Per-user quotas for production

### Authentication (Future)

- API key authentication for server-to-server
- JWT tokens for user sessions
- OAuth2 for third-party integrations

---

## Observability

### Logging

- Structured JSON logging (structlog)
- Correlation IDs for request tracing
- Log levels: DEBUG, INFO, WARNING, ERROR

### Metrics

- Prometheus format at `/metrics`
- Key metrics:
  - Request latency histograms
  - Memory count gauge
  - Query/ingest counters
  - Error rates

### Tracing

- OpenTelemetry instrumentation
- Distributed tracing across services
- Span attributes for debugging

### Dashboards

- Grafana templates provided
- Key panels:
  - P50/P95/P99 latencies
  - Request rates
  - Error rates
  - Memory growth

---

## Future Enhancements

1. **Multi-Modal Memory**
   - Image embedding and retrieval
   - Audio snippet storage

2. **Advanced Retrieval**
   - Hybrid search (keyword + semantic)
   - Cross-encoder reranking
   - Query expansion

3. **Conversation Memory**
   - Multi-turn context tracking
   - Entity extraction and linking

4. **Edge Deployment**
   - Local embedding models
   - Reduced cloud dependency

5. **Analytics**
   - Query patterns analysis
   - Memory usage insights
   - Recommendation engine

---

*Document Version: 1.0*
*Last Updated: January 2025*
