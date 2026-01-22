# Mira Memory Engine

<div align="center">

![Mira Logo](frontend/public/mira-icon.svg)

**Real-time Voice-to-Memory RAG System**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev)

</div>

---

## Overview

Mira Memory Engine is a high-performance, real-time voice-to-memory system that combines streaming speech-to-text, intelligent chunking, semantic embedding, and retrieval-augmented generation (RAG) to create a seamless conversational memory experience.

### Key Features

- **🎙️ Real-time Voice Streaming**: Live transcription via Deepgram with <150ms latency
- **🧠 Intelligent Memory Storage**: Automatic chunking and semantic embedding
- **🔍 Semantic Search**: Vector-based retrieval using ChromaDB/Pinecone
- **💬 RAG-Powered Responses**: Context-aware answers using Claude
- **⚡ Low Latency**: <400ms query response time (first token)
- **📊 Performance Monitoring**: Built-in metrics and observability

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                          │
│                      (React + WebSocket)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     ┌──────────────┐                ┌──────────────┐
     │ AUDIO STREAM │                │  TEXT QUERY  │
     └──────┬───────┘                └──────┬───────┘
            │                               │
            ▼                               │
┌───────────────────────┐                   │
│    STT SERVICE        │                   │
│  (Deepgram nova-2)    │                   │
│     ~150ms            │                   │
└───────────┬───────────┘                   │
            │                               │
            ▼                               │
┌───────────────────────────────────────────────────────────────┐
│                    MEMORY PIPELINE                            │
│  Chunker → Embedder (OpenAI) → Vector DB (Chroma)             │
│   ~5ms       ~80ms                ~20ms                       │
└───────────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│               RETRIEVAL + GENERATION                          │
│  Query Embed → Vector Search → Context → LLM (Claude)         │
│     ~80ms         ~30ms        ~10ms      ~250ms              │
└───────────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                          Response Stream
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- API Keys:
  - [Deepgram](https://console.deepgram.com/) (Speech-to-Text)
  - [OpenAI](https://platform.openai.com/) (Embeddings)
  - [Anthropic](https://console.anthropic.com/) (LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mira-ai/mira-memory-engine.git
   cd mira-memory-engine
   ```

2. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

3. **Install backend dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Run the application**
   ```bash
   # Terminal 1: Backend
   make run
   
   # Terminal 2: Frontend
   make frontend-dev
   ```

6. **Open the app**
   Navigate to [http://localhost:3000](http://localhost:3000)

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/memory/ingest` | Ingest text into memory |
| `POST` | `/api/v1/memory/query` | Query memories with RAG |
| `GET` | `/api/v1/memory/recent` | Get recent memories |
| `DELETE` | `/api/v1/memory/clear` | Clear all memories |
| `GET` | `/api/v1/metrics` | Get performance metrics |
| `GET` | `/api/v1/health` | Health check |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/audio` | Stream audio for transcription |
| `/ws/query` | Stream query responses |

### Example: Ingest Memory

```bash
curl -X POST http://localhost:8000/api/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The meeting discussed Q4 revenue projections.",
    "source": "meeting_notes",
    "session_id": "meeting-2024-01-15"
  }'
```

### Example: Query Memory

```bash
curl -X POST http://localhost:8000/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the Q4 projections?",
    "top_k": 5,
    "include_context": true
  }'
```

## Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_DB_PROVIDER` | Vector store (`chroma` or `pinecone`) | `chroma` |
| `CHUNK_SIZE` | Words per chunk | `150` |
| `CHUNK_OVERLAP` | Overlap between chunks | `30` |
| `RETRIEVAL_TOP_K` | Results to retrieve | `5` |
| `LLM_MODEL` | Claude model version | `claude-sonnet-4-20250514` |

See [env.example](env.example) for all options.

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# Type check
make typecheck

# Run benchmarks
make benchmark

# Seed test data
make seed
```

## Deployment

### Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f infra/k8s/

# Check status
kubectl get pods -l app=mira
```

### AWS (Terraform)

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

## Performance

### Latency Budget

| Pipeline | Target | Components |
|----------|--------|------------|
| **Ingest** | ~200ms | STT (150ms) + Chunk (5ms) + Embed (80ms) + Store (20ms) |
| **Query** | ~400ms | Embed (80ms) + Search (30ms) + Context (10ms) + LLM (250ms) |

### Benchmarks

Run the benchmark suite:

```bash
python scripts/benchmark.py --all
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11+ |
| Frontend | React 18, TypeScript, Tailwind |
| STT | Deepgram (nova-2) |
| Embeddings | OpenAI (text-embedding-3-small) |
| Vector DB | ChromaDB / Pinecone |
| LLM | Anthropic Claude |
| Observability | OpenTelemetry, Prometheus |

## Project Structure

```
mira-memory-engine/
├── app/                    # Backend application
│   ├── api/               # REST & WebSocket endpoints
│   ├── core/              # Memory pipeline
│   ├── services/          # External integrations
│   ├── models/            # Data models
│   └── utils/             # Utilities
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/   # UI components
│   │   ├── hooks/        # React hooks
│   │   └── services/     # API client
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── infra/                 # Infrastructure (Docker, Terraform, K8s)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## Acknowledgments

- [Deepgram](https://deepgram.com/) for real-time speech-to-text
- [OpenAI](https://openai.com/) for embeddings
- [Anthropic](https://anthropic.com/) for Claude LLM
- [ChromaDB](https://trychroma.com/) for vector storage

---
>
