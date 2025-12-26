# BenchAI Architecture

**Version:** 3.0
**Last Updated:** December 2025

---

## System Overview

BenchAI is a self-hosted AI orchestration platform designed for software engineering tasks. It runs entirely on local hardware, providing privacy, customization, and zero API costs.

```
                    ┌────────────────────────────────────────────────────────────┐
                    │                      CLIENT LAYER                          │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
                    │  │   CLI    │  │  VS Code │  │  Neovim  │  │  WebUI   │   │
                    │  │ (benchai)│  │(Continue)│  │ (Avante) │  │(Open-WUI)│   │
                    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
                    └───────┼─────────────┼─────────────┼─────────────┼─────────┘
                            │             │             │             │
                            └─────────────┴──────┬──────┴─────────────┘
                                                 │
                    ┌────────────────────────────▼────────────────────────────────┐
                    │                    ROUTER LAYER (:8085)                     │
                    │  ┌─────────────────────────────────────────────────────┐   │
                    │  │                   FastAPI Server                      │   │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
                    │  │  │ OpenAI   │  │ Session  │  │ Request  │           │   │
                    │  │  │ Compat   │  │ Manager  │  │  Cache   │           │   │
                    │  │  └──────────┘  └──────────┘  └──────────┘           │   │
                    │  └─────────────────────────────────────────────────────┘   │
                    │                                                             │
                    │  ┌─────────────────────────────────────────────────────┐   │
                    │  │              ORCHESTRATION LAYER                     │   │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
                    │  │  │ Agentic  │  │  Tool    │  │  Model   │           │   │
                    │  │  │ Planner  │  │ Executor │  │ Selector │           │   │
                    │  │  └──────────┘  └──────────┘  └──────────┘           │   │
                    │  └─────────────────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────────────────┘
                                                 │
          ┌──────────────────┬──────────────────┬┴────────────────┬───────────────┐
          │                  │                  │                 │               │
┌─────────▼────────┐ ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼────┐ ┌────────▼────┐
│   MODEL LAYER    │ │ MEMORY LAYER  │ │   RAG LAYER   │ │ TOOL LAYER │ │  EXTERNAL   │
│  ┌────────────┐  │ │               │ │               │ │            │ │             │
│  │ Phi-3 Mini │  │ │   SQLite DB   │ │   ChromaDB    │ │  88+ Tools │ │  - Obsidian │
│  │  :8091     │  │ │   + FTS5      │ │   (HNSW)      │ │            │ │  - GitHub   │
│  ├────────────┤  │ │               │ │               │ │  - Shell   │ │  - Docker   │
│  │ Qwen2.5 7B │  │ │  Categories:  │ │  Documents:   │ │  - Git     │ │  - SearXNG  │
│  │  :8092     │  │ │  - Facts      │ │  - Code files │ │  - Files   │ │             │
│  ├────────────┤  │ │  - Prefs      │ │  - Docs       │ │  - Web     │ │             │
│  │ DeepSeek   │  │ │  - Context    │ │  - Notes      │ │  - Vision  │ │             │
│  │ Coder :8093│  │ │               │ │               │ │  - ...     │ │             │
│  ├────────────┤  │ └───────────────┘ └───────────────┘ └────────────┘ └─────────────┘
│  │ Qwen2-VL   │  │
│  │  :8094     │  │
│  └────────────┘  │
│                  │
│  llama.cpp       │
│  (llama-server)  │
└──────────────────┘
```

---

## Core Components

### 1. Router (FastAPI Server)

**Location:** `router/llm_router.py`
**Port:** 8085

The router is the central orchestration layer that:
- Provides OpenAI-compatible API endpoints
- Routes requests to appropriate models
- Manages sessions and conversation context
- Implements request caching for performance
- Coordinates tool execution

**Key Features:**
- CORS-enabled for browser clients
- Streaming SSE support
- Health monitoring with auto-restart
- Graceful shutdown handling

### 2. Model Manager

**Responsibilities:**
- Spawns and manages llama-server processes
- Monitors model health (60-second intervals)
- Handles VRAM allocation (GPU vs CPU mode)
- Implements model hot-swapping
- Automatic restart on crashes

**Model Configuration:**
```python
MODELS = {
    "general": {
        "file": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "port": 8091,
        "mode": "cpu",
        "context": 4096,
        "threads": 12
    },
    "code": {
        "file": "deepseek-coder-6.7b-instruct.Q5_K_M.gguf",
        "port": 8093,
        "mode": "gpu",
        "context": 8192,
        "gpu_layers": 35
    }
    # ... etc
}
```

### 3. Agentic Planner

The planner implements a multi-step reasoning workflow:

```
User Request
     │
     ▼
┌─────────────┐
│   Analyze   │  ← Determine complexity and required tools
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Plan      │  ← Break into subtasks, identify dependencies
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Execute    │  ← Run tools in parallel where possible
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Synthesize │  ← Combine results into response
└─────────────┘
```

**Parallel Execution:**
- Independent tool calls run concurrently
- Dependency graph ensures correct ordering
- Timeout handling for hung operations

### 4. Memory System

**Technology:** SQLite with FTS5 (Full-Text Search)
**Location:** `~/llm-storage/memory/benchai_memory.db`

**Schema:**
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id'
);
```

**Categories:**
- `user_stated` - Explicit user facts
- `preference` - User preferences
- `code` - Code-related context
- `analysis` - Analysis results
- `general` - Miscellaneous

### 5. RAG Pipeline

**Technology:** ChromaDB with HNSW indexing
**Location:** `~/llm-storage/rag/chroma_db`

**Workflow:**
1. **Indexing:** Code files are chunked and embedded
2. **Storage:** Vectors stored with metadata
3. **Query:** Semantic similarity search
4. **Retrieval:** Top-k relevant chunks returned

**Configuration:**
- Embedding: sentence-transformers (local)
- Index: HNSW (Hierarchical Navigable Small World)
- Distance: Cosine similarity
- Chunk size: 512 tokens with 50-token overlap

### 6. Request Cache

**Purpose:** Reduce redundant LLM calls for identical requests

**Implementation:**
```python
class RequestCache:
    TTL = 300  # 5 minutes
    MAX_SIZE = 100  # Maximum cached responses

    # Key: SHA256(model + messages + max_tokens)
    # Value: (response, timestamp)
```

**Performance:**
- Cache hit: ~0.03s
- Cache miss: ~4-10s (depending on model)
- Hit rate: 40-60% typical usage

---

## Data Flow

### Chat Request Flow

```
1. Client POST /v1/chat/completions
                │
2. Cache Check ─┼─▶ HIT ──▶ Return cached response
                │
3. Session Lookup/Create
                │
4. Model Selection (auto-routing or manual)
                │
5. Tool Detection (if needed)
                │
6. LLM Inference ──▶ llama-server
                │
7. Tool Execution (if applicable)
                │
8. Response Assembly
                │
9. Cache Store (non-streaming)
                │
10. Return to Client
```

### Memory Search Flow

```
1. Query text received
        │
2. FTS5 full-text search
        │
3. Rank by relevance + recency + importance
        │
4. Return top matches with context
```

---

## Deployment Architecture

### Systemd Service

```ini
[Unit]
Description=BenchAI LLM Router
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/benchai/router
ExecStart=/usr/bin/python3 llm_router.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Port Allocation

| Port | Service | Description |
|------|---------|-------------|
| 8085 | Router | Main API endpoint |
| 8091 | Phi-3 Mini | General model |
| 8092 | Qwen2.5 7B | Planner/Research |
| 8093 | DeepSeek Coder | Code model |
| 8094 | Qwen2-VL | Vision model |

### Storage Layout

```
~/llm-storage/
├── memory/
│   └── benchai_memory.db    # SQLite database
├── rag/
│   └── chroma_db/           # ChromaDB vectors
├── cache/
│   └── audio/               # TTS cache
├── lora/                    # LoRA adapters
├── training/                # Training data
└── checkpoints/             # Training checkpoints

~/llama.cpp/
├── models/                  # GGUF model files
├── build/bin/               # Compiled binaries
└── cache/                   # KV cache slots
```

---

## Performance Considerations

### GPU Memory Management

**12GB VRAM Budget:**
```
DeepSeek Coder (GPU): ~8.5GB
Remaining: ~3.5GB for KV cache and overhead
```

**Strategy:**
- Code model on GPU (performance critical)
- General/Planner on CPU (acceptable latency)
- Vision model loaded on-demand

### Optimization Flags

```bash
# GPU model
llama-server \
  -ngl 35           # Full GPU offload
  -c 8192           # Large context
  -t 8              # Reduced threads (GPU does work)
  --flash-attn on   # Flash attention
  --cont-batching   # Continuous batching
  -b 512 -ub 256    # Batch sizes
  --mlock           # Lock in RAM

# CPU model
llama-server \
  -ngl 0            # No GPU
  -c 4096           # Smaller context
  -t 12             # More threads
  --cont-batching
  --mlock
```

### Scaling Limits

| Metric | Current System | Bottleneck |
|--------|----------------|------------|
| Concurrent requests | 4-6 | RAM/VRAM |
| Context window | 8K (GPU), 4K (CPU) | VRAM |
| Models loaded | 3-4 | VRAM |
| RAG documents | ~500 | RAM |
| Memory entries | Unlimited | Disk |

---

## Security Model

### Network Exposure
- Router binds to `0.0.0.0:8085` (configurable)
- Model servers bind to `127.0.0.1` (local only)
- No authentication by default (add reverse proxy for production)

### Data Privacy
- All processing local (no external API calls by default)
- Memory stored in local SQLite
- RAG vectors stored locally
- No telemetry or analytics

### Recommended Production Setup
```
                Internet
                    │
            ┌───────▼───────┐
            │   Firewall    │
            └───────┬───────┘
                    │
            ┌───────▼───────┐
            │ Reverse Proxy │  ← nginx/caddy with TLS
            │ + Auth        │
            └───────┬───────┘
                    │
            ┌───────▼───────┐
            │    BenchAI    │  ← binds to localhost only
            │    Router     │
            └───────────────┘
```

---

## Extension Points

### Adding New Models

1. Download GGUF file to `~/llama.cpp/models/`
2. Add configuration to `MODELS` dict in `llm_router.py`
3. Update model selection logic if needed
4. Restart service

### Adding New Tools

1. Define tool function with async signature
2. Add to `TOOLS` dictionary with schema
3. Implement execution logic
4. Tool automatically available to planner

### Custom LoRA Adapters

1. Fine-tune using `llama-finetune`
2. Export adapter to GGUF
3. Add `--lora` flag to model config
4. Restart model

---

*Architecture document version 1.0*
