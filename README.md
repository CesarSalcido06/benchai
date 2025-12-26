# BenchAI

**Self-hosted AI orchestration platform for software engineering**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)](CHANGELOG.md)

BenchAI is a local LLM router that orchestrates multiple AI models for software engineering tasks. Run entirely on your hardware with zero API costs, full privacy, and complete customization.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Routing** | Automatic selection between 4 specialized models |
| **Agentic Planner** | Multi-step task orchestration with parallel tool execution |
| **88+ Tools** | Shell, Git, GitHub, Docker, file ops, web search, and more |
| **Persistent Memory** | SQLite with FTS5 full-text search |
| **RAG Pipeline** | ChromaDB vector database for codebase indexing |
| **Request Caching** | 100x+ faster responses on repeated queries |
| **Streaming** | Real-time SSE responses |
| **Monitoring Dashboard** | Web UI with real-time metrics |
| **OpenAI Compatible** | Drop-in replacement for OpenAI API |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/benchai.git
cd benchai

# Run installer
./scripts/install.sh

# Start service
sudo systemctl start benchai

# Verify
curl http://localhost:8085/health

# Open dashboard
xdg-open http://localhost:8085/dashboard
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Clients                              │
│   CLI  │  VS Code  │  Neovim  │  WebUI  │  Any HTTP     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                BenchAI Router (:8085)                    │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌─────────────┐  │
│  │ Planner  │ │  Memory  │ │   RAG   │ │    Cache    │  │
│  └──────────┘ └──────────┘ └─────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────┘
         │              │           │            │
    ┌────▼────┐    ┌────▼───┐  ┌────▼───┐   ┌────▼────┐
    │  LLMs   │    │ SQLite │  │ChromaDB│   │  Tools  │
    │ (4 models)│   │ +FTS5  │  │ (HNSW) │   │  (88+)  │
    └─────────┘    └────────┘  └────────┘   └─────────┘
```

---

## Models

| Model | Purpose | Size | Mode |
|-------|---------|------|------|
| **Phi-3 Mini** | Fast general queries | 2.4GB | CPU |
| **Qwen2.5 7B** | Planning, research, analysis | 5.1GB | CPU |
| **DeepSeek Coder 6.7B** | Code generation, debugging | 4.5GB | GPU |
| **Qwen2-VL 7B** | Vision, OCR, image analysis | 4.5GB | On-demand |

Use `model: "auto"` for automatic routing based on query type.

---

## Hardware Requirements

| Tier | GPU | RAM | Performance |
|------|-----|-----|-------------|
| Minimum | 8GB VRAM | 16GB | Basic functionality |
| Recommended | 12GB VRAM | 32GB | Full features |
| Optimal | 24GB+ VRAM | 64GB | Maximum performance |

---

## API Examples

### Chat Completion

```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain async/await"}],
    "stream": false
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "code",
    "messages": [{"role": "user", "content": "Write a sorting function"}],
    "stream": true
  }'
```

### Memory

```bash
# Add memory
curl -X POST http://localhost:8085/v1/memory/add \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers Python", "category": "preference"}'

# Search
curl "http://localhost:8085/v1/memory/search?q=Python"
```

### RAG Search

```bash
curl "http://localhost:8085/v1/rag/search?q=authentication&limit=5"
```

### Metrics

```bash
curl http://localhost:8085/v1/metrics | jq
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/INSTALLATION.md) | Setup and configuration |
| [User Guide](docs/USER-GUIDE.md) | Features and usage |
| [API Reference](docs/API.md) | Endpoints and parameters |
| [Architecture](docs/ARCHITECTURE.md) | System design |
| [Performance](docs/PERFORMANCE.md) | Benchmarks and tuning |
| [Tools Reference](docs/TOOLS.md) | All 88+ tools |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues |
| [Research](docs/RESEARCH.md) | LLM optimization techniques |

---

## Monitoring

Access the dashboard at `http://localhost:8085/dashboard`

![Dashboard Preview](docs/images/dashboard-preview.png)

Features:
- GPU memory and temperature
- Model status (running/stopped)
- Cache hit rate
- Memory entries
- RAG document count
- System uptime

---

## Client Integrations

### CLI Tool
```bash
cd benchai-client
./install.sh
benchai "explain this code"
```

### VS Code (Continue.dev)
See [Continue.dev setup](docs/INSTALLATION.md#vs-code)

### Neovim (Avante.nvim)
See [Avante.nvim setup](docs/INSTALLATION.md#neovim)

---

## Performance

| Metric | Value |
|--------|-------|
| Cache hit latency | ~30ms |
| Code model (GPU) | 30-40 tok/s |
| General model (CPU) | 12-15 tok/s |
| Cache speedup | 100-300x |

See [Performance Guide](docs/PERFORMANCE.md) for optimization details.

---

## Service Management

```bash
# Start
sudo systemctl start benchai

# Stop
sudo systemctl stop benchai

# Restart
sudo systemctl restart benchai

# Logs
sudo journalctl -u benchai -f

# Status
sudo systemctl status benchai
```

---

## Development

### Prerequisites
- Python 3.10+
- llama.cpp (with CUDA support)
- ChromaDB
- SQLite with FTS5

### Local Development
```bash
cd router
python3 llm_router.py
```

### Testing
```bash
# Health check
curl http://localhost:8085/health

# Chat test
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general","messages":[{"role":"user","content":"ping"}],"max_tokens":10}'
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference engine
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [HuggingFace](https://huggingface.co/) - Model hosting
