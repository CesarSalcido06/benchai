# BenchAI - Local AI Engineering Assistant

A full-featured AI orchestration platform running locally on your hardware. Features agentic planning, 88+ tools, persistent memory, RAG, TTS, streaming responses, and IDE integrations.

## Features

- **Agentic Planner** - Multi-step task orchestration with parallel tool execution
- **88+ Tools** - Docker, Git, GitHub, file ops, web search, code execution, learning tools
- **Persistent Memory** - SQLite with FTS5 full-text search
- **RAG Pipeline** - ChromaDB vector database for codebase search
- **Text-to-Speech** - Piper TTS with high-quality voices
- **Streaming** - Real-time SSE responses
- **Multi-Model** - Automatic model routing (general, code, research, vision)
- **IDE Integration** - VS Code (Continue.dev), Neovim (Avante.nvim)
- **OpenAI Compatible** - Works with any OpenAI-compatible client

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Open WebUI (:3000)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   BenchAI Router (:8085)                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Planner │  │ Memory  │  │   RAG   │  │   TTS   │        │
│  │ (Qwen)  │  │(SQLite) │  │(Chroma) │  │ (Piper) │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│  ┌────▼────────────▼────────────▼────────────▼────┐        │
│  │              Tool Executor (Parallel)           │        │
│  └─────────────────────┬──────────────────────────┘        │
└────────────────────────┼────────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───▼───┐  ┌────────┐  ┌─▼──────┐  ┌─────────▼─────────┐
│ LLMs  │  │SearXNG │  │Docker  │  │ Local Tools       │
│:8091-4│  │(:8081) │  │  API   │  │ code,file,shell   │
└───────┘  └────────┘  └────────┘  └───────────────────┘
```

## Hardware Requirements

- **Minimum:** 16GB RAM, 8GB VRAM GPU (RTX 3060 or equivalent)
- **Recommended:** 32GB RAM, 12GB+ VRAM GPU
- **Storage:** 50GB+ for models

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/benchai.git
cd benchai

# Run the installer
./scripts/install.sh

# Start the service
sudo systemctl start benchai

# Check health
curl http://localhost:8085/health
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Full setup instructions |
| [User Guide](docs/USER-GUIDE.md) | How to use BenchAI |
| [API Reference](docs/API.md) | API endpoints and examples |
| [Tools Reference](docs/TOOLS.md) | All 88+ available tools |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and fixes |
| [Architecture](docs/ARCHITECTURE.md) | System design details |

## Models

| Model | Role | Size | Port |
|-------|------|------|------|
| Phi-3 Mini | Fast/General | 2.4GB | 8091 |
| Qwen2.5 7B | Planner/Research | 4.4GB | 8092 |
| DeepSeek Coder 6.7B | Code/Math | 4.0GB | 8093 |
| Qwen2-VL 7B | Vision/OCR | 4.5GB | 8094 |

## Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| BenchAI Router | 8085 | Main API endpoint |
| Open WebUI | 3000 | Chat interface |
| SearXNG | 8081 | Web search |
| Jellyfin | 8096 | Media server |
| LLM Servers | 8091-8094 | Model inference |

## Client Tools

See [benchai-client](https://github.com/YOUR_USERNAME/benchai-client) for:
- CLI tool (`benchai`)
- VS Code integration
- Neovim integration

## API Examples

```bash
# Health check
curl http://localhost:8085/health

# Chat
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Memory
curl -X POST http://localhost:8085/v1/memory/add \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "category": "preference"}'

# RAG search
curl "http://localhost:8085/v1/rag/search?q=authentication"

# TTS
curl -X POST http://localhost:8085/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' -o speech.wav
```

## License

MIT License - See [LICENSE](LICENSE)

## Contributing

Contributions welcome! Please read the docs first and open an issue before PRs.
