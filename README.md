# BenchAI - Local AI Engineering Assistant

Self-hosted AI orchestration platform with agentic planning, 88+ tools, persistent memory, RAG, and IDE integrations. Run entirely on your own hardware.

## Features

- **Agentic Planner** - Multi-step task orchestration with parallel execution
- **88+ Tools** - Docker, Git, GitHub, file ops, web search, code execution
- **Persistent Memory** - SQLite with FTS5 full-text search
- **RAG Pipeline** - ChromaDB vector database for codebase indexing
- **Streaming** - Real-time Server-Sent Events responses
- **Multi-Model Routing** - Auto-select between general, code, research, and vision models
- **IDE Integration** - VS Code (Continue.dev), Neovim (Avante.nvim), CLI tool
- **OpenAI Compatible** - Standard OpenAI API format

## Architecture

```
┌──────────────────────────────────────────────────────┐
│            Clients (CLI, IDE, WebUI)                 │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│              BenchAI Router (:8085)                   │
│  ┌─────────┐  ┌────────┐  ┌─────┐  ┌──────────┐    │
│  │ Planner │  │ Memory │  │ RAG │  │ Executor │    │
│  └────┬────┘  └───┬────┘  └──┬──┘  └────┬─────┘    │
└───────┼───────────┼──────────┼──────────┼───────────┘
        │           │          │          │
   ┌────▼──────┬────▼───┬──────▼────┬─────▼────┐
   │ LLM Pool  │ SQLite │ ChromaDB  │  Tools   │
   │ (4 models)│  +FTS5 │           │   (88+)  │
   └───────────┴────────┴───────────┴──────────┘
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB+ |
| GPU | 8GB VRAM (RTX 3060) | 12GB+ VRAM |
| Storage | 50GB free | 100GB+ SSD |
| CPU | 4+ cores | 8+ cores |

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/benchai.git
cd benchai

# 2. Run installer
./scripts/install.sh

# 3. Start service
sudo systemctl start benchai

# 4. Verify health
curl http://localhost:8085/health

# 5. Install client tools (optional)
cd ../benchai-client
./install.sh
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Server setup and configuration |
| [User Guide](docs/USER-GUIDE.md) | Using BenchAI features |
| [API Reference](docs/API.md) | Endpoints and examples |
| [Tools Reference](docs/TOOLS.md) | All 88+ available tools |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and fixes |

## Models

BenchAI runs 4 local models with automatic routing:

| Model | Use Case | Size | Port |
|-------|----------|------|------|
| Phi-3 Mini | General queries, fast responses | 2.4GB | 8091 |
| Qwen2.5 7B | Planning, research, analysis | 4.4GB | 8092 |
| DeepSeek Coder 6.7B | Code generation, debugging | 4.0GB | 8093 |
| Qwen2-VL 7B | Vision, OCR, image analysis | 4.5GB | 8094 |

**Auto-routing** selects the best model based on your query.

## Client Tools

Install the [benchai-client](../benchai-client) for:

- **CLI Tool** - Terminal-based chat with streaming
- **VS Code** - Continue.dev integration
- **Neovim** - Avante.nvim plugin

```bash
cd benchai-client
./install.sh
```

## API Examples

### Health Check
```bash
curl http://localhost:8085/health
```

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

### Streaming Chat
```bash
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "code",
    "messages": [{"role": "user", "content": "Write a sorting function"}],
    "stream": true
  }'
```

### Memory Operations
```bash
# Add memory
curl -X POST http://localhost:8085/v1/memory/add \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers Python", "category": "preference"}'

# Search memory
curl "http://localhost:8085/v1/memory/search?q=Python"

# Get stats
curl http://localhost:8085/v1/memory/stats
```

### RAG Search
```bash
# Search indexed codebase
curl "http://localhost:8085/v1/rag/search?q=authentication&limit=5"
```

See [API Reference](docs/API.md) for complete documentation.

## Service Management

```bash
# Start service
sudo systemctl start benchai

# Stop service
sudo systemctl stop benchai

# View logs
sudo journalctl -u benchai -f

# Restart service
sudo systemctl restart benchai
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

For bugs and feature requests, open an issue with details.
