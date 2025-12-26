# Changelog

All notable changes to BenchAI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [3.0.0] - 2025-12-26

### Added
- **Monitoring Dashboard** - Real-time web UI at `/dashboard` with auto-refresh
  - GPU status (memory, utilization, temperature)
  - Model status and health
  - Cache performance metrics
  - Memory and RAG statistics
  - System uptime tracking

- **Request Caching** - TTL-based cache for LLM responses
  - 160x faster response on cache hits
  - 300-second TTL, configurable max size
  - Cache stats endpoint at `/v1/cache/stats`

- **Q5_K_M Model Upgrades** - Higher quality quantization
  - Qwen2.5 7B upgraded to Q5_K_M
  - DeepSeek Coder 6.7B upgraded to Q5_K_M
  - Better output quality with minimal speed impact

- **ChromaDB Refresh Mechanism** - Prevents stale RAG data
  - 5-minute refresh interval
  - Automatic collection reload

- **Metrics API** - Comprehensive system metrics at `/v1/metrics`
  - JSON format for monitoring integration
  - GPU, models, memory, RAG, cache stats

- **LoRA Fine-Tuning Support** - Documentation and infrastructure
  - Step-by-step guide for custom fine-tuning
  - Directory structure for adapters and training data

### Changed
- Improved model startup reliability
- Enhanced error handling in tool execution
- Better session management with cleanup

### Fixed
- Flash attention flag syntax (`--flash-attn on`)
- Model crash recovery with cooldown period
- Memory leak in session cleanup

---

## [2.0.0] - 2025-12-25

### Added
- **Agentic Planner** - Multi-step task orchestration
  - Automatic task decomposition
  - Parallel tool execution
  - Complexity-based routing

- **88+ Tools** - Comprehensive tool library
  - Shell commands, Git, GitHub
  - Docker operations
  - File operations
  - Web search (SearXNG)
  - Vision (image analysis)
  - Code execution

- **Persistent Memory** - SQLite with FTS5
  - Full-text search
  - Categorized memories
  - Importance scoring

- **RAG Pipeline** - ChromaDB integration
  - Codebase indexing
  - Semantic search
  - HNSW optimized

- **Streaming Support** - SSE responses
  - Real-time token streaming
  - OpenAI-compatible format

- **Multi-Model Routing** - Automatic model selection
  - Task-based routing
  - Manual model override

### Changed
- Migrated from single model to multi-model architecture
- Upgraded to FastAPI for better async support

---

## [1.0.0] - 2025-12-24

### Added
- Initial release
- Basic LLM router with llama.cpp backend
- Single model support (Phi-3 Mini)
- OpenAI-compatible API
- Health check endpoint
- Systemd service integration

---

## Roadmap

### Planned for v3.1
- [ ] Streaming response caching
- [ ] Model hot-swap without restart
- [ ] Prometheus metrics export
- [ ] WebSocket support

### Planned for v4.0
- [ ] Multi-GPU support
- [ ] Distributed model serving
- [ ] Fine-tuning UI
- [ ] Plugin system

---

*Maintained by the BenchAI team*
