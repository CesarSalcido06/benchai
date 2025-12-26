# Performance Guide

Benchmarks, optimization techniques, and tuning recommendations for BenchAI.

---

## Hardware Requirements

### Minimum Configuration
| Component | Specification |
|-----------|---------------|
| CPU | 4+ cores (Intel i5/AMD Ryzen 5) |
| RAM | 16GB DDR4 |
| GPU | 8GB VRAM (RTX 3060 or equivalent) |
| Storage | 50GB SSD |

### Recommended Configuration
| Component | Specification |
|-----------|---------------|
| CPU | 8+ cores (Intel i7/AMD Ryzen 7) |
| RAM | 32GB+ DDR4 |
| GPU | 12GB+ VRAM (RTX 3060 12GB, RTX 4070) |
| Storage | 100GB+ NVMe SSD |

### Optimal Configuration
| Component | Specification |
|-----------|---------------|
| CPU | 12+ cores (Intel i9/AMD Ryzen 9) |
| RAM | 64GB+ DDR5 |
| GPU | 24GB+ VRAM (RTX 4090) |
| Storage | 500GB+ NVMe SSD |

---

## Benchmarks

### Response Time (RTX 3060 12GB)

| Model | Mode | First Token | Full Response (100 tokens) |
|-------|------|-------------|---------------------------|
| Phi-3 Mini (Q4_K_M) | CPU | ~2s | ~8s |
| Qwen2.5 7B (Q5_K_M) | CPU | ~4s | ~45s |
| DeepSeek Coder (Q5_K_M) | GPU | ~0.5s | ~5s |
| Cached Response | - | ~0.03s | ~0.03s |

### Tokens Per Second

| Model | GPU Mode | CPU Mode |
|-------|----------|----------|
| Phi-3 Mini 4K | N/A | 12-15 tok/s |
| Qwen2.5 7B | 25-35 tok/s | 2-4 tok/s |
| DeepSeek Coder 6.7B | 30-40 tok/s | 3-5 tok/s |

### Memory Usage

| Model | VRAM (GPU) | RAM (CPU) |
|-------|------------|-----------|
| Phi-3 Mini Q4_K_M | 2.5GB | 4GB |
| Qwen2.5 7B Q5_K_M | 5.5GB | 7GB |
| DeepSeek Coder Q5_K_M | 5GB | 6.5GB |

### Cache Performance

| Metric | Value |
|--------|-------|
| Cache Hit Latency | 25-40ms |
| Cache Miss Latency | 4-10s |
| Speedup Factor | 100-300x |
| Typical Hit Rate | 40-60% |

---

## Optimization Techniques

### 1. Quantization Selection

| Use Case | Recommended | Notes |
|----------|-------------|-------|
| VRAM-constrained | Q4_K_M | 3.3x smaller, minimal quality loss |
| Balanced | Q5_K_M | Sweet spot for quality/size |
| Quality-critical | Q6_K | Near-original quality |
| Small models (<7B) | Q8_0 | Negligible quality loss |

### 2. GPU Layer Offloading

```bash
# Full GPU (if model fits)
-ngl 99

# Partial offload (balance speed/memory)
-ngl 25

# CPU only
-ngl 0
```

**Performance Impact:**
- Full GPU: ~40 tok/s
- 25 layers GPU: ~8.6 tok/s
- CPU only: ~2-5 tok/s

### 3. Context Window Tuning

| Context Size | VRAM Impact | Use Case |
|--------------|-------------|----------|
| 2048 | Low | Simple queries |
| 4096 | Medium | General use |
| 8192 | High | Code analysis |
| 16384 | Very High | Large documents |

**VRAM Formula:**
```
KV Cache = 2 * layers * context * (head_dim * num_kv_heads) * bytes_per_param
```

### 4. Batch Size Optimization

```bash
# Recommended settings
-b 512    # Logical batch size
-ub 256   # Physical batch size
```

Larger batches improve throughput but increase latency.

### 5. Flash Attention

```bash
--flash-attn on
```

Benefits:
- 2-3x faster attention computation
- Reduced memory usage
- Requires compatible hardware

### 6. Threading

| Mode | Recommended Threads |
|------|---------------------|
| GPU | 8 (less work for CPU) |
| CPU | 12+ (full utilization) |

```bash
# GPU mode
-t 8

# CPU mode
-t 12
```

---

## Memory Management

### VRAM Budget (12GB Example)

```
Total VRAM:           12,288 MB
├── DeepSeek Coder:    8,500 MB (GPU model)
├── KV Cache:          2,500 MB
├── CUDA overhead:       500 MB
└── Available:           788 MB
```

### RAM Budget (32GB Example)

```
Total RAM:            32,768 MB
├── System:            4,000 MB
├── Phi-3 Mini:        4,000 MB
├── Qwen2.5 7B:        7,000 MB
├── SQLite/ChromaDB:   1,000 MB
├── Python/Router:       500 MB
└── Available:        16,268 MB
```

---

## Caching Strategy

### Request Cache

```python
# Configuration
TTL_SECONDS = 300      # 5 minute expiry
MAX_SIZE = 100         # Maximum entries
```

**Best Practices:**
1. Use consistent message formatting
2. Avoid timestamp-based queries
3. Monitor hit rate via `/v1/cache/stats`

### KV Cache Slots

```bash
# Enable slot saving
--slot-save-path /path/to/cache/
```

Benefits:
- Faster context restoration
- Reduced memory pressure
- Better multi-user performance

---

## Monitoring

### Dashboard

Access at: `http://localhost:8085/dashboard`

Displays:
- GPU memory and utilization
- Model status (running/stopped)
- Cache hit rate
- Memory entries
- RAG documents

### Metrics API

```bash
curl http://localhost:8085/v1/metrics | jq
```

Returns JSON with:
- `gpu`: Memory, utilization, temperature
- `models`: Status, PIDs, configuration
- `cache`: Hits, misses, hit rate
- `memory`: Total entries, FTS5 status
- `rag`: Document count, status

### Health Checks

```bash
# Router health
curl http://localhost:8085/health

# Model health (direct)
curl http://localhost:8091/health  # Phi-3
curl http://localhost:8092/health  # Qwen2.5
curl http://localhost:8093/health  # DeepSeek
```

---

## Troubleshooting Performance

### Slow Responses

1. **Check GPU utilization:**
   ```bash
   nvidia-smi
   ```
   - If 0%, model may be on CPU

2. **Verify model mode:**
   ```bash
   ps aux | grep llama-server | grep ngl
   ```
   - Look for `-ngl 35` (GPU) vs `-ngl 0` (CPU)

3. **Check cache hit rate:**
   ```bash
   curl http://localhost:8085/v1/cache/stats
   ```
   - Low hit rate = more cold queries

### Out of Memory

1. **Reduce context window:**
   ```bash
   -c 4096  # Instead of 8192
   ```

2. **Use smaller quantization:**
   ```bash
   # Q4_K_M instead of Q5_K_M
   ```

3. **Offload fewer layers:**
   ```bash
   -ngl 25  # Instead of 35
   ```

### High Latency

1. **Enable flash attention:**
   ```bash
   --flash-attn on
   ```

2. **Increase thread count (CPU):**
   ```bash
   -t 16
   ```

3. **Enable continuous batching:**
   ```bash
   --cont-batching
   ```

---

## Recommended Configurations

### 8GB VRAM (RTX 3060 8GB)

```bash
# Single GPU model
llama-server \
  -m deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
  -ngl 30 \
  -c 4096 \
  -t 8 \
  --flash-attn on
```

### 12GB VRAM (RTX 3060 12GB)

```bash
# Primary GPU model
llama-server \
  -m deepseek-coder-6.7b-instruct.Q5_K_M.gguf \
  -ngl 35 \
  -c 8192 \
  -t 8 \
  --flash-attn on \
  --cont-batching
```

### 24GB VRAM (RTX 4090)

```bash
# Multiple GPU models possible
llama-server \
  -m qwen2.5-14b-instruct.Q5_K_M.gguf \
  -ngl 99 \
  -c 16384 \
  -t 8 \
  --flash-attn on \
  --cont-batching
```

---

*Performance guide version 1.0*
