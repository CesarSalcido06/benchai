# Comprehensive Guide: Local LLM Optimization & Router Architecture (2024-2025)

*Research compiled: December 2025*

---

## Table of Contents

1. [llama.cpp Optimization Techniques](#1-llamacpp-optimization-techniques)
2. [Local LLM Router Architecture Best Practices](#2-local-llm-router-architecture-best-practices)
3. [Model Recommendations for Different Tasks](#3-model-recommendations-for-different-tasks)
4. [Performance Benchmarks and Metrics](#4-performance-benchmarks-and-metrics)

---

## 1. llama.cpp Optimization Techniques

### 1.1 Quantization Formats

Quantization is the most important optimization for local LLM deployment. The K-quant family offers the best balance between model size and accuracy.

#### Quantization Format Comparison

| Format | Bits | Size (7B model) | Perplexity Impact | Use Case |
|--------|------|-----------------|-------------------|----------|
| **Q4_K_M** | 4-bit | ~3.8-4.1 GB | +0.0535 ppl | **Recommended** - Best balance for constrained systems |
| **Q5_K_M** | 5-bit | ~4.5-4.8 GB | Minimal | Sweet spot for quality vs. size on desktop |
| **Q6_K** | 6-bit | ~5.5-6.0 GB | Very small | Near-original quality, ideal for 7B-14B models |
| **Q8_0** | 8-bit | ~7.0-7.5 GB | Negligible | Minimal quality loss, good for smaller models |
| **FP16** | 16-bit | ~13.5 GB | Baseline | Full precision, for reference only |

#### Key Recommendations

- **Memory-constrained (8-12GB VRAM):** Start with Q4_K_M - provides 3.3x size reduction with minimal quality loss
- **Desktop (16-24GB VRAM):** Use Q5_K_M for excellent quality-to-size ratio
- **Small models (7B-14B):** Q6_K is recommended as the perplexity change is extremely small
- **RAG applications:** Prefer Q5_K_M or Q8_0 for better context handling

#### Mixed-Precision Details

Q4_K_M and Q5_K_M implement higher precision for critical layers:
- Q4_K_M uses Q6_K for half of `attention.wv` and `feed_forward.w2` tensors
- This mixed precision proves to be a good tradeoff between accuracy and resource usage

#### Using Importance Matrix (imatrix)

For better quantization quality, use calibration data to compute an importance matrix:
```bash
llama-imatrix -m model.gguf -f calibration_data.txt -o imatrix.dat
llama-quantize --imatrix imatrix.dat model.gguf model-Q4_K_M.gguf Q4_K_M
```

**Sources:**
- [Quantize Llama models with GGUF and llama.cpp | Towards Data Science](https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172/)
- [Difference in different quantization methods | GitHub Discussion](https://github.com/ggml-org/llama.cpp/discussions/2094)
- [Practical Quantization Guide for iPhone and Mac](https://enclaveai.app/blog/2025/11/12/practical-quantization-guide-iphone-mac-gguf/)
- [Demystifying LLM Quantization Suffixes | Medium](https://medium.com/@paul.ilvez/demystifying-llm-quantization-suffixes-what-q4-k-m-q8-0-and-q6-k-really-mean-0ec2770f17d3)

---

### 1.2 GPU Offloading Strategies for 12GB VRAM

#### Model Fit Guidelines for 12GB VRAM

| Model Size | Quantization | Context Window | Fits in 12GB? |
|------------|--------------|----------------|---------------|
| 7-8B | Q4_K_M | ~32-45K tokens | Yes |
| 12-14B | Q4_K_M | ~16K tokens | Yes (tight) |
| 14B | Q4_K_M | ~4K tokens | Barely |
| 30-34B | Q4_K_M | Any | No - requires hybrid |

#### GPU Layer Offloading

Use the `-ngl` (number of GPU layers) flag to control offloading:

```bash
# Full GPU offload (if model fits)
llama-cli -m model.gguf -ngl 99

# Partial offload (32 layers to GPU, rest to CPU)
llama-cli -m model.gguf -ngl 32
```

**Performance Impact of Partial Offloading:**
- Full GPU offload: ~40 tokens/sec (example)
- Partial offload (25 layers): ~8.6 tokens/sec (4.7x slower)
- CPU only: ~2-5 tokens/sec

> **Warning:** Partial GPU offloading significantly reduces performance. The more layers on GPU, the better. Never let GPU use RAM as extension - this results in performance worse than CPU alone.

#### Advanced Tensor Offloading (for MoE models)

For fine-grained control with limited VRAM:
```bash
# Selectively offload later up-projection layers to CPU
llama-cli -m model.gguf -ot "\.([2-9][0-9])\.ffn_up_exps.=CPU"
```
This technique can achieve ~67 tok/sec with only 600MB VRAM headroom.

#### Hybrid CPU/GPU Strategy

Best approach for 12GB VRAM:
1. **Primary choice:** 8B models with Q4_K_M - full GPU offload with generous context
2. **If more power needed:** 14B models at Q4_K_M with reduced context (~16K)
3. **Avoid:** 32B+ models unless you accept significant CPU offload performance penalty

**Sources:**
- [Ollama VRAM Requirements: Complete 2025 Guide](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [llama.cpp: CPU vs GPU, shared VRAM and Inference Speed](https://dev.to/maximsaplin/llamacpp-cpu-vs-gpu-shared-vram-and-inference-speed-3jpl)
- [Context Kills VRAM: How to Run LLMs on Consumer GPUs](https://medium.com/@lyx_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632)
- [Guide: Running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)

---

### 1.3 Batch Processing Optimization

#### Key Parameters

```bash
llama-server \
  --batch-size 1024 \      # Tokens processed per batch (prompt processing)
  --ubatch-size 512 \      # Micro-batch size for generation
  --parallel 4 \           # Number of parallel sequences
  --cont-batching          # Enable continuous batching
```

#### Optimization Guidelines

1. **Batch Size (`--batch-size` / `-b`):** Controls tokens fed to LLM in a single processing step
   - Larger values improve prompt processing throughput
   - Must fit within VRAM constraints
   - Optimal value depends on hardware and model

2. **Micro-batch Size (`--ubatch-size` / `-ub`):** Sub-batch for token generation
   - Typically set to half of batch-size
   - Helps with memory management

3. **Parallel Processing (`--parallel` / `-np`):** Number of concurrent request slots
   - Set based on expected concurrent users
   - Each slot requires additional KV cache memory

4. **Continuous Batching (`--cont-batching`):**
   - Dynamically batches multiple requests
   - Essential for server deployments

#### Automated Optimization with llama-optimus

Use the [llama-optimus](https://github.com/ggml-org/llama.cpp/discussions/14191) tool to find optimal parameters:
- Auto-detects safe `-ngl` upper limit
- Optimizes for token generation (tg), prompt processing (pp), or average
- Counter-intuitive finding: reducing threads from 32 to 2 can improve performance for GPU-accelerated models

**Sources:**
- [llama.cpp Guide - Running LLMs Locally](https://blog.steelph0enix.dev/posts/llama-cpp-guide/)
- [Optimizing llama.cpp flags for max performance](https://github.com/ggml-org/llama.cpp/discussions/14191)
- [Qwen llama.cpp Documentation](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html)

---

### 1.4 Flash Attention Usage

#### Enabling Flash Attention

```bash
# Enable with flag
llama-cli -m model.gguf --flash-attn

# Or in server mode
llama-server -m model.gguf -fa
```

#### Benefits

- **Faster prompt processing:** Up to 3x improvement for long contexts
- **Reduced memory usage:** More efficient attention computation
- **Required for KV cache quantization**

#### Benchmark Results (M3 Max)

| Metric | Without Flash Attention | With Flash Attention |
|--------|------------------------|---------------------|
| Time to First Token | 80 seconds | 72 seconds |
| Tokens/sec | 11 tok/sec | 32 tok/sec |

#### Important Caveats

1. **Hardware compatibility:** Flash Attention on Vulkan only supports NVIDIA GPUs with coopmat2; otherwise falls back to CPU
2. **AMD GPUs:** Slight slowdown possible on integrated GPUs with limited memory bandwidth
3. **Automatic fallback:** llama.cpp will warn and fall back if model doesn't support flash attention

**Sources:**
- [llama.cpp Flash Attention PR Discussion](https://github.com/ggml-org/llama.cpp/issues/3365)
- [Flash Attention Performance on M3 Max](https://x.com/N8Programs/status/1785871197932028331)

---

### 1.5 Context Window Optimization

#### KV Cache Memory Usage

The KV cache grows linearly with context and is the primary VRAM consumer after model weights:

| Model Size | Context Length | KV Cache (FP16) | Total VRAM |
|------------|----------------|-----------------|------------|
| 8B | 8K | ~1.1 GB | ~6-7 GB |
| 8B | 32K | ~4.5 GB | ~9-10 GB |
| 8B | 128K | ~18 GB | ~23 GB |

> **Key insight:** On consumer GPUs, pick either a big model OR a long context - rarely both.

#### KV Cache Quantization

Reduce KV cache memory by 50%+ with quantization:

```bash
# Enable KV cache quantization (requires flash attention)
llama-server -m model.gguf -fa -ctk q8_0 -ctv q4_0
```

**Sensitivity findings:**
- K cache is more sensitive to quantization than V cache
- Using q8_0 for K cache and q4_0 for V cache provides good balance
- Q8_0 for full KV cache: negligible quality loss, 50% memory savings

#### Recommended KV Cache Settings

| Setting | Memory Savings | Quality Impact |
|---------|---------------|----------------|
| `-ctk q8_0 -ctv q8_0` | ~50% | Negligible |
| `-ctk q8_0 -ctv q4_0` | ~60% | Minimal |
| `-ctk q4_0 -ctv q4_0` | ~75% | Noticeable degradation |

#### Context Window Strategies

1. **For 12GB VRAM with 8B model:**
   - Maximum context: ~32-45K tokens with FP16 KV
   - With Q8 KV cache: ~64-90K tokens possible

2. **Sliding window attention:** Some models (Mistral) use sliding window natively

3. **Context compression:** For very long contexts, consider:
   - Summarization of older messages
   - Chunked processing with retrieval

**Sources:**
- [KV Cache Quantization Implementation](https://github.com/ggml-org/llama.cpp/issues/6863)
- [Bringing K/V Context Quantisation to Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [KVQuant: Towards 10 Million Context Length](https://arxiv.org/abs/2401.18079)
- [Why does llama.cpp use so much VRAM?](https://github.com/ggml-org/llama.cpp/discussions/9784)

---

## 2. Local LLM Router Architecture Best Practices

### 2.1 Model Management Strategies

#### Multi-Model Architecture

A robust local LLM router should manage multiple models for different purposes:

```
┌─────────────────────────────────────────────────────────┐
│                    LLM Router                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Coding     │  │  General    │  │  Reasoning  │     │
│  │  Model      │  │  Purpose    │  │  Model      │     │
│  │  (8B)       │  │  (8B)       │  │  (14B)      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

#### Model Loading Strategies

1. **Hot-swap models:** Load models on-demand based on task type
   - Pros: Minimal VRAM usage
   - Cons: Cold start latency (10-30 seconds)

2. **Keep primary model loaded:** Always keep one model ready
   - Recommended: Keep most-used model in VRAM
   - Swap for specialized tasks only when needed

3. **Multiple smaller models:** For sufficient VRAM, run 2-3 small models simultaneously
   - Example: Two 7B models in Q4_K_M on 24GB GPU

#### Lifecycle Management

```python
# Pseudo-code for model lifecycle
class ModelManager:
    def __init__(self, vram_budget_gb=12):
        self.vram_budget = vram_budget_gb
        self.loaded_models = {}
        self.model_last_used = {}

    def get_model(self, task_type):
        model_name = self.route_to_model(task_type)

        if model_name not in self.loaded_models:
            self.ensure_vram_available(model_name)
            self.load_model(model_name)

        self.model_last_used[model_name] = time.now()
        return self.loaded_models[model_name]

    def ensure_vram_available(self, new_model):
        # LRU eviction of least recently used model
        while self.current_vram_usage() + model_size > self.vram_budget:
            oldest_model = min(self.model_last_used, key=self.model_last_used.get)
            self.unload_model(oldest_model)
```

**Sources:**
- [RouteLLM Framework | LMSYS](https://lmsys.org/blog/2024-07-01-routellm/)
- [NVIDIA LLM Router Blueprint](https://build.nvidia.com/nvidia/llm-router)
- [LLM Model Routing with Ollama and LiteLLM](https://medium.com/@michael.hannecke/implementing-llm-model-routing-a-practical-guide-with-ollama-and-litellm-b62c1562f50f)

---

### 2.2 Request Routing Algorithms

#### Routing Approaches

1. **Task-Based Routing**
   ```python
   def route_request(query):
       task_type = classify_task(query)

       if task_type == "code":
           return "qwen-coder-7b"
       elif task_type == "reasoning":
           return "deepseek-r1-14b"
       elif task_type == "creative":
           return "llama-3.1-8b"
       else:
           return "qwen-2.5-7b"  # General purpose default
   ```

2. **Complexity-Based Routing (RouteLLM approach)**
   - Train a small router model to predict query complexity
   - Route simple queries to smaller/faster models
   - Route complex queries to larger/better models
   - Can reduce costs by 85% while maintaining 95% quality

3. **Semantic Routing**
   - Use embeddings to match queries to model capabilities
   - More nuanced than keyword-based classification
   - Better for edge cases

4. **Hybrid Gateway-Router Architecture**
   ```
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │   Gateway    │────▶│    Router    │────▶│   Models     │
   │              │     │              │     │              │
   │ • Auth       │     │ • Classify   │     │ • Coding     │
   │ • Rate limit │     │ • Route      │     │ • General    │
   │ • Logging    │     │ • Fallback   │     │ • Reasoning  │
   └──────────────┘     └──────────────┘     └──────────────┘
   ```

#### Key Router Features

- **Low latency:** Router models should be tiny (add <100ms)
- **Fallback handling:** If primary model fails, route to backup
- **Load balancing:** Distribute requests across available resources
- **Quality monitoring:** Track response quality per model

#### When to Implement Routing

| Query Volume | Recommendation |
|--------------|----------------|
| <1K/day | Single model, no router needed |
| 1K-10K/day | Basic 3-tier routing (simple/medium/complex) |
| >10K/day | Full router with semantic classification |

**Sources:**
- [RouteLLM GitHub Repository](https://github.com/lm-sys/RouteLLM)
- [Building an LLM Router | Anyscale](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses)
- [LLM Semantic Router | Red Hat](https://developers.redhat.com/articles/2025/05/20/llm-semantic-router-intelligent-request-routing)
- [The Inference Router | Medium](https://medium.com/@balajibal/the-inference-router-a-critical-component-in-the-llm-ecosystem-15b77df499fc)

---

### 2.3 Caching Strategies for LLM Responses

#### Types of Caching

1. **Exact Match Caching**
   - Simple key-value store
   - Fast but low hit rate
   - Good for structured/repeated queries

2. **Semantic Caching**
   - Store embeddings of queries
   - Return cached response for semantically similar queries
   - 30-40% of LLM requests are similar to previous queries

3. **KV Cache Reuse**
   - Reuse computed attention states for shared prefixes
   - Useful for system prompts and few-shot examples

#### Semantic Caching Implementation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = {}  # query_embedding -> response
        self.embeddings = []
        self.threshold = similarity_threshold

    def get(self, query):
        query_embedding = self.encoder.encode(query)

        for cached_embedding, response in self.cache.items():
            similarity = np.dot(query_embedding, cached_embedding)
            if similarity > self.threshold:
                return response

        return None

    def set(self, query, response):
        embedding = self.encoder.encode(query)
        self.cache[tuple(embedding)] = response
```

#### Caching Frameworks

| Framework | Features | Hit Rate |
|-----------|----------|----------|
| **GPTCache** | LangChain/LlamaIndex integration, tensor caching | Variable |
| **MeanCache** | Privacy-preserving, local caching | 31% same-user queries |
| **GPT Semantic Cache** | Redis-based, up to 68.8% reduction | 61.6-68.8% |
| **Portkey** | Commercial, 99% accuracy | ~20% general, 18-60% RAG |

#### Performance Benefits

- **15x faster** response times for cache hits
- **Up to 68.8% reduction** in API/inference calls
- **30% reduction** in GPU memory usage with tensor caching

#### Best Practices

1. **Set appropriate thresholds:** Too low = cache poisoning, too high = low hit rate
2. **Monitor cache freshness:** Invalidate when model updates
3. **Handle context sensitivity:** Verify cached answer fits new context
4. **Use approximate nearest neighbor (ANN)** search for large caches

**Sources:**
- [GPTCache GitHub](https://github.com/zilliztech/GPTCache)
- [Semantic Caching for LLMs | Redis](https://redis.io/blog/what-is-semantic-caching/)
- [Caching Strategies in LLM Services](https://www.rohan-paul.com/p/caching-strategies-in-llm-services)
- [GPT Semantic Cache Paper](https://arxiv.org/abs/2411.05276)
- [MeanCache Paper](https://arxiv.org/abs/2403.02694)

---

### 2.4 Memory/Context Management

#### Conversation Memory Strategies

1. **Buffer Memory (Naive)**
   - Store all messages
   - Simple but consumes context quickly
   - Only for short conversations

2. **Sliding Window Memory**
   - Keep last N messages
   - Recent context preserved
   - Older context lost completely

3. **Summary Memory**
   - Periodically summarize older messages
   - Compress history into running summary
   - Better long-term coherence

4. **Hybrid (Recommended)**
   - Last 10-15 messages verbatim
   - Summarize older context
   - Best balance for production

#### Implementation Example

```python
class ConversationMemory:
    def __init__(self, window_size=10, max_summary_tokens=500):
        self.window_size = window_size
        self.max_summary_tokens = max_summary_tokens
        self.messages = []
        self.summary = ""

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

        if len(self.messages) > self.window_size * 2:
            self.compress_old_messages()

    def compress_old_messages(self):
        # Summarize oldest messages
        old_messages = self.messages[:self.window_size]
        summary_prompt = f"Summarize this conversation:\n{old_messages}"
        new_summary = llm.generate(summary_prompt)

        self.summary = f"{self.summary}\n{new_summary}"
        self.messages = self.messages[self.window_size:]

    def get_context(self):
        context = []
        if self.summary:
            context.append({"role": "system", "content": f"Previous context: {self.summary}"})
        context.extend(self.messages)
        return context
```

#### Context Degradation Prevention

- **Problem:** Long conversations lead to incoherence ("Context Degradation Syndrome")
- **Solutions:**
  - Regular summarization checkpoints
  - Priority tagging for important information
  - Periodic context refresh/reset
  - Use RAG for fact retrieval instead of relying on context

#### Memory Blocks Architecture (Letta/MemGPT)

Structure context into discrete functional units:
```
┌─────────────────────────────────────────────────────────┐
│                    Context Window                        │
├─────────────────────────────────────────────────────────┤
│ [System Block] Core instructions, personality           │
│ [Persona Block] User preferences, facts                 │
│ [Summary Block] Conversation summary                    │
│ [Working Block] Recent messages (sliding window)        │
│ [Tool Block] Available functions/tools                  │
└─────────────────────────────────────────────────────────┘
```

**Sources:**
- [How to Manage Memory for LLM Chatbot | Vellum](https://www.vellum.ai/blog/how-should-i-manage-memory-for-my-llm-chatbot)
- [Conversational Memory with LangChain | Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/)
- [Memory Blocks | Letta](https://www.letta.com/blog/memory-blocks)
- [LLM Chat History Summarization Guide](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025)
- [Context Management Guide](https://fieldguidetoai.com/guides/context-management)

---

## 3. Model Recommendations for Different Tasks

### 3.1 Best Coding Models (Under 10GB)

| Model | Parameters | VRAM (Q4_K_M) | Strengths |
|-------|------------|---------------|-----------|
| **Qwen3-Coder-7B-Instruct** | 7B | ~4.5 GB | Best for agentic coding, tool calling, debugging |
| **DeepSeek-R1-Distill-7B** | 7B | ~4.2 GB | Strong reasoning for complex coding problems |
| **Code Llama 7B-Instruct** | 7B | ~4.0 GB | Solid all-around, good Python support |
| **Mamba-Codestral-7B** | 7B | ~4.0 GB | Fast inference, Mamba architecture |
| **Phi-3 Mini** | 3.8B | ~2.5 GB | Ultra-compact, good for simple tasks |

#### Top Pick: Qwen3-Coder-7B-Instruct
- Specialized for code generation, debugging, and tool calling
- Excellent at multi-file code understanding
- Strong performance on HumanEval and MBPP benchmarks

**Sources:**
- [Top Local LLMs for Coding 2025 | MarkTechPost](https://www.marktechpost.com/2025/07/31/top-local-llms-for-coding-2025/)
- [Best Local LLMs for Coding | mslinn.com](https://www.mslinn.com/llm/7900-coding-llms.html)
- [10 Best Small Local LLMs | Apidog](https://apidog.com/blog/small-local-llm/)

---

### 3.2 Best General Purpose Models (Under 10GB)

| Model | Parameters | VRAM (Q4_K_M) | Strengths |
|-------|------------|---------------|-----------|
| **Llama 3.1 8B** | 8B | ~5.0 GB | Most reliable, excellent ecosystem support |
| **Qwen 2.5 7B** | 7B | ~4.5 GB | Strong technical accuracy, fewer hallucinations |
| **Mistral 7B** | 7B | ~4.2 GB | Fast, good instruction following |
| **DeepSeek-R1-Distill-7B** | 7B | ~4.2 GB | Best reasoning in size class |
| **Gemma 2 9B** | 9B | ~5.5 GB | Google's efficient architecture |

#### Top Pick: Llama 3.1 8B
- "The Toyota Camry of language models" - reliable and widely compatible
- 128K context window
- Excellent for summarization, basic code, professional writing
- Best ecosystem support across all tools

#### For Technical Work: Qwen 2.5 7B
- Better at code review and technical documentation
- Fewer factual errors
- Better structured output handling
- Apache 2.0 license (fully permissive)

**Sources:**
- [How to Run a Local LLM Guide 2025](https://localllm.in/blog/how-to-run-local-llm-guide-2025)
- [Top 10 Local LLMs for 2025](https://adambernard.com/kb/top-10-local-llms-for-2025/)
- [Local LLM Benchmarks 2025](https://www.practicalwebtools.com/blog/local-llm-benchmarks-consumer-hardware-guide-2025)
- [11 Best Open-Source LLMs 2025 | n8n](https://blog.n8n.io/open-source-llm/)

---

### 3.3 Best Planning/Reasoning Models

| Model | Parameters | VRAM (Q4_K_M) | Strengths |
|-------|------------|---------------|-----------|
| **DeepSeek-R1-Distill-14B** | 14B | ~9.0 GB | Best reasoning in compact size |
| **DeepSeek-R1-Distill-7B** | 7B | ~4.2 GB | Strong chain-of-thought |
| **Qwen3-30B-A3B-Thinking** | 30B (3B active) | ~8.0 GB | MoE efficiency with 256K context |
| **Qwen 2.5 14B** | 14B | ~9.0 GB | Excellent for structured planning |

#### Top Pick for 12GB VRAM: DeepSeek-R1-Distill-14B
- Achieves performance comparable to OpenAI o1 on reasoning tasks
- Excellent at multi-step planning and analysis
- Chain-of-thought reasoning built-in
- Fits in 12GB with Q4_K_M and reasonable context

#### Budget Option: DeepSeek-R1-Distill-7B
- "Probably the best reasoning model in the 8B size"
- Great for scratchpad workflows
- Fits easily in 8GB VRAM

**Sources:**
- [Best Open Source LLM for Planning Tasks 2025 | SiliconFlow](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Planning-Tasks)
- [10 Best Open-Source LLM Models 2025 | Hugging Face](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Top 5 Local LLM Tools and Models 2025 | Pinggy](https://pinggy.io/blog/top_5_local_llm_tools_and_models_2025/)

---

## 4. Performance Benchmarks and Metrics

### 4.1 Typical Response Times

#### Token Generation Speed by Hardware

| GPU | 8B Model | 14B Model | 32B Model |
|-----|----------|-----------|-----------|
| RTX 5090 (32GB) | 213 tok/s | ~120 tok/s | 61 tok/s |
| RTX 4090 (24GB) | 128 tok/s | ~70 tok/s | ~35 tok/s |
| RTX 5080 (16GB) | 132 tok/s | ~75 tok/s | - |
| RTX 4070 (12GB) | ~70 tok/s | 42 tok/s | - |
| RTX 3060 (12GB) | 45 tok/s | ~25 tok/s | - |
| Intel Arc B580 (12GB) | 62 tok/s | ~35 tok/s | - |

#### Performance Tiers

| Speed | Typical Setup | User Experience |
|-------|---------------|-----------------|
| 5-15 tok/s | CPU-only or old hardware | Noticeable delay |
| 20-40 tok/s | Mid-range GPU, 7B models | Good for reading along |
| 50-100 tok/s | High-end GPU, 7-12B models | Fast, good for coding |
| 100-200+ tok/s | Top consumer GPU | Near-instant response |

#### Key Metrics

1. **Time to First Token (TTFT):** Latency before generation starts
   - Affected by prompt length and batch size
   - Flash attention significantly improves this

2. **Tokens Per Second (TPS):** Generation speed
   - Most user-visible metric
   - 20-40 tok/s is sufficient for most use cases

3. **End-to-End Latency:** Total request time
   - Includes TTFT + generation + any post-processing

**Sources:**
- [Best GPUs for Local LLM Inference 2025](https://localllm.in/blog/best-gpus-llm-inference-2025)
- [LLM Inference Speed Benchmarks | Spare Cores](https://sparecores.com/article/llm-inference-speed)
- [LLM Inference Benchmarking | NVIDIA](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)

---

### 4.2 VRAM Usage Patterns

#### VRAM Formula

```
Total VRAM = Model Weights + KV Cache + Overhead

Model Weights (GB) ≈ (Parameters in B × Bits) / 8
KV Cache (GB) ≈ 2 × Layers × Hidden × Context / 1e9 × Precision
Overhead ≈ 0.5-1.0 GB (CUDA context, activations)
```

#### Example Calculations (Q4_K_M)

| Model | Weights | 4K Context | 16K Context | 32K Context |
|-------|---------|------------|-------------|-------------|
| 7B | ~3.5 GB | ~4.5 GB | ~6.5 GB | ~9.5 GB |
| 8B | ~4.0 GB | ~5.0 GB | ~7.5 GB | ~11.0 GB |
| 14B | ~7.0 GB | ~8.5 GB | ~11.5 GB | ~16.0 GB |
| 32B | ~16 GB | ~18 GB | ~23 GB | ~32 GB |

#### VRAM by Quantization (8B Model, 8K Context)

| Quantization | Model Size | Total VRAM |
|--------------|------------|------------|
| FP16 | 16 GB | ~18 GB |
| Q8_0 | 8 GB | ~10 GB |
| Q6_K | 6 GB | ~8 GB |
| Q5_K_M | 5 GB | ~7 GB |
| Q4_K_M | 4 GB | ~6 GB |

**Sources:**
- [Ollama VRAM Requirements 2025](https://localllm.in/blog/ollama-vram-requirements-for-local-llms)
- [VRAM Calculator for Local LLMs](https://localllm.in/blog/interactive-vram-calculator)
- [LLM VRAM Usage Compared](https://www.hardware-corner.net/llm-vram-usage-compared/)

---

### 4.3 Quality vs Speed Trade-offs

#### Quantization Quality Impact

| Quantization | Speed Boost | Quality Impact | Recommendation |
|--------------|-------------|----------------|----------------|
| FP16 | Baseline | Baseline | Only if VRAM permits |
| Q8_0 | ~1.5x | Negligible (<1% ppl) | Best quality/speed |
| Q6_K | ~2x | Minimal | Great for 7-14B models |
| Q5_K_M | ~2.5x | Small | Sweet spot for desktop |
| Q4_K_M | ~3x | Acceptable (+0.05 ppl) | Best for constrained VRAM |
| Q3_K | ~4x | Noticeable | Only if necessary |
| Q2_K | ~5x | Severe degradation | Not recommended |

#### Key Findings from 500K+ Evaluations

1. **FP8 is effectively lossless** across all model scales
2. **INT8 achieves 1-3% accuracy degradation** with good tuning
3. **INT4 (Q4_K_M) is more competitive than expected**, rivaling 8-bit in many tasks
4. **Below 3-bit:** Quality drops precipitously - avoid for production

#### Model Size vs Quantization Trade-off

> A quantized larger model often outperforms a smaller FP16 model at the same VRAM budget.

Example at 12GB VRAM:
- Llama 3.1 8B @ FP16: Won't fit
- Llama 3.1 8B @ Q4_K_M: Fits with 32K context, better quality
- Qwen 2.5 14B @ Q4_K_M: Fits with 8K context, best quality

#### Practical Guidelines

1. **For reading responses (chat):** 20 tok/s is plenty fast
2. **For coding/iteration:** Aim for 50+ tok/s
3. **Quality priority:** Use Q5_K_M or Q6_K with smaller models
4. **Speed priority:** Use Q4_K_M with larger models
5. **Never go below Q3_K** for production use

**Sources:**
- [Half Million Evaluations on Quantized LLMs | Red Hat](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)
- [Exploring Quantization Trade-Offs | arXiv](https://arxiv.org/abs/2409.11055)
- [Accuracy-Performance Trade-Offs in LLM Quantization | arXiv](https://arxiv.org/abs/2411.02355)
- [Quantization in LLMs: Why Does It Matter | Medium](https://medium.com/data-from-the-trenches/quantization-in-llms-why-does-it-matter-7c32d2513c9e)

---

## Quick Reference: Recommended Configurations

### For 12GB VRAM (RTX 3060, 4070, Arc B580)

| Use Case | Model | Quantization | Context | Expected Speed |
|----------|-------|--------------|---------|----------------|
| **General Purpose** | Llama 3.1 8B | Q4_K_M | 32K | 45-70 tok/s |
| **Coding** | Qwen3-Coder-7B | Q4_K_M | 32K | 45-70 tok/s |
| **Reasoning** | DeepSeek-R1-14B | Q4_K_M | 8K | 25-40 tok/s |
| **Fast Chat** | Mistral 7B | Q4_K_M | 16K | 50-75 tok/s |

### Optimal llama.cpp Server Configuration

```bash
llama-server \
  -m /path/to/model-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \                    # Full GPU offload
  -c 32768 \                   # 32K context
  -b 1024 \                    # Batch size
  -ub 512 \                    # Micro-batch size
  -fa \                        # Flash attention
  -ctk q8_0 \                  # KV cache key quantization
  -ctv q4_0 \                  # KV cache value quantization
  --parallel 2 \               # Concurrent slots
  --cont-batching              # Continuous batching
```

---

## Additional Resources

### Tools
- [Ollama](https://ollama.ai) - Easiest way to run local LLMs
- [LM Studio](https://lmstudio.ai) - GUI for local LLM management
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - High-performance inference engine
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API for multiple LLM providers

### Model Sources
- [Hugging Face](https://huggingface.co) - Model repository
- [TheBloke](https://huggingface.co/TheBloke) - Pre-quantized GGUF models

### Benchmarks & Leaderboards
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models)

---

*Document generated: December 2025*
*For the latest information, always check the source links and official documentation.*
