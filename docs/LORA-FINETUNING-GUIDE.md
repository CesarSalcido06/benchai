# LoRA Fine-Tuning Guide for BenchAI

**Purpose:** Fine-tune local models with custom data using LoRA (Low-Rank Adaptation)

---

## Prerequisites

Your system already has everything needed:
- `llama-finetune`: `/home/user/llama.cpp/build/bin/llama-finetune`
- `llama-export-lora`: `/home/user/llama.cpp/build/bin/llama-export-lora`
- RTX 3060 (12GB VRAM) - suitable for fine-tuning smaller models

---

## Quick Start

### 1. Prepare Training Data

Create a training file with Q&A pairs. Format depends on the model's chat template:

**train.txt** (Llama2/general format):
```
<SFT>
<s>[INST] What is BenchAI? [/INST]
BenchAI is a local LLM router that manages multiple models for different tasks.
</s>
<SFT>
<s>[INST] How do I restart BenchAI? [/INST]
Run: sudo systemctl restart benchai
</s>
```

**train_chatml.txt** (Qwen/DeepSeek format):
```
<|im_start|>system
You are a helpful coding assistant.
<|im_end|>
<|im_start|>user
Write a hello world in Python
<|im_end|>
<|im_start|>assistant
print("Hello, World!")
<|im_end|>
```

### 2. Run Fine-Tuning

```bash
# Basic fine-tuning (CPU)
~/llama.cpp/build/bin/llama-finetune \
  -m ~/llama.cpp/models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --lora ~/llm-storage/lora/my-adapter.gguf \
  -o ~/llm-storage/lora/my-finetuned.gguf \
  -lr 1e-5 \
  --epochs 2 \
  -t 12

# With GPU acceleration (recommended)
~/llama.cpp/build/bin/llama-finetune \
  -m ~/llama.cpp/models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --lora ~/llm-storage/lora/my-adapter.gguf \
  -o ~/llm-storage/lora/my-finetuned.gguf \
  -lr 1e-5 \
  --epochs 2 \
  -ngl 35 \
  -t 8
```

### 3. Use the Fine-Tuned Model

**Option A: Load LoRA adapter at runtime**
```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/llama.cpp/models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --lora ~/llm-storage/lora/my-adapter.gguf \
  --port 8094
```

**Option B: Merge into single GGUF**
```bash
~/llama.cpp/build/bin/llama-export-lora \
  -m ~/llama.cpp/models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --lora ~/llm-storage/lora/my-adapter.gguf \
  -o ~/llama.cpp/models/phi-3-custom.gguf
```

---

## Recommended Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `-lr` | 1e-5 to 5e-5 | Learning rate (start low) |
| `--epochs` | 2-5 | More epochs = risk of overfitting |
| `-wd` | 1e-9 | Weight decay (prevents overfitting) |
| `--val-split` | 0.1 | 10% of data for validation |
| `-t` | 12 | CPU threads |
| `-ngl` | 35 | GPU layers (if VRAM available) |
| `-c` | 2048 | Context size (lower = less VRAM) |

---

## Use Cases for BenchAI

### 1. Custom Code Assistant
Fine-tune DeepSeek Coder with your codebase patterns:
```bash
# Prepare training data from your common code patterns
python3 ~/benchai/scripts/prepare_code_training.py \
  --input ~/my-projects \
  --output ~/llm-storage/training/code_patterns.txt

# Fine-tune
~/llama.cpp/build/bin/llama-finetune \
  -m ~/llama.cpp/models/deepseek-coder-6.7b-instruct.Q5_K_M.gguf \
  --lora ~/llm-storage/lora/my-code-adapter.gguf \
  -lr 2e-5 --epochs 3 -ngl 35
```

### 2. Domain-Specific Assistant
Fine-tune for specific knowledge domains:
```bash
# Example: Medical terminology
~/llama.cpp/build/bin/llama-finetune \
  -m ~/llama.cpp/models/qwen2.5-7b-instruct.Q5_K_M.gguf \
  --lora ~/llm-storage/lora/medical-adapter.gguf \
  -lr 1e-5 --epochs 2
```

### 3. Personality/Style Tuning
Fine-tune for consistent response style:
```bash
# Example: Concise technical responses
~/llama.cpp/build/bin/llama-finetune \
  -m ~/llama.cpp/models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
  --lora ~/llm-storage/lora/concise-style.gguf \
  -lr 3e-5 --epochs 2
```

---

## Directory Structure

```
~/llm-storage/
├── lora/                    # LoRA adapters
│   ├── my-adapter.gguf
│   └── code-adapter.gguf
├── training/                # Training data
│   ├── train.txt
│   └── code_patterns.txt
└── checkpoints/             # Training checkpoints
    └── checkpoint-500.gguf
```

Create directories:
```bash
mkdir -p ~/llm-storage/{lora,training,checkpoints}
```

---

## Integration with BenchAI Router

To use a LoRA-enhanced model in the router, update `llm_router.py`:

```python
MODEL_CONFIGS = {
    # ... existing models ...
    "custom": {
        "name": "Custom Fine-Tuned",
        "file": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "port": 8094,
        "mode": "cpu",
        "context": 4096,
        "extra_args": ["--lora", "/home/user/llm-storage/lora/my-adapter.gguf"]
    }
}
```

---

## Tips & Best Practices

1. **Start small**: Fine-tune Phi-3 Mini first (fastest, lowest VRAM)
2. **Monitor loss**: Stop when validation loss stops decreasing
3. **Save checkpoints**: Use `--checkpoint-in` to resume interrupted training
4. **Data quality > quantity**: 100 high-quality examples beat 1000 mediocre ones
5. **Test before integrating**: Verify the model works before adding to router

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce context and batch size
-c 1024 -b 256 -ub 128
```

### Slow Training
```bash
# Use GPU layers if available
-ngl 20  # Offload some layers to GPU
```

### Poor Results
- Increase training data quality
- Adjust learning rate (try 5e-5 or 1e-5)
- Add more epochs (3-5)

---

## Resources

- [llama.cpp Fine-Tuning Tutorial](https://docs.gaianet.ai/tutorial/llamacpp/)
- [GGUF-my-LoRA (HuggingFace)](https://huggingface.co/spaces/ggml-org/gguf-my-lora)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

*Guide version 1.0 - December 26, 2025*
