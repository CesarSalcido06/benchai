# BenchAI Installation Guide

Complete installation guide for setting up BenchAI on your system.

## Prerequisites

### Hardware
- **CPU:** Modern x86_64 processor (AMD/Intel)
- **RAM:** 16GB minimum, 32GB recommended
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3060 12GB recommended)
- **Storage:** 50GB+ free space for models

### Software
- Ubuntu 22.04 / Pop!_OS 22.04 or similar
- NVIDIA drivers 535+
- CUDA 12.x
- Python 3.10+
- Docker & Docker Compose

## Step 1: Install NVIDIA Drivers & CUDA

```bash
# Check current driver
nvidia-smi

# If not installed, install drivers
sudo apt update
sudo apt install nvidia-driver-535

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify
nvcc --version
```

## Step 2: Install llama.cpp with CUDA

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA support
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Verify
./bin/llama-server --help
```

## Step 3: Download Models

```bash
cd ~/llama.cpp/models

# Phi-3 Mini (fast, general)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf \
  -O phi-3-mini-4k-instruct.Q4_K_M.gguf

# Qwen2.5 7B (planner, research)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf \
  -O qwen2.5-7b-instruct.Q4_K_M.gguf

# DeepSeek Coder (code)
wget https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
  -O deepseek-coder-6.7b-instruct.Q4_K_M.gguf

# Qwen2-VL (vision) - optional
wget https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF/resolve/main/Qwen2-VL-7B-Instruct-Q4_K_M.gguf \
  -O Qwen2-VL-7B-Instruct-Q4_K_M.gguf
wget https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2-VL-7B-Instruct-f16.gguf \
  -O mmproj-Qwen2-VL-7B-Instruct-f16.gguf
```

## Step 4: Install Python Dependencies

```bash
cd ~/llama.cpp/router

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn httpx aiosqlite pydantic
pip install chromadb sentence-transformers  # For RAG
pip install piper-tts                        # For TTS

# Or use requirements.txt
pip install -r requirements.txt
```

## Step 5: Set Up Storage Directories

```bash
# Create LLM storage directory
mkdir -p ~/llm-storage/{memory,rag,cache,models}

# Set permissions
chmod -R 755 ~/llm-storage
```

## Step 6: Configure Environment

```bash
cd ~/llama.cpp/router

# Create .env file
cat > .env << 'EOF'
# Obsidian Local REST API (optional)
OBSIDIAN_API_KEY=your-api-key-here
OBSIDIAN_API_URL=https://localhost:27124
OBSIDIAN_VAULT_PATH=/path/to/your/vault

# Proton Mail via Hydroxide (optional)
PROTON_BRIDGE_USER=your-email@proton.me
PROTON_IMAP_HOST=127.0.0.1
PROTON_IMAP_PORT=1143
PROTON_SMTP_HOST=127.0.0.1
PROTON_SMTP_PORT=1025
EOF

# Secure the file
chmod 600 .env
```

## Step 7: Install Piper TTS Voice (Optional)

```bash
mkdir -p ~/llama.cpp/router/voices
cd ~/llama.cpp/router/voices

# Download voice model
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Step 8: Set Up Docker Services

```bash
cd ~/docker-server  # Or your docker directory

# Copy the docker-compose template
cp /path/to/benchai/docker/docker-compose.yml .

# Start services
docker compose up -d

# Verify
docker ps
```

## Step 9: Create Systemd Service

```bash
# Copy service file
sudo cp ~/benchai/services/benchai.service /etc/systemd/system/

# Edit to match your paths if needed
sudo nano /etc/systemd/system/benchai.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable benchai
sudo systemctl start benchai

# Check status
sudo systemctl status benchai
```

## Step 10: Verify Installation

```bash
# Health check
curl http://localhost:8085/health

# Expected response:
# {"status":"ok","service":"benchai-router-v3","features":{"streaming":true,"memory":true,"tts":true,"rag":true}}

# Test chat
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Check memory
curl http://localhost:8085/v1/memory/stats

# Check RAG
curl http://localhost:8085/v1/rag/stats
```

## Step 11: Optional Web Interface

If using Open WebUI or another web interface:

1. Point it to: `http://localhost:8085/v1`
2. Use API Key: `not-needed` (any value works)
3. Select model: `auto` for routing

**Recommended:** Install [benchai-client](../benchai-client) for CLI, VS Code, and Neovim instead.

## Post-Installation

### Index Your Codebase (RAG)

```bash
curl -X POST http://localhost:8085/v1/rag/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/myproject", "recursive": true}'
```

### Optimize Memory Database

```bash
curl -X POST http://localhost:8085/v1/memory/optimize
```

### Set Up Remote Access (Optional)

For remote access, configure:
- VPN (Tailscale, WireGuard, etc.)
- Reverse proxy with SSL (Nginx, Caddy)
- Port forwarding (less secure)

**BenchAI Endpoint:** `your-server-ip:8085`

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## Next Steps

- Read the [User Guide](USER-GUIDE.md)
- Set up [client tools](https://github.com/YOUR_USERNAME/benchai-client)
- Explore the [API Reference](API.md)
