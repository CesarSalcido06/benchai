#!/bin/bash
# Setup script for BenchAI Learning System
# This installs all dependencies for the self-improving AI architecture

set -e

echo "=== BenchAI Learning System Setup ==="
echo ""

# Check CUDA
echo "[1/5] Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "  CUDA ${CUDA_VERSION} found"
else
    echo "  WARNING: CUDA not found. GPU training will not be available."
fi

# Check GPU
echo ""
echo "[2/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPU: ${GPU_NAME}"
    echo "  Memory: ${GPU_MEM}"
else
    echo "  WARNING: nvidia-smi not found."
fi

# Install base dependencies
echo ""
echo "[3/5] Installing base Python dependencies..."
pip3 install --quiet --upgrade pip
pip3 install --quiet aiosqlite>=0.19.0
pip3 install --quiet datasets>=2.14.0
pip3 install --quiet accelerate>=0.25.0
pip3 install --quiet bitsandbytes>=0.41.0
pip3 install --quiet scipy

echo "  Base dependencies installed"

# Install training libraries
echo ""
echo "[4/5] Installing training libraries..."
pip3 install --quiet peft>=0.7.0
pip3 install --quiet trl>=0.7.0

# Try to install Unsloth
echo ""
echo "[5/5] Installing Unsloth (optimized training)..."
if pip3 install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null; then
    echo "  Unsloth installed successfully"
    UNSLOTH_INSTALLED=true
else
    echo "  WARNING: Unsloth installation failed. Using transformers fallback."
    echo "  For optimal performance, install Unsloth manually:"
    echo "    pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
    UNSLOTH_INSTALLED=false
fi

# Create directories
echo ""
echo "Creating storage directories..."
mkdir -p ~/llm-storage/learning
mkdir -p ~/llm-storage/adapters
mkdir -p ~/llm-storage/memory
mkdir -p ~/llm-storage/models

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Learning System Directories:"
echo "  - Memory DB: ~/llm-storage/memory/"
echo "  - Experiences: ~/llm-storage/learning/"
echo "  - LoRA Adapters: ~/llm-storage/adapters/"
echo ""
echo "To start the learning system, add to your router startup:"
echo "  from learning.integration import create_learning_system"
echo "  learning_system = create_learning_system()"
echo "  await learning_system.initialize()"
echo ""

if [ "$UNSLOTH_INSTALLED" = true ]; then
    echo "Unsloth: INSTALLED (2.5x faster training)"
else
    echo "Unsloth: NOT INSTALLED (using fallback)"
fi
