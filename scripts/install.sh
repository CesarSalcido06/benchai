#!/bin/bash
#
# BenchAI Server Installation Script
# Installs and configures BenchAI on a fresh system
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "  BenchAI Server Installer"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    log_error "Please run as normal user, not root"
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required. Install with: sudo apt install python3 python3-pip"
        exit 1
    fi

    # NVIDIA GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "NVIDIA drivers not found. GPU acceleration will not be available."
    else
        log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi

    # Docker
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found. Some features will be limited."
    fi

    # Git
    if ! command -v git &> /dev/null; then
        log_error "Git is required. Install with: sudo apt install git"
        exit 1
    fi
}

# Install llama.cpp
install_llama_cpp() {
    log_info "Installing llama.cpp..."

    if [ -d "$HOME/llama.cpp" ]; then
        log_info "llama.cpp already exists, updating..."
        cd "$HOME/llama.cpp"
        git pull
    else
        cd "$HOME"
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp
    fi

    # Build with CUDA if available
    mkdir -p build && cd build
    if command -v nvcc &> /dev/null; then
        log_info "Building with CUDA support..."
        cmake .. -DGGML_CUDA=ON
    else
        log_info "Building CPU-only version..."
        cmake ..
    fi
    cmake --build . --config Release -j$(nproc)

    log_info "llama.cpp installed successfully"
}

# Set up directories
setup_directories() {
    log_info "Setting up directories..."

    mkdir -p "$HOME/llama.cpp/router"
    mkdir -p "$HOME/llama.cpp/models"
    mkdir -p "$HOME/llm-storage/memory"
    mkdir -p "$HOME/llm-storage/rag"
    mkdir -p "$HOME/llm-storage/cache"

    log_info "Directories created"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."

    pip3 install --user -r "$PROJECT_DIR/configs/requirements.txt"

    log_info "Python dependencies installed"
}

# Copy router files
setup_router() {
    log_info "Setting up router..."

    # Copy router if exists
    if [ -f "$PROJECT_DIR/router/llm_router.py" ]; then
        cp "$PROJECT_DIR/router/llm_router.py" "$HOME/llama.cpp/router/"
    fi

    # Copy env template
    if [ ! -f "$HOME/llama.cpp/router/.env" ]; then
        cp "$PROJECT_DIR/configs/.env.example" "$HOME/llama.cpp/router/.env"
        chmod 600 "$HOME/llama.cpp/router/.env"
        log_warn "Created .env file - please edit with your API keys"
    fi

    log_info "Router setup complete"
}

# Install systemd service
install_service() {
    log_info "Installing systemd service..."

    # Update paths in service file
    sed "s|/home/user|$HOME|g" "$PROJECT_DIR/services/benchai.service" | \
        sed "s|User=user|User=$USER|g" | \
        sed "s|Group=user|Group=$USER|g" > /tmp/benchai.service

    sudo cp /tmp/benchai.service /etc/systemd/system/benchai.service
    sudo touch /var/log/benchai-router.log
    sudo chown "$USER:$USER" /var/log/benchai-router.log

    sudo systemctl daemon-reload
    sudo systemctl enable benchai

    log_info "Systemd service installed"
}

# Download models
download_models() {
    log_info "Downloading models..."

    cd "$HOME/llama.cpp/models"

    # Phi-3 Mini (required - small and fast)
    if [ ! -f "phi-3-mini-4k-instruct.Q4_K_M.gguf" ]; then
        log_info "Downloading Phi-3 Mini..."
        wget -q --show-progress \
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf" \
            -O "phi-3-mini-4k-instruct.Q4_K_M.gguf"
    fi

    log_info "Models downloaded"
    log_warn "For additional models, see docs/INSTALLATION.md"
}

# Print summary
print_summary() {
    echo ""
    echo "================================"
    echo "  Installation Complete!"
    echo "================================"
    echo ""
    echo "Next steps:"
    echo "  1. Edit config: nano ~/llama.cpp/router/.env"
    echo "  2. Download more models (see docs/INSTALLATION.md)"
    echo "  3. Start service: sudo systemctl start benchai"
    echo "  4. Check health: curl http://localhost:8085/health"
    echo ""
    echo "Documentation: $PROJECT_DIR/docs/"
    echo ""
}

# Main
main() {
    check_prerequisites

    read -p "Continue with installation? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        exit 0
    fi

    install_llama_cpp
    setup_directories
    install_python_deps
    setup_router
    install_service
    download_models
    print_summary
}

main "$@"
