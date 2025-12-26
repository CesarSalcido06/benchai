# BenchAI Troubleshooting Guide

Common issues and solutions for BenchAI.

## Table of Contents

1. [Service Won't Start](#service-wont-start)
2. [Connection Issues](#connection-issues)
3. [Model/GPU Issues](#modelgpu-issues)
4. [Memory/Database Issues](#memorydatabase-issues)
5. [RAG Issues](#rag-issues)
6. [TTS Issues](#tts-issues)
7. [Docker Issues](#docker-issues)
8. [Client/IDE Issues](#clientide-issues)
9. [Remote Access Issues](#remote-access-issues)

---

## Service Won't Start

### Error: "File exists" on LLM storage path

**Symptom:** Service fails with `FileExistsError: [Errno 17] File exists: '/home/user/llm-storage'`

**Cause:** The storage path exists as a symlink but the target drive isn't mounted.

**Solution:**
```bash
# Check if it's a symlink
ls -la ~/llm-storage

# If symlink, check target
readlink ~/llm-storage

# Mount the drive
udisksctl mount -b /dev/sda1  # Or your drive

# Or remove symlink and create directory
rm ~/llm-storage
mkdir -p ~/llm-storage/{memory,rag,cache}
```

### Error: "Address already in use" on port 8085

**Symptom:** Service repeatedly restarts with port binding error.

**Solution:**
```bash
# Find process using the port
sudo lsof -i :8085

# Kill it
sudo kill -9 <PID>

# Or kill all BenchAI processes
pkill -9 -f llm_router
pkill -9 -f llama-server

# Restart service
sudo systemctl restart benchai
```

### Error: Permission denied

**Symptom:** Service can't read files or write to directories.

**Solution:**
```bash
# Fix ownership
sudo chown -R $USER:$USER ~/llama.cpp
sudo chown -R $USER:$USER ~/llm-storage

# Fix permissions
chmod -R 755 ~/llama.cpp/router
chmod 600 ~/llama.cpp/router/.env
```

---

## Connection Issues

### Can't connect from another machine

**Symptom:** `curl: (7) Failed to connect to 192.168.x.x port 8085`

**Solutions:**

1. **Check firewall:**
```bash
sudo ufw status
sudo ufw allow 8085/tcp
```

2. **Check service is listening on all interfaces:**
```bash
ss -tlnp | grep 8085
# Should show 0.0.0.0:8085, not 127.0.0.1:8085
```

3. **Check router config:** Ensure FastAPI binds to `0.0.0.0`:
```python
# In llm_router.py
uvicorn.run(app, host="0.0.0.0", port=8085)
```

### SSL/TLS errors from client

**Symptom:** `LibreSSL error: tlsv1 unrecognized name`

**Cause:** Client trying HTTPS on HTTP endpoint, or proxy interference.

**Solution:**
```bash
# Disable proxy
unset http_proxy https_proxy

# Use explicit HTTP
curl --noproxy '*' http://192.168.0.213:8085/health

# On Mac, check for corporate proxy
echo $http_proxy $https_proxy
```

### Connection refused

**Symptom:** `Connection refused` error

**Check:**
```bash
# Is service running?
systemctl status benchai

# Is port open?
nc -zv localhost 8085

# Check logs
journalctl -u benchai -n 50
```

---

## Model/GPU Issues

### Out of VRAM

**Symptom:** Model loading fails with CUDA out of memory.

**Solutions:**

1. **Reduce GPU layers:**
```python
# In MODELS config, lower gpu_layers
"gpu_layers": 20,  # Instead of 35
```

2. **Use smaller quantization:**
```bash
# Use Q3_K_M instead of Q4_K_M models
```

3. **Kill other GPU processes:**
```bash
nvidia-smi
# Kill any other CUDA processes
```

### Model file not found

**Symptom:** `FileNotFoundError: model.gguf`

**Solution:**
```bash
# Check models directory
ls -la ~/llama.cpp/models/

# Verify model names match config
# In llm_router.py, check MODELS dict
```

### llama-server not found

**Symptom:** `FileNotFoundError: llama-server`

**Solution:**
```bash
# Rebuild llama.cpp
cd ~/llama.cpp/build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Verify
ls -la ~/llama.cpp/build/bin/llama-server
```

---

## Memory/Database Issues

### Database locked

**Symptom:** `sqlite3.OperationalError: database is locked`

**Solution:**
```bash
# Check for multiple processes
pgrep -af llm_router

# Kill duplicates
pkill -9 -f llm_router

# Restart cleanly
sudo systemctl restart benchai
```

### Memory search returns nothing

**Symptom:** Memory search returns empty results.

**Solutions:**

1. **Check if FTS5 is enabled:**
```bash
curl http://localhost:8085/v1/memory/stats
# Should show "fts5_enabled": true
```

2. **Migrate to FTS5:**
```bash
curl -X POST http://localhost:8085/v1/memory/migrate-fts
```

3. **Optimize database:**
```bash
curl -X POST http://localhost:8085/v1/memory/optimize
```

### Database corruption

**Symptom:** SQLite errors, missing data.

**Solution:**
```bash
# Backup current database
cp ~/llm-storage/memory/benchai_memory.db ~/llm-storage/memory/benchai_memory.db.bak

# Try integrity check
sqlite3 ~/llm-storage/memory/benchai_memory.db "PRAGMA integrity_check;"

# If corrupted, recover:
sqlite3 ~/llm-storage/memory/benchai_memory.db ".recover" | sqlite3 ~/llm-storage/memory/benchai_memory_new.db
mv ~/llm-storage/memory/benchai_memory_new.db ~/llm-storage/memory/benchai_memory.db
```

---

## RAG Issues

### ChromaDB not available

**Symptom:** `[WARN] ChromaDB not available - RAG disabled`

**Solution:**
```bash
pip install chromadb sentence-transformers
```

### RAG search returns nothing

**Solutions:**

1. **Check if documents are indexed:**
```bash
curl http://localhost:8085/v1/rag/stats
# Should show count > 0
```

2. **Re-index documents:**
```bash
curl -X POST http://localhost:8085/v1/rag/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/your/code", "recursive": true}'
```

---

## TTS Issues

### Piper not available

**Symptom:** `[WARN] Piper not available - TTS disabled`

**Solution:**
```bash
pip install piper-tts

# Download voice
mkdir -p ~/llama.cpp/router/voices
cd ~/llama.cpp/router/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### TTS returns 422 error

**Symptom:** TTS endpoint returns validation error.

**Solution:** Check request format:
```bash
# Correct format (uses "text" field)
curl -X POST http://localhost:8085/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

---

## Docker Issues

### Container can't reach BenchAI

**Symptom:** Docker container can't connect to host's port 8085.

**Solution:** Use `host.docker.internal` or host network mode:
```yaml
# In docker-compose.yml
services:
  open-webui:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### SearXNG not working

**Symptom:** Web search fails.

**Solution:**
```bash
# Check if running
docker ps | grep searxng

# Check logs
docker logs searxng

# Restart
docker compose restart searxng
```

---

## Client/IDE Issues

### Neovim Avante asks for API key

**Symptom:** Avante prompts for OpenAI API key.

**Solution:**
```bash
# Set dummy key
echo 'export OPENAI_API_KEY="not-needed"' >> ~/.zshrc
source ~/.zshrc
```

### Continue.dev not connecting

**Symptom:** VS Code Continue can't reach BenchAI.

**Solutions:**

1. **Check config:**
```bash
cat ~/.continue/config.json
# Verify apiBase is correct
```

2. **Test connection:**
```bash
curl http://192.168.0.213:8085/health
```

3. **Restart VS Code**

### CLI benchai command not found

**Symptom:** `benchai: command not found`

**Solution:**
```bash
# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Make permanent
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

## Remote Access Issues

### Twingate can't connect

**Solutions:**

1. **Check connector is running:**
```bash
docker ps | grep twingate
docker logs twingate-connector
```

2. **Verify resources in admin portal:** Go to your Twingate admin URL and ensure resources are added.

3. **Check client is connected:** Verify Twingate client shows "Connected" on your device.

### NTFS drive not mounting on boot

**Symptom:** External drive with models doesn't mount, service fails.

**Solution:**
```bash
# Get UUID
lsblk -f /dev/sda1

# Add to fstab
echo 'UUID=YOUR-UUID "/media/user/New Volume" ntfs-3g defaults,nofail,uid=1000,gid=1000 0 0' | sudo tee -a /etc/fstab

# Test
sudo mount -a
```

---

## Logs and Diagnostics

### View service logs
```bash
journalctl -u benchai -f
journalctl -u benchai -n 100 --no-pager
```

### View router log file
```bash
tail -f /var/log/benchai-router.log
```

### Full system check
```bash
# Service status
systemctl status benchai hydroxide docker

# Health check
curl http://localhost:8085/health

# Memory stats
curl http://localhost:8085/v1/memory/stats

# RAG stats
curl http://localhost:8085/v1/rag/stats

# GPU status
nvidia-smi

# Docker containers
docker ps
```

---

## Getting Help

1. Check logs first: `journalctl -u benchai -n 100`
2. Search this document for your error
3. Check GitHub issues
4. Open a new issue with:
   - Error message
   - System info (OS, GPU, RAM)
   - Steps to reproduce
