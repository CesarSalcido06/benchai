# BenchAI Maintenance Runbook

**Purpose:** Step-by-step procedures for maintaining and troubleshooting BenchAI

---

## Daily Health Check (2 minutes)

```bash
# 1. Check router health
curl -s http://localhost:8085/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('Router:', d['status'], '| Features:', list(d['features'].keys()))"

# 2. Check all models running
ss -tlnp | grep -E "8091|8092|8093" | wc -l
# Expected: 3

# 3. Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
# Expected: ~9000 MiB / 12288 MiB

# 4. Quick chat test
curl -s -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"general","messages":[{"role":"user","content":"ping"}],"max_tokens":10}' \
  | python3 -c "import sys,json; print('Response OK' if 'choices' in json.load(sys.stdin) else 'FAILED')"
```

---

## Common Issues & Fixes

### Issue 1: Router Not Responding

**Symptoms:** `curl http://localhost:8085/health` times out or refuses connection

**Fix:**
```bash
# Check if process exists
ps aux | grep llm_router | grep -v grep

# If not running, check logs
cat /tmp/benchai-router.log | tail -50

# Restart
sudo systemctl restart benchai
# OR manual:
cd /home/user/llama.cpp/router
pkill -f llm_router
python3 llm_router.py 2>&1 &
```

### Issue 2: Model Not Responding (Empty Response)

**Symptoms:** Chat returns "empty response from model"

**Fix:**
```bash
# Check which models are running
ss -tlnp | grep -E "8091|8092|8093|8094"

# Check model health directly
curl http://localhost:8091/health  # general
curl http://localhost:8092/health  # planner
curl http://localhost:8093/health  # code

# If model down, router should auto-restart
# Force restart by restarting router:
sudo systemctl restart benchai
```

### Issue 3: Port Already in Use

**Symptoms:** `ERROR: address already in use`

**Fix:**
```bash
# Find what's using the port
fuser 8085/tcp
# Or
ss -tlnp | grep 8085

# Kill it
fuser -k 8085/tcp

# Wait and restart
sleep 3
sudo systemctl restart benchai
```

### Issue 4: GPU Out of Memory

**Symptoms:** Model crashes, CUDA out of memory errors

**Fix:**
```bash
# Check what's using GPU
nvidia-smi

# Kill zombie processes
pkill -9 -f llama-server

# Clear GPU memory
sudo fuser -k /dev/nvidia*

# Wait for release
sleep 5

# Restart router (will start models with proper allocation)
sudo systemctl restart benchai
```

### Issue 5: Slow Responses (>30 seconds)

**Symptoms:** Code model taking 30+ seconds

**Fix:**
```bash
# Check if code model is on GPU
ps aux | grep "port 8093" | grep -o '\-ngl [0-9]*'
# Should show: -ngl 35

# If showing -ngl 0 (CPU mode), GPU was unavailable
# Free GPU memory and restart:
pkill -9 -f llama-server
sleep 3
sudo systemctl restart benchai
```

### Issue 6: Docker Containers Down

**Symptoms:** SearXNG or Open WebUI not working

**Fix:**
```bash
# Check container status
docker ps

# Restart specific container
docker restart searxng
docker restart open-webui

# Restart all
cd ~/docker-server
docker compose down
docker compose up -d
```

### Issue 7: Memory Database Corrupted

**Symptoms:** Memory search returns errors

**Fix:**
```bash
# Backup current database
cp ~/llm-storage/memory/benchai_memory.db ~/llm-storage/memory/benchai_memory.db.bak

# Try to optimize/repair
curl -X POST http://localhost:8085/v1/memory/optimize

# If still broken, check integrity
sqlite3 ~/llm-storage/memory/benchai_memory.db "PRAGMA integrity_check;"

# Last resort: delete and let it recreate
# WARNING: Loses all memories
rm ~/llm-storage/memory/benchai_memory.db
sudo systemctl restart benchai
```

---

## Scheduled Maintenance

### Weekly (5 minutes)

```bash
# 1. Optimize memory database
curl -X POST http://localhost:8085/v1/memory/optimize
echo "Memory optimized"

# 2. Check disk space
df -h /home/user | tail -1

# 3. Check log sizes
du -sh /tmp/benchai-router.log /var/log/benchai-router.log 2>/dev/null

# 4. Rotate logs if needed (>100MB)
LOG=/var/log/benchai-router.log
if [ $(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG") -gt 100000000 ]; then
  sudo mv $LOG ${LOG}.old
  sudo systemctl restart benchai
fi
```

### Monthly (15 minutes)

```bash
# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Update Docker containers
cd ~/docker-server
docker compose pull
docker compose up -d

# 3. Check llama.cpp updates
cd ~/llama.cpp
git fetch
git log HEAD..origin/master --oneline | head -5
# If updates available, rebuild:
# git pull && mkdir -p build && cd build && cmake .. -DGGML_CUDA=ON && cmake --build . -j

# 4. Check model updates
# Visit: https://huggingface.co/search?q=gguf
# Download newer quantizations if available

# 5. Full system restart
sudo systemctl restart benchai
docker restart $(docker ps -q)
```

---

## Backup Procedures

### Quick Backup (Critical Data Only)

```bash
BACKUP_DIR=~/backups/benchai-$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Memory database
cp ~/llm-storage/memory/benchai_memory.db $BACKUP_DIR/

# Configuration
cp ~/benchai/router/.env $BACKUP_DIR/ 2>/dev/null
cp -r ~/benchai-client/configs $BACKUP_DIR/

# Neovim config
cp ~/.config/nvim/lua/plugins/benchai.lua $BACKUP_DIR/ 2>/dev/null

echo "Backup saved to: $BACKUP_DIR"
ls -la $BACKUP_DIR
```

### Full Backup (Including RAG)

```bash
BACKUP_DIR=~/backups/benchai-full-$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# All data
cp -r ~/llm-storage/memory $BACKUP_DIR/
cp -r ~/llm-storage/rag $BACKUP_DIR/

# All configs
cp -r ~/benchai $BACKUP_DIR/
cp -r ~/benchai-client $BACKUP_DIR/

# Compress
tar -czf $BACKUP_DIR.tar.gz -C ~/backups $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

echo "Backup: $BACKUP_DIR.tar.gz"
ls -lh $BACKUP_DIR.tar.gz
```

### Restore Procedure

```bash
BACKUP_FILE=~/backups/benchai-full-YYYYMMDD.tar.gz

# Stop services
sudo systemctl stop benchai

# Extract
tar -xzf $BACKUP_FILE -C ~/backups/
BACKUP_DIR=${BACKUP_FILE%.tar.gz}

# Restore memory
cp $BACKUP_DIR/memory/benchai_memory.db ~/llm-storage/memory/

# Restore RAG (optional - can be re-indexed)
cp -r $BACKUP_DIR/rag/* ~/llm-storage/rag/

# Restart
sudo systemctl start benchai
```

---

## Performance Tuning

### GPU Optimization

```bash
# Check current GPU layers
ps aux | grep llama-server | grep -v grep | grep -o '\-ngl [0-9]*'

# For code model (DeepSeek), ensure full GPU offload:
# -ngl 35 uses ~8GB VRAM

# If VRAM constrained, reduce context:
# -c 4096 instead of -c 8192 saves ~1GB
```

### CPU Optimization

```bash
# Check thread allocation
ps aux | grep llama-server | grep -v grep | grep -o '\-t [0-9]*'

# Recommended:
# GPU mode: -t 8 (less threads = more GPU work)
# CPU mode: -t 12 (more threads = faster)
```

### Memory Optimization

```bash
# Check memory usage
free -h

# If RAM constrained:
# 1. Reduce max concurrent requests in llm_router.py
# 2. Lower context window (-c 2048)
# 3. Use smaller models (Phi-3 instead of Qwen)
```

---

## Emergency Procedures

### Complete System Restart

```bash
# 1. Stop everything
sudo systemctl stop benchai
pkill -9 -f llama-server
docker stop $(docker ps -q)

# 2. Clear GPU
sudo fuser -k /dev/nvidia*

# 3. Wait
sleep 10

# 4. Start in order
docker start $(docker ps -aq)
sleep 5
sudo systemctl start benchai
sleep 30

# 5. Verify
curl http://localhost:8085/health
```

### Rollback to Previous State

```bash
# 1. Stop current
sudo systemctl stop benchai

# 2. Restore from backup
cp ~/backups/benchai-YYYYMMDD/benchai_memory.db ~/llm-storage/memory/

# 3. Restart
sudo systemctl start benchai
```

### Factory Reset

```bash
# WARNING: Deletes all data!

# 1. Stop services
sudo systemctl stop benchai
pkill -9 -f llama-server

# 2. Remove data
rm -rf ~/llm-storage/memory/*
rm -rf ~/llm-storage/rag/*
rm -rf ~/llm-storage/cache/*

# 3. Restart (will recreate empty databases)
sudo systemctl start benchai
```

---

## Monitoring Commands

### Real-time Monitoring

```bash
# Watch router logs
tail -f /tmp/benchai-router.log

# Watch GPU usage
watch -n 1 nvidia-smi

# Watch system resources
htop

# Watch Docker logs
docker logs -f open-webui
```

### Generate Status Report

```bash
echo "=== BenchAI Status Report ==="
echo "Generated: $(date)"
echo ""
echo "=== Router Health ==="
curl -s http://localhost:8085/health | python3 -m json.tool
echo ""
echo "=== Model Status ==="
ss -tlnp | grep -E "8091|8092|8093|8094"
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo ""
echo "=== Memory Stats ==="
curl -s http://localhost:8085/v1/memory/stats | python3 -m json.tool
echo ""
echo "=== RAG Stats ==="
curl -s http://localhost:8085/v1/rag/stats
echo ""
echo "=== Docker Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}"
echo ""
echo "=== Disk Space ==="
df -h /home/user | tail -1
```

---

## Contact & Escalation

### Self-Help Resources
1. Check `/tmp/benchai-router.log` for errors
2. Review `docs/ARCHITECTURE.md` for system design
3. Check `docs/TROUBLESHOOTING.md` for known issues

### GitHub Issues
- BenchAI Server: https://github.com/YOUR_USERNAME/benchai/issues
- BenchAI Client: https://github.com/YOUR_USERNAME/benchai-client/issues

---

*Runbook version 1.0 - December 25, 2025*
