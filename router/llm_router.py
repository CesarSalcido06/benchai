#!/usr/bin/env python3
"""
BenchAI - Intelligent LLM Orchestration Platform v3
Full-featured AI orchestrator with streaming, memory, TTS, RAG, MCP, and parallel execution.

Features:
- Agentic Planner workflow with parallel tool execution
- VRAM cold-swapping for GPU management
- Streaming responses (SSE)
- Persistent memory (SQLite)
- Text-to-Speech (Piper)
- RAG pipeline (ChromaDB)
- Obsidian integration via Local REST API
- Tools: Web Search, Vision, Image Gen, Code Execution, File Reading, Shell Commands, Obsidian
"""

import asyncio
import subprocess
import time
import re
import os
import signal
import json
import random
import tempfile
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from pathlib import Path
from urllib.parse import quote
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
import httpx
import aiosqlite

# Optional imports with fallbacks
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[WARN] ChromaDB not available - RAG disabled")

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("[WARN] Piper not available - TTS disabled")

# Learning System Integration
try:
    from learning_integration import (
        setup_learning_system, shutdown_learning_system, get_learning_system,
        learning_router, log_chat_completion, get_experience_context,
        remember_important, recall_relevant, record_task_outcome,
        notify_agent_online, notify_agent_offline, share_with_agents
    )
    LEARNING_AVAILABLE = True
except ImportError as e:
    LEARNING_AVAILABLE = False
    print(f"[WARN] Learning system not available - {e}")

# Load environment variables from .env file
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

# Obsidian REST API Configuration
OBSIDIAN_API_KEY = os.environ.get("OBSIDIAN_API_KEY", "")
OBSIDIAN_API_URL = os.environ.get("OBSIDIAN_API_URL", "https://localhost:27124")
OBSIDIAN_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", "")
OBSIDIAN_AVAILABLE = bool(OBSIDIAN_API_KEY)

if OBSIDIAN_AVAILABLE:
    print(f"[OBSIDIAN] API configured at {OBSIDIAN_API_URL}")
else:
    print("[WARN] Obsidian API key not set - Obsidian integration disabled")

# --- CONFIGURATION ---
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
MODELS_DIR = LLAMA_CPP_DIR / "models"
LLAMA_SERVER = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
ROUTER_DIR = LLAMA_CPP_DIR / "router"

# Database paths - LLM storage on dedicated SSD (SDA)
LLM_STORAGE = Path.home() / "llm-storage"
LLM_STORAGE.mkdir(parents=True, exist_ok=True)
DB_PATH = LLM_STORAGE / "memory" / "benchai_memory.db"
CHROMA_PATH = LLM_STORAGE / "rag" / "chroma_db"
VOICES_DIR = ROUTER_DIR / "voices"
AUDIO_CACHE_DIR = LLM_STORAGE / "cache" / "audio"
# Ensure directories exist
(LLM_STORAGE / "memory").mkdir(parents=True, exist_ok=True)
(LLM_STORAGE / "rag").mkdir(parents=True, exist_ok=True)
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

COMFY_DIR = Path.home() / "ComfyUI"
COMFY_WORKFLOW_FILE = COMFY_DIR / "flux_api.json"
COMFY_CMD = ["python3", str(COMFY_DIR / "main.py"), "--listen", "127.0.0.1", "--port", "8188"]

SEARXNG_URL = "http://127.0.0.1:8081"

# Allowed shell commands (safety whitelist)
ALLOWED_SHELL_COMMANDS = [
    "ls", "pwd", "cat", "head", "tail", "wc", "grep", "find", "which", "whoami",
    "date", "uptime", "df", "du", "free", "ps", "top", "htop", "nvidia-smi",
    "git", "docker", "python3", "pip", "node", "npm", "cargo", "rustc", "go",
    "curl", "wget", "ping", "dig", "nslookup", "ifconfig", "ip",
    "tree", "file", "stat", "md5sum", "sha256sum", "base64", "jq", "yq",
    "tar", "zip", "unzip", "gzip", "gunzip", "xz",
    "sort", "uniq", "cut", "awk", "sed", "tr", "diff", "comm",
    "echo", "printf", "env", "printenv", "uname", "hostname", "lsb_release",
    # C/C++ development tools
    "gcc", "g++", "clang", "clang++", "make", "cmake", "ninja",
    "gdb", "lldb", "valgrind", "strace", "ltrace",
    "objdump", "nm", "readelf", "ldd", "size", "strings",
    "clang-tidy", "clang-format", "cppcheck", "cpplint",
    "ar", "ranlib", "strip", "addr2line", "c++filt",
    # CUDA tools
    "nvcc", "cuda-gdb", "nvprof", "nsys", "ncu", "cuda-memcheck",
    # Web dev / JS/TS tools
    "bun", "deno", "npx", "yarn", "pnpm", "tsc", "tsx", "esbuild", "vite",
    "eslint", "prettier", "jest", "vitest", "playwright", "cypress"
]

# --- REQUEST CACHE ---
# Cache identical requests to avoid redundant LLM calls (from deep research)
class RequestCache:
    """TTL-based cache for LLM responses to reduce duplicate work."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self.cache: Dict[str, tuple] = {}  # key -> (response, timestamp)
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, model: str, messages: List[Dict], max_tokens: int) -> str:
        """Create cache key from request parameters."""
        content = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, model: str, messages: List[Dict], max_tokens: int) -> Optional[Dict]:
        """Get cached response if exists and not expired."""
        key = self._make_key(model, messages, max_tokens)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                print(f"[CACHE] HIT - returning cached response (hits={self.hits})")
                return response
            else:
                del self.cache[key]  # Expired
        self.misses += 1
        return None

    def set(self, model: str, messages: List[Dict], max_tokens: int, response: Dict):
        """Cache a response."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        key = self._make_key(model, messages, max_tokens)
        self.cache[key] = (response, time.time())

    def stats(self) -> Dict:
        """Return cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hits / max(1, self.hits + self.misses) * 100:.1f}%",
            "ttl_seconds": self.ttl
        }

# Global cache instance
request_cache = RequestCache(ttl_seconds=300, max_size=100)

# Model Definitions
# --- MODULAR MODEL REGISTRY ---
# Easy to add/swap models. Each model has:
# - role: What it's used for (planner, code, vision, etc.)
# - capabilities: What it can do (reasoning, coding, vision, math, etc.)
# - quality: Relative quality for routing (1-10)
# - speed: Relative speed (1-10, higher = faster)

MODELS = {
    "general": {
        "name": "Phi-3 Mini",
        "file": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "port": 8091,
        "gpu_layers": 35,
        "context": 4096,
        "description": "Fast responses for simple questions",
        "role": "assistant",
        "capabilities": ["chat", "reasoning"],
        "quality": 6,
        "speed": 9
    },
    "planner": {
        "name": "Qwen2.5 7B (Planner)",
        "file": "qwen2.5-7b-instruct.Q5_K_M.gguf",  # Upgraded from Q4_K_M for better quality
        "port": 8092,
        "gpu_layers": 35,
        "context": 4096,  # Reduced from 8192 for CPU stability
        "description": "Orchestrates multi-step tasks",
        "role": "orchestrator",
        "capabilities": ["planning", "reasoning", "tool_selection"],
        "quality": 8,
        "speed": 7
    },
    "research": {
        "name": "Qwen2.5 7B",
        "file": "qwen2.5-7b-instruct.Q5_K_M.gguf",  # Upgraded from Q4_K_M for better quality
        "port": 8092,
        "gpu_layers": 35,
        "context": 4096,  # Reduced from 8192 for CPU stability
        "description": "Deep analysis and reasoning",
        "role": "researcher",
        "capabilities": ["reasoning", "analysis", "synthesis", "explanation"],
        "quality": 8,
        "speed": 7
    },
    "code": {
        "name": "DeepSeek Coder 6.7B",
        "file": "deepseek-coder-6.7b-instruct.Q5_K_M.gguf",  # Upgraded from Q4_K_M for better code quality
        "port": 8093,
        "gpu_layers": 35,
        "context": 8192,
        "description": "Programming, math, and code assistance",
        "role": "coder",
        "capabilities": ["coding", "debugging", "code_review", "testing", "math"],
        "quality": 8,
        "speed": 7
    },
    "vision": {
        "name": "Qwen2-VL 7B",
        "file": "Qwen2-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
        "port": 8094,
        "gpu_layers": 35,
        "context": 8192,
        "description": "Vision analysis and OCR",
        "role": "vision",
        "capabilities": ["vision", "ocr", "image_analysis"],
        "quality": 8,
        "speed": 6
    }
}

# Claude is the "mastermind" - escalate to it for:
CLAUDE_ESCALATION_TRIGGERS = [
    "complex architecture",
    "code review",
    "best practice",
    "design pattern",
    r"explain.*(concept|what is|how does|to me|like i)",  # Explain concepts
    "why.*not working",
    "debug.*issue",
    "help.*understand",
    "teach.*how",
    "compare.*approaches",
    r"what is.*(recursion|polymorphism|inheritance|closure|callback|async|await|promise)",  # Programming concepts
    r"how (do|does|can|should|would)"  # Explanation requests
]

IDLE_TIMEOUT = 600  # 10 minutes

# --- GPU CONFIGURATION ---
# RTX 3060 has 12GB VRAM - optimize allocation:
# - DeepSeek Coder: GPU (primary coding model, ~4GB)
# - Phi-3 Mini: CPU (fast, small, always available)
# - Qwen2.5: CPU (planner needs to always be available)
# - Vision: On-demand GPU (swap when needed)
GPU_VRAM_THRESHOLD = 4000  # Minimum free VRAM (MiB) needed for GPU mode
CPU_MODELS = ["planner", "general", "research"]  # Models that always run on CPU for availability
GPU_PREFERRED = ["code"]  # Models that prefer GPU when available
GPU_ONLY_MODELS = ["vision"]  # Models that require GPU (can't run on CPU effectively)

# Core models that should always be running
CORE_MODELS = ["general", "planner", "code"]  # Essential models to start on boot
HEALTH_CHECK_INTERVAL = 60  # Seconds between health checks

# --- REQUEST CONCURRENCY LIMITING ---
# Limit concurrent requests per model to prevent severe queueing
# llama-server processes requests sequentially, so limiting prevents long queues
MAX_CONCURRENT_REQUESTS = {
    "general": 2,   # Fast model, can handle slight queueing
    "planner": 2,   # Critical for orchestration
    "research": 1,  # Slower model, avoid queueing
    "code": 1,      # CPU-heavy, no queueing to prevent blocking
    "vision": 1,    # GPU-heavy, one at a time
}
# Semaphores are created lazily when first request comes in
_model_semaphores: Dict[str, asyncio.Semaphore] = {}

def get_model_semaphore(model_type: str) -> asyncio.Semaphore:
    """Get or create semaphore for a model type."""
    if model_type not in _model_semaphores:
        max_concurrent = MAX_CONCURRENT_REQUESTS.get(model_type, 1)
        _model_semaphores[model_type] = asyncio.Semaphore(max_concurrent)
    return _model_semaphores[model_type]

def check_gpu_available(min_free_mb: int = GPU_VRAM_THRESHOLD) -> tuple[bool, int, int]:
    """
    Check if GPU has enough free VRAM for a model.
    Returns: (is_available, used_mb, total_mb)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            used = int(parts[0])
            total = int(parts[1])
            free = total - used
            return (free >= min_free_mb, used, total)
    except Exception as e:
        print(f"[GPU] Check failed: {e}")
    return (False, 0, 0)

def find_gpu_hogs() -> List[tuple]:
    """Find processes using significant GPU memory (>1GB) that can be killed/restarted."""
    hogs = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        pid = int(parts[0])
                        mem_mb = int(parts[1].replace(' MiB', ''))
                        name = parts[2]
                        # Only consider processes using >1GB
                        if mem_mb > 1000:
                            # Check process cmdline for known restartable apps
                            try:
                                with open(f"/proc/{pid}/cmdline", "r") as f:
                                    cmdline = f.read()
                                    cmdline_lower = cmdline.lower()
                                    restartable_apps = ['comfyui', 'automatic1111', 'stable-diffusion', 'kohya', 'webui', 'invoke']
                                    if any(app in cmdline_lower for app in restartable_apps):
                                        # Store full cmdline for restart
                                        cmd_parts = [p for p in cmdline.split('\x00') if p]
                                        hogs.append((pid, mem_mb, cmd_parts))
                            except:
                                pass
    except Exception as e:
        print(f"[GPU] Error finding GPU hogs: {e}")
    return hogs

def stop_gpu_apps_for_vision() -> List[dict]:
    """Kill GPU-heavy apps to free VRAM. Returns info needed to restart them."""
    stopped = []
    hogs = find_gpu_hogs()
    for pid, mem, cmd_parts in hogs:
        try:
            # Get working directory before killing
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
            except:
                cwd = "/home/user"

            app_info = {
                "pid": pid,
                "cmd": cmd_parts,
                "cwd": cwd,
                "mem": mem
            }

            # Kill the process
            os.kill(pid, signal.SIGTERM)
            print(f"[GPU] Stopped {cmd_parts[0] if cmd_parts else 'app'} (PID {pid}, {mem} MiB) to free VRAM")
            stopped.append(app_info)

        except Exception as e:
            print(f"[GPU] Failed to stop PID {pid}: {e}")

    if stopped:
        time.sleep(3)  # Wait for VRAM to be released
    return stopped

def restart_gpu_apps(stopped_apps: List[dict]):
    """Restart previously stopped GPU apps."""
    for app in stopped_apps:
        try:
            cmd = app["cmd"]
            cwd = app["cwd"]

            # Start the process in background
            print(f"[GPU] Restarting {cmd[0]}...")
            subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            print(f"[GPU] Restarted {cmd[0]}")
        except Exception as e:
            print(f"[GPU] Failed to restart {app.get('cmd', ['unknown'])[0]}: {e}")

# --- DETECTION PATTERNS ---

CODE_PATTERNS = re.compile(
    r'\b(code|coding|program|programming|function|func|def|class|method|'
    r'script|python|javascript|typescript|java|rust|golang|go|cpp|c\+\+|'
    r'bash|shell|zsh|powershell|api|debug|debugging|error|bug|bugfix|'
    r'compile|compiler|syntax|algorithm|variable|array|list|dict|loop|'
    r'regex|regexp|sql|query|html|css|json|xml|yaml|git|github|'
    r'docker|kubernetes|k8s|npm|pip|cargo|maven|gradle|webpack|vite|'
    r'implement|implementation|refactor|optimize|optimization|'
    r'unittest|test|testing|pytest|jest|mocha|'
    r'import|export|require|module|package|library|framework|'
    r'async|await|promise|callback|thread|process|'
    r'react|vue|angular|svelte|next|nuxt|express|fastapi|flask|django|'
    r'database|mongodb|postgres|mysql|redis|sqlite|'
    r'write.*code|code.*write|fix.*code|code.*fix|'
    r'create.*function|function.*create|build.*app|app.*build)\b',
    re.IGNORECASE
)

MATH_PATTERNS = re.compile(
    r'\b(math|calculate|calculation|compute|computation|solve|equation|'
    r'formula|algebra|calculus|derivative|integral|matrix|vector|'
    r'statistics|probability|mean|median|average|sum|product|'
    r'factorial|fibonacci|prime|sqrt|square root|logarithm|log|'
    r'trigonometry|sin|cos|tan|geometry|area|volume|perimeter|'
    r'arithmetic|addition|subtraction|multiplication|division|'
    r'percentage|ratio|proportion|fraction|decimal|binary|hex|'
    r'what is \d|how much|how many|\d+\s*[\+\-\*\/\^]\s*\d+)\b',
    re.IGNORECASE
)

IMAGE_GEN_PATTERNS = re.compile(
    r'\b(generate|create|make|draw|render|visualize|design|sketch|paint|illustrate)\s+'
    r'(an?\s+)?(image|picture|photo|drawing|illustration|diagram|chart|graph|'
    r'schematic|blueprint|sketch|design|concept|art|artwork|portrait|landscape|logo|icon)\b',
    re.IGNORECASE
)

VISION_PATTERNS = re.compile(
    r'\b(analyze|describe|explain|read|extract|ocr|transcribe|identify|recognize|'
    r'what is in|what do you see|look at|examine|inspect)\s*(this|the|my)?\s*'
    r'(image|picture|photo|screenshot|diagram|chart|document|pdf|scan|file)?\b',
    re.IGNORECASE
)

RESEARCH_PATTERNS = re.compile(
    r'\b(research|analyze|analysis|explain|compare|contrast|evaluate|'
    r'study|investigate|examine|review|assess|summarize|synthesis|'
    r'theory|hypothesis|evidence|conclusion|methodology|'
    r'comprehensive|detailed|in-depth|thorough|pros and cons|advantages|disadvantages)\b',
    re.IGNORECASE
)

MEMORY_PATTERNS = re.compile(
    r'\b(remember|recall|forgot|forget|what did I|previously|last time|'
    r'you told me|I told you|my name is|I prefer|I like|I hate|'
    r'save this|note this|keep in mind)\b',
    re.IGNORECASE
)

META_PATTERNS = re.compile(
    r'^(Suggest\s+\d+|Generate\s+a\s+(brief\s+)?title|Create\s+tags|Provide\s+follow-up)',
    re.IGNORECASE
)

# --- DATA MODELS ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024  # Reduced from 2048 for faster responses
    stream: Optional[bool] = False

class ImageRequest(BaseModel):
    prompt: str

class MemoryRequest(BaseModel):
    content: str
    category: Optional[str] = "general"

class TTSRequest(BaseModel):
    text: str

class IndexRequest(BaseModel):
    path: str
    recursive: Optional[bool] = True

# --- MEMORY MANAGER (SQLite with Full Optimization) ---

class MemoryManager:
    """
    Optimized SQLite memory manager with:
    - WAL mode for concurrent access
    - FTS5 full-text search for fast queries
    - Proper indexes on frequently queried columns
    - Memory-mapped I/O for fast access
    - Optimized cache settings for 48GB RAM system
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False
        self._connection_pool = []  # Simple connection reuse

    async def _optimize_connection(self, db):
        """Apply performance optimizations to SQLite connection."""
        # WAL mode for concurrent read/write
        await db.execute("PRAGMA journal_mode = WAL")
        # Sync less often (safe with WAL)
        await db.execute("PRAGMA synchronous = NORMAL")
        # 80MB cache (10000 pages * 8KB)
        await db.execute("PRAGMA cache_size = -80000")
        # Use memory for temp tables
        await db.execute("PRAGMA temp_store = MEMORY")
        # Memory-mapped I/O up to 1GB
        await db.execute("PRAGMA mmap_size = 1073741824")
        # Auto-checkpoint at 1000 pages
        await db.execute("PRAGMA wal_autocheckpoint = 1000")
        # Enable foreign keys
        await db.execute("PRAGMA foreign_keys = ON")
        # Page size 8KB (optimal for modern SSDs)
        await db.execute("PRAGMA page_size = 8192")

    async def initialize(self):
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
            # Apply optimizations first
            await self._optimize_connection(db)

            # Main memories table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    importance INTEGER DEFAULT 1
                )
            ''')

            # Create indexes for faster queries
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_category
                ON memories(category)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_category_importance
                ON memories(category, importance DESC, created_at DESC)
            ''')

            # FTS5 virtual table for full-text search
            await db.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    category,
                    tokenize='porter unicode61'
                )
            ''')

            # Triggers to keep FTS in sync
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category);
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.id;
                    INSERT INTO memories_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category);
                END
            ''')

            await db.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_created
                ON conversations(created_at DESC)
            ''')

            await db.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Run ANALYZE for query optimizer
            await db.execute("ANALYZE")

            await db.commit()
        self._initialized = True
        print("[MEMORY] Database initialized (WAL mode, FTS5, indexed)")

    async def add_memory(self, content: str, category: str = "general", importance: int = 1):
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO memories (content, category, importance) VALUES (?, ?, ?)",
                (content, category, importance)
            )
            await db.commit()

    async def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories using FTS5 full-text search with ranking."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            # Use FTS5 for fast, ranked full-text search
            # bm25() provides relevance ranking
            try:
                cursor = await db.execute('''
                    SELECT m.*, bm25(memories_fts) as rank
                    FROM memories m
                    JOIN memories_fts ON m.id = memories_fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank, m.importance DESC, m.created_at DESC
                    LIMIT ?
                ''', (query, limit))
                rows = await cursor.fetchall()
                if rows:
                    return [dict(row) for row in rows]
            except Exception:
                pass  # Fall back to LIKE search if FTS fails

            # Fallback: keyword search for partial matches
            keywords = query.lower().split()
            conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]

            cursor = await db.execute(
                f"SELECT * FROM memories WHERE {conditions} ORDER BY importance DESC, created_at DESC LIMIT ?",
                params + [limit]
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_recent_memories(self, limit: int = 10) -> List[Dict]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def set_preference(self, key: str, value: str):
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.now())
            )
            await db.commit()

    async def get_preference(self, key: str) -> Optional[str]:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM user_preferences WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def get_context_summary(self) -> str:
        """Get a summary of relevant memories for context injection."""
        await self.initialize()
        memories = await self.get_recent_memories(5)
        if not memories:
            return ""

        summary_parts = ["[User Context from Memory]"]
        for mem in memories:
            summary_parts.append(f"- {mem['content']}")
        return "\n".join(summary_parts)

    async def log_conversation(self, role: str, content: str):
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO conversations (role, content) VALUES (?, ?)",
                (role, content[:2000])  # Limit size
            )
            # Keep only last 100 conversations
            await db.execute(
                "DELETE FROM conversations WHERE id NOT IN (SELECT id FROM conversations ORDER BY created_at DESC LIMIT 100)"
            )
            await db.commit()

memory_manager = MemoryManager(DB_PATH)

# --- RAG MANAGER (ChromaDB) ---

class RAGManager:
    """ChromaDB-based RAG manager with periodic refresh to prevent stale data."""

    REFRESH_INTERVAL = 300  # Refresh collection every 5 minutes

    def __init__(self, persist_path: Path):
        self.persist_path = persist_path
        self.client = None
        self.collection = None
        self._last_refresh = 0

    async def initialize(self):
        if not CHROMADB_AVAILABLE:
            return False

        # Refresh collection if stale (prevents stale data issue from critical analysis)
        now = time.time()
        if self.client is not None and (now - self._last_refresh) > self.REFRESH_INTERVAL:
            print("[RAG] Refreshing collection to prevent stale data...")
            self.collection = self.client.get_collection("benchai_knowledge")
            self._last_refresh = now

        if self.client is None:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_path))
            # Optimized HNSW settings for RTX 3060 / 48GB RAM system
            self.collection = self.client.get_or_create_collection(
                name="benchai_knowledge",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,  # Higher = better index quality
                    "hnsw:M": 32,                  # Max edges per node (more = better recall)
                    "hnsw:search_ef": 100,         # Higher = more accurate but slower queries
                    "hnsw:batch_size": 10000,      # Batch size before flushing
                }
            )
            self._last_refresh = now
            print(f"[RAG] ChromaDB initialized with {self.collection.count()} documents (HNSW optimized)")
        return True

    async def add_document(self, content: str, metadata: Dict = None, doc_id: str = None):
        if not await self.initialize():
            return False

        if doc_id is None:
            doc_id = hashlib.md5(content.encode()).hexdigest()

        # Check if document already exists
        existing = self.collection.get(ids=[doc_id])
        if existing['ids']:
            return False  # Already indexed

        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        return True

    async def search(self, query: str, n_results: int = 3) -> List[Dict]:
        if not await self.initialize():
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        docs = []
        for i, doc in enumerate(results['documents'][0]):
            docs.append({
                'content': doc,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0
            })
        return docs

    async def index_file(self, file_path: Path) -> bool:
        if not file_path.exists():
            return False

        try:
            content = file_path.read_text(errors='replace')

            # Chunk large files
            chunks = self._chunk_text(content, chunk_size=1000, overlap=100)

            for i, chunk in enumerate(chunks):
                await self.add_document(
                    content=chunk,
                    metadata={
                        'source': str(file_path),
                        'chunk': i,
                        'total_chunks': len(chunks)
                    },
                    doc_id=f"{file_path}_{i}"
                )

            return True
        except Exception as e:
            print(f"[RAG] Error indexing {file_path}: {e}")
            return False

    async def index_directory(self, dir_path: Path, extensions: List[str] = None):
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.md', '.txt', '.json', '.yaml', '.yml', '.sh']

        indexed = 0
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                if await self.index_file(file_path):
                    indexed += 1

        return indexed

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def get_stats(self) -> Dict:
        if self.collection is None:
            return {"status": "not initialized", "count": 0}
        return {"status": "ready", "count": self.collection.count()}

rag_manager = RAGManager(CHROMA_PATH)

# --- TTS MANAGER (Piper) ---

class TTSManager:
    def __init__(self, voices_dir: Path, cache_dir: Path):
        self.voices_dir = voices_dir
        self.cache_dir = cache_dir
        self.voice = None
        self._initialized = False

    async def initialize(self):
        if not PIPER_AVAILABLE:
            return False

        if self._initialized:
            return True

        voice_file = self.voices_dir / "en_US-lessac-medium.onnx"
        if not voice_file.exists():
            print(f"[TTS] Voice file not found: {voice_file}")
            return False

        try:
            self.voice = PiperVoice.load(str(voice_file))
            self._initialized = True
            print("[TTS] Piper voice loaded")
            return True
        except Exception as e:
            print(f"[TTS] Error loading voice: {e}")
            return False

    async def synthesize(self, text: str) -> Optional[Path]:
        if not await self.initialize():
            return None

        # Create cache key from text
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{text_hash}.wav"

        if cache_file.exists():
            return cache_file

        try:
            # Run synthesis in thread pool to not block async
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._synthesize_sync,
                text,
                cache_file
            )
            return cache_file
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    def _synthesize_sync(self, text: str, output_path: Path):
        import wave

        with wave.open(str(output_path), 'wb') as wav_file:
            self.voice.synthesize_wav(text, wav_file)

tts_manager = TTSManager(VOICES_DIR, AUDIO_CACHE_DIR)

# --- SESSION MANAGER (Conversation Context) ---

class SessionManager:
    """Manages conversation context per session for multi-turn workflows.

    Solves the problem where follow-up requests lack context from previous
    tool executions (e.g., vision analysis results not available on follow-up).
    """

    def __init__(self, max_history: int = 10, session_ttl: int = 3600):
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
        self.session_ttl = session_ttl  # 1 hour default

    def _get_session_id(self, messages: List[Dict]) -> str:
        """Generate session ID from conversation - use hash of ALL messages for uniqueness."""
        # For multi-turn conversations, hash ALL messages to avoid session collision
        # Single-turn requests (IDE) will get unique sessions per request
        if messages and len(messages) > 0:
            # Collect all message content for unique hash
            all_content = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multimodal content
                    text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    content = " ".join(text_parts)
                all_content.append(f"{role}:{content}")

            # Hash the full conversation for unique session ID
            full_hash = hashlib.sha256("|||".join(all_content).encode()).hexdigest()[:16]
            return full_hash
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    def get_or_create_session(self, messages: List[Dict]) -> tuple:
        """Get existing session or create new one. Returns (session_id, session_data)."""
        session_id = self._get_session_id(messages)

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created": time.time(),
                "last_access": time.time(),
                "context": "",  # Accumulated tool results
                "history": [],  # Conversation turns
                "tool_results": {},  # Named results (vision_analysis, code_task, etc.)
                "workflow_phase": "idle"  # idle, planning, executing, waiting
            }

        session = self.sessions[session_id]
        session["last_access"] = time.time()

        # Cleanup old sessions
        self._cleanup_expired()

        return session_id, session

    def add_tool_result(self, session_id: str, tool_name: str, result: str):
        """Store a tool execution result in the session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session["tool_results"][tool_name] = result

            # Also append to running context
            summary = result[:500] + "..." if len(result) > 500 else result
            session["context"] += f"\n\n**Previous {tool_name}:**\n{summary}"

            print(f"[SESSION] Stored {tool_name} result ({len(result)} chars)")

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn to history."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session["history"].append({"role": role, "content": content[:1000]})

            # Keep only recent history
            if len(session["history"]) > self.max_history * 2:
                session["history"] = session["history"][-self.max_history:]

    def get_context_for_planner(self, session_id: str) -> str:
        """Get accumulated context for the planner prompt."""
        if session_id not in self.sessions:
            return ""

        session = self.sessions[session_id]
        context_parts = []

        # Add tool results
        if session["tool_results"]:
            context_parts.append("**Previous Tool Results in This Conversation:**")
            for tool, result in session["tool_results"].items():
                summary = result[:300] + "..." if len(result) > 300 else result
                context_parts.append(f"- {tool}: {summary}")

        # Add conversation history summary
        if session["history"]:
            context_parts.append("\n**Recent Conversation:**")
            for turn in session["history"][-4:]:  # Last 4 turns
                role = turn["role"].upper()
                content = turn["content"][:200]
                context_parts.append(f"- {role}: {content}")

        return "\n".join(context_parts)

    def set_phase(self, session_id: str, phase: str):
        """Set workflow phase: idle, planning, executing, waiting."""
        if session_id in self.sessions:
            self.sessions[session_id]["workflow_phase"] = phase

    def get_phase(self, session_id: str) -> str:
        """Get current workflow phase."""
        if session_id in self.sessions:
            return self.sessions[session_id].get("workflow_phase", "idle")
        return "idle"

    def clear_session(self, session_id: str):
        """Clear a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["last_access"] > self.session_ttl
        ]
        for sid in expired:
            del self.sessions[sid]
            print(f"[SESSION] Expired session {sid}")

session_manager = SessionManager()

# --- MODEL MANAGER ---

class ModelManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.last_used: Dict[str, float] = {}
        self.comfy_process: Optional[subprocess.Popen] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._model_lock = asyncio.Lock()  # Prevent race conditions
        self._current_operation: Optional[str] = None  # Track what's happening
        # Kill orphaned llama-server processes on startup
        self._cleanup_orphans()

    def _cleanup_orphans(self):
        """Kill any orphaned llama-server processes from previous router instances."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"[CLEANUP] Found {len(pids)} orphaned llama-server processes, killing...")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except:
                        pass
                time.sleep(2)
                print("[CLEANUP] Orphaned processes killed")
        except Exception as e:
            print(f"[CLEANUP] Error: {e}")

    async def start_cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            await self._cleanup_idle_models()

    async def _cleanup_idle_models(self):
        current_time = time.time()
        to_unload = []
        for model_type, last_time in self.last_used.items():
            if current_time - last_time > IDLE_TIMEOUT:
                # Never unload core models
                if model_type in CORE_MODELS:
                    continue
                if model_type in self.processes:
                    to_unload.append(model_type)
        for model_type in to_unload:
            print(f"[CLEANUP] Auto-unloading idle model: {model_type}")
            await self.stop_model(model_type)

    async def ensure_core_models(self):
        """
        Ensure all core models are running on startup.
        This prevents the 'empty response from model' error.
        """
        print("[STARTUP] Ensuring core models are running...")

        for model_type in CORE_MODELS:
            if model_type not in MODELS:
                print(f"[WARN] Core model '{model_type}' not in MODELS config")
                continue

            config = MODELS[model_type]
            port = config["port"]

            # Check if model is already running
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                    if resp.status_code == 200:
                        print(f"[STARTUP] {model_type} already running on port {port}")
                        self.last_used[model_type] = time.time()
                        continue
            except:
                pass

            # Start the model
            print(f"[STARTUP] Starting core model: {model_type}")
            success = await self.start_model(model_type)
            if success:
                print(f"[STARTUP] {model_type} started successfully")
            else:
                print(f"[ERROR] Failed to start core model: {model_type}")

        print("[STARTUP] Core models check complete")

    async def health_monitor_loop(self):
        """
        Background task to monitor model health and restart crashed models.
        Runs every HEALTH_CHECK_INTERVAL seconds.
        """
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            await self._check_and_restart_models()

    async def _check_and_restart_models(self):
        """Check health of core models and restart any that have crashed."""
        for model_type in CORE_MODELS:
            if model_type not in MODELS:
                continue

            config = MODELS[model_type]
            port = config["port"]

            # Check if model responds to health check
            is_healthy = False
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/health", timeout=3.0)
                    is_healthy = resp.status_code == 200
            except:
                pass

            if not is_healthy:
                # Check if process is still running but not responding
                exit_reason = ""
                if model_type in self.processes:
                    proc = self.processes[model_type]
                    if proc.poll() is not None:
                        exit_code = proc.returncode
                        # Decode exit code for better diagnostics
                        if exit_code == -9:
                            exit_reason = "SIGKILL (OOM or forced kill)"
                        elif exit_code == -11:
                            exit_reason = "SIGSEGV (memory corruption)"
                        elif exit_code == -6:
                            exit_reason = "SIGABRT (assertion failed)"
                        else:
                            exit_reason = f"code {exit_code}"
                        print(f"[HEALTH] {model_type} process died ({exit_reason})")
                        del self.processes[model_type]
                    else:
                        print(f"[HEALTH] {model_type} not responding but process alive - killing")
                        await self.stop_model(model_type)

                # Add cooldown to prevent rapid restart loops
                last_restart = getattr(self, '_last_restart', {}).get(model_type, 0)
                if time.time() - last_restart < 30:
                    print(f"[HEALTH] {model_type} restart cooldown active, waiting...")
                    continue
                if not hasattr(self, '_last_restart'):
                    self._last_restart = {}
                self._last_restart[model_type] = time.time()

                # Restart the model
                print(f"[HEALTH] Restarting {model_type}...")
                success = await self.start_model(model_type)
                if success:
                    print(f"[HEALTH] {model_type} restarted successfully")
                else:
                    print(f"[ERROR] Failed to restart {model_type}")
            else:
                # Update last used time for healthy models
                self.last_used[model_type] = time.time()

    def is_running(self, model_type: str) -> bool:
        if model_type not in self.processes:
            return False
        return self.processes[model_type].poll() is None

    async def stop_all(self):
        """Stop all models with proper locking."""
        async with self._model_lock:
            await self._stop_all_internal()

    async def _stop_all_internal(self):
        """Internal stop_all without locking (use inside locked context)."""
        running = list(self.processes.keys())
        if running:
            print(f"[MODEL] Stopping models: {running}")
        for m in running:
            await self._stop_model_internal(m)
        await self._stop_comfy_internal()

    async def _stop_model_internal(self, model_type: str):
        """Internal stop_model without locking."""
        if model_type in self.processes:
            proc = self.processes[model_type]
            if proc.poll() is None:
                print(f"[MODEL] Killing {model_type} (PID {proc.pid})...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    await asyncio.sleep(1)
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        await asyncio.sleep(1)
                    print(f"[MODEL] {model_type} stopped")
                except Exception as e:
                    print(f"[WARN] Error stopping {model_type}: {e}")
            del self.processes[model_type]
            if model_type in self.last_used:
                del self.last_used[model_type]

    async def _stop_comfy_internal(self):
        """Internal stop_comfy without locking."""
        if self.comfy_process:
            if self.comfy_process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.comfy_process.pid), signal.SIGTERM)
                    await asyncio.sleep(1)
                    if self.comfy_process.poll() is None:
                        os.killpg(os.getpgid(self.comfy_process.pid), signal.SIGKILL)
                except Exception as e:
                    print(f"[WARN] Error stopping ComfyUI: {e}")
            self.comfy_process = None

    async def stop_comfy(self):
        """Stop ComfyUI with proper locking."""
        async with self._model_lock:
            await self._stop_comfy_internal()

    async def stop_model(self, model_type: str):
        """Stop a specific model with proper locking."""
        async with self._model_lock:
            await self._stop_model_internal(model_type)

    async def start_model(self, model_type: str) -> bool:
        """Start a model with proper locking to prevent race conditions."""
        async with self._model_lock:
            self._current_operation = f"starting {model_type}"
            try:
                return await self._start_model_internal(model_type)
            finally:
                self._current_operation = None

    async def _start_model_internal(self, model_type: str) -> bool:
        if model_type not in MODELS:
            print(f"[ERROR] Unknown model type: {model_type}")
            return False

        config = MODELS[model_type]

        # Check if same port model already running
        for running_type, proc in self.processes.items():
            if MODELS.get(running_type, {}).get("port") == config["port"] and proc.poll() is None:
                self.last_used[model_type] = time.time()
                return True

        if self.is_running(model_type):
            self.last_used[model_type] = time.time()
            return True

        # Determine if we should use GPU or CPU
        # Strategy for RTX 3060 (12GB):
        # - CPU_MODELS: Always CPU (general, planner, research) - ensures availability
        # - GPU_PREFERRED: Use GPU if available (code) - best performance
        # - GPU_ONLY: Must have GPU (vision) - may need to free VRAM
        use_gpu = False
        gpu_available, gpu_used, gpu_total = check_gpu_available()
        gpu_free = gpu_total - gpu_used if gpu_total > 0 else 0

        if model_type in CPU_MODELS:
            # Always run on CPU for guaranteed availability
            use_gpu = False
            print(f"[MODEL] {model_type} → CPU mode (always available)")
        elif model_type in GPU_ONLY_MODELS:
            # Must use GPU - may need to free VRAM from other GPU models
            if not gpu_available:
                # Try to free GPU memory by stopping GPU_PREFERRED models
                print(f"[GPU] {model_type} requires GPU, freeing VRAM...")
                for other_model in list(self.processes.keys()):
                    if other_model in GPU_PREFERRED and self.is_running(other_model):
                        print(f"[GPU] Stopping {other_model} to free VRAM")
                        await self._stop_model_internal(other_model)
                await asyncio.sleep(3)
                # Check again
                gpu_available, gpu_used, gpu_total = check_gpu_available()
                gpu_free = gpu_total - gpu_used if gpu_total > 0 else 0
                if not gpu_available:
                    print(f"[ERROR] {model_type} requires GPU but only {gpu_free} MiB free")
                    return False
            use_gpu = True
            print(f"[MODEL] {model_type} → GPU mode (required)")
        elif model_type in GPU_PREFERRED:
            # Prefer GPU but can fall back to CPU
            if gpu_available:
                use_gpu = True
                print(f"[MODEL] {model_type} → GPU mode (preferred, {gpu_free} MiB free)")
            else:
                use_gpu = False
                print(f"[MODEL] {model_type} → CPU fallback ({gpu_free} MiB free, need {GPU_VRAM_THRESHOLD})")
        else:
            # Default: use CPU for stability
            use_gpu = False
            print(f"[MODEL] {model_type} → CPU mode (default)")

        # Log current GPU state
        print(f"[GPU] Memory: {gpu_used}, {gpu_total} MiB | Mode: {'GPU' if use_gpu else 'CPU'}")

        print(f"[MODEL] Starting {config['name']} ({model_type}) on port {config['port']}...")

        cache_dir = LLAMA_CPP_DIR / "cache" / model_type
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Set GPU layers: full offload if GPU available, 0 for CPU-only
        gpu_layers = config["gpu_layers"] if use_gpu else 0
        # Use more threads for CPU mode
        threads = 8 if use_gpu else 12

        cmd = [
            str(LLAMA_SERVER), "-m", str(MODELS_DIR / config["file"]),
            "--host", "127.0.0.1", "--port", str(config["port"]),
            "-ngl", str(gpu_layers), "-c", str(config["context"]),
            "-t", str(threads), "--slot-save-path", str(cache_dir),
            "--cont-batching", "-b", "512", "-ub", "256"
        ]
        # Add flash attention for GPU mode (better memory efficiency)
        if use_gpu and gpu_layers > 0:
            cmd.extend(["--flash-attn", "on"])
            # Note: Speculative decoding disabled - requires too much VRAM on RTX 3060
            # Would need ~8GB free for both target + draft models
        # Add mlock for CPU mode (prevents swapping/OOM kills)
        if not use_gpu or gpu_layers == 0:
            cmd.extend(["--mlock"])
        if "mmproj" in config:
            cmd.extend(["--mmproj", str(MODELS_DIR / config["mmproj"])])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, start_new_session=True, text=True
            )
            self.processes[model_type] = proc
        except Exception as e:
            print(f"[ERROR] Failed to start {model_type}: {e}")
            return False

        for i in range(90):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{config['port']}/health", timeout=2.0)
                    if resp.status_code == 200:
                        mode = "GPU" if use_gpu else "CPU"
                        print(f"[MODEL] {config['name']} Ready on {mode}. (took {i+1}s)")
                        self.last_used[model_type] = time.time()
                        return True
            except:
                pass

        # Get process output for debugging
        proc = self.processes.get(model_type)
        if proc:
            try:
                # Read any available output
                import select
                if proc.stdout and select.select([proc.stdout], [], [], 0)[0]:
                    output = proc.stdout.read(2000)
                    print(f"[MODEL] Process output: {output[:500]}")
            except:
                pass
            # Check if process died
            if proc.poll() is not None:
                print(f"[ERROR] Process exited with code {proc.returncode}")

        print(f"[ERROR] {config['name']} failed to start (timeout after 90s)")
        await self._stop_model_internal(model_type)  # Use internal version (already locked)
        return False

    async def start_comfy(self) -> bool:
        if self.comfy_process and self.comfy_process.poll() is None:
            return True

        await self.stop_all()
        await asyncio.sleep(5)

        print("[COMFY] Starting ComfyUI...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            self.comfy_process = subprocess.Popen(
                COMFY_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, cwd=COMFY_DIR, start_new_session=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to start ComfyUI: {e}")
            return False

        for _ in range(120):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get("http://127.0.0.1:8188/system_stats", timeout=1.0)
                    if resp.status_code == 200:
                        print("[COMFY] ComfyUI Ready.")
                        return True
            except:
                pass

        print("[ERROR] ComfyUI failed to start (timeout)")
        await self.stop_comfy()
        return False

manager = ModelManager()

# --- TOOLS ---

async def execute_web_search(query: str) -> str:
    """Search the web using SearXNG."""
    print(f"[TOOL:SEARCH] Query: {query}")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                timeout=15.0
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])[:5]
                if not results:
                    return "No search results found."
                formatted = []
                for r in results:
                    formatted.append(f"- **{r.get('title', 'No title')}**\n  {r.get('content', 'No description')}\n  URL: {r.get('url', '')}")
                return "\n\n".join(formatted)
            else:
                return f"Search failed with status {resp.status_code}"
    except httpx.TimeoutException:
        return "Search timed out. Please try again."
    except Exception as e:
        return f"Search error: {str(e)}"


async def execute_claude_assist(prompt: str, context: str = "") -> str:
    """Call Claude CLI for complex reasoning and orchestration.

    Claude serves as the 'mastermind' for tasks that require:
    - Complex multi-step reasoning
    - Code review and architectural guidance
    - Synthesis of multiple information sources
    - High-quality explanations
    """
    print(f"[CLAUDE] Requesting assistance: {prompt[:100]}...")

    try:
        # Build the full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"

        # Use Claude CLI in print mode (non-interactive)
        cmd = [
            "claude",
            "-p", full_prompt,
            "--output-format", "text"
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path.home()
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=120.0  # 2 minute timeout for Claude
        )

        if proc.returncode == 0:
            response = stdout.decode().strip()
            print(f"[CLAUDE] Response received ({len(response)} chars)")

            # Auto-save important Claude insights to memory
            if len(response) > 200:
                await memory_manager.add_memory(
                    f"[Claude Insight] {prompt[:100]}: {response[:300]}...",
                    category="claude", importance=3
                )

            return response
        else:
            error = stderr.decode().strip()
            print(f"[CLAUDE] Error: {error}")
            return f"Claude assist failed: {error}"

    except asyncio.TimeoutError:
        print("[CLAUDE] Request timed out after 120s")
        return "Claude request timed out. Try breaking the task into smaller parts."
    except FileNotFoundError:
        print("[CLAUDE] CLI not found - not installed or not in PATH")
        return "Claude CLI not available. Install with: npm install -g @anthropic-ai/claude-code"
    except Exception as e:
        print(f"[CLAUDE] Error: {e}")
        return f"Claude assist error: {str(e)}"

async def execute_image_gen(prompt: str) -> str:
    """Generate an image using ComfyUI + Flux."""
    print(f"[TOOL:ART] Prompt: {prompt}")

    if not await manager.start_comfy():
        return "Image generation failed: Could not start ComfyUI"

    try:
        with open(COMFY_WORKFLOW_FILE, "r") as f:
            workflow = json.load(f)

        workflow["6"]["inputs"]["text"] = prompt
        workflow["3"]["inputs"]["seed"] = random.randint(1, 10**14)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow})
            resp_json = resp.json()

            if resp.status_code != 200 or "prompt_id" not in resp_json:
                error_msg = resp_json.get("error", {}).get("message", str(resp_json))
                return f"ComfyUI error: {error_msg}"

            prompt_id = resp_json["prompt_id"]

            for _ in range(120):
                await asyncio.sleep(1)
                hist_resp = await client.get(f"http://127.0.0.1:8188/history/{prompt_id}")
                hist = hist_resp.json()

                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    if "9" in outputs and "images" in outputs["9"]:
                        filename = outputs["9"]["images"][0]["filename"]
                        return f"![Generated Image](http://127.0.0.1:8188/view?filename={filename}&type=output)"

            return "Image generation timed out after 2 minutes"

    except json.JSONDecodeError as e:
        return f"Workflow file error: {str(e)}"
    except Exception as e:
        return f"Image generation error: {str(e)}"


def clean_model_output(text: str) -> str:
    """Clean up model artifacts like stop tokens and hallucinations."""
    if not text:
        return text

    # Stop tokens that models sometimes output
    stop_tokens = [
        "<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|end|>",
        "<|eot_id|>", "<|end_of_text|>", "</s>", "<s>",
        "<|assistant|>", "<|user|>", "<|system|>",
        "```\n\"\"\"", "\n<|"
    ]

    for token in stop_tokens:
        if token in text:
            text = text.split(token)[0]

    # Clean up any remaining artifacts at the end
    text = text.rstrip()

    return text


async def execute_vision(messages: List[Dict]) -> str:
    """Analyze images using Qwen2-VL. Stops GPU apps if needed, restarts after."""
    print("[TOOL:VISION] Analyzing image...")

    stopped_apps = []

    # Check GPU availability - if not enough, stop GPU-heavy apps
    gpu_available, gpu_used, gpu_total = check_gpu_available()
    if not gpu_available:
        gpu_free = gpu_total - gpu_used
        print(f"[GPU] Vision needs GPU but only {gpu_free} MiB free - stopping GPU apps...")

        # First stop any running llama-server models
        await manager.stop_all()
        await asyncio.sleep(2)

        # Check again
        gpu_available, gpu_used, gpu_total = check_gpu_available()
        if not gpu_available:
            # Need to stop ComfyUI/etc
            stopped_apps = stop_gpu_apps_for_vision()

            if stopped_apps:
                await asyncio.sleep(3)
                gpu_available, gpu_used, gpu_total = check_gpu_available()
                if not gpu_available:
                    print("[GPU] Still not enough VRAM after stopping apps")
                    restart_gpu_apps(stopped_apps)
                    return f"Vision analysis requires more GPU memory. Could not free enough VRAM."
            else:
                return f"Vision analysis requires GPU but only {gpu_free} MiB VRAM free (need {GPU_VRAM_THRESHOLD} MiB). No stoppable GPU apps found."

    try:
        if not await manager.start_model("vision"):
            if stopped_apps:
                restart_gpu_apps(stopped_apps)
            return "Vision analysis failed: Could not start vision model"

        port = MODELS["vision"]["port"]

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"messages": messages, "max_tokens": 1024, "temperature": 0.1}
            )

            if resp.status_code != 200:
                if stopped_apps:
                    await manager.stop_all()
                    await asyncio.sleep(2)
                    restart_gpu_apps(stopped_apps)
                return f"Vision model error: HTTP {resp.status_code}"

            result = resp.json()
            response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response from vision model")
            response = clean_model_output(response)  # Clean up stop tokens
            print(f"[VISION] Analysis result: {response[:500]}..." if len(response) > 500 else f"[VISION] Analysis result: {response}")

            # Stop vision model and restart stopped apps
            if stopped_apps:
                print("[GPU] Vision complete - restarting stopped apps...")
                await manager.stop_all()
                await asyncio.sleep(2)
                restart_gpu_apps(stopped_apps)

            return response

    except httpx.TimeoutException:
        if stopped_apps:
            await manager.stop_all()
            await asyncio.sleep(2)
            restart_gpu_apps(stopped_apps)
        return "Vision analysis timed out"
    except Exception as e:
        if stopped_apps:
            await manager.stop_all()
            await asyncio.sleep(2)
            restart_gpu_apps(stopped_apps)
        return f"Vision analysis error: {str(e)}"

async def execute_reasoning(messages: List[Dict], model_type: str = "research") -> str:
    """Execute reasoning/text generation with specified model."""
    print(f"[TOOL:REASONING] Model: {model_type}")

    if model_type not in MODELS:
        model_type = "research"

    # Check concurrency limit - avoid queueing
    semaphore = get_model_semaphore(model_type)
    try:
        # Try to acquire within 5 seconds, otherwise return busy
        await asyncio.wait_for(semaphore.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        print(f"[BUSY] Model {model_type} at capacity, rejecting request")
        return f"Model {model_type} is currently busy. Please try again in a moment."

    try:
        if not await manager.start_model(model_type):
            return f"Reasoning failed: Could not start {model_type} model"

        port = MODELS[model_type]["port"]

        # Adjust for CPU mode
        gpu_available, _, _ = check_gpu_available()
        timeout = 300.0 if gpu_available else 600.0
        max_tokens = 1024 if gpu_available else 512  # Reduced for faster responses

        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload
            )

            if resp.status_code != 200:
                print(f"[ERROR] Model returned {resp.status_code}: {resp.text[:500]}")
                print(f"[DEBUG] Payload was: {json.dumps(payload)[:500]}")
                return f"Model error: HTTP {resp.status_code}"

            result = resp.json()
            response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response from model")
            return clean_model_output(response)  # Clean up stop tokens

    except httpx.TimeoutException:
        return "Request timed out after 5 minutes"
    except Exception as e:
        return f"Reasoning error: {str(e)}"
    finally:
        semaphore.release()

async def execute_reasoning_stream(messages: List[Dict], model_type: str = "research") -> AsyncGenerator[str, None]:
    """Stream reasoning/text generation with specified model."""
    print(f"[TOOL:REASONING:STREAM] Model: {model_type}")

    if model_type not in MODELS:
        model_type = "research"

    if not await manager.start_model(model_type):
        yield "Reasoning failed: Could not start model"
        return

    port = MODELS[model_type]["port"]

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"messages": messages, "max_tokens": 1024, "temperature": 0.7, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"Streaming error: {str(e)}"

async def execute_code_task(task: str, language: str = "python", execute: bool = True) -> str:
    """
    Use DeepSeek Coder to generate code from a natural language task description.
    This is the PROPER flow: Planner decides task → DeepSeek writes code → Execute.
    """
    print(f"[TOOL:CODE_TASK] Language: {language}, Task: {task[:100]}...")

    # Normalize language
    lang_map = {
        "python": "python", "py": "python", "python3": "python",
        "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
        "c": "c",
        "cuda": "cuda", "cu": "cuda",
        "javascript": "javascript", "js": "javascript",
        "typescript": "typescript", "ts": "typescript",
        "rust": "rust", "rs": "rust",
        "go": "go", "golang": "go",
    }
    lang = lang_map.get(language.lower(), language.lower())

    # Check concurrency limit for code model
    semaphore = get_model_semaphore("code")
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        print(f"[BUSY] Code model at capacity, rejecting request")
        return "Code model is currently busy processing another request. Please try again in a moment."

    try:
        # Start DeepSeek Coder
        if not await manager.start_model("code"):
            return "Code generation failed: Could not start DeepSeek Coder model"

        port = MODELS["code"]["port"]

        # Prompt DeepSeek to generate code
        lang_hints = {
            "cpp": "Use #include <iostream>, #include <vector>, etc. as needed.",
            "c": "Use #include <stdio.h>, #include <stdlib.h>, etc. as needed.",
            "cuda": "Use #include <cuda_runtime.h> and proper __global__ kernel syntax.",
            "python": "Use standard library imports as needed.",
            "javascript": "Use modern ES6+ syntax.",
            "typescript": "Include proper type annotations.",
        }
        hint = lang_hints.get(lang, "")

        code_prompt = f"""Write complete, runnable {lang} code for the following task.

Task: {task}

Requirements:
1. MUST include ALL necessary imports/headers at the top
2. MUST include a main function (or entry point)
3. MUST be complete and self-contained - no missing functions
4. Add brief comments for complex logic
5. {hint}

Return ONLY the complete code, no explanations before or after.

```{lang}"""

        # Check if GPU is available - adjust timeout and tokens for CPU mode
        gpu_available, _, _ = check_gpu_available()
        if gpu_available:
            timeout = 90.0    # GPU: fast inference, reduced from 120s
            max_tokens = 1024  # Reduced from 2048 for faster responses
        else:
            timeout = 300.0   # CPU: reduced from 600s to prevent queue blocking
            max_tokens = 768   # Balanced: enough for useful code, fast enough
            print(f"[CODE] CPU mode: timeout={timeout}s, max_tokens={max_tokens}")

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": code_prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3  # Lower temp for more deterministic code
                }
            )

            if resp.status_code != 200:
                return f"Code generation error: HTTP {resp.status_code}"

            result = resp.json()
            generated_code = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"[CODE] Raw response: {generated_code[:200] if generated_code else 'EMPTY'}...")

            # Clean up model artifacts (stop tokens, hallucinations)
            generated_code = clean_model_output(generated_code)

            # Extract code from markdown blocks if present
            if "```" in generated_code:
                import re
                # Try to find code block with language tag
                code_match = re.search(r'```(?:python|cpp|c|cuda|javascript|typescript|rust|go|java)?\s*\n?(.*?)```', generated_code, re.DOTALL | re.IGNORECASE)
                if code_match:
                    generated_code = code_match.group(1).strip()
                else:
                    # Fallback: just strip everything before first ``` and after last ```
                    first_block = generated_code.find("```")
                    if first_block != -1:
                        # Skip past the opening ``` and language tag
                        start = generated_code.find("\n", first_block)
                        if start != -1:
                            end = generated_code.rfind("```")
                            if end > start:
                                generated_code = generated_code[start:end].strip()

            # Remove preamble text that models sometimes add
            preamble_markers = [
                "here is the", "here's the", "below is", "the following",
                "i've created", "i have created", "complete code:", "solution:"
            ]
            for marker in preamble_markers:
                if generated_code.lower().startswith(marker):
                    # Find first newline after preamble
                    newline_pos = generated_code.find("\n")
                    if newline_pos != -1:
                        generated_code = generated_code[newline_pos:].strip()
                    break

            # Final cleanup
            generated_code = generated_code.strip()

            if not generated_code:
                return "Code generation failed: Empty response from model"

            output = f"**Generated {lang.upper()} Code:**\n```{lang}\n{generated_code}\n```\n\n"

            # Execute if requested
            if execute:
                if lang == "python":
                    exec_result = await execute_code(generated_code)
                    output += exec_result
                elif lang == "cpp":
                    exec_result = await execute_run_cpp(generated_code)
                    output += exec_result
                elif lang == "c":
                    exec_result = await execute_run_c(generated_code)
                    output += exec_result
                elif lang == "cuda":
                    exec_result = await execute_run_cuda(generated_code)
                    output += exec_result
                elif lang in ["javascript", "typescript"]:
                    # Run with node/bun
                    exec_result = await execute_run_js(generated_code, lang)
                    output += exec_result
                else:
                    output += f"*Execution not yet supported for {lang}*"

            return output

    except httpx.TimeoutException:
        return "Code generation timed out"
    except Exception as e:
        return f"Code generation error: {str(e)}"
    finally:
        semaphore.release()


# --- ENGINEERING ASSISTANT TOOLS ---

async def execute_code_review(code: str, language: str = "python", focus: str = "all") -> str:
    """Review code for issues, best practices, and improvements.

    Uses Claude for high-quality reviews when available, falls back to local model.
    Focus can be: 'all', 'security', 'performance', 'style', 'bugs'
    """
    print(f"[TOOL:CODE_REVIEW] Reviewing {language} code (focus: {focus})")

    prompt = f"""Review this {language} code with focus on {focus}:

```{language}
{code}
```

Provide:
1. **Issues Found** - Bugs, security vulnerabilities, potential errors
2. **Best Practices** - What could be improved
3. **Suggestions** - Specific code improvements with examples
4. **Rating** - Overall code quality (1-10)

Be constructive and specific."""

    # Use Claude for better reviews
    result = await execute_claude_assist(prompt, f"Code review for {language}")
    if "error" not in result.lower() and len(result) > 100:
        return f"**Code Review ({language}):**\n\n{result}"

    # Fallback to local model
    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "code")


async def execute_explain_concept(concept: str, level: str = "intermediate") -> str:
    """Explain a programming/engineering concept with examples.

    Level: 'beginner', 'intermediate', 'advanced'
    """
    print(f"[TOOL:EXPLAIN] Explaining '{concept}' at {level} level")

    prompt = f"""Explain the concept of "{concept}" for a {level}-level programmer.

Include:
1. **What it is** - Clear definition
2. **Why it matters** - Real-world importance
3. **How it works** - Step-by-step explanation
4. **Example** - Code example with comments
5. **Common mistakes** - What to avoid
6. **Related concepts** - What to learn next

Use simple language and practical examples."""

    # Use Claude for better explanations
    result = await execute_claude_assist(prompt, f"Explaining {concept}")
    if "error" not in result.lower() and len(result) > 100:
        return f"**{concept}:**\n\n{result}"

    # Fallback to local model
    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "research")


async def execute_generate_tests(code: str, language: str = "python", framework: str = "") -> str:
    """Generate unit tests for the given code.

    Automatically detects appropriate testing framework.
    """
    print(f"[TOOL:GENERATE_TESTS] Creating tests for {language} code")

    # Detect framework if not specified
    if not framework:
        framework_map = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "go": "testing",
            "rust": "cargo test",
            "java": "junit"
        }
        framework = framework_map.get(language, "pytest")

    prompt = f"""Generate comprehensive unit tests for this {language} code using {framework}:

```{language}
{code}
```

Include:
1. **Happy path tests** - Normal expected behavior
2. **Edge cases** - Boundary conditions, empty inputs
3. **Error cases** - Invalid inputs, exceptions
4. **Setup/teardown** - If needed

Follow {framework} conventions and best practices."""

    # Generate tests using code model
    task = f"Write {framework} tests for this code:\n{code}"
    return await execute_code_task(task, language, execute=False)


async def execute_debug_help(error_message: str, code: str = "", language: str = "python") -> str:
    """Help debug an error with explanation and fix suggestions.

    Uses Claude for complex debugging.
    """
    print(f"[TOOL:DEBUG] Analyzing error in {language}")

    prompt = f"""Debug this {language} error:

**Error:**
```
{error_message}
```
"""
    if code:
        prompt += f"""
**Code:**
```{language}
{code}
```
"""

    prompt += """
Provide:
1. **What the error means** - Plain English explanation
2. **Root cause** - Why this happened
3. **How to fix it** - Step-by-step solution
4. **Fixed code** - If code was provided
5. **Prevention** - How to avoid this in the future"""

    # Use Claude for better debugging
    result = await execute_claude_assist(prompt, "Debugging assistance")
    if "error" not in result.lower() and len(result) > 100:
        return f"**Debug Analysis:**\n\n{result}"

    # Fallback to local model
    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "code")


# --- LEARNING TOOLS ---

async def execute_create_flashcard(front: str, back: str, topic: str = "general") -> str:
    """Create a flashcard for spaced repetition learning."""
    print(f"[TOOL:FLASHCARD] Creating flashcard for topic: {topic}")

    flashcard_data = {
        "front": front,
        "back": back,
        "topic": topic,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reviews": 0,
        "next_review": time.strftime("%Y-%m-%d")
    }

    # Store in memory with flashcard category
    content = f"[FLASHCARD:{topic}] Q: {front} | A: {back}"
    await memory_manager.add_memory(content, category="flashcard", importance=2)

    return f"""✅ **Flashcard Created!**

**Topic:** {topic}
**Front (Question):** {front}
**Back (Answer):** {back}

Use `study_flashcards("{topic}")` to review your flashcards."""


async def execute_study_flashcards(topic: str = "") -> str:
    """Retrieve flashcards for study session."""
    print(f"[TOOL:STUDY] Retrieving flashcards for: {topic or 'all topics'}")

    search_query = f"FLASHCARD:{topic}" if topic else "FLASHCARD"
    results = await memory_manager.search(search_query, limit=10)

    if not results:
        return f"No flashcards found{' for topic: ' + topic if topic else ''}. Create some with `create_flashcard`!"

    output = f"## 📚 Study Session: {topic or 'All Topics'}\n\n"
    output += "*Try to answer each question before revealing the answer!*\n\n"

    for i, r in enumerate(results, 1):
        content = r.get("content", "")
        # Parse flashcard format: [FLASHCARD:topic] Q: ... | A: ...
        if "Q:" in content and "A:" in content:
            parts = content.split("|")
            question = parts[0].split("Q:")[-1].strip() if "Q:" in parts[0] else parts[0]
            answer = parts[1].split("A:")[-1].strip() if len(parts) > 1 and "A:" in parts[1] else "N/A"
            output += f"### Card {i}\n**Q:** {question}\n\n<details><summary>Show Answer</summary>\n\n**A:** {answer}\n</details>\n\n---\n\n"

    return output


async def execute_generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "intermediate") -> str:
    """Generate a quiz on a programming topic."""
    print(f"[TOOL:QUIZ] Generating {num_questions} questions on {topic} ({difficulty})")

    prompt = f"""Create a {difficulty} level quiz about {topic} with {num_questions} multiple choice questions.

Format each question as:
## Question N
[Question text]

A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

<details><summary>Answer</summary>
[Correct letter]: [Brief explanation]
</details>

Make questions practical and relevant to real-world programming."""

    result = await execute_claude_assist(prompt, f"Quiz generation for {topic}")
    if "error" not in result.lower() and len(result) > 100:
        return f"# 📝 Quiz: {topic}\n**Difficulty:** {difficulty}\n\n{result}"

    # Fallback to local model
    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "research")


async def execute_learning_path(goal: str) -> str:
    """Generate a personalized learning path for a programming goal."""
    print(f"[TOOL:LEARNING_PATH] Creating path for: {goal}")

    prompt = f"""Create a structured learning path for someone who wants to: {goal}

Include:
1. **Prerequisites** - What they should know first
2. **Core Concepts** - Main topics to learn (in order)
3. **Hands-on Projects** - Practical projects for each stage
4. **Resources** - Types of resources to use (not specific URLs)
5. **Milestones** - How to know when they've mastered each stage
6. **Timeline** - Rough estimate for dedicated learning

Make it actionable and progressive."""

    result = await execute_claude_assist(prompt, f"Learning path for: {goal}")
    if "error" not in result.lower() and len(result) > 100:
        # Save to memory for reference
        await memory_manager.add_memory(
            f"[LEARNING_PATH] Goal: {goal[:100]}",
            category="learning",
            importance=3
        )
        return f"# 🎯 Learning Path: {goal}\n\n{result}"

    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "research")


async def execute_code_challenge(topic: str, difficulty: str = "medium") -> str:
    """Generate a coding challenge for practice."""
    print(f"[TOOL:CHALLENGE] Generating {difficulty} challenge for: {topic}")

    prompt = f"""Create a {difficulty} coding challenge about {topic}.

Include:
1. **Problem Statement** - Clear description of what to build
2. **Requirements** - Specific features/constraints
3. **Examples** - Input/output examples
4. **Hints** - Progressive hints (hidden by default)
5. **Bonus** - Extra challenges for those who finish early

Make it educational and fun!"""

    result = await execute_claude_assist(prompt, f"Coding challenge: {topic}")
    if "error" not in result.lower() and len(result) > 100:
        return f"# 💻 Coding Challenge: {topic}\n**Difficulty:** {difficulty}\n\n{result}"

    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "code")


async def execute_explain_error(error_type: str, language: str = "python") -> str:
    """Explain a common error type with examples and solutions."""
    print(f"[TOOL:EXPLAIN_ERROR] Explaining {error_type} in {language}")

    prompt = f"""Explain the {error_type} error in {language}:

1. **What it means** - Simple explanation
2. **Common causes** - Why this error occurs (with code examples)
3. **How to fix it** - Step-by-step solutions
4. **How to prevent it** - Best practices to avoid it
5. **Related errors** - Similar errors to know about

Use practical examples that beginners can understand."""

    result = await execute_claude_assist(prompt, f"Error explanation: {error_type}")
    if "error" not in result.lower() and len(result) > 100:
        return f"# ⚠️ Understanding: {error_type}\n**Language:** {language}\n\n{result}"

    msgs = [{"role": "user", "content": prompt}]
    return await execute_reasoning(msgs, "code")


async def execute_run_js(code: str, variant: str = "javascript") -> str:
    """Run JavaScript/TypeScript code with Node.js or Bun."""
    print(f"[TOOL:RUN_JS] Running {variant}")

    tmp_dir = Path("/tmp/benchai_js")
    tmp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time() * 1000)
    ext = ".ts" if variant == "typescript" else ".js"
    src_file = tmp_dir / f"code_{timestamp}{ext}"

    try:
        src_file.write_text(code)

        # Try bun first (faster), fall back to node
        runner = "bun" if subprocess.run("which bun", shell=True, capture_output=True).returncode == 0 else "node"

        if variant == "typescript" and runner == "node":
            # Need ts-node for TypeScript with node
            runner = "npx ts-node"

        run_result = subprocess.run(
            f"{runner} {src_file}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = f"**{variant.title()} Execution ({runner}):**\n"
        if run_result.stdout:
            output += f"```\n{run_result.stdout[:5000]}\n```\n"
        if run_result.stderr:
            output += f"\n**stderr:**\n```\n{run_result.stderr[:2000]}\n```\n"
        output += f"**Exit code:** {run_result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "JavaScript execution timed out (30 second limit)"
    except Exception as e:
        return f"JavaScript error: {str(e)}"
    finally:
        if src_file.exists():
            src_file.unlink()


async def execute_code(code: str, language: str = "python") -> str:
    """Execute code in a sandboxed environment."""
    print(f"[TOOL:CODE] Executing {language} code...")

    if language.lower() not in ["python", "python3", "py"]:
        return f"Language '{language}' not supported. Currently only Python is available."

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["python3", temp_file],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tempfile.gettempdir()
        )

        output = ""
        if result.stdout:
            output += f"**Output:**\n```\n{result.stdout}\n```\n"
        if result.stderr:
            output += f"**Errors:**\n```\n{result.stderr}\n```\n"
        if result.returncode != 0:
            output += f"**Exit code:** {result.returncode}\n"

        return output if output else "Code executed successfully with no output."

    except subprocess.TimeoutExpired:
        return "Code execution timed out (30 second limit)"
    except Exception as e:
        return f"Code execution error: {str(e)}"
    finally:
        os.unlink(temp_file)

async def execute_file_read(file_path: str) -> str:
    """Read content from a file."""
    print(f"[TOOL:FILE] Reading: {file_path}")

    path = Path(file_path).expanduser()

    if not path.exists():
        return f"File not found: {file_path}"

    if not path.is_file():
        return f"Not a file: {file_path}"

    if path.stat().st_size > 100 * 1024:
        return f"File too large (>100KB). Consider reading specific sections."

    try:
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            try:
                result = subprocess.run(
                    ["pdftotext", "-layout", str(path), "-"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return f"**PDF Content ({path.name}):**\n```\n{result.stdout[:10000]}\n```"
                else:
                    return f"PDF extraction failed: {result.stderr}"
            except FileNotFoundError:
                return "PDF reading requires 'pdftotext' (install: sudo apt install poppler-utils)"

        elif suffix in ['.json']:
            with open(path, 'r') as f:
                data = json.load(f)
                return f"**JSON Content ({path.name}):**\n```json\n{json.dumps(data, indent=2)[:10000]}\n```"

        else:
            with open(path, 'r', errors='replace') as f:
                content = f.read()

            lang_map = {
                '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c',
                '.rs': 'rust', '.go': 'go', '.rb': 'ruby', '.php': 'php',
                '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
                '.html': 'html', '.css': 'css', '.sql': 'sql',
                '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml',
                '.md': 'markdown', '.xml': 'xml'
            }
            lang = lang_map.get(suffix, '')

            return f"**File Content ({path.name}):**\n```{lang}\n{content[:10000]}\n```"

    except UnicodeDecodeError:
        return f"Cannot read file: appears to be binary"
    except Exception as e:
        return f"Error reading file: {str(e)}"

async def execute_shell(command: str) -> str:
    """Execute a shell command (with safety restrictions)."""
    print(f"[TOOL:SHELL] Command: {command}")

    parts = command.strip().split()
    if not parts:
        return "No command provided"

    base_cmd = parts[0]

    if base_cmd not in ALLOWED_SHELL_COMMANDS:
        return f"Command '{base_cmd}' not allowed. Allowed: {', '.join(sorted(ALLOWED_SHELL_COMMANDS)[:20])}..."

    dangerous_patterns = ['rm -rf', 'mkfs', '> /dev', 'dd if=', ':(){', 'fork bomb']
    for pattern in dangerous_patterns:
        if pattern in command.lower():
            return f"Dangerous pattern detected: '{pattern}'"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path.home())
        )

        output = ""
        if result.stdout:
            output += f"```\n{result.stdout[:5000]}\n```\n"
        if result.stderr:
            output += f"**stderr:**\n```\n{result.stderr[:2000]}\n```\n"
        if result.returncode != 0:
            output += f"**Exit code:** {result.returncode}"

        return output if output else "Command completed with no output."

    except subprocess.TimeoutExpired:
        return "Command timed out (30 second limit)"
    except Exception as e:
        return f"Shell error: {str(e)}"


# --- C/C++ DEVELOPMENT TOOLS ---

async def execute_compile_cpp(code: str, flags: str = "-std=c++17 -O2 -Wall") -> str:
    """Compile C++ code and return compilation result."""
    print(f"[TOOL:COMPILE_CPP] Compiling with flags: {flags}")

    tmp_dir = Path("/tmp/benchai_cpp")
    tmp_dir.mkdir(exist_ok=True)

    # Create unique filenames
    timestamp = int(time.time() * 1000)
    src_file = tmp_dir / f"code_{timestamp}.cpp"
    out_file = tmp_dir / f"code_{timestamp}.out"

    try:
        src_file.write_text(code)

        # Compile with g++
        cmd = f"g++ {flags} -o {out_file} {src_file}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = ""
        if result.returncode == 0:
            output = f"**Compilation successful!**\nBinary: `{out_file}`\n"
            # Get binary info
            size_result = subprocess.run(f"size {out_file}", shell=True, capture_output=True, text=True)
            if size_result.stdout:
                output += f"\n**Binary size info:**\n```\n{size_result.stdout}\n```"
        else:
            output = f"**Compilation failed (exit {result.returncode}):**\n"
            if result.stderr:
                output += f"```\n{result.stderr[:3000]}\n```"
            if result.stdout:
                output += f"\n```\n{result.stdout[:1000]}\n```"

        return output

    except subprocess.TimeoutExpired:
        return "Compilation timed out (30 second limit)"
    except Exception as e:
        return f"Compile error: {str(e)}"


async def execute_run_cpp(code: str, flags: str = "-std=c++17 -O2 -Wall", input_data: str = "") -> str:
    """Compile and run C++ code, returning output."""
    print(f"[TOOL:RUN_CPP] Compile and run")

    tmp_dir = Path("/tmp/benchai_cpp")
    tmp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time() * 1000)
    src_file = tmp_dir / f"code_{timestamp}.cpp"
    out_file = tmp_dir / f"code_{timestamp}.out"

    try:
        src_file.write_text(code)

        # Compile
        compile_cmd = f"g++ {flags} -o {out_file} {src_file}"
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if compile_result.returncode != 0:
            return f"**Compilation failed:**\n```\n{compile_result.stderr[:3000]}\n```"

        # Run
        run_result = subprocess.run(
            str(out_file),
            shell=True,
            capture_output=True,
            text=True,
            input=input_data if input_data else None,
            timeout=30
        )

        output = "**Compilation:** Success\n\n**Execution Output:**\n"
        if run_result.stdout:
            output += f"```\n{run_result.stdout[:5000]}\n```\n"
        if run_result.stderr:
            output += f"\n**stderr:**\n```\n{run_result.stderr[:2000]}\n```\n"
        output += f"**Exit code:** {run_result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "Execution timed out (30 second limit)"
    except Exception as e:
        return f"Run error: {str(e)}"
    finally:
        # Cleanup
        if src_file.exists():
            src_file.unlink()
        if out_file.exists():
            out_file.unlink()


async def execute_compile_c(code: str, flags: str = "-std=c11 -O2 -Wall") -> str:
    """Compile C code and return compilation result."""
    print(f"[TOOL:COMPILE_C] Compiling with flags: {flags}")

    tmp_dir = Path("/tmp/benchai_c")
    tmp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time() * 1000)
    src_file = tmp_dir / f"code_{timestamp}.c"
    out_file = tmp_dir / f"code_{timestamp}.out"

    try:
        src_file.write_text(code)

        cmd = f"gcc {flags} -o {out_file} {src_file}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = ""
        if result.returncode == 0:
            output = f"**Compilation successful!**\nBinary: `{out_file}`\n"
            size_result = subprocess.run(f"size {out_file}", shell=True, capture_output=True, text=True)
            if size_result.stdout:
                output += f"\n**Binary size info:**\n```\n{size_result.stdout}\n```"
        else:
            output = f"**Compilation failed (exit {result.returncode}):**\n"
            if result.stderr:
                output += f"```\n{result.stderr[:3000]}\n```"

        return output

    except subprocess.TimeoutExpired:
        return "Compilation timed out (30 second limit)"
    except Exception as e:
        return f"Compile error: {str(e)}"


async def execute_run_c(code: str, flags: str = "-std=c11 -O2 -Wall", input_data: str = "") -> str:
    """Compile and run C code, returning output."""
    print(f"[TOOL:RUN_C] Compile and run")

    tmp_dir = Path("/tmp/benchai_c")
    tmp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time() * 1000)
    src_file = tmp_dir / f"code_{timestamp}.c"
    out_file = tmp_dir / f"code_{timestamp}.out"

    try:
        src_file.write_text(code)

        compile_cmd = f"gcc {flags} -o {out_file} {src_file}"
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        if compile_result.returncode != 0:
            return f"**Compilation failed:**\n```\n{compile_result.stderr[:3000]}\n```"

        run_result = subprocess.run(
            str(out_file),
            shell=True,
            capture_output=True,
            text=True,
            input=input_data if input_data else None,
            timeout=30
        )

        output = "**Compilation:** Success\n\n**Execution Output:**\n"
        if run_result.stdout:
            output += f"```\n{run_result.stdout[:5000]}\n```\n"
        if run_result.stderr:
            output += f"\n**stderr:**\n```\n{run_result.stderr[:2000]}\n```\n"
        output += f"**Exit code:** {run_result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "Execution timed out (30 second limit)"
    except Exception as e:
        return f"Run error: {str(e)}"
    finally:
        if src_file.exists():
            src_file.unlink()
        if out_file.exists():
            out_file.unlink()


async def execute_cmake_build(project_dir: str, build_type: str = "Release") -> str:
    """Run CMake configure and build on a project directory."""
    print(f"[TOOL:CMAKE] Building {project_dir} ({build_type})")

    project_path = Path(project_dir).expanduser().resolve()
    if not project_path.exists():
        return f"Project directory not found: {project_path}"

    cmake_file = project_path / "CMakeLists.txt"
    if not cmake_file.exists():
        return f"No CMakeLists.txt found in {project_path}"

    build_dir = project_path / "build"
    build_dir.mkdir(exist_ok=True)

    try:
        # Configure
        configure_cmd = f"cmake -S {project_path} -B {build_dir} -DCMAKE_BUILD_TYPE={build_type}"
        configure_result = subprocess.run(
            configure_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = "**CMake Configure:**\n"
        if configure_result.returncode != 0:
            output += f"```\n{configure_result.stderr[:3000]}\n```"
            return output + "\n**Configure failed!**"
        output += "Success\n\n"

        # Build
        build_cmd = f"cmake --build {build_dir} -j$(nproc)"
        build_result = subprocess.run(
            build_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120
        )

        output += "**CMake Build:**\n"
        if build_result.returncode == 0:
            output += f"```\n{build_result.stdout[-2000:]}\n```\n"
            output += "**Build successful!**"
        else:
            output += f"```\n{build_result.stderr[:3000]}\n```"
            output += "\n**Build failed!**"

        return output

    except subprocess.TimeoutExpired:
        return "Build timed out (120 second limit)"
    except Exception as e:
        return f"CMake error: {str(e)}"


async def execute_analyze_cpp(code: str) -> str:
    """Static analysis of C++ code using available tools."""
    print(f"[TOOL:ANALYZE_CPP] Running static analysis")

    tmp_dir = Path("/tmp/benchai_cpp")
    tmp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time() * 1000)
    src_file = tmp_dir / f"analyze_{timestamp}.cpp"

    try:
        src_file.write_text(code)
        output = ""

        # Try g++ with extra warnings
        warn_cmd = f"g++ -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -fsyntax-only {src_file}"
        warn_result = subprocess.run(warn_cmd, shell=True, capture_output=True, text=True, timeout=30)

        output += "**G++ Static Analysis (-Wall -Wextra -Wpedantic -Wconversion):**\n"
        if warn_result.stderr:
            output += f"```\n{warn_result.stderr[:2000]}\n```\n"
        else:
            output += "No warnings or errors.\n"

        # Try cppcheck if available
        cppcheck_path = subprocess.run("which cppcheck", shell=True, capture_output=True, text=True)
        if cppcheck_path.returncode == 0:
            cppcheck_cmd = f"cppcheck --enable=all --std=c++17 {src_file}"
            cppcheck_result = subprocess.run(cppcheck_cmd, shell=True, capture_output=True, text=True, timeout=30)
            output += "\n**Cppcheck Analysis:**\n"
            if cppcheck_result.stderr:
                output += f"```\n{cppcheck_result.stderr[:2000]}\n```\n"
            else:
                output += "No issues found.\n"

        return output if output else "Analysis complete - no issues detected."

    except subprocess.TimeoutExpired:
        return "Analysis timed out"
    except Exception as e:
        return f"Analysis error: {str(e)}"
    finally:
        if src_file.exists():
            src_file.unlink()


async def execute_run_cuda(code: str, flags: str = "-O2", input_data: str = "") -> str:
    """Compile and run CUDA code using nvcc."""
    print(f"[TOOL:RUN_CUDA] Compile and run CUDA")

    tmp_dir = Path("/tmp/benchai_cuda")
    tmp_dir.mkdir(exist_ok=True)

    timestamp = int(time.time() * 1000)
    src_file = tmp_dir / f"code_{timestamp}.cu"
    out_file = tmp_dir / f"code_{timestamp}.out"

    try:
        src_file.write_text(code)

        # Compile with nvcc (use full path for subprocess)
        nvcc_path = "/usr/local/cuda-12.4/bin/nvcc"
        compile_cmd = f"{nvcc_path} {flags} -o {out_file} {src_file}"
        compile_result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if compile_result.returncode != 0:
            return f"**CUDA Compilation failed:**\n```\n{compile_result.stderr[:3000]}\n```"

        # Run
        run_result = subprocess.run(
            str(out_file),
            shell=True,
            capture_output=True,
            text=True,
            input=input_data if input_data else None,
            timeout=60
        )

        output = "**CUDA Compilation:** Success\n\n**Execution Output:**\n"
        if run_result.stdout:
            output += f"```\n{run_result.stdout[:5000]}\n```\n"
        if run_result.stderr:
            output += f"\n**stderr:**\n```\n{run_result.stderr[:2000]}\n```\n"
        output += f"**Exit code:** {run_result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return "CUDA execution timed out (60 second limit)"
    except Exception as e:
        return f"CUDA error: {str(e)}"
    finally:
        if src_file.exists():
            src_file.unlink()
        if out_file.exists():
            out_file.unlink()


async def execute_debug_info(binary_path: str) -> str:
    """Get debugging information about a compiled binary."""
    print(f"[TOOL:DEBUG_INFO] Analyzing {binary_path}")

    bin_path = Path(binary_path).expanduser().resolve()
    if not bin_path.exists():
        return f"Binary not found: {bin_path}"

    output = f"**Binary Analysis: {bin_path.name}**\n\n"

    try:
        # File type
        file_result = subprocess.run(f"file {bin_path}", shell=True, capture_output=True, text=True)
        output += f"**Type:** {file_result.stdout.strip()}\n\n"

        # Size info
        size_result = subprocess.run(f"size {bin_path}", shell=True, capture_output=True, text=True)
        if size_result.stdout:
            output += f"**Size breakdown:**\n```\n{size_result.stdout}\n```\n"

        # Shared libraries
        ldd_result = subprocess.run(f"ldd {bin_path}", shell=True, capture_output=True, text=True)
        if ldd_result.stdout and "not a dynamic executable" not in ldd_result.stdout:
            output += f"**Shared libraries:**\n```\n{ldd_result.stdout[:1500]}\n```\n"

        # Symbol table summary
        nm_result = subprocess.run(f"nm {bin_path} 2>/dev/null | head -30", shell=True, capture_output=True, text=True)
        if nm_result.stdout:
            output += f"**Symbols (first 30):**\n```\n{nm_result.stdout}\n```\n"

        return output

    except Exception as e:
        return f"Debug info error: {str(e)}"


async def execute_memory_search(query: str) -> str:
    """Search memories for relevant information."""
    print(f"[TOOL:MEMORY] Searching: {query}")
    memories = await memory_manager.search_memories(query)
    if not memories:
        return "No relevant memories found."

    result = "**Relevant Memories:**\n"
    for mem in memories:
        result += f"- [{mem['category']}] {mem['content']}\n"
    return result

async def execute_rag_search(query: str) -> str:
    """Search indexed documents using RAG."""
    print(f"[TOOL:RAG] Searching: {query}")

    if not CHROMADB_AVAILABLE:
        return "RAG not available (ChromaDB not installed)"

    docs = await rag_manager.search(query)
    if not docs:
        return "No relevant documents found in knowledge base."

    result = "**Relevant Documents:**\n"
    for doc in docs:
        source = doc['metadata'].get('source', 'Unknown')
        result += f"\n**From {source}:**\n{doc['content'][:500]}...\n"
    return result


# --- OBSIDIAN TOOLS ---

async def obsidian_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make a request to the Obsidian Local REST API."""
    if not OBSIDIAN_AVAILABLE:
        return {"error": "Obsidian API not configured"}

    headers = {"Authorization": f"Bearer {OBSIDIAN_API_KEY}"}
    # URL-encode path segments while preserving slashes
    if endpoint.startswith("/vault/"):
        path_part = endpoint[7:]  # Remove '/vault/'
        # Encode each path segment separately to preserve directory structure
        encoded_parts = [quote(part, safe='') for part in path_part.split('/')]
        endpoint = "/vault/" + "/".join(encoded_parts)
    url = f"{OBSIDIAN_API_URL}{endpoint}"

    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                resp = await client.post(url, headers=headers, json=data)
            elif method == "PUT":
                headers["Content-Type"] = "text/markdown"
                resp = await client.put(url, headers=headers, content=data.get("content", "") if data else "")
            elif method == "PATCH":
                headers["Content-Type"] = "application/json"
                resp = await client.patch(url, headers=headers, json=data)
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return {"error": f"Unknown method: {method}"}

            if resp.status_code >= 400:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            return {"content": resp.text}

    except Exception as e:
        return {"error": str(e)}


async def execute_obsidian_list(path: str = "/") -> str:
    """List files and folders in the Obsidian vault."""
    print(f"[TOOL:OBSIDIAN] Listing: {path}")

    if not OBSIDIAN_AVAILABLE:
        return "Obsidian integration not available (API key not set)"

    # Build correct path - /vault/ for root, /vault/path/ for subdirs
    clean_path = path.strip("/")
    endpoint = f"/vault/{clean_path}/" if clean_path else "/vault/"

    result = await obsidian_request("GET", endpoint)

    if "error" in result:
        return f"Obsidian error: {result['error']}"

    files = result.get("files", [])
    if not files:
        return f"No files found in {path}"

    output = f"**Obsidian Vault ({path}):**\n"
    for f in files[:50]:  # Limit to 50 items
        icon = "📁" if f.endswith("/") else "📄"
        output += f"- {icon} {f}\n"

    if len(files) > 50:
        output += f"\n... and {len(files) - 50} more items"

    return output


async def execute_obsidian_read(path: str) -> str:
    """Read a note from the Obsidian vault."""
    print(f"[TOOL:OBSIDIAN] Reading: {path}")

    if not OBSIDIAN_AVAILABLE:
        return "Obsidian integration not available"

    # Clean path
    path = path.strip("/")
    if not path.endswith(".md"):
        path += ".md"

    result = await obsidian_request("GET", f"/vault/{path}")

    if "error" in result:
        return f"Obsidian error: {result['error']}"

    content = result.get("content", "")
    return f"**Note: {path}**\n\n{content[:5000]}"


async def execute_obsidian_search(query: str) -> str:
    """Search for notes in the Obsidian vault by listing and filtering."""
    print(f"[TOOL:OBSIDIAN] Searching: {query}")

    if not OBSIDIAN_AVAILABLE:
        return "Obsidian integration not available"

    # Get all files recursively
    async def get_all_files(path: str = "") -> list:
        endpoint = f"/vault/{path}/" if path else "/vault/"
        result = await obsidian_request("GET", endpoint)
        if "error" in result:
            return []

        files = []
        for f in result.get("files", []):
            if f.endswith("/"):
                # Recurse into directory
                subfiles = await get_all_files(f"{path}/{f.rstrip('/')}" if path else f.rstrip("/"))
                files.extend(subfiles)
            else:
                full_path = f"{path}/{f}" if path else f
                files.append(full_path)
        return files

    all_files = await get_all_files()

    # Filter files by query (case-insensitive)
    query_lower = query.lower()
    matches = [f for f in all_files if query_lower in f.lower() and f.endswith(".md")]

    if not matches:
        return f"No notes found matching '{query}'"

    output = f"**Notes matching '{query}':**\n\n"
    for match in matches[:15]:
        output += f"- 📄 {match}\n"

    if len(matches) > 15:
        output += f"\n... and {len(matches) - 15} more matches"

    return output


async def execute_obsidian_write(path: str, content: str) -> str:
    """Create or update a note in the Obsidian vault."""
    print(f"[TOOL:OBSIDIAN] Writing: {path}")

    if not OBSIDIAN_AVAILABLE:
        return "Obsidian integration not available"

    # Clean path
    path = path.strip("/")
    if not path.endswith(".md"):
        path += ".md"

    result = await obsidian_request("PUT", f"/vault/{path}", {"content": content})

    if "error" in result:
        return f"Obsidian write error: {result['error']}"

    return f"Successfully saved note: {path}"


async def execute_obsidian_append(path: str, content: str) -> str:
    """Append content to an existing note."""
    print(f"[TOOL:OBSIDIAN] Appending to: {path}")

    if not OBSIDIAN_AVAILABLE:
        return "Obsidian integration not available"

    path = path.strip("/")
    if not path.endswith(".md"):
        path += ".md"

    result = await obsidian_request("POST", f"/vault/{path}", {
        "content": content,
        "append": True
    })

    if "error" in result:
        return f"Obsidian append error: {result['error']}"

    return f"Successfully appended to: {path}"


async def execute_obsidian_daily() -> str:
    """Get or create today's daily note."""
    print("[TOOL:OBSIDIAN] Getting daily note")

    if not OBSIDIAN_AVAILABLE:
        return "Obsidian integration not available"

    result = await obsidian_request("GET", "/periodic/daily/")

    if "error" in result:
        return f"Obsidian daily note error: {result['error']}"

    content = result.get("content", "")
    path = result.get("path", "Daily Note")

    return f"**Daily Note ({path}):**\n\n{content[:3000]}"


# --- GIT/GITHUB TOOLS ---

async def execute_git_status(path: str = ".") -> str:
    """Get git status for a repository."""
    print(f"[TOOL:GIT] Status: {path}")
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True, text=True, timeout=30,
            cwd=Path(path).expanduser()
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return f"**Git Status ({path}):**\n```\n{result.stdout or 'Clean working tree'}\n```"
    except Exception as e:
        return f"Git status failed: {e}"


async def execute_git_diff(path: str = ".", staged: bool = False) -> str:
    """Get git diff for a repository."""
    print(f"[TOOL:GIT] Diff: {path} (staged={staged})")
    try:
        cmd = ["git", "diff", "--stat"]
        if staged:
            cmd.append("--cached")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                               cwd=Path(path).expanduser())
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return f"**Git Diff:**\n```\n{result.stdout[:3000] or 'No changes'}\n```"
    except Exception as e:
        return f"Git diff failed: {e}"


async def execute_git_log(path: str = ".", count: int = 10) -> str:
    """Get recent git commits."""
    print(f"[TOOL:GIT] Log: {path} (count={count})")
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--oneline", "--decorate"],
            capture_output=True, text=True, timeout=30,
            cwd=Path(path).expanduser()
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return f"**Recent Commits:**\n```\n{result.stdout}\n```"
    except Exception as e:
        return f"Git log failed: {e}"


async def execute_git_branch(path: str = ".") -> str:
    """List git branches."""
    print(f"[TOOL:GIT] Branches: {path}")
    try:
        result = subprocess.run(
            ["git", "branch", "-a", "-v"],
            capture_output=True, text=True, timeout=30,
            cwd=Path(path).expanduser()
        )
        if result.returncode != 0:
            return f"Git error: {result.stderr}"
        return f"**Git Branches:**\n```\n{result.stdout}\n```"
    except Exception as e:
        return f"Git branch failed: {e}"


async def execute_git_commit(path: str = ".", message: str = "", add_all: bool = True) -> str:
    """Create a git commit with optional staging."""
    print(f"[TOOL:GIT] Commit: {path} - {message[:50]}...")
    try:
        cwd = Path(path).expanduser()

        # Optionally stage all changes
        if add_all:
            add_result = subprocess.run(
                ["git", "add", "-A"],
                capture_output=True, text=True, timeout=30, cwd=cwd
            )
            if add_result.returncode != 0:
                return f"Git add failed: {add_result.stderr}"

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, cwd=cwd
        )
        if not status.stdout.strip():
            return "Nothing to commit - working tree clean."

        # Create commit
        if not message:
            message = "Auto-commit by BenchAI"

        result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True, text=True, timeout=30, cwd=cwd
        )
        if result.returncode != 0:
            return f"Git commit failed: {result.stderr}"

        return f"**Commit Created:**\n```\n{result.stdout}\n```"
    except Exception as e:
        return f"Git commit failed: {e}"


async def execute_git_push(path: str = ".", remote: str = "origin", branch: str = "") -> str:
    """Push commits to remote repository."""
    print(f"[TOOL:GIT] Push: {path} -> {remote}/{branch or 'current'}")
    try:
        cwd = Path(path).expanduser()

        # Get current branch if not specified
        if not branch:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=30, cwd=cwd
            )
            branch = branch_result.stdout.strip()

        result = subprocess.run(
            ["git", "push", remote, branch],
            capture_output=True, text=True, timeout=60, cwd=cwd
        )
        if result.returncode != 0:
            return f"Git push failed: {result.stderr}"

        output = result.stdout + result.stderr
        return f"**Pushed to {remote}/{branch}:**\n```\n{output}\n```"
    except Exception as e:
        return f"Git push failed: {e}"


async def execute_git_create_branch(path: str = ".", branch_name: str = "", checkout: bool = True) -> str:
    """Create a new git branch."""
    print(f"[TOOL:GIT] Create branch: {branch_name}")
    if not branch_name:
        return "Branch name is required."
    try:
        cwd = Path(path).expanduser()

        if checkout:
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                capture_output=True, text=True, timeout=30, cwd=cwd
            )
        else:
            result = subprocess.run(
                ["git", "branch", branch_name],
                capture_output=True, text=True, timeout=30, cwd=cwd
            )

        if result.returncode != 0:
            return f"Git branch creation failed: {result.stderr}"

        action = "Created and switched to" if checkout else "Created"
        return f"**{action} branch '{branch_name}'**"
    except Exception as e:
        return f"Git branch creation failed: {e}"


async def execute_git_pull(path: str = ".", remote: str = "origin", branch: str = "") -> str:
    """Pull latest changes from remote."""
    print(f"[TOOL:GIT] Pull: {remote}/{branch or 'current'}")
    try:
        cwd = Path(path).expanduser()

        cmd = ["git", "pull", remote]
        if branch:
            cmd.append(branch)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd)
        output = result.stdout + result.stderr

        if result.returncode != 0:
            return f"Git pull failed: {output}"

        return f"**Pull Complete:**\n```\n{output}\n```"
    except Exception as e:
        return f"Git pull failed: {e}"


# --- DOCKER TOOLS ---

async def execute_docker_ps(all_containers: bool = False) -> str:
    """List Docker containers."""
    print(f"[TOOL:DOCKER] PS (all={all_containers})")
    try:
        cmd = ["docker", "ps", "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"]
        if all_containers:
            cmd.insert(2, "-a")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"Docker error: {result.stderr}"
        return f"**Docker Containers:**\n```\n{result.stdout}\n```"
    except Exception as e:
        return f"Docker ps failed: {e}"


async def execute_docker_logs(container: str, lines: int = 50) -> str:
    """Get Docker container logs."""
    print(f"[TOOL:DOCKER] Logs: {container} (lines={lines})")
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return f"**Logs ({container}):**\n```\n{output[:4000]}\n```"
    except Exception as e:
        return f"Docker logs failed: {e}"


async def execute_docker_stats() -> str:
    """Get Docker container resource usage."""
    print("[TOOL:DOCKER] Stats")
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
             "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return f"Docker error: {result.stderr}"
        return f"**Container Resources:**\n```\n{result.stdout}\n```"
    except Exception as e:
        return f"Docker stats failed: {e}"


async def execute_docker_control(container: str, action: str) -> str:
    """Start, stop, or restart a Docker container."""
    print(f"[TOOL:DOCKER] {action}: {container}")
    if action not in ["start", "stop", "restart"]:
        return f"Invalid action: {action}. Use start, stop, or restart."
    try:
        result = subprocess.run(
            ["docker", action, container],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return f"Docker {action} failed: {result.stderr}"
        return f"Container '{container}' {action}ed successfully."
    except Exception as e:
        return f"Docker {action} failed: {e}"


# --- NPM/NODE TOOLS ---

async def execute_npm_run(script: str, path: str = ".") -> str:
    """Run an npm script."""
    print(f"[TOOL:NPM] Run: {script} in {path}")
    try:
        result = subprocess.run(
            ["npm", "run", script],
            capture_output=True, text=True, timeout=120,
            cwd=Path(path).expanduser()
        )
        output = result.stdout + result.stderr
        status = "Success" if result.returncode == 0 else "Failed"
        return f"**npm run {script} ({status}):**\n```\n{output[:4000]}\n```"
    except Exception as e:
        return f"npm run failed: {e}"


async def execute_npm_install(package: str = "", path: str = ".") -> str:
    """Install npm packages."""
    print(f"[TOOL:NPM] Install: {package or 'all'} in {path}")
    try:
        cmd = ["npm", "install"]
        if package:
            cmd.append(package)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                               cwd=Path(path).expanduser())
        output = result.stdout + result.stderr
        return f"**npm install ({package or 'all'}):**\n```\n{output[:3000]}\n```"
    except Exception as e:
        return f"npm install failed: {e}"


async def execute_npm_list(path: str = ".") -> str:
    """List installed npm packages."""
    print(f"[TOOL:NPM] List: {path}")
    try:
        result = subprocess.run(
            ["npm", "list", "--depth=0"],
            capture_output=True, text=True, timeout=30,
            cwd=Path(path).expanduser()
        )
        return f"**Installed Packages:**\n```\n{result.stdout[:3000]}\n```"
    except Exception as e:
        return f"npm list failed: {e}"


# --- HOMELAB TOOLS ---

async def execute_homelab_status() -> str:
    """Get comprehensive homelab status."""
    print("[TOOL:HOMELAB] Status check")
    results = []

    # System resources
    try:
        mem = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10)
        disk = subprocess.run(["df", "-h", "/", "/home"], capture_output=True, text=True, timeout=10)
        results.append(f"**Memory:**\n```\n{mem.stdout}\n```")
        results.append(f"**Disk:**\n```\n{disk.stdout}\n```")
    except Exception as e:
        results.append(f"System info error: {e}")

    # GPU status
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if gpu.returncode == 0:
            results.append(f"**GPU:** {gpu.stdout.strip()}")
    except:
        pass

    # Docker containers
    try:
        containers = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}: {{.Status}}"],
            capture_output=True, text=True, timeout=10
        )
        if containers.returncode == 0:
            results.append(f"**Containers:**\n```\n{containers.stdout}\n```")
    except:
        pass

    # Running LLM models
    try:
        ollama = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if ollama.returncode == 0:
            results.append(f"**Ollama Models:**\n```\n{ollama.stdout}\n```")
    except:
        pass

    return "\n".join(results) if results else "Could not retrieve homelab status"


async def execute_service_health() -> str:
    """Check health of key homelab services."""
    print("[TOOL:HOMELAB] Service health check")
    services = [
        ("BenchAI Router", "http://localhost:8085/health"),
        ("Open WebUI", "http://localhost:8080/health"),
        ("Searxng", "http://localhost:8081/healthz"),
        ("Jellyfin", "http://localhost:8096/health"),
    ]

    results = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services:
            try:
                resp = await client.get(url)
                status = "✅ UP" if resp.status_code == 200 else f"⚠️ {resp.status_code}"
            except:
                status = "❌ DOWN"
            results.append(f"- {name}: {status}")

    return "**Service Health:**\n" + "\n".join(results)


# --- PYTHON DEV TOOLS ---

async def execute_pytest(path: str = ".", args: str = "") -> str:
    """Run pytest on a Python project."""
    print(f"[TOOL:PYTEST] Running tests in {path}")
    try:
        cmd = f"cd {path} && python3 -m pytest {args} --tb=short -q"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        status = "PASSED" if result.returncode == 0 else "FAILED"
        return f"**Pytest ({status}):**\n```\n{output[:4000]}\n```"
    except Exception as e:
        return f"Pytest failed: {e}"


async def execute_python_lint(path: str, tool: str = "ruff") -> str:
    """Lint Python code with ruff, black, or mypy."""
    print(f"[TOOL:LINT] {tool} on {path}")
    try:
        if tool == "black":
            cmd = f"python3 -m black --check --diff {path}"
        elif tool == "mypy":
            cmd = f"python3 -m mypy {path} --ignore-missing-imports"
        else:  # ruff
            cmd = f"python3 -m ruff check {path}"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        status = "Clean" if result.returncode == 0 else "Issues found"
        return f"**{tool.capitalize()} ({status}):**\n```\n{output[:3000]}\n```"
    except Exception as e:
        return f"Lint failed: {e}"


async def execute_python_format(path: str) -> str:
    """Format Python code with black."""
    print(f"[TOOL:FORMAT] Formatting {path}")
    try:
        result = subprocess.run(
            f"python3 -m black {path}",
            shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        return f"**Black Format:**\n```\n{output[:2000]}\n```"
    except Exception as e:
        return f"Format failed: {e}"


# --- GITHUB TOOLS ---

async def execute_gh_pr_list(repo: str = "") -> str:
    """List GitHub pull requests."""
    print(f"[TOOL:GITHUB] Listing PRs for {repo or 'current repo'}")
    try:
        cmd = f"gh pr list {f'-R {repo}' if repo else ''} --limit 10"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"GitHub error: {result.stderr}"
        return f"**Pull Requests:**\n```\n{result.stdout or 'No open PRs'}\n```"
    except Exception as e:
        return f"GitHub PR list failed: {e}"


async def execute_gh_issue_list(repo: str = "") -> str:
    """List GitHub issues."""
    print(f"[TOOL:GITHUB] Listing issues for {repo or 'current repo'}")
    try:
        cmd = f"gh issue list {f'-R {repo}' if repo else ''} --limit 10"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"GitHub error: {result.stderr}"
        return f"**Issues:**\n```\n{result.stdout or 'No open issues'}\n```"
    except Exception as e:
        return f"GitHub issue list failed: {e}"


async def execute_gh_repo_view(repo: str) -> str:
    """View GitHub repository info."""
    print(f"[TOOL:GITHUB] Viewing repo {repo}")
    try:
        result = subprocess.run(
            f"gh repo view {repo} --json name,description,stargazerCount,forkCount,url",
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return f"GitHub error: {result.stderr}"
        import json
        data = json.loads(result.stdout)
        return f"""**{data.get('name')}**
- URL: {data.get('url')}
- Description: {data.get('description', 'N/A')}
- Stars: {data.get('stargazerCount', 0)} | Forks: {data.get('forkCount', 0)}"""
    except Exception as e:
        return f"GitHub repo view failed: {e}"


async def execute_gh_workflow_list(repo: str = "") -> str:
    """List GitHub Actions workflow runs."""
    print(f"[TOOL:GITHUB] Listing workflows for {repo or 'current repo'}")
    try:
        cmd = f"gh run list {f'-R {repo}' if repo else ''} --limit 5"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"GitHub error: {result.stderr}"
        return f"**Recent Workflow Runs:**\n```\n{result.stdout or 'No runs'}\n```"
    except Exception as e:
        return f"GitHub workflow list failed: {e}"


async def execute_gh_pr_create(title: str, body: str = "", base: str = "main", draft: bool = False) -> str:
    """Create a GitHub pull request."""
    print(f"[TOOL:GITHUB] Creating PR: {title}")
    if not title:
        return "PR title is required."
    try:
        cmd = ["gh", "pr", "create", "--title", title, "--base", base]
        if body:
            cmd.extend(["--body", body])
        if draft:
            cmd.append("--draft")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return f"GitHub PR creation failed: {result.stderr}"

        return f"**Pull Request Created:**\n{result.stdout}"
    except Exception as e:
        return f"GitHub PR creation failed: {e}"


async def execute_gh_issue_create(title: str, body: str = "", labels: str = "") -> str:
    """Create a GitHub issue."""
    print(f"[TOOL:GITHUB] Creating issue: {title}")
    if not title:
        return "Issue title is required."
    try:
        cmd = ["gh", "issue", "create", "--title", title]
        if body:
            cmd.extend(["--body", body])
        if labels:
            cmd.extend(["--label", labels])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return f"GitHub issue creation failed: {result.stderr}"

        return f"**Issue Created:**\n{result.stdout}"
    except Exception as e:
        return f"GitHub issue creation failed: {e}"


async def execute_gh_pr_merge(pr_number: int, method: str = "merge") -> str:
    """Merge a GitHub pull request."""
    print(f"[TOOL:GITHUB] Merging PR #{pr_number}")
    try:
        cmd = ["gh", "pr", "merge", str(pr_number), f"--{method}"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return f"GitHub PR merge failed: {result.stderr}"

        return f"**PR #{pr_number} Merged:**\n{result.stdout}"
    except Exception as e:
        return f"GitHub PR merge failed: {e}"


# --- ADVANCED SYSTEM TOOLS ---

async def execute_system_info() -> str:
    """Get comprehensive system information."""
    print("[TOOL:SYSTEM] Getting system info")
    results = []

    # CPU info
    try:
        cpu = subprocess.run("lscpu | grep -E 'Model name|CPU\\(s\\)|Thread|Core'",
                           shell=True, capture_output=True, text=True, timeout=10)
        results.append(f"**CPU:**\n```\n{cpu.stdout}\n```")
    except: pass

    # Memory
    try:
        mem = subprocess.run("free -h", shell=True, capture_output=True, text=True, timeout=10)
        results.append(f"**Memory:**\n```\n{mem.stdout}\n```")
    except: pass

    # Disk
    try:
        disk = subprocess.run("df -h | grep -E '^/dev|Filesystem'",
                            shell=True, capture_output=True, text=True, timeout=10)
        results.append(f"**Disks:**\n```\n{disk.stdout}\n```")
    except: pass

    # GPU
    try:
        gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader",
                           shell=True, capture_output=True, text=True, timeout=10)
        if gpu.returncode == 0:
            results.append(f"**GPU:** {gpu.stdout.strip()}")
    except: pass

    # Network
    try:
        net = subprocess.run("ip -4 addr show | grep -E 'inet ' | head -5",
                           shell=True, capture_output=True, text=True, timeout=10)
        results.append(f"**Network:**\n```\n{net.stdout}\n```")
    except: pass

    return "\n".join(results) if results else "Could not get system info"


async def execute_process_list(filter_str: str = "") -> str:
    """List running processes, optionally filtered."""
    print(f"[TOOL:SYSTEM] Process list (filter: {filter_str or 'none'})")
    try:
        if filter_str:
            cmd = f"ps aux | grep -i '{filter_str}' | grep -v grep | head -20"
        else:
            cmd = "ps aux --sort=-%mem | head -15"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return f"**Processes:**\n```\n{result.stdout}\n```"
    except Exception as e:
        return f"Process list failed: {e}"


# --- BROWSER AUTOMATION TOOLS ---

def _sync_browser_fetch(url: str, wait_for: str = None, screenshot: bool = False) -> str:
    """Sync helper for browser fetch."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)

        if wait_for:
            page.wait_for_selector(wait_for, timeout=10000)

        title = page.title()
        text = page.evaluate("() => document.body.innerText")

        result = f"**{title}**\nURL: {url}\n\n{text[:5000]}"

        if screenshot:
            screenshot_path = f"/tmp/benchai_screenshot_{int(time.time())}.png"
            page.screenshot(path=screenshot_path)
            result += f"\n\nScreenshot saved: {screenshot_path}"

        browser.close()
        return result


async def execute_browser_fetch(url: str, wait_for: str = None, screenshot: bool = False) -> str:
    """Fetch a webpage using Playwright with JavaScript rendering."""
    print(f"[TOOL:BROWSER] Fetching: {url}")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_browser_fetch, url, wait_for, screenshot)
    except Exception as e:
        return f"Browser fetch failed: {e}"


def _sync_browser_extract(url: str, selector: str) -> str:
    """Sync helper for browser extract."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)

        elements = page.query_selector_all(selector)
        texts = [el.inner_text() for el in elements[:20]]

        browser.close()
        return f"**Found {len(elements)} elements matching '{selector}':**\n" + "\n".join(f"- {t[:200]}" for t in texts)


async def execute_browser_extract(url: str, selector: str) -> str:
    """Extract specific elements from a webpage using CSS selectors."""
    print(f"[TOOL:BROWSER] Extracting {selector} from {url}")
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_browser_extract, url, selector)
    except Exception as e:
        return f"Browser extract failed: {e}"


def _sync_browser_screenshot(url: str, full_page: bool = False) -> str:
    """Sync helper for browser screenshot."""
    from playwright.sync_api import sync_playwright

    screenshot_path = f"/tmp/benchai_screenshot_{int(time.time())}.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)
        page.screenshot(path=screenshot_path, full_page=full_page)
        browser.close()

    return screenshot_path


async def execute_browser_screenshot(url: str, full_page: bool = False) -> str:
    """Take a screenshot of a webpage."""
    print(f"[TOOL:BROWSER] Screenshot: {url}")
    try:
        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(None, _sync_browser_screenshot, url, full_page)
        return f"Screenshot saved: {path}"
    except Exception as e:
        return f"Screenshot failed: {e}"


# --- SPREADSHEET TOOLS ---

async def execute_read_excel(path: str, sheet: str = None) -> str:
    """Read an Excel file and return its contents."""
    print(f"[TOOL:EXCEL] Reading: {path}")
    try:
        import pandas as pd

        df = pd.read_excel(path, sheet_name=sheet)
        info = f"**Excel: {path}**\n"
        info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        info += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        info += f"**Preview:**\n```\n{df.head(20).to_string()}\n```"
        return info

    except Exception as e:
        return f"Excel read failed: {e}"


async def execute_read_csv(path: str) -> str:
    """Read a CSV file and return its contents."""
    print(f"[TOOL:CSV] Reading: {path}")
    try:
        import pandas as pd

        df = pd.read_csv(path)
        info = f"**CSV: {path}**\n"
        info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        info += f"Columns: {', '.join(df.columns.tolist())}\n\n"
        info += f"**Preview:**\n```\n{df.head(20).to_string()}\n```"
        return info

    except Exception as e:
        return f"CSV read failed: {e}"


async def execute_analyze_data(path: str, query: str) -> str:
    """Analyze a CSV/Excel file with pandas based on a natural language query."""
    print(f"[TOOL:DATA] Analyzing: {path} - {query}")
    try:
        import pandas as pd

        # Load the file
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        result = f"**Data Analysis: {path}**\n"
        result += f"Shape: {df.shape}\n"
        result += f"Columns: {list(df.columns)}\n\n"

        # Basic statistics
        result += f"**Statistics:**\n```\n{df.describe().to_string()}\n```\n\n"

        # Data types
        result += f"**Data Types:**\n```\n{df.dtypes.to_string()}\n```"

        return result

    except Exception as e:
        return f"Data analysis failed: {e}"


async def execute_create_excel(path: str, data: dict) -> str:
    """Create an Excel file from data."""
    print(f"[TOOL:EXCEL] Creating: {path}")
    try:
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_excel(path, index=False)
        return f"Excel file created: {path} ({df.shape[0]} rows × {df.shape[1]} columns)"

    except Exception as e:
        return f"Excel creation failed: {e}"


# --- LIBRARY DOCS TOOLS ---

async def execute_context7_search(library: str) -> str:
    """Search for library documentation using Context7."""
    print(f"[TOOL:DOCS] Searching Context7 for: {library}")
    try:
        result = subprocess.run(
            ["npx", "context7", "search", library],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return f"**Library Search: {library}**\n```\n{output[:3000]}\n```"
    except Exception as e:
        return f"Context7 search failed: {e}"


async def execute_context7_docs(library: str, query: str) -> str:
    """Get specific documentation from a library using Context7."""
    print(f"[TOOL:DOCS] Context7: {library} - {query}")
    try:
        result = subprocess.run(
            ["npx", "context7", library, query],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return f"**{library} Documentation: {query}**\n```\n{output[:4000]}\n```"
    except Exception as e:
        return f"Context7 docs failed: {e}"


# --- MCP SERVER BUILDER ---

async def execute_create_mcp_server(name: str, description: str, tools: list) -> str:
    """Generate a basic MCP server template."""
    print(f"[TOOL:MCP] Creating server: {name}")

    template = f'''#!/usr/bin/env python3
"""
{name} - MCP Server
{description}

Generated by BenchAI MCP Builder
"""

import asyncio
import json
from typing import Any

# MCP Server Template
class MCPServer:
    def __init__(self):
        self.name = "{name}"
        self.version = "1.0.0"
        self.tools = {json.dumps(tools, indent=8)}

    async def handle_request(self, method: str, params: dict) -> dict:
        if method == "initialize":
            return {{
                "protocolVersion": "2024-11-05",
                "serverInfo": {{"name": self.name, "version": self.version}},
                "capabilities": {{"tools": {{}}}}
            }}
        elif method == "tools/list":
            return {{"tools": self.tools}}
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {{}})
            return await self.execute_tool(tool_name, args)
        return {{"error": "Unknown method"}}

    async def execute_tool(self, name: str, args: dict) -> dict:
        # Implement your tool logic here
        return {{"content": [{{"type": "text", "text": f"Tool {{name}} executed with {{args}}"}}]}}

    async def run(self):
        """Run the MCP server using stdio transport."""
        import sys
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            try:
                request = json.loads(line)
                response = await self.handle_request(request.get("method"), request.get("params", {{}}))
                response["jsonrpc"] = "2.0"
                response["id"] = request.get("id")
                print(json.dumps(response), flush=True)
            except Exception as e:
                print(json.dumps({{"jsonrpc": "2.0", "error": {{"message": str(e)}}, "id": None}}), flush=True)

if __name__ == "__main__":
    server = MCPServer()
    asyncio.run(server.run())
'''

    # Save the template
    server_dir = Path.home() / "mcp-servers" / name
    server_dir.mkdir(parents=True, exist_ok=True)
    server_file = server_dir / "server.py"

    with open(server_file, 'w') as f:
        f.write(template)

    # Make executable
    server_file.chmod(0o755)

    return f"""**MCP Server Created: {name}**

Location: {server_file}

To use this server:
1. Edit {server_file} to implement your tool logic
2. Add to Claude Code config:
   ```json
   {{
     "mcpServers": {{
       "{name}": {{
         "command": "python3",
         "args": ["{server_file}"]
       }}
     }}
   }}
   ```
3. Restart Claude Code

Tools defined: {', '.join(t.get('name', 'unnamed') for t in tools)}"""


# --- PLANNER ---

async def generate_plan(user_query: str, has_image: bool, session_context: str = "") -> List[Dict]:
    """Generate an execution plan using the planner model.

    Args:
        user_query: The user's request
        has_image: Whether an image is attached
        session_context: Context from previous turns in this conversation
    """
    print(f"[PLANNER] Generating plan... (has_image={has_image}, has_session_context={bool(session_context)})")

    # Filter out Open WebUI system requests (title generation, etc.)
    system_request_patterns = [
        r'generate.*title',
        r'summariz.*chat',
        r'create.*emoji.*title',
        r'^[A-Z][a-z]+ \d+ words?',  # "Generate 5 words"
    ]
    for pattern in system_request_patterns:
        if re.search(pattern, user_query, re.IGNORECASE):
            print(f"[PLANNER] Detected system request, using simple reasoning")
            return [{"tool": "reasoning", "args": user_query}]

    # Check for Claude escalation triggers - complex tasks that benefit from Claude
    for trigger in CLAUDE_ESCALATION_TRIGGERS:
        if re.search(trigger, user_query, re.IGNORECASE):
            print(f"[PLANNER] Claude escalation triggered: {trigger}")
            return [{"tool": "claude_assist", "args": {"prompt": user_query, "context": session_context}}]

    if not await manager.start_model("planner"):
        return [{"tool": "reasoning", "args": user_query}]

    # Get memory context
    memory_context = await memory_manager.get_context_summary()

    # Combine memory context with session context
    if session_context:
        memory_context = f"{session_context}\n\n{memory_context}" if memory_context else session_context

    system_prompt = f"""You are an AI Orchestrator. Analyze the user's request and create an execution plan.

**Available Tools:**

*General:*
1. `web_search(query)` - Search the internet for information
2. `vision_analysis(prompt)` - Analyze attached images (OCR, describe, extract info)
3. `generate_image(prompt)` - Create images from text descriptions
4. `read_file(path)` - Read content from files (text, code, PDF)
5. `shell(command)` - Run shell commands (ls, git, docker, etc.)
6. `memory_search(query)` - Search user's saved memories/preferences
7. `rag_search(query)` - Search indexed documents/codebase
8. `reasoning(prompt)` - Think through problems, synthesize information
9. `claude_assist(prompt, context)` - Call Claude (mastermind) for complex reasoning, code review, architectural decisions

*Code Generation (uses DeepSeek Coder):*
10. `code_task(task, language)` - Generate and run code. DeepSeek writes the code.
   - Languages: python, cpp, c, cuda, javascript, typescript, rust, go
   - Example: {{"tool": "code_task", "args": {{"task": "fibonacci sequence", "language": "cpp"}}}}
11. `run_code(code)` - Execute user-provided Python code directly (no generation)

*Engineering/Learning Tools (uses Claude when available):*
12. `code_review(code, language, focus)` - Review code for issues and best practices
   - Focus: 'all', 'security', 'performance', 'style', 'bugs'
   - Example: {{"tool": "code_review", "args": {{"code": "...", "language": "python", "focus": "security"}}}}
13. `explain_concept(concept, level)` - Explain programming concepts with examples
   - Levels: 'beginner', 'intermediate', 'advanced'
   - Example: {{"tool": "explain_concept", "args": {{"concept": "recursion", "level": "beginner"}}}}
14. `generate_tests(code, language, framework)` - Generate unit tests for code
   - Auto-detects framework (pytest, jest, etc.)
15. `debug_help(error_message, code, language)` - Help debug errors with explanations

*Learning & Study Tools:*
16. `create_flashcard(front, back, topic)` - Create a flashcard for spaced repetition
   - Example: {{"tool": "create_flashcard", "args": {{"front": "What is a closure?", "back": "A function that captures variables from its enclosing scope", "topic": "javascript"}}}}
17. `study_flashcards(topic)` - Start a study session with your flashcards
   - Leave topic empty for all flashcards
18. `generate_quiz(topic, num_questions, difficulty)` - Generate a quiz on any programming topic
   - Difficulty: 'beginner', 'intermediate', 'advanced'
19. `learning_path(goal)` - Get a personalized learning roadmap
   - Example: {{"tool": "learning_path", "args": "become a full-stack developer"}}
20. `code_challenge(topic, difficulty)` - Get a coding challenge for practice
21. `explain_error(error_type, language)` - Deep dive into common error types

*C/C++ Build Tools:*
22. `cmake_build(project_dir)` - Build CMake project (configure + make)
23. `analyze_cpp(code)` - Static analysis (g++ warnings + cppcheck)
24. `debug_info(binary_path)` - Analyze compiled binary (symbols, libs, size)

*Obsidian Knowledge Base:*
19. `obsidian_list(path)` - List files/folders in vault (default: root)
20. `obsidian_read(path)` - Read a note's content
21. `obsidian_search(query)` - Search notes by content
22. `obsidian_write(path, content)` - Create or update a note
23. `obsidian_append(path, content)` - Append to existing note
24. `obsidian_daily()` - Get today's daily note

*Git/GitHub:*
25. `git_status(path)` - Get git status (default: current dir)
26. `git_diff(path, staged)` - Show changes (staged=true for staged only)
27. `git_log(path, count)` - Recent commits (default: 10)
28. `git_branch(path)` - List all branches
29. `git_commit(path, message, add_all)` - Create a commit (add_all=true stages all changes)
30. `git_push(path, remote, branch)` - Push to remote (default: origin/current)
31. `git_pull(path, remote, branch)` - Pull from remote
32. `git_create_branch(path, branch_name, checkout)` - Create new branch
33. `gh_pr_create(title, body, base, draft)` - Create a pull request
34. `gh_issue_create(title, body, labels)` - Create an issue
35. `gh_pr_merge(pr_number, method)` - Merge a PR (method: merge/squash/rebase)

*Docker Management:*
29. `docker_ps(all)` - List containers (all=true includes stopped)
30. `docker_logs(container, lines)` - Get container logs (default: 50 lines)
31. `docker_stats()` - Container resource usage (CPU, memory, network)
32. `docker_control(container, action)` - Start/stop/restart container

*Web Development:*
33. `npm_run(script, path)` - Run npm script in project
34. `npm_install(package, path)` - Install packages (empty = all)
35. `npm_list(path)` - List installed packages

*Homelab:*
36. `homelab_status()` - System resources, GPU, containers, models
37. `service_health()` - Check health of all homelab services
38. `system_info()` - Detailed CPU, memory, disk, GPU, network info
39. `process_list(filter)` - List processes (optional filter string)

*Python Development:*
40. `pytest(path, args)` - Run pytest tests
41. `python_lint(path, tool)` - Lint with ruff/black/mypy
42. `python_format(path)` - Format with black

*GitHub:*
43. `gh_pr_list(repo)` - List pull requests
44. `gh_issue_list(repo)` - List issues
45. `gh_repo_view(repo)` - View repo info
46. `gh_workflow_list(repo)` - List GitHub Actions runs

*Browser Automation (Playwright):*
47. `browser_fetch(url, wait_for)` - Fetch webpage with JS rendering
48. `browser_extract(url, selector)` - Extract elements by CSS selector
49. `browser_screenshot(url, full_page)` - Screenshot a webpage

*Spreadsheet/Data:*
50. `read_excel(path, sheet)` - Read Excel file
51. `read_csv(path)` - Read CSV file
52. `analyze_data(path, query)` - Analyze CSV/Excel with pandas
53. `create_excel(path, data)` - Create Excel from data

*Library Docs:*
54. `context7_search(library)` - Search library documentation
55. `context7_docs(library, query)` - Get specific docs

*MCP Builder:*
56. `create_mcp_server(name, description, tools)` - Generate MCP server template

*Email (Proton):*
57. `email_list(folder, count)` - List recent emails (default: INBOX, 10)
58. `email_read(email_id, folder)` - Read full email by ID
59. `email_send(to, subject, body, cc)` - Send email via Proton
60. `email_search(query, folder)` - Search emails by text
61. `email_unread(folder)` - Get unread count

*API Testing:*
62. `api_request(method, url, headers, body)` - Make HTTP request (GET/POST/PUT/DELETE)
63. `api_test_suite(tests)` - Run multiple API tests

*Diagrams:*
64. `create_diagram(type, code, output_path)` - Create Mermaid diagram
65. `flowchart(description)` - Generate flowchart template

*PDF Tools:*
66. `pdf_read(path, pages)` - Read PDF text content
67. `pdf_merge(files, output)` - Merge multiple PDFs
68. `pdf_create(content, output, title)` - Create PDF from text

*Transcription:*
69. `transcribe(audio_path, language)` - Transcribe audio with Whisper

*Database:*
70. `sql_query(database, query)` - Execute SQL on SQLite DB
71. `db_schema(database)` - Get database schema

{memory_context}

**User Request:** "{user_query}"
**Image Attached:** {has_image}

**Rules:**
1. Return ONLY a valid JSON array of steps
2. Each step: {{"tool": "tool_name", "args": ...}}
3. For PARALLEL execution, wrap steps in: {{"parallel": [step1, step2, ...]}}
4. For image generation requests, use `generate_image` as the final step
5. For ALL coding tasks, use `code_task` with task description and language - DO NOT write code yourself!
6. Only use `run_code` when user provides literal Python code to execute
7. If an image is attached, start with `vision_analysis`
8. Use `memory_search` when user references past conversations
9. Use `rag_search` when user asks about their codebase/documents
10. Use `cmake_build` for CMake projects, not shell commands
11. **Use `web_search` for ANY request about current events, news, prices, latest info, real-world facts**
12. **NEVER use `reasoning` when a specific tool exists for the task!**
13. **CRITICAL ROUTING:**
    - Docker/containers → `docker_ps`, `docker_logs`, `docker_stats`, `docker_control`
    - Git/commits/branches → `git_status`, `git_log`, `git_diff`, `git_branch`, `git_commit`, `git_push`
    - GitHub/PRs/issues → `gh_pr_list`, `gh_issue_list`, `gh_pr_create`
    - System/homelab → `homelab_status`, `system_info`, `service_health`
    - Files/directories → `read_file`, `shell` with ls/find
    - Memory/remember → `memory_search`
14. Only use `reasoning` for: synthesis, explanation, general knowledge questions, opinions

**Example Plans:**
- [Image attached] "Solve this" → [{{"tool": "vision_analysis", "args": "Extract and solve the problem"}}, {{"tool": "code_task", "args": {{"task": "verify the solution", "language": "python"}}}}]
- [Image attached] "What's in this image?" → [{{"tool": "vision_analysis", "args": "Describe the image in detail"}}]
- [Image attached] "Check my work" → [{{"tool": "vision_analysis", "args": "Read the problem and solution, verify correctness"}}]
- "Calculate 25!" → [{{"tool": "code_task", "args": {{"task": "calculate factorial of 25", "language": "python"}}}}]
- "Search X, Y, Z" → [{{"parallel": [{{"tool": "web_search", "args": "X"}}, {{"tool": "web_search", "args": "Y"}}, {{"tool": "web_search", "args": "Z"}}]}}, {{"tool": "reasoning", "args": "Synthesize"}}]
- "What's the latest on RTX 5090?" → [{{"tool": "web_search", "args": "RTX 5090 specs price release date 2025"}}]
- "Search the web for Python tutorials" → [{{"tool": "web_search", "args": "Python tutorials beginners 2025"}}]
- "Find information about..." → [{{"tool": "web_search", "args": "..."}}]
- "What is the current price of Bitcoin?" → [{{"tool": "web_search", "args": "Bitcoin price today"}}]
- "What did I tell you about my project?" → [{{"tool": "memory_search", "args": "project"}}]
- "Write hello world in C++" → [{{"tool": "code_task", "args": {{"task": "print hello world", "language": "cpp"}}}}]
- "Write Fibonacci in CUDA" → [{{"tool": "code_task", "args": {{"task": "compute first 20 fibonacci numbers using GPU", "language": "cuda"}}}}]
- "Build my CMake project" → [{{"tool": "cmake_build", "args": "/path/to/project"}}]
- "Create a React component" → [{{"tool": "code_task", "args": {{"task": "create a button component with click counter", "language": "typescript"}}}}]
- "What notes do I have about Python?" → [{{"tool": "obsidian_search", "args": "python"}}]
- "Read my project notes" → [{{"tool": "obsidian_read", "args": "Projects/MyProject"}}]
- "Save this to my vault" → [{{"tool": "obsidian_write", "args": {{"path": "Notes/NewNote", "content": "..."}}}}]
- "What's the git status?" → [{{"tool": "git_status", "args": "."}}]
- "Show recent commits" → [{{"tool": "git_log", "args": {{"path": ".", "count": 5}}}}]
- "List running containers" → [{{"tool": "docker_ps", "args": false}}]
- "Get jellyfin logs" → [{{"tool": "docker_logs", "args": {{"container": "jellyfin", "lines": 100}}}}]
- "Restart sonarr" → [{{"tool": "docker_control", "args": {{"container": "sonarr", "action": "restart"}}}}]
- "Check homelab status" → [{{"tool": "homelab_status", "args": null}}]
- "Are all services running?" → [{{"tool": "service_health", "args": null}}]
- "Run npm build" → [{{"tool": "npm_run", "args": {{"script": "build", "path": "."}}}}]
- "Review this architecture" → [{{"tool": "claude_assist", "args": {{"prompt": "Review this architecture design", "context": "..."}}}}]
- "Help me design a complex system" → [{{"tool": "claude_assist", "args": "Design guidance for complex multi-service architecture"}}]
- "Create a flashcard about closures" → [{{"tool": "create_flashcard", "args": {{"front": "What is a closure in JavaScript?", "back": "A function that has access to variables from its outer (enclosing) function scope, even after the outer function has returned.", "topic": "javascript"}}}}]
- "Quiz me on Python basics" → [{{"tool": "generate_quiz", "args": {{"topic": "Python basics", "num_questions": 5, "difficulty": "beginner"}}}}]
- "Give me a learning path for machine learning" → [{{"tool": "learning_path", "args": "learn machine learning from scratch"}}]
- "Give me a coding challenge about recursion" → [{{"tool": "code_challenge", "args": {{"topic": "recursion", "difficulty": "medium"}}}}]
- "Explain NullPointerException in Java" → [{{"tool": "explain_error", "args": {{"error_type": "NullPointerException", "language": "java"}}}}]

**Follow-up Examples (when Previous Tool Results are provided):**
- [Has vision_analysis result] "Now write the code" → [{{"tool": "code_task", "args": {{"task": "implement based on the vision analysis above", "language": "python"}}}}]
- [Has vision_analysis result] "Explain the solution" → [{{"tool": "reasoning", "args": "Explain the problem and solution from the vision analysis"}}]
- [Has code_task result] "Make it better" → [{{"tool": "code_task", "args": {{"task": "improve the previous code", "language": "python"}}}}]

Respond with ONLY the JSON array:"""

    port = MODELS["planner"]["port"]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": system_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 512
                }
            )

            if resp.status_code != 200:
                print(f"[PLANNER] HTTP error: {resp.status_code}")
                return [{"tool": "reasoning", "args": user_query}]

            content = resp.json()["choices"][0]["message"]["content"]

            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()
            plan = json.loads(content)

            if not isinstance(plan, list):
                plan = [plan]

            # Force vision_analysis first if image is attached but planner missed it
            if has_image:
                has_vision_step = any(
                    (isinstance(s, dict) and s.get("tool") == "vision_analysis")
                    for s in plan
                )
                if not has_vision_step:
                    print("[PLANNER] Image detected but no vision step - adding vision_analysis")
                    plan.insert(0, {"tool": "vision_analysis", "args": user_query})

            # Post-process: Fix incorrect "reasoning" routing
            plan = fix_reasoning_routing(plan, user_query)

            return plan

    except json.JSONDecodeError as e:
        print(f"[PLANNER] JSON parse error: {e}")
        # If image attached, default to vision analysis
        if has_image:
            return [{"tool": "vision_analysis", "args": user_query}]
        return keyword_based_routing(user_query)
    except Exception as e:
        print(f"[PLANNER] Error: {e}")
        if has_image:
            return [{"tool": "vision_analysis", "args": user_query}]
        return keyword_based_routing(user_query)

def keyword_based_routing(query: str) -> List[Dict]:
    """Fallback routing based on keywords when planner fails."""
    q = query.lower()

    # Docker
    if any(kw in q for kw in ['docker', 'container', 'containers']):
        if any(kw in q for kw in ['log', 'logs']):
            return [{"tool": "docker_logs", "args": {"container": "", "lines": 50}}]
        if any(kw in q for kw in ['stat', 'stats', 'usage', 'resource']):
            return [{"tool": "docker_stats", "args": None}]
        if any(kw in q for kw in ['restart', 'stop', 'start']):
            return [{"tool": "docker_control", "args": {"container": "", "action": "restart"}}]
        return [{"tool": "docker_ps", "args": False}]

    # Git
    if any(kw in q for kw in ['git status', 'git diff', 'commit', 'branch']):
        if 'status' in q:
            return [{"tool": "git_status", "args": "."}]
        if 'diff' in q:
            return [{"tool": "git_diff", "args": {"path": ".", "staged": False}}]
        if 'log' in q or 'commit' in q:
            return [{"tool": "git_log", "args": {"path": ".", "count": 10}}]
        if 'branch' in q:
            return [{"tool": "git_branch", "args": "."}]

    # System/Homelab
    if any(kw in q for kw in ['homelab', 'system', 'service', 'health', 'status']):
        if 'service' in q or 'health' in q:
            return [{"tool": "service_health", "args": None}]
        return [{"tool": "homelab_status", "args": None}]

    # Web search triggers
    if any(kw in q for kw in ['search', 'find', 'look up', 'what is', 'latest', 'current', 'price', 'news']):
        return [{"tool": "web_search", "args": query}]

    # Memory
    if any(kw in q for kw in ['remember', 'recall', 'told you', 'said', 'memory']):
        return [{"tool": "memory_search", "args": query}]

    # Default to reasoning
    return [{"tool": "reasoning", "args": query}]

def fix_reasoning_routing(plan: List[Dict], query: str) -> List[Dict]:
    """Fix plans that incorrectly use 'reasoning' for specific tool tasks."""
    if not plan or len(plan) != 1:
        return plan

    step = plan[0]
    if step.get("tool") != "reasoning":
        return plan

    # Check if this should be routed to a specific tool
    corrected = keyword_based_routing(query)
    if corrected[0].get("tool") != "reasoning":
        print(f"[PLANNER] Correcting routing: reasoning -> {corrected[0].get('tool')}")
        return corrected

    return plan

def complexity_score(query: str) -> float:
    """Calculate query complexity for smarter routing (from deep research).

    Score 0-1 where higher = more complex, needs smarter model.
    """
    factors = {
        "length": min(len(query) / 1000, 0.3),  # Long queries are complex
        "code_blocks": min(query.count("```") * 0.15, 0.3),  # Code blocks
        "questions": min(query.count("?") * 0.1, 0.2),  # Multiple questions
        "technical": 0.1 if any(t in query.lower() for t in [
            "algorithm", "architecture", "optimize", "implement", "refactor",
            "database", "api", "async", "concurrent", "performance"
        ]) else 0,
        "multi_step": 0.15 if any(w in query.lower() for w in [
            "then", "after that", "next", "finally", "step", "first"
        ]) else 0,
    }
    return min(sum(factors.values()), 1.0)

def detect_intent(query: str, has_image: bool) -> str:
    """Quickly detect the primary intent without using the planner."""
    if IMAGE_GEN_PATTERNS.search(query):
        return "image_gen"
    if has_image:
        return "vision"
    if MATH_PATTERNS.search(query):
        return "code"
    if CODE_PATTERNS.search(query):
        return "code"
    if MEMORY_PATTERNS.search(query):
        return "memory"
    if RESEARCH_PATTERNS.search(query):
        return "research"

    # Use complexity scoring for general queries (from deep research)
    score = complexity_score(query)
    if score > 0.5:
        print(f"[ROUTING] Complex query (score={score:.2f}) → planner")
        return "planner"  # Complex general → use smarter model

    return "general"

# --- PARALLEL EXECUTOR ---

async def execute_parallel_steps(steps: List[Dict], messages_dicts: List[Dict], has_image: bool) -> Dict[str, str]:
    """Execute multiple steps in parallel."""
    print(f"[PARALLEL] Executing {len(steps)} steps concurrently...")

    async def execute_single(step: Dict) -> tuple:
        tool = step.get("tool") or step.get("step") or step.get("action")
        args = step.get("args") or step.get("query") or step.get("prompt") or ""

        result = ""
        try:
            if tool == "web_search":
                result = await execute_web_search(args)
            elif tool == "claude_assist":
                # Claude mastermind - for complex reasoning
                if isinstance(args, dict):
                    prompt = args.get("prompt") or args.get("task") or args.get("query") or str(args)
                    ctx = args.get("context", "")
                else:
                    prompt = str(args)
                    ctx = ""
                result = await execute_claude_assist(prompt, ctx)
            elif tool == "memory_search":
                result = await execute_memory_search(args)
            elif tool == "rag_search":
                result = await execute_rag_search(args)
            elif tool == "read_file":
                result = await execute_file_read(args)
            elif tool == "shell":
                result = await execute_shell(args)
            # C/C++ tools
            elif tool == "run_cpp":
                result = await execute_run_cpp(args)
            elif tool == "run_c":
                result = await execute_run_c(args)
            elif tool == "compile_cpp":
                result = await execute_compile_cpp(args)
            elif tool == "compile_c":
                result = await execute_compile_c(args)
            elif tool == "cmake_build":
                result = await execute_cmake_build(args)
            elif tool == "analyze_cpp":
                result = await execute_analyze_cpp(args)
            elif tool == "debug_info":
                result = await execute_debug_info(args)
            elif tool == "run_cuda":
                result = await execute_run_cuda(args)
            elif tool == "code_task":
                # args can be dict {"task": "...", "language": "..."} or string
                if isinstance(args, dict):
                    task = args.get("task", "")
                    language = args.get("language", "python")
                else:
                    task = str(args)
                    language = "python"
                result = await execute_code_task(task, language)
            elif tool == "run_js":
                result = await execute_run_js(args)

            # Engineering/Learning tools
            elif tool == "code_review":
                if isinstance(args, dict):
                    result = await execute_code_review(
                        args.get("code", ""),
                        args.get("language", "python"),
                        args.get("focus", "all")
                    )
                else:
                    result = await execute_code_review(str(args))
            elif tool == "explain_concept":
                if isinstance(args, dict):
                    result = await execute_explain_concept(
                        args.get("concept", ""),
                        args.get("level", "intermediate")
                    )
                else:
                    result = await execute_explain_concept(str(args))
            elif tool == "generate_tests":
                if isinstance(args, dict):
                    result = await execute_generate_tests(
                        args.get("code", ""),
                        args.get("language", "python"),
                        args.get("framework", "")
                    )
                else:
                    result = await execute_generate_tests(str(args))
            elif tool == "debug_help":
                if isinstance(args, dict):
                    result = await execute_debug_help(
                        args.get("error_message", ""),
                        args.get("code", ""),
                        args.get("language", "python")
                    )
                else:
                    result = await execute_debug_help(str(args))

            # Learning tools
            elif tool == "create_flashcard":
                if isinstance(args, dict):
                    result = await execute_create_flashcard(
                        args.get("front", ""),
                        args.get("back", ""),
                        args.get("topic", "general")
                    )
                else:
                    result = "create_flashcard requires {front, back, topic}"
            elif tool == "study_flashcards":
                topic = args.get("topic", "") if isinstance(args, dict) else (args or "")
                result = await execute_study_flashcards(topic)
            elif tool == "generate_quiz":
                if isinstance(args, dict):
                    result = await execute_generate_quiz(
                        args.get("topic", "programming"),
                        args.get("num_questions", 5),
                        args.get("difficulty", "intermediate")
                    )
                else:
                    result = await execute_generate_quiz(str(args))
            elif tool == "learning_path":
                goal = args.get("goal", args) if isinstance(args, dict) else args
                result = await execute_learning_path(goal)
            elif tool == "code_challenge":
                if isinstance(args, dict):
                    result = await execute_code_challenge(
                        args.get("topic", "algorithms"),
                        args.get("difficulty", "medium")
                    )
                else:
                    result = await execute_code_challenge(str(args))
            elif tool == "explain_error":
                if isinstance(args, dict):
                    result = await execute_explain_error(
                        args.get("error_type", ""),
                        args.get("language", "python")
                    )
                else:
                    result = await execute_explain_error(str(args))

            # Obsidian tools - handle both dict and string args
            elif tool == "obsidian_list":
                path = args.get("path", "/") if isinstance(args, dict) else (args or "/")
                result = await execute_obsidian_list(path)
            elif tool == "obsidian_read":
                path = args.get("path", args) if isinstance(args, dict) else args
                result = await execute_obsidian_read(path)
            elif tool == "obsidian_search":
                query = args.get("query", args) if isinstance(args, dict) else args
                result = await execute_obsidian_search(query)
            elif tool == "obsidian_write":
                if isinstance(args, dict):
                    result = await execute_obsidian_write(args.get("path", ""), args.get("content", ""))
                else:
                    result = "obsidian_write requires {path, content}"
            elif tool == "obsidian_append":
                if isinstance(args, dict):
                    result = await execute_obsidian_append(args.get("path", ""), args.get("content", ""))
                else:
                    result = "obsidian_append requires {path, content}"
            elif tool == "obsidian_daily":
                result = await execute_obsidian_daily()

            # Git tools
            elif tool == "git_status":
                path = args.get("path", ".") if isinstance(args, dict) else (args or ".")
                result = await execute_git_status(path)
            elif tool == "git_diff":
                if isinstance(args, dict):
                    result = await execute_git_diff(args.get("path", "."), args.get("staged", False))
                else:
                    result = await execute_git_diff(args or ".")
            elif tool == "git_log":
                if isinstance(args, dict):
                    result = await execute_git_log(args.get("path", "."), args.get("count", 10))
                else:
                    result = await execute_git_log(args or ".")
            elif tool == "git_branch":
                path = args.get("path", ".") if isinstance(args, dict) else (args or ".")
                result = await execute_git_branch(path)
            elif tool == "git_commit":
                if isinstance(args, dict):
                    result = await execute_git_commit(
                        args.get("path", "."),
                        args.get("message", ""),
                        args.get("add_all", True)
                    )
                else:
                    result = await execute_git_commit(".", str(args))
            elif tool == "git_push":
                if isinstance(args, dict):
                    result = await execute_git_push(
                        args.get("path", "."),
                        args.get("remote", "origin"),
                        args.get("branch", "")
                    )
                else:
                    result = await execute_git_push()
            elif tool == "git_pull":
                if isinstance(args, dict):
                    result = await execute_git_pull(
                        args.get("path", "."),
                        args.get("remote", "origin"),
                        args.get("branch", "")
                    )
                else:
                    result = await execute_git_pull()
            elif tool == "git_create_branch":
                if isinstance(args, dict):
                    result = await execute_git_create_branch(
                        args.get("path", "."),
                        args.get("branch_name", ""),
                        args.get("checkout", True)
                    )
                else:
                    result = await execute_git_create_branch(".", str(args))
            elif tool == "gh_pr_create":
                if isinstance(args, dict):
                    result = await execute_gh_pr_create(
                        args.get("title", ""),
                        args.get("body", ""),
                        args.get("base", "main"),
                        args.get("draft", False)
                    )
                else:
                    result = await execute_gh_pr_create(str(args))
            elif tool == "gh_issue_create":
                if isinstance(args, dict):
                    result = await execute_gh_issue_create(
                        args.get("title", ""),
                        args.get("body", ""),
                        args.get("labels", "")
                    )
                else:
                    result = await execute_gh_issue_create(str(args))
            elif tool == "gh_pr_merge":
                if isinstance(args, dict):
                    result = await execute_gh_pr_merge(
                        args.get("pr_number", 0),
                        args.get("method", "merge")
                    )
                else:
                    result = await execute_gh_pr_merge(int(args))

            # Docker tools
            elif tool == "docker_ps":
                all_c = args.get("all", False) if isinstance(args, dict) else bool(args)
                result = await execute_docker_ps(all_c)
            elif tool == "docker_logs":
                if isinstance(args, dict):
                    result = await execute_docker_logs(args.get("container", ""), args.get("lines", 50))
                else:
                    result = await execute_docker_logs(args)
            elif tool == "docker_stats":
                result = await execute_docker_stats()
            elif tool == "docker_control":
                if isinstance(args, dict):
                    result = await execute_docker_control(args.get("container", ""), args.get("action", ""))
                else:
                    result = "docker_control requires {container, action}"

            # NPM tools
            elif tool == "npm_run":
                if isinstance(args, dict):
                    result = await execute_npm_run(args.get("script", ""), args.get("path", "."))
                else:
                    result = await execute_npm_run(args)
            elif tool == "npm_install":
                if isinstance(args, dict):
                    result = await execute_npm_install(args.get("package", ""), args.get("path", "."))
                else:
                    result = await execute_npm_install(args or "")
            elif tool == "npm_list":
                path = args.get("path", ".") if isinstance(args, dict) else (args or ".")
                result = await execute_npm_list(path)

            # Homelab tools
            elif tool == "homelab_status":
                result = await execute_homelab_status()
            elif tool == "service_health":
                result = await execute_service_health()

            # System tools
            elif tool == "system_info":
                result = await execute_system_info()
            elif tool == "process_list":
                filter_str = args.get("filter", args) if isinstance(args, dict) else (args or "")
                result = await execute_process_list(filter_str)

            # Python tools
            elif tool == "pytest":
                if isinstance(args, dict):
                    result = await execute_pytest(args.get("path", "."), args.get("args", ""))
                else:
                    result = await execute_pytest(args or ".")
            elif tool == "python_lint":
                if isinstance(args, dict):
                    result = await execute_python_lint(args.get("path", "."), args.get("tool", "ruff"))
                else:
                    result = await execute_python_lint(args or ".")
            elif tool == "python_format":
                path = args.get("path", args) if isinstance(args, dict) else args
                result = await execute_python_format(path)

            # GitHub tools
            elif tool == "gh_pr_list":
                repo = args.get("repo", args) if isinstance(args, dict) else (args or "")
                result = await execute_gh_pr_list(repo)
            elif tool == "gh_issue_list":
                repo = args.get("repo", args) if isinstance(args, dict) else (args or "")
                result = await execute_gh_issue_list(repo)
            elif tool == "gh_repo_view":
                repo = args.get("repo", args) if isinstance(args, dict) else args
                result = await execute_gh_repo_view(repo)
            elif tool == "gh_workflow_list":
                repo = args.get("repo", args) if isinstance(args, dict) else (args or "")
                result = await execute_gh_workflow_list(repo)

            # Browser tools
            elif tool == "browser_fetch":
                if isinstance(args, dict):
                    result = await execute_browser_fetch(args.get("url", ""), args.get("wait_for"))
                else:
                    result = await execute_browser_fetch(args)
            elif tool == "browser_extract":
                if isinstance(args, dict):
                    result = await execute_browser_extract(args.get("url", ""), args.get("selector", ""))
                else:
                    result = "browser_extract requires {url, selector}"
            elif tool == "browser_screenshot":
                if isinstance(args, dict):
                    result = await execute_browser_screenshot(args.get("url", ""), args.get("full_page", False))
                else:
                    result = await execute_browser_screenshot(args)

            # Spreadsheet tools
            elif tool == "read_excel":
                if isinstance(args, dict):
                    result = await execute_read_excel(args.get("path", ""), args.get("sheet"))
                else:
                    result = await execute_read_excel(args)
            elif tool == "read_csv":
                path = args.get("path", args) if isinstance(args, dict) else args
                result = await execute_read_csv(path)
            elif tool == "analyze_data":
                if isinstance(args, dict):
                    result = await execute_analyze_data(args.get("path", ""), args.get("query", ""))
                else:
                    result = await execute_analyze_data(args, "basic analysis")
            elif tool == "create_excel":
                if isinstance(args, dict):
                    result = await execute_create_excel(args.get("path", ""), args.get("data", {}))
                else:
                    result = "create_excel requires {path, data}"

            # Library docs tools
            elif tool == "context7_search":
                library = args.get("library", args) if isinstance(args, dict) else args
                result = await execute_context7_search(library)
            elif tool == "context7_docs":
                if isinstance(args, dict):
                    result = await execute_context7_docs(args.get("library", ""), args.get("query", ""))
                else:
                    result = "context7_docs requires {library, query}"

            # MCP builder
            elif tool == "create_mcp_server":
                if isinstance(args, dict):
                    result = await execute_create_mcp_server(
                        args.get("name", "my-server"),
                        args.get("description", "MCP Server"),
                        args.get("tools", [])
                    )
                else:
                    result = "create_mcp_server requires {name, description, tools}"

            # Email tools
            elif tool == "email_list":
                folder = args.get("folder", "INBOX") if isinstance(args, dict) else "INBOX"
                count = int(args.get("count", 10)) if isinstance(args, dict) else 10
                result = await execute_email_list(folder, count)
            elif tool == "email_read":
                if isinstance(args, dict) and args.get("email_id"):
                    result = await execute_email_read(args["email_id"], args.get("folder", "INBOX"))
                else:
                    result = "email_read requires {email_id, folder}"
            elif tool == "email_send":
                if isinstance(args, dict) and args.get("to") and args.get("subject") and args.get("body"):
                    result = await execute_email_send(args["to"], args["subject"], args["body"], args.get("cc", ""))
                else:
                    result = "email_send requires {to, subject, body}"
            elif tool == "email_search":
                if isinstance(args, dict) and args.get("query"):
                    result = await execute_email_search(args["query"], args.get("folder", "INBOX"))
                else:
                    result = "email_search requires {query}"
            elif tool == "email_unread":
                folder = args.get("folder", "INBOX") if isinstance(args, dict) else "INBOX"
                result = await execute_email_unread(folder)

            # API Testing
            elif tool == "api_request":
                if isinstance(args, dict) and args.get("url"):
                    result = await execute_api_request(
                        args.get("method", "GET"),
                        args["url"],
                        args.get("headers", {}),
                        args.get("body", ""),
                        args.get("timeout", 30)
                    )
                else:
                    result = "api_request requires {method, url}"
            elif tool == "api_test_suite":
                if isinstance(args, dict) and args.get("tests"):
                    result = await execute_api_test_suite(args["tests"])
                else:
                    result = "api_test_suite requires {tests: [...]}"

            # Diagrams
            elif tool == "create_diagram":
                if isinstance(args, dict) and args.get("code"):
                    result = await execute_create_diagram(
                        args.get("type", "flowchart"),
                        args["code"],
                        args.get("output_path", "")
                    )
                else:
                    result = "create_diagram requires {code}"
            elif tool == "flowchart":
                desc = args.get("description", args) if isinstance(args, dict) else str(args)
                result = await execute_flowchart(desc)

            # PDF Tools
            elif tool == "pdf_read":
                if isinstance(args, dict) and args.get("path"):
                    result = await execute_pdf_read(args["path"], args.get("pages", ""))
                else:
                    result = "pdf_read requires {path}"
            elif tool == "pdf_merge":
                if isinstance(args, dict) and args.get("files") and args.get("output"):
                    result = await execute_pdf_merge(args["files"], args["output"])
                else:
                    result = "pdf_merge requires {files: [...], output}"
            elif tool == "pdf_create":
                if isinstance(args, dict) and args.get("content") and args.get("output"):
                    result = await execute_pdf_create(args["content"], args["output"], args.get("title", "Document"))
                else:
                    result = "pdf_create requires {content, output}"

            # Whisper
            elif tool == "transcribe":
                if isinstance(args, dict) and args.get("audio_path"):
                    result = await execute_transcribe(args["audio_path"], args.get("language", ""))
                else:
                    result = "transcribe requires {audio_path}"

            # Database
            elif tool == "sql_query":
                if isinstance(args, dict) and args.get("database") and args.get("query"):
                    result = await execute_sql_query(args["database"], args["query"])
                else:
                    result = "sql_query requires {database, query}"
            elif tool == "db_schema":
                if isinstance(args, dict) and args.get("database"):
                    result = await execute_db_schema(args["database"])
                else:
                    result = "db_schema requires {database}"

        except Exception as e:
            result = f"Error: {str(e)}"

        return (tool, args, result)

    results = await asyncio.gather(*[execute_single(step) for step in steps])

    combined = {}
    for tool, args, result in results:
        key = f"{tool}:{args[:50]}"
        combined[key] = result

    return combined

# --- API ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Track startup time for uptime metrics
    app.state.start_time = time.time()

    # Initialize managers
    await memory_manager.initialize()
    await rag_manager.initialize()

    # Initialize learning system (4-layer self-improving AI)
    if LEARNING_AVAILABLE:
        try:
            await setup_learning_system()
            print("[BENCHAI] Learning system initialized (Zettelkasten, Memory, Experience, Training)")
        except Exception as e:
            print(f"[WARN] Learning system initialization failed: {e}")

    # Start background tasks
    manager._cleanup_task = asyncio.create_task(manager.start_cleanup_loop())
    manager._health_monitor_task = asyncio.create_task(manager.health_monitor_loop())

    # Ensure core models are running (prevents empty response errors)
    print("[BENCHAI] Starting core models...")
    await manager.ensure_core_models()

    print("[BENCHAI] Router v3 started on port 8085")
    print(f"[BENCHAI] Memory: {DB_PATH}")
    print(f"[BENCHAI] RAG: {rag_manager.get_stats()}")
    print(f"[BENCHAI] TTS: {'Available' if PIPER_AVAILABLE else 'Not available'}")
    print(f"[BENCHAI] Core models: {CORE_MODELS}")
    print(f"[BENCHAI] Health check interval: {HEALTH_CHECK_INTERVAL}s")
    yield

    # Cleanup on shutdown
    if manager._cleanup_task:
        manager._cleanup_task.cancel()
    if manager._health_monitor_task:
        manager._health_monitor_task.cancel()
    if LEARNING_AVAILABLE:
        await shutdown_learning_system()
    await manager.stop_all()
    print("[BENCHAI] Router stopped")

app = FastAPI(title="BenchAI LLM Router v3.5", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Include learning system router (Zettelkasten, Memory, Experience, Training, Agents)
if LEARNING_AVAILABLE:
    app.include_router(learning_router)
    print("[BENCHAI] Learning router mounted at /v1/learning/*")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "benchai-router-v3.5",
        "features": {
            "streaming": True,
            "memory": True,
            "tts": PIPER_AVAILABLE,
            "rag": CHROMADB_AVAILABLE,
            "obsidian": OBSIDIAN_AVAILABLE,
            "learning": LEARNING_AVAILABLE,
            "zettelkasten": LEARNING_AVAILABLE,
            "experience_replay": LEARNING_AVAILABLE,
            "multi_agent": LEARNING_AVAILABLE
        }
    }


# =========================================================================
# A2A v0.3 Agent Card Endpoint
# =========================================================================

BENCHAI_AGENT_CARD = {
    "name": "BenchAI",
    "description": "Central AI orchestrator with knowledge management, multi-agent coordination, and self-improving learning systems. Specializes in deep research, memory management, and coordinating tasks between specialized agents.",
    "url": "http://localhost:8085",
    "version": "3.5.0",
    "protocol_version": "0.3",
    "capabilities": {
        "streaming": True,
        "push_notifications": False,
        "state_transition_history": True
    },
    "authentication": {
        "schemes": ["none"]  # Add API key auth later
    },
    "default_input_modes": ["text/plain", "application/json"],
    "default_output_modes": ["text/plain", "application/json", "text/event-stream"],
    "skills": [
        {
            "id": "research",
            "name": "Deep Research",
            "description": "Perform deep research using Zettelkasten knowledge graph with semantic linking and automatic discovery",
            "tags": ["research", "knowledge", "zettelkasten"],
            "input_modes": ["text/plain"],
            "output_modes": ["application/json"],
            "examples": [
                "What are best practices for async database connections?",
                "Explain the relationship between transformers and attention mechanisms"
            ]
        },
        {
            "id": "memory",
            "name": "Persistent Memory",
            "description": "Store and retrieve memories with semantic search, importance decay, and automatic consolidation",
            "tags": ["memory", "knowledge", "persistent"],
            "input_modes": ["text/plain", "application/json"],
            "output_modes": ["application/json"]
        },
        {
            "id": "rag",
            "name": "RAG Pipeline",
            "description": "Retrieval-augmented generation using ChromaDB with HNSW indexing for fast similarity search",
            "tags": ["rag", "retrieval", "vector-search"],
            "input_modes": ["text/plain"],
            "output_modes": ["application/json"]
        },
        {
            "id": "orchestration",
            "name": "Multi-Agent Orchestration",
            "description": "Coordinate tasks between specialized agents (MarunochiAI for coding, DottscavisAI for creative work)",
            "tags": ["orchestration", "multi-agent", "coordination"],
            "input_modes": ["application/json"],
            "output_modes": ["application/json"]
        },
        {
            "id": "experience_replay",
            "name": "Experience Replay",
            "description": "Learn from past successes with curiosity-driven prioritized experience replay",
            "tags": ["learning", "experience", "self-improvement"],
            "input_modes": ["application/json"],
            "output_modes": ["application/json"]
        },
        {
            "id": "inference",
            "name": "Local LLM Inference",
            "description": "Run inference on local LLMs with VRAM management and model hot-swapping",
            "tags": ["llm", "inference", "local"],
            "input_modes": ["application/json"],
            "output_modes": ["text/event-stream", "application/json"]
        },
        {
            "id": "tts",
            "name": "Text-to-Speech",
            "description": "Convert text to speech using Piper TTS engine",
            "tags": ["tts", "audio", "speech"],
            "input_modes": ["text/plain", "application/json"],
            "output_modes": ["audio/wav"]
        }
    ],
    "provider": {
        "organization": "BenchAI",
        "contact": "benchai@localhost"
    }
}


@app.get("/.well-known/agent.json")
async def agent_card():
    """
    A2A v0.3 Agent Card - Standard endpoint for agent discovery.
    Other agents can fetch this to understand BenchAI's capabilities.
    """
    # Dynamically update capabilities based on current state
    card = BENCHAI_AGENT_CARD.copy()
    card["capabilities"] = {
        "streaming": True,
        "push_notifications": False,
        "state_transition_history": True,
        "tts": PIPER_AVAILABLE,
        "rag": CHROMADB_AVAILABLE,
        "learning": LEARNING_AVAILABLE,
        "obsidian": OBSIDIAN_AVAILABLE
    }

    # Update skills availability
    available_skills = []
    for skill in BENCHAI_AGENT_CARD["skills"]:
        skill_copy = skill.copy()
        if skill["id"] == "tts":
            skill_copy["available"] = PIPER_AVAILABLE
        elif skill["id"] == "rag":
            skill_copy["available"] = CHROMADB_AVAILABLE
        elif skill["id"] in ["research", "memory", "experience_replay", "orchestration"]:
            skill_copy["available"] = LEARNING_AVAILABLE
        else:
            skill_copy["available"] = True
        available_skills.append(skill_copy)

    card["skills"] = available_skills
    card["updated_at"] = datetime.now().isoformat()

    return card


@app.get("/.well-known/agent-card.json")
async def agent_card_alt():
    """Alias for agent card (some implementations use this path)."""
    return await agent_card()

@app.get("/v1/metrics")
async def metrics():
    """Get comprehensive system metrics for monitoring."""
    # GPU stats
    gpu_stats = {"available": False}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                gpu_stats = {
                    "available": True,
                    "memory_used_mb": int(parts[0]),
                    "memory_total_mb": int(parts[1]),
                    "utilization_percent": int(parts[2]),
                    "temperature_c": int(parts[3]),
                    "memory_percent": round(int(parts[0]) / int(parts[1]) * 100, 1)
                }
    except Exception:
        pass

    # Model stats
    model_stats = []
    for name, config in MODELS.items():
        port = config["port"]
        is_running = manager.is_running(name)
        pid = manager.processes.get(name, None)
        pid = pid.pid if pid else None
        mode = config.get("mode", "cpu")
        model_stats.append({
            "name": name,
            "display_name": config["name"],
            "port": port,
            "status": "running" if is_running else "stopped",
            "pid": pid,
            "mode": mode.upper(),
            "file": config["file"]
        })

    # Memory stats
    memory_stats = {"total": 0, "fts5": False}
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT COUNT(*) FROM memories") as cursor:
                row = await cursor.fetchone()
                memory_stats["total"] = row[0] if row else 0
            memory_stats["fts5"] = True
    except Exception:
        pass

    # RAG stats
    rag_stats = {"status": "disabled", "count": 0}
    if CHROMADB_AVAILABLE and rag_manager.collection:
        try:
            rag_stats = {"status": "ready", "count": rag_manager.collection.count()}
        except Exception:
            rag_stats = {"status": "error", "count": 0}

    # Cache stats
    cache_stats = request_cache.stats()

    # System uptime
    uptime = time.time() - getattr(app.state, 'start_time', time.time())

    return {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(uptime),
        "gpu": gpu_stats,
        "models": model_stats,
        "memory": memory_stats,
        "rag": rag_stats,
        "cache": cache_stats,
        "features": {
            "tts": PIPER_AVAILABLE,
            "rag": CHROMADB_AVAILABLE,
            "obsidian": OBSIDIAN_AVAILABLE
        }
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Monitoring dashboard for BenchAI."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BenchAI Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px 0 30px;
            border-bottom: 1px solid #374151;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .header .subtitle { color: #9ca3af; font-size: 0.9rem; }
        .header .status {
            display: inline-block;
            margin-top: 10px;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .status-ok { background: #065f46; color: #34d399; }
        .status-error { background: #7f1d1d; color: #fca5a5; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            font-size: 1rem;
            color: #9ca3af;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card h2 .icon { font-size: 1.2rem; }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #374151;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #9ca3af; font-size: 0.9rem; }
        .metric-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e4e4e7;
        }
        .metric-value.good { color: #34d399; }
        .metric-value.warn { color: #fbbf24; }
        .metric-value.bad { color: #f87171; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #374151;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .progress-fill.gpu { background: linear-gradient(90deg, #60a5fa, #a78bfa); }
        .progress-fill.cache { background: linear-gradient(90deg, #34d399, #22d3d3); }
        .model-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: rgba(55, 65, 81, 0.5);
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .model-item:last-child { margin-bottom: 0; }
        .model-name { font-weight: 500; }
        .model-meta { font-size: 0.8rem; color: #9ca3af; }
        .model-status {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .model-status.running { background: #065f46; color: #34d399; }
        .model-status.stopped { background: #7f1d1d; color: #fca5a5; }
        .model-status.gpu { background: #5b21b6; color: #c4b5fd; }
        .model-status.cpu { background: #1e40af; color: #93c5fd; }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .feature-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px;
            background: rgba(55, 65, 81, 0.5);
            border-radius: 8px;
            font-size: 0.9rem;
        }
        .feature-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .feature-dot.enabled { background: #34d399; }
        .feature-dot.disabled { background: #6b7280; }
        .refresh-info {
            text-align: center;
            padding: 15px;
            color: #6b7280;
            font-size: 0.85rem;
        }
        .big-number {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card.wide { grid-column: span 2; }
        @media (max-width: 700px) { .card.wide { grid-column: span 1; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>BenchAI Monitor</h1>
        <div class="subtitle">Intelligent LLM Orchestration Platform</div>
        <div id="status" class="status status-ok">● Connected</div>
    </div>

    <div class="grid">
        <div class="card">
            <h2><span class="icon">🎮</span> GPU Status</h2>
            <div id="gpu-content">Loading...</div>
        </div>

        <div class="card">
            <h2><span class="icon">⚡</span> Cache Performance</h2>
            <div id="cache-content">Loading...</div>
        </div>

        <div class="card wide">
            <h2><span class="icon">🤖</span> Models</h2>
            <div id="models-content">Loading...</div>
        </div>

        <div class="card">
            <h2><span class="icon">🧠</span> Memory System</h2>
            <div id="memory-content">Loading...</div>
        </div>

        <div class="card">
            <h2><span class="icon">📚</span> RAG Knowledge Base</h2>
            <div id="rag-content">Loading...</div>
        </div>

        <div class="card">
            <h2><span class="icon">✨</span> Features</h2>
            <div id="features-content">Loading...</div>
        </div>

        <div class="card">
            <h2><span class="icon">📊</span> System</h2>
            <div id="system-content">Loading...</div>
        </div>
    </div>

    <div class="refresh-info">Auto-refreshes every 5 seconds</div>

    <script>
        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = seconds % 60;
            if (h > 0) return `${h}h ${m}m ${s}s`;
            if (m > 0) return `${m}m ${s}s`;
            return `${s}s`;
        }

        async function fetchMetrics() {
            try {
                const res = await fetch('/v1/metrics');
                const data = await res.json();
                updateDashboard(data);
                document.getElementById('status').className = 'status status-ok';
                document.getElementById('status').textContent = '● Connected';
            } catch (e) {
                document.getElementById('status').className = 'status status-error';
                document.getElementById('status').textContent = '● Disconnected';
            }
        }

        function updateDashboard(data) {
            // GPU
            const gpu = data.gpu;
            if (gpu.available) {
                const memPct = gpu.memory_percent;
                const color = memPct > 90 ? 'bad' : memPct > 70 ? 'warn' : 'good';
                document.getElementById('gpu-content').innerHTML = `
                    <div class="metric">
                        <span class="metric-label">Memory Used</span>
                        <span class="metric-value">${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB</span>
                    </div>
                    <div class="progress-bar"><div class="progress-fill gpu" style="width: ${memPct}%"></div></div>
                    <div class="metric">
                        <span class="metric-label">Utilization</span>
                        <span class="metric-value ${color}">${gpu.utilization_percent}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Temperature</span>
                        <span class="metric-value ${gpu.temperature_c > 80 ? 'bad' : gpu.temperature_c > 60 ? 'warn' : 'good'}">${gpu.temperature_c}°C</span>
                    </div>
                `;
            } else {
                document.getElementById('gpu-content').innerHTML = '<div class="metric"><span class="metric-value bad">GPU Not Available</span></div>';
            }

            // Cache
            const cache = data.cache;
            document.getElementById('cache-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Hit Rate</span>
                    <span class="metric-value good">${cache.hit_rate}</span>
                </div>
                <div class="progress-bar"><div class="progress-fill cache" style="width: ${parseFloat(cache.hit_rate)}%"></div></div>
                <div class="metric">
                    <span class="metric-label">Hits / Misses</span>
                    <span class="metric-value">${cache.hits} / ${cache.misses}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cached Responses</span>
                    <span class="metric-value">${cache.size} / ${cache.max_size}</span>
                </div>
            `;

            // Models
            document.getElementById('models-content').innerHTML = data.models.map(m => `
                <div class="model-item">
                    <div>
                        <div class="model-name">${m.display_name}</div>
                        <div class="model-meta">Port ${m.port} • ${m.file}</div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <span class="model-status ${m.mode.toLowerCase()}">${m.mode}</span>
                        <span class="model-status ${m.status}">${m.status}</span>
                    </div>
                </div>
            `).join('');

            // Memory
            document.getElementById('memory-content').innerHTML = `
                <div style="text-align: center; padding: 20px 0;">
                    <div class="big-number">${data.memory.total}</div>
                    <div style="color: #9ca3af; margin-top: 5px;">Total Memories</div>
                </div>
                <div class="metric">
                    <span class="metric-label">FTS5 Search</span>
                    <span class="metric-value ${data.memory.fts5 ? 'good' : 'bad'}">${data.memory.fts5 ? 'Enabled' : 'Disabled'}</span>
                </div>
            `;

            // RAG
            document.getElementById('rag-content').innerHTML = `
                <div style="text-align: center; padding: 20px 0;">
                    <div class="big-number">${data.rag.count}</div>
                    <div style="color: #9ca3af; margin-top: 5px;">Documents Indexed</div>
                </div>
                <div class="metric">
                    <span class="metric-label">Status</span>
                    <span class="metric-value ${data.rag.status === 'ready' ? 'good' : 'bad'}">${data.rag.status.toUpperCase()}</span>
                </div>
            `;

            // Features
            const features = data.features;
            document.getElementById('features-content').innerHTML = `
                <div class="feature-grid">
                    <div class="feature-item"><span class="feature-dot enabled"></span>Streaming</div>
                    <div class="feature-item"><span class="feature-dot enabled"></span>Memory</div>
                    <div class="feature-item"><span class="feature-dot ${features.tts ? 'enabled' : 'disabled'}"></span>TTS</div>
                    <div class="feature-item"><span class="feature-dot ${features.rag ? 'enabled' : 'disabled'}"></span>RAG</div>
                    <div class="feature-item"><span class="feature-dot ${features.obsidian ? 'enabled' : 'disabled'}"></span>Obsidian</div>
                    <div class="feature-item"><span class="feature-dot enabled"></span>Caching</div>
                </div>
            `;

            // System
            document.getElementById('system-content').innerHTML = `
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value good">${formatUptime(data.uptime_seconds)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Last Updated</span>
                    <span class="metric-value">${new Date(data.timestamp).toLocaleTimeString()}</span>
                </div>
            `;
        }

        fetchMetrics();
        setInterval(fetchMetrics, 5000);
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/v1/models")
async def list_models():
    models_list = [
        {"id": "auto", "object": "model", "name": "Auto-Agent (Planner)", "owned_by": "benchai"},
        {"id": "general", "object": "model", "name": "General (Phi-3 Mini)", "owned_by": "benchai"},
        {"id": "research", "object": "model", "name": "Research (Qwen2.5 7B)", "owned_by": "benchai"},
        {"id": "code", "object": "model", "name": "Code (DeepSeek 6.7B)", "owned_by": "benchai"},
        {"id": "vision", "object": "model", "name": "Vision (Qwen2-VL 7B)", "owned_by": "benchai"},
    ]
    return {"object": "list", "data": models_list}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages_dicts = [m.model_dump() for m in request.messages]

    # Check cache first (skip for streaming requests)
    if not request.stream:
        cached = request_cache.get(request.model, messages_dicts, request.max_tokens)
        if cached:
            return cached

    # Get or create session for conversation context
    session_id, session = session_manager.get_or_create_session(messages_dicts)
    print(f"[SESSION] Using session {session_id} (history: {len(session['history'])}, tools: {list(session['tool_results'].keys())})")

    user_msgs = [m for m in request.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "No user message found")

    last_msg = user_msgs[-1]
    has_image = False
    text_query = ""

    if isinstance(last_msg.content, list):
        for part in last_msg.content:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    has_image = True
                    print(f"[IMAGE] Detected image in request")
                if part.get("type") == "text":
                    text_query += part.get("text", "")
    else:
        text_query = str(last_msg.content)

    if has_image:
        print(f"[REQUEST] Image attached, query: {text_query[:100]}")

    # Add user turn to session
    session_manager.add_turn(session_id, "user", text_query)

    # Log conversation
    await memory_manager.log_conversation("user", text_query)

    # --- LEARNING SYSTEM: Experience Context Injection ---
    # Inject relevant past experiences for 15-20% performance boost
    experience_context = ""
    memory_context = ""
    if LEARNING_AVAILABLE and len(text_query) > 50:  # Only for substantial queries
        try:
            # Get relevant experiences from past successes
            experience_context = await get_experience_context(text_query, domain="general")
            # Get relevant memories
            memories = await recall_relevant(text_query, limit=3)
            if memories:
                memory_context = "\n".join([f"- {m.get('content', '')[:200]}" for m in memories[:3]])

            # Inject into first system message if we have useful context
            if experience_context or memory_context:
                context_injection = ""
                if experience_context:
                    context_injection += f"\n\n## Relevant Past Experiences:\n{experience_context}"
                if memory_context:
                    context_injection += f"\n\n## Relevant Knowledge:\n{memory_context}"

                # Find and enhance system message
                for i, msg in enumerate(messages_dicts):
                    if msg.get("role") == "system":
                        messages_dicts[i]["content"] = msg["content"] + context_injection
                        print(f"[LEARNING] Injected {len(experience_context)} chars experience + {len(memory_context)} chars memory")
                        break
        except Exception as e:
            print(f"[LEARNING] Experience injection skipped: {e}")

    # Check for memory-related queries and auto-save
    if MEMORY_PATTERNS.search(text_query):
        # Check for "remember this" type commands
        if re.search(r'\b(remember|save|note|keep in mind)\b', text_query, re.IGNORECASE):
            await memory_manager.add_memory(text_query, category="user_stated", importance=2)

    # Meta-prompt fast path
    if META_PATTERNS.match(text_query.strip()):
        print("[FAST-PATH] Meta-prompt detected")
        ans = await execute_reasoning(messages_dicts, "general")
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "general",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ans}, "finish_reason": "stop"}]
        }

    # Manual model override
    if request.model and request.model != "auto" and request.model in MODELS:
        print(f"[MANUAL] Using {request.model}")

        if request.stream:
            async def stream_generator():
                if request.model == "vision":
                    content = await execute_vision(messages_dicts)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                else:
                    async for chunk in execute_reasoning_stream(messages_dicts, request.model):
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        if request.model == "vision":
            ans = await execute_vision(messages_dicts)
        else:
            ans = await execute_reasoning(messages_dicts, request.model)

        await memory_manager.log_conversation("assistant", ans)

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ans}, "finish_reason": "stop"}]
        }

        # Cache the response
        request_cache.set(request.model, messages_dicts, request.max_tokens, response)
        return response

    # Agentic Plan & Execute
    session_context = session_manager.get_context_for_planner(session_id)
    plan = await generate_plan(text_query, has_image, session_context)
    print(f"[PLAN] {json.dumps(plan, indent=2)}")

    context = session.get("context", "")  # Start with existing session context
    final_response = ""

    pending_tool = None  # For malformed plans like ["tool_name", {"args": "..."}]
    for step in plan:
        if isinstance(step, str):
            # This might be a tool name for the next step (malformed plan)
            pending_tool = step
            continue

        # If step has no tool key but we have a pending tool name, use it
        if pending_tool and isinstance(step, dict) and "tool" not in step:
            step["tool"] = pending_tool
            pending_tool = None

        # Handle parallel execution
        if "parallel" in step:
            parallel_results = await execute_parallel_steps(step["parallel"], messages_dicts, has_image)
            for key, result in parallel_results.items():
                context += f"\n\n**{key}:**\n{result}"
            continue

        tool = step.get("tool") or step.get("step") or step.get("action")
        args = step.get("args") or step.get("query") or step.get("prompt") or step.get("input") or ""

        # Log tool execution
        print(f"[EXECUTE] Tool: {tool}")

        # Normalize args: if it's a dict with 'prompt' or 'query', extract the string
        if isinstance(args, dict):
            args = args.get("prompt") or args.get("query") or args.get("task") or args.get("input") or str(args)

        prompt_with_context = f"Previous context:\n{context}\n\nCurrent task: {args}" if context else str(args)

        try:
            if tool == "web_search":
                res = await execute_web_search(args)
                context += f"\n\n**Web Search Results for '{args}':**\n{res}"

            elif tool == "claude_assist":
                # Claude mastermind - for complex reasoning tasks
                if isinstance(args, dict):
                    prompt = args.get("prompt") or args.get("task") or args.get("query") or str(args)
                    ctx = args.get("context", context)  # Use accumulated context
                else:
                    prompt = str(args)
                    ctx = context
                res = await execute_claude_assist(prompt, ctx)
                context += f"\n\n**Claude Analysis:**\n{res}"
                final_response += f"\n\n**Claude's Analysis:**\n{res}"

            elif tool == "vision_analysis":
                res = await execute_vision(messages_dicts)
                context += f"\n\n**Image Analysis:**\n{res}"
                # Store in session for follow-up requests
                session_manager.add_tool_result(session_id, "vision_analysis", res)
                # Auto-save important vision analysis to memory
                if len(res) > 100 and not res.startswith("Vision analysis"):
                    # Save a summary to memory for future reference
                    summary = res[:500] + "..." if len(res) > 500 else res
                    await memory_manager.add_memory(
                        f"[Vision Analysis] {summary}",
                        category="analysis",
                        importance=2
                    )
                    print("[MEMORY] Auto-saved vision analysis result")
                # If vision is the primary tool (e.g., "solve this" with image), include in response
                if not final_response:
                    final_response = res

            elif tool == "generate_image":
                full_prompt = f"{args}. {context[:300]}" if context else args
                res = await execute_image_gen(full_prompt)
                final_response += f"\n{res}\n"

            elif tool == "run_code":
                res = await execute_code(args)
                context += f"\n\n**Code Execution:**\n{res}"
                final_response += res

            elif tool == "read_file":
                res = await execute_file_read(args)
                context += f"\n\n**File Content:**\n{res}"

            elif tool == "shell":
                res = await execute_shell(args)
                context += f"\n\n**Shell Output:**\n{res}"

            # C/C++ tools
            elif tool == "run_cpp":
                res = await execute_run_cpp(args)
                context += f"\n\n**C++ Execution:**\n{res}"
                final_response += res

            elif tool == "run_c":
                res = await execute_run_c(args)
                context += f"\n\n**C Execution:**\n{res}"
                final_response += res

            elif tool == "compile_cpp":
                res = await execute_compile_cpp(args)
                context += f"\n\n**C++ Compilation:**\n{res}"
                final_response += res

            elif tool == "compile_c":
                res = await execute_compile_c(args)
                context += f"\n\n**C Compilation:**\n{res}"
                final_response += res

            elif tool == "cmake_build":
                res = await execute_cmake_build(args)
                context += f"\n\n**CMake Build:**\n{res}"
                final_response += res

            elif tool == "analyze_cpp":
                res = await execute_analyze_cpp(args)
                context += f"\n\n**Static Analysis:**\n{res}"
                final_response += res

            elif tool == "debug_info":
                res = await execute_debug_info(args)
                context += f"\n\n**Binary Analysis:**\n{res}"
                final_response += res

            elif tool == "run_cuda":
                res = await execute_run_cuda(args)
                context += f"\n\n**CUDA Execution:**\n{res}"
                final_response += res

            elif tool == "code_task":
                # args can be dict {"task": "...", "language": "..."} or string
                if isinstance(args, dict):
                    task = args.get("task", "")
                    language = args.get("language", "python")
                else:
                    task = str(args)
                    language = "python"
                # Include context from previous steps (e.g., vision analysis)
                full_task = f"{task}\n\nContext from previous analysis:\n{context}" if context else task
                res = await execute_code_task(full_task, language)
                context += f"\n\n**Code Task ({language}):**\n{res}"
                final_response += res
                # Store in session for follow-up requests
                session_manager.add_tool_result(session_id, "code_task", res)
                # Auto-save successful code results to memory
                if "Output:" in res and "Error" not in res[:100]:
                    summary = f"[Code: {language}] Task: {task[:100]}... Result saved."
                    await memory_manager.add_memory(summary, category="code", importance=2)
                    print("[MEMORY] Auto-saved code task result")

            elif tool == "run_js":
                res = await execute_run_js(args)
                context += f"\n\n**JavaScript Execution:**\n{res}"
                final_response += res

            # Engineering/Learning tools
            elif tool == "code_review":
                if isinstance(args, dict):
                    res = await execute_code_review(
                        args.get("code", ""),
                        args.get("language", "python"),
                        args.get("focus", "all")
                    )
                else:
                    res = await execute_code_review(str(args))
                context += f"\n\n**Code Review:**\n{res}"
                final_response += res
                session_manager.add_tool_result(session_id, "code_review", res)

            elif tool == "explain_concept":
                if isinstance(args, dict):
                    res = await execute_explain_concept(
                        args.get("concept", ""),
                        args.get("level", "intermediate")
                    )
                else:
                    res = await execute_explain_concept(str(args))
                context += f"\n\n**Concept Explanation:**\n{res}"
                final_response += res
                session_manager.add_tool_result(session_id, "explain_concept", res)

            elif tool == "generate_tests":
                if isinstance(args, dict):
                    res = await execute_generate_tests(
                        args.get("code", ""),
                        args.get("language", "python"),
                        args.get("framework", "")
                    )
                else:
                    res = await execute_generate_tests(str(args))
                context += f"\n\n**Generated Tests:**\n{res}"
                final_response += res
                session_manager.add_tool_result(session_id, "generate_tests", res)

            elif tool == "debug_help":
                if isinstance(args, dict):
                    res = await execute_debug_help(
                        args.get("error_message", ""),
                        args.get("code", ""),
                        args.get("language", "python")
                    )
                else:
                    res = await execute_debug_help(str(args))
                context += f"\n\n**Debug Analysis:**\n{res}"
                final_response += res
                session_manager.add_tool_result(session_id, "debug_help", res)

            # Learning tools
            elif tool == "create_flashcard":
                if isinstance(args, dict):
                    res = await execute_create_flashcard(
                        args.get("front", ""),
                        args.get("back", ""),
                        args.get("topic", "general")
                    )
                else:
                    res = "create_flashcard requires {front, back, topic}"
                context += f"\n\n**Flashcard:**\n{res}"
                final_response += res
            elif tool == "study_flashcards":
                topic = args.get("topic", "") if isinstance(args, dict) else (args or "")
                res = await execute_study_flashcards(topic)
                context += f"\n\n**Study Session:**\n{res}"
                final_response += res
            elif tool == "generate_quiz":
                if isinstance(args, dict):
                    res = await execute_generate_quiz(
                        args.get("topic", "programming"),
                        args.get("num_questions", 5),
                        args.get("difficulty", "intermediate")
                    )
                else:
                    res = await execute_generate_quiz(str(args))
                context += f"\n\n**Quiz:**\n{res}"
                final_response += res
            elif tool == "learning_path":
                goal = args.get("goal", args) if isinstance(args, dict) else args
                res = await execute_learning_path(goal)
                context += f"\n\n**Learning Path:**\n{res}"
                final_response += res
            elif tool == "code_challenge":
                if isinstance(args, dict):
                    res = await execute_code_challenge(
                        args.get("topic", "algorithms"),
                        args.get("difficulty", "medium")
                    )
                else:
                    res = await execute_code_challenge(str(args))
                context += f"\n\n**Coding Challenge:**\n{res}"
                final_response += res
            elif tool == "explain_error":
                if isinstance(args, dict):
                    res = await execute_explain_error(
                        args.get("error_type", ""),
                        args.get("language", "python")
                    )
                else:
                    res = await execute_explain_error(str(args))
                context += f"\n\n**Error Explanation:**\n{res}"
                final_response += res

            # Obsidian tools - handle both dict and string args
            elif tool == "obsidian_list":
                path = args.get("path", "/") if isinstance(args, dict) else (args or "/")
                res = await execute_obsidian_list(path)
                context += f"\n\n**Obsidian Vault:**\n{res}"
                final_response += res

            elif tool == "obsidian_read":
                path = args.get("path", args) if isinstance(args, dict) else args
                res = await execute_obsidian_read(path)
                context += f"\n\n**Obsidian Note:**\n{res}"
                final_response += res

            elif tool == "obsidian_search":
                query = args.get("query", args) if isinstance(args, dict) else args
                res = await execute_obsidian_search(query)
                context += f"\n\n**Obsidian Search:**\n{res}"
                final_response += res

            elif tool == "obsidian_write":
                if isinstance(args, dict):
                    res = await execute_obsidian_write(args.get("path", ""), args.get("content", ""))
                else:
                    res = "obsidian_write requires {path, content}"
                context += f"\n\n**Obsidian Write:**\n{res}"
                final_response += res

            elif tool == "obsidian_append":
                if isinstance(args, dict):
                    res = await execute_obsidian_append(args.get("path", ""), args.get("content", ""))
                else:
                    res = "obsidian_append requires {path, content}"
                context += f"\n\n**Obsidian Append:**\n{res}"
                final_response += res

            elif tool == "obsidian_daily":
                res = await execute_obsidian_daily()
                context += f"\n\n**Daily Note:**\n{res}"
                final_response += res

            # Git tools
            elif tool == "git_status":
                path = args.get("path", ".") if isinstance(args, dict) else (args or ".")
                res = await execute_git_status(path)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_diff":
                if isinstance(args, dict):
                    res = await execute_git_diff(args.get("path", "."), args.get("staged", False))
                else:
                    res = await execute_git_diff(args or ".")
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_log":
                if isinstance(args, dict):
                    res = await execute_git_log(args.get("path", "."), args.get("count", 10))
                else:
                    res = await execute_git_log(args or ".")
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_branch":
                path = args.get("path", ".") if isinstance(args, dict) else (args or ".")
                res = await execute_git_branch(path)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_commit":
                if isinstance(args, dict):
                    res = await execute_git_commit(
                        args.get("path", "."),
                        args.get("message", ""),
                        args.get("add_all", True)
                    )
                else:
                    res = await execute_git_commit(".", str(args))
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_push":
                if isinstance(args, dict):
                    res = await execute_git_push(
                        args.get("path", "."),
                        args.get("remote", "origin"),
                        args.get("branch", "")
                    )
                else:
                    res = await execute_git_push()
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_pull":
                if isinstance(args, dict):
                    res = await execute_git_pull(
                        args.get("path", "."),
                        args.get("remote", "origin"),
                        args.get("branch", "")
                    )
                else:
                    res = await execute_git_pull()
                context += f"\n\n{res}"
                final_response += res

            elif tool == "git_create_branch":
                if isinstance(args, dict):
                    res = await execute_git_create_branch(
                        args.get("path", "."),
                        args.get("branch_name", ""),
                        args.get("checkout", True)
                    )
                else:
                    res = await execute_git_create_branch(".", str(args))
                context += f"\n\n{res}"
                final_response += res

            elif tool == "gh_pr_create":
                if isinstance(args, dict):
                    res = await execute_gh_pr_create(
                        args.get("title", ""),
                        args.get("body", ""),
                        args.get("base", "main"),
                        args.get("draft", False)
                    )
                else:
                    res = await execute_gh_pr_create(str(args))
                context += f"\n\n{res}"
                final_response += res

            elif tool == "gh_issue_create":
                if isinstance(args, dict):
                    res = await execute_gh_issue_create(
                        args.get("title", ""),
                        args.get("body", ""),
                        args.get("labels", "")
                    )
                else:
                    res = await execute_gh_issue_create(str(args))
                context += f"\n\n{res}"
                final_response += res

            elif tool == "gh_pr_merge":
                if isinstance(args, dict):
                    res = await execute_gh_pr_merge(
                        args.get("pr_number", 0),
                        args.get("method", "merge")
                    )
                else:
                    res = await execute_gh_pr_merge(int(args))
                context += f"\n\n{res}"
                final_response += res

            # Docker tools
            elif tool == "docker_ps":
                all_c = args.get("all", False) if isinstance(args, dict) else bool(args)
                res = await execute_docker_ps(all_c)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "docker_logs":
                if isinstance(args, dict):
                    res = await execute_docker_logs(args.get("container", ""), args.get("lines", 50))
                else:
                    res = await execute_docker_logs(args)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "docker_stats":
                res = await execute_docker_stats()
                context += f"\n\n{res}"
                final_response += res

            elif tool == "docker_control":
                if isinstance(args, dict):
                    res = await execute_docker_control(args.get("container", ""), args.get("action", ""))
                else:
                    res = "docker_control requires {container, action}"
                context += f"\n\n{res}"
                final_response += res

            # NPM tools
            elif tool == "npm_run":
                if isinstance(args, dict):
                    res = await execute_npm_run(args.get("script", ""), args.get("path", "."))
                else:
                    res = await execute_npm_run(args)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "npm_install":
                if isinstance(args, dict):
                    res = await execute_npm_install(args.get("package", ""), args.get("path", "."))
                else:
                    res = await execute_npm_install(args or "")
                context += f"\n\n{res}"
                final_response += res

            elif tool == "npm_list":
                path = args.get("path", ".") if isinstance(args, dict) else (args or ".")
                res = await execute_npm_list(path)
                context += f"\n\n{res}"
                final_response += res

            # Homelab tools
            elif tool == "homelab_status":
                res = await execute_homelab_status()
                context += f"\n\n{res}"
                final_response += res

            elif tool == "service_health":
                res = await execute_service_health()
                context += f"\n\n{res}"
                final_response += res

            # System tools
            elif tool == "system_info":
                res = await execute_system_info()
                context += f"\n\n{res}"
                final_response += res

            elif tool == "process_list":
                filter_str = args.get("filter", args) if isinstance(args, dict) else (args or "")
                res = await execute_process_list(filter_str)
                context += f"\n\n{res}"
                final_response += res

            # Python tools
            elif tool == "pytest":
                if isinstance(args, dict):
                    res = await execute_pytest(args.get("path", "."), args.get("args", ""))
                else:
                    res = await execute_pytest(args or ".")
                context += f"\n\n{res}"
                final_response += res

            elif tool == "python_lint":
                if isinstance(args, dict):
                    res = await execute_python_lint(args.get("path", "."), args.get("tool", "ruff"))
                else:
                    res = await execute_python_lint(args or ".")
                context += f"\n\n{res}"
                final_response += res

            elif tool == "python_format":
                path = args.get("path", args) if isinstance(args, dict) else args
                res = await execute_python_format(path)
                context += f"\n\n{res}"
                final_response += res

            # GitHub tools
            elif tool == "gh_pr_list":
                repo = args.get("repo", args) if isinstance(args, dict) else (args or "")
                res = await execute_gh_pr_list(repo)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "gh_issue_list":
                repo = args.get("repo", args) if isinstance(args, dict) else (args or "")
                res = await execute_gh_issue_list(repo)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "gh_repo_view":
                repo = args.get("repo", args) if isinstance(args, dict) else args
                res = await execute_gh_repo_view(repo)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "gh_workflow_list":
                repo = args.get("repo", args) if isinstance(args, dict) else (args or "")
                res = await execute_gh_workflow_list(repo)
                context += f"\n\n{res}"
                final_response += res

            # Browser tools
            elif tool == "browser_fetch":
                if isinstance(args, dict):
                    res = await execute_browser_fetch(args.get("url", ""), args.get("wait_for"))
                else:
                    res = await execute_browser_fetch(args)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "browser_extract":
                if isinstance(args, dict):
                    res = await execute_browser_extract(args.get("url", ""), args.get("selector", ""))
                else:
                    res = "browser_extract requires {url, selector}"
                context += f"\n\n{res}"
                final_response += res

            elif tool == "browser_screenshot":
                if isinstance(args, dict):
                    res = await execute_browser_screenshot(args.get("url", ""), args.get("full_page", False))
                else:
                    res = await execute_browser_screenshot(args)
                context += f"\n\n{res}"
                final_response += res

            # Spreadsheet tools
            elif tool == "read_excel":
                if isinstance(args, dict):
                    res = await execute_read_excel(args.get("path", ""), args.get("sheet"))
                else:
                    res = await execute_read_excel(args)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "read_csv":
                path = args.get("path", args) if isinstance(args, dict) else args
                res = await execute_read_csv(path)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "analyze_data":
                if isinstance(args, dict):
                    res = await execute_analyze_data(args.get("path", ""), args.get("query", ""))
                else:
                    res = await execute_analyze_data(args, "basic analysis")
                context += f"\n\n{res}"
                final_response += res

            elif tool == "create_excel":
                if isinstance(args, dict):
                    res = await execute_create_excel(args.get("path", ""), args.get("data", {}))
                else:
                    res = "create_excel requires {path, data}"
                context += f"\n\n{res}"
                final_response += res

            # Library docs tools
            elif tool == "context7_search":
                library = args.get("library", args) if isinstance(args, dict) else args
                res = await execute_context7_search(library)
                context += f"\n\n{res}"
                final_response += res

            elif tool == "context7_docs":
                if isinstance(args, dict):
                    res = await execute_context7_docs(args.get("library", ""), args.get("query", ""))
                else:
                    res = "context7_docs requires {library, query}"
                context += f"\n\n{res}"
                final_response += res

            # MCP builder
            elif tool == "create_mcp_server":
                if isinstance(args, dict):
                    res = await execute_create_mcp_server(
                        args.get("name", "my-server"),
                        args.get("description", "MCP Server"),
                        args.get("tools", [])
                    )
                else:
                    res = "create_mcp_server requires {name, description, tools}"
                context += f"\n\n{res}"
                final_response += res

            # Email tools
            elif tool == "email_list":
                folder = args.get("folder", "INBOX") if isinstance(args, dict) else "INBOX"
                count = int(args.get("count", 10)) if isinstance(args, dict) else 10
                res = await execute_email_list(folder, count)
                context += f"\n\n{res}"
                final_response += res
            elif tool == "email_read":
                if isinstance(args, dict) and args.get("email_id"):
                    res = await execute_email_read(args["email_id"], args.get("folder", "INBOX"))
                else:
                    res = "email_read requires {email_id, folder}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "email_send":
                if isinstance(args, dict) and args.get("to") and args.get("subject") and args.get("body"):
                    res = await execute_email_send(args["to"], args["subject"], args["body"], args.get("cc", ""))
                else:
                    res = "email_send requires {to, subject, body}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "email_search":
                if isinstance(args, dict) and args.get("query"):
                    res = await execute_email_search(args["query"], args.get("folder", "INBOX"))
                else:
                    res = "email_search requires {query}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "email_unread":
                folder = args.get("folder", "INBOX") if isinstance(args, dict) else "INBOX"
                res = await execute_email_unread(folder)
                context += f"\n\n{res}"
                final_response += res

            # API Testing
            elif tool == "api_request":
                if isinstance(args, dict) and args.get("url"):
                    res = await execute_api_request(
                        args.get("method", "GET"),
                        args["url"],
                        args.get("headers", {}),
                        args.get("body", ""),
                        args.get("timeout", 30)
                    )
                else:
                    res = "api_request requires {method, url}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "api_test_suite":
                if isinstance(args, dict) and args.get("tests"):
                    res = await execute_api_test_suite(args["tests"])
                else:
                    res = "api_test_suite requires {tests: [...]}"
                context += f"\n\n{res}"
                final_response += res

            # Diagrams
            elif tool == "create_diagram":
                if isinstance(args, dict) and args.get("code"):
                    res = await execute_create_diagram(
                        args.get("type", "flowchart"),
                        args["code"],
                        args.get("output_path", "")
                    )
                else:
                    res = "create_diagram requires {code}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "flowchart":
                desc = args.get("description", args) if isinstance(args, dict) else str(args)
                res = await execute_flowchart(desc)
                context += f"\n\n{res}"
                final_response += res

            # PDF Tools
            elif tool == "pdf_read":
                if isinstance(args, dict) and args.get("path"):
                    res = await execute_pdf_read(args["path"], args.get("pages", ""))
                else:
                    res = "pdf_read requires {path}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "pdf_merge":
                if isinstance(args, dict) and args.get("files") and args.get("output"):
                    res = await execute_pdf_merge(args["files"], args["output"])
                else:
                    res = "pdf_merge requires {files: [...], output}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "pdf_create":
                if isinstance(args, dict) and args.get("content") and args.get("output"):
                    res = await execute_pdf_create(args["content"], args["output"], args.get("title", "Document"))
                else:
                    res = "pdf_create requires {content, output}"
                context += f"\n\n{res}"
                final_response += res

            # Whisper
            elif tool == "transcribe":
                if isinstance(args, dict) and args.get("audio_path"):
                    res = await execute_transcribe(args["audio_path"], args.get("language", ""))
                else:
                    res = "transcribe requires {audio_path}"
                context += f"\n\n{res}"
                final_response += res

            # Database
            elif tool == "sql_query":
                if isinstance(args, dict) and args.get("database") and args.get("query"):
                    res = await execute_sql_query(args["database"], args["query"])
                else:
                    res = "sql_query requires {database, query}"
                context += f"\n\n{res}"
                final_response += res
            elif tool == "db_schema":
                if isinstance(args, dict) and args.get("database"):
                    res = await execute_db_schema(args["database"])
                else:
                    res = "db_schema requires {database}"
                context += f"\n\n{res}"
                final_response += res

            elif tool == "memory_search":
                res = await execute_memory_search(args)
                context += f"\n\n**Memory:**\n{res}"

            elif tool == "rag_search":
                res = await execute_rag_search(args)
                context += f"\n\n**Knowledge Base:**\n{res}"

            elif tool == "reasoning":
                intent = detect_intent(str(args), has_image)
                # Route based on intent: code → code model, complex → planner, else → research
                if intent == "code":
                    target_model = "code"
                elif intent == "planner":
                    target_model = "planner"
                else:
                    target_model = "research"
                print(f"[TOOL:REASONING] Intent: {intent} → Model: {target_model}")

                msgs = [
                    {"role": "system", "content": "You are BenchAI, an expert engineering assistant. Be helpful, accurate, and concise."},
                    {"role": "user", "content": prompt_with_context}
                ]

                if request.stream:
                    # For streaming, we need to handle this differently
                    res = await execute_reasoning(msgs, target_model)
                else:
                    res = await execute_reasoning(msgs, target_model)

                final_response += res

        except Exception as e:
            print(f"[ERROR] Tool '{tool}' failed: {e}")
            context += f"\n\n**Error in {tool}:** {str(e)}"

    # Synthesize if needed
    if not final_response and context:
        msgs = [
            {"role": "system", "content": "You are BenchAI. Synthesize the following information into a helpful response."},
            {"role": "user", "content": f"User asked: {text_query}\n\nGathered information:\n{context}"}
        ]
        final_response = await execute_reasoning(msgs, "research")

    if not final_response:
        final_response = "I was unable to process your request. Please try rephrasing."

    # Log response
    await memory_manager.log_conversation("assistant", final_response)

    # Add assistant turn to session for follow-up context
    session_manager.add_turn(session_id, "assistant", final_response)

    # --- LEARNING SYSTEM: Log interaction for future training ---
    if LEARNING_AVAILABLE and len(final_response) > 100:
        try:
            # Log successful completion for experience replay
            await log_chat_completion(
                request={"messages": messages_dicts, "model": request.model},
                response={"content": final_response[:2000]},  # Truncate for storage
                model=request.model or "auto",
                tokens_in=sum(len(str(m.get("content", ""))) // 4 for m in messages_dicts),
                tokens_out=len(final_response) // 4,
                duration_ms=int((time.time() - app.state.start_time) * 1000) % 60000,
                success=True,
                session_id=session_id
            )

            # For significant tasks, record as successful experience
            if len(text_query) > 100 and len(final_response) > 500:
                await record_task_outcome(
                    task=text_query[:200],
                    approach=f"Used agentic planning with tools",
                    success=True,
                    details=final_response[:500],
                    domain="general"
                )
        except Exception as e:
            print(f"[LEARNING] Logging skipped: {e}")

    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "auto-agent",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": final_response}, "finish_reason": "stop"}]
    }

    # Cache the response for future identical requests
    if not request.stream:
        request_cache.set(request.model, messages_dicts, request.max_tokens, response)

    return response

@app.post("/v1/images/generations")
async def generate_image_endpoint(request: ImageRequest):
    res = await execute_image_gen(request.prompt)
    if "http" in res:
        url = res.split("(")[1].split(")")[0]
        return {"created": int(time.time()), "data": [{"url": url}]}
    raise HTTPException(500, res)

# --- MEMORY ENDPOINTS ---

@app.get("/v1/cache/stats")
async def cache_stats():
    """Get request cache statistics."""
    return request_cache.stats()

@app.post("/v1/memory/add")
async def add_memory(request: MemoryRequest):
    await memory_manager.add_memory(request.content, request.category)
    return {"status": "ok", "message": "Memory saved"}

@app.get("/v1/memory/search")
async def search_memory(q: str = Query(...)):
    memories = await memory_manager.search_memories(q)
    return {"results": memories}

@app.get("/v1/memory/recent")
async def recent_memories(limit: int = 10):
    memories = await memory_manager.get_recent_memories(limit)
    return {"results": memories}

@app.get("/v1/memory/stats")
async def memory_stats():
    """Get detailed memory database statistics."""
    await memory_manager.initialize()
    async with aiosqlite.connect(memory_manager.db_path) as db:
        await memory_manager._optimize_connection(db)

        stats = {}

        # Memory counts by category
        cursor = await db.execute(
            "SELECT category, COUNT(*) as count FROM memories GROUP BY category"
        )
        stats["by_category"] = {row[0]: row[1] for row in await cursor.fetchall()}

        # Total counts
        cursor = await db.execute("SELECT COUNT(*) FROM memories")
        stats["total_memories"] = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(*) FROM conversations")
        stats["total_conversations"] = (await cursor.fetchone())[0]

        cursor = await db.execute("SELECT COUNT(*) FROM user_preferences")
        stats["total_preferences"] = (await cursor.fetchone())[0]

        # Database info
        cursor = await db.execute("PRAGMA journal_mode")
        stats["journal_mode"] = (await cursor.fetchone())[0]

        cursor = await db.execute("PRAGMA cache_size")
        stats["cache_size"] = (await cursor.fetchone())[0]

        cursor = await db.execute("PRAGMA page_count")
        page_count = (await cursor.fetchone())[0]
        cursor = await db.execute("PRAGMA page_size")
        page_size = (await cursor.fetchone())[0]
        stats["db_size_bytes"] = page_count * page_size

        # FTS5 status
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM memories_fts")
            stats["fts5_indexed"] = (await cursor.fetchone())[0]
            stats["fts5_enabled"] = True
        except:
            stats["fts5_enabled"] = False

        return stats

@app.post("/v1/memory/optimize")
async def optimize_memory():
    """Run maintenance tasks on the memory database."""
    await memory_manager.initialize()
    async with aiosqlite.connect(memory_manager.db_path) as db:
        await memory_manager._optimize_connection(db)

        # Run VACUUM to reclaim space and defragment
        await db.execute("VACUUM")

        # Run ANALYZE for query optimizer
        await db.execute("ANALYZE")

        # Checkpoint WAL file
        await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Rebuild FTS5 index
        try:
            await db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except:
            pass

        await db.commit()

    return {"status": "ok", "message": "Database optimized (VACUUM, ANALYZE, WAL checkpoint, FTS rebuild)"}

@app.post("/v1/memory/migrate-fts")
async def migrate_fts():
    """Migrate existing memories to FTS5 index."""
    await memory_manager.initialize()
    async with aiosqlite.connect(memory_manager.db_path) as db:
        await memory_manager._optimize_connection(db)

        # Get all memories not yet in FTS
        cursor = await db.execute('''
            SELECT id, content, category FROM memories
            WHERE id NOT IN (SELECT rowid FROM memories_fts)
        ''')
        rows = await cursor.fetchall()

        migrated = 0
        for row in rows:
            try:
                await db.execute(
                    "INSERT INTO memories_fts(rowid, content, category) VALUES (?, ?, ?)",
                    (row[0], row[1], row[2])
                )
                migrated += 1
            except:
                pass

        await db.commit()

    return {"status": "ok", "migrated": migrated}

# --- TTS ENDPOINTS ---

@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest):
    if not PIPER_AVAILABLE:
        raise HTTPException(501, "TTS not available")

    audio_file = await tts_manager.synthesize(request.text)
    if audio_file is None:
        raise HTTPException(500, "TTS synthesis failed")

    return FileResponse(audio_file, media_type="audio/wav")

# --- RAG ENDPOINTS ---

# --- API TESTER TOOLS ---

async def execute_api_request(method: str, url: str, headers: dict = None, body: str = "", timeout: int = 30) -> str:
    """Make an HTTP request and return the response."""
    print(f"[TOOL:API] {method} {url}")
    try:
        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            headers = headers or {}

            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, content=body)
            elif method.upper() == "PUT":
                response = await client.put(url, headers=headers, content=body)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            elif method.upper() == "PATCH":
                response = await client.patch(url, headers=headers, content=body)
            else:
                return f"Error: Unsupported method {method}"

            # Format response
            result = f"**{method.upper()} {url}**\n"
            result += f"**Status:** {response.status_code} {response.reason_phrase}\n\n"
            result += f"**Headers:**\n```\n"
            for k, v in response.headers.items():
                result += f"{k}: {v}\n"
            result += "```\n\n"

            # Try to parse as JSON
            try:
                body_json = response.json()
                result += f"**Body (JSON):**\n```json\n{json.dumps(body_json, indent=2)[:3000]}\n```"
            except:
                result += f"**Body:**\n```\n{response.text[:3000]}\n```"

            return result
    except Exception as e:
        return f"Error: {str(e)}"


async def execute_api_test_suite(tests: list) -> str:
    """Run multiple API tests and report results."""
    print(f"[TOOL:API] Running {len(tests)} tests")
    results = []
    passed = 0
    failed = 0

    for i, test in enumerate(tests):
        try:
            method = test.get("method", "GET")
            url = test.get("url", "")
            expected_status = test.get("expected_status", 200)
            headers = test.get("headers", {})
            body = test.get("body", "")

            async with httpx.AsyncClient(timeout=10, verify=False) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, content=body)
                else:
                    response = await client.request(method.upper(), url, headers=headers, content=body)

                if response.status_code == expected_status:
                    passed += 1
                    results.append(f"✅ Test {i+1}: {method} {url} - {response.status_code}")
                else:
                    failed += 1
                    results.append(f"❌ Test {i+1}: {method} {url} - Expected {expected_status}, got {response.status_code}")
        except Exception as e:
            failed += 1
            results.append(f"❌ Test {i+1}: {test.get('url', 'unknown')} - Error: {str(e)}")

    summary = f"**API Test Results: {passed}/{passed+failed} passed**\n\n"
    return summary + "\n".join(results)


# --- DIAGRAM GENERATION TOOLS ---

async def execute_create_diagram(diagram_type: str, code: str, output_path: str = "") -> str:
    """Create a diagram using Mermaid syntax."""
    print(f"[TOOL:DIAGRAM] Creating {diagram_type} diagram")
    try:
        # Default output path
        if not output_path:
            output_path = f"/tmp/diagram_{int(time.time())}.png"

        # Create temp mermaid file
        mmd_file = f"/tmp/diagram_{int(time.time())}.mmd"
        with open(mmd_file, 'w') as f:
            f.write(code)

        # Run mermaid-cli
        result = subprocess.run(
            ["mmdc", "-i", mmd_file, "-o", output_path, "-b", "transparent"],
            capture_output=True,
            text=True,
            timeout=30
        )

        os.remove(mmd_file)

        if result.returncode == 0:
            return f"Diagram created: {output_path}"
        else:
            return f"Error creating diagram: {result.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"


async def execute_flowchart(description: str) -> str:
    """Generate a flowchart from a description."""
    print(f"[TOOL:DIAGRAM] Creating flowchart: {description[:50]}")
    # This would ideally use an LLM to convert description to mermaid
    # For now, return a template
    template = f"""```mermaid
flowchart TD
    A[Start] --> B{{Decision}}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```

To create a custom flowchart, use `create_diagram` with Mermaid syntax.
Description requested: {description}"""
    return template


# --- PDF TOOLS ---

async def execute_pdf_read(path: str, pages: str = "") -> str:
    """Read text content from a PDF file."""
    print(f"[TOOL:PDF] Reading: {path}")
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        total_pages = len(reader.pages)

        # Parse page range
        if pages:
            try:
                if "-" in pages:
                    start, end = pages.split("-")
                    page_nums = range(int(start)-1, min(int(end), total_pages))
                else:
                    page_nums = [int(pages)-1]
            except:
                page_nums = range(min(10, total_pages))
        else:
            page_nums = range(min(10, total_pages))

        text = ""
        for i in page_nums:
            if 0 <= i < total_pages:
                text += f"\n--- Page {i+1} ---\n"
                text += reader.pages[i].extract_text() or "[No text extracted]"

        result = f"**PDF: {path}**\n"
        result += f"**Pages:** {total_pages}\n\n"
        result += text[:5000]

        return result
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


async def execute_pdf_merge(files: list, output: str) -> str:
    """Merge multiple PDF files into one."""
    print(f"[TOOL:PDF] Merging {len(files)} files")
    try:
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()

        for pdf_path in files:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)

        with open(output, 'wb') as f:
            writer.write(f)

        return f"Merged {len(files)} PDFs into: {output}"
    except Exception as e:
        return f"Error merging PDFs: {str(e)}"


async def execute_pdf_create(content: str, output: str, title: str = "Document") -> str:
    """Create a PDF from text content."""
    print(f"[TOOL:PDF] Creating: {output}")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        c = canvas.Canvas(output, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, height - 1*inch, title)

        # Content
        c.setFont("Helvetica", 11)
        y = height - 1.5*inch

        for line in content.split('\n'):
            if y < 1*inch:
                c.showPage()
                y = height - 1*inch
                c.setFont("Helvetica", 11)
            c.drawString(1*inch, y, line[:80])
            y -= 14

        c.save()
        return f"PDF created: {output}"
    except Exception as e:
        return f"Error creating PDF: {str(e)}"


# --- WHISPER TRANSCRIPTION ---

async def execute_transcribe(audio_path: str, language: str = "") -> str:
    """Transcribe audio file using Whisper."""
    print(f"[TOOL:WHISPER] Transcribing: {audio_path}")
    try:
        import whisper

        # Load model (use base for speed, can change to medium/large for accuracy)
        model = whisper.load_model("base")

        # Transcribe
        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(audio_path, **options)

        text = result["text"]
        detected_lang = result.get("language", "unknown")

        response = f"**Transcription of {audio_path}**\n"
        response += f"**Language:** {detected_lang}\n\n"
        response += f"**Text:**\n{text}"

        return response
    except Exception as e:
        return f"Error transcribing: {str(e)}"


# --- DATABASE TOOLS ---

async def execute_sql_query(database: str, query: str) -> str:
    """Execute SQL query on SQLite database."""
    print(f"[TOOL:SQL] Query on {database}")
    try:
        import sqlite3

        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # Execute query
        cursor.execute(query)

        if query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            result = f"**Query:** `{query}`\n"
            result += f"**Rows:** {len(rows)}\n\n"

            if columns:
                result += "| " + " | ".join(columns) + " |\n"
                result += "| " + " | ".join(["---"] * len(columns)) + " |\n"

                for row in rows[:50]:
                    result += "| " + " | ".join(str(v)[:30] for v in row) + " |\n"
        else:
            conn.commit()
            result = f"Query executed. Rows affected: {cursor.rowcount}"

        conn.close()
        return result
    except Exception as e:
        return f"SQL Error: {str(e)}"


async def execute_db_schema(database: str) -> str:
    """Get database schema for SQLite database."""
    print(f"[TOOL:SQL] Getting schema for {database}")
    try:
        import sqlite3

        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        result = f"**Database Schema: {database}**\n\n"

        for (table_name,) in tables:
            result += f"### {table_name}\n"
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            result += "| Column | Type | Nullable | Default | PK |\n"
            result += "|--------|------|----------|---------|----|\n"

            for col in columns:
                cid, name, ctype, notnull, default, pk = col
                result += f"| {name} | {ctype} | {'NO' if notnull else 'YES'} | {default or ''} | {'✓' if pk else ''} |\n"
            result += "\n"

        conn.close()
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# --- PROTON EMAIL TOOLS ---

# Proton Mail Configuration
PROTON_IMAP_HOST = os.environ.get("PROTON_IMAP_HOST", "127.0.0.1")
PROTON_IMAP_PORT = int(os.environ.get("PROTON_IMAP_PORT", "1143"))
PROTON_SMTP_HOST = os.environ.get("PROTON_SMTP_HOST", "127.0.0.1")
PROTON_SMTP_PORT = int(os.environ.get("PROTON_SMTP_PORT", "1025"))
PROTON_USER = os.environ.get("PROTON_BRIDGE_USER", "")
PROTON_PASS = os.environ.get("PROTON_BRIDGE_PASS", "")


def get_proton_imap():
    """Create IMAP connection to ProtonMail Bridge/Hydroxide."""
    import imaplib
    try:
        imap = imaplib.IMAP4(PROTON_IMAP_HOST, PROTON_IMAP_PORT)
        # Don't use starttls for localhost (hydroxide doesn't support it)
        if PROTON_IMAP_HOST not in ("127.0.0.1", "localhost"):
            imap.starttls()
        imap.login(PROTON_USER, PROTON_PASS)
        return imap
    except Exception as e:
        print(f"[PROTON] IMAP error: {e}")
        return None


def decode_email_header(header):
    """Decode email header to readable string."""
    from email.header import decode_header
    if header is None:
        return ""
    decoded_parts = decode_header(header)
    result = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            result += part.decode(encoding or 'utf-8', errors='replace')
        else:
            result += part
    return result


async def execute_email_list(folder: str = "INBOX", count: int = 10) -> str:
    """List recent emails from a folder."""
    print(f"[TOOL:EMAIL] Listing {count} emails from {folder}")
    import imaplib
    import email

    if not PROTON_USER or not PROTON_PASS:
        return "Error: PROTON_BRIDGE_USER and PROTON_BRIDGE_PASS not configured"

    try:
        imap = get_proton_imap()
        if not imap:
            return "Error: Could not connect to ProtonMail Bridge"

        status, _ = imap.select(folder)
        if status != 'OK':
            imap.logout()
            return f"Error: Could not select folder '{folder}'"

        status, messages = imap.search(None, 'ALL')
        if status != 'OK':
            imap.logout()
            return "Error: Could not search emails"

        email_ids = messages[0].split()
        email_ids = email_ids[-count:]
        email_ids.reverse()

        results = []
        for eid in email_ids:
            status, msg_data = imap.fetch(eid, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])')
            if status == 'OK':
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                from_addr = decode_email_header(msg.get('From', ''))
                subject = decode_email_header(msg.get('Subject', '(No Subject)'))
                date = msg.get('Date', '')[:16]
                results.append(f"[{eid.decode()}] {date}\n   From: {from_addr[:50]}\n   Subject: {subject[:60]}")

        imap.logout()

        if not results:
            return f"No emails found in {folder}"

        return f"**Emails in {folder}:**\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error: {str(e)}"


async def execute_email_read(email_id: str, folder: str = "INBOX") -> str:
    """Read full email content by ID."""
    print(f"[TOOL:EMAIL] Reading email {email_id} from {folder}")
    import imaplib
    import email

    if not PROTON_USER or not PROTON_PASS:
        return "Error: PROTON_BRIDGE_USER and PROTON_BRIDGE_PASS not configured"

    try:
        imap = get_proton_imap()
        if not imap:
            return "Error: Could not connect to ProtonMail Bridge"

        status, _ = imap.select(folder)
        if status != 'OK':
            imap.logout()
            return f"Error: Could not select folder '{folder}'"

        status, msg_data = imap.fetch(email_id.encode(), '(RFC822)')
        if status != 'OK':
            imap.logout()
            return f"Error: Could not fetch email {email_id}"

        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        from_addr = decode_email_header(msg.get('From', ''))
        to_addr = decode_email_header(msg.get('To', ''))
        subject = decode_email_header(msg.get('Subject', '(No Subject)'))
        date = msg.get('Date', '')

        # Get body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        break
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            except:
                body = str(msg.get_payload())

        imap.logout()

        return f"""**Email #{email_id}**

**From:** {from_addr}
**To:** {to_addr}
**Date:** {date}
**Subject:** {subject}

---
{body[:3000]}
"""

    except Exception as e:
        return f"Error: {str(e)}"


async def execute_email_send(to: str, subject: str, body: str, cc: str = "") -> str:
    """Send an email via ProtonMail Bridge."""
    print(f"[TOOL:EMAIL] Sending to {to}")
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    if not PROTON_USER or not PROTON_PASS:
        return "Error: PROTON_BRIDGE_USER and PROTON_BRIDGE_PASS not configured"

    try:
        msg = MIMEMultipart()
        msg['From'] = PROTON_USER
        msg['To'] = to
        msg['Subject'] = subject
        if cc:
            msg['Cc'] = cc

        msg.attach(MIMEText(body, 'plain'))

        smtp = smtplib.SMTP(PROTON_SMTP_HOST, PROTON_SMTP_PORT)
        # Don't use starttls for localhost (hydroxide doesn't support it)
        if PROTON_SMTP_HOST not in ("127.0.0.1", "localhost"):
            smtp.starttls()
        smtp.login(PROTON_USER, PROTON_PASS)

        recipients = [to]
        if cc:
            recipients.extend([c.strip() for c in cc.split(',')])

        smtp.sendmail(PROTON_USER, recipients, msg.as_string())
        smtp.quit()

        return f"Email sent successfully to {to}"

    except Exception as e:
        return f"Error: {str(e)}"


async def execute_email_search(query: str, folder: str = "INBOX") -> str:
    """Search emails in a folder."""
    print(f"[TOOL:EMAIL] Searching '{query}' in {folder}")
    import imaplib
    import email

    if not PROTON_USER or not PROTON_PASS:
        return "Error: PROTON_BRIDGE_USER and PROTON_BRIDGE_PASS not configured"

    try:
        imap = get_proton_imap()
        if not imap:
            return "Error: Could not connect to ProtonMail Bridge"

        status, _ = imap.select(folder)
        if status != 'OK':
            imap.logout()
            return f"Error: Could not select folder '{folder}'"

        status, messages = imap.search(None, f'TEXT "{query}"')
        if status != 'OK':
            imap.logout()
            return "Error: Search failed"

        email_ids = messages[0].split()[-20:]

        results = []
        for eid in email_ids:
            status, msg_data = imap.fetch(eid, '(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])')
            if status == 'OK':
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                from_addr = decode_email_header(msg.get('From', ''))[:40]
                subject = decode_email_header(msg.get('Subject', ''))[:50]
                results.append(f"[{eid.decode()}] From: {from_addr} | {subject}")

        imap.logout()

        if not results:
            return f"No emails matching '{query}'"

        return f"**Search results for '{query}':**\n\n" + "\n".join(results)

    except Exception as e:
        return f"Error: {str(e)}"


async def execute_email_unread(folder: str = "INBOX") -> str:
    """Get count of unread emails."""
    print(f"[TOOL:EMAIL] Getting unread count for {folder}")
    import imaplib

    if not PROTON_USER or not PROTON_PASS:
        return "Error: PROTON_BRIDGE_USER and PROTON_BRIDGE_PASS not configured"

    try:
        imap = get_proton_imap()
        if not imap:
            return "Error: Could not connect to ProtonMail Bridge"

        status, _ = imap.select(folder)
        if status != 'OK':
            imap.logout()
            return f"Error: Could not select folder '{folder}'"

        status, messages = imap.search(None, 'UNSEEN')
        if status != 'OK':
            imap.logout()
            return "Error: Could not search"

        unread_count = len(messages[0].split()) if messages[0] else 0
        imap.logout()

        return f"**Unread emails in {folder}:** {unread_count}"

    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/v1/rag/index")
async def index_path(request: IndexRequest):
    if not CHROMADB_AVAILABLE:
        raise HTTPException(501, "RAG not available")

    path = Path(request.path).expanduser()

    if path.is_file():
        success = await rag_manager.index_file(path)
        return {"status": "ok" if success else "failed", "indexed": 1 if success else 0}
    elif path.is_dir():
        count = await rag_manager.index_directory(path)
        return {"status": "ok", "indexed": count}
    else:
        raise HTTPException(404, f"Path not found: {request.path}")

@app.get("/v1/rag/search")
async def search_rag(q: str = Query(...), n: int = 3):
    if not CHROMADB_AVAILABLE:
        raise HTTPException(501, "RAG not available")

    results = await rag_manager.search(q, n)
    return {"results": results}

@app.get("/v1/rag/stats")
async def rag_stats():
    return rag_manager.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)
