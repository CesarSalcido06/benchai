# BenchAI User Guide

How to use BenchAI for daily development tasks.

## Access Methods

| Method | URL/Command | Best For |
|--------|-------------|----------|
| Open WebUI | http://localhost:3000 | Full chat interface |
| CLI | `benchai "your query"` | Quick terminal queries |
| API | `curl localhost:8085/v1/...` | Scripting, automation |
| VS Code | Cmd/Ctrl+L | In-editor assistance |
| Neovim | `<leader>aa` | Vim users |

---

## Basic Usage

### Open WebUI

Simply chat naturally:

```
"what containers are running?"          → Lists Docker containers
"check homelab status"                   → System health report
"search the web for Python tutorials"    → Web search via SearXNG
"remember that I prefer dark mode"       → Saves to memory
"what do you remember about me?"         → Recalls memories
```

### CLI Tool

```bash
benchai --status                         # Health check
benchai "what's the weather?"            # Single query
benchai -i                               # Interactive mode
benchai --memory-stats                   # Show memory info
benchai --search "python"                # Search memories
```

### API

```bash
# Chat
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'

# Stream responses
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

---

## Workflows

### Daily Development

**Morning Check:**
```
"check homelab status"
"are all services healthy?"
"show docker container stats"
```

**During Development:**
```
"search my codebase for authentication"    → RAG search
"explain this error: [paste error]"        → Error explanation
"review this code: [paste code]"           → Code review
"generate tests for: [paste function]"     → Test generation
```

**End of Day:**
```
"git status in /home/user/myproject"
"git diff"
"commit with message: Added feature X"
```

### Learning

**Create Study Materials:**
```
"create a flashcard: Q: What is a closure? A: A function that captures outer scope variables"
"generate a quiz on Python basics with 5 questions"
"give me a learning path for machine learning"
```

**Study Session:**
```
"study my flashcards on JavaScript"
"explain recursion for a beginner"
```

### Research

**Web Research:**
```
"search the web for React best practices 2025"
"what is the current price of Bitcoin?"
```

**Codebase Research:**
```
"search my codebase for API endpoints"
"what files handle authentication?"
```

### DevOps

**Container Management:**
```
"list all containers"
"get jellyfin logs"
"restart sonarr"
"show docker stats"
```

### Git/GitHub

```
"git status in /path/to/repo"
"create a PR with title: Add feature X"
"list open issues in myrepo"
"merge PR 123"
```

---

## Available Tools (88+)

### Infrastructure
| Pattern | Tool | Example |
|---------|------|---------|
| "containers", "docker" | docker_ps | "what containers are running?" |
| "container logs" | docker_logs | "get jellyfin logs" |
| "restart/stop" | docker_control | "restart sonarr" |
| "homelab status" | homelab_status | "check homelab status" |

### Development
| Pattern | Tool | Example |
|---------|------|---------|
| "git status" | git_status | "git status in /path" |
| "git diff" | git_diff | "show git changes" |
| "commit" | git_commit | "commit: Fix bug" |
| "PR", "pull request" | gh_pr_* | "create a PR" |

### Learning
| Pattern | Tool | Example |
|---------|------|---------|
| "flashcard" | create_flashcard | "create flashcard about X" |
| "quiz" | generate_quiz | "quiz me on React" |
| "learning path" | learning_path | "learning path for ML" |

### Research
| Pattern | Tool | Example |
|---------|------|---------|
| "search the web" | web_search | "search for tutorials" |
| "remember" | memory_add | "remember I like X" |
| "search codebase" | rag_search | "search for auth code" |

### Code
| Pattern | Tool | Example |
|---------|------|---------|
| "write code" | code_task | "write fibonacci" |
| "run this" | run_code | "run: print('hi')" |
| "review" | code_review | "review this function" |

---

## Model Selection

BenchAI auto-selects the best model, but you can specify:

| Model | Best For | Specify With |
|-------|----------|--------------|
| `auto` | Automatic selection | Default |
| `general` | Quick questions | `-m general` |
| `code` | Programming | `-m code` |
| `research` | Analysis | `-m research` |
| `vision` | Images | `-m vision` |

```bash
benchai -m code "write a sorting algorithm"
```

---

## Memory System

BenchAI remembers things you tell it.

### Save to Memory
```
"remember that I prefer tabs over spaces"
"remember my email is user@example.com"
"remember the API key is stored in .env"
```

### Recall from Memory
```
"what do you remember about my preferences?"
"what did I tell you about the API?"
```

### Memory API
```bash
# Add
curl -X POST http://localhost:8085/v1/memory/add \
  -d '{"content": "User prefers Python", "category": "preference"}'

# Search
curl "http://localhost:8085/v1/memory/search?q=python"

# Stats
curl http://localhost:8085/v1/memory/stats
```

---

## RAG (Codebase Search)

Search your indexed codebase semantically.

### Index Your Code
```bash
curl -X POST http://localhost:8085/v1/rag/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/myproject", "recursive": true}'
```

### Search
```
"search my codebase for authentication"
"what files handle user login?"
"find the database connection code"
```

---

## Text-to-Speech

Generate audio from text.

```bash
curl -X POST http://localhost:8085/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am BenchAI"}' \
  -o speech.wav

# Play it
aplay speech.wav  # Linux
afplay speech.wav # Mac
```

---

## Tips for Best Results

1. **Be Specific:** Include paths, names, details
   - Good: "git status in /home/user/myproject"
   - Bad: "check git"

2. **Use Keywords:** Include trigger words
   - "docker", "containers" for Docker
   - "search the web" for web search
   - "remember", "recall" for memory
   - "git status/diff/commit" for Git

3. **Chain Requests:** Ask follow-ups in same session

4. **Use RAG:** Index your codebase for semantic search

5. **Maintain Memory:** Save important info for later recall

---

## Maintenance

### Weekly
```bash
# Optimize memory database
curl -X POST http://localhost:8085/v1/memory/optimize

# Check stats
curl http://localhost:8085/v1/memory/stats
```

### Monthly
```bash
# Re-index codebase
curl -X POST http://localhost:8085/v1/rag/index \
  -d '{"path": "/home/user/projects"}'

# Clear logs
sudo truncate -s 0 /var/log/benchai-router.log
```

---

## Common Commands Cheatsheet

```bash
# Status
benchai --status
curl http://localhost:8085/health

# Service
sudo systemctl status benchai
sudo systemctl restart benchai

# Logs
journalctl -u benchai -f
tail -f /var/log/benchai-router.log

# Docker
docker ps
docker logs <container>

# Memory
curl http://localhost:8085/v1/memory/stats
curl -X POST http://localhost:8085/v1/memory/optimize

# RAG
curl http://localhost:8085/v1/rag/stats
```
