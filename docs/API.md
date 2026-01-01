# BenchAI API Reference

Complete API documentation for BenchAI router.

## Base URL

```
http://localhost:8085
```

## Authentication

No authentication required for local access. For remote access, use Twingate or similar VPN.

---

## Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "benchai-router-v3.5",
  "features": {
    "streaming": true,
    "memory": true,
    "tts": true,
    "rag": true,
    "obsidian": true
  }
}
```

---

### List Models

```http
GET /v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"id": "auto", "name": "Auto-Agent (Planner)"},
    {"id": "general", "name": "General (Phi-3 Mini)"},
    {"id": "research", "name": "Research (Qwen2.5 7B)"},
    {"id": "code", "name": "Code (Qwen2.5-Coder 14B)"},
    {"id": "vision", "name": "Vision (Qwen2-VL 7B)"},
    {"id": "math", "name": "Math (DeepSeek-Math 7B)"}
  ]
}
```

---

### Chat Completions

```http
POST /v1/chat/completions
```

**Request:**
```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false,
  "max_tokens": 2048,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1703520000,
  "model": "auto",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! I'm doing well, thank you for asking."
    },
    "finish_reason": "stop"
  }]
}
```

**Streaming:** Set `"stream": true` for Server-Sent Events (SSE) response.

---

### Memory Endpoints

#### Add Memory

```http
POST /v1/memory/add
```

**Request:**
```json
{
  "content": "User prefers dark mode",
  "category": "preference"
}
```

**Categories:** `preference`, `fact`, `context`, `code`, `general`, `user_stated`

**Response:**
```json
{
  "status": "ok",
  "message": "Memory saved"
}
```

#### Search Memories

```http
GET /v1/memory/search?q=dark+mode&limit=10
```

**Response:**
```json
[
  {
    "id": 1,
    "content": "User prefers dark mode",
    "category": "preference",
    "created_at": "2025-12-25T12:00:00",
    "relevance": 0.95
  }
]
```

#### Get Recent Memories

```http
GET /v1/memory/recent?limit=20
```

#### Get Memory Stats

```http
GET /v1/memory/stats
```

**Response:**
```json
{
  "total_memories": 42,
  "total_conversations": 100,
  "by_category": {
    "preference": 10,
    "fact": 15,
    "code": 17
  },
  "fts5_enabled": true,
  "journal_mode": "wal",
  "db_size_bytes": 217088
}
```

#### Optimize Memory Database

```http
POST /v1/memory/optimize
```

Runs VACUUM and ANALYZE on the database.

#### Migrate to FTS5

```http
POST /v1/memory/migrate-fts
```

Migrates existing memories to FTS5 full-text search index.

---

### RAG Endpoints

#### Index Documents

```http
POST /v1/rag/index
```

**Request:**
```json
{
  "path": "/home/user/myproject",
  "recursive": true
}
```

**Response:**
```json
{
  "status": "ok",
  "indexed": 150,
  "path": "/home/user/myproject"
}
```

#### Search Documents

```http
GET /v1/rag/search?q=authentication&n=5
```

**Response:**
```json
{
  "results": [
    {
      "content": "def authenticate_user(...)...",
      "source": "/home/user/myproject/auth.py",
      "score": 0.89
    }
  ]
}
```

#### Get RAG Stats

```http
GET /v1/rag/stats
```

**Response:**
```json
{
  "status": "ready",
  "count": 346
}
```

---

### Text-to-Speech

```http
POST /v1/audio/speech
```

**Request:**
```json
{
  "text": "Hello, I am BenchAI.",
  "voice": "default"
}
```

**Response:** WAV audio file (binary)

**Example:**
```bash
curl -X POST http://localhost:8085/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  -o speech.wav
```

---

### Image Generation

```http
POST /v1/images/generations
```

**Request:**
```json
{
  "prompt": "A cyberpunk city at night",
  "size": "1024x1024",
  "n": 1
}
```

**Note:** Requires ComfyUI with Flux model running on port 8188.

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found
- `422` - Validation error
- `500` - Internal server error
- `503` - Service unavailable (model not loaded)

---

## Rate Limits

No rate limits for local usage. Consider implementing if exposing publicly.

---

## Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8085"

# Chat
response = requests.post(f"{BASE_URL}/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello"}]
})
print(response.json()["choices"][0]["message"]["content"])

# Memory
requests.post(f"{BASE_URL}/v1/memory/add", json={
    "content": "User prefers Python",
    "category": "preference"
})

# Search
results = requests.get(f"{BASE_URL}/v1/memory/search?q=python").json()
```

### JavaScript

```javascript
const BASE_URL = "http://localhost:8085";

// Chat
const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "Hello" }]
  })
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### cURL

```bash
# Chat
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

# Stream
curl -X POST http://localhost:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Memory
curl -X POST http://localhost:8085/v1/memory/add \
  -H "Content-Type: application/json" \
  -d '{"content":"Test","category":"general"}'
```
