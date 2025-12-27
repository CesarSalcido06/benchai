# MarunochiAI Engineer Guide

## Agent Identity
- **Name**: MarunochiAI
- **Role**: Programmer Agent
- **Hardware**: M4 Pro Mac 24GB RAM
- **Specialization**: Coding, debugging, testing, code review, refactoring

## Connection to BenchAI Orchestrator

BenchAI is running at `http://[BENCHAI_IP]:8085` and serves as the central knowledge repository and orchestrator.

### 1. Register Your Agent

On startup, register with BenchAI:

```python
import httpx
import asyncio

BENCHAI_URL = "http://[BENCHAI_IP]:8085"

async def register_agent():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BENCHAI_URL}/v1/learning/agents/register",
            json={
                "agent_id": "marunochiAI",
                "name": "MarunochiAI",
                "role": "programmer",
                "capabilities": ["coding", "debugging", "testing", "code_review", "refactoring"],
                "endpoint": "http://[YOUR_IP]:8086"  # Your local endpoint
            }
        )
        print(f"Registration: {response.json()}")
```

### 2. Send Heartbeats

Every 30 seconds, send a heartbeat to stay visible:

```python
async def heartbeat_loop():
    while True:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BENCHAI_URL}/v1/learning/a2a/heartbeat",
                json={
                    "agent_id": "marunochiAI",
                    "status": "online",  # or "busy", "idle"
                    "current_task": "Reviewing code for project X",
                    "capabilities": ["coding", "debugging"],
                    "load": 0.3  # 0.0 to 1.0
                }
            )
        await asyncio.sleep(30)
```

### 3. Request Research from BenchAI

When you need deep research while coding, submit an async query:

```python
async def request_research(query: str, priority: str = "normal"):
    """
    Submit research request to BenchAI's Zettelkasten knowledge graph.
    Priority: "critical", "high", "normal", "low"
    """
    async with httpx.AsyncClient() as client:
        # Submit query
        response = await client.post(
            f"{BENCHAI_URL}/v1/learning/research/query",
            json={
                "query": query,
                "agent_id": "marunochiAI",
                "priority": priority,
                "expand_graph": True,
                "graph_depth": 3,
                "max_results": 15
            }
        )
        result = response.json()
        query_id = result["query_id"]

        # Continue working while query processes...
        # Later, retrieve result:
        result_response = await client.get(
            f"{BENCHAI_URL}/v1/learning/research/result/{query_id}",
            params={"wait": True, "timeout": 30}
        )
        return result_response.json()

# Example usage:
research = await request_research(
    "Best practices for implementing async database connections in Python",
    priority="high"
)
```

### 4. A2A Task Protocol

Submit tasks to other agents or receive tasks:

```python
# Submit task to BenchAI for research
async def submit_task_to_benchai(task_description: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BENCHAI_URL}/v1/learning/a2a/task",
            json={
                "from_agent": "marunochiAI",
                "to_agent": "benchai",
                "task_type": "research",
                "task_description": task_description,
                "priority": "normal",
                "context": {"project": "current_project_name"}
            }
        )
        return response.json()

# Submit task to DottscavisAI for creative work
async def request_creative_asset(description: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BENCHAI_URL}/v1/learning/a2a/task",
            json={
                "from_agent": "marunochiAI",
                "to_agent": "dottscavisAI",
                "task_type": "creative",
                "task_description": description,
                "priority": "normal"
            }
        )
        return response.json()
```

### 5. Memory & Knowledge APIs

Store and retrieve knowledge:

```python
# Store a memory (useful finding, solution, etc.)
async def remember(content: str, category: str = "coding"):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{BENCHAI_URL}/v1/learning/memory/store",
            json={
                "content": content,
                "memory_type": "procedural",  # or "semantic", "episodic"
                "category": category,
                "importance": 4,
                "source": "marunochiAI"
            }
        )

# Search memories
async def recall(query: str, limit: int = 5):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BENCHAI_URL}/v1/learning/memory/search",
            params={"query": query, "limit": limit}
        )
        return response.json()

# Search Zettelkasten for deep knowledge
async def search_knowledge(query: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BENCHAI_URL}/v1/learning/zettelkasten/search",
            params={"query": query, "limit": 10, "expand_graph": True}
        )
        return response.json()
```

### 6. Experience Recording

Record successful solutions for future learning:

```python
async def record_success(task: str, approach: str, details: str):
    """Record successful programming task for experience replay."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{BENCHAI_URL}/v1/learning/experience/record",
            json={
                "task": task,
                "approach": approach,
                "trajectory": [
                    {"action": "analyze", "result": "understood requirements"},
                    {"action": "implement", "result": "wrote solution"},
                    {"action": "test", "result": "all tests passed"}
                ],
                "domain": "coding",
                "outcome": "success",
                "outcome_details": details,
                "success_score": 0.9,
                "agent": "marunochiAI"
            }
        )
```

### 7. Discover Other Agents

Find available agents for collaboration:

```python
async def discover_agents():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BENCHAI_URL}/v1/learning/a2a/discover")
        return response.json()

# Returns:
# {
#   "agents": [
#     {"id": "benchai", "role": "orchestrator", "status": "online", ...},
#     {"id": "dottscavisAI", "role": "creative", "status": "offline", ...}
#   ],
#   "orchestrator": {"id": "benchai", "endpoint": "http://...", "capabilities": [...]}
# }
```

### 8. Broadcast Messages

Share updates with all agents:

```python
async def broadcast(message: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{BENCHAI_URL}/v1/learning/a2a/broadcast",
            params={
                "message": message,
                "from_agent": "marunochiAI",
                "category": "status"
            }
        )
```

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/learning/agents/register` | POST | Register your agent |
| `/v1/learning/a2a/heartbeat` | POST | Send heartbeat |
| `/v1/learning/a2a/task` | POST | Submit task to agent |
| `/v1/learning/a2a/discover` | GET | Discover all agents |
| `/v1/learning/a2a/broadcast` | POST | Broadcast message |
| `/v1/learning/a2a/messages` | GET | Get shared messages |
| `/v1/learning/research/query` | POST | Submit research query |
| `/v1/learning/research/result/{id}` | GET | Get research result |
| `/v1/learning/memory/store` | POST | Store memory |
| `/v1/learning/memory/search` | GET | Search memories |
| `/v1/learning/zettelkasten/search` | GET | Search knowledge graph |
| `/v1/learning/experience/record` | POST | Record experience |
| `/v1/learning/experience/similar` | GET | Find similar experiences |
| `/health` | GET | Health check |

## Recommended Architecture

```
MarunochiAI (M4 Pro 24GB)
├── Local LLM Server (Ollama/llama.cpp)
│   └── Fine-tuned coding model (Qwen2.5-Coder, DeepSeek-Coder)
├── FastAPI Backend
│   ├── /v1/chat/completions - Local inference
│   ├── /v1/code/analyze - Code analysis
│   ├── /v1/code/review - Code review
│   └── /a2a/task - Receive tasks from other agents
├── BenchAI Client
│   ├── Heartbeat loop (30s)
│   ├── Research API client
│   └── Memory client
└── MLX Optimization
    └── Apple Silicon optimized inference
```

## Example: Full Integration

```python
import asyncio
import httpx
from fastapi import FastAPI

BENCHAI_URL = "http://192.168.1.100:8085"  # Replace with actual IP

app = FastAPI(title="MarunochiAI")

@app.on_event("startup")
async def startup():
    # Register with BenchAI
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{BENCHAI_URL}/v1/learning/agents/register",
            json={
                "agent_id": "marunochiAI",
                "name": "MarunochiAI",
                "role": "programmer",
                "capabilities": ["coding", "debugging", "testing"],
                "endpoint": "http://YOUR_LOCAL_IP:8086"
            }
        )

    # Start heartbeat
    asyncio.create_task(heartbeat_loop())

async def heartbeat_loop():
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BENCHAI_URL}/v1/learning/a2a/heartbeat",
                    json={
                        "agent_id": "marunochiAI",
                        "status": "online",
                        "load": 0.2
                    }
                )
        except:
            pass
        await asyncio.sleep(30)

@app.post("/a2a/task")
async def receive_task(task: dict):
    """Receive task from another agent."""
    # Process the task...
    return {"status": "received", "task_id": task.get("task_id")}
```

---

*Generated for BenchAI Multi-Agent System - December 2025*
