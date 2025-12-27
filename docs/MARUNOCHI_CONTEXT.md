# MarunochiAI Integration Context

**Generated**: 2025-12-26 22:15 UTC
**From**: BenchAI Orchestrator
**To**: MarunochiAI Programmer Agent

---

## Current System State

### BenchAI Orchestrator Status
- **Version**: v3.5 (Router)
- **Status**: ONLINE at `http://[BENCHAI_IP]:8085`
- **Features Active**:
  - Streaming: ✅
  - Memory: ✅ (35 memories, 4 from MarunochiAI)
  - TTS: ✅
  - RAG: ✅ (346 documents)
  - Zettelkasten: ✅ (15 zettels, 45 links)
  - Experience Replay: ✅ (3 experiences)
  - Multi-Agent A2A: ✅

### Agent Registry
| Agent | Role | Status | Endpoint | Load |
|-------|------|--------|----------|------|
| BenchAI | orchestrator | ONLINE | localhost:8085 | 0% |
| MarunochiAI | programmer | ONLINE | 192.168.1.101:8086 | 20% |
| DottscavisAI | creative | ONLINE | (pending) | 10% |

---

## Completed Work (BenchAI Side)

### 1. A2A v0.3 Agent Cards
- `/.well-known/agent.json` endpoint implemented
- Full Agent Card schema with skills, capabilities, input/output modes
- Agent registration stores full Agent Cards
- Endpoint: `GET /v1/learning/agents/{agent_id}/card`

### 2. Semantic Task Router
Location: `router/learning/semantic_router.py`

**Routing Logic**:
- Coding tasks → MarunochiAI (confidence: 100%)
- Creative tasks → DottscavisAI (confidence: 85%)
- Research tasks → BenchAI (confidence: 98%)

**Usage**:
```python
# Auto-route a task
response = await client.post(
    f"{BENCHAI_URL}/v1/learning/a2a/task",
    json={
        "from_agent": "marunochiAI",
        "to_agent": "auto",  # <-- Semantic routing
        "task_type": "research",
        "task_description": "Best practices for async database connections"
    }
)
# Response includes routing info with confidence and reasoning
```

**Preview routing without submitting**:
```python
response = await client.post(
    f"{BENCHAI_URL}/v1/learning/a2a/route",
    params={"task_description": "Debug the Python TypeError in auth module"}
)
# Returns recommended agent + alternatives with confidence scores
```

---

## API Endpoints Available

### Core A2A Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/learning/agents/register` | POST | Register with full Agent Card |
| `/v1/learning/agents/{id}/card` | GET | Get agent's Agent Card |
| `/v1/learning/a2a/heartbeat` | POST | Send heartbeat (every 30s) |
| `/v1/learning/a2a/task` | POST | Submit task (supports `to_agent="auto"`) |
| `/v1/learning/a2a/route` | POST | Preview routing decision |
| `/v1/learning/a2a/discover` | GET | Discover all agents with cards |
| `/v1/learning/a2a/broadcast` | POST | Broadcast to all agents |
| `/v1/learning/a2a/messages` | GET | Get shared context |

### Research & Knowledge
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/learning/research/query` | POST | Submit async research query |
| `/v1/learning/research/result/{id}` | GET | Get research results |
| `/v1/learning/zettelkasten/search` | GET | Search knowledge graph |
| `/v1/learning/memory/store` | POST | Store memory |
| `/v1/learning/memory/search` | GET | Search memories |

### Experience & Learning
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/learning/experience/record` | POST | Record successful task |
| `/v1/learning/experience/similar` | GET | Find similar experiences |
| `/v1/learning/experience/examples` | GET | Get in-context examples |

---

## Recommended MarunochiAI Implementation

### 1. FastAPI Server Structure
```python
from fastapi import FastAPI, BackgroundTasks
import httpx
import asyncio

BENCHAI_URL = "http://[BENCHAI_IP]:8085"
app = FastAPI(title="MarunochiAI", version="1.0.0")

@app.on_event("startup")
async def startup():
    # Register with full Agent Card
    await register_with_benchai()
    # Start heartbeat loop
    asyncio.create_task(heartbeat_loop())

@app.post("/a2a/task")
async def receive_task(task: dict):
    """Receive tasks from other agents."""
    task_type = task.get("task_type")
    if task_type == "code":
        return await handle_coding_task(task)
    elif task_type == "debug":
        return await handle_debug_task(task)
    elif task_type == "review":
        return await handle_review_task(task)
    return {"status": "unknown_task_type"}

@app.get("/.well-known/agent.json")
async def agent_card():
    """Return MarunochiAI's Agent Card."""
    return {
        "name": "MarunochiAI",
        "description": "Professional coding agent for Python, TypeScript, full-stack",
        "url": "http://YOUR_IP:8086",
        "version": "1.0.0",
        "protocol_version": "0.3",
        "capabilities": ["coding", "debugging", "testing", "code_review"],
        "skills": [...]
    }
```

### 2. Task Handling Pattern
```python
async def handle_coding_task(task: dict):
    task_id = task.get("task_id")
    description = task.get("task_description")

    # 1. Get relevant context from BenchAI
    context = await get_research_context(description)

    # 2. Execute the task with your local LLM
    result = await execute_with_llm(description, context)

    # 3. Record success to BenchAI for learning
    await record_experience(task, result)

    # 4. Return result
    return {"status": "completed", "result": result}

async def get_research_context(query: str):
    """Get relevant knowledge from BenchAI."""
    async with httpx.AsyncClient() as client:
        # Get similar experiences
        exp = await client.get(
            f"{BENCHAI_URL}/v1/learning/experience/examples",
            params={"task": query, "limit": 3}
        )
        # Get knowledge graph context
        zk = await client.get(
            f"{BENCHAI_URL}/v1/learning/zettelkasten/search",
            params={"query": query, "limit": 5}
        )
        return {"experiences": exp.json(), "knowledge": zk.json()}
```

### 3. Recording Experiences
```python
async def record_experience(task: dict, result: dict):
    """Record successful task for collective learning."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{BENCHAI_URL}/v1/learning/experience/record",
            json={
                "task": task.get("task_description"),
                "approach": "Used systematic debugging with error analysis",
                "trajectory": [
                    {"action": "analyze", "result": "identified root cause"},
                    {"action": "implement", "result": "applied fix"},
                    {"action": "verify", "result": "tests passing"}
                ],
                "domain": "coding",
                "outcome": "success",
                "outcome_details": result.get("summary", ""),
                "success_score": 0.95,
                "agent": "marunochiAI"
            }
        )
```

---

## Pending Work (Collaboration Needed)

### 1. OpenTelemetry Monitoring
- Track latency across agent calls
- Monitor task success rates
- Distributed tracing for multi-agent workflows

### 2. Collective Learning Pipeline
- Cross-agent experience sharing
- LoRA fine-tuning from collective experiences
- Performance improvement tracking

### 3. Context Passing Protocol
- Rich context transfer between agents
- State preservation across handoffs
- Conversation history sharing

---

## Next Steps for MarunochiAI

1. **Implement FastAPI server** with `/a2a/task` endpoint
2. **Set up heartbeat loop** (every 30 seconds)
3. **Integrate with local LLM** (Qwen2.5-Coder or similar)
4. **Test task reception** from BenchAI
5. **Record experiences** back to BenchAI for learning

---

## Quick Test Commands

```bash
# Check BenchAI is reachable
curl http://[BENCHAI_IP]:8085/health

# Get your Agent Card
curl http://[BENCHAI_IP]:8085/v1/learning/agents/marunochiAI/card

# Send a test heartbeat
curl -X POST http://[BENCHAI_IP]:8085/v1/learning/a2a/heartbeat \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "marunochiAI", "status": "online", "load": 0.2}'

# Test semantic routing
curl -X POST "http://[BENCHAI_IP]:8085/v1/learning/a2a/route?task_description=Debug+Python+code"
```

---

*This context was generated by BenchAI to help MarunochiAI integrate successfully.*
