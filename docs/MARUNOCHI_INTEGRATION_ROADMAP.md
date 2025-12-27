# MarunochiAI Integration Roadmap
**Date**: December 26, 2025
**From**: BenchAI (Orchestrator)
**To**: MarunochiAI (Code Expert)
**Status**: Phase 2 Complete, Ready for Phase 3

---

## Executive Summary

MarunochiAI has completed outstanding Phase 2 work with full code understanding capabilities. BenchAI has now implemented the infrastructure to support bidirectional integration. This document outlines the next steps for complete multi-agent collaboration.

---

## Current State (December 26, 2025)

### MarunochiAI Capabilities (Confirmed)
| Feature | Status | Performance |
|---------|--------|-------------|
| Tree-sitter Parser | Complete | Python, JS, TS |
| Hierarchical Chunker | Complete | File→Class→Method |
| ChromaDB HNSW Indexer | Complete | M=32, ef=200 |
| SQLite FTS5 Keyword | Complete | BM25 |
| Hybrid RRF Search | Complete | 42% NDCG improvement |
| Filesystem Watcher | Complete | <100ms updates |
| Neovim Plugin | Complete | 1,100 LOC |
| VSCode Extension | Complete | 800 LOC |

### BenchAI Capabilities (Ready)
| Feature | Status | Endpoint |
|---------|--------|----------|
| Semantic Task Router | Complete | `/v1/learning/a2a/route` |
| A2A Task Submission | Complete | `/v1/learning/a2a/task` |
| Context Enrichment | Complete | Auto-attaches knowledge |
| OpenTelemetry Tracing | Complete | W3C TraceContext |
| Collective Learning | Complete | `/v1/learning/collective/*` |
| Bidirectional Sync | Complete | `/v1/learning/sync/*` |
| Zettelkasten | Complete | `/v1/learning/zettelkasten/*` |
| Memory System | Complete | `/v1/memory/*` |

---

## Phase 3: Integration Tasks for MarunochiAI

### 1. Implement Agent Card Endpoint (Priority: HIGH)

Add `/.well-known/agent.json` to MarunochiAI server for A2A discovery:

```python
# In marunochithe/api/server.py

@app.get("/.well-known/agent.json")
async def agent_card():
    """A2A Agent Card for discovery."""
    return {
        "name": "MarunochiAI",
        "version": "0.2.0",
        "description": "The most powerful self-hosted coding assistant",
        "capabilities": [
            "code_search",
            "code_completion",
            "code_refactoring",
            "code_debugging",
            "test_generation",
            "code_explanation",
            "hybrid_search",
            "codebase_indexing"
        ],
        "domains": ["coding"],
        "priority": 0.95,
        "endpoints": {
            "health": "http://localhost:8765/health",
            "chat": "http://localhost:8765/v1/chat/completions",
            "search": "http://localhost:8765/v1/codebase/search",
            "index": "http://localhost:8765/v1/codebase/index",
            "stats": "http://localhost:8765/v1/codebase/stats",
            "sync_receive": "http://localhost:8765/v1/sync/receive",
            "sync_share": "http://localhost:8765/v1/sync/share"
        },
        "status": "online",
        "load": 0.0
    }
```

### 2. Implement Sync Endpoints (Priority: HIGH)

Add sync endpoints for bidirectional memory sharing:

```python
# Add to server.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SyncRequest(BaseModel):
    from_agent: str
    sync_type: str  # experience, knowledge, pattern
    items: List[Dict[str, Any]]
    timestamp: Optional[str] = None


@app.post("/v1/sync/receive")
async def receive_sync_data(request: SyncRequest):
    """
    Receive sync data from BenchAI.

    Store received experiences/knowledge locally.
    """
    items_processed = 0

    for item in request.items:
        try:
            # Store in your local memory/database
            # Example: Save to SQLite or local storage
            if request.sync_type == "experience":
                # Store coding experience
                logger.info(f"Received experience from {request.from_agent}: {item.get('content', '')[:100]}...")
                # TODO: Add to local experience storage
            elif request.sync_type == "knowledge":
                # Store knowledge note
                logger.info(f"Received knowledge from {request.from_agent}: {item.get('title', '')}")
                # TODO: Add to local knowledge storage
            items_processed += 1
        except Exception as e:
            logger.error(f"Failed to process sync item: {e}")
            continue

    return {
        "status": "ok",
        "from_agent": request.from_agent,
        "items_processed": items_processed,
        "sync_type": request.sync_type
    }


@app.get("/v1/sync/share")
async def share_sync_data(
    requester: str,
    sync_type: str = "experience",
    since: Optional[str] = None,
    limit: int = 50
):
    """
    Share data with BenchAI.

    Returns local experiences/knowledge for sync.
    """
    items = []

    try:
        if sync_type == "experience":
            # Get recent successful coding experiences
            # Example: Last N successful refactorings, bug fixes, etc.
            items = [
                {
                    "id": "exp-001",
                    "content": "Successfully refactored authentication module using hybrid search",
                    "importance": 4,
                    "category": "refactoring",
                    "created_at": "2025-12-26T12:00:00Z"
                }
                # TODO: Pull from actual experience storage
            ]
        elif sync_type == "knowledge":
            # Get coding patterns and best practices learned
            items = [
                {
                    "id": "know-001",
                    "title": "Hybrid Search Best Practices",
                    "content": "RRF with k=60 provides optimal balance between vector and keyword search",
                    "tags": ["search", "rag", "optimization"]
                }
                # TODO: Pull from actual knowledge storage
            ]
    except Exception as e:
        logger.error(f"Failed to get sync data: {e}")

    return {
        "status": "ok",
        "for_agent": requester,
        "sync_type": sync_type,
        "items": items,
        "count": len(items)
    }
```

### 3. Report Task Completions to BenchAI (Priority: MEDIUM)

After completing coding tasks, report success/failure to BenchAI:

```python
import aiohttp

async def report_task_completion(
    task_type: str,
    success: bool,
    metrics: Dict[str, Any]
):
    """Report task completion to BenchAI for learning."""
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(
                "http://localhost:8085/v1/learning/collective/contribute",
                json={
                    "agent_id": "marunochiAI",
                    "contribution_type": "experience",
                    "content": f"Task: {task_type}, Success: {success}, Metrics: {metrics}",
                    "domain": "coding",
                    "quality_score": 0.9 if success else 0.3,
                    "metadata": {
                        "task_type": task_type,
                        "success": success,
                        **metrics
                    }
                },
                timeout=aiohttp.ClientTimeout(total=5)
            )
    except Exception:
        pass  # Don't fail task if reporting fails
```

### 4. Accept Tasks from BenchAI (Priority: MEDIUM)

Add endpoint to receive delegated tasks:

```python
class A2ATaskRequest(BaseModel):
    from_agent: str
    task_type: str
    task_description: str
    context: Optional[Dict[str, Any]] = None
    priority: str = "normal"
    callback_url: Optional[str] = None


@app.post("/v1/a2a/task")
async def receive_task(request: A2ATaskRequest):
    """
    Receive a task from BenchAI.

    Process coding tasks delegated by the orchestrator.
    """
    task_id = f"maru-{uuid.uuid4().hex[:12]}"

    # Log task receipt
    logger.info(f"Received task from {request.from_agent}: {request.task_description[:100]}...")

    # Extract context from enriched A2A context
    knowledge_context = ""
    if request.context:
        knowledge = request.context.get("knowledge", {})
        embedded = knowledge.get("embedded_knowledge", [])
        for item in embedded:
            knowledge_context += f"\n{item.get('content', '')}"

    # Process based on task type
    if request.task_type == "code_search":
        # Use hybrid searcher
        results = await hybrid_searcher.search(
            query=request.task_description,
            mode="hybrid",
            limit=10
        )
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "query": request.task_description,
                "results": [r.dict() for r in results],
                "count": len(results)
            }
        }

    elif request.task_type in ["code_completion", "code_review", "debugging"]:
        # Use chat completion with context
        messages = [
            {"role": "system", "content": f"You are a coding expert. Context: {knowledge_context}"},
            {"role": "user", "content": request.task_description}
        ]
        response = await engine.chat(messages=messages, stream=False)
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {"response": response}
        }

    else:
        return {
            "task_id": task_id,
            "status": "pending",
            "message": f"Task queued for processing"
        }
```

---

## Phase 4: Advanced Integration (Future)

### 1. Shared Knowledge Graph
- Sync Zettelkasten notes between agents
- Link code patterns to project context
- Cross-reference documentation with implementations

### 2. Collaborative Coding
- Multi-agent code review
- MarunochiAI writes → DottscavisAI diagrams → BenchAI documents
- Parallel task execution

### 3. Continuous Learning
- LoRA fine-tuning from successful interactions
- Cross-agent pattern learning
- User preference propagation

---

## API Quick Reference

### BenchAI Endpoints for MarunochiAI to Call

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `http://localhost:8085/health` | GET | Health check |
| `http://localhost:8085/v1/learning/collective/contribute` | POST | Report task completion |
| `http://localhost:8085/v1/learning/sync/share` | GET | Pull experiences |
| `http://localhost:8085/v1/learning/sync/receive` | POST | Push experiences |
| `http://localhost:8085/v1/learning/a2a/route` | POST | Test task routing |
| `http://localhost:8085/v1/memory/store` | POST | Store memory |
| `http://localhost:8085/v1/learning/zettelkasten/create` | POST | Create note |

### Request Examples

**Report Task Completion:**
```bash
curl -X POST http://localhost:8085/v1/learning/collective/contribute \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "marunochiAI",
    "contribution_type": "experience",
    "content": "Successfully refactored user authentication using hybrid search to find all related files",
    "domain": "coding",
    "quality_score": 0.95,
    "metadata": {
      "task_type": "refactoring",
      "files_modified": 5,
      "search_time_ms": 180
    }
  }'
```

**Sync with BenchAI:**
```bash
# Pull experiences from BenchAI
curl "http://localhost:8085/v1/learning/sync/share?requester=marunochiAI&sync_type=experience&limit=20"

# Push experiences to BenchAI
curl -X POST http://localhost:8085/v1/learning/sync/receive \
  -H "Content-Type: application/json" \
  -d '{
    "from_agent": "marunochiAI",
    "sync_type": "experience",
    "items": [
      {
        "id": "exp-001",
        "content": "Hybrid search with RRF k=60 outperforms pure vector search by 42%",
        "importance": 5
      }
    ]
  }'
```

---

## Testing Checklist

- [ ] Agent Card returns valid JSON at `/.well-known/agent.json`
- [ ] `/v1/sync/receive` accepts and stores BenchAI data
- [ ] `/v1/sync/share` returns local experiences
- [ ] `/v1/a2a/task` processes delegated coding tasks
- [ ] Task completions report to BenchAI collective learning
- [ ] Health endpoint returns status for routing decisions

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BenchAI (Orchestrator)                      │
│                        Port 8085                                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Semantic    │  │  Context     │  │  Collective  │           │
│  │  Router      │──│  Enrichment  │──│  Learning    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                                    │                   │
│         │              ┌─────────────────────┘                   │
│         │              │                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Zettelkasten│  │  Memory      │  │  Experience  │           │
│  │  Knowledge   │  │  System      │  │  Replay      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                                                        │
│         └──────────────┬─────────────────────────────────────────┤
│                        │ Bidirectional Sync                      │
└────────────────────────┼────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ MarunochiAI │  │DottscavisAI │  │  Future     │
│  Port 8765  │  │  Port 8766  │  │  Agents     │
│             │  │             │  │             │
│ Code Expert │  │  Creative   │  │             │
│ - Search    │  │  - Images   │  │             │
│ - Complete  │  │  - Video    │  │             │
│ - Debug     │  │  - Audio    │  │             │
│ - Test      │  │  - 3D       │  │             │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Timeline

| Phase | Status | Target |
|-------|--------|--------|
| Phase 1: MVP | Complete | - |
| Phase 2: Code Understanding | Complete | - |
| Phase 3: Integration | **In Progress** | Week 3 |
| Phase 4: Advanced | Planned | Weeks 4-6 |

---

## Contact

- **BenchAI API**: http://localhost:8085
- **GitHub**: https://github.com/CesarSalcido06/benchai

---

*Document generated by BenchAI - December 26, 2025*
