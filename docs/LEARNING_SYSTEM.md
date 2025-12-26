# BenchAI Learning System v2.0

A self-improving AI architecture with Zettelkasten-inspired knowledge graph, experience replay, and LoRA fine-tuning capabilities.

## Architecture Overview

BenchAI implements a **four-layer learning system** that enables continuous improvement through knowledge accumulation, experience replay, and periodic fine-tuning.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BENCHAI v2.0 ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 0: Zettelkasten Knowledge Graph                              │
│  ├── Atomic notes with emergent linking                             │
│  ├── Sleep consolidation (strengthen/weaken/compress)               │
│  └── Async Research API for parallel agent queries                  │
│                                                                      │
│  LAYER 1: Enhanced Memory System                                    │
│  ├── Typed memories (episodic, semantic, procedural, agent)        │
│  ├── Importance scoring with decay                                  │
│  └── Cross-agent memory sharing                                     │
│                                                                      │
│  LAYER 2: Experience Replay                                         │
│  ├── Success/failure trajectory tracking                            │
│  ├── Curious replay (prioritize novel experiences)                  │
│  └── In-context example injection (15-20% performance gain)         │
│                                                                      │
│  LAYER 3: LoRA Fine-Tuning Pipeline                                 │
│  ├── Unsloth for 2.5x faster training                               │
│  ├── Multiple specialized adapters                                  │
│  └── Catastrophic forgetting prevention                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Research Foundation

This system is built on cutting-edge 2025 research:

| Research | Source | Key Contribution |
|----------|--------|------------------|
| A-MEM | NeurIPS 2025 | Zettelkasten for AI agents |
| LOIRE | ICLR 2025 | Lifelong learning framework |
| Curious Replay | Stanford HAI | Prioritizing novel experiences |
| Graphiti | Zep/Neo4j | Temporal knowledge graphs |
| Unsloth | NVIDIA | 2.5x faster LoRA training |

## Components

### Layer 0: Zettelkasten Knowledge Graph

**File:** `router/learning/zettelkasten.py` (778 lines)

The Zettelkasten layer implements a self-organizing knowledge graph based on the slip-box method developed by Niklas Luhmann and adapted for AI agents by A-MEM (NeurIPS 2025).

#### Principles
- **Atomicity**: Each note contains exactly one idea
- **Unique IDs**: Permanent identifiers for each note
- **Emergent Linking**: Connections form organically via semantic similarity
- **Multi-box Membership**: Notes can belong to multiple conceptual clusters

#### Note Types
| Type | Purpose |
|------|---------|
| `fleeting` | Quick captures, to be processed |
| `literature` | From external sources (research, docs) |
| `permanent` | Refined, atomic knowledge |
| `hub` | Index/entry points connecting many notes |
| `structure` | Organizational notes |

#### Link Types
- `relates_to`, `supports`, `contradicts`, `extends`
- `example_of`, `caused_by`, `part_of`, `sequence`

#### Sleep Consolidation
Inspired by how the brain consolidates memories during sleep:
1. Strengthen frequently co-accessed links
2. Weaken rarely used connections
3. Consolidate old fleeting notes into summaries
4. Update importance scores based on access patterns
5. Apply importance decay to old, unaccessed notes

### Layer 1: Enhanced Memory System

**File:** `router/learning/memory_enhanced.py` (625 lines)

Typed, categorized memory storage with importance scoring and decay.

#### Memory Types
| Type | Description | Example |
|------|-------------|---------|
| `episodic` | Events/interactions | "User asked about X on date Y" |
| `semantic` | Facts/knowledge | "MarunochiAI runs on M4 Pro" |
| `procedural` | How-to guides | "To deploy, run docker-compose up" |
| `agent` | Cross-agent state | "DottscavisAI is rendering" |
| `experience` | Learning trajectories | Success/failure records |
| `architecture` | System design | Architecture decisions |

#### Features
- FTS5 full-text search
- Importance scoring (1-5) with automatic decay
- Memory consolidation (summarize old memories)
- Deduplication via content hashing
- Access tracking for relevance scoring

### Layer 2: Experience Replay

**File:** `router/learning/experience_replay.py` (657 lines)

Based on Stanford's "Curious Replay" and NeurIPS 2025 research on self-generated in-context examples.

#### Components
- **Success Library**: Trajectories that worked
- **Failure Repair Library**: Lessons learned from mistakes
- **Training Queue**: High-quality examples for fine-tuning

#### Performance
- In-context example injection provides 15-20% performance gain
- No fine-tuning required for this improvement
- Stanford research: ALFWorld performance 73% → 89% → 93%

### Layer 3: LoRA Fine-Tuning Pipeline

**File:** `router/learning/learning_pipeline.py` (837 lines)

Periodic model improvement using Unsloth for efficient LoRA training.

#### Adapter Types
| Adapter | Specialization |
|---------|---------------|
| `research` | Deep research and analysis |
| `orchestration` | Multi-agent coordination |
| `coding` | Code understanding and generation |
| `general` | General conversation |

#### Training Schedule
- **Weekly**: Collect interaction data, filter quality examples
- **Monthly**: Train LoRA adapter on accumulated data
- **Quarterly**: Merge best adapters, evaluate, create new baseline

#### Catastrophic Forgetting Prevention
- Replay Buffer: Mix 20% old data with new
- LoRA Isolation: Keep base model frozen
- Progressive Adapters: Stack adapters, don't replace

## Async Research API

**File:** `router/learning/research_api.py` (352 lines)

Allows agents to query BenchAI's knowledge while working in parallel.

```python
# Agent submits query
query_id = await research_api.submit_query(
    query="How does Zettelkasten linking work?",
    agent_id="marunochiAI",
    priority=QueryPriority.NORMAL
)

# Agent continues working...

# Later, retrieve results
result = await research_api.get_result(query_id, wait=True)
```

### Priority Levels
| Priority | Use Case |
|----------|----------|
| `critical` | Blocking agent work |
| `high` | Needed soon |
| `normal` | Background research |
| `low` | Nice to have |

## Multi-Agent Integration

### Registered Agents
| Agent | Hardware | Role |
|-------|----------|------|
| BenchAI | Linux Server | Orchestrator, knowledge repository |
| MarunochiAI | M4 Pro 24GB | Programmer, code specialist |
| DottscavisAI | M1 Pro 32GB | Creative, media/3D/video |

### Communication
- A2A Protocol (JSON-RPC/gRPC/REST hybrid)
- REST for external/simple queries
- gRPC for high-speed internal calls
- Message queue for async long-running tasks

## API Endpoints

All endpoints are prefixed with `/v1/learning/`

### Memory
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memory/store` | POST | Store typed memory |
| `/memory/search` | GET | Search with filtering |
| `/memory/by-type/{type}` | GET | Get by memory type |
| `/memory/stats` | GET | Memory statistics |
| `/memory/consolidate` | POST | Trigger consolidation |

### Experience
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/experience/record` | POST | Record success/failure |
| `/experience/similar` | GET | Find similar experiences |
| `/experience/examples` | GET | Get in-context examples |
| `/experience/curious` | GET | Curious replay examples |

### Agents
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agents/register` | POST | Register new agent |
| `/agents` | GET | List agents |
| `/agents/{id}/status` | PUT | Update status |
| `/agents/context` | POST/GET | Share/get context |

### Training
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/training/trigger` | POST | Start training run |
| `/training/status/{type}` | GET | Check if should train |
| `/training/adapters` | GET | List active adapters |
| `/training/runs` | GET | Training history |

## Installation

```bash
# Install dependencies
./scripts/setup_learning.sh

# Initialize the system
python3 scripts/init_learning_system.py

# Run tests
python3 scripts/test_learning.py
python3 scripts/test_zettelkasten.py
```

## Integration

Add to your FastAPI router:

```python
from learning_integration import setup_learning_system, learning_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_learning_system()
    yield

app.include_router(learning_router)
```

## Storage Structure

```
~/llm-storage/
├── memory/          # Enhanced memory database
├── learning/        # Experience replay, interactions, pipeline
├── zettelkasten/    # Knowledge graph database
├── adapters/        # Trained LoRA adapters
├── models/          # Base models
├── cache/           # Audio, query caches
└── rag/             # ChromaDB vector store
```

## Statistics (Current)

- **Enhanced Memory**: 20 memories across 6 types
- **Zettelkasten**: 14 zettels with 45 auto-generated links
- **Graph Density**: 0.4 (well-connected knowledge)
- **Registered Agents**: 3 (BenchAI, MarunochiAI, DottscavisAI)

## Research Sources

1. [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) - NeurIPS 2025
2. [LOIRE: Lifelong Learning Framework](https://openreview.net/pdf?id=F5PlYMC5ik) - ICLR 2025
3. [Curious Replay](https://hai.stanford.edu/news/ai-agents-self-reflect-perform-better-changing-environments) - Stanford HAI
4. [Graphiti Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/) - Neo4j
5. [Unsloth Fine-Tuning](https://github.com/unslothai/unsloth) - NVIDIA Partnership
6. [ACM Computing Surveys: Continual Learning of LLMs](https://dl.acm.org/doi/10.1145/3716629) - 2025
7. [Zep: Temporal Knowledge Graph](https://arxiv.org/abs/2501.13956) - 2025

---

*Generated with Claude Code on December 26, 2025*
