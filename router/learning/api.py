"""
Learning System API Endpoints

FastAPI router with endpoints for the learning system.
Import and include this in the main router.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path

from .integration import LearningSystem, create_learning_system
from .memory_enhanced import MemoryType
from .experience_replay import TaskDomain, ExperienceOutcome
from .interaction_logger import InteractionType
from .learning_pipeline import AdapterType

# Create the router
router = APIRouter(prefix="/v1/learning", tags=["learning"])

# Global learning system instance (will be set by main router)
_learning_system: Optional[LearningSystem] = None


def get_learning_system() -> LearningSystem:
    """Get or create the learning system instance."""
    global _learning_system
    if _learning_system is None:
        _learning_system = create_learning_system()
    return _learning_system


def set_learning_system(system: LearningSystem):
    """Set the learning system instance (called by main router)."""
    global _learning_system
    _learning_system = system


# =========================================================================
# Request/Response Models
# =========================================================================

class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: str = "semantic"
    category: str = "general"
    importance: int = Field(default=3, ge=1, le=5)
    source: str = "user"
    metadata: Optional[Dict[str, Any]] = None


class MemorySearchRequest(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    category: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class ExperienceRecordRequest(BaseModel):
    task: str
    approach: str
    trajectory: List[Dict[str, Any]] = []
    domain: str = "general"
    outcome: str = "success"
    outcome_details: str = ""
    success_score: float = Field(default=0.8, ge=0, le=1)
    agent: str = "benchai"


class AgentRegisterRequest(BaseModel):
    agent_id: str
    name: str
    role: str
    capabilities: List[str]
    endpoint: Optional[str] = None


class AgentStatusRequest(BaseModel):
    status: str  # online, offline, busy


class ContextShareRequest(BaseModel):
    content: str
    from_agent: str
    category: str = "shared"


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: int = Field(ge=1, le=5)


class TrainingTriggerRequest(BaseModel):
    adapter_type: str  # research, orchestration, coding, general


# =========================================================================
# Memory Endpoints
# =========================================================================

@router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    """Store a new memory with type categorization."""
    system = get_learning_system()
    await system.initialize()

    try:
        memory_type = MemoryType(request.memory_type)
    except ValueError:
        memory_type = MemoryType.SEMANTIC

    memory_id = await system.remember(
        content=request.content,
        memory_type=memory_type,
        category=request.category,
        importance=request.importance,
        source=request.source,
        metadata=request.metadata
    )

    return {"status": "ok", "memory_id": memory_id}


@router.get("/memory/search")
async def search_memory(
    q: str = Query(..., description="Search query"),
    types: Optional[str] = Query(None, description="Comma-separated memory types"),
    category: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50)
):
    """Search memories with optional type filtering."""
    system = get_learning_system()
    await system.initialize()

    memory_types = None
    if types:
        memory_types = [MemoryType(t.strip()) for t in types.split(",") if t.strip()]

    results = await system.recall(
        query=q,
        memory_types=memory_types,
        limit=limit
    )

    return {"results": results, "count": len(results)}


@router.get("/memory/by-type/{memory_type}")
async def get_memories_by_type(
    memory_type: str,
    limit: int = Query(20, ge=1, le=100)
):
    """Get memories of a specific type."""
    system = get_learning_system()
    await system.initialize()

    try:
        mtype = MemoryType(memory_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid memory type: {memory_type}")

    results = await system.memory.get_by_type(mtype, limit=limit)
    return {"results": results, "count": len(results)}


@router.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics."""
    system = get_learning_system()
    await system.initialize()
    return await system.memory.get_stats()


@router.post("/memory/consolidate")
async def consolidate_memories(
    days_old: int = Query(30, ge=7, le=365),
    background_tasks: BackgroundTasks = None
):
    """Consolidate old memories into summaries."""
    system = get_learning_system()
    await system.initialize()

    result = await system.memory.consolidate_old_memories(days_old=days_old)
    return result


# =========================================================================
# Experience Endpoints
# =========================================================================

@router.post("/experience/record")
async def record_experience(request: ExperienceRecordRequest):
    """Record an experience (success or failure) for learning."""
    system = get_learning_system()
    await system.initialize()

    try:
        domain = TaskDomain(request.domain)
    except ValueError:
        domain = TaskDomain.GENERAL

    try:
        outcome = ExperienceOutcome(request.outcome)
    except ValueError:
        outcome = ExperienceOutcome.SUCCESS if request.success_score > 0.5 else ExperienceOutcome.FAILURE

    exp_id = await system.experiences.record_experience(
        task_description=request.task,
        domain=domain,
        approach=request.approach,
        trajectory=request.trajectory,
        outcome=outcome,
        outcome_details=request.outcome_details,
        success_score=request.success_score,
        agent_source=request.agent
    )

    return {"status": "ok", "experience_id": exp_id}


@router.get("/experience/similar")
async def get_similar_experiences(
    task: str = Query(..., description="Task description to find similar experiences"),
    domain: Optional[str] = None,
    limit: int = Query(5, ge=1, le=20)
):
    """Find similar successful experiences for in-context learning."""
    system = get_learning_system()
    await system.initialize()

    domain_enum = None
    if domain:
        try:
            domain_enum = TaskDomain(domain)
        except ValueError:
            pass

    examples = await system.experiences.get_similar_successes(
        task_description=task,
        domain=domain_enum,
        limit=limit
    )

    return {"results": examples, "count": len(examples)}


@router.get("/experience/examples")
async def get_in_context_examples(
    task: str = Query(..., description="Task to get examples for"),
    domain: Optional[str] = None,
    limit: int = Query(3, ge=1, le=5)
):
    """Get formatted in-context examples for prompting."""
    system = get_learning_system()
    await system.initialize()

    domain_enum = None
    if domain:
        try:
            domain_enum = TaskDomain(domain)
        except ValueError:
            pass

    formatted = await system.get_relevant_examples(
        task=task,
        domain=domain_enum,
        limit=limit
    )

    return {"examples": formatted}


@router.get("/experience/curious")
async def get_curious_replay_examples(limit: int = Query(10, ge=1, le=50)):
    """Get high-novelty experiences for curious replay."""
    system = get_learning_system()
    await system.initialize()

    examples = await system.experiences.get_curious_replay_examples(limit=limit)
    return {"results": examples, "count": len(examples)}


@router.get("/experience/stats")
async def get_experience_stats():
    """Get experience replay statistics."""
    system = get_learning_system()
    await system.initialize()
    return await system.experiences.get_stats()


# =========================================================================
# Agent Coordination Endpoints
# =========================================================================

@router.post("/agents/register")
async def register_agent(request: AgentRegisterRequest):
    """Register an agent in the multi-agent system."""
    system = get_learning_system()
    await system.initialize()

    await system.register_agent(
        agent_id=request.agent_id,
        name=request.name,
        role=request.role,
        capabilities=request.capabilities,
        endpoint=request.endpoint
    )

    return {"status": "ok", "agent_id": request.agent_id}


@router.get("/agents")
async def list_agents(status: Optional[str] = None):
    """List registered agents."""
    system = get_learning_system()
    await system.initialize()

    agents = await system.get_agents(status=status)
    return {"agents": agents, "count": len(agents)}


@router.put("/agents/{agent_id}/status")
async def update_agent_status(agent_id: str, request: AgentStatusRequest):
    """Update an agent's status."""
    system = get_learning_system()
    await system.initialize()

    await system.update_agent_status(agent_id, request.status)
    return {"status": "ok", "agent_id": agent_id, "new_status": request.status}


@router.post("/agents/context")
async def share_agent_context(request: ContextShareRequest):
    """Share context between agents."""
    system = get_learning_system()
    await system.initialize()

    memory_id = await system.share_context(
        content=request.content,
        from_agent=request.from_agent,
        category=request.category
    )

    return {"status": "ok", "memory_id": memory_id}


@router.get("/agents/context")
async def get_agent_context(agent_id: Optional[str] = None):
    """Get shared context for agents."""
    system = get_learning_system()
    await system.initialize()

    context = await system.get_shared_context(agent_id=agent_id)
    return {"context": context, "count": len(context)}


# =========================================================================
# Interaction & Feedback Endpoints
# =========================================================================

@router.post("/feedback")
async def add_feedback(request: FeedbackRequest):
    """Add user feedback to an interaction."""
    system = get_learning_system()
    await system.initialize()

    await system.add_feedback(request.interaction_id, request.rating)
    return {"status": "ok"}


@router.get("/interactions/stats")
async def get_interaction_stats():
    """Get interaction statistics."""
    system = get_learning_system()
    await system.initialize()
    return await system.logger.get_stats()


@router.get("/interactions/trends")
async def get_performance_trends(days: int = Query(7, ge=1, le=30)):
    """Get performance trends over time."""
    system = get_learning_system()
    await system.initialize()
    return await system.logger.get_performance_trends(days=days)


@router.get("/interactions/failures")
async def get_recent_failures(hours: int = Query(24, ge=1, le=168)):
    """Get recent failed interactions for analysis."""
    system = get_learning_system()
    await system.initialize()

    failures = await system.logger.get_recent_failures(hours=hours)
    return {"failures": failures, "count": len(failures)}


# =========================================================================
# Training Endpoints
# =========================================================================

@router.post("/training/trigger")
async def trigger_training(request: TrainingTriggerRequest, background_tasks: BackgroundTasks):
    """Trigger a training run for an adapter type."""
    system = get_learning_system()
    await system.initialize()

    try:
        adapter_type = AdapterType(request.adapter_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid adapter type: {request.adapter_type}")

    # Check if training should proceed
    should_train, reason = await system.pipeline.should_trigger_training(adapter_type)
    if not should_train:
        return {"status": "skipped", "reason": reason}

    # Run training in background
    run_id = await system.pipeline.create_training_run(adapter_type)

    async def run_training():
        try:
            result = await system.pipeline.run_training(run_id)
            if result["status"] == "completed":
                await system.pipeline.activate_adapter(run_id)
        except Exception as e:
            print(f"[TRAINING] Error: {e}")

    background_tasks.add_task(run_training)

    return {"status": "started", "run_id": run_id, "adapter_type": adapter_type.value}


@router.get("/training/status/{adapter_type}")
async def check_training_status(adapter_type: str):
    """Check if training should be triggered for an adapter type."""
    system = get_learning_system()
    await system.initialize()

    try:
        atype = AdapterType(adapter_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid adapter type: {adapter_type}")

    return await system.check_training_status(atype)


@router.get("/training/adapters")
async def list_active_adapters():
    """List currently active adapters."""
    system = get_learning_system()
    await system.initialize()
    return await system.get_active_adapters()


@router.get("/training/runs")
async def list_training_runs(
    adapter_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """List training run history."""
    system = get_learning_system()
    await system.initialize()

    atype = None
    if adapter_type:
        try:
            atype = AdapterType(adapter_type)
        except ValueError:
            pass

    runs = await system.pipeline.get_training_runs(
        adapter_type=atype,
        limit=limit
    )

    return {"runs": runs, "count": len(runs)}


# =========================================================================
# System Endpoints
# =========================================================================

@router.get("/health")
async def learning_health():
    """Check health of the learning system."""
    system = get_learning_system()
    return await system.health_check()


@router.get("/stats")
async def learning_stats():
    """Get comprehensive learning system statistics."""
    system = get_learning_system()
    await system.initialize()
    return await system.get_stats()


@router.post("/maintenance")
async def run_maintenance(background_tasks: BackgroundTasks):
    """Trigger maintenance tasks (consolidation, decay, etc.)."""
    system = get_learning_system()
    await system.initialize()

    background_tasks.add_task(system.run_maintenance)
    return {"status": "started", "message": "Maintenance tasks running in background"}
