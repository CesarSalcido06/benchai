"""
Learning System Integration for Main Router

This module provides easy integration of the learning system into the main BenchAI router.
Import and call setup_learning_system() in your lifespan context.

Usage in llm_router.py:

    from learning_integration import setup_learning_system, learning_system, learning_router

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ... existing startup code ...
        await setup_learning_system()
        yield
        # ... existing shutdown code ...

    # Add the learning router
    app.include_router(learning_router)
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import APIRouter

# Import learning system components
from learning.integration import LearningSystem, create_learning_system
from learning.memory_enhanced import MemoryType
from learning.experience_replay import TaskDomain, ExperienceOutcome
from learning.api import router as learning_api_router, set_learning_system

# Global learning system instance
learning_system: Optional[LearningSystem] = None

# Re-export the API router
learning_router = learning_api_router


async def setup_learning_system(storage_dir: Optional[Path] = None) -> LearningSystem:
    """
    Initialize the learning system for use with the main router.

    Call this during FastAPI lifespan startup.

    Args:
        storage_dir: Custom storage directory. Defaults to ~/llm-storage

    Returns:
        The initialized LearningSystem instance
    """
    global learning_system

    if storage_dir is None:
        storage_dir = Path.home() / "llm-storage"

    learning_system = create_learning_system(storage_dir)
    await learning_system.initialize()

    # Set the global instance for the API router
    set_learning_system(learning_system)

    # Start maintenance loop (runs every 24 hours)
    await learning_system.start_maintenance_loop(interval_hours=24)

    print("[ROUTER] Learning system integrated successfully")
    return learning_system


async def shutdown_learning_system():
    """Shutdown the learning system gracefully."""
    global learning_system

    if learning_system:
        await learning_system.stop_maintenance_loop()
        print("[ROUTER] Learning system shutdown complete")


def get_learning_system() -> Optional[LearningSystem]:
    """Get the current learning system instance."""
    return learning_system


# =========================================================================
# Helper Functions for Chat Integration
# =========================================================================

async def log_chat_completion(
    request: Dict[str, Any],
    response: Dict[str, Any],
    model: str,
    tokens_in: int,
    tokens_out: int,
    duration_ms: int,
    success: bool = True,
    session_id: Optional[str] = None
) -> Optional[int]:
    """
    Log a chat completion for learning.
    Call this after processing each chat request.
    """
    if not learning_system:
        return None

    return await learning_system.log_chat(
        request=request,
        response=response,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=duration_ms,
        success=success,
        session_id=session_id
    )


async def get_experience_context(task: str, domain: str = "general") -> str:
    """
    Get relevant experience examples for in-context learning.
    Call this before processing complex tasks to inject past successes.
    """
    if not learning_system:
        return ""

    try:
        domain_enum = TaskDomain(domain)
    except ValueError:
        domain_enum = TaskDomain.GENERAL

    return await learning_system.get_relevant_examples(
        task=task,
        domain=domain_enum,
        limit=3
    )


async def remember_important(
    content: str,
    category: str = "general",
    importance: int = 4
) -> Optional[int]:
    """
    Store an important memory for future reference.
    Use this for significant facts, decisions, or learnings.
    """
    if not learning_system:
        return None

    return await learning_system.remember(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        category=category,
        importance=importance,
        source="benchai"
    )


async def recall_relevant(query: str, limit: int = 5) -> List[Dict]:
    """
    Recall relevant memories for a query.
    Use this to augment responses with learned knowledge.
    """
    if not learning_system:
        return []

    return await learning_system.recall(query=query, limit=limit)


async def record_task_outcome(
    task: str,
    approach: str,
    success: bool,
    details: str = "",
    domain: str = "general"
) -> Optional[int]:
    """
    Record the outcome of a task for learning.
    Call this after completing significant tasks.
    """
    if not learning_system:
        return None

    try:
        domain_enum = TaskDomain(domain)
    except ValueError:
        domain_enum = TaskDomain.GENERAL

    if success:
        return await learning_system.record_success(
            task=task,
            approach=approach,
            trajectory=[{"action": "complete", "result": details}],
            domain=domain_enum,
            score=0.8
        )
    else:
        return await learning_system.record_failure(
            task=task,
            approach=approach,
            trajectory=[],
            failure_reason=details,
            domain=domain_enum
        )


# =========================================================================
# Multi-Agent Helpers
# =========================================================================

async def notify_agent_online(agent_id: str, name: str, role: str, capabilities: List[str], endpoint: str):
    """Register/update an agent as online."""
    if not learning_system:
        return

    await learning_system.register_agent(
        agent_id=agent_id,
        name=name,
        role=role,
        capabilities=capabilities,
        endpoint=endpoint
    )


async def notify_agent_offline(agent_id: str):
    """Mark an agent as offline."""
    if not learning_system:
        return

    await learning_system.update_agent_status(agent_id, "offline")


async def share_with_agents(content: str, category: str = "shared"):
    """Share context with all agents."""
    if not learning_system:
        return None

    return await learning_system.share_context(
        content=content,
        from_agent="benchai",
        category=category
    )


async def get_agent_shared_context() -> List[Dict]:
    """Get all shared context from agents."""
    if not learning_system:
        return []

    return await learning_system.get_shared_context()


# =========================================================================
# Example Integration Points
# =========================================================================

INTEGRATION_EXAMPLE = '''
# Example: Integrating learning into chat completions

async def chat_completions_with_learning(request, ...):
    import time
    start_time = time.time()

    # 1. Get relevant experiences for complex tasks
    user_message = request.messages[-1].content if request.messages else ""
    experience_context = await get_experience_context(user_message)

    # 2. Recall relevant memories
    memories = await recall_relevant(user_message, limit=3)
    memory_context = "\\n".join([m["content"] for m in memories])

    # 3. Inject into system prompt (if valuable)
    if experience_context or memory_context:
        enhanced_system = f"""
{original_system_prompt}

## Relevant Past Experiences:
{experience_context}

## Relevant Knowledge:
{memory_context}
"""

    # 4. Process the request normally
    response = await process_chat(request)

    # 5. Log the interaction
    duration_ms = int((time.time() - start_time) * 1000)
    await log_chat_completion(
        request=request.dict(),
        response=response,
        model=request.model,
        tokens_in=...,
        tokens_out=...,
        duration_ms=duration_ms
    )

    # 6. If task completed successfully, record it
    if is_significant_task(user_message):
        await record_task_outcome(
            task=user_message[:200],
            approach=response["choices"][0]["message"]["content"][:500],
            success=True,
            domain="general"
        )

    return response
'''

print(__doc__) if __name__ == "__main__" else None
