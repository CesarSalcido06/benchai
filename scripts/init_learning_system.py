#!/usr/bin/env python3
"""
Initialize the BenchAI Learning System and save the architecture to memory.
Run this once to bootstrap the learning system with the core architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add router to path
sys.path.insert(0, str(Path(__file__).parent.parent / "router"))

from learning.integration import create_learning_system
from learning.memory_enhanced import MemoryType
from learning.experience_replay import TaskDomain


ARCHITECTURE_MEMORY = """BENCHAI LEARNING ARCHITECTURE v1.0

Three-Layer Self-Improving System:

LAYER 1 - INSTANT MEMORY (RAG):
- Episodic Memory: Events/interactions - "User asked about X on date Y"
- Semantic Memory: Facts/knowledge - "MarunochiAI runs on M4 Pro"
- Procedural Memory: How-to guides - "To deploy, run docker-compose up"
- Agent Context: Cross-agent state - "DottscavisAI is rendering"
- Storage: ChromaDB + SQLite with FTS5 full-text search

LAYER 2 - EXPERIENCE REPLAY (Behavioral):
- Success Library: What worked (trajectories for in-context learning)
- Failure Repair Library: Lessons learned from mistakes
- Curious Replay: Prioritize novel/interesting experiences
- In-context example injection for 15-20% performance gain without fine-tuning

LAYER 3 - PERIODIC FINE-TUNING (LoRA):
- Base Model: Frozen to prevent catastrophic forgetting
- LoRA Adapters: Specialized for research, orchestration, coding
- Unsloth + QLoRA: 2.5x faster training, 90% less VRAM
- Schedule: Weekly collect data, Monthly train, Quarterly merge

RAFT HYBRID APPROACH:
Combines RAG + Experience Replay + Fine-tuned weights for optimal performance.

CATASTROPHIC FORGETTING PREVENTION:
- Replay Buffer: Mix 20% old data with new during training
- LoRA Isolation: Keep base model frozen, only train adapters
- Progressive Adapters: Stack adapters, don't replace

MULTI-AGENT ORCHESTRATION:
- BenchAI (Server): Mastermind brain, always-on, learning hub
- MarunochiAI (M4 Pro 24GB): Programmer agent, fine-tuned for code
- DottscavisAI (M1 Pro 32GB): Creative agent, media/3D/video
- Communication: A2A Protocol, gRPC for internal, REST for external"""


async def main():
    print("=" * 60)
    print("BenchAI Learning System Initialization")
    print("=" * 60)

    # Use production storage
    storage_dir = Path.home() / "llm-storage"
    print(f"\nStorage directory: {storage_dir}")

    system = create_learning_system(storage_dir)
    await system.initialize()
    print("[OK] Learning system initialized")

    # Save architecture to memory
    print("\nSaving architecture to memory...")

    arch_id = await system.remember(
        content=ARCHITECTURE_MEMORY,
        memory_type=MemoryType.ARCHITECTURE,
        category="system",
        importance=5,
        source="initialization"
    )
    print(f"[OK] Architecture saved (Memory ID: {arch_id})")

    # Save procedural memory for learning system usage
    await system.remember(
        content="To initialize the learning system: from learning_integration import setup_learning_system; await setup_learning_system()",
        memory_type=MemoryType.PROCEDURAL,
        category="learning_system",
        importance=4,
        source="initialization"
    )

    await system.remember(
        content="To add the learning API endpoints: app.include_router(learning_router)",
        memory_type=MemoryType.PROCEDURAL,
        category="learning_system",
        importance=4,
        source="initialization"
    )

    await system.remember(
        content="To log chat interactions: await log_chat_completion(request, response, model, tokens_in, tokens_out, duration_ms)",
        memory_type=MemoryType.PROCEDURAL,
        category="learning_system",
        importance=4,
        source="initialization"
    )

    await system.remember(
        content="To get in-context examples from past successes: context = await get_experience_context(task_description)",
        memory_type=MemoryType.PROCEDURAL,
        category="learning_system",
        importance=4,
        source="initialization"
    )

    print("[OK] Procedural memories saved")

    # Register agents
    print("\nRegistering multi-agent system...")

    await system.register_agent(
        agent_id="benchai",
        name="BenchAI",
        role="orchestrator",
        capabilities=["research", "planning", "orchestration", "memory", "rag", "tts"],
        endpoint="http://localhost:8085"
    )
    print("[OK] BenchAI registered")

    await system.register_agent(
        agent_id="marunochiAI",
        name="MarunochiAI",
        role="programmer",
        capabilities=["coding", "debugging", "testing", "code_review", "refactoring"],
        endpoint=None  # Will be set when connected
    )
    await system.memory.update_agent_status("marunochiAI", "offline")
    print("[OK] MarunochiAI registered (offline)")

    await system.register_agent(
        agent_id="dottscavisAI",
        name="DottscavisAI",
        role="creative",
        capabilities=["image_generation", "video_editing", "3d_modeling", "audio_processing"],
        endpoint=None
    )
    await system.memory.update_agent_status("dottscavisAI", "offline")
    print("[OK] DottscavisAI registered (offline)")

    # Record initial experience
    await system.record_success(
        task="Initialize BenchAI learning system",
        approach="Created three-layer learning architecture with enhanced memory, experience replay, and LoRA fine-tuning pipeline",
        trajectory=[
            {"action": "design", "result": "Designed three-layer architecture based on 2025 research"},
            {"action": "implement", "result": "Created memory_enhanced.py, experience_replay.py, interaction_logger.py, learning_pipeline.py"},
            {"action": "integrate", "result": "Created integration.py and api.py for FastAPI integration"},
            {"action": "test", "result": "All tests passed"}
        ],
        domain=TaskDomain.SYSTEM,
        score=0.95
    )
    print("[OK] Initial experience recorded")

    # Print stats
    print("\n" + "=" * 60)
    print("System Statistics")
    print("=" * 60)

    stats = await system.get_stats()

    print(f"\nMemory System:")
    print(f"  Total memories: {stats['memory']['total_memories']}")
    for mtype, info in stats['memory'].get('memory_types', {}).items():
        print(f"  - {mtype}: {info.get('count', 0)} (avg importance: {info.get('avg_importance', 'N/A')})")

    print(f"\nExperience Replay:")
    print(f"  Total experiences: {stats['experiences']['total_experiences']}")

    print(f"\nRegistered Agents: {stats['memory'].get('registered_agents', 0)}")

    agents = await system.get_agents()
    for agent in agents:
        print(f"  - {agent['name']} ({agent['role']}): {agent['status']}")

    print("\n" + "=" * 60)
    print("INITIALIZATION COMPLETE")
    print("=" * 60)
    print("\nThe learning system is ready. To integrate into the router:")
    print("  1. Add to lifespan: await setup_learning_system()")
    print("  2. Add router: app.include_router(learning_router)")
    print("  3. API endpoints available at: /v1/learning/*")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
