#!/usr/bin/env python3
"""
Test script for the BenchAI Learning System.
Run this to verify all components are working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add router to path
sys.path.insert(0, str(Path(__file__).parent.parent / "router"))

from learning.integration import create_learning_system
from learning.memory_enhanced import MemoryType
from learning.experience_replay import TaskDomain, ExperienceOutcome


async def test_memory_system(system):
    """Test the enhanced memory system."""
    print("\n=== Testing Enhanced Memory System ===")

    # Store different types of memories
    mem_id = await system.remember(
        content="BenchAI uses a three-layer learning architecture",
        memory_type=MemoryType.ARCHITECTURE,
        category="system",
        importance=5,
        source="test"
    )
    print(f"  [OK] Stored architecture memory (ID: {mem_id})")

    mem_id = await system.remember(
        content="User asked about multi-agent coordination on 2025-12-26",
        memory_type=MemoryType.EPISODIC,
        category="interaction",
        importance=3,
        source="test"
    )
    print(f"  [OK] Stored episodic memory (ID: {mem_id})")

    mem_id = await system.remember(
        content="To deploy the learning system, run setup_learning.sh first",
        memory_type=MemoryType.PROCEDURAL,
        category="deployment",
        importance=4,
        source="test"
    )
    print(f"  [OK] Stored procedural memory (ID: {mem_id})")

    # Search memories
    results = await system.recall("learning architecture", limit=5)
    print(f"  [OK] Search returned {len(results)} results")

    # Get stats
    stats = await system.memory.get_stats()
    print(f"  [OK] Memory stats: {stats['total_memories']} total memories")

    return True


async def test_experience_replay(system):
    """Test the experience replay system."""
    print("\n=== Testing Experience Replay System ===")

    # Record a successful experience
    exp_id = await system.record_success(
        task="Implement multi-agent communication",
        approach="Used gRPC for fast internal calls and REST for external API",
        trajectory=[
            {"action": "research", "result": "Found A2A protocol specification"},
            {"action": "implement", "result": "Created gRPC endpoints"},
            {"action": "test", "result": "All tests passed"}
        ],
        domain=TaskDomain.ORCHESTRATION,
        score=0.9
    )
    print(f"  [OK] Recorded success experience (ID: {exp_id})")

    # Record a failure
    fail_id = await system.record_failure(
        task="Optimize database queries",
        approach="Added indexes without analyzing query patterns",
        trajectory=[
            {"action": "add_indexes", "result": "Added 5 indexes"},
            {"action": "test", "result": "Performance worse"}
        ],
        failure_reason="Wrong columns indexed, should have used EXPLAIN first",
        domain=TaskDomain.CODING
    )
    print(f"  [OK] Recorded failure experience (ID: {fail_id})")

    # Get in-context examples
    examples = await system.get_relevant_examples(
        task="Implement agent communication",
        domain=TaskDomain.ORCHESTRATION
    )
    print(f"  [OK] Got in-context examples ({len(examples)} chars)")

    # Get curious replay examples
    curious = await system.experiences.get_curious_replay_examples(limit=5)
    print(f"  [OK] Curious replay returned {len(curious)} examples")

    # Get stats
    stats = await system.experiences.get_stats()
    print(f"  [OK] Experience stats: {stats['total_experiences']} total experiences")

    return True


async def test_interaction_logging(system):
    """Test the interaction logging system."""
    print("\n=== Testing Interaction Logging ===")

    # Start a session
    session_id = await system.logger.start_session("test-session", agent_source="test")
    print(f"  [OK] Started session: {session_id}")

    # Log a chat interaction
    int_id = await system.log_chat(
        request={"messages": [{"role": "user", "content": "Hello, how do I use the learning system?"}]},
        response={"choices": [{"message": {"content": "The learning system has three layers..."}}]},
        model="test-model",
        tokens_in=50,
        tokens_out=100,
        duration_ms=500,
        session_id=session_id
    )
    print(f"  [OK] Logged chat interaction (ID: {int_id})")

    # Log a tool use
    tool_id = await system.log_tool_use(
        tool_name="memory_search",
        tool_input={"query": "learning architecture"},
        tool_output={"results": []},
        duration_ms=50,
        session_id=session_id
    )
    print(f"  [OK] Logged tool use (ID: {tool_id})")

    # Add feedback
    await system.add_feedback(int_id, 5)
    print(f"  [OK] Added feedback to interaction")

    # End session
    await system.logger.end_session(session_id)
    print(f"  [OK] Ended session")

    # Get stats
    stats = await system.logger.get_stats()
    print(f"  [OK] Logger stats: {stats['total_interactions']} total interactions")

    return True


async def test_agent_coordination(system):
    """Test multi-agent coordination."""
    print("\n=== Testing Agent Coordination ===")

    # Register agents
    await system.register_agent(
        agent_id="marunochiAI",
        name="MarunochiAI",
        role="programmer",
        capabilities=["coding", "debugging", "testing"],
        endpoint="http://marunochi.local:8080"
    )
    print("  [OK] Registered MarunochiAI")

    await system.register_agent(
        agent_id="dottscavisAI",
        name="DottscavisAI",
        role="creative",
        capabilities=["image_gen", "video_edit", "3d_modeling"],
        endpoint="http://dottscavis.local:8080"
    )
    print("  [OK] Registered DottscavisAI")

    # List agents
    agents = await system.get_agents()
    print(f"  [OK] Listed {len(agents)} agents")

    # Share context
    ctx_id = await system.share_context(
        content="Working on multi-agent orchestration test",
        from_agent="benchai",
        category="test"
    )
    print(f"  [OK] Shared context (ID: {ctx_id})")

    # Get shared context
    context = await system.get_shared_context()
    print(f"  [OK] Retrieved {len(context)} shared context items")

    return True


async def test_training_pipeline(system):
    """Test the training pipeline (without actually training)."""
    print("\n=== Testing Training Pipeline ===")

    from learning.learning_pipeline import AdapterType

    # Check training status
    status = await system.check_training_status(AdapterType.RESEARCH)
    print(f"  [OK] Training status check: {status['reason']}")

    # Get active adapters (should be empty initially)
    adapters = await system.get_active_adapters()
    print(f"  [OK] Active adapters: {len(adapters)}")

    # Get pipeline stats
    stats = await system.pipeline.get_stats()
    print(f"  [OK] Pipeline stats: {stats}")

    return True


async def test_system_health(system):
    """Test overall system health."""
    print("\n=== Testing System Health ===")

    health = await system.health_check()
    print(f"  Status: {health['status']}")
    for component, status in health['components'].items():
        print(f"    - {component}: {status}")

    return health['status'] == 'healthy'


async def main():
    print("=" * 60)
    print("BenchAI Learning System Test Suite")
    print("=" * 60)

    # Create learning system with test storage
    test_storage = Path("/tmp/benchai_learning_test")
    test_storage.mkdir(parents=True, exist_ok=True)

    system = create_learning_system(test_storage)
    await system.initialize()
    print("[OK] Learning system initialized")

    # Run tests
    all_passed = True

    try:
        all_passed &= await test_memory_system(system)
    except Exception as e:
        print(f"  [FAIL] Memory system: {e}")
        all_passed = False

    try:
        all_passed &= await test_experience_replay(system)
    except Exception as e:
        print(f"  [FAIL] Experience replay: {e}")
        all_passed = False

    try:
        all_passed &= await test_interaction_logging(system)
    except Exception as e:
        print(f"  [FAIL] Interaction logging: {e}")
        all_passed = False

    try:
        all_passed &= await test_agent_coordination(system)
    except Exception as e:
        print(f"  [FAIL] Agent coordination: {e}")
        all_passed = False

    try:
        all_passed &= await test_training_pipeline(system)
    except Exception as e:
        print(f"  [FAIL] Training pipeline: {e}")
        all_passed = False

    try:
        all_passed &= await test_system_health(system)
    except Exception as e:
        print(f"  [FAIL] System health: {e}")
        all_passed = False

    # Get comprehensive stats
    print("\n=== Final Statistics ===")
    stats = await system.get_stats()
    print(f"  Memory: {stats['memory']['total_memories']} memories")
    print(f"  Experiences: {stats['experiences']['total_experiences']} experiences")
    print(f"  Interactions: {stats['interactions']['total_interactions']} logged")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree(test_storage, ignore_errors=True)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
