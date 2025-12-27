"""
Learning System Integration Module

This module integrates all learning components and provides:
1. Unified API for the learning system
2. Automatic experience capture from chat interactions
3. Scheduled maintenance tasks (decay, consolidation, training)
4. Multi-agent coordination
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .memory_enhanced import EnhancedMemoryManager, MemoryType
from .experience_replay import ExperienceReplayManager, Experience, ExperienceOutcome, TaskDomain
from .interaction_logger import InteractionLogger, InteractionType
from .learning_pipeline import LearningPipeline, AdapterType, TrainingConfig
from .zettelkasten import ZettelkastenKnowledgeGraph, ZettelType, LinkType
from .research_api import ResearchAPI, QueryPriority


class LearningSystem:
    """
    Unified interface for the BenchAI learning system.

    This class coordinates all learning components and provides:
    - Automatic logging of interactions
    - Experience extraction from successful completions
    - In-context example injection for prompts
    - Scheduled maintenance and training
    - Multi-agent coordination
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.memory = EnhancedMemoryManager(storage_dir / "memory" / "enhanced.db")
        self.experiences = ExperienceReplayManager(storage_dir / "learning" / "experiences.db")
        self.logger = InteractionLogger(storage_dir / "learning" / "interactions.db")
        self.pipeline = LearningPipeline(
            db_path=storage_dir / "learning" / "pipeline.db",
            models_dir=storage_dir / "models",
            adapters_dir=storage_dir / "adapters",
            experience_manager=self.experiences,
            interaction_logger=self.logger
        )

        # Zettelkasten knowledge graph (Layer 0)
        self.zettelkasten = ZettelkastenKnowledgeGraph(storage_dir / "zettelkasten" / "knowledge.db")
        self.research_api: Optional[ResearchAPI] = None

        self._initialized = False
        self._maintenance_task = None

    async def initialize(self):
        """Initialize all learning system components."""
        if self._initialized:
            return

        await self.memory.initialize()
        await self.experiences.initialize()
        await self.logger.initialize()
        await self.pipeline.initialize()

        # Initialize Zettelkasten and Research API
        await self.zettelkasten.initialize()
        self.research_api = ResearchAPI(self.zettelkasten)
        await self.research_api.start()

        self._initialized = True
        print("[LEARNING-SYSTEM] All components initialized (including Zettelkasten)")

    async def start_maintenance_loop(self, interval_hours: int = 24):
        """Start background maintenance tasks."""
        async def maintenance_loop():
            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)
                    await self.run_maintenance()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[LEARNING-SYSTEM] Maintenance error: {e}")

        self._maintenance_task = asyncio.create_task(maintenance_loop())
        print(f"[LEARNING-SYSTEM] Maintenance loop started (every {interval_hours}h)")

    async def stop_maintenance_loop(self):
        """Stop the background maintenance loop."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

        # Stop research API
        if self.research_api:
            await self.research_api.stop()

    async def run_maintenance(self) -> Dict:
        """Run all maintenance tasks."""
        await self.initialize()

        results = {}

        # 1. Memory consolidation
        consolidation = await self.memory.consolidate_old_memories()
        results["memory_consolidation"] = consolidation

        # 2. Importance decay
        await self.memory.apply_importance_decay()
        results["importance_decay"] = "applied"

        # 3. Novelty decay for experiences
        await self.experiences.apply_novelty_decay()
        results["novelty_decay"] = "applied"

        # 4. Aggregate metrics
        await self.logger.aggregate_metrics()
        results["metrics_aggregation"] = "completed"

        # 5. Zettelkasten sleep consolidation (brain-inspired memory processing)
        zettel_consolidation = await self.zettelkasten.sleep_consolidation()
        results["zettelkasten_consolidation"] = zettel_consolidation

        # 6. Check if training should be triggered
        for adapter_type in AdapterType:
            should_train, reason = await self.pipeline.should_trigger_training(adapter_type)
            results[f"training_{adapter_type.value}"] = {"should_train": should_train, "reason": reason}

        print(f"[LEARNING-SYSTEM] Maintenance completed: {results}")
        return results

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        category: str = "general",
        importance: int = 3,
        source: str = "system",
        metadata: Optional[Dict] = None
    ) -> int:
        """Store a new memory."""
        await self.initialize()
        return await self.memory.store(
            content=content,
            memory_type=memory_type,
            category=category,
            importance=importance,
            source=source,
            metadata=metadata
        )

    async def recall(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search memories by query."""
        await self.initialize()
        return await self.memory.search(
            query=query,
            memory_types=memory_types,
            limit=limit
        )

    # =========================================================================
    # Experience Operations
    # =========================================================================

    async def record_success(
        self,
        task: str,
        approach: str,
        trajectory: List[Dict],
        domain: TaskDomain = TaskDomain.GENERAL,
        score: float = 0.8,
        agent: str = "benchai"
    ) -> int:
        """Record a successful experience for learning."""
        await self.initialize()
        return await self.experiences.record_experience(
            task_description=task,
            domain=domain,
            approach=approach,
            trajectory=trajectory,
            outcome=ExperienceOutcome.SUCCESS,
            outcome_details="Task completed successfully",
            success_score=score,
            agent_source=agent
        )

    async def record_failure(
        self,
        task: str,
        approach: str,
        trajectory: List[Dict],
        failure_reason: str,
        domain: TaskDomain = TaskDomain.GENERAL,
        agent: str = "benchai"
    ) -> int:
        """Record a failed experience for learning."""
        await self.initialize()
        return await self.experiences.record_experience(
            task_description=task,
            domain=domain,
            approach=approach,
            trajectory=trajectory,
            outcome=ExperienceOutcome.FAILURE,
            outcome_details=failure_reason,
            success_score=0.0,
            agent_source=agent
        )

    async def get_relevant_examples(
        self,
        task: str,
        domain: Optional[TaskDomain] = None,
        limit: int = 3
    ) -> str:
        """
        Get relevant successful examples for in-context learning.
        Returns formatted string ready to inject into prompts.
        """
        await self.initialize()

        examples = await self.experiences.get_similar_successes(
            task_description=task,
            domain=domain,
            limit=limit
        )

        return self.experiences.format_as_in_context_examples(examples)

    # =========================================================================
    # Interaction Logging
    # =========================================================================

    async def log_chat(
        self,
        request: Dict,
        response: Dict,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: int,
        success: bool = True,
        session_id: Optional[str] = None
    ) -> int:
        """Log a chat interaction."""
        await self.initialize()
        return await self.logger.log(
            interaction_type=InteractionType.CHAT,
            request=request,
            response=response,
            model_used=model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            duration_ms=duration_ms,
            success=success,
            session_id=session_id
        )

    async def log_tool_use(
        self,
        tool_name: str,
        tool_input: Dict,
        tool_output: Dict,
        duration_ms: int,
        success: bool = True,
        session_id: Optional[str] = None
    ) -> int:
        """Log a tool use interaction."""
        await self.initialize()
        return await self.logger.log(
            interaction_type=InteractionType.TOOL_USE,
            request={"tool": tool_name, "input": tool_input},
            response=tool_output,
            duration_ms=duration_ms,
            success=success,
            session_id=session_id
        )

    async def log_agent_call(
        self,
        agent_id: str,
        task: str,
        result: Dict,
        duration_ms: int,
        success: bool = True
    ) -> int:
        """Log a cross-agent call."""
        await self.initialize()
        return await self.logger.log(
            interaction_type=InteractionType.AGENT_CALL,
            request={"agent": agent_id, "task": task},
            response=result,
            duration_ms=duration_ms,
            success=success,
            agent_source=agent_id
        )

    async def add_feedback(self, interaction_id: int, rating: int):
        """Add user feedback to an interaction."""
        await self.initialize()
        await self.logger.add_feedback(interaction_id, rating)

    # =========================================================================
    # Multi-Agent Coordination
    # =========================================================================

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        capabilities: List[str],
        endpoint: Optional[str] = None
    ):
        """Register an agent in the multi-agent system."""
        await self.initialize()
        await self.memory.register_agent(
            agent_id=agent_id,
            name=name,
            role=role,
            capabilities=capabilities,
            endpoint=endpoint
        )

    async def get_agents(self, status: Optional[str] = None) -> List[Dict]:
        """Get registered agents."""
        await self.initialize()
        return await self.memory.get_agents(status=status)

    async def update_agent_status(self, agent_id: str, status: str):
        """Update an agent's status."""
        await self.initialize()
        await self.memory.update_agent_status(agent_id, status)

    async def share_context(
        self,
        content: str,
        from_agent: str,
        category: str = "shared"
    ) -> int:
        """Share context between agents."""
        await self.initialize()
        return await self.memory.store(
            content=content,
            memory_type=MemoryType.AGENT_CONTEXT,
            category=category,
            importance=4,
            source=from_agent
        )

    async def get_shared_context(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get shared context for an agent."""
        await self.initialize()
        return await self.memory.get_agent_context(agent_id=agent_id)

    # =========================================================================
    # Training Operations
    # =========================================================================

    async def trigger_training(
        self,
        adapter_type: AdapterType,
        config: Optional[TrainingConfig] = None
    ) -> Dict:
        """Trigger a training run for an adapter type."""
        await self.initialize()

        run_id = await self.pipeline.create_training_run(adapter_type, config)
        result = await self.pipeline.run_training(run_id)

        if result["status"] == "completed":
            await self.pipeline.activate_adapter(run_id)

        return result

    async def check_training_status(self, adapter_type: AdapterType) -> Dict:
        """Check if training should be triggered."""
        await self.initialize()
        should_train, reason = await self.pipeline.should_trigger_training(adapter_type)
        return {"should_train": should_train, "reason": reason}

    async def get_active_adapters(self) -> Dict:
        """Get currently active adapters."""
        await self.initialize()
        return await self.pipeline.get_active_adapters()

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> Dict:
        """Get comprehensive statistics for the learning system."""
        await self.initialize()

        return {
            "memory": await self.memory.get_stats(),
            "experiences": await self.experiences.get_stats(),
            "interactions": await self.logger.get_stats(),
            "training": await self.pipeline.get_stats(),
            "performance_trends": await self.logger.get_performance_trends(days=7)
        }

    async def health_check(self) -> Dict:
        """Check health of all learning system components."""
        await self.initialize()

        health = {
            "status": "healthy",
            "components": {}
        }

        # Check each component
        try:
            await self.memory.get_stats()
            health["components"]["memory"] = "ok"
        except Exception as e:
            health["components"]["memory"] = f"error: {e}"
            health["status"] = "degraded"

        try:
            await self.experiences.get_stats()
            health["components"]["experiences"] = "ok"
        except Exception as e:
            health["components"]["experiences"] = f"error: {e}"
            health["status"] = "degraded"

        try:
            await self.logger.get_stats()
            health["components"]["logger"] = "ok"
        except Exception as e:
            health["components"]["logger"] = f"error: {e}"
            health["status"] = "degraded"

        try:
            await self.pipeline.get_stats()
            health["components"]["pipeline"] = "ok"
        except Exception as e:
            health["components"]["pipeline"] = f"error: {e}"
            health["status"] = "degraded"

        return health


# Convenience function to create a learning system instance
def create_learning_system(storage_dir: Optional[Path] = None) -> LearningSystem:
    """Create a learning system with default or custom storage directory."""
    if storage_dir is None:
        storage_dir = Path.home() / "llm-storage"
    return LearningSystem(storage_dir)
