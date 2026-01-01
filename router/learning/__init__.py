"""
BenchAI Learning System - Self-Improving AI Architecture

Four-Layer Learning System:
1. Zettelkasten Knowledge Graph - Atomic notes with emergent linking
2. Instant Memory (RAG) - Episodic, Semantic, Procedural, Agent Context
3. Experience Replay - Success/Failure trajectories for in-context learning
4. Periodic Fine-Tuning - LoRA adapters via Unsloth

This module enables BenchAI to learn continuously from interactions,
accumulate knowledge, and improve over time. The Zettelkasten layer
provides a "second brain" that other agents can query asynchronously
for deep research while working in parallel.
"""

from .memory_enhanced import EnhancedMemoryManager, MemoryType
from .experience_replay import ExperienceReplayManager, Experience, ExperienceOutcome
from .interaction_logger import InteractionLogger
from .learning_pipeline import LearningPipeline
from .zettelkasten import ZettelkastenKnowledgeGraph, Zettel, ZettelType, LinkType
from .research_api import ResearchAPI, ResearchQuery, QueryPriority, QueryStatus
from .agent_config import get_config, get_marunochi_url, get_dottscavis_url, refresh_config
from .semantic_router import SemanticTaskRouter, route_task, classify_domain
from .agent_sync import AgentSyncManager

__all__ = [
    # Memory
    'EnhancedMemoryManager',
    'MemoryType',
    # Experience
    'ExperienceReplayManager',
    'Experience',
    'ExperienceOutcome',
    # Logging
    'InteractionLogger',
    # Training
    'LearningPipeline',
    # Zettelkasten
    'ZettelkastenKnowledgeGraph',
    'Zettel',
    'ZettelType',
    'LinkType',
    # Research API
    'ResearchAPI',
    'ResearchQuery',
    'QueryPriority',
    'QueryStatus',
    # Agent Configuration
    'get_config',
    'get_marunochi_url',
    'get_dottscavis_url',
    'refresh_config',
    # Routing
    'SemanticTaskRouter',
    'route_task',
    'classify_domain',
    # Sync
    'AgentSyncManager',
]
