"""
Semantic Task Router for Multi-Agent System

Routes tasks to the appropriate agent based on:
- Task content analysis (keywords, domain)
- Agent capabilities and skills
- Agent load and availability
- Historical success patterns
"""

import re
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


async def call_marunochi_search(query: str, limit: int = 10) -> Optional[Dict[str, Any]]:
    """
    Call MarunochiAI's codebase search endpoint.

    Args:
        query: Search query
        limit: Max results

    Returns:
        Search results or None if unavailable
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8765/v1/codebase/search",
                json={"query": query, "limit": limit},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None


async def check_marunochi_health() -> bool:
    """Check if MarunochiAI is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8765/health",
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("status") in ["healthy", "degraded"]
    except Exception:
        pass
    return False


class TaskDomain(Enum):
    """High-level task domains for routing."""
    RESEARCH = "research"
    CODING = "coding"
    CREATIVE = "creative"
    ORCHESTRATION = "orchestration"
    GENERAL = "general"


@dataclass
class RouteResult:
    """Result of task routing decision."""
    target_agent: str
    confidence: float  # 0.0 to 1.0
    domain: TaskDomain
    matched_capabilities: List[str]
    reasoning: str


# Keyword patterns for domain classification
DOMAIN_PATTERNS = {
    TaskDomain.RESEARCH: {
        "keywords": [
            "research", "analyze", "explain", "understand", "learn", "study",
            "investigate", "find information", "knowledge", "documentation",
            "best practices", "how does", "what is", "why does", "compare",
            "deep dive", "explore", "discover", "literature", "paper"
        ],
        "weight": 1.0
    },
    TaskDomain.CODING: {
        "keywords": [
            "code", "implement", "debug", "fix", "refactor", "test", "program",
            "function", "class", "api", "endpoint", "database", "python",
            "javascript", "typescript", "rust", "compile", "build", "deploy",
            "git", "commit", "pull request", "bug", "error", "exception",
            "algorithm", "data structure", "optimize", "performance"
        ],
        "weight": 1.0
    },
    TaskDomain.CREATIVE: {
        "keywords": [
            "image", "video", "audio", "render", "generate", "create art",
            "design", "animation", "3d model", "texture", "visualize",
            "illustration", "graphic", "photo", "music", "sound", "voice",
            "style", "aesthetic", "artistic", "creative", "composition",
            "blender", "stable diffusion", "midjourney", "dalle"
        ],
        "weight": 1.0
    },
    TaskDomain.ORCHESTRATION: {
        "keywords": [
            "coordinate", "orchestrate", "manage agents", "delegate", "workflow",
            "pipeline", "multi-step", "complex task", "project", "plan"
        ],
        "weight": 0.8
    }
}

# Agent capability mappings
AGENT_CAPABILITIES = {
    "benchai": {
        "domains": [TaskDomain.RESEARCH, TaskDomain.ORCHESTRATION],
        "capabilities": [
            "research", "memory", "rag", "zettelkasten", "training",
            "experience_replay", "orchestration", "tts", "knowledge_graph"
        ],
        "keywords": [
            "remember", "recall", "knowledge", "learn", "zettelkasten",
            "memory", "rag", "vector search", "semantic search"
        ],
        "always_available": True,
        "priority": 0.9  # High priority for its domains
    },
    "marunochiAI": {
        "domains": [TaskDomain.CODING],
        "capabilities": [
            "coding", "debugging", "testing", "code_review", "refactoring",
            "architecture", "python", "typescript", "fullstack",
            # Phase 2: Code Understanding
            "code_search", "semantic_search", "codebase_indexing",
            "hybrid_search", "code_completion", "code_explanation"
        ],
        "keywords": [
            "write code", "fix bug", "review code", "implement feature",
            "unit test", "integration test", "api", "backend", "frontend",
            # Phase 2 keywords
            "search code", "find function", "where is", "code for",
            "index codebase", "search codebase", "find class", "find method"
        ],
        "endpoints": {
            "chat": "http://localhost:8765/v1/chat/completions",
            "search": "http://localhost:8765/v1/codebase/search",
            "index": "http://localhost:8765/v1/codebase/index",
            "stats": "http://localhost:8765/v1/codebase/stats"
        },
        "always_available": False,
        "priority": 0.95  # Very high for coding tasks
    },
    "dottscavisAI": {
        "domains": [TaskDomain.CREATIVE],
        "capabilities": [
            "image_generation", "video_editing", "3d_modeling",
            "audio_processing", "animation", "compositing"
        ],
        "keywords": [
            "generate image", "create video", "3d model", "render",
            "animation", "visual", "audio", "transcribe", "voice"
        ],
        "always_available": False,
        "priority": 0.95  # Very high for creative tasks
    }
}


class SemanticTaskRouter:
    """
    Routes tasks to appropriate agents using semantic analysis.

    Features:
    - Keyword-based domain classification
    - Capability matching
    - Load-aware routing
    - Fallback to orchestrator for ambiguous tasks
    """

    def __init__(self):
        self.domain_patterns = DOMAIN_PATTERNS
        self.agent_capabilities = AGENT_CAPABILITIES

    def classify_domain(self, task_description: str) -> Tuple[TaskDomain, float]:
        """
        Classify the task into a domain.

        Returns:
            Tuple of (domain, confidence)
        """
        task_lower = task_description.lower()
        domain_scores: Dict[TaskDomain, float] = {}

        for domain, config in self.domain_patterns.items():
            score = 0.0
            matched_count = 0

            for keyword in config["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower in task_lower:
                    # Exact word match scores higher
                    if f" {keyword_lower} " in f" {task_lower} " or task_lower.startswith(f"{keyword_lower} ") or task_lower.endswith(f" {keyword_lower}"):
                        score += 3.0
                    else:
                        score += 1.5
                    matched_count += 1

            # Normalize and boost based on matches
            if config["keywords"]:
                # Base score from matches
                base_score = score / len(config["keywords"])
                # Significant boost for multiple matches
                if matched_count >= 3:
                    base_score *= 2.0
                elif matched_count >= 2:
                    base_score *= 1.5
                elif matched_count >= 1:
                    base_score *= 1.2
                score = base_score * config["weight"]

            domain_scores[domain] = score

        # Find best domain
        if not domain_scores:
            return TaskDomain.GENERAL, 0.0

        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]

        # Normalize confidence to 0-1 (scale up for better routing)
        confidence = min(best_score * 2.0, 1.0)

        # Lower threshold for domain classification
        if confidence < 0.05:
            return TaskDomain.GENERAL, 0.0

        return best_domain, confidence

    def match_capabilities(
        self,
        task_description: str,
        agent_id: str
    ) -> Tuple[List[str], float]:
        """
        Match task against agent capabilities.

        Returns:
            Tuple of (matched_capabilities, match_score)
        """
        agent = self.agent_capabilities.get(agent_id, {})
        capabilities = agent.get("capabilities", [])
        keywords = agent.get("keywords", [])

        task_lower = task_description.lower()
        matched = []

        # Check capabilities
        for cap in capabilities:
            cap_lower = cap.lower().replace("_", " ")
            if cap_lower in task_lower or cap in task_lower:
                matched.append(cap)

        # Check agent-specific keywords
        for keyword in keywords:
            if keyword.lower() in task_lower:
                matched.append(f"keyword:{keyword}")

        # Calculate match score
        total_items = len(capabilities) + len(keywords)
        if total_items == 0:
            return [], 0.0

        score = len(matched) / total_items
        return matched, score

    def route_task(
        self,
        task_description: str,
        available_agents: Optional[List[Dict]] = None,
        prefer_agent: Optional[str] = None
    ) -> RouteResult:
        """
        Route a task to the most appropriate agent.

        Args:
            task_description: The task to route
            available_agents: List of agents with their status/load
            prefer_agent: If specified, prefer this agent if capable

        Returns:
            RouteResult with target agent and reasoning
        """
        # Classify the task domain
        domain, domain_confidence = self.classify_domain(task_description)

        # Build agent scores
        agent_scores: Dict[str, Tuple[float, List[str], str]] = {}

        for agent_id, agent_config in self.agent_capabilities.items():
            # Check domain match
            domain_match = domain in agent_config["domains"]

            # Check capability match
            matched_caps, cap_score = self.match_capabilities(task_description, agent_id)

            # Calculate base score - heavily favor domain match
            base_score = 0.0
            if domain_match:
                base_score += 0.6 * domain_confidence  # Strong domain weight
            base_score += 0.4 * cap_score  # Capability weight
            base_score *= agent_config["priority"]

            # Bonus for matching domain even with low confidence
            if domain_match and domain_confidence > 0.1:
                base_score += 0.15

            # Check availability
            is_available = True
            load = 0.0

            if available_agents:
                agent_info = next(
                    (a for a in available_agents if a.get("id") == agent_id),
                    None
                )
                if agent_info:
                    status = agent_info.get("status", "offline")
                    is_available = status in ["online", "idle"]
                    load = agent_info.get("load", 0.0)
                elif not agent_config["always_available"]:
                    is_available = False

            # Adjust score based on availability and load
            if not is_available and not agent_config["always_available"]:
                base_score *= 0.1  # Heavily penalize unavailable agents
            else:
                # Reduce score based on load (higher load = lower score)
                base_score *= (1 - load * 0.3)

            # Prefer specified agent if capable
            if prefer_agent and agent_id == prefer_agent and base_score > 0.1:
                base_score *= 1.5

            reasoning = f"Domain: {domain.value} (conf: {domain_confidence:.2f}), "
            reasoning += f"Caps matched: {len(matched_caps)}, "
            reasoning += f"Available: {is_available}, Load: {load:.1%}"

            agent_scores[agent_id] = (base_score, matched_caps, reasoning)

        # Select best agent
        best_agent = max(agent_scores, key=lambda x: agent_scores[x][0])
        score, matched_caps, reasoning = agent_scores[best_agent]

        # If score is too low AND domain is general, fall back to orchestrator
        # Otherwise, trust the domain classification
        if score < 0.1 and domain == TaskDomain.GENERAL:
            best_agent = "benchai"
            matched_caps = ["orchestration", "general"]
            reasoning = f"Low confidence routing, falling back to orchestrator. Original: {reasoning}"

        return RouteResult(
            target_agent=best_agent,
            confidence=min(score * 1.5, 1.0),  # Boost confidence for display
            domain=domain,
            matched_capabilities=matched_caps,
            reasoning=reasoning
        )

    def suggest_agents(
        self,
        task_description: str,
        available_agents: Optional[List[Dict]] = None,
        top_n: int = 3
    ) -> List[RouteResult]:
        """
        Suggest multiple agents that could handle a task.

        Returns top N agents sorted by suitability.
        """
        domain, domain_confidence = self.classify_domain(task_description)
        results = []

        for agent_id, agent_config in self.agent_capabilities.items():
            domain_match = domain in agent_config["domains"]
            matched_caps, cap_score = self.match_capabilities(task_description, agent_id)

            base_score = 0.0
            if domain_match:
                base_score += 0.4 * domain_confidence
            base_score += 0.3 * cap_score
            base_score *= agent_config["priority"]

            # Check availability
            is_available = True
            load = 0.0
            if available_agents:
                agent_info = next(
                    (a for a in available_agents if a.get("id") == agent_id),
                    None
                )
                if agent_info:
                    is_available = agent_info.get("status", "offline") in ["online", "idle"]
                    load = agent_info.get("load", 0.0)
                elif not agent_config["always_available"]:
                    is_available = False

            if not is_available and not agent_config["always_available"]:
                base_score *= 0.1
            else:
                base_score *= (1 - load * 0.3)

            reasoning = f"Domain: {domain.value}, Available: {is_available}"

            results.append(RouteResult(
                target_agent=agent_id,
                confidence=min(base_score, 1.0),
                domain=domain,
                matched_capabilities=matched_caps,
                reasoning=reasoning
            ))

        # Sort by confidence and return top N
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_n]


# Singleton instance
_router = SemanticTaskRouter()


def get_router() -> SemanticTaskRouter:
    """Get the semantic router instance."""
    return _router


def route_task(
    task_description: str,
    available_agents: Optional[List[Dict]] = None,
    prefer_agent: Optional[str] = None
) -> RouteResult:
    """Convenience function to route a task."""
    return _router.route_task(task_description, available_agents, prefer_agent)


def classify_domain(task_description: str) -> Tuple[TaskDomain, float]:
    """Convenience function to classify task domain."""
    return _router.classify_domain(task_description)
