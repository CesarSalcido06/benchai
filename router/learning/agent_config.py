"""
Agent Configuration Module

Centralized configuration for multi-agent system endpoints.
All agent URLs are configurable via environment variables.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class AgentEndpoints:
    """Endpoint configuration for an agent."""
    base_url: str
    health: str = "/health"
    sync_push: str = "/v1/sync/receive"
    sync_pull: str = "/v1/sync/share"
    capabilities: str = "/v1/capabilities"
    chat: str = "/v1/chat/completions"
    search: str = "/v1/codebase/search"
    index: str = "/v1/codebase/index"
    stats: str = "/v1/codebase/stats"

    def get_url(self, endpoint: str) -> str:
        """Get full URL for an endpoint."""
        path = getattr(self, endpoint, None)
        if path:
            return f"{self.base_url}{path}"
        return f"{self.base_url}/{endpoint}"


def get_agent_config() -> Dict[str, AgentEndpoints]:
    """
    Get agent endpoint configuration from environment variables.

    Environment variables:
        MARUNOCHI_URL: MarunochiAI base URL (default: http://localhost:8765)
        DOTTSCAVIS_URL: DottscavisAI base URL (default: http://localhost:8766)

    Returns:
        Dict mapping agent ID to AgentEndpoints
    """
    return {
        "marunochiAI": AgentEndpoints(
            base_url=os.environ.get("MARUNOCHI_URL", "http://localhost:8765")
        ),
        "dottscavisAI": AgentEndpoints(
            base_url=os.environ.get("DOTTSCAVIS_URL", "http://localhost:8766")
        ),
    }


def get_marunochi_url() -> str:
    """Get MarunochiAI base URL from environment."""
    return os.environ.get("MARUNOCHI_URL", "http://localhost:8765")


def get_dottscavis_url() -> str:
    """Get DottscavisAI base URL from environment."""
    return os.environ.get("DOTTSCAVIS_URL", "http://localhost:8766")


# Singleton config instance
_config: Dict[str, AgentEndpoints] = None


def get_config() -> Dict[str, AgentEndpoints]:
    """Get or initialize the agent configuration."""
    global _config
    if _config is None:
        _config = get_agent_config()
    return _config


def refresh_config():
    """Refresh configuration from environment (useful after env changes)."""
    global _config
    _config = get_agent_config()
    return _config
