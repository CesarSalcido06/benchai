"""
Bidirectional Memory Sync for Multi-Agent System

Enables experience and knowledge sharing between agents:
- Push learnings to remote agents
- Pull experiences from remote agents
- Sync Zettelkasten notes across agents
- Federated learning coordination
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from .agent_config import get_config, AgentEndpoints


@dataclass
class SyncResult:
    """Result of a sync operation."""
    agent_id: str
    direction: str  # push, pull, bidirectional
    items_sent: int
    items_received: int
    success: bool
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AgentSyncManager:
    """
    Manages bidirectional sync between BenchAI and other agents.

    Features:
    - Experience sharing (successful task patterns)
    - Knowledge sync (Zettelkasten notes)
    - Capability updates
    - Health-aware sync (only sync with healthy agents)

    Agent endpoints are configured via environment variables:
    - MARUNOCHI_URL: MarunochiAI base URL (default: http://localhost:8765)
    - DOTTSCAVIS_URL: DottscavisAI base URL (default: http://localhost:8766)
    """

    def __init__(self, local_memory, local_zettelkasten):
        """
        Initialize sync manager.

        Args:
            local_memory: Local memory system instance
            local_zettelkasten: Local Zettelkasten instance
        """
        self.memory = local_memory
        self.zettelkasten = local_zettelkasten
        self.sync_history: List[SyncResult] = []

    def _get_agent_endpoints(self) -> Dict[str, AgentEndpoints]:
        """Get agent endpoints from configuration."""
        return get_config()

    async def check_agent_health(self, agent_id: str) -> bool:
        """Check if an agent is available for sync."""
        config = self._get_agent_endpoints().get(agent_id)
        if not config:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                url = config.get_url("health")
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status") in ["healthy", "ok", "degraded"]
        except Exception:
            pass
        return False

    async def push_experiences(
        self,
        agent_id: str,
        experiences: List[Dict[str, Any]],
        sync_type: str = "experience"
    ) -> SyncResult:
        """
        Push experiences to a remote agent.

        Args:
            agent_id: Target agent
            experiences: List of experience records
            sync_type: Type of sync (experience, knowledge, pattern)

        Returns:
            SyncResult with operation details
        """
        config = self._get_agent_endpoints().get(agent_id)
        if not config:
            return SyncResult(
                agent_id=agent_id,
                direction="push",
                items_sent=0,
                items_received=0,
                success=False,
                error=f"Unknown agent: {agent_id}"
            )

        # Check health first
        if not await self.check_agent_health(agent_id):
            return SyncResult(
                agent_id=agent_id,
                direction="push",
                items_sent=0,
                items_received=0,
                success=False,
                error=f"Agent {agent_id} is not available"
            )

        try:
            async with aiohttp.ClientSession() as session:
                url = config.get_url("sync_push")
                payload = {
                    "from_agent": "benchai",
                    "sync_type": sync_type,
                    "items": experiences,
                    "timestamp": datetime.now().isoformat()
                }

                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        sync_result = SyncResult(
                            agent_id=agent_id,
                            direction="push",
                            items_sent=len(experiences),
                            items_received=result.get("items_processed", 0),
                            success=True
                        )
                    else:
                        sync_result = SyncResult(
                            agent_id=agent_id,
                            direction="push",
                            items_sent=0,
                            items_received=0,
                            success=False,
                            error=f"HTTP {resp.status}"
                        )

        except Exception as e:
            sync_result = SyncResult(
                agent_id=agent_id,
                direction="push",
                items_sent=0,
                items_received=0,
                success=False,
                error=str(e)
            )

        self.sync_history.append(sync_result)
        return sync_result

    async def pull_experiences(
        self,
        agent_id: str,
        sync_type: str = "experience",
        since: Optional[str] = None,
        limit: int = 50
    ) -> SyncResult:
        """
        Pull experiences from a remote agent.

        Args:
            agent_id: Source agent
            sync_type: Type of sync
            since: ISO timestamp to fetch updates since
            limit: Max items to fetch

        Returns:
            SyncResult with operation details
        """
        config = self._get_agent_endpoints().get(agent_id)
        if not config:
            return SyncResult(
                agent_id=agent_id,
                direction="pull",
                items_sent=0,
                items_received=0,
                success=False,
                error=f"Unknown agent: {agent_id}"
            )

        if not await self.check_agent_health(agent_id):
            return SyncResult(
                agent_id=agent_id,
                direction="pull",
                items_sent=0,
                items_received=0,
                success=False,
                error=f"Agent {agent_id} is not available"
            )

        try:
            async with aiohttp.ClientSession() as session:
                url = config.get_url("sync_pull")
                params = {
                    "requester": "benchai",
                    "sync_type": sync_type,
                    "limit": limit
                }
                if since:
                    params["since"] = since

                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])

                        # Store received items in local memory
                        for item in items:
                            await self._store_received_item(agent_id, item, sync_type)

                        sync_result = SyncResult(
                            agent_id=agent_id,
                            direction="pull",
                            items_sent=0,
                            items_received=len(items),
                            success=True
                        )
                    else:
                        sync_result = SyncResult(
                            agent_id=agent_id,
                            direction="pull",
                            items_sent=0,
                            items_received=0,
                            success=False,
                            error=f"HTTP {resp.status}"
                        )

        except Exception as e:
            sync_result = SyncResult(
                agent_id=agent_id,
                direction="pull",
                items_sent=0,
                items_received=0,
                success=False,
                error=str(e)
            )

        self.sync_history.append(sync_result)
        return sync_result

    async def _store_received_item(
        self,
        from_agent: str,
        item: Dict[str, Any],
        sync_type: str
    ):
        """Store a received sync item in local memory/zettelkasten."""
        try:
            if sync_type == "experience":
                # Store in memory
                await self.memory.store(
                    content=item.get("content", ""),
                    memory_type="episodic",
                    category="agent_experience",
                    importance=item.get("importance", 3),
                    source=from_agent,
                    metadata={
                        "synced_from": from_agent,
                        "original_id": item.get("id"),
                        "sync_type": sync_type
                    }
                )
            elif sync_type == "knowledge":
                # Store in Zettelkasten
                await self.zettelkasten.create_note(
                    title=item.get("title", f"Synced from {from_agent}"),
                    content=item.get("content", ""),
                    tags=item.get("tags", []) + [f"synced:{from_agent}"],
                    source=from_agent
                )
        except Exception:
            pass  # Log error but don't fail sync

    async def bidirectional_sync(
        self,
        agent_id: str,
        sync_types: List[str] = None
    ) -> Dict[str, SyncResult]:
        """
        Perform bidirectional sync with an agent.

        Args:
            agent_id: Agent to sync with
            sync_types: Types to sync (default: experience, knowledge)

        Returns:
            Dict with push and pull results
        """
        if sync_types is None:
            sync_types = ["experience", "knowledge"]

        results = {}

        for sync_type in sync_types:
            # Get local items to push
            if sync_type == "experience":
                local_items = await self._get_local_experiences(agent_id)
            else:
                local_items = await self._get_local_knowledge(agent_id)

            # Push our data
            push_result = await self.push_experiences(agent_id, local_items, sync_type)
            results[f"push_{sync_type}"] = push_result

            # Pull their data
            pull_result = await self.pull_experiences(agent_id, sync_type)
            results[f"pull_{sync_type}"] = pull_result

        return results

    async def _get_local_experiences(
        self,
        for_agent: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get local experiences to share with an agent."""
        try:
            # Get high-quality recent experiences
            experiences = await self.memory.search(
                query=f"successful {for_agent}",
                limit=limit,
                min_importance=3
            )
            return [
                {
                    "id": e.get("id"),
                    "content": e.get("content"),
                    "importance": e.get("importance", 3),
                    "category": e.get("category"),
                    "created_at": e.get("created_at")
                }
                for e in experiences
            ]
        except Exception:
            return []

    async def _get_local_knowledge(
        self,
        for_agent: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get local Zettelkasten notes to share."""
        try:
            # Get relevant notes
            notes = await self.zettelkasten.search(
                query=for_agent,
                limit=limit
            )
            return [
                {
                    "id": n.get("id"),
                    "title": n.get("title"),
                    "content": n.get("content"),
                    "tags": n.get("tags", [])
                }
                for n in notes
            ]
        except Exception:
            return []

    async def sync_all_agents(self) -> Dict[str, Dict[str, SyncResult]]:
        """Sync with all known agents."""
        all_results = {}

        for agent_id in self._get_agent_endpoints().keys():
            if await self.check_agent_health(agent_id):
                results = await self.bidirectional_sync(agent_id)
                all_results[agent_id] = results

        return all_results

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        total_pushed = sum(r.items_sent for r in self.sync_history)
        total_pulled = sum(r.items_received for r in self.sync_history)
        successful = sum(1 for r in self.sync_history if r.success)
        failed = sum(1 for r in self.sync_history if not r.success)

        return {
            "total_syncs": len(self.sync_history),
            "successful": successful,
            "failed": failed,
            "items_pushed": total_pushed,
            "items_pulled": total_pulled,
            "last_sync": self.sync_history[-1].timestamp if self.sync_history else None
        }


# Singleton instance
_sync_manager: Optional[AgentSyncManager] = None


def get_sync_manager(memory=None, zettelkasten=None) -> AgentSyncManager:
    """Get or create the sync manager instance."""
    global _sync_manager
    if _sync_manager is None and memory and zettelkasten:
        _sync_manager = AgentSyncManager(memory, zettelkasten)
    return _sync_manager


def init_sync_manager(memory, zettelkasten) -> AgentSyncManager:
    """Initialize the sync manager."""
    global _sync_manager
    _sync_manager = AgentSyncManager(memory, zettelkasten)
    return _sync_manager
