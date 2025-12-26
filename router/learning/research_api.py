"""
Async Research API for Multi-Agent Knowledge Queries

This provides the interface for agents (MarunochiAI, DottscavisAI) to query
BenchAI's knowledge graph while working in parallel on their own tasks.

Features:
- Async queries that don't block agent work
- Deep knowledge traversal with graph expansion
- Relevance scoring based on Zettelkasten connections
- Query caching for repeated requests
- Priority queuing for urgent research needs
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import heapq


class QueryPriority(str, Enum):
    """Priority levels for research queries."""
    CRITICAL = "critical"  # Blocking agent work
    HIGH = "high"          # Needed soon
    NORMAL = "normal"      # Background research
    LOW = "low"           # Nice to have


class QueryStatus(str, Enum):
    """Status of a research query."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class ResearchQuery:
    """A research query from an agent."""
    id: str
    query: str
    agent_id: str
    priority: QueryPriority
    status: QueryStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    options: Dict = field(default_factory=dict)

    def __lt__(self, other):
        # Priority ordering for heap
        priority_order = {
            QueryPriority.CRITICAL: 0,
            QueryPriority.HIGH: 1,
            QueryPriority.NORMAL: 2,
            QueryPriority.LOW: 3
        }
        return priority_order[self.priority] < priority_order[other.priority]


class ResearchAPI:
    """
    Async Research API for agent knowledge queries.

    Agents call this to request research from BenchAI's knowledge graph.
    Queries are processed asynchronously so agents can continue working.

    Usage:
        # Agent submits query
        query_id = await research_api.submit_query(
            query="How does the Zettelkasten method handle linking?",
            agent_id="marunochiAI",
            priority=QueryPriority.NORMAL
        )

        # Agent continues working...

        # Later, check if result is ready
        result = await research_api.get_result(query_id)
        if result['status'] == 'completed':
            knowledge = result['data']
    """

    def __init__(self, knowledge_graph, cache_ttl_minutes: int = 30):
        self.knowledge_graph = knowledge_graph
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Query queue and results
        self._queue: List[ResearchQuery] = []
        self._results: Dict[str, ResearchQuery] = {}
        self._cache: Dict[str, Dict] = {}

        # Processing
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        # Callbacks for query completion
        self._callbacks: Dict[str, List[Callable[[ResearchQuery], Awaitable[None]]]] = {}

    def _generate_query_id(self, query: str, agent_id: str) -> str:
        """Generate unique query ID."""
        timestamp = datetime.now().strftime("%H%M%S%f")
        content_hash = hashlib.sha256(f"{query}:{agent_id}".encode()).hexdigest()[:8]
        return f"rq-{timestamp}-{content_hash}"

    def _cache_key(self, query: str, options: Dict) -> str:
        """Generate cache key for a query."""
        content = json.dumps({"query": query, "options": options}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def start(self):
        """Start the query processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        print("[RESEARCH-API] Query processor started")

    async def stop(self):
        """Stop the query processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        print("[RESEARCH-API] Query processor stopped")

    async def submit_query(
        self,
        query: str,
        agent_id: str,
        priority: QueryPriority = QueryPriority.NORMAL,
        expand_graph: bool = True,
        graph_depth: int = 2,
        max_results: int = 10,
        include_hubs: bool = True,
        callback: Optional[Callable[[ResearchQuery], Awaitable[None]]] = None
    ) -> str:
        """
        Submit a research query for async processing.

        Args:
            query: The research query
            agent_id: Which agent is asking
            priority: Query priority
            expand_graph: Whether to traverse knowledge graph
            graph_depth: How many hops to traverse
            max_results: Maximum results to return
            include_hubs: Include hub notes as entry points
            callback: Optional async callback when complete

        Returns:
            Query ID for tracking
        """
        options = {
            "expand_graph": expand_graph,
            "graph_depth": graph_depth,
            "max_results": max_results,
            "include_hubs": include_hubs
        }

        # Check cache first
        cache_key = self._cache_key(query, options)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now() < cached['expires_at']:
                # Return cached result
                query_id = self._generate_query_id(query, agent_id)
                research_query = ResearchQuery(
                    id=query_id,
                    query=query,
                    agent_id=agent_id,
                    priority=priority,
                    status=QueryStatus.CACHED,
                    created_at=datetime.now(),
                    completed_at=datetime.now(),
                    result=cached['result'],
                    options=options
                )
                self._results[query_id] = research_query
                print(f"[RESEARCH-API] Cache hit for query from {agent_id}")
                return query_id

        # Create new query
        query_id = self._generate_query_id(query, agent_id)
        research_query = ResearchQuery(
            id=query_id,
            query=query,
            agent_id=agent_id,
            priority=priority,
            status=QueryStatus.PENDING,
            created_at=datetime.now(),
            options=options
        )

        # Register callback
        if callback:
            self._callbacks[query_id] = [callback]

        # Add to queue
        async with self._lock:
            heapq.heappush(self._queue, research_query)
            self._results[query_id] = research_query

        print(f"[RESEARCH-API] Query {query_id} submitted by {agent_id} (priority: {priority.value})")
        return query_id

    async def get_result(
        self,
        query_id: str,
        wait: bool = False,
        timeout: float = 30.0
    ) -> Dict:
        """
        Get the result of a research query.

        Args:
            query_id: The query ID
            wait: Whether to wait for completion
            timeout: Max time to wait (seconds)

        Returns:
            Query result with status
        """
        if query_id not in self._results:
            return {"status": "not_found", "error": "Query ID not found"}

        query = self._results[query_id]

        if wait and query.status in [QueryStatus.PENDING, QueryStatus.PROCESSING]:
            start_time = time.time()
            while query.status in [QueryStatus.PENDING, QueryStatus.PROCESSING]:
                if time.time() - start_time > timeout:
                    return {"status": "timeout", "error": "Query still processing"}
                await asyncio.sleep(0.1)

        return {
            "id": query.id,
            "status": query.status.value,
            "query": query.query,
            "agent_id": query.agent_id,
            "created_at": query.created_at.isoformat(),
            "completed_at": query.completed_at.isoformat() if query.completed_at else None,
            "data": query.result,
            "error": query.error
        }

    async def _process_loop(self):
        """Background loop processing research queries."""
        while self._running:
            try:
                # Get next query from priority queue
                query = None
                async with self._lock:
                    if self._queue:
                        query = heapq.heappop(self._queue)

                if query:
                    await self._process_query(query)
                else:
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[RESEARCH-API] Processing error: {e}")
                await asyncio.sleep(1)

    async def _process_query(self, query: ResearchQuery):
        """Process a single research query."""
        query.status = QueryStatus.PROCESSING
        start_time = time.time()

        try:
            # Perform the research
            result = await self._execute_research(query)

            query.status = QueryStatus.COMPLETED
            query.result = result
            query.completed_at = datetime.now()

            # Cache the result
            cache_key = self._cache_key(query.query, query.options)
            self._cache[cache_key] = {
                'result': result,
                'expires_at': datetime.now() + self.cache_ttl
            }

            duration_ms = int((time.time() - start_time) * 1000)
            print(f"[RESEARCH-API] Query {query.id} completed in {duration_ms}ms")

        except Exception as e:
            query.status = QueryStatus.FAILED
            query.error = str(e)
            query.completed_at = datetime.now()
            print(f"[RESEARCH-API] Query {query.id} failed: {e}")

        # Trigger callbacks
        if query.id in self._callbacks:
            for callback in self._callbacks[query.id]:
                try:
                    await callback(query)
                except Exception as e:
                    print(f"[RESEARCH-API] Callback error: {e}")
            del self._callbacks[query.id]

    async def _execute_research(self, query: ResearchQuery) -> Dict:
        """Execute the actual research query."""
        options = query.options

        # Search the knowledge graph
        results = await self.knowledge_graph.search(
            query=query.query,
            limit=options.get('max_results', 10),
            expand_graph=options.get('expand_graph', True),
            graph_depth=options.get('graph_depth', 2)
        )

        # Find relevant hub notes if requested
        hubs = []
        if options.get('include_hubs', True):
            all_hubs = await self.knowledge_graph.find_hubs(limit=5)
            # Filter hubs relevant to the query
            for hub in all_hubs:
                if any(keyword in query.query.lower() for keyword in hub.get('title', '').lower().split()):
                    hubs.append(hub)

        # Build the research response
        response = {
            "query": query.query,
            "timestamp": datetime.now().isoformat(),
            "zettels": results,
            "zettel_count": len(results),
            "hubs": hubs,
            "graph_expanded": options.get('expand_graph', True),
            "depth": options.get('graph_depth', 2)
        }

        # Generate a synthesis if we have results
        if results:
            response["synthesis"] = self._synthesize_results(query.query, results)

        return response

    def _synthesize_results(self, query: str, results: List[Dict]) -> str:
        """Synthesize search results into a coherent summary."""
        if not results:
            return "No relevant knowledge found."

        # Sort by relevance/importance
        sorted_results = sorted(
            results,
            key=lambda x: (x.get('_search_rank', 0), x.get('importance', 0)),
            reverse=True
        )

        # Build synthesis
        synthesis_parts = [f"Found {len(results)} relevant knowledge nodes:\n"]

        for i, zettel in enumerate(sorted_results[:5], 1):
            title = zettel.get('title', 'Untitled')
            content = zettel.get('content', '')[:200]
            link_count = len(zettel.get('links', []))

            synthesis_parts.append(f"\n{i}. **{title}**")
            synthesis_parts.append(f"   {content}...")
            if link_count > 0:
                synthesis_parts.append(f"   (Connected to {link_count} other notes)")

        if len(results) > 5:
            synthesis_parts.append(f"\n... and {len(results) - 5} more related notes.")

        return "\n".join(synthesis_parts)

    async def get_pending_queries(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get pending queries, optionally filtered by agent."""
        queries = []
        for query in self._results.values():
            if query.status in [QueryStatus.PENDING, QueryStatus.PROCESSING]:
                if agent_id is None or query.agent_id == agent_id:
                    queries.append({
                        "id": query.id,
                        "query": query.query,
                        "agent_id": query.agent_id,
                        "priority": query.priority.value,
                        "status": query.status.value,
                        "created_at": query.created_at.isoformat()
                    })
        return queries

    async def get_stats(self) -> Dict:
        """Get API statistics."""
        status_counts = {}
        agent_counts = {}

        for query in self._results.values():
            status = query.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            agent = query.agent_id
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            "total_queries": len(self._results),
            "queue_size": len(self._queue),
            "cache_size": len(self._cache),
            "by_status": status_counts,
            "by_agent": agent_counts,
            "processor_running": self._running
        }

    def clear_cache(self):
        """Clear the query cache."""
        self._cache.clear()
        print("[RESEARCH-API] Cache cleared")


# Convenience function to create and start the research API
async def create_research_api(knowledge_graph) -> ResearchAPI:
    """Create and start a research API instance."""
    api = ResearchAPI(knowledge_graph)
    await api.start()
    return api
