"""
Interaction Logger for Continuous Learning

Captures all interactions with the system for:
1. Experience replay analysis
2. Fine-tuning data generation
3. Performance monitoring
4. Multi-agent coordination tracking
"""

import asyncio
import aiosqlite
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class InteractionType(str, Enum):
    """Types of interactions to log."""
    CHAT = "chat"
    TOOL_USE = "tool_use"
    AGENT_CALL = "agent_call"
    RAG_QUERY = "rag_query"
    MEMORY_ACCESS = "memory_access"
    FINE_TUNE = "fine_tune"
    SYSTEM = "system"


@dataclass
class Interaction:
    """A single interaction record."""
    id: Optional[int]
    session_id: str
    interaction_type: InteractionType
    request: Dict[str, Any]
    response: Dict[str, Any]
    model_used: str
    tokens_input: int
    tokens_output: int
    duration_ms: int
    success: bool
    error_message: Optional[str]
    agent_source: str
    user_feedback: Optional[int]  # 1-5 rating if provided
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class InteractionLogger:
    """
    Logs all interactions for analysis and learning.

    Features:
    - Session tracking for conversation continuity
    - Performance metrics per model/agent
    - Automatic quality assessment
    - Training data extraction
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False
        self._current_session = None
        self._session_start = None

        # Performance tracking
        self._metrics_cache = {}
        self._metrics_last_update = None

    async def _optimize_connection(self, db):
        """Apply performance optimizations."""
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        await db.execute("PRAGMA cache_size = -40000")
        await db.execute("PRAGMA temp_store = MEMORY")

    async def initialize(self):
        """Initialize the interaction logging database."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Sessions table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    agent_source TEXT DEFAULT 'benchai',
                    interaction_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    avg_response_time_ms REAL,
                    success_rate REAL,
                    metadata JSON DEFAULT '{}'
                )
            ''')

            # Interactions table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    request JSON NOT NULL,
                    response JSON,
                    model_used TEXT,
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    agent_source TEXT DEFAULT 'benchai',
                    user_feedback INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Indexes
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_int_session
                ON interactions(session_id, created_at DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_int_type
                ON interactions(interaction_type, success)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_int_model
                ON interactions(model_used, duration_ms)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_int_agent
                ON interactions(agent_source)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_int_feedback
                ON interactions(user_feedback) WHERE user_feedback IS NOT NULL
            ''')

            # Performance metrics table (aggregated)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_date DATE NOT NULL,
                    model_used TEXT,
                    agent_source TEXT,
                    interaction_type TEXT,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    avg_duration_ms REAL,
                    p95_duration_ms REAL,
                    avg_feedback REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(metric_date, model_used, agent_source, interaction_type)
                )
            ''')

            await db.execute("ANALYZE")
            await db.commit()

        self._initialized = True
        print("[LOGGER] Interaction logger initialized")

    async def start_session(self, session_id: str, agent_source: str = "benchai", metadata: Optional[Dict] = None) -> str:
        """Start a new interaction session."""
        await self.initialize()

        self._current_session = session_id
        self._session_start = time.time()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''INSERT OR REPLACE INTO sessions (id, agent_source, metadata)
                   VALUES (?, ?, ?)''',
                (session_id, agent_source, json.dumps(metadata or {}))
            )
            await db.commit()

        return session_id

    async def end_session(self, session_id: Optional[str] = None):
        """End a session and compute aggregates."""
        await self.initialize()

        session_id = session_id or self._current_session
        if not session_id:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Compute session aggregates
            cursor = await db.execute(
                '''SELECT
                       COUNT(*) as count,
                       SUM(tokens_input + tokens_output) as total_tokens,
                       AVG(duration_ms) as avg_duration,
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate
                   FROM interactions WHERE session_id = ?''',
                (session_id,)
            )
            row = await cursor.fetchone()

            await db.execute(
                '''UPDATE sessions SET
                       ended_at = CURRENT_TIMESTAMP,
                       interaction_count = ?,
                       total_tokens = ?,
                       avg_response_time_ms = ?,
                       success_rate = ?
                   WHERE id = ?''',
                (row[0], row[1] or 0, row[2], row[3], session_id)
            )
            await db.commit()

        if session_id == self._current_session:
            self._current_session = None
            self._session_start = None

    async def log(
        self,
        interaction_type: InteractionType,
        request: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        model_used: str = "",
        tokens_input: int = 0,
        tokens_output: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        agent_source: str = "benchai",
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Log an interaction.

        Args:
            interaction_type: Type of interaction (chat, tool_use, etc.)
            request: The request data
            response: The response data
            model_used: Which model was used
            tokens_input/output: Token counts
            duration_ms: How long it took
            success: Whether it succeeded
            error_message: Error if failed
            agent_source: Which agent made this
            session_id: Session to associate with
            metadata: Additional data

        Returns:
            ID of the logged interaction
        """
        await self.initialize()

        session_id = session_id or self._current_session or "default"

        # Ensure session exists
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (id, agent_source) VALUES (?, ?)",
                (session_id, agent_source)
            )

            cursor = await db.execute(
                '''INSERT INTO interactions
                   (session_id, interaction_type, request, response, model_used,
                    tokens_input, tokens_output, duration_ms, success,
                    error_message, agent_source, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    session_id, interaction_type.value,
                    json.dumps(request), json.dumps(response or {}),
                    model_used, tokens_input, tokens_output, duration_ms,
                    1 if success else 0, error_message, agent_source,
                    json.dumps(metadata or {})
                )
            )
            interaction_id = cursor.lastrowid
            await db.commit()

            return interaction_id

    async def add_feedback(self, interaction_id: int, feedback: int):
        """Add user feedback to an interaction (1-5 scale)."""
        await self.initialize()

        feedback = max(1, min(5, feedback))  # Clamp to 1-5

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE interactions SET user_feedback = ? WHERE id = ?",
                (feedback, interaction_id)
            )
            await db.commit()

    async def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get interaction history for a session."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                '''SELECT * FROM interactions
                   WHERE session_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?''',
                (session_id, limit)
            )

            interactions = []
            async for row in cursor:
                interaction = dict(row)
                interaction['request'] = json.loads(interaction['request'])
                interaction['response'] = json.loads(interaction['response'])
                interaction['metadata'] = json.loads(interaction['metadata'])
                interactions.append(interaction)

            return list(reversed(interactions))  # Chronological order

    async def get_recent_failures(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """Get recent failed interactions for analysis."""
        await self.initialize()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                '''SELECT * FROM interactions
                   WHERE success = 0 AND created_at > ?
                   ORDER BY created_at DESC
                   LIMIT ?''',
                (cutoff, limit)
            )

            failures = []
            async for row in cursor:
                failure = dict(row)
                failure['request'] = json.loads(failure['request'])
                failure['response'] = json.loads(failure['response'])
                failures.append(failure)

            return failures

    async def get_high_quality_interactions(
        self,
        min_feedback: int = 4,
        interaction_type: Optional[InteractionType] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get high-quality interactions for training data.
        These are interactions with positive user feedback.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            conditions = ["user_feedback >= ?", "success = 1"]
            params = [min_feedback]

            if interaction_type:
                conditions.append("interaction_type = ?")
                params.append(interaction_type.value)

            params.append(limit)
            where_clause = " AND ".join(conditions)

            cursor = await db.execute(f'''
                SELECT * FROM interactions
                WHERE {where_clause}
                ORDER BY user_feedback DESC, created_at DESC
                LIMIT ?
            ''', params)

            interactions = []
            async for row in cursor:
                interaction = dict(row)
                interaction['request'] = json.loads(interaction['request'])
                interaction['response'] = json.loads(interaction['response'])
                interactions.append(interaction)

            return interactions

    async def aggregate_metrics(self, date: Optional[str] = None):
        """
        Aggregate performance metrics for a given date.
        Run this periodically (e.g., daily) to maintain performance history.
        """
        await self.initialize()

        date = date or datetime.now().strftime("%Y-%m-%d")

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Get aggregated metrics by model, agent, and type
            cursor = await db.execute(
                '''SELECT
                       model_used, agent_source, interaction_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                       SUM(tokens_input + tokens_output) as tokens,
                       AVG(duration_ms) as avg_duration,
                       AVG(user_feedback) as avg_feedback
                   FROM interactions
                   WHERE DATE(created_at) = ?
                   GROUP BY model_used, agent_source, interaction_type''',
                (date,)
            )

            async for row in cursor:
                # Calculate P95 duration for this group
                p95_cursor = await db.execute(
                    '''SELECT duration_ms FROM interactions
                       WHERE DATE(created_at) = ?
                       AND model_used = ? AND agent_source = ? AND interaction_type = ?
                       ORDER BY duration_ms DESC
                       LIMIT 1 OFFSET (
                           SELECT CAST(COUNT(*) * 0.05 AS INTEGER) FROM interactions
                           WHERE DATE(created_at) = ?
                           AND model_used = ? AND agent_source = ? AND interaction_type = ?
                       )''',
                    (date, row[0], row[1], row[2], date, row[0], row[1], row[2])
                )
                p95_row = await p95_cursor.fetchone()
                p95_duration = p95_row[0] if p95_row else row[5]

                await db.execute(
                    '''INSERT OR REPLACE INTO performance_metrics
                       (metric_date, model_used, agent_source, interaction_type,
                        total_requests, successful_requests, total_tokens,
                        avg_duration_ms, p95_duration_ms, avg_feedback)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (date, row[0], row[1], row[2], row[3], row[4], row[5], row[6], p95_duration, row[7])
                )

            await db.commit()

    async def get_performance_trends(self, days: int = 7) -> Dict:
        """Get performance trends over recent days."""
        await self.initialize()

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Daily totals
            cursor = await db.execute(
                '''SELECT
                       metric_date,
                       SUM(total_requests) as requests,
                       SUM(successful_requests) as successes,
                       AVG(avg_duration_ms) as avg_duration,
                       AVG(avg_feedback) as avg_feedback
                   FROM performance_metrics
                   WHERE metric_date >= ?
                   GROUP BY metric_date
                   ORDER BY metric_date''',
                (cutoff,)
            )

            daily = []
            async for row in cursor:
                daily.append({
                    "date": row['metric_date'],
                    "requests": row['requests'],
                    "success_rate": round(row['successes'] / max(1, row['requests']) * 100, 1),
                    "avg_duration_ms": round(row['avg_duration'] or 0, 1),
                    "avg_feedback": round(row['avg_feedback'] or 0, 2)
                })

            # By model
            cursor = await db.execute(
                '''SELECT
                       model_used,
                       SUM(total_requests) as requests,
                       AVG(avg_duration_ms) as avg_duration
                   FROM performance_metrics
                   WHERE metric_date >= ?
                   GROUP BY model_used
                   ORDER BY requests DESC''',
                (cutoff,)
            )

            by_model = {}
            async for row in cursor:
                by_model[row['model_used'] or 'unknown'] = {
                    "requests": row['requests'],
                    "avg_duration_ms": round(row['avg_duration'] or 0, 1)
                }

            return {
                "daily": daily,
                "by_model": by_model,
                "period_days": days
            }

    async def get_stats(self) -> Dict:
        """Get overall statistics."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            stats = {}

            # Total interactions
            cursor = await db.execute("SELECT COUNT(*) FROM interactions")
            stats["total_interactions"] = (await cursor.fetchone())[0]

            # Total sessions
            cursor = await db.execute("SELECT COUNT(*) FROM sessions")
            stats["total_sessions"] = (await cursor.fetchone())[0]

            # Success rate
            cursor = await db.execute(
                "SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) FROM interactions"
            )
            stats["overall_success_rate"] = round((await cursor.fetchone())[0] or 0, 3)

            # Average feedback
            cursor = await db.execute(
                "SELECT AVG(user_feedback) FROM interactions WHERE user_feedback IS NOT NULL"
            )
            stats["avg_feedback"] = round((await cursor.fetchone())[0] or 0, 2)

            # Interactions with feedback
            cursor = await db.execute(
                "SELECT COUNT(*) FROM interactions WHERE user_feedback IS NOT NULL"
            )
            stats["interactions_with_feedback"] = (await cursor.fetchone())[0]

            # By type
            cursor = await db.execute(
                "SELECT interaction_type, COUNT(*) FROM interactions GROUP BY interaction_type"
            )
            stats["by_type"] = {row[0]: row[1] async for row in cursor}

            # By agent
            cursor = await db.execute(
                "SELECT agent_source, COUNT(*) FROM interactions GROUP BY agent_source"
            )
            stats["by_agent"] = {row[0]: row[1] async for row in cursor}

            # Today's stats
            today = datetime.now().strftime("%Y-%m-%d")
            cursor = await db.execute(
                "SELECT COUNT(*) FROM interactions WHERE DATE(created_at) = ?",
                (today,)
            )
            stats["today_interactions"] = (await cursor.fetchone())[0]

            return stats

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old interaction data to save space."""
        await self.initialize()

        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Delete old interactions (keep high-feedback ones longer)
            cursor = await db.execute(
                '''DELETE FROM interactions
                   WHERE created_at < ?
                   AND (user_feedback IS NULL OR user_feedback < 4)''',
                (cutoff,)
            )
            deleted_interactions = cursor.rowcount

            # Delete old sessions with no interactions
            await db.execute(
                '''DELETE FROM sessions
                   WHERE ended_at < ?
                   AND id NOT IN (SELECT DISTINCT session_id FROM interactions)''',
                (cutoff,)
            )

            await db.execute("VACUUM")
            await db.commit()

            return {"deleted_interactions": deleted_interactions}
