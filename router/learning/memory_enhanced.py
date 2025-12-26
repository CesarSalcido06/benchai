"""
Enhanced Memory Manager with Categorized Memory Types

Memory Types:
- EPISODIC: Events and interactions (what happened)
- SEMANTIC: Facts and knowledge (what we know)
- PROCEDURAL: How-to guides and processes (how to do things)
- AGENT_CONTEXT: Cross-agent state and coordination
- EXPERIENCE: Successful/failed trajectories for learning

Features:
- Importance decay over time
- Memory consolidation (summarize old memories)
- Cross-agent memory sharing
- Automatic categorization
"""

import asyncio
import aiosqlite
import json
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


class MemoryType(str, Enum):
    """Types of memories in the learning system."""
    EPISODIC = "episodic"      # Events/interactions - "User asked about X on date Y"
    SEMANTIC = "semantic"       # Facts/knowledge - "MarunochiAI runs on M4 Pro"
    PROCEDURAL = "procedural"   # How-to - "To deploy, run docker-compose up"
    AGENT_CONTEXT = "agent"     # Cross-agent state - "DottscavisAI is rendering"
    EXPERIENCE = "experience"   # Trajectories for learning
    ARCHITECTURE = "architecture"  # System design decisions
    USER_PREFERENCE = "preference"  # User preferences and settings


@dataclass
class Memory:
    """A single memory entry."""
    id: Optional[int]
    content: str
    memory_type: MemoryType
    category: str  # Sub-category within type
    importance: int  # 1-5 scale
    source: str  # Where this came from (user, agent, system)
    metadata: Dict[str, Any]  # Additional structured data
    created_at: datetime
    accessed_at: datetime
    access_count: int
    embedding_hash: Optional[str]  # For deduplication


class EnhancedMemoryManager:
    """
    Enhanced SQLite memory manager with:
    - Typed memories (episodic, semantic, procedural, agent_context)
    - Importance decay over time
    - Memory consolidation
    - Cross-agent sharing via API
    - Automatic categorization hints
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False
        self._consolidation_running = False

        # Memory importance decay settings
        self.decay_rate = 0.1  # Reduce importance by 10% per week for episodic
        self.consolidation_threshold = 100  # Consolidate when over this many old memories

    async def _optimize_connection(self, db):
        """Apply performance optimizations to SQLite connection."""
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        await db.execute("PRAGMA cache_size = -80000")
        await db.execute("PRAGMA temp_store = MEMORY")
        await db.execute("PRAGMA mmap_size = 1073741824")
        await db.execute("PRAGMA wal_autocheckpoint = 1000")
        await db.execute("PRAGMA foreign_keys = ON")

    async def initialize(self):
        """Initialize the enhanced memory database schema."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Enhanced memories table with types
            await db.execute('''
                CREATE TABLE IF NOT EXISTS memories_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'semantic',
                    category TEXT DEFAULT 'general',
                    importance INTEGER DEFAULT 3,
                    source TEXT DEFAULT 'system',
                    metadata JSON DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    embedding_hash TEXT,
                    consolidated_into INTEGER,
                    FOREIGN KEY (consolidated_into) REFERENCES memories_v2(id)
                )
            ''')

            # Indexes for fast queries
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_mem_type
                ON memories_v2(memory_type, importance DESC, accessed_at DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_mem_category
                ON memories_v2(category, memory_type)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_mem_source
                ON memories_v2(source)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_mem_hash
                ON memories_v2(embedding_hash)
            ''')

            # FTS5 for full-text search
            await db.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_v2_fts USING fts5(
                    content,
                    category,
                    memory_type,
                    source,
                    tokenize='porter unicode61'
                )
            ''')

            # Triggers for FTS sync
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS mem_v2_ai AFTER INSERT ON memories_v2 BEGIN
                    INSERT INTO memories_v2_fts(rowid, content, category, memory_type, source)
                    VALUES (new.id, new.content, new.category, new.memory_type, new.source);
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS mem_v2_ad AFTER DELETE ON memories_v2 BEGIN
                    DELETE FROM memories_v2_fts WHERE rowid = old.id;
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS mem_v2_au AFTER UPDATE ON memories_v2 BEGIN
                    DELETE FROM memories_v2_fts WHERE rowid = old.id;
                    INSERT INTO memories_v2_fts(rowid, content, category, memory_type, source)
                    VALUES (new.id, new.content, new.category, new.memory_type, new.source);
                END
            ''')

            # Agent registry table for multi-agent coordination
            await db.execute('''
                CREATE TABLE IF NOT EXISTS agent_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    capabilities JSON DEFAULT '[]',
                    endpoint TEXT,
                    status TEXT DEFAULT 'offline',
                    last_seen TIMESTAMP,
                    metadata JSON DEFAULT '{}'
                )
            ''')

            # Memory consolidation log
            await db.execute('''
                CREATE TABLE IF NOT EXISTS consolidation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consolidated_ids JSON NOT NULL,
                    summary_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (summary_id) REFERENCES memories_v2(id)
                )
            ''')

            await db.execute("ANALYZE")
            await db.commit()

        self._initialized = True
        print("[MEMORY-V2] Enhanced memory system initialized")

    def _compute_hash(self, content: str) -> str:
        """Compute hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        category: str = "general",
        importance: int = 3,
        source: str = "system",
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store a new memory with type categorization.

        Args:
            content: The memory content
            memory_type: Type of memory (episodic, semantic, procedural, agent)
            category: Sub-category for organization
            importance: 1-5 scale (5 = most important)
            source: Where this came from (user, benchai, marunochiAI, etc.)
            metadata: Additional structured data

        Returns:
            The ID of the stored memory
        """
        await self.initialize()

        content_hash = self._compute_hash(content)
        metadata = metadata or {}

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Check for duplicates
            cursor = await db.execute(
                "SELECT id FROM memories_v2 WHERE embedding_hash = ?",
                (content_hash,)
            )
            existing = await cursor.fetchone()
            if existing:
                # Update access count instead of duplicating
                await db.execute(
                    "UPDATE memories_v2 SET access_count = access_count + 1, accessed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (existing[0],)
                )
                await db.commit()
                return existing[0]

            cursor = await db.execute(
                '''INSERT INTO memories_v2
                   (content, memory_type, category, importance, source, metadata, embedding_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (content, memory_type.value, category, importance, source, json.dumps(metadata), content_hash)
            )
            await db.commit()
            return cursor.lastrowid

    async def search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        min_importance: int = 1,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search memories with filtering by type, category, and source.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            # Build query with filters
            conditions = ["memories_v2_fts MATCH ?"]
            params = [query]

            if memory_types:
                placeholders = ",".join("?" * len(memory_types))
                conditions.append(f"m.memory_type IN ({placeholders})")
                params.extend([t.value for t in memory_types])

            if category:
                conditions.append("m.category = ?")
                params.append(category)

            if source:
                conditions.append("m.source = ?")
                params.append(source)

            conditions.append("m.importance >= ?")
            params.append(min_importance)

            conditions.append("m.consolidated_into IS NULL")  # Don't return consolidated memories

            where_clause = " AND ".join(conditions)
            params.append(limit)

            try:
                cursor = await db.execute(f'''
                    SELECT m.*, bm25(memories_v2_fts) as rank
                    FROM memories_v2 m
                    JOIN memories_v2_fts ON m.id = memories_v2_fts.rowid
                    WHERE {where_clause}
                    ORDER BY rank, m.importance DESC, m.accessed_at DESC
                    LIMIT ?
                ''', params)

                results = []
                async for row in cursor:
                    # Update access tracking
                    await db.execute(
                        "UPDATE memories_v2 SET access_count = access_count + 1, accessed_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (row['id'],)
                    )
                    results.append({
                        "id": row['id'],
                        "content": row['content'],
                        "memory_type": row['memory_type'],
                        "category": row['category'],
                        "importance": row['importance'],
                        "source": row['source'],
                        "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                        "created_at": row['created_at'],
                        "accessed_at": row['accessed_at'],
                        "access_count": row['access_count'],
                        "relevance": -row['rank']  # Higher is better
                    })

                await db.commit()
                return results

            except Exception as e:
                # Fallback to LIKE search if FTS fails
                print(f"[MEMORY] FTS search failed, falling back to LIKE: {e}")
                cursor = await db.execute(
                    '''SELECT * FROM memories_v2
                       WHERE content LIKE ? AND consolidated_into IS NULL
                       ORDER BY importance DESC, accessed_at DESC
                       LIMIT ?''',
                    (f"%{query}%", limit)
                )
                results = []
                async for row in cursor:
                    results.append(dict(row))
                return results

    async def get_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 20,
        min_importance: int = 1
    ) -> List[Dict]:
        """Get recent memories of a specific type."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                '''SELECT * FROM memories_v2
                   WHERE memory_type = ? AND importance >= ? AND consolidated_into IS NULL
                   ORDER BY importance DESC, accessed_at DESC
                   LIMIT ?''',
                (memory_type.value, min_importance, limit)
            )

            return [dict(row) async for row in cursor]

    async def get_agent_context(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get current agent context for coordination."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            if agent_id:
                cursor = await db.execute(
                    '''SELECT * FROM memories_v2
                       WHERE memory_type = 'agent' AND source = ?
                       ORDER BY created_at DESC LIMIT 10''',
                    (agent_id,)
                )
            else:
                cursor = await db.execute(
                    '''SELECT * FROM memories_v2
                       WHERE memory_type = 'agent'
                       ORDER BY created_at DESC LIMIT 20'''
                )

            return [dict(row) async for row in cursor]

    async def register_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        capabilities: List[str],
        endpoint: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Register an agent in the multi-agent system."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''INSERT OR REPLACE INTO agent_registry
                   (id, name, role, capabilities, endpoint, status, last_seen, metadata)
                   VALUES (?, ?, ?, ?, ?, 'online', CURRENT_TIMESTAMP, ?)''',
                (agent_id, name, role, json.dumps(capabilities), endpoint, json.dumps(metadata or {}))
            )
            await db.commit()

        # Store as memory too
        await self.store(
            content=f"Agent registered: {name} ({agent_id}) - Role: {role}, Capabilities: {', '.join(capabilities)}",
            memory_type=MemoryType.AGENT_CONTEXT,
            category="registration",
            importance=4,
            source=agent_id,
            metadata={"agent_id": agent_id, "capabilities": capabilities, "endpoint": endpoint}
        )

    async def get_agents(self, status: Optional[str] = None) -> List[Dict]:
        """Get registered agents, optionally filtered by status."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if status:
                cursor = await db.execute(
                    "SELECT * FROM agent_registry WHERE status = ?",
                    (status,)
                )
            else:
                cursor = await db.execute("SELECT * FROM agent_registry")

            agents = []
            async for row in cursor:
                agent = dict(row)
                agent['capabilities'] = json.loads(agent['capabilities'])
                agent['metadata'] = json.loads(agent['metadata'])
                agents.append(agent)

            return agents

    async def update_agent_status(self, agent_id: str, status: str):
        """Update an agent's status (online/offline/busy)."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE agent_registry SET status = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?",
                (status, agent_id)
            )
            await db.commit()

    async def consolidate_old_memories(self, days_old: int = 30, batch_size: int = 20):
        """
        Consolidate old episodic memories into summaries to save space.
        This implements memory consolidation like human sleep does.
        """
        if self._consolidation_running:
            return {"status": "already_running"}

        self._consolidation_running = True

        try:
            await self.initialize()
            cutoff = datetime.now() - timedelta(days=days_old)
            consolidated_count = 0

            async with aiosqlite.connect(self.db_path) as db:
                await self._optimize_connection(db)
                db.row_factory = aiosqlite.Row

                # Get old episodic memories that haven't been consolidated
                cursor = await db.execute(
                    '''SELECT * FROM memories_v2
                       WHERE memory_type = 'episodic'
                       AND created_at < ?
                       AND consolidated_into IS NULL
                       AND importance < 4
                       ORDER BY created_at ASC
                       LIMIT ?''',
                    (cutoff.isoformat(), batch_size)
                )

                old_memories = [dict(row) async for row in cursor]

                if len(old_memories) >= 5:
                    # Group by category and summarize
                    by_category = {}
                    for mem in old_memories:
                        cat = mem['category']
                        if cat not in by_category:
                            by_category[cat] = []
                        by_category[cat].append(mem)

                    for cat, mems in by_category.items():
                        if len(mems) >= 3:
                            # Create summary
                            contents = [m['content'][:200] for m in mems]
                            summary = f"[Consolidated {len(mems)} memories from {cat}]: " + " | ".join(contents[:5])

                            if len(contents) > 5:
                                summary += f" ... and {len(contents) - 5} more"

                            # Store summary
                            cursor = await db.execute(
                                '''INSERT INTO memories_v2
                                   (content, memory_type, category, importance, source, metadata)
                                   VALUES (?, 'semantic', ?, 3, 'consolidation', ?)''',
                                (summary, cat, json.dumps({"consolidated_count": len(mems)}))
                            )
                            summary_id = cursor.lastrowid

                            # Mark originals as consolidated
                            ids = [m['id'] for m in mems]
                            placeholders = ",".join("?" * len(ids))
                            await db.execute(
                                f"UPDATE memories_v2 SET consolidated_into = ? WHERE id IN ({placeholders})",
                                [summary_id] + ids
                            )

                            # Log consolidation
                            await db.execute(
                                "INSERT INTO consolidation_log (consolidated_ids, summary_id) VALUES (?, ?)",
                                (json.dumps(ids), summary_id)
                            )

                            consolidated_count += len(mems)

                await db.commit()

            return {
                "status": "completed",
                "consolidated_count": consolidated_count
            }

        finally:
            self._consolidation_running = False

    async def apply_importance_decay(self):
        """
        Apply importance decay to old episodic memories.
        This helps prioritize recent, frequently accessed information.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Decay episodic memories older than 7 days that haven't been accessed
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()

            await db.execute(
                '''UPDATE memories_v2
                   SET importance = MAX(1, importance - 1)
                   WHERE memory_type = 'episodic'
                   AND accessed_at < ?
                   AND importance > 1
                   AND access_count < 3''',
                (week_ago,)
            )

            # Boost frequently accessed memories
            await db.execute(
                '''UPDATE memories_v2
                   SET importance = MIN(5, importance + 1)
                   WHERE access_count >= 10
                   AND importance < 5'''
            )

            await db.commit()

    async def get_stats(self) -> Dict:
        """Get detailed statistics about the memory system."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            stats = {"memory_types": {}, "categories": {}, "sources": {}}

            # Count by memory type
            cursor = await db.execute(
                '''SELECT memory_type, COUNT(*) as count, AVG(importance) as avg_importance
                   FROM memories_v2 WHERE consolidated_into IS NULL
                   GROUP BY memory_type'''
            )
            async for row in cursor:
                stats["memory_types"][row['memory_type']] = {
                    "count": row['count'],
                    "avg_importance": round(row['avg_importance'], 2)
                }

            # Count by category
            cursor = await db.execute(
                '''SELECT category, COUNT(*) as count
                   FROM memories_v2 WHERE consolidated_into IS NULL
                   GROUP BY category ORDER BY count DESC LIMIT 10'''
            )
            async for row in cursor:
                stats["categories"][row['category']] = row['count']

            # Count by source
            cursor = await db.execute(
                '''SELECT source, COUNT(*) as count
                   FROM memories_v2 WHERE consolidated_into IS NULL
                   GROUP BY source'''
            )
            async for row in cursor:
                stats["sources"][row['source']] = row['count']

            # Total counts
            cursor = await db.execute("SELECT COUNT(*) FROM memories_v2")
            stats["total_memories"] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM memories_v2 WHERE consolidated_into IS NOT NULL")
            stats["consolidated_memories"] = (await cursor.fetchone())[0]

            # Agent count
            cursor = await db.execute("SELECT COUNT(*) FROM agent_registry")
            stats["registered_agents"] = (await cursor.fetchone())[0]

            return stats
