"""
Zettelkasten Knowledge Graph System for BenchAI

Based on:
- A-MEM: Agentic Memory for LLM Agents (NeurIPS 2025)
- Graphiti: Temporal Knowledge Graph Memory
- Wake-Sleep Memory Consolidation Architecture

This implements a self-organizing knowledge graph where:
1. Atomic notes (Zettels) capture single units of knowledge
2. Links emerge organically through semantic similarity + LLM analysis
3. Notes evolve as new connections form
4. Sleep consolidation compresses and strengthens important memories
5. Agents can query asynchronously for deep knowledge retrieval

The system serves as BenchAI's "second brain" - a repository of connected
knowledge that other agents can query while working in parallel.
"""

import asyncio
import aiosqlite
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field, asdict
import heapq


class ZettelType(str, Enum):
    """Types of Zettels in the knowledge graph."""
    FLEETING = "fleeting"      # Quick captures, to be processed
    LITERATURE = "literature"   # From external sources (research, docs)
    PERMANENT = "permanent"     # Refined, atomic knowledge
    HUB = "hub"                # Index/entry point connecting many notes
    STRUCTURE = "structure"    # Organizational notes
    PROJECT = "project"        # Related to specific projects/tasks


class LinkType(str, Enum):
    """Types of connections between Zettels."""
    RELATES_TO = "relates_to"       # General relation
    SUPPORTS = "supports"           # Evidence/support
    CONTRADICTS = "contradicts"     # Conflicting information
    EXTENDS = "extends"             # Builds upon
    EXAMPLE_OF = "example_of"       # Instance of a concept
    CAUSED_BY = "caused_by"         # Causal relationship
    PART_OF = "part_of"            # Component relationship
    SEQUENCE = "sequence"           # Temporal/logical order


@dataclass
class Zettel:
    """
    An atomic note in the knowledge graph.

    Following Zettelkasten principles:
    - Atomicity: One idea per note
    - Unique ID: Permanent identifier
    - Self-contained: Understandable without context
    - Linkable: Connected to related notes
    """
    id: Optional[str]  # Unique identifier (not auto-increment, semantic ID)
    content: str       # The actual knowledge (300-500 words max)
    title: str         # Brief title
    zettel_type: ZettelType
    keywords: List[str]  # LLM-extracted keywords
    tags: List[str]      # User/system tags
    context: str         # LLM-generated context description
    source: str          # Where this came from
    source_agent: str    # Which agent created this

    # Embedding and linking
    embedding: Optional[List[float]] = None
    links: List[Dict[str, str]] = field(default_factory=list)  # [{target_id, link_type, reason}]
    backlinks: List[str] = field(default_factory=list)  # IDs that link to this

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)

    # Importance and evolution
    access_count: int = 0
    importance: float = 0.5  # 0-1, evolves over time
    maturity: int = 0  # 0=fleeting, increases as note is refined

    # Consolidation
    consolidated_from: List[str] = field(default_factory=list)
    is_consolidated: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)


class ZettelkastenKnowledgeGraph:
    """
    A self-organizing knowledge graph implementing Zettelkasten principles.

    Features:
    - Atomic note creation with automatic keyword/tag extraction
    - Semantic similarity-based link discovery
    - LLM-driven link analysis for nuanced connections
    - Memory evolution as new notes are added
    - Wake-Sleep consolidation cycle
    - Asynchronous query API for parallel agent access
    """

    def __init__(self, db_path: Path, embedding_model: Optional[Any] = None):
        self.db_path = db_path
        self.embedding_model = embedding_model  # For vector similarity
        self._initialized = False
        self._consolidation_running = False

        # Configuration
        self.similarity_threshold = 0.7  # For automatic linking
        self.max_links_per_note = 20
        self.consolidation_age_days = 14
        self.importance_decay_rate = 0.02

        # Caches
        self._embedding_cache: Dict[str, List[float]] = {}
        self._graph_cache: Optional[Dict[str, Set[str]]] = None
        self._hub_cache: List[str] = []

    async def _optimize_connection(self, db):
        """Apply SQLite optimizations."""
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        await db.execute("PRAGMA cache_size = -80000")
        await db.execute("PRAGMA temp_store = MEMORY")
        await db.execute("PRAGMA mmap_size = 1073741824")

    async def initialize(self):
        """Initialize the knowledge graph database."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Main Zettels table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS zettels (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    title TEXT NOT NULL,
                    zettel_type TEXT DEFAULT 'permanent',
                    keywords JSON DEFAULT '[]',
                    tags JSON DEFAULT '[]',
                    context TEXT,
                    source TEXT,
                    source_agent TEXT DEFAULT 'benchai',
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    maturity INTEGER DEFAULT 0,
                    consolidated_from JSON DEFAULT '[]',
                    is_consolidated INTEGER DEFAULT 0,
                    metadata JSON DEFAULT '{}'
                )
            ''')

            # Links table (graph edges)
            await db.execute('''
                CREATE TABLE IF NOT EXISTS zettel_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    link_type TEXT DEFAULT 'relates_to',
                    reason TEXT,
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES zettels(id),
                    FOREIGN KEY (target_id) REFERENCES zettels(id),
                    UNIQUE(source_id, target_id, link_type)
                )
            ''')

            # Indexes
            await db.execute('CREATE INDEX IF NOT EXISTS idx_zettels_type ON zettels(zettel_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_zettels_importance ON zettels(importance DESC)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_zettels_accessed ON zettels(accessed_at DESC)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_links_source ON zettel_links(source_id)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_links_target ON zettel_links(target_id)')

            # FTS5 for full-text search
            await db.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS zettels_fts USING fts5(
                    id,
                    title,
                    content,
                    context,
                    keywords,
                    tags,
                    tokenize='porter unicode61'
                )
            ''')

            # FTS triggers
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS zettels_ai AFTER INSERT ON zettels BEGIN
                    INSERT INTO zettels_fts(id, title, content, context, keywords, tags)
                    VALUES (new.id, new.title, new.content, new.context,
                            new.keywords, new.tags);
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS zettels_ad AFTER DELETE ON zettels BEGIN
                    DELETE FROM zettels_fts WHERE id = old.id;
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS zettels_au AFTER UPDATE ON zettels BEGIN
                    DELETE FROM zettels_fts WHERE id = old.id;
                    INSERT INTO zettels_fts(id, title, content, context, keywords, tags)
                    VALUES (new.id, new.title, new.content, new.context,
                            new.keywords, new.tags);
                END
            ''')

            # Consolidation log
            await db.execute('''
                CREATE TABLE IF NOT EXISTS consolidation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_type TEXT,
                    zettels_processed INTEGER,
                    links_created INTEGER,
                    notes_consolidated INTEGER,
                    duration_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Query cache for agent requests
            await db.execute('''
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    result JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')

            await db.execute("ANALYZE")
            await db.commit()

        self._initialized = True
        print("[ZETTELKASTEN] Knowledge graph initialized")

    def _generate_id(self, content: str, title: str) -> str:
        """Generate a unique, semi-semantic ID for a Zettel."""
        # Format: YYYYMMDD-HHMMSS-hash[:8]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        content_hash = hashlib.sha256(f"{title}:{content}".encode()).hexdigest()[:8]
        return f"{timestamp}-{content_hash}"

    def _simple_embedding(self, text: str) -> List[float]:
        """
        Simple embedding using character n-grams.
        Replace with proper embedding model in production.
        """
        # Create a simple 256-dim embedding based on character trigrams
        text = text.lower()[:1000]
        embedding = [0.0] * 256

        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(trigram) % 256
            embedding[idx] += 1

        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(x*x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def create_zettel(
        self,
        content: str,
        title: str,
        zettel_type: ZettelType = ZettelType.PERMANENT,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        source: str = "user",
        source_agent: str = "benchai",
        auto_link: bool = True,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new Zettel (atomic note) in the knowledge graph.

        Args:
            content: The knowledge content (aim for 300-500 words)
            title: Brief descriptive title
            zettel_type: Type of note (fleeting, literature, permanent, hub)
            keywords: Key terms (auto-extracted if not provided)
            tags: Categorization tags
            context: Contextual description (auto-generated if not provided)
            source: Origin of the knowledge
            source_agent: Which agent created this
            auto_link: Whether to automatically discover and create links
            metadata: Additional structured data

        Returns:
            The unique ID of the created Zettel
        """
        await self.initialize()

        zettel_id = self._generate_id(content, title)

        # Auto-extract keywords if not provided
        if keywords is None:
            keywords = self._extract_keywords(content)

        # Auto-generate context if not provided
        if context is None:
            context = self._generate_context(content, title)

        tags = tags or []
        metadata = metadata or {}

        # Generate embedding
        embedding = self._simple_embedding(f"{title} {content} {context}")
        embedding_bytes = json.dumps(embedding).encode()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            await db.execute('''
                INSERT INTO zettels
                (id, content, title, zettel_type, keywords, tags, context,
                 source, source_agent, embedding, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                zettel_id, content, title, zettel_type.value,
                json.dumps(keywords), json.dumps(tags), context,
                source, source_agent, embedding_bytes,
                0.5 if zettel_type == ZettelType.PERMANENT else 0.3,
                json.dumps(metadata)
            ))
            await db.commit()

        # Cache embedding
        self._embedding_cache[zettel_id] = embedding
        self._graph_cache = None  # Invalidate

        # Discover and create links
        if auto_link:
            await self._discover_links(zettel_id, embedding)

        print(f"[ZETTELKASTEN] Created Zettel: {zettel_id} ({title})")
        return zettel_id

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content (simple implementation)."""
        # In production, use LLM or NLP library
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                      'during', 'before', 'after', 'above', 'below', 'between',
                      'under', 'again', 'further', 'then', 'once', 'and', 'but',
                      'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not',
                      'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
                      'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
                      'them', 'we', 'our', 'you', 'your', 'he', 'she', 'him', 'her'}

        words = content.lower().split()
        word_freq = {}
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top keywords by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:10]]

    def _generate_context(self, content: str, title: str) -> str:
        """Generate contextual description (simple implementation)."""
        # In production, use LLM
        sentences = content.split('.')
        first_sentence = sentences[0].strip() if sentences else ""
        return f"Note about {title}. {first_sentence}."[:200]

    async def _discover_links(self, zettel_id: str, embedding: List[float]):
        """
        Discover and create links to related Zettels.

        Uses embedding similarity for candidate discovery,
        then semantic analysis for link type determination.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            # Get all other Zettels with embeddings
            cursor = await db.execute(
                "SELECT id, title, embedding FROM zettels WHERE id != ?",
                (zettel_id,)
            )

            candidates = []
            async for row in cursor:
                if row['embedding']:
                    other_embedding = json.loads(row['embedding'])
                    similarity = self._cosine_similarity(embedding, other_embedding)
                    if similarity >= self.similarity_threshold:
                        candidates.append({
                            'id': row['id'],
                            'title': row['title'],
                            'similarity': similarity
                        })

            # Sort by similarity and take top candidates
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            candidates = candidates[:self.max_links_per_note]

            # Create links
            for candidate in candidates:
                link_type = self._infer_link_type(candidate['similarity'])
                await db.execute('''
                    INSERT OR IGNORE INTO zettel_links
                    (source_id, target_id, link_type, strength, reason)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    zettel_id, candidate['id'], link_type.value,
                    candidate['similarity'],
                    f"Auto-linked: {candidate['similarity']:.2f} similarity"
                ))

            await db.commit()

            if candidates:
                print(f"[ZETTELKASTEN] Created {len(candidates)} links for {zettel_id}")

    def _infer_link_type(self, similarity: float) -> LinkType:
        """Infer link type based on similarity (simplified)."""
        if similarity > 0.9:
            return LinkType.EXTENDS
        elif similarity > 0.8:
            return LinkType.RELATES_TO
        else:
            return LinkType.RELATES_TO

    async def search(
        self,
        query: str,
        limit: int = 10,
        zettel_types: Optional[List[ZettelType]] = None,
        include_links: bool = True,
        expand_graph: bool = True,
        graph_depth: int = 1
    ) -> List[Dict]:
        """
        Search the knowledge graph with optional graph expansion.

        This is the primary query interface for agents.

        Args:
            query: Search query
            limit: Maximum results
            zettel_types: Filter by type
            include_links: Include linked notes in results
            expand_graph: Traverse graph to find connected knowledge
            graph_depth: How many hops to traverse

        Returns:
            List of relevant Zettels with their connections
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            # Full-text search
            try:
                cursor = await db.execute('''
                    SELECT z.*, bm25(zettels_fts) as rank
                    FROM zettels z
                    JOIN zettels_fts ON z.id = zettels_fts.id
                    WHERE zettels_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (query, limit))
            except:
                # Fallback to LIKE search
                cursor = await db.execute('''
                    SELECT *, 0 as rank FROM zettels
                    WHERE content LIKE ? OR title LIKE ? OR context LIKE ?
                    ORDER BY importance DESC
                    LIMIT ?
                ''', (f"%{query}%", f"%{query}%", f"%{query}%", limit))

            results = []
            seen_ids = set()

            async for row in cursor:
                zettel = self._row_to_dict(row)
                zettel['_search_rank'] = -row['rank'] if 'rank' in row.keys() else 0
                results.append(zettel)
                seen_ids.add(row['id'])

                # Update access tracking
                await db.execute('''
                    UPDATE zettels SET
                        access_count = access_count + 1,
                        accessed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (row['id'],))

            # Expand graph if requested
            if expand_graph and results:
                for depth in range(graph_depth):
                    new_ids = set()
                    for zettel in list(results):
                        if zettel['id'] in seen_ids:
                            # Get linked Zettels
                            cursor = await db.execute('''
                                SELECT z.* FROM zettels z
                                JOIN zettel_links l ON z.id = l.target_id
                                WHERE l.source_id = ?
                                LIMIT 5
                            ''', (zettel['id'],))

                            async for linked_row in cursor:
                                if linked_row['id'] not in seen_ids:
                                    linked = self._row_to_dict(linked_row)
                                    linked['_linked_from'] = zettel['id']
                                    linked['_link_depth'] = depth + 1
                                    results.append(linked)
                                    new_ids.add(linked_row['id'])

                    seen_ids.update(new_ids)

            # Include link information
            if include_links:
                for zettel in results:
                    cursor = await db.execute('''
                        SELECT target_id, link_type, reason, strength
                        FROM zettel_links WHERE source_id = ?
                    ''', (zettel['id'],))
                    zettel['links'] = [dict(row) async for row in cursor]

                    cursor = await db.execute('''
                        SELECT source_id FROM zettel_links WHERE target_id = ?
                    ''', (zettel['id'],))
                    zettel['backlinks'] = [row['source_id'] async for row in cursor]

            await db.commit()

            return results

    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        d = dict(row)
        for field in ['keywords', 'tags', 'consolidated_from', 'metadata']:
            if field in d and d[field]:
                try:
                    d[field] = json.loads(d[field])
                except:
                    pass
        if 'embedding' in d:
            del d['embedding']  # Don't return raw embedding
        return d

    async def get_connected_knowledge(
        self,
        zettel_id: str,
        max_depth: int = 2,
        max_nodes: int = 50
    ) -> Dict:
        """
        Get all knowledge connected to a Zettel via graph traversal.

        Uses BFS to explore the knowledge graph, returning a subgraph
        of connected notes.
        """
        await self.initialize()

        visited = set()
        to_visit = [(zettel_id, 0)]
        nodes = []
        edges = []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            while to_visit and len(nodes) < max_nodes:
                current_id, depth = to_visit.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)

                # Get node
                cursor = await db.execute(
                    "SELECT * FROM zettels WHERE id = ?",
                    (current_id,)
                )
                row = await cursor.fetchone()
                if row:
                    nodes.append({
                        'id': row['id'],
                        'title': row['title'],
                        'type': row['zettel_type'],
                        'importance': row['importance'],
                        'depth': depth
                    })

                    # Get outgoing links
                    cursor = await db.execute(
                        "SELECT * FROM zettel_links WHERE source_id = ?",
                        (current_id,)
                    )
                    async for link in cursor:
                        edges.append({
                            'source': link['source_id'],
                            'target': link['target_id'],
                            'type': link['link_type'],
                            'strength': link['strength']
                        })
                        if link['target_id'] not in visited:
                            to_visit.append((link['target_id'], depth + 1))

        return {
            'root': zettel_id,
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }

    async def find_hubs(self, limit: int = 10) -> List[Dict]:
        """
        Find hub notes - highly connected knowledge nodes.

        These are entry points into the knowledge graph.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute('''
                SELECT z.id, z.title, z.zettel_type, z.importance,
                       COUNT(DISTINCT l1.target_id) as outlinks,
                       COUNT(DISTINCT l2.source_id) as inlinks
                FROM zettels z
                LEFT JOIN zettel_links l1 ON z.id = l1.source_id
                LEFT JOIN zettel_links l2 ON z.id = l2.target_id
                GROUP BY z.id
                ORDER BY (outlinks + inlinks) DESC
                LIMIT ?
            ''', (limit,))

            return [dict(row) async for row in cursor]

    async def sleep_consolidation(self) -> Dict:
        """
        Run sleep consolidation cycle.

        This is inspired by how the brain consolidates memories during sleep:
        1. Strengthen frequently accessed connections
        2. Weaken rarely used connections
        3. Consolidate similar fleeting notes into permanent notes
        4. Update importance scores
        5. Prune low-importance, unconnected notes
        """
        if self._consolidation_running:
            return {"status": "already_running"}

        self._consolidation_running = True
        start_time = datetime.now()

        try:
            await self.initialize()

            stats = {
                "links_strengthened": 0,
                "links_weakened": 0,
                "notes_consolidated": 0,
                "importance_updated": 0,
                "notes_pruned": 0
            }

            async with aiosqlite.connect(self.db_path) as db:
                await self._optimize_connection(db)

                # 1. Strengthen frequently co-accessed links
                await db.execute('''
                    UPDATE zettel_links SET strength = MIN(1.0, strength + 0.1)
                    WHERE source_id IN (
                        SELECT id FROM zettels WHERE access_count > 5
                    ) AND target_id IN (
                        SELECT id FROM zettels WHERE access_count > 5
                    )
                ''')
                stats["links_strengthened"] = db.total_changes

                # 2. Weaken old, unused links
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                await db.execute('''
                    UPDATE zettel_links SET strength = MAX(0.1, strength - 0.05)
                    WHERE created_at < ?
                    AND source_id IN (
                        SELECT id FROM zettels WHERE accessed_at < ?
                    )
                ''', (week_ago, week_ago))
                stats["links_weakened"] = db.total_changes

                # 3. Update importance based on connections and access
                await db.execute('''
                    UPDATE zettels SET importance = MIN(1.0, (
                        0.3 * importance +
                        0.3 * MIN(1.0, access_count / 20.0) +
                        0.4 * (
                            SELECT COALESCE(COUNT(*), 0) / 10.0
                            FROM zettel_links
                            WHERE source_id = zettels.id OR target_id = zettels.id
                        )
                    ))
                ''')
                stats["importance_updated"] = db.total_changes

                # 4. Consolidate old fleeting notes
                month_ago = (datetime.now() - timedelta(days=30)).isoformat()
                cursor = await db.execute('''
                    SELECT * FROM zettels
                    WHERE zettel_type = 'fleeting'
                    AND created_at < ?
                    AND access_count < 3
                    LIMIT 10
                ''', (month_ago,))

                fleeting_notes = [dict(row) async for row in cursor]
                if len(fleeting_notes) >= 3:
                    # Consolidate into a single note
                    combined_content = "\n\n".join([
                        f"## {n['title']}\n{n['content']}"
                        for n in fleeting_notes
                    ])
                    consolidated_id = await self.create_zettel(
                        content=combined_content[:2000],
                        title=f"Consolidated notes ({len(fleeting_notes)} items)",
                        zettel_type=ZettelType.PERMANENT,
                        source="consolidation",
                        auto_link=True
                    )

                    # Mark originals as consolidated
                    for note in fleeting_notes:
                        await db.execute('''
                            UPDATE zettels SET is_consolidated = 1
                            WHERE id = ?
                        ''', (note['id'],))

                    stats["notes_consolidated"] = len(fleeting_notes)

                # 5. Apply importance decay
                await db.execute('''
                    UPDATE zettels SET importance = MAX(0.1, importance - ?)
                    WHERE accessed_at < ?
                    AND importance > 0.2
                ''', (self.importance_decay_rate, week_ago))

                await db.commit()

                # Log consolidation
                duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                await db.execute('''
                    INSERT INTO consolidation_log
                    (cycle_type, zettels_processed, links_created, notes_consolidated, duration_ms)
                    VALUES ('sleep', ?, ?, ?, ?)
                ''', (
                    stats["importance_updated"],
                    stats["links_strengthened"],
                    stats["notes_consolidated"],
                    duration_ms
                ))
                await db.commit()

            self._graph_cache = None  # Invalidate cache
            print(f"[ZETTELKASTEN] Sleep consolidation complete: {stats}")
            return stats

        finally:
            self._consolidation_running = False

    async def get_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            stats = {}

            # Total Zettels by type
            cursor = await db.execute('''
                SELECT zettel_type, COUNT(*) as count, AVG(importance) as avg_importance
                FROM zettels WHERE is_consolidated = 0
                GROUP BY zettel_type
            ''')
            stats["by_type"] = {row['zettel_type']: {
                'count': row['count'],
                'avg_importance': round(row['avg_importance'] or 0, 2)
            } async for row in cursor}

            # Total links by type
            cursor = await db.execute('''
                SELECT link_type, COUNT(*) as count, AVG(strength) as avg_strength
                FROM zettel_links
                GROUP BY link_type
            ''')
            stats["links_by_type"] = {row['link_type']: {
                'count': row['count'],
                'avg_strength': round(row['avg_strength'] or 0, 2)
            } async for row in cursor}

            # Overall counts
            cursor = await db.execute("SELECT COUNT(*) FROM zettels WHERE is_consolidated = 0")
            stats["total_zettels"] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM zettel_links")
            stats["total_links"] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM zettels WHERE is_consolidated = 1")
            stats["consolidated_zettels"] = (await cursor.fetchone())[0]

            # Graph density
            if stats["total_zettels"] > 1:
                max_edges = stats["total_zettels"] * (stats["total_zettels"] - 1)
                stats["graph_density"] = round(stats["total_links"] / max_edges, 4)
            else:
                stats["graph_density"] = 0

            # Most connected
            cursor = await db.execute('''
                SELECT z.title, COUNT(*) as connections
                FROM zettels z
                JOIN zettel_links l ON z.id = l.source_id OR z.id = l.target_id
                WHERE z.is_consolidated = 0
                GROUP BY z.id
                ORDER BY connections DESC
                LIMIT 5
            ''')
            stats["top_connected"] = [
                {'title': row['title'], 'connections': row['connections']}
                async for row in cursor
            ]

            return stats
