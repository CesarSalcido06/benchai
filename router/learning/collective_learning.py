"""
Collective Learning Pipeline for Multi-Agent System

Enables cross-agent experience sharing and collective model improvement:
- Experience aggregation from all agents
- Collective success pattern identification
- Cross-agent knowledge transfer
- Federated learning coordination
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import aiosqlite


class LearningContributionType(Enum):
    """Types of learning contributions from agents."""
    EXPERIENCE = "experience"          # Successful task experience
    KNOWLEDGE = "knowledge"            # Knowledge/facts discovered
    PATTERN = "pattern"                # Identified pattern or strategy
    CORRECTION = "correction"          # Error correction or fix
    FEEDBACK = "feedback"              # User or peer feedback


@dataclass
class LearningContribution:
    """A learning contribution from an agent."""
    id: str
    agent_id: str
    contribution_type: LearningContributionType
    content: str
    domain: str
    quality_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    aggregated: bool = False


@dataclass
class CollectiveInsight:
    """An insight derived from collective learning."""
    id: str
    insight_type: str  # pattern, best_practice, common_error, strategy
    description: str
    domains: List[str]
    contributing_agents: List[str]
    confidence: float
    evidence_count: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CollectiveLearningPipeline:
    """
    Coordinates collective learning across all agents.

    Features:
    - Experience aggregation from MarunochiAI, DottscavisAI, BenchAI
    - Pattern identification across agent experiences
    - Collective knowledge graph enrichment
    - Federated improvement coordination
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.db_path = storage_dir / "collective_learning.db"
        self._initialized = False
        self._aggregation_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the collective learning database."""
        if self._initialized:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Contributions table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS contributions (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    contribution_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.5,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    aggregated INTEGER DEFAULT 0,
                    aggregated_into TEXT
                )
            ''')

            # Collective insights table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    insight_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    domains TEXT NOT NULL,
                    contributing_agents TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    active INTEGER DEFAULT 1
                )
            ''')

            # Agent learning stats
            await db.execute('''
                CREATE TABLE IF NOT EXISTS agent_learning_stats (
                    agent_id TEXT PRIMARY KEY,
                    total_contributions INTEGER DEFAULT 0,
                    total_quality_score REAL DEFAULT 0,
                    patterns_identified INTEGER DEFAULT 0,
                    last_contribution TEXT,
                    specializations TEXT
                )
            ''')

            # Indexes
            await db.execute('CREATE INDEX IF NOT EXISTS idx_contrib_agent ON contributions(agent_id)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_contrib_domain ON contributions(domain)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_contrib_type ON contributions(contribution_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type)')

            await db.commit()

        self._initialized = True

    async def submit_contribution(
        self,
        agent_id: str,
        contribution_type: LearningContributionType,
        content: str,
        domain: str,
        quality_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a learning contribution from an agent.

        Args:
            agent_id: ID of the contributing agent
            contribution_type: Type of contribution
            content: The learning content
            domain: Domain (coding, creative, research, etc.)
            quality_score: Quality score (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            Contribution ID
        """
        await self.initialize()

        import uuid
        contribution_id = f"contrib-{uuid.uuid4().hex[:12]}"

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''INSERT INTO contributions
                   (id, agent_id, contribution_type, content, domain, quality_score, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    contribution_id,
                    agent_id,
                    contribution_type.value,
                    content,
                    domain,
                    quality_score,
                    json.dumps(metadata or {}),
                    datetime.now().isoformat()
                )
            )

            # Update agent stats
            await db.execute(
                '''INSERT INTO agent_learning_stats (agent_id, total_contributions, total_quality_score, last_contribution)
                   VALUES (?, 1, ?, ?)
                   ON CONFLICT(agent_id) DO UPDATE SET
                   total_contributions = total_contributions + 1,
                   total_quality_score = total_quality_score + excluded.total_quality_score,
                   last_contribution = excluded.last_contribution''',
                (agent_id, quality_score, datetime.now().isoformat())
            )

            await db.commit()

        return contribution_id

    async def aggregate_experiences(
        self,
        domain: Optional[str] = None,
        min_quality: float = 0.6,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Aggregate unaggregated experiences for collective learning.

        Returns experiences grouped by similarity for pattern identification.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            query = '''
                SELECT * FROM contributions
                WHERE aggregated = 0
                AND quality_score >= ?
                AND contribution_type = 'experience'
            '''
            params = [min_quality]

            if domain:
                query += ' AND domain = ?'
                params.append(domain)

            query += ' ORDER BY quality_score DESC LIMIT ?'
            params.append(limit)

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            return [dict(row) for row in rows]

    async def identify_patterns(
        self,
        min_occurrences: int = 3,
        domains: Optional[List[str]] = None
    ) -> List[CollectiveInsight]:
        """
        Identify patterns from collective experiences.

        Looks for:
        - Common success strategies
        - Frequent error patterns
        - Best practices across agents
        """
        await self.initialize()

        async with self._aggregation_lock:
            # Get recent high-quality contributions
            contributions = await self.aggregate_experiences(min_quality=0.7, limit=200)

            if len(contributions) < min_occurrences:
                return []

            # Group by domain and content similarity
            domain_groups: Dict[str, List[Dict]] = {}
            for contrib in contributions:
                domain = contrib['domain']
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(contrib)

            insights = []

            for domain, group in domain_groups.items():
                if domains and domain not in domains:
                    continue

                if len(group) >= min_occurrences:
                    # Extract common patterns from this domain
                    agents = list(set(c['agent_id'] for c in group))
                    avg_quality = sum(c['quality_score'] for c in group) / len(group)

                    # Create insight
                    import uuid
                    insight = CollectiveInsight(
                        id=f"insight-{uuid.uuid4().hex[:8]}",
                        insight_type="pattern",
                        description=f"Collective pattern in {domain}: {len(group)} successful experiences from {len(agents)} agents",
                        domains=[domain],
                        contributing_agents=agents,
                        confidence=avg_quality,
                        evidence_count=len(group)
                    )
                    insights.append(insight)

                    # Store insight
                    await self._store_insight(insight)

                    # Mark contributions as aggregated
                    await self._mark_aggregated([c['id'] for c in group], insight.id)

            return insights

    async def _store_insight(self, insight: CollectiveInsight):
        """Store a collective insight."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''INSERT OR REPLACE INTO insights
                   (id, insight_type, description, domains, contributing_agents, confidence, evidence_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    insight.id,
                    insight.insight_type,
                    insight.description,
                    json.dumps(insight.domains),
                    json.dumps(insight.contributing_agents),
                    insight.confidence,
                    insight.evidence_count,
                    insight.created_at
                )
            )
            await db.commit()

    async def _mark_aggregated(self, contribution_ids: List[str], insight_id: str):
        """Mark contributions as aggregated into an insight."""
        async with aiosqlite.connect(self.db_path) as db:
            for cid in contribution_ids:
                await db.execute(
                    'UPDATE contributions SET aggregated = 1, aggregated_into = ? WHERE id = ?',
                    (insight_id, cid)
                )
            await db.commit()

    async def get_insights(
        self,
        insight_type: Optional[str] = None,
        domain: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get collective insights."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            query = 'SELECT * FROM insights WHERE active = 1 AND confidence >= ?'
            params = [min_confidence]

            if insight_type:
                query += ' AND insight_type = ?'
                params.append(insight_type)

            if domain:
                query += ' AND domains LIKE ?'
                params.append(f'%{domain}%')

            query += ' ORDER BY confidence DESC, evidence_count DESC LIMIT ?'
            params.append(limit)

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                r = dict(row)
                r['domains'] = json.loads(r['domains'])
                r['contributing_agents'] = json.loads(r['contributing_agents'])
                results.append(r)

            return results

    async def get_agent_learning_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get learning statistics for agents."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if agent_id:
                cursor = await db.execute(
                    'SELECT * FROM agent_learning_stats WHERE agent_id = ?',
                    (agent_id,)
                )
                row = await cursor.fetchone()
                if row:
                    r = dict(row)
                    r['specializations'] = json.loads(r['specializations']) if r['specializations'] else []
                    return r
                return {}
            else:
                cursor = await db.execute('SELECT * FROM agent_learning_stats ORDER BY total_contributions DESC')
                rows = await cursor.fetchall()
                return {
                    row['agent_id']: {
                        **dict(row),
                        'specializations': json.loads(row['specializations']) if row['specializations'] else []
                    }
                    for row in rows
                }

    async def get_collective_stats(self) -> Dict[str, Any]:
        """Get overall collective learning statistics."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Total contributions
            cursor = await db.execute('SELECT COUNT(*) FROM contributions')
            total_contributions = (await cursor.fetchone())[0]

            # By type
            cursor = await db.execute(
                'SELECT contribution_type, COUNT(*) as count FROM contributions GROUP BY contribution_type'
            )
            by_type = {row[0]: row[1] for row in await cursor.fetchall()}

            # By domain
            cursor = await db.execute(
                'SELECT domain, COUNT(*) as count FROM contributions GROUP BY domain'
            )
            by_domain = {row[0]: row[1] for row in await cursor.fetchall()}

            # By agent
            cursor = await db.execute(
                'SELECT agent_id, COUNT(*) as count FROM contributions GROUP BY agent_id'
            )
            by_agent = {row[0]: row[1] for row in await cursor.fetchall()}

            # Total insights
            cursor = await db.execute('SELECT COUNT(*) FROM insights WHERE active = 1')
            total_insights = (await cursor.fetchone())[0]

            # Avg quality
            cursor = await db.execute('SELECT AVG(quality_score) FROM contributions')
            avg_quality = (await cursor.fetchone())[0] or 0

            # Get aggregated count (properly awaiting fetchone)
            aggregation_rate = 0
            if total_contributions > 0:
                cursor = await db.execute('SELECT COUNT(*) FROM contributions WHERE aggregated = 1')
                aggregated_count = (await cursor.fetchone())[0]
                aggregation_rate = round(aggregated_count / total_contributions, 3)

            return {
                "total_contributions": total_contributions,
                "by_contribution_type": by_type,
                "by_domain": by_domain,
                "by_agent": by_agent,
                "total_insights": total_insights,
                "avg_quality_score": round(avg_quality, 3),
                "aggregation_rate": aggregation_rate
            }

    async def share_learning_with_agent(
        self,
        agent_id: str,
        domains: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Package collective learnings to share with an agent.

        Returns insights and patterns relevant to the agent's domains.
        """
        await self.initialize()

        # Get relevant insights
        insights = []
        if domains:
            for domain in domains:
                domain_insights = await self.get_insights(domain=domain, limit=limit)
                insights.extend(domain_insights)
        else:
            insights = await self.get_insights(limit=limit)

        # Get agent's learning stats
        agent_stats = await self.get_agent_learning_stats(agent_id)

        # Get peer contributions (from other agents)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                '''SELECT agent_id, content, domain, quality_score
                   FROM contributions
                   WHERE agent_id != ? AND quality_score >= 0.8
                   ORDER BY created_at DESC LIMIT ?''',
                (agent_id, limit)
            )
            peer_contributions = [dict(row) for row in await cursor.fetchall()]

        return {
            "agent_id": agent_id,
            "insights": insights,
            "peer_contributions": peer_contributions,
            "agent_stats": agent_stats,
            "recommendation": self._generate_recommendation(agent_stats, insights)
        }

    def _generate_recommendation(
        self,
        agent_stats: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> str:
        """Generate a learning recommendation for the agent."""
        if not agent_stats:
            return "Start contributing experiences to participate in collective learning."

        contributions = agent_stats.get('total_contributions', 0)
        avg_quality = agent_stats.get('total_quality_score', 0) / max(contributions, 1)

        if contributions < 5:
            return "Keep contributing high-quality experiences to build your learning profile."
        elif avg_quality < 0.6:
            return "Focus on improving the quality of your contributions for better collective insights."
        elif len(insights) > 0:
            domains = set()
            for i in insights:
                domains.update(i.get('domains', []))
            return f"Review collective insights in: {', '.join(domains)}"
        else:
            return "Excellent contributions! Continue sharing experiences for collective improvement."


# Singleton instance
_collective_pipeline: Optional[CollectiveLearningPipeline] = None


def get_collective_pipeline(storage_dir: Optional[Path] = None) -> CollectiveLearningPipeline:
    """Get or create the collective learning pipeline."""
    global _collective_pipeline
    if _collective_pipeline is None:
        if storage_dir is None:
            storage_dir = Path.home() / "llm-storage" / "collective"
        _collective_pipeline = CollectiveLearningPipeline(storage_dir)
    return _collective_pipeline
