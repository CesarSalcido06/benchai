"""
Experience Replay System for Self-Improving AI

This implements the "experience replay for prompting" technique from NeurIPS 2025.
Key insight: Using past successful trajectories as in-context examples
can improve performance by 15-20% without any fine-tuning.

Features:
- Success Library: Store successful task completions with trajectories
- Failure Repair: Store failed attempts with corrected solutions
- Curious Replay: Prioritize novel/interesting experiences
- In-context example injection
"""

import asyncio
import aiosqlite
import json
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field


class ExperienceOutcome(str, Enum):
    """Outcome of a task attempt."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    REPAIRED = "repaired"  # Failed but later corrected


class TaskDomain(str, Enum):
    """Domain categories for experiences."""
    CODING = "coding"
    RESEARCH = "research"
    CREATIVE = "creative"
    SYSTEM = "system"
    ORCHESTRATION = "orchestration"
    GENERAL = "general"


@dataclass
class Experience:
    """A single experience/trajectory."""
    id: Optional[int]
    task_description: str
    domain: TaskDomain
    approach: str  # What approach was taken
    trajectory: List[Dict]  # Steps taken (can be used for in-context learning)
    outcome: ExperienceOutcome
    outcome_details: str  # Why it succeeded/failed
    tokens_used: int
    duration_ms: int
    novelty_score: float  # How novel/interesting (for curious replay)
    success_score: float  # Quality of the success (0-1)
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For repaired experiences
    original_approach: Optional[str] = None
    repair_explanation: Optional[str] = None


class ExperienceReplayManager:
    """
    Manages the experience replay system for self-improving AI.

    Based on:
    - Stanford's "Curious Replay" for prioritizing novel experiences
    - NeurIPS 2025 "Self-Generated In-Context Examples"
    - SiriuS multi-agent experience sharing
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._initialized = False

        # Curious replay settings
        self.novelty_decay = 0.05  # Reduce novelty over time
        self.min_success_score = 0.6  # Minimum quality to use as example
        self.max_examples = 5  # Max in-context examples to inject

    async def _optimize_connection(self, db):
        """Apply performance optimizations."""
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        await db.execute("PRAGMA cache_size = -40000")
        await db.execute("PRAGMA temp_store = MEMORY")

    async def initialize(self):
        """Initialize the experience database."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Main experiences table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_description TEXT NOT NULL,
                    task_hash TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT 'general',
                    approach TEXT NOT NULL,
                    trajectory JSON NOT NULL,
                    outcome TEXT NOT NULL,
                    outcome_details TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    novelty_score REAL DEFAULT 1.0,
                    success_score REAL DEFAULT 0.0,
                    original_approach TEXT,
                    repair_explanation TEXT,
                    agent_source TEXT DEFAULT 'benchai',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    metadata JSON DEFAULT '{}'
                )
            ''')

            # Indexes
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_exp_domain_outcome
                ON experiences(domain, outcome, success_score DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_exp_novelty
                ON experiences(novelty_score DESC, created_at DESC)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_exp_task_hash
                ON experiences(task_hash)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_exp_agent
                ON experiences(agent_source)
            ''')

            # FTS for semantic search of experiences
            await db.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
                    task_description,
                    approach,
                    outcome_details,
                    domain,
                    tokenize='porter unicode61'
                )
            ''')

            # FTS triggers
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS exp_ai AFTER INSERT ON experiences BEGIN
                    INSERT INTO experiences_fts(rowid, task_description, approach, outcome_details, domain)
                    VALUES (new.id, new.task_description, new.approach, new.outcome_details, new.domain);
                END
            ''')
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS exp_ad AFTER DELETE ON experiences BEGIN
                    DELETE FROM experiences_fts WHERE rowid = old.id;
                END
            ''')

            # Training data queue for fine-tuning
            await db.execute('''
                CREATE TABLE IF NOT EXISTS training_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experience_id INTEGER NOT NULL,
                    instruction TEXT NOT NULL,
                    input TEXT,
                    output TEXT NOT NULL,
                    domain TEXT,
                    quality_score REAL,
                    used_in_training INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experience_id) REFERENCES experiences(id)
                )
            ''')

            await db.execute("ANALYZE")
            await db.commit()

        self._initialized = True
        print("[EXPERIENCE] Experience replay system initialized")

    def _compute_task_hash(self, task: str) -> str:
        """Compute semantic hash of task for deduplication."""
        # Normalize: lowercase, remove extra whitespace
        normalized = " ".join(task.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _estimate_novelty(self, task: str, domain: str, existing_count: int) -> float:
        """
        Estimate novelty of a task.
        Novel tasks are prioritized in curious replay.
        """
        # Base novelty is high for new tasks
        novelty = 1.0

        # Reduce novelty based on similar existing experiences
        if existing_count > 0:
            novelty *= max(0.3, 1.0 - (existing_count * 0.1))

        # Certain domains are more valuable for learning
        domain_weights = {
            TaskDomain.CODING.value: 1.2,
            TaskDomain.RESEARCH.value: 1.1,
            TaskDomain.ORCHESTRATION.value: 1.3,
            TaskDomain.CREATIVE.value: 1.0,
            TaskDomain.SYSTEM.value: 0.9,
            TaskDomain.GENERAL.value: 0.8,
        }
        novelty *= domain_weights.get(domain, 1.0)

        return min(1.0, novelty)

    async def record_experience(
        self,
        task_description: str,
        domain: TaskDomain,
        approach: str,
        trajectory: List[Dict],
        outcome: ExperienceOutcome,
        outcome_details: str = "",
        tokens_used: int = 0,
        duration_ms: int = 0,
        success_score: float = 0.0,
        agent_source: str = "benchai",
        metadata: Optional[Dict] = None,
        original_approach: Optional[str] = None,
        repair_explanation: Optional[str] = None
    ) -> int:
        """
        Record a new experience in the replay buffer.

        Args:
            task_description: What task was being performed
            domain: Task domain category
            approach: What approach was taken
            trajectory: List of steps taken (for in-context learning)
            outcome: SUCCESS, PARTIAL, FAILURE, or REPAIRED
            outcome_details: Why it succeeded/failed
            tokens_used: Token consumption
            duration_ms: Time taken
            success_score: Quality score 0-1
            agent_source: Which agent recorded this
            metadata: Additional data
            original_approach: For repaired experiences, what was tried first
            repair_explanation: For repaired, what was learned

        Returns:
            ID of the recorded experience
        """
        await self.initialize()

        task_hash = self._compute_task_hash(task_description)

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)

            # Check for similar existing experiences
            cursor = await db.execute(
                "SELECT COUNT(*) FROM experiences WHERE task_hash = ?",
                (task_hash,)
            )
            existing_count = (await cursor.fetchone())[0]

            novelty = self._estimate_novelty(task_description, domain.value, existing_count)

            cursor = await db.execute(
                '''INSERT INTO experiences
                   (task_description, task_hash, domain, approach, trajectory,
                    outcome, outcome_details, tokens_used, duration_ms,
                    novelty_score, success_score, agent_source, metadata,
                    original_approach, repair_explanation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    task_description, task_hash, domain.value, approach,
                    json.dumps(trajectory), outcome.value, outcome_details,
                    tokens_used, duration_ms, novelty, success_score,
                    agent_source, json.dumps(metadata or {}),
                    original_approach, repair_explanation
                )
            )
            exp_id = cursor.lastrowid

            # If successful with high quality, add to training queue
            if outcome in [ExperienceOutcome.SUCCESS, ExperienceOutcome.REPAIRED] and success_score >= self.min_success_score:
                await self._add_to_training_queue(db, exp_id, task_description, approach, trajectory, domain.value, success_score)

            await db.commit()
            return exp_id

    async def _add_to_training_queue(
        self,
        db,
        experience_id: int,
        task: str,
        approach: str,
        trajectory: List[Dict],
        domain: str,
        quality_score: float
    ):
        """Add high-quality experience to training queue for fine-tuning."""
        # Create instruction-following format
        instruction = f"Task: {task}"

        # Build output from trajectory
        if trajectory:
            output_parts = []
            for step in trajectory:
                if isinstance(step, dict):
                    if 'action' in step:
                        output_parts.append(f"Action: {step['action']}")
                    if 'result' in step:
                        output_parts.append(f"Result: {step['result'][:500]}")
                    if 'reasoning' in step:
                        output_parts.append(f"Reasoning: {step['reasoning']}")
            output = "\n".join(output_parts) if output_parts else approach
        else:
            output = approach

        await db.execute(
            '''INSERT INTO training_queue
               (experience_id, instruction, output, domain, quality_score)
               VALUES (?, ?, ?, ?, ?)''',
            (experience_id, instruction, output, domain, quality_score)
        )

    async def get_similar_successes(
        self,
        task_description: str,
        domain: Optional[TaskDomain] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find similar successful experiences for in-context learning.
        This is the core of "experience replay for prompting".
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            try:
                # Search using FTS
                conditions = ["experiences_fts MATCH ?", "e.outcome IN ('success', 'repaired')", "e.success_score >= ?"]
                params = [task_description, self.min_success_score]

                if domain:
                    conditions.append("e.domain = ?")
                    params.append(domain.value)

                params.append(limit)
                where_clause = " AND ".join(conditions)

                cursor = await db.execute(f'''
                    SELECT e.*, bm25(experiences_fts) as rank
                    FROM experiences e
                    JOIN experiences_fts ON e.id = experiences_fts.rowid
                    WHERE {where_clause}
                    ORDER BY rank, e.success_score DESC, e.novelty_score DESC
                    LIMIT ?
                ''', params)

                results = []
                async for row in cursor:
                    # Update access tracking
                    await db.execute(
                        "UPDATE experiences SET access_count = access_count + 1, accessed_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (row['id'],)
                    )
                    results.append({
                        "id": row['id'],
                        "task_description": row['task_description'],
                        "domain": row['domain'],
                        "approach": row['approach'],
                        "trajectory": json.loads(row['trajectory']),
                        "outcome": row['outcome'],
                        "outcome_details": row['outcome_details'],
                        "success_score": row['success_score'],
                        "novelty_score": row['novelty_score'],
                        "relevance": -row['rank']
                    })

                await db.commit()
                return results

            except Exception as e:
                print(f"[EXPERIENCE] FTS search failed: {e}")
                return []

    async def get_curious_replay_examples(self, limit: int = 10) -> List[Dict]:
        """
        Get most novel/interesting experiences for curious replay.
        Based on Stanford HAI research on prioritizing novel experiences.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            # Get high-novelty successful experiences
            cursor = await db.execute(
                '''SELECT * FROM experiences
                   WHERE outcome IN ('success', 'repaired')
                   AND success_score >= ?
                   ORDER BY novelty_score DESC, success_score DESC, created_at DESC
                   LIMIT ?''',
                (self.min_success_score, limit)
            )

            return [dict(row) async for row in cursor]

    async def get_repaired_failures(self, domain: Optional[TaskDomain] = None, limit: int = 10) -> List[Dict]:
        """
        Get experiences where failures were repaired.
        These are valuable learning examples showing what NOT to do and the correction.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            if domain:
                cursor = await db.execute(
                    '''SELECT * FROM experiences
                       WHERE outcome = 'repaired' AND domain = ?
                       ORDER BY success_score DESC, created_at DESC
                       LIMIT ?''',
                    (domain.value, limit)
                )
            else:
                cursor = await db.execute(
                    '''SELECT * FROM experiences
                       WHERE outcome = 'repaired'
                       ORDER BY success_score DESC, created_at DESC
                       LIMIT ?''',
                    (limit,)
                )

            return [dict(row) async for row in cursor]

    async def repair_failure(
        self,
        failed_experience_id: int,
        corrected_approach: str,
        corrected_trajectory: List[Dict],
        repair_explanation: str,
        success_score: float = 0.8
    ) -> int:
        """
        Create a repaired version of a failed experience.
        This implements "hindsight experience replay" for learning from mistakes.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            await self._optimize_connection(db)
            db.row_factory = aiosqlite.Row

            # Get the failed experience
            cursor = await db.execute(
                "SELECT * FROM experiences WHERE id = ?",
                (failed_experience_id,)
            )
            failed_exp = await cursor.fetchone()

            if not failed_exp:
                raise ValueError(f"Experience {failed_experience_id} not found")

            # Create repaired experience
            cursor = await db.execute(
                '''INSERT INTO experiences
                   (task_description, task_hash, domain, approach, trajectory,
                    outcome, outcome_details, tokens_used, duration_ms,
                    novelty_score, success_score, agent_source, metadata,
                    original_approach, repair_explanation)
                   VALUES (?, ?, ?, ?, ?, 'repaired', ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    failed_exp['task_description'],
                    failed_exp['task_hash'],
                    failed_exp['domain'],
                    corrected_approach,
                    json.dumps(corrected_trajectory),
                    repair_explanation,
                    failed_exp['tokens_used'],
                    failed_exp['duration_ms'],
                    failed_exp['novelty_score'] * 1.2,  # Boost novelty for repaired
                    success_score,
                    failed_exp['agent_source'],
                    failed_exp['metadata'],
                    failed_exp['approach'],  # Original failed approach
                    repair_explanation
                )
            )
            repaired_id = cursor.lastrowid

            # Add to training queue
            await self._add_to_training_queue(
                db, repaired_id,
                failed_exp['task_description'],
                corrected_approach,
                corrected_trajectory,
                failed_exp['domain'],
                success_score
            )

            await db.commit()
            return repaired_id

    def format_as_in_context_examples(self, experiences: List[Dict]) -> str:
        """
        Format experiences as in-context examples for prompting.
        This is the key technique for "experience replay for prompting".
        """
        if not experiences:
            return ""

        examples = []
        for i, exp in enumerate(experiences[:self.max_examples], 1):
            example = f"""### Example {i}: {exp['task_description'][:100]}
**Approach:** {exp['approach'][:300]}
**Outcome:** {exp['outcome']} (Score: {exp.get('success_score', 'N/A')})
"""
            if exp.get('outcome_details'):
                example += f"**Details:** {exp['outcome_details'][:200]}\n"

            if exp.get('original_approach') and exp.get('repair_explanation'):
                example += f"""
**What didn't work:** {exp['original_approach'][:200]}
**Lesson learned:** {exp['repair_explanation'][:200]}
"""
            examples.append(example)

        return "\n---\n".join(examples)

    async def get_training_data(
        self,
        domain: Optional[str] = None,
        min_quality: float = 0.7,
        unused_only: bool = True,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get training data for fine-tuning in instruction format.
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            conditions = ["quality_score >= ?"]
            params = [min_quality]

            if domain:
                conditions.append("domain = ?")
                params.append(domain)

            if unused_only:
                conditions.append("used_in_training = 0")

            params.append(limit)
            where_clause = " AND ".join(conditions)

            cursor = await db.execute(f'''
                SELECT * FROM training_queue
                WHERE {where_clause}
                ORDER BY quality_score DESC, created_at DESC
                LIMIT ?
            ''', params)

            return [dict(row) async for row in cursor]

    async def mark_training_used(self, ids: List[int]):
        """Mark training examples as used."""
        await self.initialize()

        if not ids:
            return

        async with aiosqlite.connect(self.db_path) as db:
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"UPDATE training_queue SET used_in_training = 1 WHERE id IN ({placeholders})",
                ids
            )
            await db.commit()

    async def apply_novelty_decay(self):
        """Apply decay to novelty scores over time."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Decay novelty for experiences older than 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()

            await db.execute(
                '''UPDATE experiences
                   SET novelty_score = MAX(0.1, novelty_score - ?)
                   WHERE created_at < ?
                   AND novelty_score > 0.1''',
                (self.novelty_decay, week_ago)
            )
            await db.commit()

    async def get_stats(self) -> Dict:
        """Get statistics about the experience replay system."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            stats = {"by_outcome": {}, "by_domain": {}, "training_queue": {}}

            # By outcome
            cursor = await db.execute(
                '''SELECT outcome, COUNT(*) as count, AVG(success_score) as avg_score
                   FROM experiences GROUP BY outcome'''
            )
            async for row in cursor:
                stats["by_outcome"][row['outcome']] = {
                    "count": row['count'],
                    "avg_score": round(row['avg_score'] or 0, 2)
                }

            # By domain
            cursor = await db.execute(
                '''SELECT domain, COUNT(*) as count FROM experiences GROUP BY domain'''
            )
            async for row in cursor:
                stats["by_domain"][row['domain']] = row['count']

            # Training queue
            cursor = await db.execute("SELECT COUNT(*) FROM training_queue WHERE used_in_training = 0")
            stats["training_queue"]["pending"] = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM training_queue WHERE used_in_training = 1")
            stats["training_queue"]["used"] = (await cursor.fetchone())[0]

            # Total experiences
            cursor = await db.execute("SELECT COUNT(*) FROM experiences")
            stats["total_experiences"] = (await cursor.fetchone())[0]

            # Average novelty
            cursor = await db.execute("SELECT AVG(novelty_score) FROM experiences")
            stats["avg_novelty"] = round((await cursor.fetchone())[0] or 0, 2)

            return stats
