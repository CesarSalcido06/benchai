"""
BenchAI Background Task Queue

Allows async task processing for:
- Research queries
- Document indexing
- Model inference
- Multi-agent coordination
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Task:
    id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    requester: str = "system"


class BackgroundTaskQueue:
    """
    Async task queue for background processing.

    Features:
    - Priority queue
    - Concurrent workers
    - Task persistence
    - Webhook callbacks
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.queue = None  # Lazy init
        self.workers: List[asyncio.Task] = []
        self.handlers: Dict[str, Callable] = {}
        self.running = False
        self._lock = None  # Lazy init

    async def _ensure_started(self):
        """Ensure queue is initialized (lazy init)."""
        if self.queue is None:
            self.queue = asyncio.PriorityQueue()
            self._lock = asyncio.Lock()
            self.running = True

    async def start(self):
        """Start the task queue workers."""
        await self._ensure_started()
        if len(self.workers) > 0:
            return  # Already have workers

        # Start workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        print(f"[TASK-QUEUE] Started with {self.max_workers} workers")

    async def stop(self):
        """Stop all workers."""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        self.workers = []
        print("[TASK-QUEUE] Stopped")

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self.handlers[task_type] = handler
        print(f"[TASK-QUEUE] Registered handler for '{task_type}'")

    async def submit(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback_url: Optional[str] = None,
        requester: str = "system"
    ) -> str:
        """Submit a task to the queue."""
        await self._ensure_started()

        task_id = str(uuid.uuid4())[:8]

        task = Task(
            id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            callback_url=callback_url,
            requester=requester
        )

        async with self._lock:
            self.tasks[task_id] = task
            # Priority queue: lower number = higher priority, so negate
            await self.queue.put((-priority.value, task_id))

        print(f"[TASK-QUEUE] Submitted task {task_id}: {task_type} (priority={priority.name})")
        return task_id

    async def get_status(self, task_id: str) -> Optional[Dict]:
        """Get task status."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "id": task.id,
            "type": task.task_type,
            "status": task.status.value,
            "priority": task.priority.name,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }

    async def get_result(self, task_id: str, wait: bool = False, timeout: float = 60) -> Optional[Dict]:
        """Get task result, optionally waiting for completion."""
        if wait:
            start = asyncio.get_event_loop().time()
            while True:
                task = self.tasks.get(task_id)
                if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break
                if asyncio.get_event_loop().time() - start > timeout:
                    break
                await asyncio.sleep(0.5)

        return await self.get_status(task_id)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        return False

    async def list_tasks(self, status: Optional[TaskStatus] = None, limit: int = 50) -> List[Dict]:
        """List tasks, optionally filtered by status."""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return [await self.get_status(t.id) for t in tasks[:limit]]

    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks."""
        print(f"[TASK-QUEUE] Worker {worker_id} started")

        while self.running:
            try:
                # Get next task (with timeout to allow checking running flag)
                try:
                    _, task_id = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue

                # Mark as running
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()

                print(f"[TASK-QUEUE] Worker {worker_id} processing task {task_id}: {task.task_type}")

                # Get handler
                handler = self.handlers.get(task.task_type)
                if not handler:
                    task.status = TaskStatus.FAILED
                    task.error = f"No handler for task type: {task.task_type}"
                    task.completed_at = datetime.utcnow()
                    continue

                # Execute handler
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(task.payload)
                    else:
                        result = handler(task.payload)

                    task.status = TaskStatus.COMPLETED
                    task.result = result
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                finally:
                    task.completed_at = datetime.utcnow()

                print(f"[TASK-QUEUE] Task {task_id} {task.status.value}")

                # Send callback if configured
                if task.callback_url:
                    await self._send_callback(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[TASK-QUEUE] Worker {worker_id} error: {e}")

    async def _send_callback(self, task: Task):
        """Send webhook callback."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    task.callback_url,
                    json={
                        "task_id": task.id,
                        "status": task.status.value,
                        "result": task.result,
                        "error": task.error
                    },
                    timeout=10
                )
        except Exception as e:
            print(f"[TASK-QUEUE] Callback failed for task {task.id}: {e}")

    def stats(self) -> Dict:
        """Get queue statistics."""
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "queue_size": self.queue.qsize() if self.queue else 0,
            "workers": len(self.workers),
            "running": self.running,
            "by_status": status_counts
        }


# Global task queue instance
task_queue = BackgroundTaskQueue(max_workers=3)


# Pre-defined task handlers
async def handle_research_task(payload: Dict) -> Dict:
    """Handle research tasks by calling BenchAI's chat endpoint."""
    import httpx

    query = payload.get("query", "")
    max_tokens = payload.get("max_tokens", 500)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8085/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": query}],
                "max_tokens": max_tokens
            },
            timeout=120
        )
        data = resp.json()
        return {
            "query": query,
            "response": data.get("choices", [{}])[0].get("message", {}).get("content", "")
        }


async def handle_index_task(payload: Dict) -> Dict:
    """Handle document indexing tasks."""
    import httpx

    file_path = payload.get("file_path", "")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8085/v1/pdf/analyze",
            json={
                "file_path": file_path,
                "action": "index"
            },
            timeout=120
        )
        return resp.json()


async def handle_memory_task(payload: Dict) -> Dict:
    """Handle memory storage tasks."""
    import httpx

    content = payload.get("content", "")
    category = payload.get("category", "note")
    importance = payload.get("importance", 3)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8085/v1/memory/add",
            json={
                "content": content,
                "category": category,
                "importance": importance
            },
            timeout=30
        )
        return resp.json()


async def handle_agent_task(payload: Dict) -> Dict:
    """Handle multi-agent coordination tasks."""
    import httpx

    target_agent = payload.get("agent", "")
    task_type = payload.get("task_type", "")
    description = payload.get("description", "")

    # Route to appropriate agent using configured URLs
    from learning.agent_config import get_config
    config = get_config()

    endpoint = None
    if target_agent in config:
        endpoint = config[target_agent].get_url("a2a_task")
    else:
        # Fallback for unknown agents
        legacy_endpoints = {
            "dottscavisAI": "http://dottscavisai.local:8766/v1/a2a/task"
        }
        endpoint = legacy_endpoints.get(target_agent)
    if not endpoint:
        return {"error": f"Unknown agent: {target_agent}"}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            endpoint,
            json={
                "from_agent": "benchai",
                "task_type": task_type,
                "task_description": description,
                "context": payload.get("context", {})
            },
            timeout=120
        )
        return resp.json()


# Register default handlers
def register_default_handlers():
    """Register all default task handlers."""
    task_queue.register_handler("research", handle_research_task)
    task_queue.register_handler("index", handle_index_task)
    task_queue.register_handler("memory", handle_memory_task)
    task_queue.register_handler("agent", handle_agent_task)
