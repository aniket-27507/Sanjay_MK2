"""Worker pool: manage multiple simulation instances with health checking.

Each worker wraps an Isaac Sim instance and tracks its health and
availability. The pool provides round-robin assignment and health
monitoring.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Any


class WorkerState(enum.Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


@dataclass(slots=True)
class Worker:
    """A single simulation instance worker."""

    worker_id: str
    instance_name: str
    state: WorkerState = WorkerState.IDLE
    current_job_id: str = ""
    jobs_completed: int = 0
    jobs_failed: int = 0
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "instance_name": self.instance_name,
            "state": self.state.value,
            "current_job_id": self.current_job_id,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "consecutive_failures": self.consecutive_failures,
        }

    @property
    def is_available(self) -> bool:
        return self.state == WorkerState.IDLE


class WorkerPool:
    """Manage a pool of simulation workers with health checking.

    Parameters
    ----------
    max_consecutive_failures:
        Number of consecutive failures before marking a worker unhealthy.
    health_check_interval_s:
        Seconds between health checks.
    """

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        health_check_interval_s: float = 30.0,
    ) -> None:
        self._workers: dict[str, Worker] = {}
        self._max_failures = max_consecutive_failures
        self._health_interval = health_check_interval_s
        self._health_task: asyncio.Task[None] | None = None
        self._health_fn: Any = None
        self._stopped = False
        self._round_robin_idx = 0

    def add_worker(self, worker_id: str, instance_name: str, metadata: dict[str, Any] | None = None) -> Worker:
        """Register a new worker in the pool."""
        worker = Worker(
            worker_id=worker_id,
            instance_name=instance_name,
            last_health_check=time.monotonic(),
            metadata=metadata or {},
        )
        self._workers[worker_id] = worker
        return worker

    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool."""
        if worker_id in self._workers:
            del self._workers[worker_id]
            return True
        return False

    def get_worker(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    def acquire_worker(self) -> Worker | None:
        """Get the next available worker using round-robin selection.

        Returns None if no workers are available.
        """
        available = [w for w in self._workers.values() if w.is_available]
        if not available:
            return None

        # Round-robin among available workers
        idx = self._round_robin_idx % len(available)
        self._round_robin_idx += 1
        worker = available[idx]
        worker.state = WorkerState.BUSY
        return worker

    def release_worker(self, worker_id: str, success: bool = True) -> Worker | None:
        """Release a worker back to idle after job completion."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return None

        worker.current_job_id = ""

        if success:
            worker.jobs_completed += 1
            worker.consecutive_failures = 0
            worker.state = WorkerState.IDLE
        else:
            worker.jobs_failed += 1
            worker.consecutive_failures += 1
            if worker.consecutive_failures >= self._max_failures:
                worker.state = WorkerState.UNHEALTHY
            else:
                worker.state = WorkerState.IDLE

        return worker

    def assign_job(self, worker_id: str, job_id: str) -> bool:
        """Associate a job with a worker."""
        worker = self._workers.get(worker_id)
        if worker is None or worker.state != WorkerState.BUSY:
            return False
        worker.current_job_id = job_id
        return True

    def drain_worker(self, worker_id: str) -> bool:
        """Put a worker into draining state (no new jobs)."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return False
        worker.state = WorkerState.DRAINING
        return True

    def recover_worker(self, worker_id: str) -> bool:
        """Move an unhealthy or draining worker back to idle."""
        worker = self._workers.get(worker_id)
        if worker is None:
            return False
        if worker.state not in (WorkerState.UNHEALTHY, WorkerState.DRAINING):
            return False
        worker.consecutive_failures = 0
        worker.state = WorkerState.IDLE
        return True

    def set_health_check(self, fn: Any) -> None:
        """Set an async health check function: async fn(worker) -> bool."""
        self._health_fn = fn

    async def start_health_checks(self) -> None:
        """Start periodic health checking."""
        if self._health_task is not None:
            return
        self._stopped = False
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop_health_checks(self) -> None:
        """Stop periodic health checking."""
        self._stopped = True
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _health_loop(self) -> None:
        while not self._stopped:
            try:
                await self._check_all_workers()
                await asyncio.sleep(self._health_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self._health_interval)

    async def _check_all_workers(self) -> None:
        """Run health check on all idle/unhealthy workers."""
        if self._health_fn is None:
            return

        for worker in self._workers.values():
            if worker.state == WorkerState.BUSY:
                continue  # Don't health-check busy workers
            try:
                healthy = await self._health_fn(worker)
                worker.last_health_check = time.monotonic()
                if healthy and worker.state == WorkerState.UNHEALTHY:
                    worker.consecutive_failures = 0
                    worker.state = WorkerState.IDLE
                elif not healthy and worker.state == WorkerState.IDLE:
                    worker.consecutive_failures += 1
                    if worker.consecutive_failures >= self._max_failures:
                        worker.state = WorkerState.UNHEALTHY
            except Exception:
                worker.consecutive_failures += 1
                if worker.consecutive_failures >= self._max_failures:
                    worker.state = WorkerState.UNHEALTHY

    @property
    def total_workers(self) -> int:
        return len(self._workers)

    @property
    def available_workers(self) -> int:
        return sum(1 for w in self._workers.values() if w.is_available)

    @property
    def busy_workers(self) -> int:
        return sum(1 for w in self._workers.values() if w.state == WorkerState.BUSY)

    def list_workers(self) -> list[Worker]:
        return list(self._workers.values())

    def get_stats(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        total_completed = 0
        total_failed = 0
        for w in self._workers.values():
            by_state[w.state.value] = by_state.get(w.state.value, 0) + 1
            total_completed += w.jobs_completed
            total_failed += w.jobs_failed

        return {
            "total_workers": self.total_workers,
            "available": self.available_workers,
            "busy": self.busy_workers,
            "by_state": by_state,
            "total_jobs_completed": total_completed,
            "total_jobs_failed": total_failed,
        }
