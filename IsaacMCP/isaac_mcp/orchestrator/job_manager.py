"""Job orchestration with priority queue and lifecycle management.

Jobs represent units of work (batch experiments, parameter sweeps,
regression suites, etc.) that are queued, scheduled, and tracked
through their lifecycle.
"""

from __future__ import annotations

import asyncio
import enum
import heapq
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine


class JobStatus(enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class JobPriority(enum.IntEnum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass(slots=True)
class Job:
    """A unit of work to be scheduled and executed."""

    job_id: str
    job_type: str
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    instance: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    timeout_s: float = 300.0
    retries: int = 0
    max_retries: int = 1
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "priority": self.priority.name.lower(),
            "status": self.status.value,
            "instance": self.instance,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "timeout_s": self.timeout_s,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "tags": self.tags,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        )

    @property
    def duration_s(self) -> float:
        if not self.started_at:
            return 0.0
        start = datetime.fromisoformat(self.started_at)
        if self.completed_at:
            end = datetime.fromisoformat(self.completed_at)
        else:
            end = datetime.now(timezone.utc)
        return (end - start).total_seconds()


# Priority queue entry: (priority, sequence_number, job_id)
_QueueEntry = tuple[int, int, str]


class JobManager:
    """Manage job lifecycle: create, queue, track, and complete jobs.

    The manager maintains a priority queue and a registry of all jobs.
    It does NOT execute jobs -- that is the scheduler/worker pool's role.
    """

    def __init__(self, max_history: int = 500) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: list[_QueueEntry] = []
        self._seq = 0
        self._max_history = max_history
        self._on_complete_callbacks: list[Callable[[Job], Coroutine[Any, Any, None]]] = []

    def create_job(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        priority: str = "normal",
        timeout_s: float = 300.0,
        max_retries: int = 1,
        tags: list[str] | None = None,
    ) -> Job:
        """Create a new job and add it to the queue."""
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        priority_enum = _parse_priority(priority)

        job = Job(
            job_id=job_id,
            job_type=job_type,
            priority=priority_enum,
            status=JobStatus.QUEUED,
            payload=payload or {},
            created_at=now,
            timeout_s=timeout_s,
            max_retries=max_retries,
            tags=tags or [],
        )

        self._jobs[job_id] = job
        self._enqueue(job)
        self._trim_history()
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> Job | None:
        """Cancel a pending or queued job."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.is_terminal:
            return None
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        return job

    def dequeue(self) -> Job | None:
        """Pop the highest-priority job from the queue.

        Skips jobs that are no longer in QUEUED state (e.g. cancelled).
        """
        while self._queue:
            _, _, job_id = heapq.heappop(self._queue)
            job = self._jobs.get(job_id)
            if job is not None and job.status == JobStatus.QUEUED:
                return job
        return None

    def mark_running(self, job_id: str, instance: str = "") -> Job | None:
        """Transition a job to RUNNING state."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return None
        job.status = JobStatus.RUNNING
        job.instance = instance
        job.started_at = datetime.now(timezone.utc).isoformat()
        return job

    def mark_completed(self, job_id: str, result: dict[str, Any] | None = None) -> Job | None:
        """Transition a job to COMPLETED state."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return None
        job.status = JobStatus.COMPLETED
        job.result = result or {}
        job.completed_at = datetime.now(timezone.utc).isoformat()
        return job

    def mark_failed(self, job_id: str, error: str = "") -> Job | None:
        """Transition a job to FAILED state, or re-queue if retries remain."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return None

        job.retries += 1
        if job.retries < job.max_retries:
            # Re-queue for retry
            job.status = JobStatus.QUEUED
            job.error = error
            job.instance = ""
            job.started_at = ""
            self._enqueue(job)
            return job

        job.status = JobStatus.FAILED
        job.error = error
        job.completed_at = datetime.now(timezone.utc).isoformat()
        return job

    def mark_timed_out(self, job_id: str) -> Job | None:
        """Transition a running job to TIMED_OUT."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return None
        job.status = JobStatus.TIMED_OUT
        job.error = "Job exceeded timeout"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        return job

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())

        if status is not None:
            try:
                status_enum = JobStatus(status)
                jobs = [j for j in jobs if j.status == status_enum]
            except ValueError:
                pass

        if job_type is not None:
            jobs = [j for j in jobs if j.job_type == job_type]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    @property
    def queue_size(self) -> int:
        """Number of jobs in QUEUED state."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)

    @property
    def running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics."""
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for job in self._jobs.values():
            by_status[job.status.value] = by_status.get(job.status.value, 0) + 1
            by_type[job.job_type] = by_type.get(job.job_type, 0) + 1

        return {
            "total_jobs": len(self._jobs),
            "queue_size": self.queue_size,
            "running": self.running_count,
            "by_status": by_status,
            "by_type": by_type,
        }

    def check_timeouts(self) -> list[Job]:
        """Check running jobs for timeout and mark them accordingly."""
        timed_out: list[Job] = []
        now = datetime.now(timezone.utc)
        for job in list(self._jobs.values()):
            if job.status != JobStatus.RUNNING or not job.started_at:
                continue
            started = datetime.fromisoformat(job.started_at)
            elapsed = (now - started).total_seconds()
            if elapsed > job.timeout_s:
                self.mark_timed_out(job.job_id)
                timed_out.append(job)
        return timed_out

    def on_complete(self, callback: Callable[[Job], Coroutine[Any, Any, None]]) -> None:
        """Register a callback for job completion."""
        self._on_complete_callbacks.append(callback)

    def _enqueue(self, job: Job) -> None:
        self._seq += 1
        heapq.heappush(self._queue, (job.priority.value, self._seq, job.job_id))

    def _trim_history(self) -> None:
        """Remove oldest terminal jobs when exceeding max_history."""
        terminal = [
            j for j in self._jobs.values() if j.is_terminal
        ]
        if len(terminal) <= self._max_history:
            return
        terminal.sort(key=lambda j: j.completed_at or j.created_at)
        to_remove = len(terminal) - self._max_history
        for job in terminal[:to_remove]:
            del self._jobs[job.job_id]


def _parse_priority(value: str) -> JobPriority:
    mapping = {
        "urgent": JobPriority.URGENT,
        "high": JobPriority.HIGH,
        "normal": JobPriority.NORMAL,
        "low": JobPriority.LOW,
    }
    return mapping.get(value.lower(), JobPriority.NORMAL)
