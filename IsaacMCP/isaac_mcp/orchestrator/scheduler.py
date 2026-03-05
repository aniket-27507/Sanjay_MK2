"""Job scheduler: assigns queued jobs to available worker instances.

The scheduler polls the job manager's queue and dispatches jobs to
workers in the pool. It handles concurrency limits per instance and
basic load balancing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from isaac_mcp.orchestrator.job_manager import Job, JobManager, JobStatus

logger = logging.getLogger(__name__)

# Type for async job executor: (job) -> result dict
JobExecutor = Callable[[Job], Coroutine[Any, Any, dict[str, Any]]]


class Scheduler:
    """Poll job queue and dispatch to available workers.

    Parameters
    ----------
    job_manager:
        The job manager that owns the queue.
    max_concurrent:
        Maximum number of jobs running simultaneously.
    poll_interval_s:
        Seconds between queue polls.
    """

    def __init__(
        self,
        job_manager: JobManager,
        max_concurrent: int = 4,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._jm = job_manager
        self._max_concurrent = max(1, max_concurrent)
        self._poll_interval = poll_interval_s
        self._running: dict[str, asyncio.Task[None]] = {}
        self._executor: JobExecutor | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._stopped = False

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def set_executor(self, executor: JobExecutor) -> None:
        """Set the function that actually executes jobs."""
        self._executor = executor

    async def start(self) -> None:
        """Start the scheduling loop."""
        if self._loop_task is not None:
            return
        self._stopped = False
        self._loop_task = asyncio.create_task(self._poll_loop())

    async def stop(self, cancel_running: bool = False) -> None:
        """Stop the scheduling loop."""
        self._stopped = True
        if cancel_running:
            for task in self._running.values():
                task.cancel()
            if self._running:
                await asyncio.gather(*self._running.values(), return_exceptions=True)
            self._running.clear()

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def _poll_loop(self) -> None:
        """Main scheduling loop."""
        while not self._stopped:
            try:
                self._cleanup_finished()
                self._jm.check_timeouts()
                await self._dispatch_pending()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Scheduler error: %s", exc)
                await asyncio.sleep(self._poll_interval)

    async def _dispatch_pending(self) -> None:
        """Dequeue and dispatch jobs up to the concurrency limit."""
        while len(self._running) < self._max_concurrent:
            job = self._jm.dequeue()
            if job is None:
                break
            self._jm.mark_running(job.job_id)
            task = asyncio.create_task(self._execute_job(job))
            self._running[job.job_id] = task

    async def _execute_job(self, job: Job) -> None:
        """Execute a single job and update its status."""
        try:
            if self._executor is None:
                # No executor: simulate success
                result = {"simulated": True}
            else:
                result = await asyncio.wait_for(
                    self._executor(job),
                    timeout=job.timeout_s,
                )
            self._jm.mark_completed(job.job_id, result)
        except asyncio.TimeoutError:
            self._jm.mark_timed_out(job.job_id)
        except asyncio.CancelledError:
            self._jm.mark_failed(job.job_id, error="cancelled")
        except Exception as exc:
            self._jm.mark_failed(job.job_id, error=str(exc))
        finally:
            self._running.pop(job.job_id, None)

    def _cleanup_finished(self) -> None:
        """Remove completed tasks from the running set."""
        finished = [jid for jid, task in self._running.items() if task.done()]
        for jid in finished:
            self._running.pop(jid, None)

    def dispatch_one(self) -> Job | None:
        """Synchronously dequeue and mark one job as running (for testing)."""
        job = self._jm.dequeue()
        if job is None:
            return None
        self._jm.mark_running(job.job_id)
        return job

    def get_stats(self) -> dict[str, Any]:
        return {
            "max_concurrent": self._max_concurrent,
            "running": self.running_count,
            "available_slots": self._max_concurrent - self.running_count,
            "is_running": self.is_running,
            "queue_size": self._jm.queue_size,
        }
