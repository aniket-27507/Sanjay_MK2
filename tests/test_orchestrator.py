"""Tests for job orchestration: JobManager, Scheduler, WorkerPool."""

import asyncio

import pytest

from isaac_mcp.orchestrator.job_manager import (
    Job,
    JobManager,
    JobPriority,
    JobStatus,
)
from isaac_mcp.orchestrator.scheduler import Scheduler
from isaac_mcp.orchestrator.worker_pool import Worker, WorkerPool, WorkerState


# --- JobManager tests ---


class TestJobManager:
    def test_create_job(self):
        jm = JobManager()
        job = jm.create_job("batch_experiment", payload={"scenario": "s1"})
        assert job.job_type == "batch_experiment"
        assert job.status == JobStatus.QUEUED
        assert job.payload == {"scenario": "s1"}
        assert job.created_at != ""

    def test_get_job(self):
        jm = JobManager()
        job = jm.create_job("test")
        found = jm.get_job(job.job_id)
        assert found is not None
        assert found.job_id == job.job_id

    def test_dequeue_priority_order(self):
        jm = JobManager()
        low = jm.create_job("low_job", priority="low")
        high = jm.create_job("high_job", priority="high")
        urgent = jm.create_job("urgent_job", priority="urgent")

        first = jm.dequeue()
        assert first is not None
        assert first.job_id == urgent.job_id

        second = jm.dequeue()
        assert second is not None
        assert second.job_id == high.job_id

        third = jm.dequeue()
        assert third is not None
        assert third.job_id == low.job_id

    def test_dequeue_empty(self):
        jm = JobManager()
        assert jm.dequeue() is None

    def test_cancel_job(self):
        jm = JobManager()
        job = jm.create_job("test")
        cancelled = jm.cancel_job(job.job_id)
        assert cancelled is not None
        assert cancelled.status == JobStatus.CANCELLED
        # Cancelled jobs are skipped when dequeuing
        assert jm.dequeue() is None

    def test_cancel_terminal_job(self):
        jm = JobManager()
        job = jm.create_job("test")
        jm.cancel_job(job.job_id)
        # Can't cancel an already-cancelled job
        assert jm.cancel_job(job.job_id) is None

    def test_mark_running(self):
        jm = JobManager()
        job = jm.create_job("test")
        dequeued = jm.dequeue()
        running = jm.mark_running(dequeued.job_id, instance="primary")
        assert running is not None
        assert running.status == JobStatus.RUNNING
        assert running.instance == "primary"

    def test_mark_completed(self):
        jm = JobManager()
        job = jm.create_job("test")
        jm.dequeue()
        jm.mark_running(job.job_id)
        completed = jm.mark_completed(job.job_id, result={"success": True})
        assert completed is not None
        assert completed.status == JobStatus.COMPLETED
        assert completed.result == {"success": True}

    def test_mark_failed_with_retry(self):
        jm = JobManager()
        job = jm.create_job("test", max_retries=2)
        jm.dequeue()
        jm.mark_running(job.job_id)
        failed = jm.mark_failed(job.job_id, error="timeout")
        assert failed is not None
        # Should be re-queued (retries=1 < max_retries=2)
        assert failed.status == JobStatus.QUEUED
        assert failed.retries == 1

    def test_mark_failed_no_retries(self):
        jm = JobManager()
        job = jm.create_job("test", max_retries=1)
        jm.dequeue()
        jm.mark_running(job.job_id)
        failed = jm.mark_failed(job.job_id, error="crash")
        assert failed is not None
        assert failed.status == JobStatus.FAILED
        assert failed.error == "crash"

    def test_mark_timed_out(self):
        jm = JobManager()
        job = jm.create_job("test")
        jm.dequeue()
        jm.mark_running(job.job_id)
        timed_out = jm.mark_timed_out(job.job_id)
        assert timed_out is not None
        assert timed_out.status == JobStatus.TIMED_OUT

    def test_list_jobs(self):
        jm = JobManager()
        jm.create_job("type_a")
        jm.create_job("type_b")
        jm.create_job("type_a")
        assert len(jm.list_jobs()) == 3
        assert len(jm.list_jobs(job_type="type_a")) == 2

    def test_list_jobs_by_status(self):
        jm = JobManager()
        jm.create_job("test")
        job2 = jm.create_job("test")
        jm.dequeue()
        jm.mark_running(job2.job_id)
        # One queued, one running
        queued = jm.list_jobs(status="queued")
        running = jm.list_jobs(status="running")
        assert len(queued) == 1
        assert len(running) == 1

    def test_queue_size_and_running_count(self):
        jm = JobManager()
        jm.create_job("test")
        jm.create_job("test")
        assert jm.queue_size == 2
        assert jm.running_count == 0

        job = jm.dequeue()
        jm.mark_running(job.job_id)
        assert jm.queue_size == 1
        assert jm.running_count == 1

    def test_get_stats(self):
        jm = JobManager()
        jm.create_job("batch")
        jm.create_job("sweep")
        stats = jm.get_stats()
        assert stats["total_jobs"] == 2
        assert stats["queue_size"] == 2
        assert "batch" in stats["by_type"]

    def test_job_to_dict(self):
        jm = JobManager()
        job = jm.create_job("test", tags=["regression"])
        d = job.to_dict()
        assert d["job_type"] == "test"
        assert d["tags"] == ["regression"]
        assert d["priority"] == "normal"

    def test_is_terminal(self):
        jm = JobManager()
        job = jm.create_job("test")
        assert not job.is_terminal
        jm.cancel_job(job.job_id)
        assert job.is_terminal

    def test_trim_history(self):
        jm = JobManager(max_history=2)
        jobs = [jm.create_job("test") for _ in range(5)]
        for job in jobs:
            jm.dequeue()
            jm.mark_running(job.job_id)
            jm.mark_completed(job.job_id)
        # Should have trimmed old terminal jobs
        assert len(jm.list_jobs()) <= 5  # some may be trimmed


# --- Scheduler tests ---


class TestScheduler:
    @pytest.mark.asyncio
    async def test_dispatch_one(self):
        jm = JobManager()
        sched = Scheduler(jm, max_concurrent=2)
        jm.create_job("test")
        job = sched.dispatch_one()
        assert job is not None
        assert job.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_dispatch_respects_concurrency(self):
        jm = JobManager()
        sched = Scheduler(jm, max_concurrent=1)
        jm.create_job("test")
        jm.create_job("test")
        first = sched.dispatch_one()
        assert first is not None
        # After dispatching one, the queue still has one job
        assert jm.queue_size == 1

    @pytest.mark.asyncio
    async def test_scheduler_with_executor(self):
        jm = JobManager()
        sched = Scheduler(jm, max_concurrent=2, poll_interval_s=0.05)

        async def mock_executor(job):
            return {"done": True}

        sched.set_executor(mock_executor)
        jm.create_job("test")
        jm.create_job("test")

        await sched.start()
        await asyncio.sleep(0.2)
        await sched.stop()

        completed = jm.list_jobs(status="completed")
        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_scheduler_handles_failure(self):
        jm = JobManager()
        sched = Scheduler(jm, max_concurrent=2, poll_interval_s=0.05)

        async def failing_executor(job):
            raise RuntimeError("boom")

        sched.set_executor(failing_executor)
        jm.create_job("test", max_retries=1)

        await sched.start()
        await asyncio.sleep(0.2)
        await sched.stop()

        failed = jm.list_jobs(status="failed")
        assert len(failed) == 1
        assert failed[0].error == "boom"

    @pytest.mark.asyncio
    async def test_scheduler_stats(self):
        jm = JobManager()
        sched = Scheduler(jm, max_concurrent=4)
        stats = sched.get_stats()
        assert stats["max_concurrent"] == 4
        assert stats["available_slots"] == 4


# --- WorkerPool tests ---


class TestWorkerPool:
    def test_add_and_get_worker(self):
        pool = WorkerPool()
        worker = pool.add_worker("w1", "primary")
        assert worker.worker_id == "w1"
        assert worker.instance_name == "primary"
        assert pool.get_worker("w1") is not None

    def test_remove_worker(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        assert pool.remove_worker("w1")
        assert pool.get_worker("w1") is None
        assert not pool.remove_worker("w1")

    def test_acquire_and_release(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        pool.add_worker("w2", "secondary")

        worker = pool.acquire_worker()
        assert worker is not None
        assert worker.state == WorkerState.BUSY
        assert pool.available_workers == 1

        pool.release_worker(worker.worker_id, success=True)
        assert worker.state == WorkerState.IDLE
        assert worker.jobs_completed == 1
        assert pool.available_workers == 2

    def test_acquire_none_available(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        pool.acquire_worker()
        assert pool.acquire_worker() is None

    def test_release_failure(self):
        pool = WorkerPool(max_consecutive_failures=2)
        pool.add_worker("w1", "primary")

        w = pool.acquire_worker()
        pool.release_worker(w.worker_id, success=False)
        assert w.state == WorkerState.IDLE  # 1 failure < 2 threshold
        assert w.consecutive_failures == 1

        w = pool.acquire_worker()
        pool.release_worker(w.worker_id, success=False)
        assert w.state == WorkerState.UNHEALTHY  # 2 failures >= 2 threshold

    def test_assign_job(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        worker = pool.acquire_worker()
        assert pool.assign_job("w1", "job123")
        assert worker.current_job_id == "job123"

    def test_drain_worker(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        assert pool.drain_worker("w1")
        assert pool.get_worker("w1").state == WorkerState.DRAINING
        # Drained workers are not available
        assert pool.acquire_worker() is None

    def test_recover_worker(self):
        pool = WorkerPool(max_consecutive_failures=1)
        pool.add_worker("w1", "primary")
        w = pool.acquire_worker()
        pool.release_worker(w.worker_id, success=False)
        assert w.state == WorkerState.UNHEALTHY

        assert pool.recover_worker("w1")
        assert w.state == WorkerState.IDLE
        assert w.consecutive_failures == 0

    def test_list_workers(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        pool.add_worker("w2", "secondary")
        workers = pool.list_workers()
        assert len(workers) == 2

    def test_get_stats(self):
        pool = WorkerPool()
        pool.add_worker("w1", "primary")
        pool.add_worker("w2", "secondary")
        pool.acquire_worker()
        stats = pool.get_stats()
        assert stats["total_workers"] == 2
        assert stats["busy"] == 1
        assert stats["available"] == 1

    def test_worker_to_dict(self):
        pool = WorkerPool()
        worker = pool.add_worker("w1", "primary")
        d = worker.to_dict()
        assert d["worker_id"] == "w1"
        assert d["state"] == "idle"

    @pytest.mark.asyncio
    async def test_health_check_recovers_worker(self):
        pool = WorkerPool(max_consecutive_failures=1, health_check_interval_s=0.05)
        pool.add_worker("w1", "primary")
        w = pool.acquire_worker()
        pool.release_worker(w.worker_id, success=False)
        assert w.state == WorkerState.UNHEALTHY

        async def healthy_check(worker):
            return True

        pool.set_health_check(healthy_check)
        await pool.start_health_checks()
        await asyncio.sleep(0.15)
        await pool.stop_health_checks()

        assert w.state == WorkerState.IDLE
