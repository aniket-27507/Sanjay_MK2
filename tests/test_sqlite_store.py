"""Tests for SQLite experiment store."""

from __future__ import annotations

import pytest
import pytest_asyncio

from isaac_mcp.storage.sqlite_store import ExperimentStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ExperimentStore(db_path=str(tmp_path / "test.db"))
    await s.init_db()
    return s


@pytest.mark.asyncio
async def test_init_creates_tables(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "test.db"))
    await store.init_db()
    # Calling init again should be idempotent
    await store.init_db()


@pytest.mark.asyncio
async def test_save_and_get_experiment(store):
    exp_id = await store.save_experiment("scenario_a", "batch", {"count": 5})
    assert exp_id

    result = await store.get_experiment(exp_id)
    assert result is not None
    assert result["scenario_id"] == "scenario_a"
    assert result["type"] == "batch"
    assert result["config"]["count"] == 5
    assert result["summary"]["total_runs"] == 0


@pytest.mark.asyncio
async def test_save_runs_and_summary(store):
    exp_id = await store.save_experiment("scenario_b", "batch")

    await store.save_run(exp_id, 0, success=True, duration_s=1.5)
    await store.save_run(exp_id, 1, success=False, duration_s=2.0, failure_reason="timeout")
    await store.save_run(exp_id, 2, success=True, duration_s=1.2)

    result = await store.get_experiment(exp_id)
    assert result is not None
    assert result["summary"]["total_runs"] == 3
    assert result["summary"]["successes"] == 2
    assert result["summary"]["failures"] == 1
    assert result["summary"]["success_rate"] == pytest.approx(2 / 3, abs=0.01)


@pytest.mark.asyncio
async def test_list_experiments(store):
    await store.save_experiment("s1", "batch")
    await store.save_experiment("s2", "sweep")

    experiments = await store.list_experiments(limit=10)
    assert len(experiments) == 2


@pytest.mark.asyncio
async def test_get_experiment_not_found(store):
    result = await store.get_experiment("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_sweep_results_grouping(store):
    exp_id = await store.save_experiment("s1", "sweep", {"parameter": "friction"})

    await store.save_run(exp_id, 0, True, 1.0, telemetry={"sweep_value": 0.1})
    await store.save_run(exp_id, 1, True, 1.1, telemetry={"sweep_value": 0.1})
    await store.save_run(exp_id, 2, False, 2.0, failure_reason="fell", telemetry={"sweep_value": 0.5})

    result = await store.get_sweep_results(exp_id)
    assert result is not None
    assert result["parameter"] == "friction"
    assert len(result["sweep_points"]) == 2  # Two distinct values


@pytest.mark.asyncio
async def test_scenario_save_and_results(store):
    sid = await store.save_scenario("base_01", {"friction": 0.3, "gravity_scale": 1.0})
    assert sid

    await store.save_scenario_result(sid, success=True, duration_s=2.5)
    await store.save_scenario_result(sid, success=False, failure_type="robot_fell", duration_s=1.0)

    results = await store.get_scenario_results([sid])
    assert len(results) == 2
    assert results[0]["success"] or results[1]["success"]
