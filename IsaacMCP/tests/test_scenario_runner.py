"""Tests for batch scenario runner."""

from __future__ import annotations

import pytest

from isaac_mcp.autonomous_loop.simulation_runner import SimulationResult
from isaac_mcp.experiments.scenario_runner import ScenarioRunner
from isaac_mcp.storage.sqlite_store import ExperimentStore


class FakeWS:
    def __init__(self, fail_on: set[int] | None = None):
        self.sent = []
        self._fail_on = fail_on or set()
        self._run_count = 0

    async def send_command(self, command, **params):
        self.sent.append((command, params))
        return {}

    def get_cached_state(self):
        self._run_count += 1
        if self._run_count in self._fail_on:
            return {"drones": [{"status": "crashed"}]}
        return {"drones": [{"status": "ok", "name": "d0", "position": [0, 0, 1]}]}


class FakeRunner:
    """Controllable mock runner for testing batch logic."""

    def __init__(self, results: list[SimulationResult] | None = None):
        self._results = results or []
        self._index = 0

    async def run_with_monitoring(self, ws, kit, ssh, scenario_id, timeout_s=60.0):
        if self._results:
            result = self._results[self._index % len(self._results)]
            self._index += 1
            return result
        return SimulationResult(success=True, duration_s=1.0)


@pytest.mark.asyncio
async def test_batch_run_all_success():
    runner = FakeRunner()
    scenario = ScenarioRunner(runner=runner)
    result = await scenario.run_batch(
        ws=FakeWS(), kit=None, ssh=None,
        scenario_id="test", count=3, timeout_s=1.0,
    )

    assert result.total_runs == 3
    assert result.successes == 3
    assert result.success_rate == 1.0


@pytest.mark.asyncio
async def test_batch_run_with_failures():
    results = [
        SimulationResult(success=True, duration_s=1.0),
        SimulationResult(success=False, duration_s=0.5, failure_reason="timeout"),
        SimulationResult(success=True, duration_s=1.2),
    ]
    runner = FakeRunner(results=results)
    scenario = ScenarioRunner(runner=runner)
    result = await scenario.run_batch(
        ws=FakeWS(), kit=None, ssh=None,
        scenario_id="test", count=3, timeout_s=1.0,
    )

    assert result.total_runs == 3
    assert result.successes == 2
    assert result.failures == 1
    assert result.success_rate == pytest.approx(2 / 3, abs=0.01)


@pytest.mark.asyncio
async def test_batch_run_with_store(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "test.db"))
    runner = FakeRunner()
    scenario = ScenarioRunner(runner=runner, store=store)

    result = await scenario.run_batch(
        ws=FakeWS(), kit=None, ssh=None,
        scenario_id="stored_test", count=2, timeout_s=1.0,
    )

    assert result.experiment_id
    exp = await store.get_experiment(result.experiment_id)
    assert exp is not None
    assert exp["summary"]["total_runs"] == 2
