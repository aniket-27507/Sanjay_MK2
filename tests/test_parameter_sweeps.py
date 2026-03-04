"""Tests for parameter sweep engine."""

from __future__ import annotations

import pytest

from isaac_mcp.autonomous_loop.simulation_runner import SimulationResult
from isaac_mcp.experiments.parameter_sweeps import ParameterSweeper
from isaac_mcp.storage.sqlite_store import ExperimentStore


class FakeRunner:
    def __init__(self):
        self.call_count = 0

    async def run_with_monitoring(self, ws, kit, ssh, scenario_id, timeout_s=60.0):
        self.call_count += 1
        return SimulationResult(success=True, duration_s=1.0)


class FakeWS:
    async def send_command(self, command, **params):
        return {}

    def get_cached_state(self):
        return {"drones": []}


@pytest.mark.asyncio
async def test_sweep_basic():
    runner = FakeRunner()
    sweeper = ParameterSweeper(runner=runner)
    result = await sweeper.sweep(
        ws=FakeWS(), kit=None, ssh=None,
        scenario_id="test", parameter="friction",
        min_val=0.1, max_val=0.5, steps=3, runs_per_value=2, timeout_s=1.0,
    )

    assert result.parameter == "friction"
    assert len(result.sweep_points) == 3
    assert result.total_runs == 6  # 3 steps * 2 runs
    assert runner.call_count == 6

    # Check values are evenly spaced
    values = [sp.parameter_value for sp in result.sweep_points]
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(0.3)
    assert values[2] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_sweep_single_step():
    runner = FakeRunner()
    sweeper = ParameterSweeper(runner=runner)
    result = await sweeper.sweep(
        ws=FakeWS(), kit=None, ssh=None,
        scenario_id="test", parameter="gravity",
        min_val=1.0, max_val=1.0, steps=1, runs_per_value=3, timeout_s=1.0,
    )

    assert len(result.sweep_points) == 1
    assert result.total_runs == 3


@pytest.mark.asyncio
async def test_sweep_with_store(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "test.db"))
    runner = FakeRunner()
    sweeper = ParameterSweeper(runner=runner, store=store)

    result = await sweeper.sweep(
        ws=FakeWS(), kit=None, ssh=None,
        scenario_id="test", parameter="friction",
        min_val=0.1, max_val=0.2, steps=2, runs_per_value=2, timeout_s=1.0,
    )

    assert result.experiment_id
    sweep_data = await store.get_sweep_results(result.experiment_id)
    assert sweep_data is not None
    assert sweep_data["parameter"] == "friction"
    assert len(sweep_data["sweep_points"]) == 2
