"""Tests for robustness tester."""

from __future__ import annotations

import pytest

from isaac_mcp.autonomous_loop.simulation_runner import SimulationResult
from isaac_mcp.scenario_lab.robustness_tester import RobustnessTester
from isaac_mcp.scenario_lab.scenario_generator import ScenarioGenerator
from isaac_mcp.scenario_lab.failure_detector import FailureDetector
from isaac_mcp.storage.sqlite_store import ExperimentStore


class FakeRunner:
    def __init__(self, results=None):
        self._results = results or []
        self._index = 0

    async def run_with_monitoring(self, ws, kit, ssh, scenario_id, timeout_s=60.0):
        if self._results:
            result = self._results[self._index % len(self._results)]
            self._index += 1
            return result
        return SimulationResult(success=True, duration_s=1.0,
                                telemetry={"robots": [{"name": "d0", "position": [0, 0, 1], "status": "ok"}]})


class FakeWS:
    async def send_command(self, command, **params):
        return {}

    def get_cached_state(self):
        return {"drones": []}


@pytest.mark.asyncio
async def test_robustness_all_success():
    runner = FakeRunner()
    tester = RobustnessTester(runner=runner, generator=ScenarioGenerator(seed=42))

    report = await tester.run_robustness_test(
        ws=FakeWS(), kit=None, ssh=None,
        base_scenario_id="test", count=5, timeout_s=1.0,
    )

    assert report.total_runs == 5
    assert report.successes == 5
    assert report.success_rate == 1.0
    assert len(report.failure_breakdown) == 0


@pytest.mark.asyncio
async def test_robustness_with_failures():
    results = [
        SimulationResult(success=True, duration_s=1.0,
                        telemetry={"robots": [{"name": "d0", "position": [0, 0, 1], "status": "ok"}]}),
        SimulationResult(success=False, duration_s=0.5, failure_reason="simulation_timeout",
                        telemetry={"robots": []}),
        SimulationResult(success=True, duration_s=1.2,
                        telemetry={"robots": [{"name": "d0", "position": [0, 0, 1], "status": "ok"}]}),
    ]
    runner = FakeRunner(results=results)
    tester = RobustnessTester(runner=runner, generator=ScenarioGenerator(seed=42))

    report = await tester.run_robustness_test(
        ws=FakeWS(), kit=None, ssh=None,
        base_scenario_id="test", count=3, timeout_s=1.0,
    )

    assert report.total_runs == 3
    assert report.successes == 2
    assert report.failures == 1
    assert len(report.failure_breakdown) > 0


@pytest.mark.asyncio
async def test_robustness_with_store(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "test.db"))
    runner = FakeRunner()
    tester = RobustnessTester(runner=runner, generator=ScenarioGenerator(seed=42), store=store)

    report = await tester.run_robustness_test(
        ws=FakeWS(), kit=None, ssh=None,
        base_scenario_id="test", count=3, timeout_s=1.0,
    )

    assert report.test_id
    exp = await store.get_experiment(report.test_id)
    assert exp is not None


@pytest.mark.asyncio
async def test_robustness_report_structure():
    runner = FakeRunner()
    tester = RobustnessTester(runner=runner, generator=ScenarioGenerator(seed=42))

    report = await tester.run_robustness_test(
        ws=FakeWS(), kit=None, ssh=None,
        base_scenario_id="test", count=2, timeout_s=1.0,
    )

    d = report.to_dict()
    assert "test_id" in d
    assert "base_scenario_id" in d
    assert "scenario_results" in d
    assert len(d["scenario_results"]) == 2
    for sr in d["scenario_results"]:
        assert "parameters" in sr
        assert "success" in sr
