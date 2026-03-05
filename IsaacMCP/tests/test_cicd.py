"""Tests for CI/CD pipeline runner and regression suites."""

import json
import os

import pytest
import pytest_asyncio

from isaac_mcp.cicd.regression_suite import (
    RegressionSuite,
    SuiteManager,
    SuiteTestCase,
)
from isaac_mcp.cicd.pipeline_runner import (
    CaseResult,
    PipelineRunner,
    PipelineResult,
)


# --- SuiteTestCase tests ---


class TestSuiteTestCase:
    def test_to_dict(self):
        tc = SuiteTestCase(name="test1", scenario_id="s1", run_count=5)
        d = tc.to_dict()
        assert d["name"] == "test1"
        assert d["run_count"] == 5

    def test_from_dict(self):
        tc = SuiteTestCase.from_dict({
            "name": "test1",
            "scenario_id": "s1",
            "run_count": 10,
            "min_success_rate": 0.9,
        })
        assert tc.name == "test1"
        assert tc.run_count == 10
        assert tc.min_success_rate == 0.9


# --- SuiteManager tests ---


class TestSuiteManager:
    def test_create_suite(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        suite = mgr.create_suite(
            name="basic_tests",
            description="Basic regression tests",
            test_cases=[
                {"name": "hover", "scenario_id": "hover_test", "run_count": 3},
                {"name": "land", "scenario_id": "land_test", "run_count": 5},
            ],
            tags=["regression"],
        )
        assert suite.name == "basic_tests"
        assert len(suite.test_cases) == 2
        assert suite.tags == ["regression"]

    def test_get_suite_by_id(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        suite = mgr.create_suite(name="test_suite")
        found = mgr.get_suite(suite.suite_id)
        assert found is not None
        assert found.name == "test_suite"

    def test_get_suite_by_name(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        mgr.create_suite(name="my_suite")
        found = mgr.get_suite("my_suite")
        assert found is not None

    def test_list_suites(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        mgr.create_suite(name="suite1")
        mgr.create_suite(name="suite2")
        suites = mgr.list_suites()
        assert len(suites) == 2

    def test_add_test_case(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        suite = mgr.create_suite(name="test_suite")
        updated = mgr.add_test_case(suite.suite_id, {"name": "new_test", "scenario_id": "s1"})
        assert updated is not None
        assert len(updated.test_cases) == 1

    def test_remove_test_case(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        suite = mgr.create_suite(
            name="test_suite",
            test_cases=[
                {"name": "test1", "scenario_id": "s1"},
                {"name": "test2", "scenario_id": "s2"},
            ],
        )
        updated = mgr.remove_test_case(suite.suite_id, "test1")
        assert updated is not None
        assert len(updated.test_cases) == 1
        assert updated.test_cases[0].name == "test2"

    def test_delete_suite(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        suite = mgr.create_suite(name="to_delete")
        assert mgr.delete_suite(suite.suite_id)
        assert mgr.get_suite(suite.suite_id) is None

    def test_persistence(self, tmp_path):
        mgr1 = SuiteManager(str(tmp_path))
        suite = mgr1.create_suite(name="persistent", test_cases=[{"name": "t1", "scenario_id": "s1"}])

        # Create new manager, should load from disk
        mgr2 = SuiteManager(str(tmp_path))
        found = mgr2.get_suite(suite.suite_id)
        assert found is not None
        assert found.name == "persistent"
        assert len(found.test_cases) == 1

    def test_suite_to_dict(self, tmp_path):
        mgr = SuiteManager(str(tmp_path))
        suite = mgr.create_suite(name="s1", test_cases=[{"name": "t1", "scenario_id": "sc1"}])
        d = suite.to_dict()
        assert d["name"] == "s1"
        assert d["total_test_cases"] == 1


# --- PipelineRunner tests ---


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_run_suite_no_backend(self):
        """Without a run_fn, the runner simulates passes."""
        runner = PipelineRunner()
        suite = RegressionSuite(
            suite_id="s1",
            name="test_suite",
            test_cases=[
                SuiteTestCase(name="test1", scenario_id="sc1"),
                SuiteTestCase(name="test2", scenario_id="sc2"),
            ],
        )
        result = await runner.run_suite(suite)
        assert result.passed
        assert result.total_cases == 2
        assert result.passed_cases == 2
        assert result.failed_cases == 0

    @pytest.mark.asyncio
    async def test_run_suite_with_backend(self):
        """With a mock run_fn, test pass/fail logic."""
        async def mock_run(scenario_id, count, timeout_s, params):
            if scenario_id == "fail_scenario":
                return {
                    "success_rate": 0.5,
                    "successes": 1,
                    "failures": 1,
                    "total_runs": 2,
                    "runs": [{"failure_reason": "collision"}],
                }
            return {
                "success_rate": 1.0,
                "successes": count,
                "failures": 0,
                "total_runs": count,
                "runs": [],
            }

        runner = PipelineRunner(run_fn=mock_run)
        suite = RegressionSuite(
            suite_id="s1",
            name="mixed_suite",
            test_cases=[
                SuiteTestCase(name="pass_test", scenario_id="good_scenario"),
                SuiteTestCase(name="fail_test", scenario_id="fail_scenario", min_success_rate=1.0),
            ],
        )
        result = await runner.run_suite(suite)
        assert not result.passed
        assert result.passed_cases == 1
        assert result.failed_cases == 1

    @pytest.mark.asyncio
    async def test_stop_on_first_failure(self):
        async def mock_run(scenario_id, count, timeout_s, params):
            return {"success_rate": 0.0, "successes": 0, "failures": count, "total_runs": count, "runs": []}

        runner = PipelineRunner(run_fn=mock_run)
        suite = RegressionSuite(
            suite_id="s1",
            name="stop_early",
            test_cases=[
                SuiteTestCase(name="fail1", scenario_id="s1"),
                SuiteTestCase(name="skip1", scenario_id="s2"),
                SuiteTestCase(name="skip2", scenario_id="s3"),
            ],
        )
        result = await runner.run_suite(suite, stop_on_first_failure=True)
        assert result.failed_cases == 1
        assert result.skipped_cases == 2

    @pytest.mark.asyncio
    async def test_run_fn_exception(self):
        async def bad_run(scenario_id, count, timeout_s, params):
            raise RuntimeError("Connection lost")

        runner = PipelineRunner(run_fn=bad_run)
        suite = RegressionSuite(
            suite_id="s1",
            name="error_suite",
            test_cases=[SuiteTestCase(name="error_test", scenario_id="s1")],
        )
        result = await runner.run_suite(suite)
        assert not result.passed
        assert result.test_results[0].error == "Connection lost"

    @pytest.mark.asyncio
    async def test_to_junit_xml(self):
        runner = PipelineRunner()
        suite = RegressionSuite(
            suite_id="s1",
            name="junit_test",
            test_cases=[SuiteTestCase(name="test1", scenario_id="s1")],
        )
        result = await runner.run_suite(suite)
        xml = result.to_junit_xml()
        assert '<?xml version="1.0"' in xml
        assert 'name="junit_test"' in xml
        assert 'name="test1"' in xml

    @pytest.mark.asyncio
    async def test_list_results(self):
        runner = PipelineRunner()
        suite = RegressionSuite(
            suite_id="s1",
            name="test",
            test_cases=[SuiteTestCase(name="t1", scenario_id="s1")],
        )
        await runner.run_suite(suite)
        results = runner.list_results()
        assert len(results) == 1
        assert results[0]["suite_name"] == "test"

    @pytest.mark.asyncio
    async def test_get_result(self):
        runner = PipelineRunner()
        suite = RegressionSuite(
            suite_id="s1",
            name="test",
            test_cases=[SuiteTestCase(name="t1", scenario_id="s1")],
        )
        result = await runner.run_suite(suite)
        found = runner.get_result(result.pipeline_id)
        assert found is not None
        assert found.suite_name == "test"

    def test_pipeline_result_to_dict(self):
        result = PipelineResult(
            pipeline_id="p1",
            suite_id="s1",
            suite_name="test",
            passed=True,
            total_cases=1,
            passed_cases=1,
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "p1"
        assert d["passed"] is True
