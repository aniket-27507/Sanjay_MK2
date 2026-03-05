"""Robotics CI/CD pipeline plugin.

Provides MCP tools for managing regression test suites and running
CI/CD pipelines against simulation scenarios.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.cicd.pipeline_runner import PipelineRunner
from isaac_mcp.cicd.regression_suite import SuiteManager
from isaac_mcp.plugin_host import PluginHost

_READONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def _success(data: Any) -> str:
    return json.dumps({"status": "ok", "data": data}, indent=2, default=str)


def _error(code: str, message: str) -> str:
    return json.dumps({"status": "error", "error": {"code": code, "message": message}})


def register(host: PluginHost) -> None:
    """Register CI/CD pipeline tools."""

    suite_mgr = SuiteManager()
    runner = PipelineRunner()

    @host.tool(
        name="create_regression_suite",
        description=(
            "Create a new regression test suite with named test cases. "
            "Each test case specifies a scenario_id, run_count, timeout, "
            "and minimum success rate threshold."
        ),
        annotations=_MUTATING,
        mutating=True,
    )
    async def create_regression_suite(
        name: str,
        description: str = "",
        test_cases_json: str = "[]",
        tags: str = "",
        instance: str = "primary",
    ) -> str:
        try:
            test_cases = json.loads(test_cases_json)
        except json.JSONDecodeError:
            return _error("invalid_json", "test_cases_json must be valid JSON array")

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        suite = suite_mgr.create_suite(
            name=name,
            description=description,
            test_cases=test_cases,
            tags=tag_list,
        )
        return _success(suite.to_dict())

    @host.tool(
        name="list_regression_suites",
        description="List all available regression test suites.",
        annotations=_READONLY,
    )
    async def list_regression_suites(instance: str = "primary") -> str:
        suites = suite_mgr.list_suites()
        return _success({
            "suites": [s.to_dict() for s in suites],
            "total": len(suites),
        })

    @host.tool(
        name="get_regression_suite",
        description="Get details of a regression suite by ID or name.",
        annotations=_READONLY,
    )
    async def get_regression_suite(
        suite_id: str,
        instance: str = "primary",
    ) -> str:
        suite = suite_mgr.get_suite(suite_id)
        if suite is None:
            return _error("not_found", f"Suite '{suite_id}' not found")
        return _success(suite.to_dict())

    @host.tool(
        name="run_regression_suite",
        description=(
            "Run a regression test suite against the simulation backend. "
            "Returns pass/fail results for each test case and an overall "
            "pipeline verdict. Optionally stops on first failure."
        ),
        annotations=_MUTATING,
        mutating=True,
    )
    async def run_regression_suite(
        suite_id: str,
        stop_on_first_failure: bool = False,
        instance: str = "primary",
    ) -> str:
        suite = suite_mgr.get_suite(suite_id)
        if suite is None:
            return _error("not_found", f"Suite '{suite_id}' not found")

        result = await runner.run_suite(
            suite,
            stop_on_first_failure=stop_on_first_failure,
        )
        return _success(result.to_dict())

    @host.tool(
        name="get_pipeline_result",
        description="Get the result of a previously run pipeline by ID.",
        annotations=_READONLY,
    )
    async def get_pipeline_result(
        pipeline_id: str,
        instance: str = "primary",
    ) -> str:
        result = runner.get_result(pipeline_id)
        if result is None:
            return _error("not_found", f"Pipeline '{pipeline_id}' not found")
        return _success(result.to_dict())

    @host.tool(
        name="list_pipeline_results",
        description="List recent pipeline execution results.",
        annotations=_READONLY,
    )
    async def list_pipeline_results(
        limit: int = 20,
        instance: str = "primary",
    ) -> str:
        results = runner.list_results(limit=limit)
        return _success({"results": results, "total": len(results)})
