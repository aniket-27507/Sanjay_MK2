"""CI/CD pipeline runner for robotic regression testing.

Executes a regression suite's test cases against a simulation backend,
collects pass/fail results, and produces structured reports compatible
with CI systems (JSON summary, optional JUnit XML).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from isaac_mcp.cicd.regression_suite import RegressionSuite, SuiteTestCase


@dataclass(slots=True)
class CaseResult:
    """Result of running a single test case."""

    name: str
    scenario_id: str
    passed: bool
    success_rate: float = 0.0
    total_runs: int = 0
    successes: int = 0
    failures: int = 0
    duration_s: float = 0.0
    failure_reasons: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "success_rate": self.success_rate,
            "total_runs": self.total_runs,
            "successes": self.successes,
            "failures": self.failures,
            "duration_s": round(self.duration_s, 3),
            "failure_reasons": self.failure_reasons,
            "error": self.error,
        }


@dataclass(slots=True)
class PipelineResult:
    """Aggregate result of running a full regression suite."""

    pipeline_id: str
    suite_id: str
    suite_name: str
    passed: bool
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    skipped_cases: int = 0
    duration_s: float = 0.0
    test_results: list[CaseResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "suite_id": self.suite_id,
            "suite_name": self.suite_name,
            "passed": self.passed,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "skipped_cases": self.skipped_cases,
            "duration_s": round(self.duration_s, 3),
            "test_results": [tr.to_dict() for tr in self.test_results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    def to_junit_xml(self) -> str:
        """Export results as JUnit XML for CI system integration."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="{_xml_escape(self.suite_name)}" '
            f'tests="{self.total_cases}" '
            f'failures="{self.failed_cases}" '
            f'skipped="{self.skipped_cases}" '
            f'time="{self.duration_s:.3f}">',
        ]
        for tr in self.test_results:
            lines.append(
                f'  <testcase name="{_xml_escape(tr.name)}" '
                f'classname="{_xml_escape(tr.scenario_id)}" '
                f'time="{tr.duration_s:.3f}">'
            )
            if not tr.passed:
                reason = tr.error or "; ".join(tr.failure_reasons[:3]) or "Test failed"
                lines.append(
                    f'    <failure message="{_xml_escape(reason)}">'
                    f"success_rate={tr.success_rate:.2%} "
                    f"({tr.successes}/{tr.total_runs})"
                    f"</failure>"
                )
            lines.append("  </testcase>")
        lines.append("</testsuite>")
        return "\n".join(lines)


class PipelineRunner:
    """Execute regression suites and produce structured results.

    The runner accepts a ``run_fn`` callback that executes a single
    scenario batch. This decouples the runner from the simulation
    backend, making it testable without Isaac Sim.

    Parameters
    ----------
    run_fn:
        An async callable with signature::

            async def run_fn(
                scenario_id: str,
                count: int,
                timeout_s: float,
                params: dict,
            ) -> dict[str, Any]

        It must return a dict with keys: ``success_rate``, ``successes``,
        ``failures``, ``total_runs``, ``runs`` (list of run dicts).
    """

    def __init__(self, run_fn: Any = None) -> None:
        self._run_fn = run_fn
        self._results: dict[str, PipelineResult] = {}

    async def run_suite(
        self,
        suite: RegressionSuite,
        *,
        stop_on_first_failure: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute all test cases in a regression suite."""
        pipeline_id = uuid.uuid4().hex[:12]
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.monotonic()

        test_results: list[CaseResult] = []
        passed_cases = 0
        failed_cases = 0
        skipped_cases = 0

        for tc in suite.test_cases:
            if stop_on_first_failure and failed_cases > 0:
                skipped_cases += 1
                test_results.append(CaseResult(
                    name=tc.name,
                    scenario_id=tc.scenario_id,
                    passed=False,
                    error="Skipped due to prior failure",
                ))
                continue

            result = await self._run_test_case(tc)
            test_results.append(result)
            if result.passed:
                passed_cases += 1
            else:
                failed_cases += 1

        total_duration = time.monotonic() - start_time
        completed_at = datetime.now(timezone.utc).isoformat()

        pipeline_result = PipelineResult(
            pipeline_id=pipeline_id,
            suite_id=suite.suite_id,
            suite_name=suite.name,
            passed=(failed_cases == 0),
            total_cases=len(suite.test_cases),
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            skipped_cases=skipped_cases,
            duration_s=total_duration,
            test_results=test_results,
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata or {},
        )

        self._results[pipeline_id] = pipeline_result
        return pipeline_result

    async def _run_test_case(self, tc: SuiteTestCase) -> CaseResult:
        """Execute a single test case."""
        start = time.monotonic()

        if self._run_fn is None:
            # No backend: simulate a pass for testing
            return CaseResult(
                name=tc.name,
                scenario_id=tc.scenario_id,
                passed=True,
                success_rate=1.0,
                total_runs=tc.run_count,
                successes=tc.run_count,
                failures=0,
                duration_s=time.monotonic() - start,
            )

        try:
            batch_result = await self._run_fn(
                scenario_id=tc.scenario_id,
                count=tc.run_count,
                timeout_s=tc.timeout_s,
                params=tc.parameters,
            )

            success_rate = batch_result.get("success_rate", 0.0)
            successes = batch_result.get("successes", 0)
            failures = batch_result.get("failures", 0)
            total = batch_result.get("total_runs", tc.run_count)

            # Collect failure reasons
            failure_reasons = []
            for run in batch_result.get("runs", []):
                reason = run.get("failure_reason", "")
                if reason and reason not in failure_reasons:
                    failure_reasons.append(reason)

            passed = success_rate >= tc.min_success_rate

            return CaseResult(
                name=tc.name,
                scenario_id=tc.scenario_id,
                passed=passed,
                success_rate=success_rate,
                total_runs=total,
                successes=successes,
                failures=failures,
                duration_s=time.monotonic() - start,
                failure_reasons=failure_reasons,
            )

        except Exception as exc:
            return CaseResult(
                name=tc.name,
                scenario_id=tc.scenario_id,
                passed=False,
                duration_s=time.monotonic() - start,
                error=str(exc),
            )

    def get_result(self, pipeline_id: str) -> PipelineResult | None:
        """Retrieve a pipeline result by ID."""
        return self._results.get(pipeline_id)

    def list_results(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent pipeline results as summaries."""
        results = sorted(
            self._results.values(),
            key=lambda r: r.started_at,
            reverse=True,
        )[:limit]
        return [
            {
                "pipeline_id": r.pipeline_id,
                "suite_name": r.suite_name,
                "passed": r.passed,
                "total_cases": r.total_cases,
                "passed_cases": r.passed_cases,
                "failed_cases": r.failed_cases,
                "duration_s": round(r.duration_s, 3),
                "started_at": r.started_at,
            }
            for r in results
        ]


def _xml_escape(text: str) -> str:
    """Escape text for XML attribute values."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
