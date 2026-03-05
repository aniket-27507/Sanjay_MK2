"""Regression test suite management.

A regression suite is a named collection of test cases, each specifying
a scenario to run with expected outcomes and pass/fail criteria.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SuiteTestCase:
    """A single test case within a regression suite."""

    name: str
    scenario_id: str
    run_count: int = 3
    timeout_s: float = 60.0
    min_success_rate: float = 1.0
    expected_failure_types: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SuiteTestCase:
        return SuiteTestCase(
            name=data.get("name", ""),
            scenario_id=data.get("scenario_id", ""),
            run_count=data.get("run_count", 3),
            timeout_s=data.get("timeout_s", 60.0),
            min_success_rate=data.get("min_success_rate", 1.0),
            expected_failure_types=data.get("expected_failure_types", []),
            parameters=data.get("parameters", {}),
        )


@dataclass(slots=True)
class RegressionSuite:
    """A named regression test suite."""

    suite_id: str
    name: str
    description: str = ""
    test_cases: list[TestCase] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "total_test_cases": len(self.test_cases),
        }


class SuiteManager:
    """CRUD manager for regression test suites.

    Suites are persisted as JSON files in a directory.
    """

    def __init__(self, storage_dir: str = "data/suites") -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._suites: dict[str, RegressionSuite] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all suites from disk."""
        for path in self._dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                suite = self._parse_suite(data)
                self._suites[suite.suite_id] = suite
            except Exception:
                pass

    def _parse_suite(self, data: dict[str, Any]) -> RegressionSuite:
        test_cases = [SuiteTestCase.from_dict(tc) for tc in data.get("test_cases", [])]
        return RegressionSuite(
            suite_id=data.get("suite_id", uuid.uuid4().hex[:12]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            test_cases=test_cases,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            tags=data.get("tags", []),
        )

    def _save_suite(self, suite: RegressionSuite) -> None:
        path = self._dir / f"{suite.suite_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(suite.to_dict(), f, indent=2)

    def create_suite(
        self,
        name: str,
        description: str = "",
        test_cases: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
    ) -> RegressionSuite:
        """Create a new regression suite."""
        now = datetime.now(timezone.utc).isoformat()
        suite = RegressionSuite(
            suite_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            test_cases=[SuiteTestCase.from_dict(tc) for tc in (test_cases or [])],
            created_at=now,
            updated_at=now,
            tags=tags or [],
        )
        self._suites[suite.suite_id] = suite
        self._save_suite(suite)
        return suite

    def get_suite(self, suite_id: str) -> RegressionSuite | None:
        """Retrieve a suite by ID or name."""
        if suite_id in self._suites:
            return self._suites[suite_id]
        # Search by name
        for suite in self._suites.values():
            if suite.name == suite_id:
                return suite
        return None

    def list_suites(self) -> list[RegressionSuite]:
        """Return all suites."""
        return list(self._suites.values())

    def add_test_case(self, suite_id: str, test_case: dict[str, Any]) -> RegressionSuite | None:
        """Add a test case to an existing suite."""
        suite = self.get_suite(suite_id)
        if suite is None:
            return None
        suite.test_cases.append(SuiteTestCase.from_dict(test_case))
        suite.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_suite(suite)
        return suite

    def remove_test_case(self, suite_id: str, test_name: str) -> RegressionSuite | None:
        """Remove a test case by name from a suite."""
        suite = self.get_suite(suite_id)
        if suite is None:
            return None
        suite.test_cases = [tc for tc in suite.test_cases if tc.name != test_name]
        suite.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_suite(suite)
        return suite

    def delete_suite(self, suite_id: str) -> bool:
        """Delete a suite by ID."""
        if suite_id not in self._suites:
            return False
        del self._suites[suite_id]
        path = self._dir / f"{suite_id}.json"
        if path.exists():
            path.unlink()
        return True
