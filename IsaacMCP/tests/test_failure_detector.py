"""Tests for failure detector."""

from __future__ import annotations

import pytest

from isaac_mcp.scenario_lab.failure_detector import (
    FailureDetector,
    ROBOT_FELL,
    PHYSICS_INSTABILITY,
    TIMEOUT,
    NAVIGATION_FAILED,
)


def test_detect_no_failures():
    detector = FailureDetector()
    telemetry = {
        "robots": [{"name": "d0", "position": [0, 0, 1.0], "velocity": [0, 0, 0], "status": "ok"}],
        "physics": {},
    }
    failures = detector.detect(telemetry)
    assert len(failures) == 0


def test_detect_robot_fell():
    detector = FailureDetector(ground_height=0.1)
    telemetry = {
        "robots": [{"name": "d0", "position": [0, 0, -0.5], "status": "ok"}],
    }
    failures = detector.detect(telemetry)
    assert any(f.failure_type == ROBOT_FELL for f in failures)


def test_detect_crashed_status():
    detector = FailureDetector()
    telemetry = {
        "robots": [{"name": "d0", "position": [0, 0, 1.0], "status": "crashed"}],
    }
    failures = detector.detect(telemetry)
    assert any(f.failure_type == ROBOT_FELL for f in failures)


def test_detect_extreme_velocity():
    detector = FailureDetector(velocity_threshold=50.0)
    telemetry = {
        "robots": [{"name": "d0", "position": [0, 0, 1.0], "velocity": [100, 0, 0], "status": "ok"}],
    }
    failures = detector.detect(telemetry)
    assert any(f.failure_type == PHYSICS_INSTABILITY for f in failures)


def test_detect_timeout():
    detector = FailureDetector()
    failures = detector.detect({}, timeout_s=60.0, duration_s=61.0, failure_reason="simulation_timeout")
    assert any(f.failure_type == TIMEOUT for f in failures)


def test_detect_physics_solver_errors():
    detector = FailureDetector()
    telemetry = {"robots": [], "physics": {"solver_errors": 5}}
    failures = detector.detect(telemetry)
    assert any(f.failure_type == PHYSICS_INSTABILITY for f in failures)


def test_get_primary_failure_type():
    detector = FailureDetector()
    telemetry = {
        "robots": [
            {"name": "d0", "position": [0, 0, -1.0], "velocity": [100, 0, 0], "status": "crashed"},
        ],
    }
    failures = detector.detect(telemetry)
    primary = detector.get_primary_failure_type(failures)
    assert primary  # Should return a non-empty failure type


def test_no_robots_no_failure():
    detector = FailureDetector()
    failures = detector.detect({"robots": [], "physics": {}})
    assert len(failures) == 0


def test_empty_primary_failure():
    detector = FailureDetector()
    assert detector.get_primary_failure_type([]) == ""
