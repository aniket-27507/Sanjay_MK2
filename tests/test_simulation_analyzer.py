"""Unit tests for the SimulationAnalyzer."""

from __future__ import annotations

import math

import pytest

from isaac_mcp.diagnostics.simulation_analyzer import SimulationAnalyzer
from isaac_mcp.error_patterns import ERROR_PATTERNS


@pytest.fixture
def analyzer():
    return SimulationAnalyzer(error_patterns=ERROR_PATTERNS)


def test_healthy_simulation(analyzer):
    telemetry = {
        "robots": [{"name": "ur5", "position": [0, 0, 1.0], "velocity": [0, 0, 0]}],
        "physics": {},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert diagnosis.root_cause == "no_issues_detected"
    assert diagnosis.category == "healthy"
    assert diagnosis.confidence == 1.0
    assert len(diagnosis.issues) == 0


def test_physics_nan_detection(analyzer):
    telemetry = {
        "robots": [{"name": "robot_0", "position": [float("nan"), 0, 1.0]}],
        "physics": {},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert len(diagnosis.issues) > 0
    assert any("NaN/Inf" in i.description for i in diagnosis.issues)
    assert diagnosis.category == "physics"


def test_robot_fell_detection(analyzer):
    telemetry = {
        "robots": [{"name": "drone_0", "position": [0, 0, -5.0]}],
        "physics": {},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert any("fallen" in i.description for i in diagnosis.issues)


def test_excessive_velocity_detection(analyzer):
    telemetry = {
        "robots": [{"name": "fast_bot", "velocity": [100, 100, 0]}],
        "physics": {},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert any("excessive velocity" in i.description for i in diagnosis.issues)


def test_log_error_classification(analyzer):
    log_entries = [
        {"raw_line": "2026-01-01 10:10:10.123 [Error] [omni.physics] PhysX Error in solver"},
        {"raw_line": "2026-01-01 10:10:11.123 [Error] [omni.kit] Out of GPU memory"},
    ]
    diagnosis = analyzer.analyze({"robots": [], "physics": {}}, log_entries, {})

    assert len(diagnosis.issues) >= 2
    categories = {i.category for i in diagnosis.issues}
    assert "physics" in categories
    assert "rendering" in categories


def test_correlated_evidence_increases_confidence(analyzer):
    telemetry = {
        "robots": [{"name": "bot", "position": [float("nan"), 0, 0]}],
        "physics": {},
    }
    log_entries = [
        {"raw_line": "2026-01-01 10:10:10.123 [Error] [omni.physics] Physics step NaN detected"},
    ]
    diagnosis = analyzer.analyze(telemetry, log_entries, {})

    # Both log and physics evidence → higher confidence
    assert diagnosis.confidence >= 0.9


def test_suggested_fixes_generated(analyzer):
    telemetry = {
        "robots": [{"name": "bot", "position": [0, 0, -3.0]}],
        "physics": {},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert len(diagnosis.suggested_fixes) > 0
    assert any("center of mass" in f.description.lower() for f in diagnosis.suggested_fixes)


def test_excessive_contacts_detection(analyzer):
    contacts = [{"a": i, "b": i + 1} for i in range(60)]
    telemetry = {
        "robots": [],
        "physics": {"contacts": contacts},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert any("contacts" in i.description.lower() for i in diagnosis.issues)


def test_active_collisions_detection(analyzer):
    telemetry = {
        "robots": [],
        "physics": {"collisions": [{"bodyA": "robot", "bodyB": "wall"}]},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})

    assert any("collision" in i.description.lower() for i in diagnosis.issues)


def test_to_dict_serialization(analyzer):
    telemetry = {
        "robots": [{"name": "bot", "position": [0, 0, -2.0]}],
        "physics": {},
    }
    diagnosis = analyzer.analyze(telemetry, [], {})
    d = diagnosis.to_dict()

    assert isinstance(d, dict)
    assert "issues" in d
    assert "root_cause" in d
    assert "confidence" in d
    assert "suggested_fixes" in d
    assert isinstance(d["issues"], list)
