"""Tests for scenario generator."""

from __future__ import annotations

import pytest

from isaac_mcp.scenario_lab.scenario_generator import ScenarioGenerator, DEFAULT_RANDOMIZATION


def test_generate_creates_parameters():
    gen = ScenarioGenerator(seed=42)
    scenario = gen.generate("base_test", scenario_index=0)

    assert scenario.base_scenario_id == "base_test"
    assert scenario.scenario_id == "base_test_rand_0"
    assert "floor_friction" in scenario.parameters
    assert "gravity_scale" in scenario.parameters
    assert "obstacle_count" in scenario.parameters
    assert "terrain_type" in scenario.parameters


def test_generate_within_bounds():
    gen = ScenarioGenerator(seed=42)
    for i in range(20):
        scenario = gen.generate("test", scenario_index=i)
        params = scenario.parameters
        assert 0.1 <= params["floor_friction"] <= 1.0
        assert 0.8 <= params["gravity_scale"] <= 1.2
        assert 0 <= params["obstacle_count"] <= 15
        assert params["terrain_type"] in ("flat", "inclined", "rough")
        assert 0.0 <= params["payload_mass"] <= 10.0
        assert 0.0 <= params["sensor_noise_scale"] <= 0.5
        assert params["lighting"] in ("bright", "dim", "dark")


def test_generate_custom_config():
    gen = ScenarioGenerator(seed=42)
    config = {"floor_friction": {"min": 0.5, "max": 0.6}}
    scenario = gen.generate("test", randomization_config=config)

    assert 0.5 <= scenario.parameters["floor_friction"] <= 0.6


def test_generate_batch():
    gen = ScenarioGenerator(seed=42)
    scenarios = gen.generate_batch("test", count=5)

    assert len(scenarios) == 5
    ids = {s.scenario_id for s in scenarios}
    assert len(ids) == 5  # All unique IDs


def test_seeded_reproducibility():
    gen1 = ScenarioGenerator(seed=123)
    gen2 = ScenarioGenerator(seed=123)
    s1 = gen1.generate("test", scenario_index=0)
    s2 = gen2.generate("test", scenario_index=0)

    assert s1.parameters == s2.parameters
