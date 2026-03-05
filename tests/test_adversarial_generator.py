"""Tests for adversarial scenario generation."""

import pytest

from isaac_mcp.scenario_lab.adversarial_generator import (
    AdversarialGenerator,
    ADVERSARIAL_PROFILES,
)


@pytest.fixture
def gen():
    return AdversarialGenerator(seed=42)


def test_generate_from_profile(gen):
    scenario = gen.generate_from_profile("base_scene", "sensor_blackout")
    assert scenario.profile == "sensor_blackout"
    assert scenario.base_scenario_id == "base_scene"
    assert "camera_occlusion_pct" in scenario.parameters
    assert len(scenario.fault_sequence) > 0


def test_generate_from_all_profiles(gen):
    for profile_name in ADVERSARIAL_PROFILES:
        scenario = gen.generate_from_profile("base", profile_name)
        assert scenario.profile == profile_name
        assert len(scenario.parameters) > 0


def test_generate_from_invalid_profile(gen):
    with pytest.raises(ValueError, match="Unknown profile"):
        gen.generate_from_profile("base", "nonexistent_profile")


def test_generate_random(gen):
    scenario = gen.generate_random("base_scene")
    assert scenario.profile == "random"
    assert len(scenario.parameters) == 5  # default num_params
    assert scenario.severity in ("low", "medium", "high", "extreme")


def test_generate_random_custom_params(gen):
    scenario = gen.generate_random("base", num_params=3)
    assert len(scenario.parameters) == 3


def test_generate_campaign(gen):
    scenarios = gen.generate_campaign("base", count=10, include_profiles=True)
    assert len(scenarios) == 10
    # First 5 should be from profiles
    profile_scenarios = [s for s in scenarios if s.profile != "random"]
    assert len(profile_scenarios) == 5  # 5 predefined profiles


def test_generate_campaign_random_only(gen):
    scenarios = gen.generate_campaign("base", count=5, include_profiles=False)
    assert len(scenarios) == 5
    assert all(s.profile == "random" for s in scenarios)


def test_fault_sequence_generation(gen):
    scenario = gen.generate_from_profile("base", "combined_failure")
    assert len(scenario.fault_sequence) > 0
    # Faults should be time-ordered
    times = [f["time_s"] for f in scenario.fault_sequence]
    assert times == sorted(times)


def test_severity_estimation(gen):
    # Extreme parameters should produce high/extreme severity
    scenario = gen.generate_from_profile("base", "combined_failure")
    assert scenario.severity in ("high", "extreme")


def test_list_profiles():
    profiles = AdversarialGenerator.list_profiles()
    assert len(profiles) == 5
    assert all("name" in p and "description" in p for p in profiles)


def test_to_dict(gen):
    scenario = gen.generate_random("base")
    d = scenario.to_dict()
    assert "scenario_id" in d
    assert "parameters" in d
    assert "fault_sequence" in d
    assert "severity" in d
