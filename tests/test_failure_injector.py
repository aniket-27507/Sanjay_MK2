"""Tests for the failure injector."""

import pytest

from isaac_mcp.scenario_lab.failure_injector import FailureInjector, FaultChain, FaultEvent


@pytest.fixture
def injector():
    return FailureInjector()


def test_build_chain_from_sequence(injector):
    sequence = [
        {"time_s": 0, "fault_type": "motor_degradation", "params": {"reduction_pct": 50}},
        {"time_s": 5, "fault_type": "sensor_noise", "params": {"lidar_noise": 1.5}},
    ]
    chain = injector.build_chain_from_sequence(sequence, name="test_chain")
    assert chain.name == "test_chain"
    assert len(chain.events) == 2
    assert chain.events[0].time_s <= chain.events[1].time_s


def test_build_correlated_chain(injector):
    chain = injector.build_correlated_chain(
        base_fault="motor_degradation",
        secondary_faults=["sensor_noise", "wind_gust"],
        base_time=5.0,
        interval=2.0,
    )
    assert len(chain.events) == 3
    assert chain.events[0].fault_type == "motor_degradation"
    assert chain.events[0].time_s == 5.0
    assert chain.events[1].time_s == 7.0
    assert chain.events[2].time_s == 9.0


def test_chain_to_dict(injector):
    chain = FaultChain(
        name="test",
        events=[
            FaultEvent(time_s=0, fault_type="test_fault"),
            FaultEvent(time_s=5, fault_type="test_fault_2", duration_s=10),
        ],
        description="test chain",
    )
    d = chain.to_dict()
    assert d["total_events"] == 2
    assert d["duration_s"] == 15.0


def test_generate_kit_script(injector):
    chain = FaultChain(
        name="adversarial",
        events=[
            FaultEvent(time_s=0, fault_type="motor_degradation", params={"reduction_pct": 50}),
            FaultEvent(time_s=5, fault_type="sensor_noise", params={"lidar_noise": 1.5}),
        ],
    )
    script = injector.generate_kit_script_for_chain(chain)
    assert "motor_degradation" in script
    assert "sensor_noise" in script
    assert "import asyncio" in script


@pytest.mark.asyncio
async def test_execute_chain(injector):
    events_logged = []

    async def mock_inject(fault_type, drone_id, duration):
        events_logged.append((fault_type, drone_id))
        return True

    chain = FaultChain(
        name="test",
        events=[
            FaultEvent(time_s=0, fault_type="fault_a"),
            FaultEvent(time_s=0.1, fault_type="fault_b"),
        ],
    )
    result = await injector.execute_chain(chain, mock_inject, sim_start_time=0)
    assert result.events_injected == 2
    assert result.events_failed == 0
    assert len(events_logged) == 2


@pytest.mark.asyncio
async def test_execute_chain_handles_failure(injector):
    async def failing_inject(fault_type, drone_id, duration):
        return False

    chain = FaultChain(name="test", events=[FaultEvent(time_s=0, fault_type="fault_a")])
    result = await injector.execute_chain(chain, failing_inject)
    assert result.events_injected == 0
    assert result.events_failed == 1
    assert len(result.errors) == 1
