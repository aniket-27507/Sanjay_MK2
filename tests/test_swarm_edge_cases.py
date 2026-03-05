"""
Project Sanjay Mk2 - Parameterized Swarm Edge Case Tests
=========================================================
Wires the 25 scenarios from swarm_edge_cases.py into pytest via
the FaultInjector, TaskRedistributor, and CBBAEngine.
"""

import time
import pytest
from typing import Dict, List

from tests.swarm_edge_cases import EDGE_TEST_SCENARIOS, TestScenario
from src.swarm.fault_injection import (
    FaultInjector,
    FaultType as RealFaultType,
    FaultSeverity as RealFaultSeverity,
    TaskRedistributor,
)
from src.swarm.cbba.cbba_engine import CBBAEngine, CBBAConfig
from src.swarm.cbba.task_types import SwarmTask, TaskType
from src.core.types.drone_types import DroneState, DroneType, Vector3

DRONE_COUNT = 6

FAULT_TYPE_MAP = {
    "MOTOR_FAILURE": RealFaultType.MOTOR_FAILURE,
    "TOTAL_POWER_LOSS": RealFaultType.TOTAL_POWER_LOSS,
    "BATTERY_CRITICAL": RealFaultType.BATTERY_CRITICAL,
    "COMMS_LOSS": RealFaultType.COMMS_LOSS,
    "COMMS_TOTAL_LOSS": RealFaultType.COMMS_LOSS,
    "COMMS_PARTIAL_LOSS": RealFaultType.COMMS_LOSS,
    "COMMS_DELAYED": RealFaultType.COMMS_DELAY,
    "COMMS_CORRUPTED": RealFaultType.COMMS_DELAY,
    "GPS_LOSS": RealFaultType.GPS_LOSS,
    "GPS_DRIFT": RealFaultType.GPS_DRIFT,
    "SENSOR_FAILURE": RealFaultType.LIDAR_FAILURE,
    "OBSTACLE_COLLISION": RealFaultType.MOTOR_FAILURE,
    "STATE_MACHINE_STUCK": RealFaultType.STATE_STUCK,
    "TASK_QUEUE_OVERFLOW": RealFaultType.STATE_STUCK,
    "MEMORY_EXHAUSTION": RealFaultType.STATE_STUCK,
    "GEOFENCE_BREACH": RealFaultType.STATE_STUCK,
    "WIND_GUST": RealFaultType.STATE_STUCK,
    "NO_FLY_ZONE_ENTERED": RealFaultType.STATE_STUCK,
    "RANDOM_REBOOT": RealFaultType.RANDOM_REBOOT,
    "BATTERY_DRAIN": RealFaultType.BATTERY_CRITICAL,
}

SEVERITY_MAP = {
    "WARNING": RealFaultSeverity.WARNING,
    "CRITICAL": RealFaultSeverity.CRITICAL,
    "FATAL": RealFaultSeverity.FATAL,
}


def _make_drone(drone_id: int, battery: float = 100.0) -> DroneState:
    return DroneState(
        drone_id=drone_id,
        drone_type=DroneType.ALPHA,
        position=Vector3(x=float(drone_id * 80), y=0.0, z=-65.0),
        velocity=Vector3(),
        battery=battery,
    )


def _make_tasks(count: int = 6) -> List[SwarmTask]:
    return [
        SwarmTask(
            task_id=f"sector_{i}",
            task_type=TaskType.SECTOR_COVERAGE,
            position=Vector3(x=float(i * 80), y=0.0, z=-65.0),
            radius=100.0,
            priority=5,
        )
        for i in range(count)
    ]


def _scenario_ids():
    return [s.id for s in EDGE_TEST_SCENARIOS]


def _scenario_by_id(scenario_id: str) -> TestScenario:
    for s in EDGE_TEST_SCENARIOS:
        if s.id == scenario_id:
            return s
    raise KeyError(scenario_id)


class TestScenarioDefinitions:
    """Verify all 25 scenario definitions are well-formed."""

    def test_all_scenarios_present(self):
        assert len(EDGE_TEST_SCENARIOS) == 25

    @pytest.mark.parametrize("scenario", EDGE_TEST_SCENARIOS, ids=_scenario_ids())
    def test_scenario_has_required_fields(self, scenario: TestScenario):
        assert scenario.id
        assert scenario.name
        assert scenario.description.strip()
        assert scenario.category
        assert isinstance(scenario.success_criteria, list)
        assert len(scenario.success_criteria) >= 1
        assert scenario.timeout > 0


class TestFaultInjectionScenarios:
    """Inject faults per scenario and verify the injector state machine."""

    @pytest.mark.parametrize("scenario", EDGE_TEST_SCENARIOS, ids=_scenario_ids())
    def test_faults_inject_and_clear(self, scenario: TestScenario):
        if not scenario.faults:
            pytest.skip("No faults defined for this scenario")

        injector = FaultInjector()
        now = time.time()

        for fault_def in scenario.faults:
            ft_name = fault_def["type"].name
            real_ft = FAULT_TYPE_MAP.get(ft_name)
            if real_ft is None:
                continue
            sev_name = fault_def["severity"].name
            real_sev = SEVERITY_MAP.get(sev_name, RealFaultSeverity.CRITICAL)
            duration = fault_def.get("params", {}).get("duration", 0)

            fid = injector.inject_fault(
                fault_type=real_ft,
                drone_id=fault_def["drone_id"],
                severity=real_sev,
                duration=duration,
                current_time=now,
            )
            assert fid is not None

        active = injector.get_active_faults()
        assert len(active) >= 1

        injector.clear_all_faults()
        assert len(injector.get_active_faults()) == 0


class TestTaskRedistributionScenarios:
    """Test task redistribution under each failure scenario."""

    @pytest.mark.parametrize("scenario_id", [
        "SDF-001", "SDF-002", "MDF-001", "MDF-002", "MDF-003",
        "REC-001", "REC-003",
    ])
    def test_redistribution_after_failure(self, scenario_id: str):
        scenario = _scenario_by_id(scenario_id)
        redistributor = TaskRedistributor(drone_count=DRONE_COUNT)
        now = time.time()

        for i in range(DRONE_COUNT):
            redistributor.update_heartbeat(i, now)
            redistributor.assign_task(f"sector_{i}", i)

        failed_ids = set()
        for fault_def in scenario.faults:
            if fault_def["severity"].name == "FATAL":
                did = fault_def["drone_id"]
                if did >= 0:
                    failed_ids.add(did)

        for fid in failed_ids:
            assignments = redistributor.redistribute_tasks(fid)
            assert isinstance(assignments, dict)

        operational = redistributor.get_operational_drones()
        alive = [d for d in range(DRONE_COUNT) if d not in failed_ids]
        for d in alive:
            assert d in operational

        coverage = redistributor.calculate_coverage_after_failure()
        if len(failed_ids) < DRONE_COUNT:
            assert coverage["can_complete_mission"]


class TestCBBAConsensusScenarios:
    """Test CBBA convergence under edge-case conditions."""

    def test_conflicting_bids_resolve_deterministically(self):
        """TRD-002: Two drones bid simultaneously, CBBA resolves."""
        tasks = _make_tasks(3)
        engines: Dict[int, CBBAEngine] = {}
        for i in range(3):
            eng = CBBAEngine(drone_id=i)
            eng.upsert_tasks(tasks)
            engines[i] = eng

        for i in range(3):
            engines[i].bundle_phase(_make_drone(i))

        for _ in range(5):
            for i in range(3):
                payload = engines[i].get_bids_payload()
                for j in range(3):
                    if i != j:
                        engines[j].ingest_remote_payload(i, payload)
            for i in range(3):
                engines[i].bundle_phase(_make_drone(i))

        assigned: Dict[str, int] = {}
        for i in range(3):
            current = engines[i].get_current_task()
            if current is not None:
                assert current.task_id not in assigned or assigned[current.task_id] == i
                assigned[current.task_id] = i

    def test_consensus_converges_after_partition_heal(self):
        """CON-001: Split-brain heals and converges."""
        tasks = _make_tasks(6)
        engines: Dict[int, CBBAEngine] = {}
        for i in range(6):
            eng = CBBAEngine(drone_id=i)
            eng.upsert_tasks(tasks)
            engines[i] = eng

        partition_a = [0, 1, 2]
        partition_b = [3, 4, 5]

        for i in partition_a:
            engines[i].bundle_phase(_make_drone(i))
        for i in partition_b:
            engines[i].bundle_phase(_make_drone(i))

        for _ in range(3):
            for i in partition_a:
                p = engines[i].get_bids_payload()
                for j in partition_a:
                    if i != j:
                        engines[j].ingest_remote_payload(i, p)
            for i in partition_b:
                p = engines[i].get_bids_payload()
                for j in partition_b:
                    if i != j:
                        engines[j].ingest_remote_payload(i, p)

        for _ in range(5):
            for i in range(6):
                p = engines[i].get_bids_payload()
                for j in range(6):
                    if i != j:
                        engines[j].ingest_remote_payload(i, p)
                engines[i].bundle_phase(_make_drone(i))

        all_winners = {}
        for i in range(6):
            for tid, agent in engines[i].winning_agents.items():
                if tid in all_winners:
                    assert all_winners[tid] == agent, (
                        f"Inconsistent winner for {tid}: "
                        f"drone {i} sees {agent}, expected {all_winners[tid]}"
                    )
                all_winners[tid] = agent

    def test_agent_departure_releases_tasks(self):
        """REC-001 / SDF-001: Departed agent tasks freed."""
        tasks = _make_tasks(3)
        eng = CBBAEngine(drone_id=0)
        eng.upsert_tasks(tasks)

        eng.winning_bids["sector_0"] = 0.8
        eng.winning_agents["sector_0"] = 5
        eng.bid_timestamps["sector_0"] = time.time()
        eng._bundle.append("sector_0")

        eng.clear_agent_claims(5)
        assert 5 not in eng.winning_agents.values()

    def test_bundle_respects_battery_feasibility(self):
        """SDF-002: Low battery prevents task pickup."""
        tasks = _make_tasks(3)
        for t in tasks:
            t.position = Vector3(x=5000.0, y=5000.0, z=0.0)

        eng = CBBAEngine(drone_id=0, config=CBBAConfig(battery_reserve=15.0))
        eng.upsert_tasks(tasks)

        low_battery_drone = _make_drone(0, battery=16.0)
        eng.bundle_phase(low_battery_drone)
        assert len(eng.get_bundle_ids()) == 0


class TestScenarioCategoryCoverage:
    """Ensure all 8 categories are represented."""

    def test_all_categories_present(self):
        categories = {s.category for s in EDGE_TEST_SCENARIOS}
        expected = {
            "Single Drone Failures",
            "Multiple Drone Failures",
            "Communication Failures",
            "Task Redistribution",
            "Coverage Maintenance",
            "Recovery Scenarios",
            "Consensus Edge Cases",
            "Stress Tests",
        }
        assert categories == expected
