"""
Tests for the scenario simulation framework.

Covers:
    - Scenario YAML loading and validation
    - Executor tick loop with baseline scenario
    - Metrics collection accuracy
    - New object types (weapon_person, fire, explosive_device)
    - Sensor degradation wrapper
"""

import os
import math
import pytest
import tempfile
from pathlib import Path

from src.simulation.scenario_loader import (
    ScenarioLoader, ScenarioDefinition, VALID_CATEGORIES,
)
from src.simulation.scenario_executor import ScenarioExecutor, ScenarioResult
from src.simulation.metrics_collector import ScenarioMetrics, BatchReport
from src.simulation.sensor_degradation import DegradedSensorWrapper
from src.surveillance.world_model import WorldModel, THERMAL_SIGNATURES, OBJECT_SIZES
from src.surveillance.change_detection import THREAT_CLASSIFICATION
from src.core.types.drone_types import ThreatLevel, Vector3


SCENARIOS_DIR = Path("config/scenarios")


# ═══════════════════════════════════════════════════════════════════
#  Scenario Loader Tests
# ═══════════════════════════════════════════════════════════════════

class TestScenarioLoader:
    """Scenario YAML loading and validation."""

    def test_load_s01(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S01_building_rooftop_intruder.yaml")
        assert s.id == "S01"
        assert s.category == "high_rise"
        assert len(s.spawn_schedule) == 1
        assert s.spawn_schedule[0].object_type == "person"
        assert s.spawn_schedule[0].is_threat is True

    def test_load_s10_baseline(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S10_baseline_patrol.yaml")
        assert s.id == "S10"
        assert s.category == "baseline"
        assert len(s.spawn_schedule) == 0
        assert s.crowd.enabled is False

    def test_load_s03_crowd(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S03_religious_gathering_buildup.yaml")
        assert s.crowd.enabled is True
        assert len(s.crowd.density_curve) >= 3
        assert s.crowd.initial_density == 1.0

    def test_load_s09_fault(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S09_drone_gps_loss.yaml")
        assert len(s.fault_schedule) == 1
        assert s.fault_schedule[0].fault_type == "gps_loss"
        assert s.fault_schedule[0].drone_id == 2

    def test_load_all_scenarios(self):
        scenarios = ScenarioLoader.load_all(SCENARIOS_DIR)
        assert len(scenarios) >= 10
        ids = [s.id for s in scenarios]
        assert "S01" in ids
        assert "S10" in ids

    def test_load_by_category(self):
        scenarios = ScenarioLoader.load_all(SCENARIOS_DIR, category="high_rise")
        assert all(s.category == "high_rise" for s in scenarios)

    def test_load_by_split(self):
        train = ScenarioLoader.load_all(SCENARIOS_DIR, split="train")
        test = ScenarioLoader.load_all(SCENARIOS_DIR, split="test")
        all_scenarios = ScenarioLoader.load_all(SCENARIOS_DIR)
        assert len(train) + len(test) <= len(all_scenarios)

    def test_invalid_yaml_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("scenario:\n  id: INVALID\n  category: nonexistent\n")
            f.flush()
            with pytest.raises(ValueError, match="Invalid category"):
                ScenarioLoader.load(f.name)
        os.unlink(f.name)

    def test_spawn_schedule_sorted(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S08_bird_flock.yaml")
        times = [e.time for e in s.spawn_schedule]
        assert times == sorted(times)

    def test_all_categories_valid(self):
        scenarios = ScenarioLoader.load_all(SCENARIOS_DIR)
        for s in scenarios:
            assert s.category in VALID_CATEGORIES, f"{s.id} has invalid category {s.category}"


# ═══════════════════════════════════════════════════════════════════
#  Executor Tests
# ═══════════════════════════════════════════════════════════════════

class TestScenarioExecutor:
    """Executor tick loop and integration."""

    def test_baseline_no_threats(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S10_baseline_patrol.yaml")
        s.duration_sec = 10  # short run
        ex = ScenarioExecutor(s, gcs_port=19999)
        ex._gcs = None  # skip GCS for unit test
        result = ex.run(realtime=False)
        assert result.completed
        assert result.threats_detected == 0
        assert result.false_positives == 0

    def test_spawn_fires(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S01_building_rooftop_intruder.yaml")
        s.duration_sec = 5
        ex = ScenarioExecutor(s, gcs_port=19998)
        ex._gcs = None
        result = ex.run(realtime=False)
        assert ex._spawn_cursor == 0  # spawn at t=30, but run only 5s

        s.duration_sec = 35
        ex2 = ScenarioExecutor(s, gcs_port=19997)
        ex2._gcs = None
        result2 = ex2.run(realtime=False)
        assert ex2._spawn_cursor == 1  # spawn should have fired

    def test_drone_count(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S10_baseline_patrol.yaml")
        ex = ScenarioExecutor(s, gcs_port=19996)
        assert len(ex.drones) == 6  # 6 Alpha-only police swarm

    def test_fault_deactivates_drone(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S09_drone_gps_loss.yaml")
        s.duration_sec = 80
        ex = ScenarioExecutor(s, gcs_port=19995)
        ex._gcs = None
        result = ex.run(realtime=False)
        # Drone 2 should have been deactivated at t=60
        assert any(e["type"] == "fault" for e in result.events)

    def test_result_has_ground_truth(self):
        s = ScenarioLoader.load(SCENARIOS_DIR / "S01_building_rooftop_intruder.yaml")
        s.duration_sec = 35
        ex = ScenarioExecutor(s, gcs_port=19994)
        ex._gcs = None
        result = ex.run(realtime=False)
        assert len(result.ground_truth) == 1
        assert result.ground_truth[0]["type"] == "person"


# ═══════════════════════════════════════════════════════════════════
#  New Object Type Tests
# ═══════════════════════════════════════════════════════════════════

class TestNewObjectTypes:
    """Verify weapon_person, fire, explosive_device integration."""

    def test_thermal_signatures_exist(self):
        assert "weapon_person" in THERMAL_SIGNATURES
        assert "fire" in THERMAL_SIGNATURES
        assert "explosive_device" in THERMAL_SIGNATURES
        assert THERMAL_SIGNATURES["fire"] > THERMAL_SIGNATURES["person"]

    def test_object_sizes_exist(self):
        assert "weapon_person" in OBJECT_SIZES
        assert OBJECT_SIZES["weapon_person"] == 1.8  # same as person

    def test_threat_classification(self):
        assert THREAT_CLASSIFICATION["weapon_person"] == ThreatLevel.CRITICAL
        assert THREAT_CLASSIFICATION["explosive_device"] == ThreatLevel.CRITICAL
        assert THREAT_CLASSIFICATION["fire"] == ThreatLevel.HIGH

    def test_world_model_spawns_weapon_person(self):
        wm = WorldModel(width=100, height=100, cell_size=5)
        wm.generate_terrain(seed=1)
        obj_id = wm.spawn_object("weapon_person", Vector3(50, 50, 0), is_threat=True)
        obj = wm.get_object(obj_id)
        assert obj is not None
        assert obj.object_type == "weapon_person"
        assert obj.thermal_signature == THERMAL_SIGNATURES["weapon_person"]

    def test_world_model_spawns_fire(self):
        wm = WorldModel(width=100, height=100, cell_size=5)
        wm.generate_terrain(seed=1)
        obj_id = wm.spawn_object("fire", Vector3(50, 50, 10), is_threat=True)
        obj = wm.get_object(obj_id)
        assert obj.thermal_signature == 0.95


# ═══════════════════════════════════════════════════════════════════
#  Sensor Degradation Tests
# ═══════════════════════════════════════════════════════════════════

class TestSensorDegradation:
    """Sensor degradation wrapper modes."""

    def _make_sensor(self):
        from src.single_drone.sensors.rgb_camera import SimulatedRGBCamera
        from src.core.types.drone_types import DroneType
        return SimulatedRGBCamera(drone_type=DroneType.ALPHA)

    def test_normal_mode_passthrough(self):
        inner = self._make_sensor()
        wrapped = DegradedSensorWrapper(inner, mode="normal")
        wm = WorldModel(width=100, height=100, cell_size=5)
        wm.generate_terrain(seed=1)
        obs = wrapped.capture(Vector3(50, 50, -65), 65.0, wm, 0)
        assert obs is not None

    def test_failed_mode_empty(self):
        inner = self._make_sensor()
        wrapped = DegradedSensorWrapper(inner, mode="failed")
        wm = WorldModel(width=100, height=100, cell_size=5)
        wm.generate_terrain(seed=1)
        obs = wrapped.capture(Vector3(50, 50, -65), 65.0, wm, 0)
        assert len(obs.detected_objects) == 0

    def test_noisy_mode_adds_detections(self):
        inner = self._make_sensor()
        wrapped = DegradedSensorWrapper(inner, mode="noisy", noise_count=5)
        wm = WorldModel(width=100, height=100, cell_size=5)
        wm.generate_terrain(seed=1)
        obs = wrapped.capture(Vector3(50, 50, -65), 65.0, wm, 0)
        noise_objs = [o for o in obs.detected_objects if "noise" in o.object_id]
        assert len(noise_objs) == 5


# ═══════════════════════════════════════════════════════════════════
#  Metrics Collector Tests
# ═══════════════════════════════════════════════════════════════════

class TestMetricsCollector:
    """Metrics collection and training data export."""

    def test_scenario_metrics_aggregates(self):
        m = ScenarioMetrics(
            scenario_id="TEST",
            scenario_name="Test",
            category="baseline",
            split="train",
            duration_sec=60,
            detection_latencies=[5.0, 10.0, 15.0],
        )
        m.compute_aggregates()
        assert m.avg_detection_latency == 10.0
        assert m.max_detection_latency == 15.0
        assert m.min_detection_latency == 5.0

    def test_batch_report(self):
        report = BatchReport()
        for i in range(3):
            m = ScenarioMetrics(
                scenario_id=f"S{i}",
                scenario_name=f"Test {i}",
                category="baseline",
                split="train",
                duration_sec=30,
                threats_detected=i,
                coverage_pct=80 + i,
            )
            report.add_scenario(m)
        report.compute_aggregates()
        assert report.scenario_count == 3
        assert report.total_threats_detected == 3  # 0+1+2

    def test_training_dict_format(self):
        m = ScenarioMetrics(
            scenario_id="S11",
            scenario_name="Armed Person",
            category="armed",
            split="train",
            duration_sec=60,
            ground_truth=[{"time": 30, "type": "weapon_person"}],
            detections=[{"time": 42, "type": "weapon_person", "confidence": 0.7}],
        )
        td = m.to_training_dict()
        assert td["scenario_id"] == "S11"
        assert td["split"] == "train"
        assert len(td["ground_truth"]) == 1
        assert len(td["detections"]) == 1
