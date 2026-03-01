"""Tests for change detection and sensor fusion pipeline."""
import pytest
from src.surveillance.world_model import WorldModel
from src.surveillance.baseline_map import BaselineMap
from src.surveillance.change_detection import ChangeDetector
from src.surveillance.sensor_fusion import SensorFusionPipeline
from src.single_drone.sensors.rgb_camera import SimulatedRGBCamera
from src.single_drone.sensors.thermal_camera import SimulatedThermalCamera
from src.core.types.drone_types import (
    Vector3, DroneType, SensorType, ThreatLevel,
    DetectedObject, SensorObservation, FusedObservation,
)


@pytest.fixture
def world():
    wm = WorldModel(width=500, height=500, cell_size=5.0)
    wm.generate_terrain(seed=42)
    return wm


@pytest.fixture
def world_with_baseline(world):
    """World with a baseline already built (no dynamic objects yet)."""
    baseline = BaselineMap(world.rows, world.cols, world.cell_size)
    baseline.build_from_world_model(world)
    return world, baseline


class TestBaselineMap:
    def test_build_marks_all_surveyed(self, world):
        bl = BaselineMap(world.rows, world.cols, world.cell_size)
        bl.build_from_world_model(world)
        assert bl.coverage_percentage() == 100.0

    def test_incremental_update(self, world):
        bl = BaselineMap(world.rows, world.cols, world.cell_size)
        assert bl.coverage_percentage() == 0.0
        bl.update_from_observation([(0, 0), (0, 1), (1, 0)])
        assert bl.surveyed_cell_count() == 3

    def test_known_objects(self, world):
        oid = world.spawn_object("vehicle", Vector3(0, 0, 0))
        bl = BaselineMap(world.rows, world.cols, world.cell_size)
        bl.build_from_world_model(world)
        assert bl.is_known_object(oid)
        assert not bl.is_known_object("nonexistent")


class TestChangeDetector:
    def test_detects_new_object(self, world_with_baseline):
        world, baseline = world_with_baseline
        detector = ChangeDetector(baseline, min_confidence=0.3)

        # Spawn a new object AFTER baseline
        world.spawn_object("person", Vector3(10, 10, 0), is_threat=True)

        # Create a fused observation with the new object
        det = DetectedObject(
            object_id="obj_0001",
            object_type="person",
            position=Vector3(10, 10, 0),
            confidence=0.7,
            thermal_signature=0.8,
        )
        fused = FusedObservation(
            drone_id=0,
            position=Vector3(0, 0, 0),
            detected_objects=[det],
        )

        changes = detector.detect_changes(fused, current_time=1.0)
        assert len(changes) >= 1
        assert changes[0].change_type == "new_object"
        assert changes[0].threat_level == ThreatLevel.HIGH  # Person = HIGH

    def test_no_false_positive_for_known_objects(self, world):
        # Add object BEFORE baseline
        oid = world.spawn_object("vehicle", Vector3(5, 5, 0))
        baseline = BaselineMap(world.rows, world.cols, world.cell_size)
        baseline.build_from_world_model(world)
        detector = ChangeDetector(baseline)

        det = DetectedObject(
            object_id=oid,
            object_type="vehicle",
            position=Vector3(5, 5, 0),
            confidence=0.8,
        )
        fused = FusedObservation(drone_id=0, detected_objects=[det])

        changes = detector.detect_changes(fused, current_time=1.0)
        assert len(changes) == 0  # Known object, no change

    def test_cooldown_prevents_duplicates(self, world_with_baseline):
        world, baseline = world_with_baseline
        detector = ChangeDetector(baseline, min_confidence=0.3)

        det = DetectedObject(
            object_id="new_obj",
            object_type="person",
            position=Vector3(20, 20, 0),
            confidence=0.7,
        )
        fused = FusedObservation(drone_id=0, detected_objects=[det])

        changes1 = detector.detect_changes(fused, current_time=1.0)
        changes2 = detector.detect_changes(fused, current_time=2.0)
        assert len(changes1) == 1
        assert len(changes2) == 0  # Cooldown prevents duplicate


class TestSensorFusion:
    def test_rgb_only_confidence(self):
        pipeline = SensorFusionPipeline()
        rgb_det = DetectedObject(
            object_id="obj_1",
            object_type="person",
            position=Vector3(10, 10, 0),
            confidence=0.5,
            sensor_type=SensorType.RGB_CAMERA,
        )
        pipeline.add_observation(SensorObservation(
            sensor_type=SensorType.RGB_CAMERA,
            drone_id=0,
            detected_objects=[rgb_det],
        ))
        fused = pipeline.fuse()
        assert fused is not None
        assert len(fused.detected_objects) == 1
        assert fused.detected_objects[0].confidence == 0.5  # No boost

    def test_thermal_corroboration_boosts_confidence(self):
        pipeline = SensorFusionPipeline()
        rgb_det = DetectedObject(
            object_id="obj_1",
            object_type="person",
            position=Vector3(10, 10, 0),
            confidence=0.5,
            sensor_type=SensorType.RGB_CAMERA,
        )
        thermal_det = DetectedObject(
            object_id="obj_1",
            object_type="thermal_contact",
            position=Vector3(12, 11, 0),  # Close enough to match
            confidence=0.6,
            thermal_signature=0.85,
            sensor_type=SensorType.THERMAL_CAMERA,
        )
        pipeline.add_observation(SensorObservation(
            sensor_type=SensorType.RGB_CAMERA,
            drone_id=0,
            detected_objects=[rgb_det],
        ))
        pipeline.add_observation(SensorObservation(
            sensor_type=SensorType.THERMAL_CAMERA,
            drone_id=0,
            detected_objects=[thermal_det],
        ))
        fused = pipeline.fuse()
        assert fused is not None
        # Confidence should be boosted above 0.5
        assert fused.detected_objects[0].confidence > 0.5

    def test_thermal_only_detection(self):
        pipeline = SensorFusionPipeline()
        thermal_det = DetectedObject(
            object_id="obj_2",
            object_type="thermal_contact",
            position=Vector3(50, 50, 0),
            confidence=0.6,
            thermal_signature=0.9,
            sensor_type=SensorType.THERMAL_CAMERA,
        )
        pipeline.add_observation(SensorObservation(
            sensor_type=SensorType.RGB_CAMERA,
            drone_id=0,
            detected_objects=[],  # RGB sees nothing
        ))
        pipeline.add_observation(SensorObservation(
            sensor_type=SensorType.THERMAL_CAMERA,
            drone_id=0,
            detected_objects=[thermal_det],
        ))
        fused = pipeline.fuse()
        assert fused is not None
        # Should include thermal-only detection at reduced confidence
        thermal_only = [d for d in fused.detected_objects if d.object_type == "thermal_only"]
        assert len(thermal_only) == 1
        assert thermal_only[0].confidence < thermal_det.confidence  # Reduced
