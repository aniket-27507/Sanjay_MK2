"""Tests for the WorldModel."""
import pytest
import numpy as np
from src.surveillance.world_model import WorldModel, TerrainType, WorldObject
from src.core.types.drone_types import Vector3


class TestWorldModelBasics:
    def test_creation(self):
        wm = WorldModel(width=100, height=100, cell_size=5.0)
        assert wm.cols == 20
        assert wm.rows == 20
        assert wm.terrain.shape == (20, 20)

    def test_coordinate_conversion_roundtrip(self):
        wm = WorldModel(width=100, height=100, cell_size=5.0)
        row, col = wm.world_to_grid(0, 0)
        x, y = wm.grid_to_world(row, col)
        # Should be near center
        assert abs(x) < wm.cell_size
        assert abs(y) < wm.cell_size

    def test_clamping(self):
        wm = WorldModel(width=100, height=100, cell_size=5.0)
        row, col = wm.world_to_grid(-9999, 9999)
        assert 0 <= row < wm.rows
        assert 0 <= col < wm.cols


class TestTerrainGeneration:
    def test_generate_terrain(self):
        wm = WorldModel(width=500, height=500, cell_size=5.0)
        wm.generate_terrain(seed=42)
        summary = wm.get_terrain_summary()
        assert summary['ROAD'] > 0
        assert summary['BUILDING'] > 0
        assert summary['VEGETATION'] > 0

    def test_deterministic(self):
        wm1 = WorldModel(width=500, height=500, cell_size=5.0)
        wm1.generate_terrain(seed=1)
        wm2 = WorldModel(width=500, height=500, cell_size=5.0)
        wm2.generate_terrain(seed=1)
        np.testing.assert_array_equal(wm1.terrain, wm2.terrain)


class TestObjectManagement:
    def test_spawn_and_get(self):
        wm = WorldModel()
        oid = wm.spawn_object("person", Vector3(10, 20, 0), is_threat=True)
        obj = wm.get_object(oid)
        assert obj is not None
        assert obj.object_type == "person"
        assert obj.is_threat

    def test_remove(self):
        wm = WorldModel()
        oid = wm.spawn_object("vehicle", Vector3(0, 0, 0))
        assert wm.remove_object(oid)
        assert wm.get_object(oid) is None

    def test_get_threats(self):
        wm = WorldModel()
        wm.spawn_object("person", Vector3(0, 0, 0), is_threat=True)
        wm.spawn_object("vehicle", Vector3(10, 0, 0), is_threat=False)
        wm.spawn_object("camp", Vector3(0, 10, 0), is_threat=True)
        threats = wm.get_threats()
        assert len(threats) == 2

    def test_get_objects_in_radius(self):
        wm = WorldModel()
        wm.spawn_object("person", Vector3(5, 5, 0))
        wm.spawn_object("vehicle", Vector3(100, 100, 0))
        near = wm.get_objects_in_radius(Vector3(0, 0, 0), 20)
        assert len(near) == 1


class TestSensorQueries:
    def test_query_fov(self):
        wm = WorldModel(width=200, height=200, cell_size=5.0)
        wm.spawn_object("person", Vector3(5, 5, 0))
        wm.spawn_object("vehicle", Vector3(150, 150, 0))  # Far away
        objects, cells = wm.query_fov(Vector3(0, 0, 0), altitude=65, fov_deg=84)
        # Person at (5,5) should be visible from (0,0) at 65m
        assert len(objects) >= 1
        assert len(cells) > 0

    def test_query_thermal(self):
        wm = WorldModel()
        wm.spawn_object("person", Vector3(10, 10, 0))  # thermal=0.85
        wm.spawn_object("equipment", Vector3(15, 15, 0))  # thermal=0.40
        hot = wm.query_thermal(Vector3(0, 0, 0), altitude=65, fov_deg=40, threshold=0.5)
        # Only person should be above 0.5 threshold
        assert all(o.thermal_signature >= 0.5 for o in hot)

    def test_elevation(self):
        wm = WorldModel(width=100, height=100, cell_size=5.0)
        elev = wm.get_elevation(0, 0)
        assert isinstance(elev, float)
"""Tests for simulated sensors."""
import pytest
from src.single_drone.sensors.rgb_camera import SimulatedRGBCamera
from src.single_drone.sensors.thermal_camera import SimulatedThermalCamera
from src.single_drone.sensors.depth_estimator import SimulatedDepthEstimator
from src.surveillance.world_model import WorldModel
from src.core.types.drone_types import Vector3, DroneType, SensorType


@pytest.fixture
def world_with_objects():
    wm = WorldModel(width=500, height=500, cell_size=5.0)
    wm.generate_terrain(seed=42)
    wm.spawn_object("person", Vector3(10, 10, 0), is_threat=True)
    wm.spawn_object("vehicle", Vector3(20, -5, 0), is_threat=False)
    return wm


class TestRGBCamera:
    def test_capture_detects_objects(self, world_with_objects):
        cam = SimulatedRGBCamera(drone_type=DroneType.ALPHA)
        obs = cam.capture(Vector3(0, 0, 0), altitude=25.0, world_model=world_with_objects, drone_id=0)
        assert obs.sensor_type == SensorType.RGB_CAMERA
        assert obs.drone_id == 0
        # At 25m altitude the person at (10,10) should often be detected
        # (probabilistic, but footprint at 25m with 84 deg FOV is ~47m radius)
        assert len(obs.coverage_cells) > 0

    def test_beta_higher_confidence(self, world_with_objects):
        alpha_cam = SimulatedRGBCamera(drone_type=DroneType.ALPHA)
        beta_cam = SimulatedRGBCamera(drone_type=DroneType.BETA)
        assert beta_cam.base_confidence > alpha_cam.base_confidence

    def test_footprint_scales_with_altitude(self):
        cam = SimulatedRGBCamera()
        r1 = cam.get_footprint_radius(25.0)
        r2 = cam.get_footprint_radius(65.0)
        assert r2 > r1


class TestThermalCamera:
    def test_captures_thermal_objects(self, world_with_objects):
        thermal = SimulatedThermalCamera()
        obs = thermal.capture(Vector3(0, 0, 0), altitude=25.0, world_model=world_with_objects, drone_id=0)
        assert obs.sensor_type == SensorType.THERMAL_CAMERA
        # Detected objects should have thermal signature
        for det in obs.detected_objects:
            assert det.thermal_signature > 0

    def test_threshold_filtering(self, world_with_objects):
        low = SimulatedThermalCamera(thermal_threshold=0.1)
        high = SimulatedThermalCamera(thermal_threshold=0.8)
        obs_low = low.capture(Vector3(0, 0, 0), 25.0, world_with_objects)
        obs_high = high.capture(Vector3(0, 0, 0), 25.0, world_with_objects)
        # Higher threshold should detect fewer or equal objects
        assert len(obs_high.detected_objects) <= len(obs_low.detected_objects)


class TestDepthEstimator:
    def test_estimate_returns_data(self, world_with_objects):
        est = SimulatedDepthEstimator()
        depths, cells = est.estimate(Vector3(0, 0, 0), 65.0, world_with_objects)
        assert len(depths) > 0
        assert len(cells) == len(depths)

    def test_accuracy_degrades_with_altitude(self):
        est = SimulatedDepthEstimator()
        acc_low = est.get_accuracy_at_altitude(25.0)
        acc_high = est.get_accuracy_at_altitude(65.0)
        assert acc_high > acc_low  # Higher altitude = worse accuracy (larger stddev)
