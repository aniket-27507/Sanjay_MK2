"""
Tests for Isaac Sim bridge logic.
Project Sanjay Mk2 - Test Suite
=================================
Isaac Sim integration bridging network tests.

@author: Aniket More
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import yaml

from src.core.types.drone_types import (
    Vector3, DroneType, SensorType, TelemetryData,
    DetectedObject, SensorObservation, FusedObservation,
)
from src.surveillance.sensor_fusion import SensorFusionPipeline
from src.integration.isaac_sim_bridge import (
    BridgeConfig,
    DroneTopicConfig,
    ImageToObservation,
    OdometryAdapter,
    is_ros2_available,
)


# ═══════════════════════════════════════════════════════════════════
#  Config Loading
# ═══════════════════════════════════════════════════════════════════


class TestBridgeConfig:
    """Tests for BridgeConfig.from_yaml()."""

    def test_loads_project_config(self):
        """Load the real config/isaac_sim.yaml and verify structure."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "isaac_sim.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config/isaac_sim.yaml not found")

        config = BridgeConfig.from_yaml(config_path)
        assert len(config.drones) >= 1
        assert config.tick_rate_hz > 0
        assert config.match_radius > 0

    def test_parses_drone_types(self):
        """Verify drone type mapping from YAML."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "isaac_sim.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config/isaac_sim.yaml not found")

        config = BridgeConfig.from_yaml(config_path)

        alpha_drones = [d for d in config.drones if d.drone_type == DroneType.ALPHA]
        beta_drones = [d for d in config.drones if d.drone_type == DroneType.BETA]

        assert len(alpha_drones) >= 1, "Should have at least one Alpha drone"
        assert len(beta_drones) >= 1, "Should have at least one Beta drone"

        for d in alpha_drones:
            assert d.altitude == 65.0
            assert d.rgb_fov_deg == 84.0

        for d in beta_drones:
            assert d.altitude == 25.0
            assert d.rgb_fov_deg == 50.0

    def test_parses_topics(self):
        """Verify topic mapping is populated."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "isaac_sim.yaml"
        )
        if not os.path.exists(config_path):
            pytest.skip("config/isaac_sim.yaml not found")

        config = BridgeConfig.from_yaml(config_path)

        for drone in config.drones:
            assert drone.topic_rgb, f"No RGB topic for {drone.name}"
            assert drone.topic_odom, f"No odom topic for {drone.name}"
            assert drone.topic_cmd_vel, f"No cmd_vel topic for {drone.name}"
            if drone.drone_type == DroneType.ALPHA:
                assert drone.topic_thermal, f"No thermal topic for {drone.name}"
                assert drone.topic_lidar_3d, f"No LiDAR topic for {drone.name}"
                assert drone.topic_lidar_3d.endswith("/lidar_3d/points")
            else:
                assert drone.topic_thermal == ""
                assert drone.topic_lidar_3d == ""

    def test_loads_from_minimal_yaml(self):
        """Load config from minimal YAML content."""
        minimal = {
            "drones": {
                "test_drone": {
                    "type": "ALPHA",
                    "altitude": 50.0,
                    "topics": {
                        "rgb": "/test/rgb",
                        "thermal": "/test/thermal",
                        "lidar_3d": "/test/lidar_3d/points",
                        "odom": "/test/odom",
                        "cmd_vel": "/test/cmd_vel",
                    },
                }
            },
            "fusion": {"tick_rate_hz": 5},
            "ros2": {"qos_depth": 5},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(minimal, f)
            f.flush()
            config = BridgeConfig.from_yaml(f.name)

        os.unlink(f.name)
        assert len(config.drones) == 1
        assert config.drones[0].name == "test_drone"
        assert config.drones[0].topic_rgb == "/test/rgb"
        assert config.drones[0].topic_thermal == "/test/thermal"
        assert config.drones[0].topic_lidar_3d == "/test/lidar_3d/points"
        assert config.tick_rate_hz == 5.0
        assert config.qos_depth == 5


# ═══════════════════════════════════════════════════════════════════
#  ImageToObservation Adapter
# ═══════════════════════════════════════════════════════════════════


class TestImageToObservation:
    """Tests for converting images to SensorObservation."""

    def test_convert_rgb_image(self):
        """Basic RGB image → SensorObservation conversion."""
        adapter = ImageToObservation(drone_id=0, sensor_type=SensorType.RGB_CAMERA)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        obs = adapter.convert(
            image=image,
            drone_position=Vector3(100, 200, 0),
            altitude=65.0,
        )

        assert obs.sensor_type == SensorType.RGB_CAMERA
        assert obs.drone_id == 0
        assert obs.drone_position.x == 100
        assert obs.drone_altitude == 65.0
        assert obs.detected_objects == []
        assert obs.timestamp > 0

    def test_convert_with_detector(self):
        """Image conversion with a detection callback."""
        def mock_detector(image: np.ndarray):
            return [
                DetectedObject(
                    object_id="det_001",
                    object_type="person",
                    position=Vector3(110, 210, 0),
                    confidence=0.72,
                    thermal_signature=0.0,
                )
            ]

        adapter = ImageToObservation(
            drone_id=1,
            sensor_type=SensorType.RGB_CAMERA,
            detector=mock_detector,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        obs = adapter.convert(
            image=image,
            drone_position=Vector3(100, 200, 0),
            altitude=65.0,
        )

        assert len(obs.detected_objects) == 1
        assert obs.detected_objects[0].object_id == "det_001"
        assert obs.detected_objects[0].confidence == 0.72

    def test_convert_with_coverage_cells(self):
        """Coverage cells are forwarded correctly."""
        adapter = ImageToObservation(drone_id=0)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cells = [(0, 0), (0, 1), (1, 0)]

        obs = adapter.convert(
            image=image,
            drone_position=Vector3(),
            altitude=65.0,
            coverage_cells=cells,
        )

        assert obs.coverage_cells == cells


# ═══════════════════════════════════════════════════════════════════
#  OdometryAdapter
# ═══════════════════════════════════════════════════════════════════


class TestOdometryAdapter:
    """Tests for ENU ↔ NED conversion."""

    def test_enu_to_ned_position(self):
        """ENU (x=East, y=North, z=Up) → NED (x=North, y=East, z=Down)."""
        # ENU: 10m East, 20m North, 30m Up
        vec = OdometryAdapter.to_vector3(x=10.0, y=20.0, z=30.0)

        # NED: 20m North, 10m East, -30m Down
        assert vec.x == 20.0   # North = ENU y
        assert vec.y == 10.0   # East = ENU x
        assert vec.z == -30.0  # Down = -ENU z

    def test_enu_origin_maps_to_ned_origin(self):
        """Origin maps correctly."""
        vec = OdometryAdapter.to_vector3(0.0, 0.0, 0.0)
        assert vec.x == 0.0
        assert vec.y == 0.0
        assert vec.z == 0.0

    def test_to_telemetry(self):
        """Full odometry → TelemetryData conversion."""
        telem = OdometryAdapter.to_telemetry(
            pos_x=5.0, pos_y=10.0, pos_z=65.0,
            quat_x=0.0, quat_y=0.0, quat_z=0.0, quat_w=1.0,
            vel_x=1.0, vel_y=2.0, vel_z=0.0,
        )

        assert isinstance(telem, TelemetryData)
        # Position: ENU→NED
        assert telem.position.x == 10.0  # North
        assert telem.position.y == 5.0   # East
        assert telem.position.z == -65.0 # Down

        # Altitude
        assert telem.altitude_rel == 65.0

        # In air
        assert telem.in_air is True

        # Velocity
        assert telem.velocity.x == 2.0  # North
        assert telem.velocity.y == 1.0  # East

    def test_telemetry_on_ground(self):
        """Drone on ground has in_air=False."""
        telem = OdometryAdapter.to_telemetry(
            pos_x=0, pos_y=0, pos_z=0.1,
            quat_x=0, quat_y=0, quat_z=0, quat_w=1,
        )
        assert telem.in_air is False


# ═══════════════════════════════════════════════════════════════════
#  Integration: Bridge + SensorFusionPipeline
# ═══════════════════════════════════════════════════════════════════


class TestFusionIntegration:
    """Verify that bridge-produced observations work with existing fusion pipeline."""

    def test_bridge_observations_fuse_correctly(self):
        """Observations from ImageToObservation feed into SensorFusionPipeline."""
        pipeline = SensorFusionPipeline()

        # Simulate RGB adapter output
        rgb_adapter = ImageToObservation(
            drone_id=0,
            sensor_type=SensorType.RGB_CAMERA,
            detector=lambda img: [
                DetectedObject(
                    object_id="obj_1",
                    object_type="person",
                    position=Vector3(100, 200, 0),
                    confidence=0.55,
                )
            ],
        )
        rgb_obs = rgb_adapter.convert(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            drone_position=Vector3(100, 200, 0),
            altitude=65.0,
        )

        # Simulate thermal observation (from existing sensor code)
        thermal_obs = SensorObservation(
            sensor_type=SensorType.THERMAL_CAMERA,
            drone_id=0,
            drone_position=Vector3(100, 200, 0),
            detected_objects=[
                DetectedObject(
                    object_id="thermal_1",
                    object_type="thermal_contact",
                    position=Vector3(102, 201, 0),  # Within match radius
                    confidence=0.65,
                    thermal_signature=0.85,
                    sensor_type=SensorType.THERMAL_CAMERA,
                )
            ],
        )

        pipeline.add_observation(rgb_obs)
        pipeline.add_observation(thermal_obs)
        fused = pipeline.fuse()

        assert fused is not None
        assert fused.drone_id == 0
        assert fused.sensor_count == 2
        # Person should have boosted confidence from thermal corroboration
        person_dets = [d for d in fused.detected_objects if d.object_type == "person"]
        assert len(person_dets) == 1
        assert person_dets[0].confidence > 0.55  # Boosted

    def test_bridge_observation_types_match_pipeline(self):
        """Bridge output is the exact SensorObservation type the pipeline expects."""
        adapter = ImageToObservation(drone_id=0)
        obs = adapter.convert(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            drone_position=Vector3(),
            altitude=65.0,
        )
        assert isinstance(obs, SensorObservation)

        pipeline = SensorFusionPipeline()
        pipeline.add_observation(obs)
        result = pipeline.fuse()
        assert isinstance(result, FusedObservation)


# ═══════════════════════════════════════════════════════════════════
#  ROS 2 Availability Check
# ═══════════════════════════════════════════════════════════════════


class TestROS2Availability:
    def test_is_ros2_available_returns_bool(self):
        """is_ros2_available() returns a boolean."""
        result = is_ros2_available()
        assert isinstance(result, bool)
        # On Windows dev machine, this will typically be False
