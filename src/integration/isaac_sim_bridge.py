"""
Project Sanjay Mk2 - Isaac Sim <-> ROS 2 Bridge
===============================================
Subscribes to Isaac Sim sensor topics (published via Isaac Sim's
built-in ROS 2 Bridge) and feeds the data into the existing
        └─ /alpha_0/cmd_vel            ◄── velocity commands ◄── autonomy logic

Requirements:
    - ROS 2 Humble (runs inside Docker on WSL2)
    - cv_bridge, opencv-python-headless
    - This module's ROS 2 features gracefully degrade when rclpy is
      unavailable (e.g. on Windows without ROS 2).

Usage (WSL2 Docker):
    python -m src.integration.isaac_sim_bridge --config config/isaac_sim.yaml

Usage (standalone, no ROS):
    # Only the adapter classes are usable
    from src.integration.isaac_sim_bridge import ImageToObservation, OdometryAdapter
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from src.core.types.drone_types import (
    DetectedObject,
    DroneType,
    FusedObservation,
    Quaternion,
    SensorObservation,
    SensorType,
    TelemetryData,
    Vector3,
)

logger = logging.getLogger(__name__)

try:
    from src.surveillance.sensor_fusion import SensorFusionPipeline
except Exception as e:
    SensorFusionPipeline = None  # type: ignore[assignment]
    logger.warning("SensorFusionPipeline unavailable, using no-op fusion: %s", e)

try:
    from src.surveillance.change_detection import ChangeDetector
    from src.surveillance.baseline_map import BaselineMap
except Exception:
    ChangeDetector = None  # type: ignore[assignment,misc]
    BaselineMap = None  # type: ignore[assignment,misc]

try:
    from src.surveillance.threat_manager import ThreatManager
except Exception:
    ThreatManager = None  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DroneTopicConfig:
    """ROS 2 topic mapping for a single drone."""

    name: str
    drone_type: DroneType
    altitude: float
    rgb_fov_deg: float = 84.0
    topic_rgb: str = ""
    topic_thermal: str = ""
    topic_odom: str = ""
    topic_imu: str = ""
    topic_lidar_3d: str = ""
    topic_cmd_vel: str = ""


@dataclass
class BridgeConfig:
    """Complete bridge configuration loaded from YAML."""

    drones: List[DroneTopicConfig] = field(default_factory=list)
    tick_rate_hz: float = 10.0
    match_radius: float = 15.0
    min_change_confidence: float = 0.35
    qos_depth: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> BridgeConfig:
        """
        Load bridge configuration from a YAML file.

        Args:
            path: Path to isaac_sim.yaml

        Returns:
            Populated BridgeConfig instance.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        drones: List[DroneTopicConfig] = []
        for name, cfg in raw.get("drones", {}).items():
            topics = cfg.get("topics", {})
            drone_type = DroneType[cfg.get("type", "ALPHA")]
            drones.append(
                DroneTopicConfig(
                    name=name,
                    drone_type=drone_type,
                    altitude=float(cfg.get("altitude", 65.0)),
                    rgb_fov_deg=float(cfg.get("rgb_fov_deg", 84.0)),
                    topic_rgb=topics.get("rgb", f"/{name}/rgb/image_raw"),
                    topic_thermal=topics.get(
                        "thermal",
                        f"/{name}/thermal/image_raw" if drone_type == DroneType.ALPHA else "",
                    ),
                    topic_odom=topics.get("odom", f"/{name}/odom"),
                    topic_imu=topics.get("imu", f"/{name}/imu"),
                    topic_lidar_3d=topics.get("lidar_3d", ""),
                    topic_cmd_vel=topics.get("cmd_vel", f"/{name}/cmd_vel"),
                )
            )

        fusion = raw.get("fusion", {})
        ros2 = raw.get("ros2", {})

        return cls(
            drones=drones,
            tick_rate_hz=float(fusion.get("tick_rate_hz", 10.0)),
            match_radius=float(fusion.get("match_radius", 15.0)),
            min_change_confidence=float(fusion.get("min_change_confidence", 0.35)),
            qos_depth=int(ros2.get("qos_depth", 10)),
        )


# ═══════════════════════════════════════════════════════════════════
#  Adapters (usable without ROS 2)
# ═══════════════════════════════════════════════════════════════════


class ImageToObservation:
    """
    Convert a raw image (numpy HxWxC array) into a SensorObservation.

    This is a lightweight adapter — it does NOT run object detection.
    The actual detection model (e.g. YOLOv8) is pluggable via
    the `detector` callback.

    Usage:
        adapter = ImageToObservation(drone_id=0, sensor_type=SensorType.RGB_CAMERA)
        obs = adapter.convert(
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            drone_position=Vector3(100, 200, 0),
            altitude=65.0,
        )
    """

    def __init__(
        self,
        drone_id: int,
        sensor_type: SensorType = SensorType.RGB_CAMERA,
        detector: Optional[Callable[[np.ndarray], List[DetectedObject]]] = None,
    ):
        self.drone_id = drone_id
        self.sensor_type = sensor_type
        self._detector = detector

    def convert(
        self,
        image: np.ndarray,
        drone_position: Vector3,
        altitude: float,
        coverage_cells: Optional[List[tuple]] = None,
    ) -> SensorObservation:
        """
        Convert an image frame to a SensorObservation.

        Args:
            image: HxWxC numpy array (BGR or RGB).
            drone_position: Current drone XY position.
            altitude: Current altitude in meters AGL.
            coverage_cells: Grid cells covered by this frame.

        Returns:
            SensorObservation with detected objects (if detector is set).
        """
        detected: List[DetectedObject] = []
        if self._detector is not None:
            detected = self._detector(image)

        return SensorObservation(
            sensor_type=self.sensor_type,
            drone_id=self.drone_id,
            drone_position=drone_position,
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells or [],
            timestamp=time.time(),
        )


class OdometryAdapter:
    """
    Convert odometry data (position + orientation) into project types.

    Handles the coordinate frame conversion from Isaac Sim's ENU / ROS
    conventions to the project's NED convention.

    Isaac Sim / ROS 2:  x=East, y=North, z=Up
    Project (NED):      x=North, y=East, z=Down
    """

    @staticmethod
    def to_vector3(x: float, y: float, z: float) -> Vector3:
        """
        Convert ENU position to NED Vector3.

        Args:
            x, y, z: Position in ENU frame (ROS convention).

        Returns:
            Vector3 in NED frame.
        """
        return Vector3(
            x=y,   # North = ROS y
            y=x,   # East  = ROS x
            z=-z,  # Down  = -ROS z
        )

    @staticmethod
    def to_telemetry(
        pos_x: float,
        pos_y: float,
        pos_z: float,
        quat_x: float,
        quat_y: float,
        quat_z: float,
        quat_w: float,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        vel_z: float = 0.0,
    ) -> TelemetryData:
        """
        Build a TelemetryData from ROS odometry fields.

        Args:
            pos_*: Position in ENU.
            quat_*: Orientation quaternion (ROS convention: x,y,z,w).
            vel_*: Linear velocity in ENU.

        Returns:
            TelemetryData with NED-converted values.
        """
        position = OdometryAdapter.to_vector3(pos_x, pos_y, pos_z)
        velocity = OdometryAdapter.to_vector3(vel_x, vel_y, vel_z)

        # Convert ROS quaternion (x,y,z,w) to project quaternion (w,x,y,z)
        orientation = Quaternion(w=quat_w, x=quat_x, y=quat_y, z=quat_z)
        attitude = orientation.to_euler()

        return TelemetryData(
            position=position,
            velocity=velocity,
            orientation=orientation,
            attitude_euler=attitude,
            altitude_rel=pos_z,  # ENU z = altitude above ground
            in_air=pos_z > 0.3,
            armed=True,
            timestamp=time.time(),
        )


# ═══════════════════════════════════════════════════════════════════
#  ROS 2 Bridge Node (requires rclpy)
# ═══════════════════════════════════════════════════════════════════

# Guard import so the module is importable on Windows without ROS 2.
_ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    _ROS2_AVAILABLE = True
except ImportError:
    Node = object  # type: ignore[assignment,misc]


def is_ros2_available() -> bool:
    """Check whether ROS 2 Python bindings are installed."""
    return _ROS2_AVAILABLE


if _ROS2_AVAILABLE:
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, Imu, PointCloud2

    class IsaacSimBridgeNode(Node):  # type: ignore[misc]
        """
        ROS 2 node that bridges Isaac Sim sensors into the autonomy pipeline.

        For each configured drone, the node:
        1. Subscribes to RGB, thermal, odometry, and optional LiDAR topics from Isaac Sim
        2. Buffers incoming sensor data
        3. At a fixed rate (default 10 Hz), fuses observations via
           SensorFusionPipeline and runs change detection
        4. Publishes velocity commands back to Isaac Sim

        Usage:
            rclpy.init()
            config = BridgeConfig.from_yaml("config/isaac_sim.yaml")
            node = IsaacSimBridgeNode(config)
            rclpy.spin(node)
        """

        def __init__(self, config: BridgeConfig):
            super().__init__("isaac_sim_bridge")
            self.config = config

            # QoS profile for sensor data
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=config.qos_depth,
            )

            # Per-drone state
            self._drone_state: Dict[str, Dict[str, Any]] = {}
            self._cmd_publishers: Dict[str, Any] = {}

            # Shared fusion pipeline
            if SensorFusionPipeline is not None:
                self._fusion = SensorFusionPipeline(
                    match_radius=config.match_radius
                )
            else:
                self._fusion = None

            # Downstream surveillance pipeline
            if ChangeDetector is not None and BaselineMap is not None:
                # Empty baseline (200x200 grid = 1000m world / 5m cells)
                # Baseline gets populated incrementally as drones survey
                baseline = BaselineMap(rows=200, cols=200, cell_size=5.0)
                self._change_detector = ChangeDetector(
                    baseline=baseline,
                    min_confidence=config.min_change_confidence,
                )
            else:
                self._change_detector = None
            self._threat_manager = ThreatManager() if ThreatManager is not None else None

            # Autonomy hooks (set via register_autonomy_hooks)
            self._on_threat_callback: Optional[Callable] = None
            self._on_lidar_callback: Optional[Callable] = None

            # Set up subscriptions for each drone
            for idx, drone_cfg in enumerate(config.drones):
                name = drone_cfg.name
                self._drone_state[name] = {
                    "config": drone_cfg,
                    "drone_id": idx,
                    "position": Vector3(),
                    "altitude": drone_cfg.altitude,
                    "telemetry": None,
                    "rgb_adapter": ImageToObservation(
                        drone_id=idx,
                        sensor_type=SensorType.RGB_CAMERA,
                    ),
                    "thermal_adapter": ImageToObservation(
                        drone_id=idx,
                        sensor_type=SensorType.THERMAL_CAMERA,
                    ),
                    "odom_adapter": OdometryAdapter(),
                    "lidar_point_count": 0,
                }

                # RGB subscriber
                self.create_subscription(
                    Image,
                    drone_cfg.topic_rgb,
                    lambda msg, n=name: self._on_rgb(n, msg),
                    sensor_qos,
                )

                # Thermal subscriber (Alpha drones only)
                if drone_cfg.topic_thermal:
                    self.create_subscription(
                        Image,
                        drone_cfg.topic_thermal,
                        lambda msg, n=name: self._on_thermal(n, msg),
                        sensor_qos,
                    )

                # Odometry subscriber
                self.create_subscription(
                    Odometry,
                    drone_cfg.topic_odom,
                    lambda msg, n=name: self._on_odom(n, msg),
                    sensor_qos,
                )

                # IMU subscriber
                self.create_subscription(
                    Imu,
                    drone_cfg.topic_imu,
                    lambda msg, n=name: self._on_imu(n, msg),
                    sensor_qos,
                )

                # LiDAR subscriber (optional, ALPHA drones)
                if drone_cfg.topic_lidar_3d:
                    self.create_subscription(
                        PointCloud2,
                        drone_cfg.topic_lidar_3d,
                        lambda msg, n=name: self._on_lidar(n, msg),
                        sensor_qos,
                    )

                # Velocity command publisher
                self._cmd_publishers[name] = self.create_publisher(
                    Twist, drone_cfg.topic_cmd_vel, 10
                )

                self.get_logger().info(
                    f"Subscribed to drone '{name}' "
                    f"(type={drone_cfg.drone_type.name}, alt={drone_cfg.altitude}m)"
                )

            # Fusion timer
            period = 1.0 / config.tick_rate_hz
            self._fusion_timer = self.create_timer(period, self._fusion_tick)

            self.get_logger().info(
                f"Isaac Sim bridge started: {len(config.drones)} drone(s), "
                f"fusion at {config.tick_rate_hz} Hz"
            )

        def register_autonomy_hooks(
            self,
            on_threat: Optional[Callable] = None,
            on_lidar: Optional[Callable] = None,
        ):
            """
            Register callbacks that connect bridge output to the autonomy loop.

            Args:
                on_threat: Called with (drone_name, threat) when a new threat
                    is detected via the surveillance pipeline.
                on_lidar: Called with (drone_name, points_nx3) when LiDAR data
                    arrives, allowing the avoidance stack to consume it.
            """
            self._on_threat_callback = on_threat
            self._on_lidar_callback = on_lidar

        # ── Sensor Callbacks ──────────────────────────────────────

        def _on_rgb(self, drone_name: str, msg: Image):
            """Handle incoming RGB image from Isaac Sim."""
            state = self._drone_state[drone_name]
            try:
                # Convert ROS Image to numpy (without cv_bridge dependency)
                image = self._ros_image_to_numpy(msg)
                obs = state["rgb_adapter"].convert(
                    image=image,
                    drone_position=state["position"],
                    altitude=state["altitude"],
                )
                if self._fusion is not None:
                    self._fusion.add_observation(obs)
            except Exception as e:
                self.get_logger().warn(
                    f"RGB callback error for {drone_name}: {e}"
                )

        def _on_thermal(self, drone_name: str, msg: Image):
            """Handle incoming thermal image from Isaac Sim."""
            state = self._drone_state[drone_name]
            try:
                image = self._ros_image_to_numpy(msg)
                obs = state["thermal_adapter"].convert(
                    image=image,
                    drone_position=state["position"],
                    altitude=state["altitude"],
                )
                if self._fusion is not None:
                    self._fusion.add_observation(obs)
            except Exception as e:
                self.get_logger().warn(
                    f"Thermal callback error for {drone_name}: {e}"
                )

        def _on_odom(self, drone_name: str, msg: Odometry):
            """Handle incoming odometry from Isaac Sim."""
            state = self._drone_state[drone_name]
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear

            telemetry = OdometryAdapter.to_telemetry(
                pos_x=p.x, pos_y=p.y, pos_z=p.z,
                quat_x=q.x, quat_y=q.y, quat_z=q.z, quat_w=q.w,
                vel_x=v.x, vel_y=v.y, vel_z=v.z,
            )
            state["telemetry"] = telemetry
            state["position"] = telemetry.position
            state["altitude"] = telemetry.altitude_rel

        def _on_imu(self, drone_name: str, msg: Imu):
            """Handle incoming IMU data (stored for future use)."""
            # IMU data can be used for attitude estimation.
            # Currently the odom callback provides orientation,
            # so this is a placeholder for higher-fidelity fusion.
            pass

        def _on_lidar(self, drone_name: str, msg: PointCloud2):
            """Handle incoming PointCloud2 from Isaac Sim LiDAR."""
            state = self._drone_state[drone_name]
            try:
                n_points = int(msg.width) * int(msg.height)
                if n_points <= 0:
                    state["lidar_point_count"] = 0
                    return
                floats_per_point = max(3, int(msg.point_step) // 4)
                points = np.frombuffer(msg.data, dtype=np.float32).reshape(n_points, floats_per_point)[:, :3]
                state["lidar_point_count"] = int(points.shape[0])

                if self._on_lidar_callback is not None:
                    self._on_lidar_callback(drone_name, points)
            except Exception as e:
                self.get_logger().warn(
                    f"LiDAR callback error for {drone_name}: {e}"
                )

        # ── Fusion Loop ───────────────────────────────────────────

        def _fusion_tick(self):
            """Periodic fusion callback — fuse buffered observations."""
            if self._fusion is None:
                return
            fused = self._fusion.fuse()
            if fused is None:
                return

            n_objects = len(fused.detected_objects)
            if n_objects > 0:
                self.get_logger().info(
                    f"Fused: {n_objects} object(s), "
                    f"sensors={fused.sensor_count}"
                )

            if self._change_detector is not None and n_objects > 0:
                changes = self._change_detector.detect(fused)
                for change in changes:
                    if self._threat_manager is not None:
                        threat = self._threat_manager.report_change(change)
                        if threat is not None and self._on_threat_callback is not None:
                            drone_name = self._drone_name_for_id(change.detected_by)
                            self._on_threat_callback(drone_name, threat)

        def _drone_name_for_id(self, drone_id: int) -> str:
            """Map a numeric drone_id back to the configured drone name."""
            for name, state in self._drone_state.items():
                if state["drone_id"] == drone_id:
                    return name
            return f"drone_{drone_id}"

        # ── Command Publishing ────────────────────────────────────

        def send_velocity(
            self,
            drone_name: str,
            vx: float,
            vy: float,
            vz: float,
            yaw_rate: float = 0.0,
        ):
            """
            Publish a velocity command to Isaac Sim.

            Args:
                drone_name: Target drone (must match config).
                vx, vy, vz: Linear velocity in NED frame (m/s).
                yaw_rate: Yaw rate (rad/s).
            """
            if drone_name not in self._cmd_publishers:
                self.get_logger().error(f"Unknown drone: {drone_name}")
                return

            cmd = Twist()
            # Convert NED → ENU for ROS
            cmd.linear.x = vy   # ROS x = East = NED y
            cmd.linear.y = vx   # ROS y = North = NED x
            cmd.linear.z = -vz  # ROS z = Up = -NED z
            cmd.angular.z = yaw_rate
            self._cmd_publishers[drone_name].publish(cmd)

        # ── Helpers ───────────────────────────────────────────────

        @staticmethod
        def _ros_image_to_numpy(msg: Image) -> np.ndarray:
            """Convert sensor_msgs/Image to numpy array (no cv_bridge needed)."""
            dtype_map = {
                "8UC1": (np.uint8, 1),
                "8UC3": (np.uint8, 3),
                "bgr8": (np.uint8, 3),
                "rgb8": (np.uint8, 3),
                "mono8": (np.uint8, 1),
                "16UC1": (np.uint16, 1),
                "32FC1": (np.float32, 1),
            }
            dtype, channels = dtype_map.get(msg.encoding, (np.uint8, 3))
            image = np.frombuffer(msg.data, dtype=dtype)
            if channels == 1:
                image = image.reshape((msg.height, msg.width))
            else:
                image = image.reshape((msg.height, msg.width, channels))
            return image

# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════


def main():
    """Launch the bridge node from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Isaac Sim <-> Sanjay MK2 ROS 2 Bridge"
    )
    parser.add_argument(
        "--config",
        default="config/isaac_sim.yaml",
        help="Path to isaac_sim.yaml config file",
    )
    args = parser.parse_args()

    if not is_ros2_available():
        print(
            "ERROR: rclpy is not available.\n"
            "The Isaac Sim bridge requires ROS 2 Humble.\n"
            "Run this inside a ROS 2 Docker container on WSL2:\n"
            "  docker compose --profile isaac up\n"
        )
        return

    config = BridgeConfig.from_yaml(args.config)

    rclpy.init()
    node = IsaacSimBridgeNode(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
