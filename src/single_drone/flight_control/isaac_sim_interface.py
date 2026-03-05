"""
Project Sanjay Mk2 - Isaac Sim Flight Interface
================================================
Low-level interface compatible with FlightController for Isaac Sim use-cases.

Supports two execution modes:
- direct: run inside Isaac Sim Kit and manipulate drone prim transforms directly
- ros2: publish cmd_vel and consume odometry via ROS 2 topics
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import yaml

from src.core.types.drone_types import TelemetryData, Vector3

logger = logging.getLogger(__name__)


def _try_import_isaac_direct():
    try:
        import omni.usd  # type: ignore
        from pxr import Gf, UsdGeom  # type: ignore
        return omni, Gf, UsdGeom
    except Exception as e:
        logger.debug("Isaac Sim direct imports unavailable: %s", e)
        return None, None, None


def _try_import_ros2():
    try:
        import rclpy  # type: ignore
        from geometry_msgs.msg import Twist  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore
        from rclpy.node import Node  # type: ignore

        return rclpy, Node, Twist, Odometry
    except Exception as e:
        logger.debug("ROS 2 imports unavailable: %s", e)
        return None, None, None, None


@dataclass
class IsaacInterfaceConfig:
    control_rate_hz: float = 30.0
    config_path: str = "config/isaac_sim.yaml"
    drone_name: Optional[str] = None
    mode: str = "auto"  # auto|direct|ros2|local


class IsaacSimInterface:
    """MAVSDK-like interface backed by Isaac Sim direct or ROS 2 transport."""

    def __init__(self, drone_id: int = 0, config: Optional[IsaacInterfaceConfig] = None):
        self._drone_id = drone_id
        self._config = config or IsaacInterfaceConfig()
        self._drone_name = self._config.drone_name or os.getenv("ISAAC_DRONE_NAME", f"alpha_{drone_id}")

        self._connected = False
        self._running = False
        self._offboard_active = False
        self._armed = False
        self._in_air = False

        self._telemetry = TelemetryData()
        self._velocity_cmd = Vector3()
        self._rtl_home = Vector3()

        self._bg_tasks: list[asyncio.Task] = []

        # Mode resolution
        omni_mod, gf_mod, usdgeom_mod = _try_import_isaac_direct()
        rclpy_mod, node_mod, twist_mod, odom_mod = _try_import_ros2()
        self._omni = omni_mod
        self._Gf = gf_mod
        self._UsdGeom = usdgeom_mod
        self._rclpy = rclpy_mod
        self._Node = node_mod
        self._Twist = twist_mod
        self._Odometry = odom_mod
        self._mode = self._resolve_mode()

        # ROS 2 handles (mode=ros2)
        self._ros_node = None
        self._cmd_pub = None
        self._odom_sub = None
        self._ros_cmd_topic = ""
        self._ros_odom_topic = ""

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def telemetry(self) -> TelemetryData:
        return self._telemetry

    def _resolve_mode(self) -> str:
        requested = (self._config.mode or "auto").lower()
        if requested in {"direct", "ros2", "local"}:
            return requested
        if self._omni is not None:
            return "direct"
        if self._rclpy is not None:
            return "ros2"
        return "local"

    async def connect(self, connection_string: str = "", timeout: float = 5.0) -> bool:
        _ = connection_string, timeout
        self._connected = True
        self._running = True
        self._telemetry.timestamp = time.time()

        if self._mode == "ros2":
            if not self._init_ros2():
                self._connected = False
                self._running = False
                return False
            self._bg_tasks.append(asyncio.create_task(self._ros_spin_loop()))
        elif self._mode == "direct":
            self._bg_tasks.append(asyncio.create_task(self._direct_sync_loop()))
        else:
            self._bg_tasks.append(asyncio.create_task(self._local_kinematics_loop()))

        logger.info("IsaacSimInterface connected in %s mode for %s", self._mode, self._drone_name)
        return True

    async def disconnect(self):
        self._running = False
        for task in self._bg_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._bg_tasks.clear()

        if self._mode == "ros2" and self._rclpy is not None and self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception as e:
                logger.debug("ROS 2 node cleanup error: %s", e)

        self._connected = False

    def get_position(self) -> Vector3:
        return Vector3(self._telemetry.position.x, self._telemetry.position.y, self._telemetry.position.z)

    def get_velocity(self) -> Vector3:
        return Vector3(self._telemetry.velocity.x, self._telemetry.velocity.y, self._telemetry.velocity.z)

    def get_altitude(self) -> float:
        return -self._telemetry.position.z

    def get_battery(self) -> float:
        return self._telemetry.battery_percent or 100.0

    def is_armed(self) -> bool:
        return self._armed

    def is_in_air(self) -> bool:
        return self._in_air

    async def arm(self) -> bool:
        self._armed = True
        self._telemetry.armed = True
        return True

    async def disarm(self) -> bool:
        if self._in_air:
            return False
        self._armed = False
        self._telemetry.armed = False
        return True

    async def takeoff(self, altitude: float = 10.0) -> bool:
        if not self._armed:
            await self.arm()
        pos = self.get_position()
        self._telemetry.position = Vector3(pos.x, pos.y, -abs(altitude))
        self._in_air = True
        self._telemetry.in_air = True
        return True

    async def land(self) -> bool:
        pos = self.get_position()
        self._telemetry.position = Vector3(pos.x, pos.y, 0.0)
        self._telemetry.velocity = Vector3()
        self._velocity_cmd = Vector3()
        self._in_air = False
        self._telemetry.in_air = False
        return True

    async def return_to_launch(self) -> bool:
        self._telemetry.position = Vector3(self._rtl_home.x, self._rtl_home.y, self._telemetry.position.z)
        return True

    async def start_offboard(self) -> bool:
        self._offboard_active = True
        return True

    async def stop_offboard(self) -> bool:
        self._offboard_active = False
        self._velocity_cmd = Vector3()
        return True

    async def set_velocity_ned(self, north: float, east: float, down: float, yaw_deg: float = 0.0) -> bool:
        _ = yaw_deg
        self._velocity_cmd = Vector3(float(north), float(east), float(down))
        self._telemetry.velocity = Vector3(self._velocity_cmd.x, self._velocity_cmd.y, self._velocity_cmd.z)
        self._publish_ros_velocity(self._velocity_cmd)
        return True

    async def wait_for_altitude(self, target_altitude: float, tolerance: float = 0.5, timeout: float = 30.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if abs(self.get_altitude() - target_altitude) <= tolerance:
                return True
            await asyncio.sleep(0.1)
        return False

    async def wait_for_landed(self, timeout: float = 30.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_in_air():
                return True
            await asyncio.sleep(0.1)
        return False

    def _init_ros2(self) -> bool:
        if self._rclpy is None or self._Node is None:
            return False
        try:
            if not self._rclpy.ok():
                self._rclpy.init(args=None)

            self._ros_node = self._Node(f"isaac_sim_interface_{self._drone_name}")
            topics = self._load_topics_for_drone(self._drone_name)
            self._ros_cmd_topic = topics.get("cmd_vel", f"/{self._drone_name}/cmd_vel")
            self._ros_odom_topic = topics.get("odom", f"/{self._drone_name}/odom")

            self._cmd_pub = self._ros_node.create_publisher(self._Twist, self._ros_cmd_topic, 10)
            self._odom_sub = self._ros_node.create_subscription(self._Odometry, self._ros_odom_topic, self._on_odom, 10)
            return True
        except Exception as exc:
            logger.error("Failed to init ROS2 mode for IsaacSimInterface: %s", exc)
            return False

    async def _ros_spin_loop(self):
        while self._running and self._rclpy is not None and self._ros_node is not None:
            try:
                self._rclpy.spin_once(self._ros_node, timeout_sec=0.01)
            except Exception as e:
                logger.debug("ROS 2 spin error: %s", e)
            await asyncio.sleep(0.01)

    async def _direct_sync_loop(self):
        period = 1.0 / max(1.0, self._config.control_rate_hz)
        while self._running:
            now = time.time()
            dt = period
            if self._offboard_active or self._in_air:
                pos = self.get_position()
                vel = self._velocity_cmd
                new_pos = Vector3(pos.x + vel.x * dt, pos.y + vel.y * dt, pos.z + vel.z * dt)
                self._telemetry.position = new_pos
                self._telemetry.velocity = Vector3(vel.x, vel.y, vel.z)
                self._write_stage_position(new_pos)
            else:
                stage_pos = self._read_stage_position()
                if stage_pos is not None:
                    self._telemetry.position = stage_pos
                    self._telemetry.velocity = Vector3()
            self._telemetry.timestamp = now
            self._telemetry.in_air = self._in_air
            await asyncio.sleep(period)

    async def _local_kinematics_loop(self):
        period = 1.0 / max(1.0, self._config.control_rate_hz)
        while self._running:
            now = time.time()
            if self._offboard_active or self._in_air:
                pos = self.get_position()
                vel = self._velocity_cmd
                self._telemetry.position = Vector3(pos.x + vel.x * period, pos.y + vel.y * period, pos.z + vel.z * period)
                self._telemetry.velocity = Vector3(vel.x, vel.y, vel.z)
            self._telemetry.timestamp = now
            self._telemetry.in_air = self._in_air
            await asyncio.sleep(period)

    def _publish_ros_velocity(self, ned: Vector3):
        if self._mode != "ros2" or self._cmd_pub is None or self._Twist is None:
            return
        try:
            msg = self._Twist()
            # NED -> ENU
            msg.linear.x = float(ned.y)
            msg.linear.y = float(ned.x)
            msg.linear.z = float(-ned.z)
            self._cmd_pub.publish(msg)
        except Exception as e:
            logger.debug("ROS 2 velocity publish failed: %s", e)

    def _on_odom(self, msg):
        # ENU -> NED
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self._telemetry.position = Vector3(x=float(p.y), y=float(p.x), z=-float(p.z))
        self._telemetry.velocity = Vector3(x=float(v.y), y=float(v.x), z=-float(v.z))
        self._telemetry.timestamp = time.time()
        self._in_air = (-self._telemetry.position.z) > 0.3
        self._telemetry.in_air = self._in_air

    def _read_stage_position(self) -> Optional[Vector3]:
        if self._omni is None or self._UsdGeom is None:
            return None
        try:
            stage = self._omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._drone_prim_path())
            if not prim.IsValid():
                return None
            xform = self._UsdGeom.Xformable(prim)
            mat = xform.GetLocalTransformation()
            t = mat.ExtractTranslation()
            return Vector3(float(t[0]), float(t[1]), -float(t[2]))
        except Exception as e:
            logger.debug("Stage position read failed: %s", e)
            return None

    def _write_stage_position(self, ned_pos: Vector3):
        if self._omni is None or self._UsdGeom is None or self._Gf is None:
            return
        try:
            stage = self._omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._drone_prim_path())
            if not prim.IsValid():
                return
            xform = self._UsdGeom.Xformable(prim)
            translate = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == self._UsdGeom.XformOp.TypeTranslate:
                    translate = op
                    break
            if translate is None:
                translate = xform.AddTranslateOp(self._UsdGeom.XformOp.PrecisionDouble)
            translate.Set(self._Gf.Vec3d(float(ned_pos.x), float(ned_pos.y), float(-ned_pos.z)))
        except Exception as e:
            logger.debug("Stage position write failed: %s", e)

    def _drone_prim_path(self) -> str:
        parts = self._drone_name.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            return f"/World/Drones/{parts[0].capitalize()}_{parts[1]}"
        return f"/World/Drones/{self._drone_name}"

    def _load_topics_for_drone(self, drone_name: str) -> dict:
        out = {}
        try:
            with open(self._config.config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            drones = raw.get("drones", {})
            cfg = drones.get(drone_name, {})
            out = cfg.get("topics", {}) or {}
        except Exception as e:
            logger.debug("Failed to load topics for %s: %s", drone_name, e)
        return out
