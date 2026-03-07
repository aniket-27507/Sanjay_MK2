"""
Project Sanjay Mk2 - Mission Runner (Decentralized Swarm)
==========================================================
Runs a full 6-drone decentralized mission in the clustered test
environment created by ``create_surveillance_scene.py``.

This script now:
    1. Initializes 6 AvoidanceManagers + 6 AlphaRegimentCoordinators
    2. Enables default-on Boids + CBBA flocking in each coordinator
    3. Exchanges CBBA/state gossip over an in-process broadcast bus
    4. Feeds synthetic LiDAR to each drone each tick
    5. Blends boids desired velocity with APF/HPL safety outputs
    6. Updates the Mission Overlay with live telemetry
    7. Detects collisions / unsafe spacing and logs failures
    8. Dumps mission telemetry to console + JSON on completion

This can be run:
    A) Inside Isaac Sim (full 3D rendering + physics-driven LiDAR)
    B) Standalone (headless with procedural obstacle geometry)

For mode (B), the script generates synthetic LiDAR returns from
the obstacle definitions so you can validate the avoidance stack
without Isaac Sim installed.

Usage:
    # Inside Isaac Sim after running create_surveillance_scene.py
    exec(open('scripts/isaac_sim/run_mission.py').read())

    # Standalone headless
    python scripts/isaac_sim/run_mission.py --headless

@author: Archishman Paul
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ── Project root on PATH ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.types.drone_types import DroneConfig, DroneState, DroneType, FlightMode, Vector3, Waypoint
from src.single_drone.flight_control.flight_controller import FlightController
from src.single_drone.flight_control.waypoint_controller import WaypointController
from src.simulation.surveillance_layout import (
    ALPHA_ALTITUDE,
    BETA_ALTITUDE,
    BETA_ID,
    FORMATION_CENTER,
    FORMATION_SPACING,
    MISSION_WAYPOINTS,
    build_obstacle_database,
)
from src.swarm.coordination import AlphaRegimentCoordinator, RegimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MissionRunner")


def _load_obstacle_database() -> List[Dict]:
    """Load obstacles from shared surveillance layout (NED frame)."""
    return build_obstacle_database(ned_frame=True)


# ═══════════════════════════════════════════════════════════════════
#  Synthetic LiDAR Generator
# ═══════════════════════════════════════════════════════════════════


class SyntheticLidar:
    """
    Generates synthetic 3D LiDAR returns from the obstacle database.

    Casts rays in a 360 x 30 deg pattern and checks intersection with
    axis-aligned bounding boxes.  Uses fully vectorized numpy operations
    for the slab-method ray-AABB test across all rays and obstacles
    simultaneously.
    """

    def __init__(
        self,
        obstacles: List[Dict],
        h_rays: int = 180,
        v_rays: int = 8,
        max_range: float = 30.0,
        cull_range: float = 60.0,
    ):
        self._max_range = max_range
        self._cull_range = cull_range

        h_angles = np.linspace(0, 2 * math.pi, h_rays, endpoint=False)
        v_angles = np.linspace(-15, 15, v_rays) * math.pi / 180
        va_grid, ha_grid = np.meshgrid(v_angles, h_angles, indexing="ij")
        cos_va = np.cos(va_grid.ravel())
        self._directions = np.column_stack([
            cos_va * np.cos(ha_grid.ravel()),
            cos_va * np.sin(ha_grid.ravel()),
            np.sin(va_grid.ravel()),
        ]).astype(np.float64)

        self._obs_centers = np.array(
            [[o["x"], o["y"], o["z"]] for o in obstacles], dtype=np.float64,
        )
        half_extents = np.array(
            [[o["w"] / 2, o["d"] / 2, o["h"] / 2] for o in obstacles], dtype=np.float64,
        )
        self._aabb_min = self._obs_centers - half_extents
        self._aabb_max = self._obs_centers + half_extents

    def scan(self, position: Vector3) -> np.ndarray:
        """
        Generate a synthetic point cloud from the given drone position.

        Returns:
            Nx3 array of hit points in body frame.
        """
        pos = np.array([position.x, position.y, position.z], dtype=np.float64)

        dists = np.linalg.norm(self._obs_centers - pos, axis=1)
        mask = dists < self._cull_range
        if not np.any(mask):
            return np.empty((0, 3), dtype=np.float32)
        aabb_min = self._aabb_min[mask]
        aabb_max = self._aabb_max[mask]

        dirs = self._directions
        n_rays = dirs.shape[0]
        n_obs = aabb_min.shape[0]

        eps = 1e-10
        safe = np.where(np.abs(dirs) > eps, dirs, eps)
        inv_dir = 1.0 / safe

        origin_expanded = pos[np.newaxis, np.newaxis, :]
        inv_expanded = inv_dir[:, np.newaxis, :]

        t_lo = (aabb_min[np.newaxis, :, :] - origin_expanded) * inv_expanded
        t_hi = (aabb_max[np.newaxis, :, :] - origin_expanded) * inv_expanded

        t_enter = np.minimum(t_lo, t_hi)
        t_exit = np.maximum(t_lo, t_hi)

        t_min = np.max(t_enter, axis=2)
        t_max = np.min(t_exit, axis=2)

        t_hit = np.where(t_min > 0, t_min, t_max)
        valid = (t_max >= t_min) & (t_hit > 0.3) & (t_hit < self._max_range)

        t_hit = np.where(valid, t_hit, self._max_range + 1.0)
        best_idx = np.argmin(t_hit, axis=1)
        best_t = t_hit[np.arange(n_rays), best_idx]

        hit_mask = best_t <= self._max_range
        if not np.any(hit_mask):
            return np.empty((0, 3), dtype=np.float32)

        hit_dirs = dirs[hit_mask]
        hit_t = best_t[hit_mask, np.newaxis]
        # Return relative points so downstream drivers can transform
        # into world frame exactly once.
        points = hit_dirs * hit_t

        return points.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Simulated Drone (headless physics-free)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SimDrone:
    """Lightweight simulated drone for headless testing."""
    drone_id: int
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    mode: FlightMode = FlightMode.NAVIGATING
    battery: float = 100.0
    is_active: bool = True

    def step(self, velocity_command: Vector3, dt: float):
        """Advance physics by one tick."""
        # Simple kinematic integration
        self.velocity = velocity_command
        self.position = Vector3(
            x=self.position.x + velocity_command.x * dt,
            y=self.position.y + velocity_command.y * dt,
            z=self.position.z + velocity_command.z * dt,
        )
        self.battery -= 0.001 * dt  # Slow drain

    def to_state(self) -> DroneState:
        return DroneState(
            drone_id=self.drone_id,
            drone_type=DroneType.ALPHA,
            position=self.position,
            velocity=self.velocity,
            mode=self.mode,
            battery=self.battery,
        )


# ═══════════════════════════════════════════════════════════════════
#  Mission Runner
# ═══════════════════════════════════════════════════════════════════


from src.core.utils.geometry import hex_positions as _hex_positions


class MissionRunner:
    """
    Drives a decentralized 6-drone Alpha Regiment mission.

    Integrates:
        - AvoidanceManager per drone (APF + HPL + Tactical A*)
        - AlphaRegimentCoordinator per drone (Boids + CBBA + gossip)
        - SyntheticLidar for headless testing
        - Mission Overlay for live telemetry in Isaac Sim
        - Debug log dump on mission failure
    """

    def __init__(self, headless: bool = True, use_waypoint_controller: bool = False):
        self._headless = headless
        self._use_waypoint_controller = use_waypoint_controller
        self._dt = 1.0 / 30.0  # 30 Hz control rate

        # Load obstacle database
        self._obstacles = _load_obstacle_database()
        logger.info(f"Loaded {len(self._obstacles)} obstacles from scene database")

        # Build synthetic LiDAR
        self._lidar = SyntheticLidar(self._obstacles)

        # Spawn drones
        self._drones: Dict[int, SimDrone] = {}
        self._formation_offsets: Dict[int, Vector3] = {}
        self._formation_goal_scale = 0.55
        hex_pos = _hex_positions(*FORMATION_CENTER, FORMATION_SPACING)
        for i, (x, y) in enumerate(hex_pos):
            self._drones[i] = SimDrone(
                drone_id=i,
                position=Vector3(x=x, y=y, z=-ALPHA_ALTITUDE),  # NED
            )
            self._formation_offsets[i] = Vector3(
                x=x - FORMATION_CENTER[0],
                y=y - FORMATION_CENTER[1],
                z=0.0,
            )

        # Avoidance managers (lazy import)
        self._avoidance_managers: Dict[int, object] = {}
        self._coordinators: Dict[int, AlphaRegimentCoordinator] = {}

        # Mission state
        self._mission_waypoints = [
            Waypoint(
                position=Vector3(x=wp["pos"][0], y=wp["pos"][1], z=-wp["pos"][2]),
                speed=5.0,
                acceptance_radius=5.0,
            )
            for wp in MISSION_WAYPOINTS
        ]
        self._mission_waypoint_index = 0
        self._waypoint_reached_radius = 36.0
        self._waypoint_quorum_ratio = 0.60

        # Mission timing
        self._start_time = 0.0
        self._max_mission_time = 600.0  # 10 minutes

        # Telemetry log (bounded to prevent memory growth in long missions)
        self._event_log: deque = deque(maxlen=10000)
        self._collision_count = 0
        self._hpl_override_count = 0
        self._min_inter_drone_distance = float("inf")

        # Mission overlay (Isaac Sim mode)
        self._overlay = None
        # Stage sync subscription (Isaac Sim mode) - runs on main Kit thread
        self._sync_sub = None
        # Bridge reference for publishing velocity commands (Isaac Sim mode)
        self._bridge = None
        self._bridge_ros_started = False

        # Optional controller-backed execution path.
        self._flight_controller: Optional[FlightController] = None
        self._waypoint_controller: Optional[WaypointController] = None

    async def run(self):
        """Execute the complete mission."""
        logger.info("=" * 65)

        if self._use_waypoint_controller:
            await self._run_waypoint_controller_path()
            return
        logger.info("  PROJECT SANJAY MK2 — Mission Runner")
        logger.info(f"  Mode: {'Headless' if self._headless else 'Isaac Sim'}")
        logger.info(f"  Drones: {len(self._drones)}")
        logger.info(f"  Obstacles: {len(self._obstacles)}")
        logger.info("  Mission: Full 6-drone decentralized autonomy")
        logger.info("=" * 65)

        # Initialize avoidance managers
        self._init_avoidance()
        await self._init_coordinators()

        # Connect to Isaac Sim overlay, stage sync, and bridge if available
        if not self._headless:
            self._init_overlay()
            self._register_stage_sync()
            self._init_bridge()

        self._start_time = time.time()
        self._log_event("SYSTEM", "Mission started")

        tick = 0
        mission_result = "UNKNOWN"

        try:
            while True:
                tick += 1
                elapsed = time.time() - self._start_time
                self._service_bridge_callbacks()

                # ── Timeout check ──
                if elapsed > self._max_mission_time:
                    self._log_event("SYSTEM", "Mission TIMEOUT", "CRITICAL")
                    mission_result = "TIMEOUT"
                    break

                # ── Feed LiDAR to all drones ──
                for drone_id, drone in self._drones.items():
                    if not drone.is_active:
                        continue

                    scan_pos = Vector3(x=drone.position.x, y=drone.position.y, z=drone.position.z)
                    points = self._lidar.scan(scan_pos)

                    if drone_id in self._avoidance_managers:
                        mgr = self._avoidance_managers[drone_id]
                        mgr.feed_lidar_points(points, drone_position=drone.position)

                # ── Local state update per decentralized coordinator ──
                for drone_id, coordinator in self._coordinators.items():
                    coordinator.update_member_state(drone_id, self._drones[drone_id].to_state())

                # ── In-process gossip broadcast ──
                gossip_payloads = {
                    drone_id: coordinator.prepare_gossip_payload()
                    for drone_id, coordinator in self._coordinators.items()
                }
                for receiver_id, coordinator in self._coordinators.items():
                    for sender_id, payload in gossip_payloads.items():
                        if sender_id == receiver_id or not payload:
                            continue
                        coordinator.ingest_gossip_payload(payload)

                # ── One coordination tick per drone ──
                for drone_id, coordinator in self._coordinators.items():
                    coordinator.set_forced_goal(self._get_drone_waypoint_goal(drone_id))
                for coordinator in self._coordinators.values():
                    coordinator.coordination_step()

                # ── Apply commands to all drones ──
                for drone_id, drone in self._drones.items():
                    if not drone.is_active:
                        continue

                    coordinator = self._coordinators.get(drone_id)
                    mgr = self._avoidance_managers.get(drone_id)
                    desired_velocity = coordinator.get_desired_velocity(drone_id) if coordinator else Vector3()
                    goal = coordinator.get_desired_goal(drone_id) if coordinator else None

                    if mgr is not None:
                        mgr.set_boids_velocity(desired_velocity)
                        if goal is not None:
                            mgr.set_goal(goal)
                        velocity = mgr.compute_avoidance(
                            drone_position=drone.position,
                            drone_velocity=drone.velocity,
                        )
                    else:
                        velocity = desired_velocity

                    drone.step(velocity, self._dt)
                    self._publish_velocity(drone_id, velocity)

                for drone_id, drone in self._drones.items():
                    if not drone.is_active:
                        continue
                    mgr = self._avoidance_managers.get(drone_id)
                    if mgr and mgr.closest_obstacle_distance < 0.5:
                        self._collision_count += 1
                        self._log_event(
                            f"Alpha_{drone_id}",
                                f"COLLISION — obstacle at {mgr.closest_obstacle_distance:.2f}m "
                            f"(total: {self._collision_count})",
                            "CRITICAL",
                        )
                        if self._collision_count >= 3:
                            mission_result = "FAILED_COLLISION"
                            break

                    if mgr and mgr.is_hpl_overriding:
                        self._hpl_override_count += 1

                if mission_result == "FAILED_COLLISION":
                    break

                min_distance = self._compute_min_inter_drone_distance()
                self._min_inter_drone_distance = min(self._min_inter_drone_distance, min_distance)
                if min_distance < 20.0:
                    self._log_event(
                        "SWARM",
                        f"Unsafe inter-drone spacing detected ({min_distance:.1f}m)",
                        "CRITICAL",
                    )
                    mission_result = "FAILED_SEPARATION"
                    break

                if self._mission_waypoint_index >= len(self._mission_waypoints):
                    mission_result = "SUCCESS"
                    break

                # ── Update overlay (Isaac Sim) ──
                if self._overlay and tick % 10 == 0:
                    for did, drone in self._drones.items():
                        mgr = self._avoidance_managers.get(did)
                        if mgr:
                            self._overlay.update_drone_state(
                                f"Alpha_{did}", mgr.get_telemetry()
                            )

                # ── Periodic console status ──
                if tick % 300 == 0:  # Every ~10s
                    mgr = self._avoidance_managers.get(0)
                    state_name = mgr.state.name if mgr else "N/A"
                    closest = mgr.closest_obstacle_distance if mgr else float("inf")
                    logger.info(
                        f"[{elapsed:>6.1f}s] Decentralized swarm active | "
                        f"WP: {self._mission_waypoint_index}/{len(self._mission_waypoints)} | "
                        f"State: {state_name} | "
                        f"Closest: {closest:.1f}m | "
                        f"MinSep: {self._min_inter_drone_distance:.1f}m | "
                        f"HPL: {self._hpl_override_count}"
                    )

                self._update_waypoint_progress()
                await asyncio.sleep(self._dt)

        except KeyboardInterrupt:
            mission_result = "ABORTED"
            self._log_event("SYSTEM", "Mission aborted by user", "WARNING")

        # ── Finalize ──
        self._finalize(mission_result)

    async def _run_waypoint_controller_path(self):
        """
        Optional execution path that uses FlightController + WaypointController.
        This is useful for validating autonomous waypoint orchestration and
        manual-overrides against the isaac_sim backend.
        """
        self._start_time = time.time()
        self._flight_controller = FlightController(drone_id=0, backend="isaac_sim")
        self._waypoint_controller = WaypointController(self._flight_controller)

        for wp in self._mission_waypoints:
            self._waypoint_controller.add_waypoint(
                position=wp.position,
                speed=wp.speed,
                acceptance_radius=wp.acceptance_radius,
                hold_time=wp.hold_time,
            )

        ok = await self._waypoint_controller.execute_mission(
            auto_arm_takeoff=True,
            takeoff_altitude_m=ALPHA_ALTITUDE,
            enable_avoidance=True,
        )
        result = "SUCCESS" if ok else "FAILED_WAYPOINT_CONTROLLER"
        await self._flight_controller.shutdown()
        self._finalize(result)

    def _init_avoidance(self):
        """Initialize an AvoidanceManager per drone."""
        try:
            from src.single_drone.obstacle_avoidance.avoidance_manager import (
                AvoidanceManager,
                AvoidanceManagerConfig,
            )

            for drone_id in self._drones:
                config = AvoidanceManagerConfig()
                config.control_rate_hz = 30.0
                mgr = AvoidanceManager(drone_id=drone_id, config=config)
                self._avoidance_managers[drone_id] = mgr
                logger.info(f"  AvoidanceManager initialized for Alpha_{drone_id}")

        except ImportError as e:
            logger.warning(f"Could not import AvoidanceManager: {e}")
            logger.warning("Running without obstacle avoidance (direct P-control)")

    async def _init_coordinators(self):
        """Initialize one decentralized regiment coordinator per drone."""
        for drone_id in self._drones:
            cfg = RegimentConfig(
                formation_spacing=FORMATION_SPACING,
                formation_altitude=ALPHA_ALTITUDE,
                total_coverage_area=1000.0,
                use_boids_flocking=True,
            )
            coordinator = AlphaRegimentCoordinator(my_drone_id=drone_id, config=cfg)
            await coordinator.initialize()
            for peer_id in self._drones:
                coordinator.register_drone(peer_id)
            coordinator.set_forced_goal(self._get_drone_waypoint_goal(drone_id))
            self._coordinators[drone_id] = coordinator

    def _get_current_waypoint_goal(self) -> Optional[Vector3]:
        if self._mission_waypoint_index >= len(self._mission_waypoints):
            return None
        return self._mission_waypoints[self._mission_waypoint_index].position

    def _get_drone_waypoint_goal(self, drone_id: int) -> Optional[Vector3]:
        waypoint_center = self._get_current_waypoint_goal()
        if waypoint_center is None:
            return None
        offset = self._formation_offsets.get(drone_id, Vector3())
        return Vector3(
            x=waypoint_center.x + offset.x * self._formation_goal_scale,
            y=waypoint_center.y + offset.y * self._formation_goal_scale,
            z=waypoint_center.z + offset.z,
        )

    def _update_waypoint_progress(self):
        if self._get_current_waypoint_goal() is None:
            return

        active = [d for d in self._drones.values() if d.is_active]
        if not active:
            return

        active_ids = [d.drone_id for d in active]
        quorum = max(1, int(math.ceil(len(active_ids) * self._waypoint_quorum_ratio)))
        reached = 0
        for drone_id in active_ids:
            drone_goal = self._get_drone_waypoint_goal(drone_id)
            drone_state = self._drones.get(drone_id)
            if drone_goal is None or drone_state is None:
                continue
            if drone_state.position.distance_to(drone_goal) <= self._waypoint_reached_radius:
                reached += 1

        if reached < quorum:
            return

        wp = MISSION_WAYPOINTS[self._mission_waypoint_index]
        self._mission_waypoint_index += 1
        self._log_event(
            "SWARM",
            f"Reached {wp['id']} ({wp['label']}) with quorum {reached}/{len(active_ids)}",
            "SUCCESS",
        )
        if self._overlay:
            try:
                self._overlay.advance_waypoint("SWARM", wp["id"])
            except Exception:
                pass

    def _compute_min_inter_drone_distance(self) -> float:
        """Compute nearest pairwise spacing among active drones."""
        active = [d for d in self._drones.values() if d.is_active]
        if len(active) < 2:
            return float("inf")

        min_dist = float("inf")
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dist = active[i].position.distance_to(active[j].position)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _init_overlay(self):
        """Connect to MissionOverlay if in Isaac Sim."""
        try:
            from scripts.isaac_sim.create_surveillance_scene import get_mission_overlay
            self._overlay = get_mission_overlay()
        except Exception:
            self._overlay = None

    def _init_bridge(self):
        """Connect to the Isaac Sim ROS 2 bridge for velocity publishing."""
        try:
            from src.integration.isaac_sim_bridge import IsaacSimBridgeNode, BridgeConfig, is_ros2_available

            if not is_ros2_available():
                logger.info("ROS 2 unavailable — bridge velocity publishing disabled")
                return

            config = BridgeConfig.from_yaml(
                os.path.join(PROJECT_ROOT, "config", "isaac_sim.yaml")
            )
            import rclpy
            if not rclpy.ok():
                rclpy.init()
                self._bridge_ros_started = True
            self._bridge = IsaacSimBridgeNode(config)

            def _lidar_hook(drone_name: str, points: np.ndarray):
                drone_id = int(drone_name.split("_")[-1]) if "_" in drone_name else 0
                mgr = self._avoidance_managers.get(drone_id)
                drone = self._drones.get(drone_id)
                if mgr is not None and drone is not None:
                    mgr.feed_lidar_points(points, drone_position=drone.position)

            def _threat_hook(drone_name: str, threat):
                logger.info(f"Bridge threat from {drone_name}: {threat}")

            self._bridge.register_autonomy_hooks(
                on_threat=_threat_hook,
                on_lidar=_lidar_hook,
            )
            logger.info("Isaac Sim bridge connected for velocity publishing")
        except Exception as e:
            logger.debug(f"Bridge init skipped: {e}")
            self._bridge = None
            self._bridge_ros_started = False

    def _service_bridge_callbacks(self):
        if self._bridge is None:
            return
        try:
            import rclpy
            rclpy.spin_once(self._bridge, timeout_sec=0.0)
        except Exception:
            pass

    def _publish_velocity(self, drone_id: int, velocity: Vector3):
        """Publish velocity command to Isaac Sim via ROS 2 bridge."""
        if self._bridge is None:
            return
        drone_name = "beta_0" if drone_id == BETA_ID else f"alpha_{drone_id}"
        self._bridge.send_velocity(
            drone_name, vx=velocity.x, vy=velocity.y, vz=velocity.z
        )

    def _register_stage_sync(self):
        """Register per-frame stage sync on main Kit thread so drones move visibly."""
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            if app:
                self._sync_sub = (
                    app.get_update_event_stream()
                    .create_subscription_to_pop(self._sync_drones_to_stage)
                )
                logger.info("Stage sync registered (drones will move in viewport)")
        except Exception as e:
            logger.debug(f"Stage sync registration skipped: {e}")

    def _sync_drones_to_stage(self, event=None):
        """Write SimDrone positions to the Isaac Sim stage so drones move visibly."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            for drone_id, drone in self._drones.items():
                if not drone.is_active:
                    continue
                prim_path = "/World/Drones/Beta_0" if drone_id == BETA_ID else f"/World/Drones/Alpha_{drone_id}"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                # NED: (x, y, z) with z negative = altitude -> Isaac (x, y, -z)
                pos = Gf.Vec3d(drone.position.x, drone.position.y, -drone.position.z)

                # Use Xformable directly (robust for VisualCuboid and Robot prims)
                xform = UsdGeom.Xformable(prim)
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op is None:
                    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                translate_op.Set(pos)
        except Exception as e:
            logger.debug(f"Stage sync skipped: {e}")

    # ── Logging ───────────────────────────────────────────────────

    def _log_event(self, source: str, message: str, level: str = "INFO"):
        elapsed = time.time() - self._start_time if self._start_time else 0
        entry = {
            "timestamp": time.time(),
            "elapsed": elapsed,
            "source": source,
            "level": level,
            "message": message,
        }
        self._event_log.append(entry)

        if level in ("CRITICAL", "WARNING", "SUCCESS"):
            log_fn = (
                logger.critical if level == "CRITICAL"
                else logger.warning if level == "WARNING"
                else logger.info
            )
            log_fn(f"[{source}] {message}")

    # ── Finalization ──────────────────────────────────────────────

    def _finalize(self, result: str):
        """Print final results and dump debug log on failure."""
        # Unregister stage sync subscription
        if self._sync_sub is not None:
            try:
                self._sync_sub.unsubscribe()
            except Exception:
                pass
            self._sync_sub = None

        if self._bridge is not None:
            try:
                self._bridge.destroy_node()
            except Exception:
                pass
            self._bridge = None
        if self._bridge_ros_started:
            try:
                import rclpy
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            self._bridge_ros_started = False

        elapsed = time.time() - self._start_time

        print()
        print("=" * 65)
        if result == "SUCCESS":
            print("  MISSION COMPLETE")
        else:
            print(f"  MISSION RESULT: {result}")
        print("=" * 65)

        print(f"\n  Duration          : {elapsed:.1f}s")
        print(f"  Min Separation    : {self._min_inter_drone_distance:.1f}m")
        print(f"  HPL Overrides     : {self._hpl_override_count}")
        print(f"  Collisions        : {self._collision_count}")

        # Per-drone final state
        print("\n  -- Drone Final Positions --")
        for did, drone in sorted(self._drones.items()):
            mgr = self._avoidance_managers.get(did)
            state = mgr.state.name if mgr else "N/A"
            print(f"    Alpha_{did}: ({drone.position.x:>7.1f}, {drone.position.y:>7.1f}, "
                  f"{drone.position.z:>7.1f})  State={state}")

        # On failure: dump full debug log
        if result != "SUCCESS":
            self._dump_debug_log(result, elapsed)

        # On success or failure: save JSON log
        self._save_log(result, elapsed)

        print("=" * 65)

    def _dump_debug_log(self, result: str, elapsed: float):
        """Dump full debug log to console on failure."""
        print(f"\n  -- DEBUG LOG ({len(self._event_log)} events) --")
        for ev in self._event_log:
            ts = ev["elapsed"]
            lvl = ev["level"]
            src = ev["source"]
            msg = ev["message"]
            marker = "!!" if lvl == "CRITICAL" else ">>" if lvl == "WARNING" else "  "
            print(f"    {marker} [{ts:>7.1f}s] [{lvl:<8s}] [{src:<12s}] {msg}")

        # Avoidance telemetry snapshots
        print("\n  -- Avoidance Telemetry at Failure --")
        for did, mgr in self._avoidance_managers.items():
            telem = mgr.get_telemetry()
            print(f"    Alpha_{did}:")
            print(f"      State          : {telem.get('avoidance_state', 'N/A')}")
            print(f"      HPL State      : {telem.get('hpl_state', 'N/A')}")
            print(f"      Closest Obs    : {telem.get('closest_obstacle_m', 'N/A')}m")
            print(f"      Sub-Waypoints  : {telem.get('active_sub_waypoints', 0)}")
            print(f"      LiDAR Points   : {telem.get('lidar', {}).get('filtered_points', 0)}")
            print(f"      Obstacles      : {telem.get('lidar', {}).get('obstacle_count', 0)}")

    def _save_log(self, result: str, elapsed: float):
        """Save mission log to JSON."""
        log_dir = os.path.join(PROJECT_ROOT, "simulation", "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"mission_{result.lower()}_{int(time.time())}.json")

        log_data = {
            "result": result,
            "duration_s": elapsed,
            "min_inter_drone_distance_m": self._min_inter_drone_distance,
            "hpl_overrides": self._hpl_override_count,
            "collisions": self._collision_count,
            "drone_finals": {
                f"alpha_{did}": {
                    "position": [d.position.x, d.position.y, d.position.z],
                    "battery": d.battery,
                }
                for did, d in self._drones.items()
            },
            "events": list(self._event_log),
        }

        try:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
            print(f"\n  Log saved: {log_path}")
        except Exception as e:
            print(f"\n  Log save failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Run Sanjay MK2 obstacle avoidance mission")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run without Isaac Sim (synthetic LiDAR)")
    parser.add_argument("--isaac", action="store_true",
                        help="Run inside Isaac Sim (requires scene to be loaded)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Mission timeout in seconds (default: 600)")
    parser.add_argument(
        "--controller-path",
        action="store_true",
        help="Run mission via FlightController + WaypointController path",
    )
    args = parser.parse_args()

    headless = not args.isaac
    runner = MissionRunner(headless=headless, use_waypoint_controller=args.controller_path)
    runner._max_mission_time = args.timeout

    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
else:
    # When loaded via Isaac Sim Script Editor with an active event loop.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        runner = MissionRunner(headless=False)
        loop.create_task(runner.run())
