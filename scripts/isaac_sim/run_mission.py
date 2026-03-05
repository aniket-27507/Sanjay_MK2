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
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root on PATH ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.types.drone_types import DroneConfig, DroneState, DroneType, FlightMode, Vector3, Waypoint
from src.single_drone.flight_control.flight_controller import FlightController
from src.single_drone.flight_control.waypoint_controller import WaypointController
from src.swarm.coordination import AlphaRegimentCoordinator, RegimentConfig
from scripts.isaac_sim.waypoint_session import get_waypoint_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MissionRunner")


# ═══════════════════════════════════════════════════════════════════
#  Obstacle Database (mirrors the scene builder definitions)
# ═══════════════════════════════════════════════════════════════════
#  Each obstacle is (x, y, z_center, width, depth, height).
#  Keep in sync with create_surveillance_scene.py.


def _load_obstacle_database() -> List[Dict]:
    """Build the obstacle database matching the scene builder."""
    obstacles = []

    # ── Zone 1: Downtown ──
    DOWNTOWN_CENTER = (200, 200)
    dt_buildings = [
        (0, 0, 18, 18, 55), (25, 0, 15, 22, 42), (48, -5, 20, 16, 60),
        (0, 28, 12, 12, 38), (18, 25, 14, 20, 50),
        (75, 0, 10, 40, 30), (88, 0, 10, 40, 28), (75, 45, 24, 10, 22),
        (0, 60, 40, 8, 35), (0, 60, 8, 40, 35), (32, 60, 8, 40, 35),
    ]
    for ox, oy, w, d, h in dt_buildings:
        x, y = DOWNTOWN_CENTER[0] + ox, DOWNTOWN_CENTER[1] + oy
        obstacles.append({
            "x": x, "y": y, "z": h / 2, "w": w, "d": d, "h": h,
            "zone": "downtown",
        })

    # ── Zone 2: Industrial ──
    IND_CENTER = (500, 150)
    ind_objects = [
        (0, 0, 12, 12, 20, 0), (18, 0, 10, 10, 25, 0), (32, 5, 14, 14, 18, 0),
        (0, 18, 40, 1.5, 1.5, 12), (6, 0, 1.5, 30, 1.5, 15),
        (20, 10, 25, 1.0, 1.0, 20), (-10, -15, 60, 5, 8, 0),
        (-15, 10, 8, 6, 5, 0), (45, 20, 10, 8, 6, 0),
    ]
    for ox, oy, w, d, h, elev in ind_objects:
        x, y = IND_CENTER[0] + ox, IND_CENTER[1] + oy
        obstacles.append({
            "x": x, "y": y, "z": elev + h / 2, "w": w, "d": d, "h": h,
            "zone": "industrial",
        })

    # ── Zone 3: Residential ──
    RES_BASE = (350, 450)
    rng = np.random.RandomState(42)
    templates = [(10, 8, 6), (12, 10, 8), (8, 8, 5), (14, 10, 10), (10, 12, 7)]
    for row in range(5):
        for col in range(5):
            t = templates[rng.randint(len(templates))]
            w, d, h = t[0] + rng.uniform(-2, 2), t[1] + rng.uniform(-2, 2), t[2] + rng.uniform(-1, 3)
            x = RES_BASE[0] + col * 22
            y = RES_BASE[1] + row * 22
            obstacles.append({
                "x": x, "y": y, "z": h / 2, "w": w, "d": d, "h": h,
                "zone": "residential",
            })

    # ── Zone 4: Forest ──
    FOREST_CENTER = (150, 600)
    rng2 = np.random.RandomState(77)
    for _ in range(80):
        angle = rng2.uniform(0, 2 * math.pi)
        radius = rng2.uniform(0, 120) ** 0.5 * 120 ** 0.5
        x = FOREST_CENTER[0] + radius * math.cos(angle)
        y = FOREST_CENTER[1] + radius * math.sin(angle)
        trunk_h = rng2.uniform(8, 22)
        canopy_r = rng2.uniform(4, 10)
        # Tree trunk
        obstacles.append({"x": x, "y": y, "z": trunk_h / 2, "w": 0.4, "d": 0.4, "h": trunk_h, "zone": "forest"})
        # Canopy
        obstacles.append({"x": x, "y": y, "z": trunk_h + canopy_r * 0.3,
                          "w": canopy_r * 2, "d": canopy_r * 2, "h": canopy_r, "zone": "forest"})

    # ── Zone 5: Powerlines ──
    for i in range(8):
        x = 600 + i * 40
        obstacles.append({"x": x, "y": 400, "z": 75 / 2, "w": 2, "d": 2, "h": 75, "zone": "powerline"})

    for ax, ay, ah, aw in [(700, 200, 80, 2.0), (720, 280, 70, 1.5), (680, 350, 85, 2.5)]:
        obstacles.append({"x": ax, "y": ay, "z": ah / 2, "w": aw, "d": aw, "h": ah, "zone": "antenna"})

    # Convert obstacle centers from Isaac/ENU-style altitude (+Z up) to
    # the project's NED convention (negative Z is up) used by the runner.
    for obs in obstacles:
        obs["z"] = -float(obs["z"])

    return obstacles


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
        points = hit_dirs * hit_t

        return points.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Simulated Drone (headless physics-free)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SimDrone:
    """Lightweight simulated drone for headless testing."""
    drone_id: int
    drone_type: DroneType = DroneType.ALPHA
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
            drone_type=self.drone_type,
            position=self.position,
            velocity=self.velocity,
            mode=self.mode,
            battery=self.battery,
        )


# ═══════════════════════════════════════════════════════════════════
#  Mission Runner
# ═══════════════════════════════════════════════════════════════════


FORMATION_CENTER = (400, 350)
FORMATION_SPACING = 80.0
ALPHA_ALTITUDE = 65.0
BETA_ALTITUDE = 25.0
BETA_DRONE_ID = 6

OBSTACLE_ENGAGE_DIST = 30.0
OBSTACLE_FULL_DIST = 10.0
REJOIN_TOLERANCE = 5.0
REJOIN_SPEED = 7.0
REJOIN_P_GAIN = 1.5

MISSION_WAYPOINTS = [
    {"id": "WP_01", "pos": (200, 200, 65), "label": "Downtown Entry"},
    {"id": "WP_02", "pos": (250, 200, 65), "label": "Tower Gap"},
    {"id": "WP_03", "pos": (280, 260, 65), "label": "U-Trap Test"},
    {"id": "WP_04", "pos": (500, 160, 65), "label": "Industrial Entry"},
    {"id": "WP_05", "pos": (530, 170, 65), "label": "Pipe Corridor"},
    {"id": "WP_06", "pos": (550, 140, 65), "label": "Tank Slalom"},
    {"id": "WP_07", "pos": (150, 600, 55), "label": "Forest Ingress"},
    {"id": "WP_08", "pos": (200, 620, 50), "label": "Canopy Skim"},
    {"id": "WP_09", "pos": (620, 400, 65), "label": "Pylon Slalom"},
    {"id": "WP_10", "pos": (700, 400, 65), "label": "Antenna Weave"},
    {"id": "WP_11", "pos": (400, 350, 65), "label": "RTB"},
]


from src.core.utils.geometry import hex_center as _hex_center
from src.core.utils.geometry import hex_positions as _hex_positions


class FormationState(Enum):
    NORMAL = auto()
    AVOIDING = auto()
    REJOINING = auto()


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

    def __init__(
        self,
        headless: bool = True,
        use_waypoint_controller: bool = False,
        use_gui_waypoints: bool = False,
    ):
        self._headless = headless
        self._use_waypoint_controller = use_waypoint_controller
        self._use_gui_waypoints = use_gui_waypoints and (not headless)
        self._dt = 1.0 / 30.0  # 30 Hz control rate

        # Load obstacle database
        self._obstacles = _load_obstacle_database()
        logger.info(f"Loaded {len(self._obstacles)} obstacles from scene database")

        # Build synthetic LiDAR
        self._lidar = SyntheticLidar(self._obstacles)

        # Spawn drones
        self._drones: Dict[int, SimDrone] = {}
        alpha_positions = _hex_positions(*FORMATION_CENTER, FORMATION_SPACING, n=6)
        for i, (x, y) in enumerate(alpha_positions):
            self._drones[i] = SimDrone(
                drone_id=i,
                drone_type=DroneType.ALPHA,
                position=Vector3(x=x, y=y, z=-ALPHA_ALTITUDE),  # NED
            )
        beta_cx, beta_cy = _hex_center(*FORMATION_CENTER)
        self._drones[BETA_DRONE_ID] = SimDrone(
            drone_id=BETA_DRONE_ID,
            drone_type=DroneType.BETA,
            position=Vector3(x=beta_cx, y=beta_cy, z=-BETA_ALTITUDE),
        )
        self._alpha_ids = [did for did, d in self._drones.items() if d.drone_type == DroneType.ALPHA]
        self._beta_ids = [did for did, d in self._drones.items() if d.drone_type == DroneType.BETA]

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
        self._mission_success_duration = 120.0

        # Mission timing
        self._start_time = 0.0
        self._max_mission_time = 600.0  # 10 minutes

        # Telemetry log (bounded to prevent memory growth in long missions)
        self._event_log: deque = deque(maxlen=10000)
        self._collision_count = 0
        self._hpl_override_count = 0
        self._min_inter_drone_distance = float("inf")
        self._last_beta_proximity_warning_ts = 0.0

        # Mission overlay (Isaac Sim mode)
        self._overlay = None
        # Stage sync subscription (Isaac Sim mode) - runs on main Kit thread
        self._sync_sub = None
        # Bridge reference for publishing velocity commands (Isaac Sim mode)
        self._bridge = None

        # Optional controller-backed execution path.
        self._flight_controller: Optional[FlightController] = None
        self._waypoint_controller: Optional[WaypointController] = None
        self._session = get_waypoint_session() if self._use_gui_waypoints else None
        self._formation_offsets: Dict[int, Vector3] = {}
        self._formation_states: Dict[int, FormationState] = {
            drone_id: FormationState.NORMAL for drone_id in self._drones
        }
        if 0 in self._drones:
            center_ref = Vector3(x=beta_cx, y=beta_cy, z=-ALPHA_ALTITUDE)
            for drone_id, drone in self._drones.items():
                self._formation_offsets[drone_id] = Vector3(
                    x=drone.position.x - center_ref.x,
                    y=drone.position.y - center_ref.y,
                    z=drone.position.z - center_ref.z,
                )

    async def run(self):
        """Execute the complete mission."""
        logger.info("=" * 65)

        if self._use_waypoint_controller:
            await self._run_waypoint_controller_path()
            return
        logger.info("  PROJECT SANJAY MK2 — Mission Runner")
        logger.info(f"  Mode: {'Headless' if self._headless else 'Isaac Sim'}")
        logger.info(f"  Drones: {len(self._drones)} ({len(self._alpha_ids)} Alpha, {len(self._beta_ids)} Beta)")
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

        if self._use_gui_waypoints:
            mission_result = await self._run_gui_waypoint_swarm()
            self._finalize(mission_result)
            return

        tick = 0
        mission_result = "UNKNOWN"

        try:
            while True:
                tick += 1
                elapsed = time.time() - self._start_time

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
                    for beta_id in self._beta_ids:
                        beta_drone = self._drones.get(beta_id)
                        if beta_drone is not None and beta_drone.is_active:
                            coordinator.update_member_state(beta_id, beta_drone.to_state())

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
                            self._drone_label(drone_id),
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
                        f"Inter-drone spacing low ({min_distance:.1f}m)",
                        "WARNING",
                    )
                if min_distance < 10.0:
                    self._log_event(
                        "SWARM",
                        f"Critical inter-drone spacing ({min_distance:.1f}m)",
                        "CRITICAL",
                    )
                    mission_result = "FAILED_SEPARATION"
                    break
                self._warn_beta_alpha_proximity()

                if elapsed >= self._mission_success_duration:
                    mission_result = "SUCCESS"
                    break

                # ── Update overlay (Isaac Sim) ──
                if self._overlay and tick % 10 == 0:
                    for did, drone in self._drones.items():
                        mgr = self._avoidance_managers.get(did)
                        if mgr:
                            self._overlay.update_drone_state(
                                self._drone_label(did), mgr.get_telemetry()
                            )

                # ── Periodic console status ──
                if tick % 300 == 0:  # Every ~10s
                    mgr = self._avoidance_managers.get(0)
                    state_name = mgr.state.name if mgr else "N/A"
                    closest = mgr.closest_obstacle_distance if mgr else float("inf")
                    logger.info(
                        f"[{elapsed:>6.1f}s] Decentralized swarm active | "
                        f"State: {state_name} | "
                        f"Closest: {closest:.1f}m | "
                        f"MinSep: {self._min_inter_drone_distance:.1f}m | "
                        f"HPL: {self._hpl_override_count}"
                    )

                await asyncio.sleep(self._dt)

        except KeyboardInterrupt:
            mission_result = "ABORTED"
            self._log_event("SYSTEM", "Mission aborted by user", "WARNING")

        # ── Finalize ──
        self._finalize(mission_result)

    async def _run_gui_waypoint_swarm(self) -> str:
        """
        GUI waypoint mode:
        - Alpha_0 follows panel-defined waypoints (leader path).
        - Alpha_1..5 retain boids/formation behavior with leader-offset bias.
        """
        if self._session is None:
            return "FAILED_GUI_SESSION"

        tick = 0
        mission_active = False
        mission_paused = False
        leader_wp_index = 0
        self._session.set_runner_state("idle", "Waiting for GUI start")

        while True:
            tick += 1
            elapsed = time.time() - self._start_time

            # Timeout is only enforced while a mission is actively running.
            if mission_active and elapsed > self._max_mission_time:
                self._log_event("SYSTEM", "GUI mission TIMEOUT", "CRITICAL")
                self._session.set_runner_state("timeout", "Mission timed out")
                return "TIMEOUT"

            # Session commands from the panel.
            command = self._session.consume_command()
            if command == "start":
                gui_wps = self._session.get_waypoints()
                if not gui_wps:
                    self._session.set_runner_state("idle", "No GUI waypoints available")
                else:
                    mission_active = True
                    mission_paused = False
                    leader_wp_index = 0
                    self._session.set_current_waypoint_index(0)
                    self._start_time = time.time()
                    self._session.set_runner_state("running", "Following GUI waypoints")
                    self._log_event("SYSTEM", f"GUI mission started with {len(gui_wps)} waypoints")
            elif command == "pause":
                mission_paused = True
                self._session.set_runner_state("paused", "Mission paused from GUI")
            elif command == "resume":
                if mission_active:
                    mission_paused = False
                    self._session.set_runner_state("running", "Mission resumed from GUI")
            elif command == "stop":
                self._session.set_runner_state("stopped", "Mission stopped from GUI")
                return "STOPPED_BY_GUI"

            toggles = self._session.get_toggles()
            self._apply_gui_toggles(toggles)
            if self._session.is_manual_override_enabled() and mission_active:
                mission_paused = True
                self._session.set_runner_state("manual_override", "Leader manual overtake active")

            # Feed LiDAR to all drones.
            for drone_id, drone in self._drones.items():
                if not drone.is_active:
                    continue
                scan_pos = Vector3(x=drone.position.x, y=drone.position.y, z=drone.position.z)
                points = self._lidar.scan(scan_pos)
                mgr = self._avoidance_managers.get(drone_id)
                if mgr is not None:
                    mgr.feed_lidar_points(points, drone_position=drone.position)

            # Decentralized state + gossip update.
            for drone_id, coordinator in self._coordinators.items():
                coordinator.update_member_state(drone_id, self._drones[drone_id].to_state())
                for beta_id in self._beta_ids:
                    beta_drone = self._drones.get(beta_id)
                    if beta_drone is not None and beta_drone.is_active:
                        coordinator.update_member_state(beta_id, beta_drone.to_state())

            gossip_payloads = {
                drone_id: coordinator.prepare_gossip_payload()
                for drone_id, coordinator in self._coordinators.items()
            }
            for receiver_id, coordinator in self._coordinators.items():
                for sender_id, payload in gossip_payloads.items():
                    if sender_id == receiver_id or not payload:
                        continue
                    coordinator.ingest_gossip_payload(payload)

            for coordinator in self._coordinators.values():
                coordinator.coordination_step()

            # Leader waypoint tracking
            leader_goal = None
            leader_override_velocity = Vector3()
            if mission_active and not mission_paused:
                gui_wps = self._session.get_waypoints()
                if not gui_wps:
                    self._session.set_runner_state("idle", "Waypoint list became empty")
                    mission_active = False
                elif leader_wp_index >= len(gui_wps):
                    self._session.set_runner_state("complete", "All GUI waypoints completed")
                    return "SUCCESS"
                else:
                    leader_goal = gui_wps[leader_wp_index].position
                    leader = self._drones[0]
                    leader_override_velocity, reached = self._compute_leader_waypoint_velocity(
                        leader.position,
                        gui_wps[leader_wp_index],
                    )
                    if reached:
                        self._log_event(
                            "LEADER",
                            f"Reached GUI_WP_{leader_wp_index + 1}",
                            "SUCCESS",
                        )
                        leader_wp_index += 1
                        self._session.set_current_waypoint_index(leader_wp_index)
                        if self._overlay is not None:
                            try:
                                self._overlay.advance_waypoint("Alpha_0", f"GUI_WP_{leader_wp_index}")
                            except Exception:
                                pass
                        if leader_wp_index >= len(gui_wps):
                            self._session.set_runner_state("complete", "All GUI waypoints completed")
                            return "SUCCESS"

            # Apply motion commands.
            for drone_id, drone in self._drones.items():
                if not drone.is_active:
                    continue

                coordinator = self._coordinators.get(drone_id)
                mgr = self._avoidance_managers.get(drone_id)
                desired_velocity = coordinator.get_desired_velocity(drone_id) if coordinator else Vector3()
                goal = coordinator.get_desired_goal(drone_id) if coordinator else None

                if mission_paused:
                    desired_velocity = Vector3()
                    goal = drone.position
                elif mission_active and not mission_paused:
                    if drone_id == 0:
                        desired_velocity = leader_override_velocity
                        goal = leader_goal
                    else:
                        desired_velocity, goal = self._blend_follower_with_leader_goal(
                            drone_id=drone_id,
                            boids_velocity=desired_velocity,
                            avoidance_mgr=mgr,
                        )

                if mgr is not None:
                    mgr.set_boids_velocity(desired_velocity)
                    if goal is not None:
                        mgr.set_goal(goal)
                    if toggles.avoidance_enabled:
                        velocity = mgr.compute_avoidance(
                            drone_position=drone.position,
                            drone_velocity=drone.velocity,
                        )
                    else:
                        velocity = desired_velocity
                else:
                    velocity = desired_velocity

                drone.step(velocity, self._dt)
                self._publish_velocity(drone_id, velocity)

            # Safety checks.
            for drone_id, drone in self._drones.items():
                if not drone.is_active:
                    continue
                mgr = self._avoidance_managers.get(drone_id)
                if mgr and mgr.closest_obstacle_distance < 0.5:
                    self._collision_count += 1
                    self._log_event(
                        self._drone_label(drone_id),
                        f"COLLISION — obstacle at {mgr.closest_obstacle_distance:.2f}m "
                        f"(total: {self._collision_count})",
                        "CRITICAL",
                    )
                    if self._collision_count >= 3:
                        self._session.set_runner_state("failed", "Collision threshold reached")
                        return "FAILED_COLLISION"

                if mgr and mgr.is_hpl_overriding:
                    self._hpl_override_count += 1

            min_distance = self._compute_min_inter_drone_distance()
            self._min_inter_drone_distance = min(self._min_inter_drone_distance, min_distance)
            if min_distance < 20.0:
                self._log_event(
                    "SWARM",
                    f"Inter-drone spacing low ({min_distance:.1f}m)",
                    "WARNING",
                )
            if min_distance < 10.0:
                self._log_event(
                    "SWARM",
                    f"Critical inter-drone spacing ({min_distance:.1f}m)",
                    "CRITICAL",
                )
                self._session.set_runner_state("failed", "Unsafe drone separation")
                return "FAILED_SEPARATION"
            self._warn_beta_alpha_proximity()

            if self._overlay and tick % 10 == 0:
                for did, _drone in self._drones.items():
                    mgr = self._avoidance_managers.get(did)
                    if mgr:
                        self._overlay.update_drone_state(self._drone_label(did), mgr.get_telemetry())

            if tick % 300 == 0:
                logger.info(
                    f"[{elapsed:>6.1f}s] GUI swarm | "
                    f"active={mission_active} paused={mission_paused} "
                    f"wp={leader_wp_index}/{len(self._session.get_waypoints())} "
                    f"MinSep={self._min_inter_drone_distance:.1f}m"
                )

            await asyncio.sleep(self._dt)

    def _compute_leader_waypoint_velocity(
        self,
        current_position: Vector3,
        waypoint: Waypoint,
    ) -> Tuple[Vector3, bool]:
        """Compute leader velocity toward active GUI waypoint."""
        delta = waypoint.position - current_position
        distance = delta.magnitude()
        if distance <= max(0.5, waypoint.acceptance_radius):
            return Vector3(), True
        max_speed = max(0.5, float(waypoint.speed))
        desired_speed = min(max_speed, distance)
        return delta.normalized() * desired_speed, False

    def _blend_follower_with_leader_goal(
        self,
        drone_id: int,
        boids_velocity: Vector3,
        avoidance_mgr: Optional[object] = None,
    ) -> Tuple[Vector3, Vector3]:
        """
        Adaptive formation/boids blending with explicit rejoin phase.
        """
        leader = self._drones[0]
        drone = self._drones[drone_id]
        center_offset = self._formation_offsets.get(0, Vector3())
        formation_center = Vector3(
            x=leader.position.x - center_offset.x,
            y=leader.position.y - center_offset.y,
            z=leader.position.z - center_offset.z,
        )
        offset = self._formation_offsets.get(drone_id, Vector3())
        target = formation_center + offset
        slot_delta = target - drone.position
        slot_error = slot_delta.magnitude()
        obstacle_dist = (
            float(avoidance_mgr.closest_obstacle_distance)
            if avoidance_mgr is not None
            else float("inf")
        )
        state = self._update_formation_state(drone_id, obstacle_dist, slot_error)
        follow_velocity = self._compute_seek_velocity(
            drone.position, target, p_gain=1.0, max_speed=5.0
        )
        if state == FormationState.NORMAL:
            return follow_velocity, target

        if state == FormationState.REJOINING:
            rejoin_velocity = self._compute_seek_velocity(
                drone.position, target, p_gain=REJOIN_P_GAIN, max_speed=REJOIN_SPEED
            )
            return rejoin_velocity, target

        if obstacle_dist >= OBSTACLE_ENGAGE_DIST:
            obstacle_factor = 0.0
        elif obstacle_dist <= OBSTACLE_FULL_DIST:
            obstacle_factor = 1.0
        else:
            obstacle_factor = 1.0 - (
                (obstacle_dist - OBSTACLE_FULL_DIST)
                / (OBSTACLE_ENGAGE_DIST - OBSTACLE_FULL_DIST)
            )

        blended = Vector3(
            x=(1.0 - obstacle_factor) * follow_velocity.x + obstacle_factor * boids_velocity.x,
            y=(1.0 - obstacle_factor) * follow_velocity.y + obstacle_factor * boids_velocity.y,
            z=(1.0 - obstacle_factor) * follow_velocity.z + obstacle_factor * boids_velocity.z,
        )
        return blended, target

    def _update_formation_state(
        self,
        drone_id: int,
        obstacle_dist: float,
        slot_error: float,
    ) -> FormationState:
        """Transition per-drone formation states using obstacle and slot error."""
        prev_state = self._formation_states.get(drone_id, FormationState.NORMAL)
        next_state = prev_state
        if prev_state == FormationState.NORMAL:
            if obstacle_dist < OBSTACLE_ENGAGE_DIST:
                next_state = FormationState.AVOIDING
        elif prev_state == FormationState.AVOIDING:
            if obstacle_dist > OBSTACLE_ENGAGE_DIST:
                next_state = FormationState.REJOINING
        elif prev_state == FormationState.REJOINING:
            if obstacle_dist < OBSTACLE_ENGAGE_DIST:
                next_state = FormationState.AVOIDING
            elif slot_error < REJOIN_TOLERANCE:
                next_state = FormationState.NORMAL
        self._formation_states[drone_id] = next_state
        if next_state != prev_state:
            self._log_event(
                self._drone_label(drone_id),
                f"{prev_state.name} -> {next_state.name}",
            )
        return next_state

    @staticmethod
    def _compute_seek_velocity(
        current: Vector3,
        target: Vector3,
        p_gain: float,
        max_speed: float,
    ) -> Vector3:
        """Compute bounded proportional seek velocity."""
        delta = target - current
        distance = delta.magnitude()
        if distance <= 1e-6:
            return Vector3()
        return delta.normalized() * min(max_speed, distance * p_gain)

    def _apply_gui_toggles(self, toggles) -> None:
        for coordinator in self._coordinators.values():
            coordinator.set_boids_enabled(toggles.boids_enabled)
            coordinator.set_cbba_enabled(toggles.cbba_enabled)
            coordinator.set_formation_enabled(toggles.formation_enabled)

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
                logger.info(f"  AvoidanceManager initialized for {self._drone_label(drone_id)}")

        except ImportError as e:
            logger.warning(f"Could not import AvoidanceManager: {e}")
            logger.warning("Running without obstacle avoidance (direct P-control)")

    async def _init_coordinators(self):
        """Initialize one decentralized regiment coordinator per drone."""
        for drone_id in self._alpha_ids:
            cfg = RegimentConfig(
                formation_spacing=FORMATION_SPACING,
                formation_altitude=ALPHA_ALTITUDE,
                total_coverage_area=1000.0,
                use_boids_flocking=True,
            )
            coordinator = AlphaRegimentCoordinator(my_drone_id=drone_id, config=cfg)
            await coordinator.initialize()
            for peer_id in self._alpha_ids:
                coordinator.register_drone(peer_id)
            self._coordinators[drone_id] = coordinator

    def _compute_min_inter_drone_distance(self) -> float:
        """Compute nearest pairwise spacing among active Alpha drones."""
        active = [
            d
            for d in self._drones.values()
            if d.is_active and d.drone_type == DroneType.ALPHA
        ]
        if len(active) < 2:
            return float("inf")

        min_dist = float("inf")
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dist = active[i].position.distance_to(active[j].position)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _warn_beta_alpha_proximity(self, threshold_m: float = 15.0) -> None:
        """Log non-fatal warning when Beta comes too close to an Alpha."""
        now = time.time()
        if now - self._last_beta_proximity_warning_ts < 1.0:
            return

        closest = float("inf")
        closest_pair: Optional[Tuple[int, int]] = None
        for beta_id in self._beta_ids:
            beta = self._drones.get(beta_id)
            if beta is None or not beta.is_active:
                continue
            for alpha_id in self._alpha_ids:
                alpha = self._drones.get(alpha_id)
                if alpha is None or not alpha.is_active:
                    continue
                dist = beta.position.distance_to(alpha.position)
                if dist < closest:
                    closest = dist
                    closest_pair = (beta_id, alpha_id)

        if closest_pair is not None and closest < threshold_m:
            beta_id, alpha_id = closest_pair
            self._last_beta_proximity_warning_ts = now
            self._log_event(
                self._drone_label(beta_id),
                f"Close to {self._drone_label(alpha_id)}: {closest:.1f}m",
                "WARNING",
            )

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

    def _publish_velocity(self, drone_id: int, velocity: Vector3):
        """Publish velocity command to Isaac Sim via ROS 2 bridge."""
        if self._bridge is None:
            return
        drone_name = self._drone_topic_name(drone_id)
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
                prim_path = self._drone_prim_path(drone_id)
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
            print(f"    {self._drone_label(did)}: ({drone.position.x:>7.1f}, {drone.position.y:>7.1f}, "
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
            print(f"    {self._drone_label(did)}:")
            print(f"      State          : {telem.get('avoidance_state', 'N/A')}")
            print(f"      HPL State      : {telem.get('hpl_state', 'N/A')}")
            print(f"      Closest Obs    : {telem.get('closest_obstacle_m', 'N/A')}m")
            print(f"      Sub-Waypoints  : {telem.get('active_sub_waypoints', 0)}")
            print(f"      LiDAR Points   : {telem.get('lidar', {}).get('filtered_points', 0)}")
            print(f"      Obstacles      : {telem.get('lidar', {}).get('obstacle_count', 0)}")

    def _drone_label(self, drone_id: int) -> str:
        """Human-readable drone label used for logs/overlay keys."""
        if drone_id in self._beta_ids:
            return f"Beta_{drone_id - BETA_DRONE_ID}"
        return f"Alpha_{drone_id}"

    def _drone_topic_name(self, drone_id: int) -> str:
        """ROS topic drone key for bridge velocity publishing."""
        if drone_id in self._beta_ids:
            return f"beta_{drone_id - BETA_DRONE_ID}"
        return f"alpha_{drone_id}"

    def _drone_prim_path(self, drone_id: int) -> str:
        """USD prim path for drone stage sync."""
        if drone_id in self._beta_ids:
            return f"/World/Drones/Beta_{drone_id - BETA_DRONE_ID}"
        return f"/World/Drones/Alpha_{drone_id}"

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
                self._drone_topic_name(did): {
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
    parser.add_argument(
        "--gui-waypoints",
        action="store_true",
        help="Use waypoint panel shared session (leader-followers in Isaac Sim)",
    )
    args = parser.parse_args()

    headless = not args.isaac
    runner = MissionRunner(
        headless=headless,
        use_waypoint_controller=args.controller_path,
        use_gui_waypoints=args.gui_waypoints,
    )
    runner._max_mission_time = args.timeout

    asyncio.run(runner.run())


def launch_gui_waypoint_swarm_runner(timeout: float = 600.0) -> MissionRunner:
    """
    Start GUI-waypoint swarm runner on the active Isaac Sim event loop.

    Intended for use from launch_waypoint_panel.py.
    """
    runner = MissionRunner(headless=False, use_gui_waypoints=True)
    runner._max_mission_time = timeout
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as exc:
        raise RuntimeError("Isaac event loop is required for GUI waypoint mode") from exc
    loop.create_task(runner.run())
    return runner


if __name__ == "__main__":
    main()
else:
    # When loaded via Isaac Sim Script Editor with an active event loop.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    auto_import_run = os.getenv("SANJAY_AUTOSTART_ON_IMPORT", "1").strip().lower() not in ("0", "false", "no")
    if loop is not None and auto_import_run:
        runner = MissionRunner(headless=False)
        loop.create_task(runner.run())
