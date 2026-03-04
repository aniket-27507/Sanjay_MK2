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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root on PATH ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.types.drone_types import DroneConfig, DroneState, DroneType, FlightMode, Vector3, Waypoint
from src.swarm.coordination import AlphaRegimentCoordinator, RegimentConfig

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

    return obstacles


# ═══════════════════════════════════════════════════════════════════
#  Synthetic LiDAR Generator
# ═══════════════════════════════════════════════════════════════════


class SyntheticLidar:
    """
    Generates synthetic 3D LiDAR returns from the obstacle database.

    Casts rays in a 360° × 30° pattern and checks intersection with
    axis-aligned bounding boxes.  This lets us test the avoidance
    stack without Isaac Sim's physics engine.
    """

    def __init__(
        self,
        obstacles: List[Dict],
        h_rays: int = 180,   # horizontal rays
        v_rays: int = 8,     # vertical channels
        max_range: float = 30.0,
    ):
        self._obstacles = obstacles
        self._h_rays = h_rays
        self._v_rays = v_rays
        self._max_range = max_range

        # Pre-compute ray directions
        h_angles = np.linspace(0, 2 * math.pi, h_rays, endpoint=False)
        v_angles = np.linspace(-15, 15, v_rays) * math.pi / 180

        self._directions = []
        for va in v_angles:
            for ha in h_angles:
                dx = math.cos(va) * math.cos(ha)
                dy = math.cos(va) * math.sin(ha)
                dz = math.sin(va)
                self._directions.append(np.array([dx, dy, dz]))

        self._directions = np.array(self._directions)

    def scan(self, position: Vector3) -> np.ndarray:
        """
        Generate a synthetic point cloud from the given drone position.

        Args:
            position: Drone position in world frame (NED).

        Returns:
            Nx3 array of hit points in body frame.
        """
        pos = np.array([position.x, position.y, position.z])
        hits = []

        for direction in self._directions:
            hit = self._cast_ray(pos, direction)
            if hit is not None:
                hits.append(hit - pos)  # Convert to body frame

        if hits:
            return np.array(hits, dtype=np.float32)
        return np.empty((0, 3), dtype=np.float32)

    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        """Cast a single ray and return the closest hit point."""
        closest_t = self._max_range
        closest_point = None

        for obs in self._obstacles:
            t = self._ray_aabb(
                origin, direction,
                obs["x"] - obs["w"] / 2, obs["y"] - obs["d"] / 2, obs["z"] - obs["h"] / 2,
                obs["x"] + obs["w"] / 2, obs["y"] + obs["d"] / 2, obs["z"] + obs["h"] / 2,
            )
            if t is not None and 0.3 < t < closest_t:
                closest_t = t
                closest_point = origin + direction * t

        return closest_point

    @staticmethod
    def _ray_aabb(
        origin: np.ndarray, direction: np.ndarray,
        x_min: float, y_min: float, z_min: float,
        x_max: float, y_max: float, z_max: float,
    ) -> Optional[float]:
        """Ray–AABB intersection test (slab method)."""
        inv_dir = np.where(np.abs(direction) > 1e-10, 1.0 / direction, 1e10)

        t1 = (x_min - origin[0]) * inv_dir[0]
        t2 = (x_max - origin[0]) * inv_dir[0]
        t3 = (y_min - origin[1]) * inv_dir[1]
        t4 = (y_max - origin[1]) * inv_dir[1]
        t5 = (z_min - origin[2]) * inv_dir[2]
        t6 = (z_max - origin[2]) * inv_dir[2]

        t_min = max(min(t1, t2), min(t3, t4), min(t5, t6))
        t_max = min(max(t1, t2), max(t3, t4), max(t5, t6))

        if t_max < 0 or t_min > t_max:
            return None
        return t_min if t_min > 0 else t_max


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


FORMATION_CENTER = (400, 350)
FORMATION_SPACING = 80.0
ALPHA_ALTITUDE = 65.0

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

    def __init__(self, headless: bool = True):
        self._headless = headless
        self._dt = 1.0 / 30.0  # 30 Hz control rate

        # Load obstacle database
        self._obstacles = _load_obstacle_database()
        logger.info(f"Loaded {len(self._obstacles)} obstacles from scene database")

        # Build synthetic LiDAR
        self._lidar = SyntheticLidar(self._obstacles)

        # Spawn drones
        self._drones: Dict[int, SimDrone] = {}
        hex_pos = _hex_positions(*FORMATION_CENTER, FORMATION_SPACING)
        for i, (x, y) in enumerate(hex_pos):
            self._drones[i] = SimDrone(
                drone_id=i,
                position=Vector3(x=x, y=y, z=-ALPHA_ALTITUDE),  # NED
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
        self._mission_success_duration = 120.0

        # Mission timing
        self._start_time = 0.0
        self._max_mission_time = 600.0  # 10 minutes

        # Telemetry log
        self._event_log: List[Dict] = []
        self._collision_count = 0
        self._hpl_override_count = 0
        self._min_inter_drone_distance = float("inf")

        # Mission overlay (Isaac Sim mode)
        self._overlay = None

    async def run(self):
        """Execute the complete mission."""
        logger.info("=" * 65)
        logger.info("  PROJECT SANJAY MK2 — Mission Runner")
        logger.info(f"  Mode: {'Headless' if self._headless else 'Isaac Sim'}")
        logger.info(f"  Drones: {len(self._drones)}")
        logger.info(f"  Obstacles: {len(self._obstacles)}")
        logger.info("  Mission: Full 6-drone decentralized autonomy")
        logger.info("=" * 65)

        # Initialize avoidance managers
        self._init_avoidance()
        await self._init_coordinators()

        # Connect to Isaac Sim overlay if available
        if not self._headless:
            self._init_overlay()

        self._start_time = time.time()
        self._log_event("SYSTEM", "Mission started")

        tick = 0
        mission_result = "UNKNOWN"

        try:
            while True:
                tick += 1
                elapsed = time.time() - self._start_time

                # ── Timeout check ──
                if elapsed > self._max_mission_time:
                    self._log_event("SYSTEM", "⏱️  Mission TIMEOUT", "CRITICAL")
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

                    if mgr and mgr.closest_obstacle_distance < 0.5:
                        self._collision_count += 1
                        self._log_event(
                            f"Alpha_{drone_id}",
                            f"💥 COLLISION — obstacle at {mgr.closest_obstacle_distance:.2f}m "
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
                        f"❌ Unsafe inter-drone spacing detected ({min_distance:.1f}m)",
                        "CRITICAL",
                    )
                    mission_result = "FAILED_SEPARATION"
                    break

                if elapsed >= self._mission_success_duration:
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
            self._coordinators[drone_id] = coordinator

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
        elapsed = time.time() - self._start_time

        print()
        print("=" * 65)
        if result == "SUCCESS":
            print("  🎯 MISSION COMPLETE")
        else:
            print(f"  ❌ MISSION RESULT: {result}")
        print("=" * 65)

        print(f"\n  Duration          : {elapsed:.1f}s")
        print(f"  Min Separation    : {self._min_inter_drone_distance:.1f}m")
        print(f"  HPL Overrides     : {self._hpl_override_count}")
        print(f"  Collisions        : {self._collision_count}")

        # Per-drone final state
        print(f"\n  ── Drone Final Positions ──")
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
        print(f"\n  ── DEBUG LOG ({len(self._event_log)} events) ──")
        for ev in self._event_log:
            ts = ev["elapsed"]
            lvl = ev["level"]
            src = ev["source"]
            msg = ev["message"]
            marker = "!!" if lvl == "CRITICAL" else ">>" if lvl == "WARNING" else "  "
            print(f"    {marker} [{ts:>7.1f}s] [{lvl:<8s}] [{src:<12s}] {msg}")

        # Avoidance telemetry snapshots
        print(f"\n  ── Avoidance Telemetry at Failure ──")
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
            "events": self._event_log,
        }

        try:
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
            print(f"\n  📄 Log saved: {log_path}")
        except Exception as e:
            print(f"\n  ⚠️  Log save failed: {e}")


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
    args = parser.parse_args()

    headless = not args.isaac
    runner = MissionRunner(headless=headless)
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
