"""
Project Sanjay Mk2 - Alpha Regiment Coordinator
================================================
Manages a 6-drone Alpha regiment for large-area surveillance
with integrated obstacle avoidance, sector-based coverage,
Collaborative SLAM (C-SLAM) data sharing, and FANET threat
broadcasting.

Architecture:
    ┌────────────────────────────────────────────────────┐
    │                Regiment Coordinator                │
    │  ┌──────────┐ ┌──────────┐       ┌──────────┐     │
    │  │ Alpha_0  │ │ Alpha_1  │  ...  │ Alpha_5  │     │
    │  │ (Leader) │ │          │       │          │     │
    │  └──────────┘ └──────────┘       └──────────┘     │
    │         │           │                  │           │
    │         └───────────┴──────────────────┘           │
    │               UDP Mesh Network                     │
    │          (Gossip Protocol + FANETs)                │
    └────────────────────────────────────────────────────┘

Each Alpha drone runs its own AvoidanceManager.  The Regiment
Coordinator handles:
    - Sector assignment (which drone covers which area)
    - Formation management (hexagonal spread at ≥50m separation)
    - C-SLAM map merging (shared obstacle maps)
    - Dynamic leader election
    - Load balancing for compute-heavy tasks
    - FANET threat relay (obstacle data from any drone shared instantly)

@author: Archishman Paul
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import (
    DroneConfig,
    DroneState,
    DroneType,
    FlightMode,
    Vector3,
    Waypoint,
)
from src.core.utils.geometry import hex_positions
from src.swarm.flock_coordinator import FlockCoordinator, FlockCoordinatorConfig
from src.swarm.formation import FormationConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Constants & Configuration
# ═══════════════════════════════════════════════════════════════════

REGIMENT_SIZE = 6
DEFAULT_ALPHA_ALTITUDE = 65.0   # meters
MIN_INTER_DRONE_DISTANCE = 50.0  # meters


class RegimentFormation(Enum):
    """Available formation patterns for the 6-drone regiment."""
    HEXAGONAL = auto()      # Standard surveillance formation
    LINEAR = auto()         # Sweep formation (line abreast)
    WEDGE = auto()          # Forward-heavy formation
    RING = auto()           # Circular perimeter patrol
    ADAPTIVE = auto()       # Auto-select based on mission
    # Urban operations (State Police deployment)
    BUILDING_PERIMETER = auto()  # Orbit around a building
    CROWD_EVENT = auto()         # Overhead coverage for crowd monitoring


class LeaderElectionCriteria(Enum):
    """Criteria for dynamic leader election."""
    COMPUTE_HEADROOM = auto()
    SENSOR_VISIBILITY = auto()
    BATTERY_LEVEL = auto()
    PROXIMITY_TO_OBJECTIVE = auto()


@dataclass
class SectorAssignment:
    """A surveillance sector assigned to a specific drone."""
    sector_id: str
    drone_id: int
    center: Vector3                 # Sector center position
    radius: float                   # Sector radius (meters)
    waypoints: List[Waypoint] = field(default_factory=list)
    coverage_percent: float = 0.0
    start_time: float = field(default_factory=time.time)


@dataclass
class TriangleSector(SectorAssignment):
    """Equilateral triangle sector per spec §3.4 (Method A).

    Each triangle has vertices at: hex_center, V_i (this drone's vertex),
    bounded by adjacent hex vertices V_{i-1} and V_{i+1}.
    """
    vertex: Vector3 = field(default_factory=Vector3)           # V_i position
    hex_center: Vector3 = field(default_factory=Vector3)       # Shared hex centre
    vertex_index: int = 0                                       # 0-5 clockwise from north
    left_neighbour_vertex: Vector3 = field(default_factory=Vector3)  # V_{(i-1) % 6}
    right_neighbour_vertex: Vector3 = field(default_factory=Vector3) # V_{(i+1) % 6}
    hex_radius: float = 80.0                                    # R — center to vertex


@dataclass
class RegimentConfig:
    """Configuration for the Alpha Regiment."""

    # ── Formation ──
    formation: RegimentFormation = RegimentFormation.HEXAGONAL
    formation_spacing: float = 80.0         # m — inter-drone distance
    formation_altitude: float = DEFAULT_ALPHA_ALTITUDE

    # ── Coverage ──
    total_coverage_area: float = 1000.0     # m — square area side length
    sector_overlap: float = 0.15            # 15% overlap between sectors

    # ── C-SLAM ──
    cslam_share_interval: float = 2.0       # Seconds between map shares
    cslam_merge_radius: float = 20.0        # Max distance for loop closure matching

    # ── FANET ──
    fanet_relay_hops: int = 3              # Max relay hops for threat data

    # ── Leader Election ──
    leader_election_interval: float = 10.0  # Seconds between elections
    leader_election_criteria: LeaderElectionCriteria = (
        LeaderElectionCriteria.COMPUTE_HEADROOM
    )

    # ── Load Balancing ──
    load_balance_interval: float = 5.0      # Seconds between load checks
    max_cpu_utilization: float = 0.80       # Trigger redistribution above this

    # ── Safety ──
    min_inter_drone_distance: float = MIN_INTER_DRONE_DISTANCE
    max_altitude_variance: float = 5.0      # m — altitude band

    # ── Decentralized Flocking ──
    use_boids_flocking: bool = True


@dataclass
class DroneRegimentMember:
    """State and metadata for a single regiment member."""
    drone_id: int
    state: DroneState = field(default_factory=DroneState)
    sector: Optional[SectorAssignment] = None
    is_leader: bool = False
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    obstacle_count: int = 0
    shared_obstacles: List[Dict] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    is_active: bool = True


# ═══════════════════════════════════════════════════════════════════
#  Regiment Coordinator
# ═══════════════════════════════════════════════════════════════════


class AlphaRegimentCoordinator:
    """
    Coordinates a 6-drone Alpha regiment for surveillance operations.

    This coordinator runs on each drone (decentralized) and also
    on the ground station (centralized fallback).  In normal operation,
    the elected leader drone runs the coordination logic and broadcasts
    assignments over the mesh network.

    Usage:
        coordinator = AlphaRegimentCoordinator(my_drone_id=0)
        await coordinator.initialize()

        # Register drones as they come online
        for i in range(6):
            coordinator.register_drone(i, drone_configs[i])

        # Start coordination loop
        await coordinator.start()

        # The coordinator will:
        # 1. Assign sectors to each drone
        # 2. Generate sweep waypoints per sector
        # 3. Share obstacle maps (C-SLAM)
        # 4. Relay threat data (FANET)
        # 5. Elect leaders dynamically
        # 6. Balance computational load
    """

    def __init__(
        self,
        my_drone_id: int = 0,
        config: Optional[RegimentConfig] = None,
    ):
        self.my_drone_id = my_drone_id
        self.config = config or RegimentConfig()

        # ── Regiment Members ──
        self._members: Dict[int, DroneRegimentMember] = {}

        # ── Leadership ──
        self._current_leader_id: int = 0
        self._last_election_time: float = 0.0

        # ── Shared Obstacle Map (C-SLAM) ──
        self._global_obstacle_map: Deque[Dict] = deque(maxlen=200)
        self._last_cslam_share: float = 0.0

        # ── FANET Threat Relay ──
        self._threat_queue: List[Dict] = []
        self._relayed_threat_ids: set = set()
        self._relayed_threat_order: Deque[str] = deque(maxlen=500)  # eviction order for bounded set

        # ── State ──
        self._running = False
        self._initialized = False
        self._shared_state_lock = asyncio.Lock()

        # ── Callbacks ──
        self._on_sector_assigned: Optional[Callable] = None
        self._on_threat_received: Optional[Callable] = None
        self._network_send: Optional[Callable] = None

        # ── Decentralized Flocking State ──
        self._flock_coordinator: Optional[FlockCoordinator] = None
        self._desired_velocities: Dict[int, Vector3] = {}
        self._desired_goals: Dict[int, Vector3] = {}
        self._last_gossip_timestamps: Dict[int, float] = {}
        self._forced_goal: Optional[Vector3] = None

        # ── Gossip Crypto (anti-hostile-takeover — stub until keys available) ──
        self._gossip_crypto: Optional[Any] = None  # GossipCryptoEngine instance

    # ── Initialization ────────────────────────────────────────────

    async def initialize(self):
        """Initialize the regiment coordinator."""
        self._initialized = True
        self.register_drone(self.my_drone_id)
        if self.config.use_boids_flocking and self._flock_coordinator is None:
            flock_cfg = FlockCoordinatorConfig(
                formation=FormationConfig(
                    spacing=self.config.formation_spacing,
                    altitude=self.config.formation_altitude,
                    min_separation=self.config.min_inter_drone_distance,
                )
            )
            self._flock_coordinator = FlockCoordinator(
                drone_id=self.my_drone_id,
                config=flock_cfg,
                num_drones=REGIMENT_SIZE,
            )
        logger.info(
            f"RegimentCoordinator initialized (drone {self.my_drone_id})"
        )

    def register_drone(
        self,
        drone_id: int,
        config: Optional[DroneConfig] = None,
    ):
        """Register a drone into the regiment."""
        if drone_id in self._members:
            return

        member = DroneRegimentMember(
            drone_id=drone_id,
            state=DroneState(
                drone_id=drone_id,
                drone_type=DroneType.ALPHA,
            ),
        )
        self._members[drone_id] = member
        logger.info(
            f"Registered Alpha_{drone_id} "
            f"(regiment size: {len(self._members)}/{REGIMENT_SIZE})"
        )

        if self.config.use_boids_flocking and self._flock_coordinator is not None:
            self._flock_coordinator.update_membership(self._members.keys())

        # Auto-assign sectors when regiment is full
        if len(self._members) == REGIMENT_SIZE:
            self._assign_sectors()

    def unregister_drone(self, drone_id: int):
        """Remove a drone from the regiment.

        When the failed drone held a ``TriangleSector``, its orphaned
        waypoints are distributed to the two adjacent Alpha drones
        (vertex_index ± 1 mod 6) instead of triggering a full
        sector reassignment.
        """
        if drone_id in self._members:
            failed_member = self._members[drone_id]
            failed_sector = failed_member.sector

            del self._members[drone_id]
            self._desired_velocities.pop(drone_id, None)
            self._desired_goals.pop(drone_id, None)
            logger.warning(f"Alpha_{drone_id} removed from regiment")

            if self.config.use_boids_flocking and self._flock_coordinator is not None:
                self._flock_coordinator.update_membership(self._members.keys())

            # Triangle sector failure recovery
            if isinstance(failed_sector, TriangleSector) and self._members:
                vidx = failed_sector.vertex_index
                orphaned_wps = list(failed_sector.waypoints)
                left_idx = (vidx - 1) % 6
                right_idx = (vidx + 1) % 6

                # Find adjacent Alphas by their vertex_index
                left_id = right_id = None
                for m in self._members.values():
                    if isinstance(m.sector, TriangleSector):
                        if m.sector.vertex_index == left_idx:
                            left_id = m.drone_id
                        elif m.sector.vertex_index == right_idx:
                            right_id = m.drone_id

                # Split orphaned waypoints between the two neighbours
                mid = len(orphaned_wps) // 2
                if left_id is not None:
                    self._expand_sector_for_adjacent(
                        left_id, orphaned_wps[:mid]
                    )
                if right_id is not None:
                    self._expand_sector_for_adjacent(
                        right_id, orphaned_wps[mid:]
                    )

                logger.warning(
                    f"Triangle failure recovery: V{vidx} orphaned waypoints "
                    f"distributed to neighbours V{left_idx} / V{right_idx}"
                )
                return

            # Fallback: redistribute all sectors
            if self._members:
                self._assign_sectors()

    def _expand_sector_for_adjacent(
        self, alpha_id: int, orphaned_waypoints: List[Waypoint]
    ):
        """Expand an adjacent Alpha's sector to absorb orphaned waypoints."""
        member = self._members.get(alpha_id)
        if member and member.sector:
            member.sector.waypoints.extend(orphaned_waypoints)
            member.sector.coverage_percent = 0.0  # reset — needs re-sweep
            logger.warning(
                f"Alpha_{alpha_id} sector expanded to cover orphaned area"
            )

    # ── Main Loop ─────────────────────────────────────────────────

    async def start(self):
        """Start the coordination loop."""
        self._running = True
        logger.info("Regiment coordination started")

        # Launch background tasks
        asyncio.create_task(self._coordination_loop())
        asyncio.create_task(self._cslam_loop())
        asyncio.create_task(self._leader_election_loop())
        asyncio.create_task(self._load_balance_loop())

    async def stop(self):
        """Stop the coordination loop."""
        self._running = False

    async def _coordination_loop(self):
        """Main coordination tick at 2 Hz."""
        while self._running:
            async with self._shared_state_lock:
                self.coordination_step()
            await asyncio.sleep(0.5)

    def coordination_step(self):
        """Run one synchronous coordination step (used by scripts and loop)."""
        try:
            self._check_member_health()
            if self.config.use_boids_flocking:
                self._run_flocking_step()
            self._update_coverage()
            self._process_threat_queue()
        except Exception as e:
            logger.error(f"Coordination error: {e}")

    # ── Sector Assignment ─────────────────────────────────────────

    def _assign_sectors(self):
        """
        Assign surveillance sectors to regiment members.

        Divides the total coverage area into sectors based on
        the current formation pattern.
        """
        active_members = [m for m in self._members.values() if m.is_active]
        n = len(active_members)

        if n == 0:
            return

        area = self.config.total_coverage_area
        formation = self.config.formation

        # Triangle sector model for full hexagonal regiment
        if formation == RegimentFormation.HEXAGONAL and n == REGIMENT_SIZE:
            self._assign_triangle_sectors(active_members)
            return

        if formation == RegimentFormation.HEXAGONAL:
            positions = self._hexagonal_positions(n, area)
        elif formation == RegimentFormation.LINEAR:
            positions = self._linear_positions(n, area)
        elif formation == RegimentFormation.RING:
            positions = self._ring_positions(n, area)
        elif formation == RegimentFormation.WEDGE:
            positions = self._wedge_positions(n, area)
        else:
            positions = self._hexagonal_positions(n, area)

        sector_radius = (area / n) ** 0.5 * (1 + self.config.sector_overlap)

        for member, (cx, cy) in zip(active_members, positions):
            sector = SectorAssignment(
                sector_id=f"sector_{member.drone_id}",
                drone_id=member.drone_id,
                center=Vector3(x=cx, y=cy, z=-self.config.formation_altitude),
                radius=sector_radius,
                waypoints=self._generate_sweep_waypoints(cx, cy, sector_radius),
            )
            member.sector = sector
            logger.info(
                f"Alpha_{member.drone_id} → Sector at ({cx:.0f}, {cy:.0f}) "
                f"radius={sector_radius:.0f}m"
            )

        if self._on_sector_assigned:
            self._on_sector_assigned()

    def _generate_sweep_waypoints(
        self,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> List[Waypoint]:
        """Generate a lawnmower sweep pattern for a sector."""
        waypoints = []
        spacing = radius * 0.4  # Sweep line spacing
        altitude = -self.config.formation_altitude  # NED

        lines = int(2 * radius / spacing)
        for i in range(lines):
            y = center_y - radius + i * spacing

            if i % 2 == 0:
                x_start = center_x - radius
                x_end = center_x + radius
            else:
                x_start = center_x + radius
                x_end = center_x - radius

            waypoints.append(Waypoint(
                position=Vector3(x=x_start, y=y, z=altitude),
                speed=5.0,
                acceptance_radius=3.0,
                hold_time=1.0,
            ))
            waypoints.append(Waypoint(
                position=Vector3(x=x_end, y=y, z=altitude),
                speed=5.0,
                acceptance_radius=3.0,
            ))

        return waypoints

    def _assign_triangle_sectors(
        self, active_members: List[DroneRegimentMember]
    ):
        """Assign equilateral triangle sectors using hex vertex positions.

        Uses ``hex_positions`` from the geometry module to obtain the 6
        vertices V_0..V_5 (no center), then greedily matches each active
        member to the closest available vertex (mimics CBBA sector auction).
        """
        spacing = self.config.formation_spacing

        # Use override center if set (from reassign_sectors_for_radius)
        if hasattr(self, '_sector_center_override') and self._sector_center_override is not None:
            center = self._sector_center_override
            cx, cy = center.x, center.y
        else:
            area = self.config.total_coverage_area
            cx = area / 2.0
            cy = area / 2.0
            center = Vector3(x=cx, y=cy, z=-self.config.formation_altitude)

        vertices_xy = hex_positions(
            cx, cy, spacing, n=6, include_center=False
        )  # List[(x, y)]

        # Build Vector3 list for the 6 hex vertices
        vertices = [
            Vector3(x=vx, y=vy, z=-self.config.formation_altitude)
            for vx, vy in vertices_xy
        ]

        # Greedy closest-vertex assignment
        available_indices = list(range(6))
        assignments: List[Tuple[DroneRegimentMember, int]] = []

        for member in active_members:
            best_idx = min(
                available_indices,
                key=lambda idx: (
                    (member.state.position.x - vertices[idx].x) ** 2
                    + (member.state.position.y - vertices[idx].y) ** 2
                ),
            )
            assignments.append((member, best_idx))
            available_indices.remove(best_idx)

        for member, vidx in assignments:
            v = vertices[vidx]
            left_v = vertices[(vidx - 1) % 6]
            right_v = vertices[(vidx + 1) % 6]

            waypoints = self._generate_radial_lawnmower(
                vertex=v,
                hex_center=center,
                left_v=left_v,
                right_v=right_v,
                hex_radius=spacing,
            )

            sector = TriangleSector(
                sector_id=f"tri_sector_{member.drone_id}",
                drone_id=member.drone_id,
                center=center,
                radius=spacing,
                waypoints=waypoints,
                vertex=v,
                hex_center=center,
                vertex_index=vidx,
                left_neighbour_vertex=left_v,
                right_neighbour_vertex=right_v,
                hex_radius=spacing,
            )
            member.sector = sector
            logger.info(
                f"Alpha_{member.drone_id} → TriangleSector V{vidx} "
                f"at ({v.x:.0f}, {v.y:.0f}), "
                f"{len(waypoints)} radial waypoints"
            )

        if self._on_sector_assigned:
            self._on_sector_assigned()

    def reassign_sectors_for_radius(
        self, hex_center: Vector3, hex_radius: float,
    ):
        """Re-assign triangle sectors scaled to a new hex radius.

        Called by the swarm runner when entering HOLD_FOR_CLIMB at each
        checkpoint to scale the survey area to the checkpoint's survey_radius.
        """
        # Temporarily override spacing for sector assignment
        original_spacing = self.config.formation_spacing
        self.config.formation_spacing = hex_radius

        active = [m for m in self._members.values() if m.state is not None]
        if active:
            # Use hex_center directly (not derived from total_coverage_area)
            self._sector_center_override = hex_center
            self._assign_triangle_sectors(active)
            self._sector_center_override = None

        # Restore original config
        self.config.formation_spacing = original_spacing

    def _generate_radial_lawnmower(
        self,
        vertex: Vector3,
        hex_center: Vector3,
        left_v: Vector3,
        right_v: Vector3,
        hex_radius: float,
        sensor_footprint: float = 40.0,  # Livox Mid-360 at 65m
    ) -> List[Waypoint]:
        """Generate a boustrophedon sweep inside a triangle sector.

        Strips run perpendicular to the vertex → center radial.  The
        triangle half-width at each depth *d* is derived from the 60°
        central angle of the hexagonal sector.
        """
        altitude = -self.config.formation_altitude  # NED

        # Radial direction: vertex → center
        dx = hex_center.x - vertex.x
        dy = hex_center.y - vertex.y
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            return []
        radial_x = dx / mag
        radial_y = dy / mag

        # Perpendicular direction: 90° clockwise rotation of radial
        perp_x = radial_y
        perp_y = -radial_x

        sin60 = math.sqrt(3.0) / 2.0
        n_strips = math.ceil(hex_radius / sensor_footprint)

        waypoints: List[Waypoint] = []

        for i in range(n_strips):
            d = (i + 0.5) * sensor_footprint  # depth along radial from vertex
            if d > hex_radius:
                d = hex_radius

            half_w = (d / hex_radius) * hex_radius * sin60

            lx = vertex.x + radial_x * d - perp_x * half_w
            ly = vertex.y + radial_y * d - perp_y * half_w
            rx = vertex.x + radial_x * d + perp_x * half_w
            ry = vertex.y + radial_y * d + perp_y * half_w

            if i % 2 == 0:
                first_x, first_y = lx, ly
                second_x, second_y = rx, ry
            else:
                first_x, first_y = rx, ry
                second_x, second_y = lx, ly

            waypoints.append(Waypoint(
                position=Vector3(x=first_x, y=first_y, z=altitude),
                speed=5.0,
                acceptance_radius=3.0,
            ))
            waypoints.append(Waypoint(
                position=Vector3(x=second_x, y=second_y, z=altitude),
                speed=5.0,
                acceptance_radius=3.0,
            ))

        # Phase 4: Reset — return to vertex
        waypoints.append(Waypoint(
            position=Vector3(x=vertex.x, y=vertex.y, z=altitude),
            speed=5.0,
            acceptance_radius=3.0,
        ))

        return waypoints

    # ── Formation Patterns ────────────────────────────────────────

    def _hexagonal_positions(
        self, n: int, area: float
    ) -> List[Tuple[float, float]]:
        """Compute hexagonal grid positions."""
        positions = []
        spacing = self.config.formation_spacing
        center = area / 2

        # Hexagonal ring layout
        positions.append((center, center))  # Center drone

        if n > 1:
            for i in range(min(n - 1, 6)):
                angle = i * (2 * np.pi / 6)
                x = center + spacing * np.cos(angle)
                y = center + spacing * np.sin(angle)
                positions.append((x, y))

        return positions[:n]

    def _linear_positions(
        self, n: int, area: float
    ) -> List[Tuple[float, float]]:
        """Compute line-abreast positions."""
        spacing = self.config.formation_spacing
        center_y = area / 2
        total_width = (n - 1) * spacing
        start_x = area / 2 - total_width / 2

        return [
            (start_x + i * spacing, center_y)
            for i in range(n)
        ]

    def _ring_positions(
        self, n: int, area: float
    ) -> List[Tuple[float, float]]:
        """Compute circular perimeter positions."""
        center = area / 2
        radius = min(area / 3, self.config.formation_spacing * n / (2 * np.pi))

        return [
            (center + radius * np.cos(i * 2 * np.pi / n),
             center + radius * np.sin(i * 2 * np.pi / n))
            for i in range(n)
        ]

    def _wedge_positions(
        self, n: int, area: float
    ) -> List[Tuple[float, float]]:
        """Compute V-wedge formation positions."""
        spacing = self.config.formation_spacing
        center = area / 2
        positions = [(center, center)]  # Lead drone

        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            x = center - row * spacing * 0.7
            y = center + side * row * spacing * 0.5
            positions.append((x, y))

        return positions[:n]

    # ── C-SLAM (Collaborative SLAM) ───────────────────────────────

    async def _cslam_loop(self):
        """Periodically share local obstacle map with the regiment."""
        while self._running:
            try:
                now = time.time()
                if now - self._last_cslam_share >= self.config.cslam_share_interval:
                    async with self._shared_state_lock:
                        await self._share_local_map()
                    self._last_cslam_share = now
            except Exception as e:
                logger.error(f"C-SLAM error: {e}")
            await asyncio.sleep(1.0)

    async def _share_local_map(self):
        """Share local obstacle observations with the swarm."""
        my_member = self._members.get(self.my_drone_id)
        if my_member is None:
            return

        map_data = {
            "type": "cslam_update",
            "drone_id": self.my_drone_id,
            "obstacles": my_member.shared_obstacles[:20],  # Cap for bandwidth
            "position": [
                my_member.state.position.x,
                my_member.state.position.y,
                my_member.state.position.z,
            ],
            "timestamp": time.time(),
        }

        if self._network_send:
            try:
                self._network_send(map_data)
            except Exception as e:
                logger.error(f"C-SLAM share failed: {e}")

    def receive_cslam_update(self, data: Dict):
        """
        Ingest C-SLAM update from another drone.

        Merges remote obstacle observations into the global map.
        Uses distance-based matching to avoid duplicates.
        """
        remote_id = data.get("drone_id", -1)
        remote_obstacles = data.get("obstacles", [])

        for obs in remote_obstacles:
            pos = obs.get("position", [0, 0, 0])
            is_duplicate = False

            # Check for duplicates within merge radius
            for existing in self._global_obstacle_map:
                ex_pos = existing.get("position", [0, 0, 0])
                dist = np.linalg.norm(
                    np.array(pos) - np.array(ex_pos)
                )
                if dist < self.config.cslam_merge_radius:
                    # Merge: boost confidence
                    existing["confidence"] = min(
                        existing.get("confidence", 0.5) + 0.1,
                        1.0,
                    )
                    existing["sources"] = list(
                        set(existing.get("sources", [])) | {remote_id}
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                obs["sources"] = [remote_id]
                self._global_obstacle_map.append(obs)

        # deque(maxlen=200) auto-prunes oldest entries

    # ── FANET Threat Relay ────────────────────────────────────────

    def broadcast_threat(self, threat_data: Dict):
        """
        Broadcast a threat to the regiment via FANET relay.

        Each drone that receives the threat will re-broadcast it
        up to `fanet_relay_hops` times, ensuring full coverage.
        """
        threat_id = str(uuid.uuid4())[:8]
        threat_data["threat_id"] = threat_id
        threat_data["hops_remaining"] = self.config.fanet_relay_hops
        threat_data["origin_drone"] = self.my_drone_id

        self._threat_queue.append(threat_data)
        self._track_relayed_threat(threat_id)

    def receive_threat_relay(self, threat_data: Dict):
        """Receive a relayed threat from the FANET mesh."""
        threat_id = threat_data.get("threat_id", "")

        # Deduplicate
        if threat_id in self._relayed_threat_ids:
            return

        self._track_relayed_threat(threat_id)

        # Process the threat
        if self._on_threat_received:
            self._on_threat_received(threat_data)

        # Re-relay if hops remaining
        hops = threat_data.get("hops_remaining", 0)
        if hops > 0:
            threat_data["hops_remaining"] = hops - 1
            self._threat_queue.append(threat_data)

    def _track_relayed_threat(self, threat_id: str):
        """Add threat_id to bounded dedup set, evicting oldest if full."""
        if len(self._relayed_threat_order) >= self._relayed_threat_order.maxlen:
            evicted = self._relayed_threat_order[0]  # will be auto-evicted by deque
            self._relayed_threat_ids.discard(evicted)
        self._relayed_threat_ids.add(threat_id)
        self._relayed_threat_order.append(threat_id)

    def _process_threat_queue(self):
        """Process and send queued threat broadcasts."""
        if not self._network_send:
            self._threat_queue.clear()
            return

        while self._threat_queue:
            threat = self._threat_queue.pop(0)
            try:
                self._network_send(threat)
            except Exception as e:
                logger.error(f"Threat relay failed: {e}")

    def _run_flocking_step(self):
        """Compute decentralized boids velocity for this coordinator's drone."""
        if self._flock_coordinator is None:
            flock_cfg = FlockCoordinatorConfig(
                formation=FormationConfig(
                    spacing=self.config.formation_spacing,
                    altitude=self.config.formation_altitude,
                    min_separation=self.config.min_inter_drone_distance,
                )
            )
            self._flock_coordinator = FlockCoordinator(
                drone_id=self.my_drone_id,
                config=flock_cfg,
                num_drones=REGIMENT_SIZE,
            )

        my_member = self._members.get(self.my_drone_id)
        if my_member is None or not my_member.is_active:
            return

        active_ids = [m.drone_id for m in self._members.values() if m.is_active]
        self._flock_coordinator.update_membership(active_ids)

        peer_states = {
            m.drone_id: m.state
            for m in self._members.values()
            if m.drone_id != self.my_drone_id and m.is_active
        }
        sectors = [m.sector for m in self._members.values() if m.sector is not None]
        # Use forced goal as home position (aligns boids cohesion with
        # formation target).  Fallback to area center if no goal set.
        if self._forced_goal is not None:
            home = self._forced_goal
        else:
            home = Vector3(
                x=self.config.total_coverage_area / 2.0,
                y=self.config.total_coverage_area / 2.0,
                z=-self.config.formation_altitude,
            )

        desired = self._flock_coordinator.tick(
            my_state=my_member.state,
            peer_states=peer_states,
            obstacles=self._global_obstacle_map,
            sector_assignments=sectors,
            home_position=home,
        )
        if self._forced_goal is not None:
            direction = self._forced_goal - my_member.state.position
            if direction.magnitude() > 1e-6:
                distance_to_goal = direction.magnitude()
                max_speed = max(0.1, float(self._flock_coordinator.config.boids.max_speed))
                forced_velocity = direction.normalized() * min(direction.magnitude(), max_speed)
                # Adaptive blending: stronger pull on long legs, gentler near goals.
                if distance_to_goal > 180.0:
                    forced_weight = 0.85
                elif distance_to_goal > 100.0:
                    forced_weight = 0.75
                elif distance_to_goal > 50.0:
                    forced_weight = 0.60
                else:
                    forced_weight = 0.45
                desired = desired * (1.0 - forced_weight) + forced_velocity * forced_weight
        self._desired_velocities[self.my_drone_id] = desired
        if self._forced_goal is not None:
            self._desired_goals[self.my_drone_id] = self._forced_goal
        elif self._flock_coordinator.current_goal is not None:
            self._desired_goals[self.my_drone_id] = self._flock_coordinator.current_goal

    # ── Dynamic Leader Election ───────────────────────────────────

    async def _leader_election_loop(self):
        """Periodically elect a leader for complex coordination tasks."""
        while self._running:
            try:
                now = time.time()
                if now - self._last_election_time >= self.config.leader_election_interval:
                    self._elect_leader()
                    self._last_election_time = now
            except Exception as e:
                logger.error(f"Leader election error: {e}")
            await asyncio.sleep(2.0)

    def _elect_leader(self):
        """
        Elect a leader based on configured criteria.

        The leader handles complex coordination tasks like
        A* pathfinding for the swarm through dense obstacle fields.
        """
        active = [m for m in self._members.values() if m.is_active]
        if not active:
            return

        criteria = self.config.leader_election_criteria

        if criteria == LeaderElectionCriteria.COMPUTE_HEADROOM:
            best = min(active, key=lambda m: m.cpu_utilization)
        elif criteria == LeaderElectionCriteria.BATTERY_LEVEL:
            best = max(active, key=lambda m: m.state.battery)
        elif criteria == LeaderElectionCriteria.SENSOR_VISIBILITY:
            best = max(active, key=lambda m: m.obstacle_count)
        else:
            best = active[0]

        # Update leadership
        for m in self._members.values():
            m.is_leader = False
        best.is_leader = True
        self._current_leader_id = best.drone_id

        if best.drone_id == self.my_drone_id:
            logger.info(f"Elected as regiment leader (drone {best.drone_id})")

    @property
    def current_leader_id(self) -> int:
        return self._current_leader_id

    @property
    def am_i_leader(self) -> bool:
        return self._current_leader_id == self.my_drone_id

    # ── Load Balancing ────────────────────────────────────────────

    async def _load_balance_loop(self):
        """Monitor and balance computational load across drones."""
        while self._running:
            try:
                overloaded = [
                    m for m in self._members.values()
                    if m.cpu_utilization > self.config.max_cpu_utilization
                       and m.is_active
                ]
                available = [
                    m for m in self._members.values()
                    if m.cpu_utilization < self.config.max_cpu_utilization * 0.6
                       and m.is_active
                ]

                for overloaded_drone in overloaded:
                    if available:
                        helper = min(available, key=lambda m: m.cpu_utilization)
                        logger.info(
                            f"Load balancing: offloading from "
                            f"Alpha_{overloaded_drone.drone_id} "
                            f"({overloaded_drone.cpu_utilization:.0%}) "
                            f"→ Alpha_{helper.drone_id} "
                            f"({helper.cpu_utilization:.0%})"
                        )
                        # In production, this would trigger task migration

            except Exception as e:
                logger.error(f"Load balance error: {e}")
            await asyncio.sleep(self.config.load_balance_interval)

    # ── Health Monitoring ─────────────────────────────────────────

    def _check_member_health(self):
        """Check heartbeats and mark inactive drones."""
        now = time.time()
        for member in self._members.values():
            if now - member.last_heartbeat > 5.0:
                if member.is_active:
                    logger.warning(
                        f"Alpha_{member.drone_id} heartbeat lost — marking inactive"
                    )
                    member.is_active = False
                    # Reassign sectors
                    self._assign_sectors()

    def update_member_state(self, drone_id: int, state: DroneState):
        """Update a member's state (called from heartbeat/gossip)."""
        if drone_id in self._members:
            self._members[drone_id].state = state
            self._members[drone_id].last_heartbeat = time.time()
            self._members[drone_id].is_active = True

    def update_member_load(
        self, drone_id: int, cpu: float, gpu: float
    ):
        """Update a member's compute utilization."""
        if drone_id in self._members:
            self._members[drone_id].cpu_utilization = cpu
            self._members[drone_id].gpu_utilization = gpu

    def update_member_obstacles(
        self, drone_id: int, obstacles: List[Dict]
    ):
        """Update shared obstacles from a member."""
        if drone_id in self._members:
            self._members[drone_id].shared_obstacles = obstacles
            self._members[drone_id].obstacle_count = len(obstacles)

    # ── Coverage Tracking ─────────────────────────────────────────

    def _update_coverage(self):
        """Track per-sector coverage progress."""
        for member in self._members.values():
            if member.sector and member.state.mode == FlightMode.NAVIGATING:
                total_wps = len(member.sector.waypoints)
                if total_wps > 0:
                    # Estimate coverage based on position proximity
                    completed = sum(
                        1 for wp in member.sector.waypoints
                        if member.state.position.distance_to(wp.position) < wp.acceptance_radius
                    )
                    member.sector.coverage_percent = completed / total_wps * 100

    # ── Callbacks & Network ───────────────────────────────────────

    def set_network_send(self, callback: Callable):
        """Set the callback for sending data over the mesh network."""
        self._network_send = callback

    def on_sector_assigned(self, callback: Callable):
        """Register callback for sector assignment events."""
        self._on_sector_assigned = callback

    def on_threat_received(self, callback: Callable):
        """Register callback for incoming threat data."""
        self._on_threat_received = callback

    def enable_gossip_crypto(self, crypto_engine: Any):
        """Attach a GossipCryptoEngine for authenticated gossip.

        Pass an instance of ``src.communication.gossip_crypto.GossipCryptoEngine``.
        When set, all outgoing gossip is wrapped in an authenticated envelope
        and all incoming gossip is verified before processing.
        """
        self._gossip_crypto = crypto_engine
        logger.info(
            "Gossip crypto enabled for drone %d (algo=%s)",
            self.my_drone_id,
            getattr(getattr(crypto_engine, 'config', None), 'algorithm', 'unknown'),
        )

    # ── Gossip Payloads ──────────────────────────────────────────

    def _get_nearest_neighbour_ids(self, count: int = 2) -> List[int]:
        """Return the `count` nearest active neighbour drone IDs (spec §4.4)."""
        my_member = self._members.get(self.my_drone_id)
        if my_member is None:
            return []
        my_pos = my_member.state.position
        candidates = [
            (m.drone_id, my_pos.distance_to(m.state.position))
            for m in self._members.values()
            if m.drone_id != self.my_drone_id and m.is_active
        ]
        candidates.sort(key=lambda pair: pair[1])
        return [cid for cid, _ in candidates[:count]]

    def prepare_gossip_payload(self) -> Dict:
        """Build a compact state + CBBA gossip payload.

        Spec §4.4: each Alpha gossips to its 2 nearest neighbours at 10 Hz.
        The payload includes patrol_progress for the state vector.
        The optional crypto envelope wraps the payload if the engine is set.
        """
        member = self._members.get(self.my_drone_id)
        if member is None:
            return {}

        # Inject patrol_progress into state (spec §4.4 state vector)
        sector = self.get_my_sector()
        if sector:
            member.state.patrol_progress = sector.coverage_percent

        cbba_payload = {}
        if self.config.use_boids_flocking and self._flock_coordinator is not None:
            cbba_payload = self._flock_coordinator.prepare_gossip_payload(member.state)

        target_ids = self._get_nearest_neighbour_ids(count=2)

        inner = {
            "type": "swarm_gossip_v1",
            "drone_id": self.my_drone_id,
            "drone_state": member.state.to_dict(),
            "cbba": cbba_payload,
            "target_ids": target_ids,
            "patrol_progress": member.state.patrol_progress,
            "timestamp": time.time(),
        }

        # Wrap with crypto envelope if engine is available
        if self._gossip_crypto is not None:
            return self._gossip_crypto.wrap_payload(inner)
        return inner

    def ingest_gossip_payload(self, payload: Dict):
        """Ingest state + CBBA gossip from a peer drone.

        Nearest-neighbour filtering: drops payloads not addressed to us
        (unless target_ids is absent for backwards compatibility).
        Crypto: unwraps authenticated envelope if engine is set.
        """
        # Unwrap crypto envelope if engine is present
        if self._gossip_crypto is not None:
            inner, is_valid = self._gossip_crypto.unwrap_payload(payload)
            if not is_valid or inner is None:
                return
            payload = inner

        if payload.get("type") != "swarm_gossip_v1":
            return

        sender_id = int(payload.get("drone_id", -1))
        if sender_id < 0 or sender_id == self.my_drone_id:
            return

        # Nearest-neighbour filter (spec §4.4): drop if not addressed to us
        target_ids = payload.get("target_ids")
        if target_ids is not None and self.my_drone_id not in target_ids:
            return

        ts = float(payload.get("timestamp", 0.0))
        if ts <= self._last_gossip_timestamps.get(sender_id, 0.0):
            return
        self._last_gossip_timestamps[sender_id] = ts

        state_payload = payload.get("drone_state")
        if state_payload:
            try:
                remote_state = DroneState.from_dict(state_payload)
                # Also ingest patrol_progress from gossip
                remote_state.patrol_progress = float(
                    payload.get("patrol_progress", remote_state.patrol_progress)
                )
                self.update_member_state(sender_id, remote_state)
            except Exception as e:
                logger.warning(f"Failed to ingest sender state from drone {sender_id}: {e}")

        cbba_payload = payload.get("cbba", {})
        if (
            self.config.use_boids_flocking
            and self._flock_coordinator is not None
            and cbba_payload
        ):
            self._flock_coordinator.ingest_gossip_payload(sender_id, cbba_payload)

    def get_desired_velocity(self, drone_id: Optional[int] = None) -> Vector3:
        """Get most recent flocking velocity command for a drone."""
        query_id = self.my_drone_id if drone_id is None else drone_id
        return self._desired_velocities.get(query_id, Vector3())

    def get_desired_goal(self, drone_id: Optional[int] = None) -> Optional[Vector3]:
        """Get most recent flocking task/slot goal for a drone."""
        query_id = self.my_drone_id if drone_id is None else drone_id
        return self._desired_goals.get(query_id)

    def set_forced_goal(self, goal: Optional[Vector3]):
        """Override the autonomous goal with a mission-level shared goal."""
        self._forced_goal = goal

    # ── Telemetry ─────────────────────────────────────────────────

    def get_regiment_status(self) -> Dict:
        """Get full regiment status for monitoring."""
        members = {}
        for m in self._members.values():
            members[f"alpha_{m.drone_id}"] = {
                "active": m.is_active,
                "leader": m.is_leader,
                "mode": m.state.mode.name,
                "battery": round(m.state.battery, 1),
                "position": [
                    round(m.state.position.x, 1),
                    round(m.state.position.y, 1),
                    round(m.state.position.z, 1),
                ],
                "cpu": round(m.cpu_utilization, 2),
                "gpu": round(m.gpu_utilization, 2),
                "obstacles": m.obstacle_count,
                "sector_coverage": (
                    round(m.sector.coverage_percent, 1) if m.sector else 0
                ),
            }

        return {
            "regiment_size": len(self._members),
            "active_count": sum(1 for m in self._members.values() if m.is_active),
            "leader_id": self._current_leader_id,
            "formation": self.config.formation.name,
            "use_boids_flocking": self.config.use_boids_flocking,
            "global_obstacles": len(self._global_obstacle_map),
            "members": members,
            "timestamp": time.time(),
        }

    def get_my_sector(self) -> Optional[SectorAssignment]:
        """Get the sector assigned to this drone."""
        member = self._members.get(self.my_drone_id)
        if member:
            return member.sector
        return None

    def get_my_waypoints(self) -> List[Waypoint]:
        """Get the sweep waypoints for this drone's sector."""
        sector = self.get_my_sector()
        if sector:
            return sector.waypoints
        return []
