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
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import (
    DroneConfig,
    DroneState,
    DroneType,
    FlightMode,
    Vector3,
    Waypoint,
)

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
        self._global_obstacle_map: List[Dict] = []
        self._last_cslam_share: float = 0.0

        # ── FANET Threat Relay ──
        self._threat_queue: List[Dict] = []
        self._relayed_threat_ids: set = set()

        # ── State ──
        self._running = False
        self._initialized = False

        # ── Callbacks ──
        self._on_sector_assigned: Optional[Callable] = None
        self._on_threat_received: Optional[Callable] = None
        self._network_send: Optional[Callable] = None

    # ── Initialization ────────────────────────────────────────────

    async def initialize(self):
        """Initialize the regiment coordinator."""
        self._initialized = True
        self.register_drone(self.my_drone_id)
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

        # Auto-assign sectors when regiment is full
        if len(self._members) == REGIMENT_SIZE:
            self._assign_sectors()

    def unregister_drone(self, drone_id: int):
        """Remove a drone from the regiment."""
        if drone_id in self._members:
            del self._members[drone_id]
            logger.warning(f"Alpha_{drone_id} removed from regiment")
            # Redistribute sectors
            if self._members:
                self._assign_sectors()

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
            try:
                self._check_member_health()
                self._update_coverage()
                self._process_threat_queue()
            except Exception as e:
                logger.error(f"Coordination error: {e}")
            await asyncio.sleep(0.5)

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

        # Prune old entries (keep last 200)
        if len(self._global_obstacle_map) > 200:
            self._global_obstacle_map = self._global_obstacle_map[-200:]

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
        self._relayed_threat_ids.add(threat_id)

    def receive_threat_relay(self, threat_data: Dict):
        """Receive a relayed threat from the FANET mesh."""
        threat_id = threat_data.get("threat_id", "")

        # Deduplicate
        if threat_id in self._relayed_threat_ids:
            return

        self._relayed_threat_ids.add(threat_id)

        # Process the threat
        if self._on_threat_received:
            self._on_threat_received(threat_data)

        # Re-relay if hops remaining
        hops = threat_data.get("hops_remaining", 0)
        if hops > 0:
            threat_data["hops_remaining"] = hops - 1
            self._threat_queue.append(threat_data)

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
