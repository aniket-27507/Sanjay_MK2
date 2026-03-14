"""
Project Sanjay Mk2 - Swarm Waypoint Runner
============================================
Drives a 7-drone regiment (6 alphas + 1 beta) through checkpoint
waypoints entered via the waypoint GUI.

Architecture:
    - Beta_0 flies at center of the hexagonal formation.
    - Alpha_0-5 fly at the 6 vertices of the hexagon.
    - Waypoints from the GUI act as **swarm checkpoints**.
    - Each checkpoint has a 5-phase lifecycle:
        TRANSIT → HOLD_FOR_CLIMB → ACHIEVED → DESCEND → READY

Reuses:
    - AlphaRegimentCoordinator (Boids + CBBA + Gossip) for alphas
    - AvoidanceManager (APF + HPL) per drone
    - FormationController for hex geometry
    - SyntheticLidar for headless testing

@author: Archishman Paul
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.types.drone_types import (
    DroneConfig,
    DroneState,
    DroneType,
    FlightMode,
    Vector3,
    Waypoint,
)
from src.core.utils.geometry import clamp_to_hex_boundary, hex_positions
from src.simulation.surveillance_layout import (
    ALPHA_ALTITUDE,
    BETA_ALTITUDE,
    BETA_ID,
    FORMATION_CENTER,
    FORMATION_SPACING,
    build_obstacle_database,
)
from src.gcs.gcs_server import GCSServer
from src.surveillance.threat_manager import ThreatManager
from src.swarm.coordination import AlphaRegimentCoordinator, RegimentConfig
from src.swarm.formation import FormationConfig, FormationController, FormationType

try:
    from src.single_drone.obstacle_avoidance.apf_3d import AvoidanceState
except ImportError:
    AvoidanceState = None  # graceful fallback

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint Phase State Machine
# ═══════════════════════════════════════════════════════════════════


class CheckpointPhase(Enum):
    """Phase of the 5-step checkpoint lifecycle."""
    TRANSIT = auto()          # Swarm navigating toward checkpoint XY
    HOLD_FOR_CLIMB = auto()   # Alphas reforming hex, beta climbing to checkpoint Z
    ACHIEVED = auto()         # Beta at checkpoint coord — checkpoint done
    DESCEND = auto()          # Beta descending back to 25m
    READY = auto()            # Waiting for alphas to fully reform hex


# ═══════════════════════════════════════════════════════════════════
#  Status Reporting
# ═══════════════════════════════════════════════════════════════════


class SwarmExecutionState(Enum):
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETE = auto()
    STOPPED = auto()
    FAILED = auto()


@dataclass
class SwarmCheckpointStatus:
    """Status snapshot for the GUI."""
    state: SwarmExecutionState = SwarmExecutionState.IDLE
    phase: CheckpointPhase = CheckpointPhase.TRANSIT
    current_index: int = 0
    total_checkpoints: int = 0
    beta_position: Vector3 = field(default_factory=Vector3)
    beta_altitude: float = 0.0
    formation_quality: float = 0.0  # 0-1, fraction of alphas at vertex
    min_inter_drone_distance: float = float("inf")


# ═══════════════════════════════════════════════════════════════════
#  Sim Drone (lightweight kinematics model for headless mode)
# ═══════════════════════════════════════════════════════════════════


class SimDrone:
    """Minimal drone with Euler integration (reused from run_mission.py)."""

    def __init__(self, drone_id: int, position: Vector3, drone_type: DroneType = DroneType.ALPHA):
        self.drone_id = drone_id
        self.drone_type = drone_type
        self.position = position
        self.velocity = Vector3()
        self.mode = FlightMode.NAVIGATING
        self.battery = 100.0
        self.is_active = True

    def step(self, velocity_command: Vector3, dt: float):
        self.velocity = velocity_command
        self.position = Vector3(
            x=self.position.x + velocity_command.x * dt,
            y=self.position.y + velocity_command.y * dt,
            z=self.position.z + velocity_command.z * dt,
        )
        self.battery -= 0.001 * dt

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
#  Swarm Waypoint Runner
# ═══════════════════════════════════════════════════════════════════


class SwarmWaypointRunner:
    """
    Drives a 7-drone regiment (6 alphas + 1 beta) through checkpoint
    waypoints.  Beta_0 sits at the hexagon center; alpha_0-5 on the
    vertices.

    Usage (from GUI):
        runner = SwarmWaypointRunner(backend="isaac_sim")
        runner.add_checkpoint(Vector3(200, 200, -65))
        runner.add_checkpoint(Vector3(500, 300, -65))
        await runner.execute()
    """

    # ── Tolerances ──
    BETA_XY_TOLERANCE = 5.0        # m — beta at checkpoint XY
    BETA_Z_TOLERANCE = 2.0         # m — beta at checkpoint Z
    ALPHA_VERTEX_TOLERANCE = 15.0  # m — alpha at hex vertex
    BETA_DEFAULT_Z = -BETA_ALTITUDE  # NED altitude (negative = up)
    ALPHA_DEFAULT_Z = -ALPHA_ALTITUDE

    # ── P-control gains ──
    BETA_P_GAIN = 0.5
    BETA_CLIMB_SPEED = 3.0  # m/s vertical
    BETA_MAX_SPEED = 6.0    # m/s horizontal
    ALPHA_FORMATION_GAIN = 0.4
    ALPHA_FORMATION_MAX_SPEED = 3.0

    # ── Flock transit ──
    FLOCK_TRANSIT_SPEED = 3.0   # m/s — virtual flock center speed
    FLOCK_LAG_TOLERANCE = 25.0  # m — max allowed lag before pausing

    def __init__(
        self,
        backend: str = "isaac_sim",
        headless: bool = False,
        formation_spacing: float = FORMATION_SPACING,
        start_center: Optional[Vector3] = None,
        start_radius: Optional[float] = None,
    ):
        self._backend = backend
        self._headless = headless
        self._dt = 1.0 / 30.0  # 30 Hz

        # ── Starting hex configuration ──
        self._start_center = start_center
        self._start_radius = start_radius

        # ── Formation ──
        self._formation_spacing = formation_spacing
        self._formation = FormationController(
            num_drones=7,
            config=FormationConfig(
                formation_type=FormationType.HEXAGONAL,
                spacing=formation_spacing,
                altitude=ALPHA_ALTITUDE,
                min_separation=min(formation_spacing * 0.6, 50.0),
            ),
        )
        # Slot 0 = center (beta), slots 1-6 = vertices (alpha_0-5)
        self._formation.assign_drones([BETA_ID, 0, 1, 2, 3, 4, 5])

        # ── Checkpoints ──
        self._checkpoints: List[Waypoint] = []
        self._current_index = 0
        self._phase = CheckpointPhase.TRANSIT

        # ── Execution state ──
        self._status = SwarmCheckpointStatus()
        self._running = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._stop_requested = False

        # ── Drones ──
        self._drones: Dict[int, SimDrone] = {}
        self._formation_offsets: Dict[int, Vector3] = {}

        # ── Coordinators & avoidance (lazy init) ──
        self._alpha_coordinators: Dict[int, AlphaRegimentCoordinator] = {}
        self._alpha_avoidance: Dict[int, object] = {}
        self._beta_avoidance: Optional[object] = None
        self._lidar: Optional[object] = None
        self._obstacles: List[Dict] = []

        # ── Isaac Sim bridge (optional) ──
        self._bridge = None
        self._bridge_ros_started = False
        self._overlay = None
        self._sync_sub = None

        # ── Callbacks ──
        self._on_checkpoint_reached: Optional[Callable] = None

        # ── Threat Manager & Shepherd Protocol ──
        self._threat_manager = ThreatManager(
            hex_center=Vector3(x=0.0, y=0.0, z=-BETA_ALTITUDE),
        )
        self._beta_mission_interrupted = False

        # ── Flock center (moves smoothly toward checkpoint during TRANSIT) ──
        self._flock_center = Vector3()
        self._flock_paused = False  # True when waiting for a lagging drone

        # ── GCS Server (spec §8) ──
        self._gcs = GCSServer(port=8765)
        self._last_gcs_state_push: float = 0.0

        # ── Telemetry ──
        self._collision_count = 0
        self._min_inter_drone = float("inf")

    # ── Public API ─────────────────────────────────────────────────

    def add_checkpoint(
        self,
        position: Vector3,
        speed: float = 5.0,
        acceptance_radius: float = 5.0,
        hold_time: float = 0.0,
        survey_radius: float = 0.0,
    ):
        """Add a swarm checkpoint (from GUI)."""
        wp = Waypoint(
            position=position,
            speed=speed,
            acceptance_radius=acceptance_radius,
            hold_time=hold_time,
            survey_radius=survey_radius,
        )
        self._checkpoints.append(wp)
        self._status.total_checkpoints = len(self._checkpoints)

    def clear_checkpoints(self):
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._current_index = 0
        self._status.total_checkpoints = 0
        self._status.current_index = 0

    @property
    def checkpoints(self) -> List[Waypoint]:
        return list(self._checkpoints)

    @property
    def status(self) -> SwarmCheckpointStatus:
        return self._status

    def set_start_config(self, center: Vector3, radius: float):
        """Set starting hex center and radius (from GUI, before mission start)."""
        self._start_center = center
        self._start_radius = radius

    def set_formation_spacing(self, spacing: float):
        """Update formation spacing (from GUI slider)."""
        spacing = max(30.0, min(150.0, spacing))
        self._formation_spacing = spacing
        self._formation.config.spacing = spacing
        self._formation.config.min_separation = min(spacing * 0.6, 50.0)
        self._formation._generate_slots()
        self._formation.assign_drones([BETA_ID, 0, 1, 2, 3, 4, 5])
        logger.info(f"Formation spacing updated to {spacing:.0f}m")

    def _get_survey_radius(self, checkpoint: Waypoint) -> float:
        """Return the effective survey radius for a checkpoint."""
        if checkpoint.survey_radius > 0:
            return checkpoint.survey_radius
        return self._formation_spacing

    def set_avoidance_enabled(self, enabled: bool):
        """Toggle avoidance for all drones."""
        # Will be used by ModeManager
        pass  # Avoidance is always on in swarm mode; toggle is a no-op for safety

    def set_boids_enabled(self, enabled: bool):
        """Toggle boids in all alpha coordinators."""
        for coord in self._alpha_coordinators.values():
            if coord._flock_coordinator is not None:
                coord._flock_coordinator.enable_boids(enabled)

    def set_cbba_enabled(self, enabled: bool):
        """Toggle CBBA in all alpha coordinators."""
        for coord in self._alpha_coordinators.values():
            if coord._flock_coordinator is not None:
                coord._flock_coordinator.enable_cbba(enabled)

    def set_formation_enabled(self, enabled: bool):
        """Toggle formation in all alpha coordinators."""
        for coord in self._alpha_coordinators.values():
            if coord._flock_coordinator is not None:
                coord._flock_coordinator.enable_formation(enabled)

    def pause(self):
        """Pause swarm execution."""
        self._pause_event.clear()
        self._status.state = SwarmExecutionState.PAUSED

    def resume(self):
        """Resume swarm execution."""
        self._pause_event.set()
        self._status.state = SwarmExecutionState.RUNNING

    def stop(self):
        """Stop swarm execution."""
        self._stop_requested = True
        self._pause_event.set()  # Unblock if paused

    # ── Main Execution ─────────────────────────────────────────────

    async def execute(self) -> bool:
        """Execute all checkpoints. Returns True on success."""
        if not self._checkpoints:
            logger.warning("No checkpoints to execute")
            return False

        self._running = True
        self._stop_requested = False
        self._current_index = 0
        self._phase = CheckpointPhase.TRANSIT
        self._status.state = SwarmExecutionState.RUNNING

        # Initialize all drones, coordinators, avoidance
        await self._initialize_all()

        try:
            while self._current_index < len(self._checkpoints):
                if self._stop_requested:
                    self._status.state = SwarmExecutionState.STOPPED
                    return False

                await self._pause_event.wait()

                checkpoint = self._checkpoints[self._current_index]
                self._status.current_index = self._current_index

                # Formation center stays at _flock_center (not checkpoint).
                # _advance_flock_center() will move it smoothly during TRANSIT.
                # Hex radius for this checkpoint (used after arrival).
                self._current_hex_radius = self._get_survey_radius(checkpoint)

                # Reset phase for this checkpoint
                self._phase = CheckpointPhase.TRANSIT
                self._status.phase = self._phase

                logger.info(
                    f"Checkpoint {self._current_index + 1}/{len(self._checkpoints)}: "
                    f"({checkpoint.position.x:.0f}, {checkpoint.position.y:.0f}, "
                    f"{checkpoint.position.z:.0f}) — Phase: TRANSIT"
                )

                # Run tick loop until checkpoint fully achieved, hex reformed,
                # and no active threats remain
                while (
                    self._phase != CheckpointPhase.READY
                    or not self._is_hex_reformed()
                    or self._beta_mission_interrupted
                    or self._threat_manager.has_active_threat_response()
                ):
                    if self._stop_requested:
                        self._status.state = SwarmExecutionState.STOPPED
                        return False
                    await self._pause_event.wait()
                    await self._tick(checkpoint)
                    await asyncio.sleep(self._dt)

                # Reset interruption flag for next checkpoint
                self._beta_mission_interrupted = False

                # Checkpoint complete, advance
                logger.info(
                    f"Checkpoint {self._current_index + 1} fully achieved, hex reformed, "
                    f"no threats active. Advancing."
                )
                self._current_index += 1

            # ── Return to start position ──
            if self._start_center is not None:
                logger.info("All checkpoints complete. Returning to start position.")
                return_wp = Waypoint(
                    position=Vector3(
                        x=self._start_center.x,
                        y=self._start_center.y,
                        z=self.BETA_DEFAULT_Z,
                    ),
                    speed=5.0,
                    acceptance_radius=5.0,
                )
                self._phase = CheckpointPhase.TRANSIT
                self._status.phase = self._phase

                while True:
                    if self._stop_requested:
                        self._status.state = SwarmExecutionState.STOPPED
                        return False
                    await self._pause_event.wait()

                    # Use the same transit logic (flock center moves to start)
                    self._feed_lidar_all()
                    arrived = self._advance_flock_center(return_wp)
                    self._cooperative_obstacle_assist()

                    # Update coordinators and apply velocities
                    for drone_id in range(6):
                        drone = self._drones.get(drone_id)
                        if drone:
                            self._alpha_coordinators[drone_id].update_member_state(
                                drone_id, drone.to_state()
                            )
                    self._gossip_exchange()
                    for drone_id in range(6):
                        goal = self._get_alpha_goal(drone_id, return_wp)
                        self._alpha_coordinators[drone_id].set_forced_goal(goal)
                    for coord in self._alpha_coordinators.values():
                        coord.coordination_step()

                    beta_goal = Vector3(
                        x=self._flock_center.x,
                        y=self._flock_center.y,
                        z=self.BETA_DEFAULT_Z,
                    )
                    self._apply_beta_velocity(beta_goal)
                    for drone_id in range(6):
                        self._apply_alpha_velocity(drone_id, return_wp)

                    self._push_gcs_state(return_wp)
                    await asyncio.sleep(self._dt)

                    if arrived and not self._any_drone_lagging():
                        logger.info("Swarm returned to start position.")
                        break

            self._status.state = SwarmExecutionState.COMPLETE
            self._status.current_index = len(self._checkpoints)
            logger.info("All checkpoints achieved. Mission complete.")
            return True

        except Exception as e:
            logger.error(f"Swarm execution failed: {e}", exc_info=True)
            self._status.state = SwarmExecutionState.FAILED
            return False
        finally:
            self._running = False

    # ── Initialization ─────────────────────────────────────────────

    async def _initialize_all(self):
        """Initialize drones, coordinators, avoidance managers, LiDAR."""
        # Load obstacles
        self._obstacles = build_obstacle_database(ned_frame=True)
        logger.info(f"Loaded {len(self._obstacles)} obstacles")

        # Build synthetic LiDAR
        self._init_lidar()

        # Spawn drones at starting hex (if provided) or first checkpoint
        if self._start_center is not None:
            center_x, center_y = self._start_center.x, self._start_center.y
        else:
            first_cp = self._checkpoints[0]
            center_x, center_y = first_cp.position.x, first_cp.position.y

        spawn_radius = self._start_radius if self._start_radius is not None else self._formation_spacing

        # Use hex_positions for initial alpha placement
        hex_pos = hex_positions(center_x, center_y, spawn_radius, n=7)

        # Track current hex center for boundary enforcement
        self._current_hex_center = Vector3(x=center_x, y=center_y, z=0.0)
        self._current_hex_radius = spawn_radius

        # Initialize flock center at spawn position
        self._flock_center = Vector3(x=center_x, y=center_y, z=self.ALPHA_DEFAULT_Z)

        # Beta_0 at center
        self._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=center_x, y=center_y, z=self.BETA_DEFAULT_Z),
            drone_type=DroneType.BETA,
        )

        # Alpha_0-5 at hex vertices
        for i in range(6):
            vx, vy = hex_pos[i + 1]  # Skip center (index 0)
            self._drones[i] = SimDrone(
                drone_id=i,
                position=Vector3(x=vx, y=vy, z=self.ALPHA_DEFAULT_Z),
                drone_type=DroneType.ALPHA,
            )
            self._formation_offsets[i] = Vector3(
                x=vx - center_x,
                y=vy - center_y,
                z=0.0,
            )

        # Initialize avoidance managers
        self._init_avoidance()

        # Initialize coordinators (alphas only)
        await self._init_coordinators()

        # Connect to Isaac Sim if available
        if not self._headless:
            self._init_overlay()
            self._register_stage_sync()
            self._init_bridge()

        # Start GCS WebSocket server
        self._gcs.start()
        self._gcs.on_override(self._handle_gcs_override)
        self._gcs.emit_audit("init", f"Swarm initialized: 7 drones, spacing={self._formation_spacing}m")

        logger.info(
            f"Swarm initialized: {len(self._drones)} drones "
            f"(6 alpha + 1 beta), spacing={self._formation_spacing}m"
        )

    def _init_lidar(self):
        """Initialize synthetic LiDAR."""
        try:
            from scripts.isaac_sim.run_mission import SyntheticLidar
            self._lidar = SyntheticLidar(self._obstacles)
        except ImportError:
            logger.warning("SyntheticLidar unavailable — running without LiDAR")
            self._lidar = None

    def _init_avoidance(self):
        """Initialize an AvoidanceManager per drone."""
        try:
            from src.single_drone.obstacle_avoidance.avoidance_manager import (
                AvoidanceManager,
                AvoidanceManagerConfig,
            )

            config = AvoidanceManagerConfig()
            config.control_rate_hz = 30.0

            # Alpha avoidance managers
            for drone_id in range(6):
                mgr = AvoidanceManager(drone_id=drone_id, config=config)
                self._alpha_avoidance[drone_id] = mgr

            # Beta avoidance manager
            self._beta_avoidance = AvoidanceManager(drone_id=BETA_ID, config=config)

            logger.info("AvoidanceManagers initialized for all 7 drones")

        except ImportError as e:
            logger.warning(f"AvoidanceManager unavailable: {e}")

    async def _init_coordinators(self):
        """Initialize one AlphaRegimentCoordinator per alpha drone."""
        for drone_id in range(6):
            cfg = RegimentConfig(
                formation_spacing=self._formation_spacing,
                formation_altitude=ALPHA_ALTITUDE,
                total_coverage_area=1000.0,
                use_boids_flocking=True,
            )
            coordinator = AlphaRegimentCoordinator(
                my_drone_id=drone_id, config=cfg
            )
            await coordinator.initialize()
            # Register all alpha peers
            for peer_id in range(6):
                coordinator.register_drone(peer_id)
            self._alpha_coordinators[drone_id] = coordinator

        logger.info("6 AlphaRegimentCoordinators initialized")

    # ── Tick Loop ──────────────────────────────────────────────────

    async def _tick(self, checkpoint: Waypoint):
        """One 30Hz tick for all 7 drones. Behavior depends on phase."""
        self._status.phase = self._phase

        # Check threat interruption for Beta during climb/descend phases
        threat_active = self._threat_manager.has_active_threat_response()
        if self._phase in (CheckpointPhase.HOLD_FOR_CLIMB, CheckpointPhase.DESCEND):
            if threat_active and not self._beta_mission_interrupted:
                self._beta_mission_interrupted = True
                logger.info(
                    f"Checkpoint {self._current_index + 1}: "
                    f"Beta mission INTERRUPTED — responding to threat"
                )
            elif not threat_active and self._beta_mission_interrupted:
                self._beta_mission_interrupted = False
                logger.info(
                    f"Checkpoint {self._current_index + 1}: "
                    f"Threat resolved — Beta RESUMING {self._phase.name}"
                )

        if self._phase == CheckpointPhase.TRANSIT:
            self._tick_transit(checkpoint)
        elif self._phase == CheckpointPhase.HOLD_FOR_CLIMB:
            self._tick_hold_for_climb(checkpoint)
        elif self._phase == CheckpointPhase.ACHIEVED:
            self._tick_achieved(checkpoint)
        elif self._phase == CheckpointPhase.DESCEND:
            self._tick_descend(checkpoint)
        elif self._phase == CheckpointPhase.READY:
            self._tick_ready(checkpoint)

        # Tick threat manager & shepherd protocols
        self._tick_shepherds()

        # Update status
        beta = self._drones.get(BETA_ID)
        if beta:
            self._status.beta_position = beta.position
            self._status.beta_altitude = -beta.position.z
        self._status.formation_quality = self._compute_formation_quality(checkpoint)
        self._status.min_inter_drone_distance = self._compute_min_inter_drone()

        # Push state to GCS at 5 Hz
        self._push_gcs_state(checkpoint)

    def _tick_transit(self, checkpoint: Waypoint):
        """TRANSIT: Flock center moves toward checkpoint; all drones follow."""
        # 1. Feed LiDAR
        self._feed_lidar_all()

        # 2. Advance flock center toward checkpoint (pauses if any drone lags)
        arrived = self._advance_flock_center(checkpoint)

        # 3. Cooperative obstacle assist (help stuck drones)
        self._cooperative_obstacle_assist()

        # 4. Update coordinator states
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            if drone:
                self._alpha_coordinators[drone_id].update_member_state(
                    drone_id, drone.to_state()
                )

        # 5. Gossip exchange
        self._gossip_exchange()

        # 6. Set forced goals (slots are around _flock_center, not checkpoint)
        for drone_id in range(6):
            goal = self._get_alpha_goal(drone_id, checkpoint)
            self._alpha_coordinators[drone_id].set_forced_goal(goal)

        # 7. Coordination step
        for coord in self._alpha_coordinators.values():
            coord.coordination_step()

        # 8. Beta navigates to flock center (NOT directly to checkpoint)
        beta_goal = Vector3(
            x=self._flock_center.x,
            y=self._flock_center.y,
            z=self.BETA_DEFAULT_Z,
        )
        self._apply_beta_velocity(beta_goal)

        # 9. Apply alpha velocities
        for drone_id in range(6):
            self._apply_alpha_velocity(drone_id, checkpoint)

        # 10. Check transition: flock center arrived AND drones in formation
        if arrived and self._is_swarm_at_checkpoint_xy(checkpoint):
            # Snap all centers to checkpoint for survey phases
            self._flock_center = Vector3(
                x=checkpoint.position.x,
                y=checkpoint.position.y,
                z=self.ALPHA_DEFAULT_Z,
            )
            self._sync_centers_to_flock()

            # Set threat manager centers for Beta RTL during survey
            self._threat_manager.set_hex_center(Vector3(
                x=checkpoint.position.x,
                y=checkpoint.position.y,
                z=self.BETA_DEFAULT_Z,
            ))
            self._threat_manager.set_hex_radius(self._current_hex_radius)

            # Reassign triangle sectors scaled to checkpoint survey radius
            survey_r = self._get_survey_radius(checkpoint)
            cp_center = Vector3(
                x=checkpoint.position.x,
                y=checkpoint.position.y,
                z=-ALPHA_ALTITUDE,
            )
            for coord in self._alpha_coordinators.values():
                coord.reassign_sectors_for_radius(cp_center, survey_r)

            self._phase = CheckpointPhase.HOLD_FOR_CLIMB
            logger.info(
                f"Checkpoint {self._current_index + 1}: "
                f"Swarm arrived at XY — Phase: HOLD_FOR_CLIMB "
                f"(survey_radius={survey_r:.0f}m)"
            )

    def _tick_hold_for_climb(self, checkpoint: Waypoint):
        """HOLD_FOR_CLIMB: Alphas survey triangle sectors, beta climbs to checkpoint Z."""
        # Defensive: ensure flock center is at checkpoint during survey phases
        self._flock_center = Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=self.ALPHA_DEFAULT_Z,
        )
        self._sync_centers_to_flock()

        # Feed LiDAR for obstacle avoidance
        self._feed_lidar_all()

        # Alphas: actively survey their triangle sectors (continue regardless of threats)
        self._apply_alpha_survey(checkpoint)

        # Beta: climb to checkpoint Z (skip if interrupted by threat response)
        if not self._beta_mission_interrupted:
            beta_goal = Vector3(
                x=checkpoint.position.x,
                y=checkpoint.position.y,
                z=checkpoint.position.z,  # User-specified Z altitude
            )
            self._apply_beta_velocity(beta_goal)

            # Check transition (only when not interrupted)
            beta = self._drones.get(BETA_ID)
            if beta and abs(beta.position.z - checkpoint.position.z) < self.BETA_Z_TOLERANCE:
                self._phase = CheckpointPhase.ACHIEVED
                logger.info(
                    f"Checkpoint {self._current_index + 1}: "
                    f"Beta at target altitude — Phase: ACHIEVED"
                )

    def _tick_achieved(self, checkpoint: Waypoint):
        """ACHIEVED: Log checkpoint, hold briefly, then descend."""
        # Alphas continue formation correction
        self._apply_alpha_formation_correction(checkpoint)

        # Beta holds position at checkpoint
        beta_goal = Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=checkpoint.position.z,
        )
        self._apply_beta_velocity(beta_goal)

        # Fire callback
        if self._on_checkpoint_reached:
            self._on_checkpoint_reached(self._current_index, checkpoint)

        logger.info(
            f"Checkpoint {self._current_index + 1}: ACHIEVED at "
            f"({checkpoint.position.x:.0f}, {checkpoint.position.y:.0f}, "
            f"{checkpoint.position.z:.0f}) — Phase: DESCEND"
        )

        self._phase = CheckpointPhase.DESCEND

    def _tick_descend(self, checkpoint: Waypoint):
        """DESCEND: Beta returns to 25m, alphas continue surveying."""
        self._feed_lidar_all()

        # Alphas: continue surveying their triangle sectors (continue regardless of threats)
        self._apply_alpha_survey(checkpoint)

        # Beta: descend to default 25m altitude (skip if interrupted by threat response)
        if not self._beta_mission_interrupted:
            beta_goal = Vector3(
                x=checkpoint.position.x,
                y=checkpoint.position.y,
                z=self.BETA_DEFAULT_Z,
            )
            self._apply_beta_velocity(beta_goal)

            # Check transition (only when not interrupted)
            beta = self._drones.get(BETA_ID)
            if beta and abs(beta.position.z - self.BETA_DEFAULT_Z) < self.BETA_Z_TOLERANCE:
                self._phase = CheckpointPhase.READY
                logger.info(
                    f"Checkpoint {self._current_index + 1}: "
                    f"Beta back at 25m — Phase: READY (waiting for hex reform)"
                )

    def _tick_ready(self, checkpoint: Waypoint):
        """READY: Beta holds, alphas reform. Advance when hex is reformed and no threats."""
        # If threats are active during READY, let shepherd handle Beta
        if self._threat_manager.has_active_threat_response():
            pass  # _tick_shepherds() will control Beta
        else:
            # Beta: hold at checkpoint XY at 25m
            beta = self._drones.get(BETA_ID)
            if beta:
                beta.step(Vector3(), self._dt)  # Zero velocity hold

        # Alphas: continue formation correction
        self._apply_alpha_formation_correction(checkpoint)

        # Transition checked in execute() loop via _is_hex_reformed()

    # ── Velocity Computation ───────────────────────────────────────

    def _apply_beta_velocity(self, goal: Vector3):
        """Compute and apply P-control velocity for beta toward goal."""
        beta = self._drones.get(BETA_ID)
        if not beta:
            return

        error = goal - beta.position
        dist = error.magnitude()

        if dist < 0.5:
            velocity = Vector3()
        else:
            direction = error.normalized()
            # Separate XY and Z control
            xy_error = Vector3(x=error.x, y=error.y, z=0.0)
            xy_dist = xy_error.magnitude()
            z_error = error.z

            # XY P-control with saturation
            if xy_dist > 0.5:
                xy_speed = min(xy_dist * self.BETA_P_GAIN, self.BETA_MAX_SPEED)
                xy_vel = xy_error.normalized() * xy_speed
            else:
                xy_vel = Vector3()

            # Z P-control with climb speed limit
            if abs(z_error) > 0.5:
                z_speed = min(abs(z_error) * self.BETA_P_GAIN, self.BETA_CLIMB_SPEED)
                z_vel = z_speed if z_error > 0 else -z_speed
            else:
                z_vel = 0.0

            velocity = Vector3(x=xy_vel.x, y=xy_vel.y, z=z_vel)

        # Apply avoidance if available
        if self._beta_avoidance is not None:
            try:
                self._beta_avoidance.set_goal(goal)
                self._beta_avoidance.set_boids_velocity(velocity)
                velocity = self._beta_avoidance.compute_avoidance(
                    drone_position=beta.position,
                    drone_velocity=beta.velocity,
                )
            except Exception:
                pass  # Fall back to raw P-control

        # Clamp to hex boundary only during survey phases (not TRANSIT —
        # during transit Beta follows the moving flock center freely)
        if (self._phase != CheckpointPhase.TRANSIT
                and hasattr(self, '_current_hex_center')
                and hasattr(self, '_current_hex_radius')):
            next_x = beta.position.x + velocity.x * self._dt
            next_y = beta.position.y + velocity.y * self._dt
            cx = self._current_hex_center.x
            cy = self._current_hex_center.y
            clamped_x, clamped_y = clamp_to_hex_boundary(
                next_x, next_y, cx, cy, self._current_hex_radius,
            )
            if (clamped_x, clamped_y) != (next_x, next_y):
                velocity = Vector3(
                    x=(clamped_x - beta.position.x) / self._dt,
                    y=(clamped_y - beta.position.y) / self._dt,
                    z=velocity.z,
                )

        beta.step(velocity, self._dt)
        self._publish_velocity(BETA_ID, velocity)

    def _apply_alpha_velocity(self, drone_id: int, checkpoint: Waypoint):
        """Compute and apply velocity for an alpha drone during TRANSIT."""
        drone = self._drones.get(drone_id)
        coord = self._alpha_coordinators.get(drone_id)
        mgr = self._alpha_avoidance.get(drone_id)

        if not drone or not coord:
            return

        desired_velocity = coord.get_desired_velocity(drone_id)
        goal = coord.get_desired_goal(drone_id)

        if mgr is not None:
            try:
                mgr.set_boids_velocity(desired_velocity)
                if goal is not None:
                    mgr.set_goal(goal)
                velocity = mgr.compute_avoidance(
                    drone_position=drone.position,
                    drone_velocity=drone.velocity,
                )
            except Exception:
                velocity = desired_velocity
        else:
            velocity = desired_velocity

        drone.step(velocity, self._dt)
        self._publish_velocity(drone_id, velocity)

    def _apply_alpha_formation_correction(self, checkpoint: Waypoint):
        """Apply formation correction velocity to reform hexagon (phases 2-5)."""
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            if not drone:
                continue

            # Target is the hex vertex position around checkpoint center
            slot_pos = self._formation.get_slot_for_drone(drone_id)
            if slot_pos is None:
                continue

            error = slot_pos - drone.position
            dist = error.magnitude()

            if dist < 1.0:
                # Already at vertex — hold position
                drone.step(Vector3(), self._dt)
                self._publish_velocity(drone_id, Vector3())
                continue

            # P-control toward vertex with deceleration
            direction = error.normalized()
            if dist < 10.0:
                speed = self.ALPHA_FORMATION_MAX_SPEED * (dist / 10.0)
            else:
                speed = self.ALPHA_FORMATION_MAX_SPEED

            velocity = direction * (speed * self.ALPHA_FORMATION_GAIN)

            # Clamp
            mag = velocity.magnitude()
            if mag > self.ALPHA_FORMATION_MAX_SPEED:
                velocity = velocity * (self.ALPHA_FORMATION_MAX_SPEED / mag)

            drone.step(velocity, self._dt)
            self._publish_velocity(drone_id, velocity)

    def _apply_alpha_survey(self, checkpoint: Waypoint):
        """Alphas navigate within their triangle sectors during survey phases.

        Each alpha reads its assigned sector's waypoints from the coordinator
        and navigates through them for area coverage. Falls back to formation
        correction if no sector is assigned.
        """
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            coord = self._alpha_coordinators.get(drone_id)
            if not drone or not coord:
                continue

            # Get sector waypoints from coordinator's member data
            member = coord._members.get(drone_id)
            sector = member.sector if member else None

            if sector and sector.waypoints:
                # Navigate through sector waypoints in sequence
                wp_idx = getattr(sector, '_survey_wp_idx', 0)
                if wp_idx >= len(sector.waypoints):
                    wp_idx = 0  # Loop survey pattern
                target_wp = sector.waypoints[wp_idx]
                target = target_wp.position

                error = target - drone.position
                # Only compare XY distance (alpha stays at its altitude)
                xy_dist = math.sqrt(error.x ** 2 + error.y ** 2)

                if xy_dist < target_wp.acceptance_radius:
                    # Reached this waypoint, advance to next
                    sector._survey_wp_idx = wp_idx + 1
                    drone.step(Vector3(), self._dt)
                    self._publish_velocity(drone_id, Vector3())
                else:
                    direction = Vector3(x=error.x, y=error.y, z=0.0).normalized()
                    speed = min(xy_dist * self.ALPHA_FORMATION_GAIN,
                                self.ALPHA_FORMATION_MAX_SPEED)
                    velocity = direction * speed
                    drone.step(velocity, self._dt)
                    self._publish_velocity(drone_id, velocity)
            else:
                # No sector assigned — fall back to formation correction
                slot_pos = self._formation.get_slot_for_drone(drone_id)
                if slot_pos is None:
                    continue
                error = slot_pos - drone.position
                dist = error.magnitude()
                if dist < 1.0:
                    drone.step(Vector3(), self._dt)
                    self._publish_velocity(drone_id, Vector3())
                else:
                    direction = error.normalized()
                    speed = min(dist * self.ALPHA_FORMATION_GAIN,
                                self.ALPHA_FORMATION_MAX_SPEED)
                    velocity = direction * speed
                    drone.step(velocity, self._dt)
                    self._publish_velocity(drone_id, velocity)

    # ── Goal Computation ───────────────────────────────────────────

    def _get_alpha_goal(self, drone_id: int, checkpoint: Waypoint) -> Vector3:
        """Get the target position for an alpha drone (checkpoint + hex offset)."""
        slot_pos = self._formation.get_slot_for_drone(drone_id)
        if slot_pos is not None:
            return slot_pos
        # Fallback: checkpoint center at alpha altitude
        return Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=self.ALPHA_DEFAULT_Z,
        )

    # ── Phase Transition Checks ────────────────────────────────────

    def _is_swarm_at_checkpoint_xy(self, checkpoint: Waypoint) -> bool:
        """Check if beta and all alphas are within XY tolerance of their targets."""
        # Beta at flock center XY (not raw checkpoint — flock center IS at
        # checkpoint when this is called from the arrived=True branch)
        beta = self._drones.get(BETA_ID)
        if not beta:
            return False
        dx = beta.position.x - self._flock_center.x
        dy = beta.position.y - self._flock_center.y
        beta_xy_dist = math.sqrt(dx * dx + dy * dy)
        if beta_xy_dist > self.BETA_XY_TOLERANCE:
            return False

        # Alphas at their hex vertex XY positions
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            slot_pos = self._formation.get_slot_for_drone(drone_id)
            if not drone or not slot_pos:
                return False
            dx = drone.position.x - slot_pos.x
            dy = drone.position.y - slot_pos.y
            alpha_xy_dist = math.sqrt(dx * dx + dy * dy)
            if alpha_xy_dist > self.ALPHA_VERTEX_TOLERANCE:
                return False

        return True

    def _is_hex_reformed(self) -> bool:
        """Check if all 6 alphas are within tolerance of their hex vertex positions."""
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            slot_pos = self._formation.get_slot_for_drone(drone_id)
            if not drone or not slot_pos:
                return False
            dist = drone.position.distance_to(slot_pos)
            if dist > self.ALPHA_VERTEX_TOLERANCE:
                return False
        return True

    # ── LiDAR & Gossip ─────────────────────────────────────────────

    def _feed_lidar_all(self):
        """Feed synthetic LiDAR to all avoidance managers."""
        if self._lidar is None:
            return

        for drone_id, drone in self._drones.items():
            if not drone.is_active:
                continue
            points = self._lidar.scan(drone.position)
            if drone_id == BETA_ID:
                if self._beta_avoidance is not None:
                    self._beta_avoidance.feed_lidar_points(
                        points, drone_position=drone.position
                    )
            else:
                mgr = self._alpha_avoidance.get(drone_id)
                if mgr is not None:
                    mgr.feed_lidar_points(points, drone_position=drone.position)

    def _gossip_exchange(self):
        """In-process gossip broadcast between alpha coordinators."""
        payloads = {
            drone_id: coord.prepare_gossip_payload()
            for drone_id, coord in self._alpha_coordinators.items()
        }
        for receiver_id, coord in self._alpha_coordinators.items():
            for sender_id, payload in payloads.items():
                if sender_id == receiver_id or not payload:
                    continue
                coord.ingest_gossip_payload(payload)

    # ── Flock Center Movement ─────────────────────────────────────

    def _any_drone_lagging(self) -> bool:
        """Check if any drone is too far from its formation slot.

        Returns True if flock center should pause (a drone is stuck).
        """
        # Check Beta distance to flock center
        beta = self._drones.get(BETA_ID)
        if beta:
            dx = beta.position.x - self._flock_center.x
            dy = beta.position.y - self._flock_center.y
            if math.sqrt(dx * dx + dy * dy) > self.FLOCK_LAG_TOLERANCE:
                return True

        # Check each Alpha distance to its hex vertex slot
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            slot_pos = self._formation.get_slot_for_drone(drone_id)
            if not drone or not slot_pos:
                continue
            dx = drone.position.x - slot_pos.x
            dy = drone.position.y - slot_pos.y
            if math.sqrt(dx * dx + dy * dy) > self.FLOCK_LAG_TOLERANCE:
                return True

        return False

    def _advance_flock_center(self, checkpoint: Waypoint) -> bool:
        """Move _flock_center toward checkpoint XY at FLOCK_TRANSIT_SPEED.

        Pauses when any drone is lagging, threats are active, or Beta
        is interrupted. After moving, updates formation center and hex
        boundary so all drones track the moving center.

        Returns True when flock center has arrived at checkpoint XY.
        """
        # Check if already at checkpoint
        error_x = checkpoint.position.x - self._flock_center.x
        error_y = checkpoint.position.y - self._flock_center.y
        dist = math.sqrt(error_x * error_x + error_y * error_y)

        if dist < 1.0:
            # Snap to checkpoint
            self._flock_center = Vector3(
                x=checkpoint.position.x,
                y=checkpoint.position.y,
                z=self.ALPHA_DEFAULT_Z,
            )
            self._sync_centers_to_flock()
            self._flock_paused = False
            return True

        # Pause conditions: don't advance if any drone is stuck or threats active
        if (self._any_drone_lagging()
                or self._threat_manager.has_active_threat_response()
                or self._beta_mission_interrupted):
            self._flock_paused = True
            # Still update formation center to current flock position
            self._sync_centers_to_flock()
            return False

        self._flock_paused = False

        # Move flock center toward checkpoint
        step = min(dist, self.FLOCK_TRANSIT_SPEED * self._dt)
        self._flock_center = Vector3(
            x=self._flock_center.x + (error_x / dist) * step,
            y=self._flock_center.y + (error_y / dist) * step,
            z=self.ALPHA_DEFAULT_Z,
        )

        self._sync_centers_to_flock()
        return False

    def _sync_centers_to_flock(self):
        """Sync formation center and hex boundary to current flock center."""
        self._formation.set_center(self._flock_center)
        self._current_hex_center = Vector3(
            x=self._flock_center.x,
            y=self._flock_center.y,
            z=0.0,
        )

    def _cooperative_obstacle_assist(self):
        """Detect stuck drones and share neighbor sensor data to help navigate.

        When a drone is stuck (avoidance state AVOIDING/STUCK/EMERGENCY or
        velocity near zero), collect obstacle data from 2-3 nearest neighbors
        and feed it to the stuck drone's avoidance manager via
        receive_swarm_threat().  This gives the stuck drone's A* planner a
        multi-angle view of the obstacle.
        """
        for drone_id, drone in self._drones.items():
            # Get avoidance manager for this drone
            if drone_id == BETA_ID:
                mgr = self._beta_avoidance
            else:
                mgr = self._alpha_avoidance.get(drone_id)
            if mgr is None:
                continue

            # Check if drone is stuck (avoidance state or velocity-based)
            is_stuck = False
            if AvoidanceState is not None:
                try:
                    if mgr.state in (AvoidanceState.STUCK, AvoidanceState.EMERGENCY):
                        is_stuck = True
                    elif mgr.state == AvoidanceState.AVOIDING:
                        speed = drone.velocity.magnitude()
                        if speed < 0.5:
                            is_stuck = True
                except Exception:
                    speed = drone.velocity.magnitude()
                    if speed < 0.3:
                        is_stuck = True
            else:
                speed = drone.velocity.magnitude()
                if speed < 0.3:
                    is_stuck = True

            if not is_stuck:
                continue

            # Find 2-3 nearest neighbor drones
            neighbors = []
            for other_id, other_drone in self._drones.items():
                if other_id == drone_id:
                    continue
                dist = drone.position.distance_to(other_drone.position)
                neighbors.append((other_id, other_drone, dist))
            neighbors.sort(key=lambda x: x[2])
            neighbors = neighbors[:3]  # Top 3 nearest

            # Share each neighbor's obstacle data with the stuck drone
            for neighbor_id, neighbor_drone, _ in neighbors:
                if neighbor_id == BETA_ID:
                    neighbor_mgr = self._beta_avoidance
                else:
                    neighbor_mgr = self._alpha_avoidance.get(neighbor_id)
                if neighbor_mgr is None:
                    continue

                # Extract obstacles from neighbor's APF
                try:
                    neighbor_obstacles = neighbor_mgr._apf._obstacles
                except Exception:
                    continue
                if not neighbor_obstacles:
                    continue

                # Build threat data for receive_swarm_threat()
                obs_list = []
                for obs in neighbor_obstacles[:10]:
                    obs_list.append({
                        "position": [obs.position.x, obs.position.y, obs.position.z],
                        "radius": obs.radius,
                        "confidence": obs.confidence,
                    })

                threat_data = {
                    "drone_id": neighbor_id,
                    "type": "cooperative_assist",
                    "position": [neighbor_drone.position.x,
                                 neighbor_drone.position.y,
                                 neighbor_drone.position.z],
                    "obstacles": obs_list,
                    "hpl_state": "PASSIVE",
                    "timestamp": time.time(),
                }
                try:
                    mgr.receive_swarm_threat(threat_data)
                except Exception:
                    pass

            logger.debug(
                f"Cooperative assist: drone {drone_id} STUCK — shared sensor "
                f"data from neighbors {[n[0] for n in neighbors]}"
            )

    # ── Shepherd Protocol ─────────────────────────────────────────

    def _tick_shepherds(self):
        """Advance all active shepherd guidance protocols one tick."""
        active = self._threat_manager.get_active_shepherds()
        if not active:
            return

        # Collect Beta positions
        beta_positions: Dict[int, Vector3] = {}
        beta = self._drones.get(BETA_ID)
        if beta:
            beta_positions[BETA_ID] = beta.position

        # Threat positions/velocities — use last known from threat manager
        threat_positions: Dict[str, Vector3] = {}
        threat_velocities: Dict[str, Vector3] = {}
        for tid, shepherd in active.items():
            threat = self._threat_manager.get_threat(tid)
            if threat:
                threat_positions[tid] = threat.position
            threat_velocities[tid] = Vector3()  # static threat for now

        # Tick all shepherds and get Beta targets
        targets = self._threat_manager.tick_shepherds(
            dt=self._dt,
            threat_positions=threat_positions,
            threat_velocities=threat_velocities,
            beta_positions=beta_positions,
        )

        # Apply shepherd targets to Beta drones
        for beta_id, (target_pos, target_speed) in targets.items():
            drone = self._drones.get(beta_id)
            if not drone:
                continue
            error = target_pos - drone.position
            dist = error.magnitude()
            if dist < 1.0:
                velocity = Vector3()
            else:
                direction = error.normalized()
                speed = min(dist * 0.5, target_speed)
                velocity = direction * speed
            drone.step(velocity, self._dt)
            self._publish_velocity(beta_id, velocity)

    # ── GCS Integration (spec §8) ─────────────────────────────────

    def _push_gcs_state(self, checkpoint: Waypoint):
        """Push composite state to GCS at 5 Hz (matches existing HTML schema)."""
        now = time.time()
        if now - self._last_gcs_state_push < 1.0 / GCSServer.MAP_VIEW_HZ:
            return
        self._last_gcs_state_push = now

        drones_list = []
        for drone_id, drone in sorted(self._drones.items()):
            role = "beta" if drone.drone_type == DroneType.BETA else "alpha"
            idx = 0 if drone.drone_type == DroneType.BETA else drone_id
            drones_list.append({
                "name": f"Beta_0" if role == "beta" else f"Alpha_{idx}",
                "position": {"x": round(drone.position.x, 1),
                              "y": round(drone.position.y, 1),
                              "z": round(-drone.position.z, 1)},
                "velocity": {"x": round(drone.velocity.x, 2),
                              "y": round(drone.velocity.y, 2),
                              "z": round(drone.velocity.z, 2)},
                "battery": round(drone.battery, 1),
                "mode": drone.mode.name if drone.mode else "NAVIGATING",
                "homeVertex": idx if role == "alpha" else -1,
                "role": role,
                "isActive": drone.is_active,
            })

        threats = self._threat_manager.get_active_threats()
        threat_list = []
        for t in threats:
            threat_list.append({
                "id": t.threat_id,
                "position": [round(t.position.x, 1), round(t.position.y, 1)],
                "level": t.threat_level.name if t.threat_level else "LOW",
                "status": t.status.name if t.status else "DETECTED",
                "score": round(getattr(t, "threat_score", 0.0), 3),
                "assigned": t.assigned_beta,
            })

        state = {
            "type": "state",
            "drones": drones_list,
            "time": round(now, 3),
            "isRunning": self._running,
            "messages": [],
            "allTasksComplete": self._status.state == SwarmExecutionState.COMPLETE,
            "config": {
                "formationSpacing": self._formation_spacing,
                "numAlpha": 6,
                "numBeta": 1,
            },
            "threats": threat_list,
            "checkpoint": {
                "index": self._current_index,
                "total": len(self._checkpoints),
                "phase": self._phase.name,
                "position": {
                    "x": round(checkpoint.position.x, 1),
                    "y": round(checkpoint.position.y, 1),
                    "z": round(checkpoint.position.z, 1),
                } if checkpoint else None,
                "surveyRadius": round(self._get_survey_radius(checkpoint), 1) if checkpoint else 0,
                "betaInterrupted": self._beta_mission_interrupted,
                "flockCenter": {
                    "x": round(self._flock_center.x, 1),
                    "y": round(self._flock_center.y, 1),
                },
                "flockPaused": self._flock_paused,
            },
            "formationQuality": round(self._status.formation_quality, 3),
            "minInterDrone": round(self._status.min_inter_drone_distance, 1),
            "operationalDrones": sum(1 for d in self._drones.values() if d.is_active),
            "failedDrones": sum(1 for d in self._drones.values() if not d.is_active),
            "scenarios": [],
            "activeScenario": None,
        }

        self._gcs.push_state(state)

    def _handle_gcs_override(self, data: dict):
        """Handle override commands from GCS operator (spec §6.5)."""
        command = data.get("command") or data.get("type")
        logger.info("GCS override received: %s", command)

        if command == "pause":
            self.pause()
        elif command == "start":
            self.resume()
        elif command == "stop":
            self.stop()
        elif command == "dispatch":
            # Manual Beta dispatch to a threat
            threat_id = data.get("threat_id")
            beta_id = data.get("target_drone", BETA_ID)
            if threat_id:
                beta = self._drones.get(beta_id)
                if beta:
                    self._threat_manager.request_confirmation(
                        threat_id,
                        [(beta_id, beta.position)],
                    )
                    self._gcs.emit_audit(
                        "manual_dispatch",
                        f"Operator dispatched Beta_{beta_id} to {threat_id}",
                    )

    # ── Telemetry ──────────────────────────────────────────────────

    def _compute_formation_quality(self, checkpoint: Waypoint) -> float:
        """Fraction of alphas at their hex vertex position (0-1)."""
        at_vertex = 0
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            slot_pos = self._formation.get_slot_for_drone(drone_id)
            if drone and slot_pos:
                if drone.position.distance_to(slot_pos) < self.ALPHA_VERTEX_TOLERANCE:
                    at_vertex += 1
        return at_vertex / 6.0

    def _compute_min_inter_drone(self) -> float:
        """Compute minimum pairwise distance among all active drones."""
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

    # ── Isaac Sim Integration ──────────────────────────────────────

    def _publish_velocity(self, drone_id: int, velocity: Vector3):
        """Publish velocity command via ROS 2 bridge (if available)."""
        if self._bridge is None:
            return
        drone_name = "beta_0" if drone_id == BETA_ID else f"alpha_{drone_id}"
        try:
            self._bridge.send_velocity(
                drone_name, vx=velocity.x, vy=velocity.y, vz=velocity.z
            )
        except Exception:
            pass

    def _init_overlay(self):
        """Connect to MissionOverlay if in Isaac Sim."""
        try:
            from scripts.isaac_sim.create_surveillance_scene import get_mission_overlay
            self._overlay = get_mission_overlay()
        except Exception:
            self._overlay = None

    def _init_bridge(self):
        """Connect to Isaac Sim ROS 2 bridge."""
        try:
            from src.integration.isaac_sim_bridge import (
                IsaacSimBridgeNode,
                BridgeConfig,
                is_ros2_available,
            )

            if not is_ros2_available():
                logger.info("ROS 2 unavailable — bridge disabled")
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
                parts = drone_name.split("_")
                if drone_name == "beta_0":
                    did = BETA_ID
                else:
                    did = int(parts[-1]) if len(parts) > 1 else 0
                drone = self._drones.get(did)
                if did == BETA_ID and self._beta_avoidance and drone:
                    self._beta_avoidance.feed_lidar_points(
                        points, drone_position=drone.position
                    )
                elif did in self._alpha_avoidance and drone:
                    self._alpha_avoidance[did].feed_lidar_points(
                        points, drone_position=drone.position
                    )

            self._bridge.register_autonomy_hooks(
                on_threat=lambda n, t: None,
                on_lidar=_lidar_hook,
            )
            logger.info("Isaac Sim bridge connected")
        except Exception as e:
            logger.debug(f"Bridge init skipped: {e}")
            self._bridge = None

    def _register_stage_sync(self):
        """Register per-frame stage sync for Isaac Sim viewport."""
        try:
            import omni.kit.app
            app = omni.kit.app.get_app()
            if app:
                self._sync_sub = (
                    app.get_update_event_stream()
                    .create_subscription_to_pop(self._sync_drones_to_stage)
                )
                logger.info("Stage sync registered for swarm")
        except Exception:
            pass

    def _sync_drones_to_stage(self, event=None):
        """Write drone positions to Isaac Sim USD stage."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if not stage:
                return

            for drone_id, drone in self._drones.items():
                if not drone.is_active:
                    continue
                if drone_id == BETA_ID:
                    prim_path = "/World/Drones/Beta_0"
                else:
                    prim_path = f"/World/Drones/Alpha_{drone_id}"

                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                # NED → Isaac (+Z up): flip Z
                pos = Gf.Vec3d(
                    drone.position.x, drone.position.y, -drone.position.z
                )
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                if ops:
                    ops[0].Set(pos)
                else:
                    xform.AddTranslateOp().Set(pos)
        except Exception:
            pass
