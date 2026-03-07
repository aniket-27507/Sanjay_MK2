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
from src.core.utils.geometry import hex_positions
from src.simulation.surveillance_layout import (
    ALPHA_ALTITUDE,
    BETA_ALTITUDE,
    BETA_ID,
    FORMATION_CENTER,
    FORMATION_SPACING,
    build_obstacle_database,
)
from src.swarm.coordination import AlphaRegimentCoordinator, RegimentConfig
from src.swarm.formation import FormationConfig, FormationController, FormationType

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

    def __init__(
        self,
        backend: str = "isaac_sim",
        headless: bool = False,
        formation_spacing: float = FORMATION_SPACING,
    ):
        self._backend = backend
        self._headless = headless
        self._dt = 1.0 / 30.0  # 30 Hz

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
    ):
        """Add a swarm checkpoint (from GUI)."""
        wp = Waypoint(
            position=position,
            speed=speed,
            acceptance_radius=acceptance_radius,
            hold_time=hold_time,
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

    def set_formation_spacing(self, spacing: float):
        """Update formation spacing (from GUI slider)."""
        spacing = max(30.0, min(150.0, spacing))
        self._formation_spacing = spacing
        self._formation.config.spacing = spacing
        self._formation.config.min_separation = min(spacing * 0.6, 50.0)
        self._formation._generate_slots()
        self._formation.assign_drones([BETA_ID, 0, 1, 2, 3, 4, 5])
        logger.info(f"Formation spacing updated to {spacing:.0f}m")

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

                # Update formation center to checkpoint XY (at alpha altitude)
                self._formation.set_center(Vector3(
                    x=checkpoint.position.x,
                    y=checkpoint.position.y,
                    z=self.ALPHA_DEFAULT_Z,
                ))

                # Reset phase for this checkpoint
                self._phase = CheckpointPhase.TRANSIT
                self._status.phase = self._phase

                logger.info(
                    f"Checkpoint {self._current_index + 1}/{len(self._checkpoints)}: "
                    f"({checkpoint.position.x:.0f}, {checkpoint.position.y:.0f}, "
                    f"{checkpoint.position.z:.0f}) — Phase: TRANSIT"
                )

                # Run tick loop until checkpoint fully achieved and hex reformed
                while self._phase != CheckpointPhase.READY or not self._is_hex_reformed():
                    if self._stop_requested:
                        self._status.state = SwarmExecutionState.STOPPED
                        return False
                    await self._pause_event.wait()
                    await self._tick(checkpoint)
                    await asyncio.sleep(self._dt)

                # Checkpoint complete, advance
                logger.info(
                    f"Checkpoint {self._current_index + 1} fully achieved, hex reformed. "
                    f"Advancing."
                )
                self._current_index += 1

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

        # Spawn drones at formation positions around first checkpoint
        first_cp = self._checkpoints[0]
        center_x, center_y = first_cp.position.x, first_cp.position.y

        # Use hex_positions for initial alpha placement
        hex_pos = hex_positions(center_x, center_y, self._formation_spacing, n=7)

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

        # Update status
        beta = self._drones.get(BETA_ID)
        if beta:
            self._status.beta_position = beta.position
            self._status.beta_altitude = -beta.position.z
        self._status.formation_quality = self._compute_formation_quality(checkpoint)
        self._status.min_inter_drone_distance = self._compute_min_inter_drone()

    def _tick_transit(self, checkpoint: Waypoint):
        """TRANSIT: All drones navigate toward checkpoint positions."""
        # 1. Feed LiDAR
        self._feed_lidar_all()

        # 2. Update coordinator states
        for drone_id in range(6):
            drone = self._drones.get(drone_id)
            if drone:
                self._alpha_coordinators[drone_id].update_member_state(
                    drone_id, drone.to_state()
                )

        # 3. Gossip exchange
        self._gossip_exchange()

        # 4. Set forced goals for alphas (checkpoint + hex offset)
        for drone_id in range(6):
            goal = self._get_alpha_goal(drone_id, checkpoint)
            self._alpha_coordinators[drone_id].set_forced_goal(goal)

        # 5. Coordination step
        for coord in self._alpha_coordinators.values():
            coord.coordination_step()

        # 6. Compute and apply beta velocity (P-control to checkpoint XY at 25m)
        beta_goal = Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=self.BETA_DEFAULT_Z,
        )
        self._apply_beta_velocity(beta_goal)

        # 7. Compute and apply alpha velocities
        for drone_id in range(6):
            self._apply_alpha_velocity(drone_id, checkpoint)

        # 8. Check transition
        if self._is_swarm_at_checkpoint_xy(checkpoint):
            self._phase = CheckpointPhase.HOLD_FOR_CLIMB
            logger.info(
                f"Checkpoint {self._current_index + 1}: "
                f"Swarm arrived at XY — Phase: HOLD_FOR_CLIMB"
            )

    def _tick_hold_for_climb(self, checkpoint: Waypoint):
        """HOLD_FOR_CLIMB: Alphas reform hex, beta climbs to checkpoint Z."""
        # Feed LiDAR for beta obstacle avoidance during climb
        self._feed_lidar_all()

        # Alphas: formation correction to reform hex
        self._apply_alpha_formation_correction(checkpoint)

        # Beta: climb to checkpoint Z
        beta_goal = Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=checkpoint.position.z,  # User-specified Z altitude
        )
        self._apply_beta_velocity(beta_goal)

        # Check transition
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
        """DESCEND: Beta returns to 25m, alphas continue reforming."""
        self._feed_lidar_all()

        # Alphas: formation correction
        self._apply_alpha_formation_correction(checkpoint)

        # Beta: descend to default 25m altitude
        beta_goal = Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=self.BETA_DEFAULT_Z,
        )
        self._apply_beta_velocity(beta_goal)

        # Check transition
        beta = self._drones.get(BETA_ID)
        if beta and abs(beta.position.z - self.BETA_DEFAULT_Z) < self.BETA_Z_TOLERANCE:
            self._phase = CheckpointPhase.READY
            logger.info(
                f"Checkpoint {self._current_index + 1}: "
                f"Beta back at 25m — Phase: READY (waiting for hex reform)"
            )

    def _tick_ready(self, checkpoint: Waypoint):
        """READY: Beta holds, alphas reform. Advance when hex is reformed."""
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
        # Beta at checkpoint XY
        beta = self._drones.get(BETA_ID)
        if not beta:
            return False
        beta_target_xy = Vector3(
            x=checkpoint.position.x,
            y=checkpoint.position.y,
            z=beta.position.z,  # Ignore Z for XY check
        )
        beta_pos_xy = Vector3(x=beta.position.x, y=beta.position.y, z=beta.position.z)
        dx = beta.position.x - checkpoint.position.x
        dy = beta.position.y - checkpoint.position.y
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
