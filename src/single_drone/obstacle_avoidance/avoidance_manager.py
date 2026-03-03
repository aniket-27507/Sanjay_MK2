"""
Project Sanjay Mk2 - Avoidance Manager
=======================================
Orchestrates the three-tier obstacle avoidance hierarchy:

    ┌─────────────────────────────────────────────────┐
    │              Strategic Layer                    │
    │         (Mission Coordinator / Waypoints)       │
    └───────────────────┬─────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │              Tactical Layer                     │
    │        (A* Pathfinder / Sub-Waypoints)          │
    └───────────────────┬─────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────┐
    │             Operational Layer                   │
    │    (APF 3D + Hardware Protection Layer)         │
    └───────────────────┬─────────────────────────────┘
                        ▼
                  FlightController

The AvoidanceManager ties sensor input (3D LiDAR + depth) to the
APF core, routes through the HPL safety gate, and escalates to the
Tactical Planner when local minima are detected.

@author: Archishman Paul
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.core.types.drone_types import Vector3, Waypoint
from src.single_drone.sensors.lidar_3d import Lidar3DDriver, Lidar3DConfig
from src.single_drone.obstacle_avoidance.apf_3d import APF3DAvoidance, APF3DConfig, AvoidanceState, Obstacle3D
from src.single_drone.obstacle_avoidance.hardware_protection import HardwareProtectionLayer, HPLConfig, HPLState
from src.single_drone.obstacle_avoidance.tactical_planner import TacticalPlanner, PlannerConfig

logger = logging.getLogger(__name__)


@dataclass
class AvoidanceManagerConfig:
    """Configuration for the AvoidanceManager."""
    apf: APF3DConfig = field(default_factory=APF3DConfig)
    hpl: HPLConfig = field(default_factory=HPLConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    lidar: Lidar3DConfig = field(default_factory=Lidar3DConfig)

    # ── Control Rate ──
    control_rate_hz: float = 30.0       # Avoidance loop rate

    # ── Tactical Escalation ──
    stuck_escalation_time: float = 3.0  # Time in STUCK before A* plan
    replan_interval: float = 5.0        # Min seconds between A* replans

    # ── Swarm Broadcast ──
    broadcast_threats: bool = True      # Share threat data over mesh


class AvoidanceManager:
    """
    Central orchestrator for the three-tier obstacle avoidance stack.

    This is the single entry point for all obstacle avoidance on an
    Alpha Drone.  The FlightController should call this manager
    instead of directly using the APF or HPL.

    Usage:
        manager = AvoidanceManager(drone_id=0)

        # Start the async avoidance loop
        await manager.start()

        # Set the current tactical waypoint
        manager.set_goal(waypoint_position)

        # Every control tick, get the safe velocity command
        velocity = manager.get_avoidance_velocity(drone_pos, drone_vel)

        # Stop when done
        await manager.stop()
    """

    def __init__(
        self,
        drone_id: int = 0,
        config: Optional[AvoidanceManagerConfig] = None,
    ):
        self.drone_id = drone_id
        self.config = config or AvoidanceManagerConfig()

        # ── Subsystem instances ──
        self._lidar = Lidar3DDriver(self.config.lidar)
        self._apf = APF3DAvoidance(self.config.apf)
        self._hpl = HardwareProtectionLayer(self.config.hpl)
        self._planner = TacticalPlanner(self.config.planner)

        # ── State ──
        self._goal: Optional[Vector3] = None
        self._active_sub_waypoints: List[Waypoint] = []
        self._sub_waypoint_index: int = 0
        self._running = False
        self._last_replan_time: float = 0.0
        self._stuck_start_time: Optional[float] = None

        # ── Output ──
        self._current_velocity = Vector3()
        self._current_state = AvoidanceState.CLEAR
        self._hpl_overriding = False

        # ── Callbacks ──
        self._threat_broadcast_callback: Optional[Callable] = None

    # ── Public Interface ──────────────────────────────────────────

    @property
    def state(self) -> AvoidanceState:
        return self._current_state

    @property
    def hpl_state(self) -> HPLState:
        return self._hpl.state

    @property
    def is_avoiding(self) -> bool:
        return self._current_state not in (AvoidanceState.CLEAR, AvoidanceState.MONITORING)

    @property
    def is_hpl_overriding(self) -> bool:
        return self._hpl_overriding

    @property
    def closest_obstacle_distance(self) -> float:
        return self._apf.closest_obstacle_distance

    def set_goal(self, goal: Vector3):
        """Set the current tactical waypoint (from Strategic Layer)."""
        self._goal = goal
        # Clear tactical sub-waypoints — goal has changed
        self._active_sub_waypoints.clear()
        self._sub_waypoint_index = 0

    def feed_lidar_points(self, points: np.ndarray, drone_position: Optional[Vector3] = None):
        """Feed raw 3D LiDAR point cloud."""
        self._lidar.update_points(points, drone_position)

    def on_threat_broadcast(self, callback: Callable):
        """Register callback for swarm threat broadcasts."""
        self._threat_broadcast_callback = callback

    async def start(self):
        """Start the avoidance processing loop."""
        self._running = True
        logger.info(f"AvoidanceManager started for drone {self.drone_id}")

    async def stop(self):
        """Stop the avoidance processing loop."""
        self._running = False
        logger.info(f"AvoidanceManager stopped for drone {self.drone_id}")

    def compute_avoidance(
        self,
        drone_position: Vector3,
        drone_velocity: Vector3,
    ) -> Vector3:
        """
        Compute the final safe velocity command.

        This is the main entry point called by the FlightController
        on every control tick.

        Pipeline:
            1. LiDAR → obstacles → APF
            2. APF computes velocity
            3. Check for STUCK → escalate to Tactical A*
            4. HPL gates the final command

        Args:
            drone_position: Current position (NED).
            drone_velocity: Current velocity (NED).

        Returns:
            Safe velocity command (NED).
        """
        if self._goal is None:
            return Vector3()

        # ── 1. Sensor Processing ────────────────────────────────
        obstacles = self._lidar.get_obstacles()
        self._apf.update_obstacles(obstacles)

        # ── 2. Determine effective goal ─────────────────────────
        effective_goal = self._get_effective_goal()

        # ── 3. APF velocity computation ─────────────────────────
        apf_velocity, apf_state = self._apf.compute(
            my_position=drone_position,
            my_velocity=drone_velocity,
            goal_position=effective_goal,
        )
        self._current_state = apf_state

        # ── 4. Tactical escalation (if stuck) ───────────────────
        if apf_state == AvoidanceState.STUCK:
            self._handle_stuck(drone_position, obstacles)

        elif apf_state in (AvoidanceState.CLEAR, AvoidanceState.MONITORING):
            self._stuck_start_time = None
            # Advance sub-waypoints if we've reached the current one
            if self._active_sub_waypoints and self._sub_waypoint_index < len(self._active_sub_waypoints):
                current_sub = self._active_sub_waypoints[self._sub_waypoint_index]
                if drone_position.distance_to(current_sub.position) < current_sub.acceptance_radius:
                    self._sub_waypoint_index += 1
                    if self._sub_waypoint_index >= len(self._active_sub_waypoints):
                        self._active_sub_waypoints.clear()
                        self._sub_waypoint_index = 0

        # ── 5. HPL safety gate ──────────────────────────────────
        sector_ranges = self._lidar.get_sector_ranges()
        self._hpl.update_scan(sector_ranges)

        safe_velocity, was_overridden = self._hpl.gate_command(
            apf_velocity, drone_position
        )
        self._hpl_overriding = was_overridden

        # ── 6. Threat broadcast (swarm) ─────────────────────────
        if was_overridden and self.config.broadcast_threats:
            self._broadcast_threat(drone_position, obstacles)

        self._current_velocity = safe_velocity
        return safe_velocity

    # ── Tactical Escalation ───────────────────────────────────────

    def _handle_stuck(self, drone_position: Vector3, obstacles: List[Obstacle3D]):
        """Handle STUCK state — escalate to A* if persistent."""
        now = time.time()

        if self._stuck_start_time is None:
            self._stuck_start_time = now
            return

        stuck_duration = now - self._stuck_start_time
        if stuck_duration < self.config.stuck_escalation_time:
            return  # Give APF more time

        # Check replan interval
        if now - self._last_replan_time < self.config.replan_interval:
            return

        if self._goal is None:
            return

        logger.warning(
            f"Drone {self.drone_id}: STUCK for {stuck_duration:.1f}s — "
            f"escalating to A* tactical planner"
        )

        # Feed obstacles to planner
        self._planner.update_obstacles(
            [obs.position for obs in obstacles],
            [obs.radius for obs in obstacles],
        )
        self._planner.update_costmap_origin(drone_position)

        # Plan detour
        sub_waypoints = self._planner.plan(drone_position, self._goal)
        if sub_waypoints:
            self._active_sub_waypoints = sub_waypoints
            self._sub_waypoint_index = 0
            self._stuck_start_time = None
            logger.info(
                f"Tactical plan accepted: {len(sub_waypoints)} sub-waypoints"
            )
        else:
            logger.warning("Tactical planner failed — APF will retry")

        self._last_replan_time = now

    def _get_effective_goal(self) -> Vector3:
        """
        Return the current effective goal (sub-waypoint or final goal).

        If we have active sub-waypoints from a tactical detour,
        use the current sub-waypoint.  Otherwise use the strategic goal.
        """
        if (
            self._active_sub_waypoints
            and self._sub_waypoint_index < len(self._active_sub_waypoints)
        ):
            return self._active_sub_waypoints[self._sub_waypoint_index].position

        return self._goal  # type: ignore[return-value]

    # ── Swarm Integration ─────────────────────────────────────────

    def _broadcast_threat(
        self,
        drone_position: Vector3,
        obstacles: List[Obstacle3D],
    ):
        """Broadcast threat data to swarm via mesh network."""
        if self._threat_broadcast_callback is None:
            return

        threat_data = {
            "drone_id": self.drone_id,
            "type": "obstacle_alert",
            "position": [drone_position.x, drone_position.y, drone_position.z],
            "obstacles": [
                {
                    "position": [o.position.x, o.position.y, o.position.z],
                    "radius": o.radius,
                    "confidence": o.confidence,
                }
                for o in obstacles[:10]  # Cap at 10 for bandwidth
            ],
            "hpl_state": self._hpl.state.name,
            "timestamp": time.time(),
        }

        try:
            self._threat_broadcast_callback(threat_data)
        except Exception as e:
            logger.error(f"Threat broadcast failed: {e}")

    def receive_swarm_threat(self, threat_data: dict):
        """
        Ingest threat data from another drone in the swarm.

        Adds remote obstacles to the local APF to enable
        proactive avoidance of threats not yet visible.
        """
        remote_obstacles = []
        for obs_dict in threat_data.get("obstacles", []):
            pos = obs_dict.get("position", [0, 0, 0])
            remote_obstacles.append(Obstacle3D(
                position=Vector3(x=pos[0], y=pos[1], z=pos[2]),
                radius=obs_dict.get("radius", 1.0),
                confidence=obs_dict.get("confidence", 0.5) * 0.7,  # Discount remote
            ))

        # Merge with local obstacles
        current = self._lidar.get_obstacles()
        self._apf.update_obstacles(current + remote_obstacles)

    # ── Telemetry ─────────────────────────────────────────────────

    def get_telemetry(self) -> Dict:
        """Get full avoidance subsystem telemetry."""
        return {
            "drone_id": self.drone_id,
            "avoidance_state": self._current_state.name,
            "hpl_state": self._hpl.state.name,
            "hpl_overriding": self._hpl_overriding,
            "closest_obstacle_m": round(self._apf.closest_obstacle_distance, 2),
            "active_sub_waypoints": len(self._active_sub_waypoints),
            "sub_waypoint_index": self._sub_waypoint_index,
            "velocity": [
                round(self._current_velocity.x, 3),
                round(self._current_velocity.y, 3),
                round(self._current_velocity.z, 3),
            ],
            "lidar": self._lidar.get_telemetry(),
            "apf": self._apf.get_telemetry(),
            "hpl": self._hpl.get_telemetry(),
            "timestamp": time.time(),
        }
