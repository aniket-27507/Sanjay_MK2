"""
Project Sanjay Mk2 - 3D Artificial Potential Field (APF)
========================================================
Proprietary obstacle avoidance algorithm for Alpha Drones.

Extended APF operating in full 3D space.  Repulsive fields are
generated from a voxel-based 3D Occupancy Grid (compatible with
RTAB-Map OctoMap output) or raw obstacle point clusters.

Key features:
    - Quadratic attractive potential toward tactical waypoint
    - Exponential repulsive potential from occupied voxels / obstacles
    - Adaptive force weighting based on threat proximity
    - Local-minima escape via random-walk perturbation + wall-following
    - Altitude-preference bias (fly over obstacles when possible)
    - Velocity clamping for safe flight envelope

Mathematical Model:
    F_net = α · F_attractive  +  β · F_repulsive  +  γ · F_escape
    where α, β, γ are dynamically weighted each tick.

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════


class AvoidanceState(Enum):
    """Current state of the avoidance system."""
    CLEAR = auto()           # No obstacles in range
    MONITORING = auto()      # Obstacles detected but far
    AVOIDING = auto()        # Actively generating avoidance force
    STUCK = auto()           # Potential local minimum detected
    EMERGENCY = auto()       # Imminent collision — HPL territory


@dataclass
class Obstacle3D:
    """A 3D obstacle cluster with spatial extent."""
    position: Vector3                       # Centre of mass (NED)
    radius: float = 0.5                     # Bounding sphere radius (m)
    confidence: float = 1.0                 # Detection confidence [0-1]
    velocity: Vector3 = field(default_factory=Vector3)  # Estimated velocity
    timestamp: float = field(default_factory=time.time)

    @property
    def is_dynamic(self) -> bool:
        return self.velocity.magnitude() > 0.1


@dataclass
class OccupancyVoxel:
    """A single occupied voxel in the 3D occupancy grid."""
    x: float
    y: float
    z: float
    occupancy: float = 1.0  # Probability [0-1]
    size: float = 0.25      # Voxel edge length (m)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class APF3DConfig:
    """Configuration for the 3D Artificial Potential Field algorithm."""

    # ── Detection Ranges ──
    detection_range: float = 15.0           # Maximum sensor range (m)
    safe_distance: float = 6.0             # Gentle avoidance starts
    danger_zone: float = 2.5              # Strong avoidance
    critical_distance: float = 1.0         # HPL takeover threshold

    # ── Force Gains ──
    attractive_gain: float = 1.5           # α — pull toward waypoint
    repulsive_gain: float = 8.0            # β — push from obstacles
    escape_gain: float = 2.0              # γ — local-minima escape

    # ── Adaptive Weighting ──
    adaptive_weighting: bool = True        # Dynamically tune α/β
    min_attractive_weight: float = 0.2     # Floor for α in close proximity
    max_repulsive_weight: float = 3.0      # Ceiling multiplier for β

    # ── Velocity Limits ──
    max_avoidance_speed: float = 4.0       # m/s — output velocity cap
    max_vertical_speed: float = 2.5        # m/s — vertical axis cap

    # ── Altitude Preference ──
    altitude_bias: float = 0.4             # Fraction of repulsive force
                                           # redirected upward

    # ── Local-Minima Escape ──
    stuck_velocity_threshold: float = 0.3  # m/s — considered stuck
    stuck_duration_threshold: float = 2.0  # s — duration before escape
    escape_perturbation_mag: float = 2.0   # m/s — random kick magnitude
    wall_follow_gain: float = 1.5          # Wall-following force multiplier

    # ── Goal Clamping ──
    goal_slow_radius: float = 3.0          # Start decelerating toward goal
    goal_arrival_radius: float = 0.8       # Waypoint considered reached

    # ── OctoMap Integration ──
    voxel_query_radius: float = 12.0       # Query radius around drone
    voxel_batch_size: int = 512            # Max voxels per force calc


# ═══════════════════════════════════════════════════════════════════
#  3D APF Core Algorithm
# ═══════════════════════════════════════════════════════════════════


class APF3DAvoidance:
    """
    3D Artificial Potential Field obstacle avoidance for Alpha Drones.

    This is the core obstacle avoidance brain.  It computes a velocity
    command that steers the drone toward its tactical waypoint while
    generating repulsive forces from all sensed obstacles.

    The algorithm runs at the tactical control rate (≥20 Hz) and outputs
    a velocity vector in the NED frame that is handed to the
    FlightController or, in emergency, overridden by the HPL.

    Usage:
        apf = APF3DAvoidance()

        # Feed obstacles (from LiDAR clustering or OctoMap)
        apf.update_obstacles(obstacle_list)

        # Every control tick
        velocity, state = apf.compute(
            my_position=drone_pos,
            my_velocity=drone_vel,
            goal_position=waypoint,
        )

        # velocity → send to FlightController
        # state   → AvoidanceState enum for telemetry
    """

    def __init__(self, config: Optional[APF3DConfig] = None):
        self.config = config or APF3DConfig()
        self._obstacles: List[Obstacle3D] = []
        self._voxels: List[OccupancyVoxel] = []

        # State tracking
        self._state = AvoidanceState.CLEAR
        self._low_speed_start: Optional[float] = None
        self._escape_direction: Optional[np.ndarray] = None
        self._last_positions: List[np.ndarray] = []

        # Telemetry
        self._last_attractive = Vector3()
        self._last_repulsive = Vector3()
        self._closest_obstacle_dist: float = float("inf")

    # ── Public Interface ──────────────────────────────────────────

    @property
    def state(self) -> AvoidanceState:
        return self._state

    @property
    def closest_obstacle_distance(self) -> float:
        return self._closest_obstacle_dist

    def update_obstacles(self, obstacles: List[Obstacle3D]):
        """Feed clustered obstacles from LiDAR / depth fusion."""
        self._obstacles = obstacles

    def update_voxels(self, voxels: List[OccupancyVoxel]):
        """Feed occupied voxels from OctoMap / RTAB-Map."""
        self._voxels = voxels

    def compute(
        self,
        my_position: Vector3,
        my_velocity: Vector3,
        goal_position: Vector3,
    ) -> Tuple[Vector3, AvoidanceState]:
        """
        Compute the net APF velocity command.

        Args:
            my_position: Drone position (NED).
            my_velocity: Drone velocity (NED).
            goal_position: Current tactical waypoint (NED).

        Returns:
            (velocity_command, avoidance_state)
        """
        pos = my_position.to_array()
        vel = my_velocity.to_array()
        goal = goal_position.to_array()

        # ── 1. Closest obstacle distance ──
        self._closest_obstacle_dist = self._compute_min_distance(pos)

        # ── 2. Determine state ──
        self._update_state(pos, vel)

        # ── 3. Emergency — hand off to HPL ──
        if self._closest_obstacle_dist < self.config.critical_distance:
            self._state = AvoidanceState.EMERGENCY
            return Vector3(), AvoidanceState.EMERGENCY

        # ── 4. Attractive force ──────────────────────────────────
        f_attract = self._attractive_force(pos, goal)

        # ── 5. Repulsive force ───────────────────────────────────
        f_repulse = self._repulsive_force(pos)

        # ── 6. Escape force (local-minima) ───────────────────────
        f_escape = np.zeros(3)
        if self._state == AvoidanceState.STUCK:
            f_escape = self._escape_force(pos, goal)

        # ── 7. Adaptive weighting ────────────────────────────────
        alpha, beta, gamma = self._adaptive_weights()

        # ── 8. Net force → velocity ──────────────────────────────
        net = alpha * f_attract + beta * f_repulse + gamma * f_escape

        # Altitude bias: redirect a fraction of repulsive upward
        repulsive_mag = np.linalg.norm(f_repulse)
        if repulsive_mag > 0.1 and self.config.altitude_bias > 0:
            net[2] -= self.config.altitude_bias * repulsive_mag  # NED: -z = up

        # Clamp speed
        net = self._clamp_velocity(net)

        # ── 9. Store telemetry ───────────────────────────────────
        self._last_attractive = Vector3.from_array(f_attract)
        self._last_repulsive = Vector3.from_array(f_repulse)

        return Vector3.from_array(net), self._state

    # ── Force Calculations ────────────────────────────────────────

    def _attractive_force(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Quadratic attractive potential toward the goal."""
        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)

        if dist < self.config.goal_arrival_radius:
            return np.zeros(3)

        direction = to_goal / max(dist, 1e-6)

        # Proportional magnitude, capped for stability
        if dist < self.config.goal_slow_radius:
            magnitude = self.config.attractive_gain * dist
        else:
            magnitude = self.config.attractive_gain * self.config.goal_slow_radius

        return direction * magnitude

    def _repulsive_force(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute total repulsive force from all obstacle sources.

        Combines both cluster-level obstacles and raw voxels.
        """
        total = np.zeros(3)

        # ── Obstacle clusters ──
        for obs in self._obstacles:
            force = self._single_repulsive(pos, obs.position.to_array(), obs.radius)
            total += force * obs.confidence

        # ── Occupied voxels (batched) ──
        if self._voxels:
            sorted_voxels = sorted(
                self._voxels,
                key=lambda v: np.linalg.norm(v.center - pos),
            )
            for voxel in sorted_voxels[: self.config.voxel_batch_size]:
                force = self._single_repulsive(pos, voxel.center, voxel.size / 2)
                total += force * voxel.occupancy

        return total

    def _single_repulsive(
        self,
        pos: np.ndarray,
        obs_center: np.ndarray,
        obs_radius: float,
    ) -> np.ndarray:
        """Compute repulsive force from a single object/voxel."""
        to_drone = pos - obs_center
        dist = np.linalg.norm(to_drone)
        effective = max(dist - obs_radius, 0.05)

        if effective > self.config.detection_range:
            return np.zeros(3)

        direction = to_drone / max(dist, 1e-6)

        # Three-zone force model
        if effective < self.config.danger_zone:
            # Exponential close-range
            magnitude = self.config.repulsive_gain * (
                (1.0 / effective) - (1.0 / self.config.danger_zone)
            ) * (1.0 / effective**2)
        elif effective < self.config.safe_distance:
            # Linear mid-range
            magnitude = self.config.repulsive_gain * (
                (self.config.safe_distance - effective)
                / self.config.safe_distance**2
            )
        else:
            # Weak long-range awareness
            magnitude = self.config.repulsive_gain * 0.05 * (
                (self.config.detection_range - effective)
                / self.config.detection_range**2
            )

        return direction * magnitude

    def _escape_force(self, pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Generate a perturbation force to escape local minima.

        Strategy: combine random lateral perturbation with a wall-following
        component perpendicular to the obstacle gradient.
        """
        if self._escape_direction is None:
            # Pick a random lateral direction perpendicular to goal direction
            to_goal = goal - pos
            to_goal_norm = to_goal / max(np.linalg.norm(to_goal), 1e-6)

            # Random vector
            rand = np.random.randn(3)
            rand[2] *= 0.3  # Dampen vertical randomness

            # Make perpendicular to goal direction (Gram-Schmidt)
            rand = rand - np.dot(rand, to_goal_norm) * to_goal_norm
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-6:
                rand = rand / rand_norm
            else:
                rand = np.array([0.0, 1.0, 0.0])

            self._escape_direction = rand

        escape = self._escape_direction * self.config.escape_perturbation_mag

        # Wall-following: steer perpendicular to closest obstacle gradient
        closest_obs = self._get_closest_obstacle(pos)
        if closest_obs is not None:
            gradient = pos - closest_obs
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1e-6:
                gradient = gradient / gradient_norm
                # perpendicular in XY
                perp = np.array([-gradient[1], gradient[0], 0.0])
                escape += perp * self.config.wall_follow_gain

        return escape

    # ── State Management ──────────────────────────────────────────

    def _update_state(self, pos: np.ndarray, vel: np.ndarray):
        """Determine the current avoidance state."""
        speed = np.linalg.norm(vel)
        dist = self._closest_obstacle_dist

        if dist > self.config.safe_distance:
            self._state = AvoidanceState.CLEAR
            self._low_speed_start = None
            self._escape_direction = None
            return

        if dist > self.config.danger_zone:
            self._state = AvoidanceState.MONITORING
            self._low_speed_start = None
            self._escape_direction = None
            return

        # We are actively avoiding
        self._state = AvoidanceState.AVOIDING

        # Check for stuck condition
        if speed < self.config.stuck_velocity_threshold:
            if self._low_speed_start is None:
                self._low_speed_start = time.time()
            elif time.time() - self._low_speed_start > self.config.stuck_duration_threshold:
                self._state = AvoidanceState.STUCK
        else:
            self._low_speed_start = None
            self._escape_direction = None  # Reset escape direction

    def _adaptive_weights(self) -> Tuple[float, float, float]:
        """
        Dynamically adjust α, β, γ based on proximity threat.

        The closer an obstacle, the more weight on repulsive forces
        and less on attractive (to prevent oscillation).
        """
        if not self.config.adaptive_weighting:
            return (
                self.config.attractive_gain,
                self.config.repulsive_gain,
                self.config.escape_gain if self._state == AvoidanceState.STUCK else 0.0,
            )

        dist = self._closest_obstacle_dist
        safe = self.config.safe_distance

        if dist >= safe:
            alpha = self.config.attractive_gain
            beta = self.config.repulsive_gain
        else:
            # Linear blend: closer → more repulsive, less attractive
            ratio = max(dist / safe, 0.0)
            alpha = max(
                self.config.attractive_gain * ratio,
                self.config.min_attractive_weight,
            )
            beta = self.config.repulsive_gain * min(
                1.0 + (1.0 - ratio) * 2.0,
                self.config.max_repulsive_weight,
            )

        gamma = self.config.escape_gain if self._state == AvoidanceState.STUCK else 0.0

        return alpha, beta, gamma

    # ── Helpers ────────────────────────────────────────────────────

    def _compute_min_distance(self, pos: np.ndarray) -> float:
        """Compute distance to closest obstacle from any source."""
        min_d = float("inf")

        for obs in self._obstacles:
            d = np.linalg.norm(pos - obs.position.to_array()) - obs.radius
            min_d = min(min_d, max(d, 0.0))

        for voxel in self._voxels:
            d = np.linalg.norm(pos - voxel.center) - voxel.size / 2
            min_d = min(min_d, max(d, 0.0))

        return min_d

    def _get_closest_obstacle(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """Get the position of the closest obstacle."""
        min_d = float("inf")
        closest = None

        for obs in self._obstacles:
            d = np.linalg.norm(pos - obs.position.to_array())
            if d < min_d:
                min_d = d
                closest = obs.position.to_array()

        return closest

    def _clamp_velocity(self, vel: np.ndarray) -> np.ndarray:
        """Clamp velocity to safe flight envelope."""
        # Horizontal speed
        horiz = np.linalg.norm(vel[:2])
        if horiz > self.config.max_avoidance_speed:
            vel[:2] *= self.config.max_avoidance_speed / horiz

        # Vertical speed
        if abs(vel[2]) > self.config.max_vertical_speed:
            vel[2] = math.copysign(self.config.max_vertical_speed, vel[2])

        return vel

    # ── Telemetry ─────────────────────────────────────────────────

    def get_telemetry(self) -> Dict:
        """Return telemetry snapshot for logging / network broadcast."""
        return {
            "state": self._state.name,
            "closest_obstacle_m": round(self._closest_obstacle_dist, 2),
            "attractive_force": [
                round(self._last_attractive.x, 3),
                round(self._last_attractive.y, 3),
                round(self._last_attractive.z, 3),
            ],
            "repulsive_force": [
                round(self._last_repulsive.x, 3),
                round(self._last_repulsive.y, 3),
                round(self._last_repulsive.z, 3),
            ],
            "num_obstacles": len(self._obstacles),
            "num_voxels": len(self._voxels),
            "timestamp": time.time(),
        }
