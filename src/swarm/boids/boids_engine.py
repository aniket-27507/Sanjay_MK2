"""
Project Sanjay Mk2 - Boids Engine
=================================
Decentralized Reynolds boids motion generator with task seeking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.types.drone_types import DroneState, Vector3

from .boids_config import BoidsConfig


@dataclass
class NormalizedObstacle:
    position: Vector3
    radius: float = 1.0


class BoidsEngine:
    """Computes boids steering vectors for autonomous swarm flight."""

    def __init__(self, config: Optional[BoidsConfig] = None):
        self.config = config or BoidsConfig()
        self._prev_velocity_by_drone: Dict[int, Vector3] = {}

    def compute(
        self,
        drone_id: int,
        states: Dict[int, DroneState],
        goal: Optional[Vector3],
        obstacles: Optional[Iterable[Any]] = None,
        formation_slot: Optional[Vector3] = None,
    ) -> Vector3:
        """Compute desired boids velocity for a single drone."""
        if drone_id not in states:
            return Vector3()

        my_state = states[drone_id]
        my_pos = my_state.position
        my_vel = my_state.velocity

        neighbors = [
            s
            for sid, s in states.items()
            if sid != drone_id
            and my_pos.distance_to(s.position) <= self.config.neighbor_radius
        ]

        v_sep = self._separation(my_pos, neighbors) * self.config.w_separation
        v_ali = self._alignment(my_vel, neighbors) * self.config.w_alignment
        v_coh = self._cohesion(my_pos, neighbors) * self.config.w_cohesion
        v_goal = self._seek(my_pos, goal) * self.config.w_goal_seeking
        v_obs = (
            self._avoid_obstacles(my_pos, obstacles or [])
            * self.config.w_obstacle_avoidance
        )
        v_form = (
            self._seek(my_pos, formation_slot) * self.config.w_formation_bias
            if formation_slot is not None
            else Vector3()
        )
        prev_velocity = self._prev_velocity_by_drone.get(drone_id, my_vel)
        v_energy = (
            self._energy_saving(my_vel, prev_velocity, goal, my_pos)
            * self.config.w_energy_saving
        )

        velocity = v_sep + v_ali + v_coh + v_goal + v_obs + v_form + v_energy
        velocity = self._clamp_velocity(velocity)

        self._prev_velocity_by_drone[drone_id] = velocity
        return velocity

    def compute_all(
        self,
        states: Dict[int, DroneState],
        goals: Dict[int, Vector3],
        obstacles: Optional[Iterable[Any]] = None,
        formation_slots: Optional[Dict[int, Vector3]] = None,
    ) -> Dict[int, Vector3]:
        """Compute boids velocity for all drones in one pass."""
        corrections: Dict[int, Vector3] = {}
        for drone_id in states:
            corrections[drone_id] = self.compute(
                drone_id=drone_id,
                states=states,
                goal=goals.get(drone_id),
                obstacles=obstacles,
                formation_slot=(formation_slots or {}).get(drone_id),
            )
        return corrections

    def _separation(self, my_pos: Vector3, neighbors: List[DroneState]) -> Vector3:
        force = Vector3()
        for n in neighbors:
            d = my_pos.distance_to(n.position)
            if 0.1 < d < self.config.min_separation:
                away = (my_pos - n.position).normalized()
                force = force + away * ((self.config.min_separation / d) ** 2)
        return force

    @staticmethod
    def _alignment(my_vel: Vector3, neighbors: List[DroneState]) -> Vector3:
        if not neighbors:
            return Vector3()

        avg = Vector3()
        for n in neighbors:
            avg = avg + n.velocity
        avg = avg / float(len(neighbors))
        return avg - my_vel

    @staticmethod
    def _cohesion(my_pos: Vector3, neighbors: List[DroneState]) -> Vector3:
        if not neighbors:
            return Vector3()

        center = Vector3()
        for n in neighbors:
            center = center + n.position
        center = center / float(len(neighbors))
        return (center - my_pos).normalized()

    @staticmethod
    def _seek(my_pos: Vector3, target: Optional[Vector3]) -> Vector3:
        if target is None:
            return Vector3()
        error = target - my_pos
        distance = error.magnitude()
        if distance < 1e-6:
            return Vector3()
        return error.normalized() * min(distance / 25.0, 1.0)

    def _energy_saving(
        self,
        velocity: Vector3,
        prev_velocity: Vector3,
        goal: Optional[Vector3],
        my_pos: Vector3,
    ) -> Vector3:
        accel = velocity - prev_velocity
        speed = velocity.magnitude()
        speed_error = self.config.cruise_speed - speed

        if speed > 1e-6:
            axis = velocity.normalized()
        elif goal is not None:
            axis = (goal - my_pos).normalized()
        else:
            axis = Vector3(1.0, 0.0, 0.0)

        return (
            axis * (speed_error * self.config.speed_convergence_rate)
            - accel * self.config.acceleration_penalty
        )

    def _avoid_obstacles(
        self,
        my_pos: Vector3,
        obstacles: Iterable[Any],
    ) -> Vector3:
        force = Vector3()
        for obstacle in self._normalize_obstacles(obstacles):
            to_me = my_pos - obstacle.position
            d_center = to_me.magnitude()
            d_surface = max(0.001, d_center - obstacle.radius)

            if d_surface > self.config.obstacle_detection_range:
                continue

            direction = to_me.normalized()
            strength = (self.config.obstacle_detection_range / d_surface) ** 2
            if d_surface < self.config.obstacle_safe_distance:
                strength *= 1.5

            force = force + direction * strength

        return force

    def _clamp_velocity(self, velocity: Vector3) -> Vector3:
        clamped_z = max(
            -self.config.max_vertical_speed,
            min(self.config.max_vertical_speed, velocity.z),
        )
        out = Vector3(velocity.x, velocity.y, clamped_z)

        speed = out.magnitude()
        if speed > self.config.max_speed and speed > 1e-6:
            scale = self.config.max_speed / speed
            out = out * scale

        return out

    @staticmethod
    def _normalize_obstacles(obstacles: Iterable[Any]) -> List[NormalizedObstacle]:
        normalized: List[NormalizedObstacle] = []
        for obs in obstacles:
            pos, radius = BoidsEngine._extract_obstacle(obs)
            if pos is not None:
                normalized.append(NormalizedObstacle(position=pos, radius=radius))
        return normalized

    @staticmethod
    def _extract_obstacle(obstacle: Any) -> Tuple[Optional[Vector3], float]:
        if isinstance(obstacle, Vector3):
            return obstacle, 1.0

        if hasattr(obstacle, "position"):
            pos = getattr(obstacle, "position")
            radius = float(getattr(obstacle, "radius", 1.0))
            if isinstance(pos, Vector3):
                return pos, radius

        if isinstance(obstacle, dict):
            raw_pos = obstacle.get("position")
            radius = float(obstacle.get("radius", 1.0))
            if isinstance(raw_pos, Vector3):
                return raw_pos, radius
            if isinstance(raw_pos, (list, tuple)) and len(raw_pos) >= 3:
                return Vector3(float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2])), radius

        return None, 1.0
