"""
Project Sanjay Mk2 - Boids Engine
=================================
Decentralized Reynolds boids motion generator with task seeking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

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
        my_arr = np.array([my_pos.x, my_pos.y, my_pos.z])
        my_vel_arr = np.array([my_vel.x, my_vel.y, my_vel.z])

        peer_ids = [sid for sid in states if sid != drone_id]
        if peer_ids:
            peer_pos = np.array([[states[s].position.x, states[s].position.y, states[s].position.z] for s in peer_ids])
            peer_vel = np.array([[states[s].velocity.x, states[s].velocity.y, states[s].velocity.z] for s in peer_ids])
            dists = np.linalg.norm(peer_pos - my_arr, axis=1)
            in_range = dists <= self.config.neighbor_radius
            n_pos = peer_pos[in_range]
            n_vel = peer_vel[in_range]
            n_dists = dists[in_range]
        else:
            n_pos = np.empty((0, 3))
            n_vel = np.empty((0, 3))
            n_dists = np.empty(0)

        v_sep = self._separation_vec(my_arr, n_pos, n_dists) * self.config.w_separation
        v_ali = self._alignment_vec(my_vel_arr, n_vel) * self.config.w_alignment
        v_coh = self._cohesion_vec(my_arr, n_pos) * self.config.w_cohesion
        v_goal = self._seek_vec(my_arr, goal) * self.config.w_goal_seeking
        v_obs = (
            self._avoid_obstacles_vec(my_arr, obstacles or [])
            * self.config.w_obstacle_avoidance
        )
        v_form = (
            self._seek_vec(my_arr, formation_slot) * self.config.w_formation_bias
            if formation_slot is not None
            else np.zeros(3)
        )
        prev = self._prev_velocity_by_drone.get(drone_id, my_vel)
        prev_arr = np.array([prev.x, prev.y, prev.z])
        v_energy = (
            self._energy_saving_vec(my_vel_arr, prev_arr, goal, my_arr)
            * self.config.w_energy_saving
        )

        vel = v_sep + v_ali + v_coh + v_goal + v_obs + v_form + v_energy
        velocity = self._clamp_velocity_vec(vel)

        result = Vector3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        self._prev_velocity_by_drone[drone_id] = result
        return result

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

    # ── Vectorized force methods (operate on numpy arrays) ──────

    def _separation_vec(self, my_pos: np.ndarray, n_pos: np.ndarray, n_dists: np.ndarray) -> np.ndarray:
        if n_pos.shape[0] == 0:
            return np.zeros(3)
        close = (n_dists > 0.1) & (n_dists < self.config.min_separation)
        if not np.any(close):
            return np.zeros(3)
        diffs = my_pos - n_pos[close]
        d = n_dists[close, np.newaxis]
        away = diffs / np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 1e-9)
        weights = (self.config.min_separation / d) ** 2
        return np.sum(away * weights, axis=0)

    @staticmethod
    def _alignment_vec(my_vel: np.ndarray, n_vel: np.ndarray) -> np.ndarray:
        if n_vel.shape[0] == 0:
            return np.zeros(3)
        return n_vel.mean(axis=0) - my_vel

    @staticmethod
    def _cohesion_vec(my_pos: np.ndarray, n_pos: np.ndarray) -> np.ndarray:
        if n_pos.shape[0] == 0:
            return np.zeros(3)
        center = n_pos.mean(axis=0)
        diff = center - my_pos
        mag = np.linalg.norm(diff)
        return diff / mag if mag > 1e-9 else np.zeros(3)

    @staticmethod
    def _seek_vec(my_pos: np.ndarray, target: Optional[Vector3]) -> np.ndarray:
        if target is None:
            return np.zeros(3)
        t = np.array([target.x, target.y, target.z])
        error = t - my_pos
        dist = np.linalg.norm(error)
        if dist < 1e-6:
            return np.zeros(3)
        return (error / dist) * min(dist / 25.0, 1.0)

    def _energy_saving_vec(
        self,
        velocity: np.ndarray,
        prev_velocity: np.ndarray,
        goal: Optional[Vector3],
        my_pos: np.ndarray,
    ) -> np.ndarray:
        accel = velocity - prev_velocity
        speed = np.linalg.norm(velocity)
        speed_error = self.config.cruise_speed - speed

        if speed > 1e-6:
            axis = velocity / speed
        elif goal is not None:
            g = np.array([goal.x, goal.y, goal.z])
            d = g - my_pos
            m = np.linalg.norm(d)
            axis = d / m if m > 1e-6 else np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([1.0, 0.0, 0.0])

        return axis * (speed_error * self.config.speed_convergence_rate) - accel * self.config.acceleration_penalty

    def _avoid_obstacles_vec(self, my_pos: np.ndarray, obstacles: Iterable[Any]) -> np.ndarray:
        normalized = self._normalize_obstacles(obstacles)
        if not normalized:
            return np.zeros(3)
        obs_pos = np.array([[o.position.x, o.position.y, o.position.z] for o in normalized])
        obs_rad = np.array([o.radius for o in normalized])

        to_me = my_pos - obs_pos
        d_center = np.linalg.norm(to_me, axis=1)
        d_surface = np.maximum(0.001, d_center - obs_rad)

        in_range = d_surface <= self.config.obstacle_detection_range
        if not np.any(in_range):
            return np.zeros(3)

        to_me = to_me[in_range]
        d_s = d_surface[in_range, np.newaxis]
        d_c = d_center[in_range, np.newaxis]
        directions = to_me / np.maximum(d_c, 1e-9)
        strength = (self.config.obstacle_detection_range / d_s) ** 2
        safe_mask = d_s.ravel() < self.config.obstacle_safe_distance
        strength[safe_mask] *= 1.5
        return np.sum(directions * strength, axis=0)

    def _clamp_velocity_vec(self, velocity: np.ndarray) -> np.ndarray:
        out = velocity.copy()
        out[2] = np.clip(out[2], -self.config.max_vertical_speed, self.config.max_vertical_speed)
        speed = np.linalg.norm(out)
        if speed > self.config.max_speed and speed > 1e-6:
            out *= self.config.max_speed / speed
        return out

    # ── Legacy Vector3-based methods (used by compute_all) ────

    def _separation(self, my_pos: Vector3, neighbors: List[DroneState]) -> Vector3:
        if not neighbors:
            return Vector3()
        n_pos = np.array([[n.position.x, n.position.y, n.position.z] for n in neighbors])
        my = np.array([my_pos.x, my_pos.y, my_pos.z])
        d = np.linalg.norm(n_pos - my, axis=1)
        r = self._separation_vec(my, n_pos, d)
        return Vector3(float(r[0]), float(r[1]), float(r[2]))

    @staticmethod
    def _alignment(my_vel: Vector3, neighbors: List[DroneState]) -> Vector3:
        if not neighbors:
            return Vector3()
        n_vel = np.array([[n.velocity.x, n.velocity.y, n.velocity.z] for n in neighbors])
        r = BoidsEngine._alignment_vec(np.array([my_vel.x, my_vel.y, my_vel.z]), n_vel)
        return Vector3(float(r[0]), float(r[1]), float(r[2]))

    @staticmethod
    def _cohesion(my_pos: Vector3, neighbors: List[DroneState]) -> Vector3:
        if not neighbors:
            return Vector3()
        n_pos = np.array([[n.position.x, n.position.y, n.position.z] for n in neighbors])
        r = BoidsEngine._cohesion_vec(np.array([my_pos.x, my_pos.y, my_pos.z]), n_pos)
        return Vector3(float(r[0]), float(r[1]), float(r[2]))

    @staticmethod
    def _seek(my_pos: Vector3, target: Optional[Vector3]) -> Vector3:
        r = BoidsEngine._seek_vec(np.array([my_pos.x, my_pos.y, my_pos.z]), target)
        return Vector3(float(r[0]), float(r[1]), float(r[2]))

    def _clamp_velocity(self, velocity: Vector3) -> Vector3:
        arr = self._clamp_velocity_vec(np.array([velocity.x, velocity.y, velocity.z]))
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

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
