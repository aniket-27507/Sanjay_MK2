"""
Wind force model with base wind, gusts, Perlin-style turbulence,
and building-proximity vortex shedding.

Applies a force vector to each drone per tick based on:
- Constant base wind (direction + speed)
- Random gust events (Poisson-arrival, exponential decay)
- Perlin noise for smooth turbulence variation
- Building-proximity multiplier for urban canyon effects
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3


@dataclass
class WindConfig:
    base_speed_ms: float = 8.0
    base_direction_deg: float = 225.0
    gust_max_ms: float = 6.0
    gust_probability_per_sec: float = 0.15
    gust_decay_rate: float = 0.5
    turbulence_intensity: float = 0.3
    turbulence_scale: float = 50.0
    building_turbulence_multiplier: float = 2.5
    building_proximity_radius_factor: float = 2.0
    drag_coefficient: float = 0.12
    drone_cross_section_m2: float = 0.06
    drone_mass_kg: float = 0.5
    seed: Optional[int] = None


@dataclass
class _ActiveGust:
    direction: np.ndarray
    magnitude: float
    decay_rate: float
    age: float = 0.0


class WindModel:
    def __init__(self, config: WindConfig | None = None):
        self.config = config or WindConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._gusts: List[_ActiveGust] = []
        self._time = 0.0
        self._perm = self._rng.permutation(256).tolist()
        self._perm += self._perm

        dir_rad = math.radians(self.config.base_direction_deg)
        self._base_vec = np.array([
            math.cos(dir_rad) * self.config.base_speed_ms,
            math.sin(dir_rad) * self.config.base_speed_ms,
            0.0,
        ])

    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _grad1d(self, h: int, x: float) -> float:
        return x if (h & 1) == 0 else -x

    def _perlin_1d(self, x: float) -> float:
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = self._fade(xf)
        a = self._perm[xi]
        b = self._perm[xi + 1]
        return (1 - u) * self._grad1d(a, xf) + u * self._grad1d(b, xf - 1)

    def _turbulence_at(self, pos: np.ndarray, t: float) -> np.ndarray:
        scale = self.config.turbulence_scale
        intensity = self.config.turbulence_intensity
        tx = self._perlin_1d((pos[0] + t * 3.0) / scale) * intensity
        ty = self._perlin_1d((pos[1] + t * 3.7) / scale + 100) * intensity
        tz = self._perlin_1d((pos[2] + t * 2.3) / scale + 200) * intensity * 0.3
        return self._base_vec * np.array([tx, ty, tz])

    def _tick_gusts(self, dt: float) -> np.ndarray:
        if self._rng.random() < self.config.gust_probability_per_sec * dt:
            angle = self._rng.uniform(0, 2 * math.pi)
            elev = self._rng.uniform(-0.2, 0.2)
            direction = np.array([math.cos(angle), math.sin(angle), elev])
            magnitude = self._rng.uniform(1.0, self.config.gust_max_ms)
            self._gusts.append(_ActiveGust(
                direction=direction,
                magnitude=magnitude,
                decay_rate=self.config.gust_decay_rate,
            ))

        total = np.zeros(3)
        surviving: List[_ActiveGust] = []
        for g in self._gusts:
            g.age += dt
            strength = g.magnitude * math.exp(-g.decay_rate * g.age)
            if strength > 0.05:
                total += g.direction * strength
                surviving.append(g)
        self._gusts = surviving
        return total

    def building_proximity_factor(
        self,
        drone_pos: Vector3,
        buildings: List[Tuple[Vector3, float]],
    ) -> float:
        """
        Returns a wind multiplier >= 1.0 based on proximity to buildings.
        buildings: list of (center_pos, characteristic_width).
        """
        max_factor = 1.0
        for center, width in buildings:
            dx = drone_pos.x - center.x
            dy = drone_pos.y - center.y
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            proximity_radius = width * self.config.building_proximity_radius_factor
            if horiz_dist < proximity_radius:
                closeness = 1.0 - (horiz_dist / proximity_radius)
                factor = 1.0 + closeness * (self.config.building_turbulence_multiplier - 1.0)
                max_factor = max(max_factor, factor)
        return max_factor

    def compute_force(
        self,
        drone_pos: Vector3,
        drone_vel: Vector3,
        dt: float,
        buildings: List[Tuple[Vector3, float]] | None = None,
    ) -> Vector3:
        """
        Compute wind force on drone for this tick (Newtons, NED frame).
        """
        self._time += dt
        pos_arr = np.array([drone_pos.x, drone_pos.y, drone_pos.z])

        wind_velocity = (
            self._base_vec.copy()
            + self._turbulence_at(pos_arr, self._time)
            + self._tick_gusts(dt)
        )

        if buildings:
            factor = self.building_proximity_factor(drone_pos, buildings)
            wind_velocity *= factor

        drone_vel_arr = np.array([drone_vel.x, drone_vel.y, drone_vel.z])
        relative_wind = wind_velocity - drone_vel_arr

        # F = 0.5 * Cd * A * rho * v^2 * direction
        # rho ~1.15 kg/m³ for hot humid Guwahati air (computed by AtmosphereModel if available)
        rho = 1.15
        speed = float(np.linalg.norm(relative_wind))
        if speed < 0.01:
            return Vector3()

        force_magnitude = (
            0.5
            * self.config.drag_coefficient
            * self.config.drone_cross_section_m2
            * rho
            * speed * speed
        )

        direction = relative_wind / speed
        force = direction * force_magnitude
        return Vector3(x=float(force[0]), y=float(force[1]), z=float(force[2]))

    def compute_acceleration(
        self,
        drone_pos: Vector3,
        drone_vel: Vector3,
        dt: float,
        buildings: List[Tuple[Vector3, float]] | None = None,
    ) -> Vector3:
        """Convenience: returns acceleration (m/s²) instead of force."""
        f = self.compute_force(drone_pos, drone_vel, dt, buildings)
        m = self.config.drone_mass_kg
        return Vector3(x=f.x / m, y=f.y / m, z=f.z / m)
