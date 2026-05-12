"""
GPS noise model with Gaussian position error and building multipath degradation.

Cheap GPS modules (u-blox NEO-6M class) have ~2.5m CEP in open sky.
Near tall buildings, multipath reflections increase CEP to 5-10m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3


@dataclass
class GPSConfig:
    horizontal_sigma_m: float = 2.5
    vertical_sigma_m: float = 4.0
    multipath_extra_sigma_m: float = 5.0
    multipath_building_radius_factor: float = 2.0
    update_rate_hz: float = 5.0
    hdop_base: float = 1.2
    hdop_multipath_max: float = 4.0
    velocity_noise_sigma_ms: float = 0.3
    glitch_probability_per_sec: float = 0.005
    glitch_offset_max_m: float = 15.0
    glitch_duration_sec: float = 0.5
    seed: Optional[int] = None


class GPSNoiseModel:
    def __init__(self, config: GPSConfig | None = None):
        self.config = config or GPSConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._update_interval = 1.0 / self.config.update_rate_hz
        self._time_since_update = 0.0
        self._last_noisy_pos: Optional[Vector3] = None
        self._glitch_remaining = 0.0
        self._glitch_offset = np.zeros(3)

    def _multipath_sigma(
        self,
        true_pos: Vector3,
        buildings: List[Tuple[Vector3, float]] | None,
    ) -> Tuple[float, float, float]:
        """Compute per-axis sigma based on building proximity."""
        h_sigma = self.config.horizontal_sigma_m
        v_sigma = self.config.vertical_sigma_m

        if buildings:
            max_extra = 0.0
            for center, width in buildings:
                dx = true_pos.x - center.x
                dy = true_pos.y - center.y
                dist = math.sqrt(dx * dx + dy * dy)
                threshold = width * self.config.multipath_building_radius_factor
                if dist < threshold:
                    closeness = 1.0 - (dist / threshold)
                    extra = closeness * self.config.multipath_extra_sigma_m
                    max_extra = max(max_extra, extra)
            h_sigma += max_extra
            v_sigma += max_extra * 0.7

        return h_sigma, h_sigma, v_sigma

    def get_hdop(
        self,
        true_pos: Vector3,
        buildings: List[Tuple[Vector3, float]] | None = None,
    ) -> float:
        """Horizontal dilution of precision (unitless, 1.0 = ideal)."""
        sx, _, _ = self._multipath_sigma(true_pos, buildings)
        ratio = sx / self.config.horizontal_sigma_m
        return min(self.config.hdop_base * ratio, self.config.hdop_multipath_max)

    def apply_noise(
        self,
        true_pos: Vector3,
        true_vel: Vector3,
        dt: float,
        buildings: List[Tuple[Vector3, float]] | None = None,
    ) -> Tuple[Vector3, Vector3]:
        """
        Returns (noisy_position, noisy_velocity).
        Respects GPS update rate — holds stale value between updates.
        """
        self._time_since_update += dt

        # GPS glitch (rare large offset)
        if self._glitch_remaining <= 0:
            if self._rng.random() < self.config.glitch_probability_per_sec * dt:
                angle = self._rng.uniform(0, 2 * math.pi)
                r = self._rng.uniform(5, self.config.glitch_offset_max_m)
                self._glitch_offset = np.array([
                    r * math.cos(angle), r * math.sin(angle), self._rng.uniform(-3, 3),
                ])
                self._glitch_remaining = self.config.glitch_duration_sec
        else:
            self._glitch_remaining -= dt
            if self._glitch_remaining <= 0:
                self._glitch_offset = np.zeros(3)

        if self._time_since_update >= self._update_interval:
            self._time_since_update = 0.0
            sx, sy, sz = self._multipath_sigma(true_pos, buildings)
            noise = np.array([
                self._rng.normal(0, sx),
                self._rng.normal(0, sy),
                self._rng.normal(0, sz),
            ]) + self._glitch_offset

            self._last_noisy_pos = Vector3(
                x=true_pos.x + noise[0],
                y=true_pos.y + noise[1],
                z=true_pos.z + noise[2],
            )

        noisy_pos = self._last_noisy_pos or true_pos

        vs = self.config.velocity_noise_sigma_ms
        noisy_vel = Vector3(
            x=true_vel.x + self._rng.normal(0, vs),
            y=true_vel.y + self._rng.normal(0, vs),
            z=true_vel.z + self._rng.normal(0, vs),
        )

        return noisy_pos, noisy_vel
