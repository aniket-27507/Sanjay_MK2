"""
LiDAR noise model for cheap 2D LiDAR (RPLiDAR A1 class) on servo.

Models:
- Range noise (Gaussian, increases with distance)
- False returns from rain, dust, or glass reflections
- Missed returns (dropout) from dark/absorptive surfaces
- Angular accuracy noise
- Servo position noise and backlash
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3


@dataclass
class LiDARNoiseConfig:
    range_noise_base_m: float = 0.02
    range_noise_per_meter: float = 0.005
    angular_noise_deg: float = 0.3
    false_return_probability: float = 0.02
    false_return_range_max_m: float = 12.0
    dropout_probability: float = 0.03
    dropout_dark_surface_extra: float = 0.05
    rain_mode: bool = False
    rain_false_return_boost: float = 0.10
    rain_dropout_boost: float = 0.08
    dust_mode: bool = False
    dust_false_return_boost: float = 0.06
    servo_backlash_deg: float = 0.5
    servo_position_noise_deg: float = 0.2
    max_range_m: float = 12.0
    min_range_m: float = 0.15
    seed: Optional[int] = None


@dataclass
class NoisyLiDARReturn:
    angle_deg: float
    range_m: float
    is_valid: bool
    is_false_return: bool = False


class LiDARNoiseModel:
    def __init__(self, config: LiDARNoiseConfig | None = None):
        self.config = config or LiDARNoiseConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._last_servo_angle = 0.0

    def _false_return_prob(self) -> float:
        p = self.config.false_return_probability
        if self.config.rain_mode:
            p += self.config.rain_false_return_boost
        if self.config.dust_mode:
            p += self.config.dust_false_return_boost
        return min(p, 0.5)

    def _dropout_prob(self) -> float:
        p = self.config.dropout_probability
        if self.config.rain_mode:
            p += self.config.rain_dropout_boost
        if self.config.dust_mode:
            p += self.config.dust_false_return_boost * 0.5
        return min(p, 0.5)

    def apply_servo_noise(self, commanded_angle_deg: float) -> float:
        """Apply servo backlash and position noise to tilt angle."""
        direction = 1.0 if commanded_angle_deg >= self._last_servo_angle else -1.0
        backlash = direction * self.config.servo_backlash_deg * self._rng.uniform(0, 0.5)
        position_noise = self._rng.normal(0, self.config.servo_position_noise_deg)
        actual_angle = commanded_angle_deg + backlash + position_noise
        self._last_servo_angle = commanded_angle_deg
        return actual_angle

    def apply_range_noise(self, true_range_m: float) -> NoisyLiDARReturn:
        """
        Apply noise to a single LiDAR range measurement.
        Returns a NoisyLiDARReturn with possibly invalid or false data.
        """
        if true_range_m < self.config.min_range_m:
            return NoisyLiDARReturn(angle_deg=0, range_m=0, is_valid=False)

        if true_range_m > self.config.max_range_m:
            if self._rng.random() < self._false_return_prob():
                fake_range = self._rng.uniform(
                    self.config.min_range_m, self.config.false_return_range_max_m,
                )
                return NoisyLiDARReturn(
                    angle_deg=0, range_m=fake_range, is_valid=True, is_false_return=True,
                )
            return NoisyLiDARReturn(angle_deg=0, range_m=0, is_valid=False)

        if self._rng.random() < self._dropout_prob():
            return NoisyLiDARReturn(angle_deg=0, range_m=0, is_valid=False)

        if self._rng.random() < self._false_return_prob():
            fake_range = self._rng.uniform(self.config.min_range_m, true_range_m * 0.8)
            return NoisyLiDARReturn(
                angle_deg=0, range_m=fake_range, is_valid=True, is_false_return=True,
            )

        sigma = self.config.range_noise_base_m + self.config.range_noise_per_meter * true_range_m
        noisy_range = true_range_m + self._rng.normal(0, sigma)
        noisy_range = max(self.config.min_range_m, min(self.config.max_range_m, noisy_range))

        return NoisyLiDARReturn(angle_deg=0, range_m=noisy_range, is_valid=True)

    def apply_scan_noise(
        self,
        true_ranges: List[Tuple[float, float]],
        servo_angle_deg: float,
    ) -> List[NoisyLiDARReturn]:
        """
        Apply noise to a full 360° scan ring.
        true_ranges: list of (angle_deg, range_m) for each ray in the scan.
        servo_angle_deg: commanded servo tilt angle.
        Returns list of noisy returns.
        """
        actual_servo = self.apply_servo_noise(servo_angle_deg)
        results = []
        for angle_deg, true_range in true_ranges:
            noisy_angle = angle_deg + self._rng.normal(0, self.config.angular_noise_deg)
            ret = self.apply_range_noise(true_range)
            ret.angle_deg = noisy_angle
            results.append(ret)
        return results
