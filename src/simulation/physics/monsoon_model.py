"""
Monsoon weather model for Guwahati June conditions.

Models rain events and their cascading effects on drone operations:
- Rain intensity (mm/hr) with sudden burst patterns typical of pre-monsoon
- LiDAR degradation: false returns from raindrops, range reduction
- Prop efficiency loss: wet air + water ingress on cheap frames
- Visibility reduction: affects camera-based detection range
- Wind intensification during rain bursts
- Temperature drop during rainfall

Guwahati June climate:
  Average rainfall: 315mm/month, ~10.5mm/day
  Rain pattern: intense bursts (50-100mm/hr) lasting 15-45 min
  Between bursts: drizzle or dry, high humidity (85-95%)
  Pre-monsoon thunderstorms: sudden onset, heavy rain + wind
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np


class RainIntensity(Enum):
    DRY = auto()           # 0 mm/hr
    DRIZZLE = auto()       # 0.5-2 mm/hr
    LIGHT = auto()         # 2-10 mm/hr
    MODERATE = auto()      # 10-30 mm/hr
    HEAVY = auto()         # 30-60 mm/hr
    TORRENTIAL = auto()    # 60-100+ mm/hr (pre-monsoon burst)


@dataclass
class MonsoonConfig:
    # Rain event parameters
    initial_intensity: RainIntensity = RainIntensity.DRY
    burst_probability_per_min: float = 0.02
    burst_duration_range_sec: tuple = (300, 1800)
    burst_peak_mmhr: float = 75.0
    drizzle_background_mmhr: float = 1.5

    # Effects on sensors
    lidar_false_return_rate_per_mmhr: float = 0.005
    lidar_range_reduction_pct_per_mmhr: float = 0.3
    lidar_dropout_rate_per_mmhr: float = 0.003
    visibility_reduction_pct_per_mmhr: float = 0.8
    camera_detection_range_floor_pct: float = 20.0

    # Effects on flight
    prop_efficiency_loss_per_mmhr: float = 0.002
    battery_drain_increase_per_mmhr: float = 0.003
    wind_gust_multiplier_in_rain: float = 1.5
    temp_drop_per_mmhr: float = 0.05

    seed: Optional[int] = None


@dataclass
class MonsoonState:
    current_intensity_mmhr: float = 0.0
    rain_category: RainIntensity = RainIntensity.DRY
    visibility_pct: float = 100.0
    lidar_false_return_boost: float = 0.0
    lidar_dropout_boost: float = 0.0
    lidar_range_multiplier: float = 1.0
    prop_efficiency_multiplier: float = 1.0
    battery_drain_multiplier: float = 1.0
    wind_gust_multiplier: float = 1.0
    temp_offset_c: float = 0.0
    is_raining: bool = False


class MonsoonModel:
    """
    Stochastic monsoon rain model with cascading sensor/flight effects.

    Rain events arrive as Poisson process, ramp up over 30-60s,
    hold at peak for configurable duration, then decay.
    Between events, background drizzle is possible.
    """

    def __init__(self, config: MonsoonConfig | None = None):
        self.config = config or MonsoonConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._time = 0.0

        self._in_burst = False
        self._burst_start = 0.0
        self._burst_duration = 0.0
        self._burst_peak = 0.0
        self._burst_ramp_sec = 45.0

        if self.config.initial_intensity != RainIntensity.DRY:
            self._trigger_burst()

    def _trigger_burst(self) -> None:
        self._in_burst = True
        self._burst_start = self._time
        lo, hi = self.config.burst_duration_range_sec
        self._burst_duration = self._rng.uniform(lo, hi)
        self._burst_peak = self._rng.uniform(
            self.config.burst_peak_mmhr * 0.5,
            self.config.burst_peak_mmhr,
        )
        self._burst_ramp_sec = self._rng.uniform(30.0, 90.0)

    def _compute_rain_intensity(self) -> float:
        if not self._in_burst:
            if self._rng.random() < 0.3:
                return self.config.drizzle_background_mmhr * self._rng.uniform(0.2, 1.0)
            return 0.0

        elapsed = self._time - self._burst_start
        ramp = self._burst_ramp_sec

        if elapsed < ramp:
            # Ramp up (sigmoid-ish)
            t = elapsed / ramp
            intensity = self._burst_peak * (3 * t * t - 2 * t * t * t)
        elif elapsed < self._burst_duration - ramp:
            # Sustained peak with noise
            intensity = self._burst_peak * self._rng.uniform(0.7, 1.0)
        elif elapsed < self._burst_duration:
            # Ramp down
            remaining = (self._burst_duration - elapsed) / ramp
            intensity = self._burst_peak * remaining
        else:
            self._in_burst = False
            intensity = 0.0

        return max(0.0, intensity)

    def _classify_intensity(self, mmhr: float) -> RainIntensity:
        if mmhr < 0.1:
            return RainIntensity.DRY
        elif mmhr < 2.0:
            return RainIntensity.DRIZZLE
        elif mmhr < 10.0:
            return RainIntensity.LIGHT
        elif mmhr < 30.0:
            return RainIntensity.MODERATE
        elif mmhr < 60.0:
            return RainIntensity.HEAVY
        else:
            return RainIntensity.TORRENTIAL

    def tick(self, dt: float) -> MonsoonState:
        self._time += dt
        cfg = self.config

        # Check for new burst
        if not self._in_burst:
            prob = cfg.burst_probability_per_min * (dt / 60.0)
            if self._rng.random() < prob:
                self._trigger_burst()

        intensity = self._compute_rain_intensity()
        category = self._classify_intensity(intensity)

        # Cascading effects
        vis = max(
            cfg.camera_detection_range_floor_pct,
            100.0 - intensity * cfg.visibility_reduction_pct_per_mmhr,
        )

        lidar_false = intensity * cfg.lidar_false_return_rate_per_mmhr
        lidar_drop = intensity * cfg.lidar_dropout_rate_per_mmhr
        lidar_range = max(0.3, 1.0 - intensity * cfg.lidar_range_reduction_pct_per_mmhr / 100.0)

        prop_eff = max(0.7, 1.0 - intensity * cfg.prop_efficiency_loss_per_mmhr)
        batt_drain = 1.0 + intensity * cfg.battery_drain_increase_per_mmhr

        wind_mult = 1.0
        if intensity > 5.0:
            wind_mult = 1.0 + (intensity / cfg.burst_peak_mmhr) * (cfg.wind_gust_multiplier_in_rain - 1.0)

        temp_off = -intensity * cfg.temp_drop_per_mmhr

        return MonsoonState(
            current_intensity_mmhr=intensity,
            rain_category=category,
            visibility_pct=vis,
            lidar_false_return_boost=lidar_false,
            lidar_dropout_boost=lidar_drop,
            lidar_range_multiplier=lidar_range,
            prop_efficiency_multiplier=prop_eff,
            battery_drain_multiplier=batt_drain,
            wind_gust_multiplier=wind_mult,
            temp_offset_c=temp_off,
            is_raining=intensity > 0.1,
        )
