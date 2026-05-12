"""
Temperature-adjusted Li-Po battery model for cheap drone frames.

Models:
- Non-linear discharge curve (Li-Po characteristic)
- Temperature-dependent capacity derating
- Voltage sag under load (current draw from thrust)
- RTL trigger at configurable threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BatteryConfig:
    capacity_mah: float = 2200.0
    cell_count: int = 3
    nominal_voltage_per_cell: float = 3.7
    full_voltage_per_cell: float = 4.2
    empty_voltage_per_cell: float = 3.3
    critical_voltage_per_cell: float = 3.0
    hover_current_a: float = 8.0
    max_current_a: float = 25.0
    internal_resistance_ohm: float = 0.08
    temp_nominal_c: float = 25.0
    temp_capacity_coeff: float = -0.005
    rtl_threshold_pct: float = 20.0
    critical_threshold_pct: float = 10.0
    seed: Optional[int] = None


class BatteryModel:
    def __init__(self, config: BatteryConfig | None = None):
        self.config = config or BatteryConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._capacity_remaining_mah = self.config.capacity_mah
        self._cycle_count = 0

    @property
    def soc_pct(self) -> float:
        """State of charge (0-100%)."""
        return max(0.0, min(100.0,
            (self._capacity_remaining_mah / self.config.capacity_mah) * 100.0))

    @property
    def should_rtl(self) -> bool:
        return self.soc_pct <= self.config.rtl_threshold_pct

    @property
    def is_critical(self) -> bool:
        return self.soc_pct <= self.config.critical_threshold_pct

    def _effective_capacity(self, ambient_temp_c: float) -> float:
        """Temperature-derated capacity."""
        delta_t = ambient_temp_c - self.config.temp_nominal_c
        factor = 1.0 + self.config.temp_capacity_coeff * delta_t
        return self.config.capacity_mah * max(0.5, min(1.1, factor))

    def _soc_to_voltage(self, soc_fraction: float) -> float:
        """Non-linear Li-Po discharge curve (per cell)."""
        if soc_fraction > 0.9:
            v = 4.2 - (1.0 - soc_fraction) * 2.0
        elif soc_fraction > 0.1:
            v = 3.6 + soc_fraction * 0.5
        else:
            v = 3.0 + soc_fraction * 6.0
        return max(self.config.critical_voltage_per_cell,
                   min(self.config.full_voltage_per_cell, v))

    def voltage(self, current_draw_a: float = 0.0) -> float:
        """Pack voltage accounting for internal resistance sag."""
        soc_frac = self.soc_pct / 100.0
        ocv = self._soc_to_voltage(soc_frac) * self.config.cell_count
        sag = current_draw_a * self.config.internal_resistance_ohm * self.config.cell_count
        return max(
            self.config.critical_voltage_per_cell * self.config.cell_count,
            ocv - sag,
        )

    def current_draw(self, thrust_fraction: float) -> float:
        """
        Estimated current draw based on thrust demand.
        thrust_fraction: 0.0 (idle) to 1.0 (full throttle).
        Hover is roughly 0.4-0.6 of max thrust for a cheap quad.
        """
        idle = 0.5
        return idle + thrust_fraction * (self.config.max_current_a - idle)

    def tick(
        self,
        dt: float,
        thrust_fraction: float,
        ambient_temp_c: float = 32.0,
    ) -> float:
        """
        Drain battery for one tick. Returns updated SoC percentage.
        thrust_fraction: 0.0 (motors off) to 1.0 (full throttle).
        """
        current = self.current_draw(thrust_fraction)
        drain_mah = (current * dt) / 3600.0

        effective_cap = self._effective_capacity(ambient_temp_c)
        capacity_ratio = self.config.capacity_mah / effective_cap
        adjusted_drain = drain_mah * capacity_ratio

        noise = self._rng.normal(1.0, 0.02)
        adjusted_drain *= max(0.8, noise)

        self._capacity_remaining_mah = max(
            0.0, self._capacity_remaining_mah - adjusted_drain,
        )
        return self.soc_pct

    def estimated_flight_time_sec(
        self,
        thrust_fraction: float = 0.5,
        ambient_temp_c: float = 32.0,
    ) -> float:
        """Estimate remaining flight time at given thrust."""
        current = self.current_draw(thrust_fraction)
        if current < 0.1:
            return float("inf")
        effective_cap = self._effective_capacity(ambient_temp_c)
        remaining_effective = self._capacity_remaining_mah * (effective_cap / self.config.capacity_mah)
        usable = remaining_effective * (1.0 - self.config.rtl_threshold_pct / 100.0)
        return max(0.0, (usable / 1000.0) / current * 3600.0)
