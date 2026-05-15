"""Brushless-motor thrust degradation over flight time.

Phase 1 Stage B.5 of the rigs plan (see docs/MINCO_PIVOT.md §5.6).

Linear thrust-efficiency falloff:
    thrust_scale(flight_hours) = max(min_efficiency,
                                     initial_efficiency
                                     - degradation_rate_per_hour * flight_hours)

The model is intentionally simple — Rig 5 needs a knob, not a physically
accurate ESC/motor characteristic. Real brushless motors degrade non-
linearly and asymmetrically across cells; that fidelity belongs in the
hardware bench-test phase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotorWearConfig:
    initial_efficiency: float = 1.0
    degradation_rate_per_hour: float = 0.02   # 2% per flight hour
    min_efficiency: float = 0.5               # floor (catastrophic = mission abort)


class MotorWear:
    def __init__(self, config: MotorWearConfig) -> None:
        if not (0.0 < config.initial_efficiency <= 1.0):
            raise ValueError("initial_efficiency must be in (0, 1]")
        if config.degradation_rate_per_hour < 0.0:
            raise ValueError("degradation_rate must be non-negative")
        if not (0.0 <= config.min_efficiency <= config.initial_efficiency):
            raise ValueError("min_efficiency must be in [0, initial_efficiency]")
        self.config = config

    def thrust_scaling(self, flight_hours: float) -> float:
        """Multiplier in (min_efficiency, initial_efficiency] applied to commanded thrust."""
        if flight_hours < 0.0:
            raise ValueError("flight_hours must be non-negative")
        raw = self.config.initial_efficiency - self.config.degradation_rate_per_hour * flight_hours
        return float(max(self.config.min_efficiency, raw))

    def mission_abort_hours(self) -> float:
        """Hours of flight at which `thrust_scaling` hits the floor."""
        if self.config.degradation_rate_per_hour <= 0.0:
            return float("inf")
        drop = self.config.initial_efficiency - self.config.min_efficiency
        return drop / self.config.degradation_rate_per_hour
