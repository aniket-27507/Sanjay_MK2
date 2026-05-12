"""
Atmosphere model for Guwahati climate conditions.

Computes air density from temperature, humidity, and altitude,
then derives thrust efficiency and power consumption multipliers.
Guwahati June baseline: 32°C, 85% humidity, 55m ASL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class AtmosphereConfig:
    temperature_c: float = 32.0
    relative_humidity_pct: float = 85.0
    station_altitude_asl_m: float = 55.0
    sea_level_pressure_pa: float = 101325.0
    lapse_rate_k_per_m: float = 0.0065
    reference_air_density: float = 1.225
    seed: Optional[int] = None


class AtmosphereModel:
    R_DRY = 287.058
    R_VAPOR = 461.495

    def __init__(self, config: AtmosphereConfig | None = None):
        self.config = config or AtmosphereConfig()

    def saturation_vapor_pressure(self, temp_c: float) -> float:
        """Magnus formula for saturation vapor pressure (Pa)."""
        return 610.78 * math.exp((17.27 * temp_c) / (temp_c + 237.3))

    def air_density(
        self,
        temp_c: float | None = None,
        humidity_pct: float | None = None,
        altitude_agl_m: float = 0.0,
    ) -> float:
        """
        Compute air density (kg/m³) accounting for temperature,
        humidity, and altitude above ground level.
        """
        T_c = temp_c if temp_c is not None else self.config.temperature_c
        RH = (humidity_pct if humidity_pct is not None else self.config.relative_humidity_pct) / 100.0

        total_alt = self.config.station_altitude_asl_m + altitude_agl_m
        T_k = T_c + 273.15 - self.config.lapse_rate_k_per_m * altitude_agl_m

        pressure = self.config.sea_level_pressure_pa * (
            (1 - self.config.lapse_rate_k_per_m * total_alt / 288.15) ** 5.2561
        )

        p_sat = self.saturation_vapor_pressure(T_c)
        p_vapor = RH * p_sat
        p_dry = pressure - p_vapor

        rho = (p_dry / (self.R_DRY * T_k)) + (p_vapor / (self.R_VAPOR * T_k))
        return rho

    def thrust_efficiency(
        self,
        altitude_agl_m: float = 0.0,
        temp_c: float | None = None,
        humidity_pct: float | None = None,
    ) -> float:
        """
        Thrust efficiency multiplier relative to standard conditions.
        < 1.0 means the props produce less thrust (need more power to hover).
        """
        rho = self.air_density(temp_c, humidity_pct, altitude_agl_m)
        return math.sqrt(rho / self.config.reference_air_density)

    def power_multiplier(
        self,
        altitude_agl_m: float = 0.0,
        temp_c: float | None = None,
        humidity_pct: float | None = None,
    ) -> float:
        """
        Power consumption multiplier to maintain hover in current conditions.
        > 1.0 means more power needed (thinner air).
        """
        eff = self.thrust_efficiency(altitude_agl_m, temp_c, humidity_pct)
        if eff < 0.01:
            return 10.0
        return 1.0 / eff

    def conditions_summary(self, altitude_agl_m: float = 12.0) -> dict:
        """Quick summary for logging/display."""
        rho = self.air_density(altitude_agl_m=altitude_agl_m)
        eff = self.thrust_efficiency(altitude_agl_m=altitude_agl_m)
        pwr = self.power_multiplier(altitude_agl_m=altitude_agl_m)
        return {
            "temperature_c": self.config.temperature_c,
            "humidity_pct": self.config.relative_humidity_pct,
            "altitude_agl_m": altitude_agl_m,
            "air_density_kg_m3": round(rho, 4),
            "thrust_efficiency": round(eff, 4),
            "power_multiplier": round(pwr, 4),
        }
