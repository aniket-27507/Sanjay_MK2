"""
Magnetometer noise model for cheap MEMS compass (HMC5883L/QMC5883L class).

Models:
- Guwahati IGRF-13 geomagnetic field (epoch 2026)
- Hard iron offset (permanent magnet bias from motors/frame)
- Soft iron distortion (ferrous material warps field lines)
- Building magnetic anomaly (rebar, steel beams, power lines)
- Power line interference (50Hz AC magnetic field)
- Temperature drift
- Quantization noise

Guwahati (26.14°N, 91.74°E, 55m ASL) IGRF-13 reference values:
  Total field:  ~46,200 nT
  Declination:  ~0.1° W (nearly true north)
  Inclination:  ~39.5° (dip angle)
  North (X):    ~35,600 nT
  East (Y):     ~-60 nT
  Down (Z):     ~29,400 nT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3


@dataclass
class MagnetometerConfig:
    # IGRF field at Guwahati (nT → converted to µT internally)
    igrf_north_nt: float = 35600.0
    igrf_east_nt: float = -60.0
    igrf_down_nt: float = 29400.0

    # Sensor noise (QMC5883L class)
    noise_density_ut: float = 0.2
    quantization_ut: float = 0.05
    sample_rate_hz: float = 100.0

    # Hard iron: constant offset from motors/frame (µT)
    hard_iron_ut: Tuple[float, float, float] = (12.0, -8.0, 5.0)

    # Soft iron: scale/cross-axis distortion matrix (dimensionless)
    # Slightly non-identity for a cheap frame with motor magnets
    soft_iron_matrix: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ] = (
        (1.02, 0.03, -0.01),
        (0.03, 0.98, 0.02),
        (-0.01, 0.02, 1.01),
    )

    # Building anomaly: ferrous structures distort field
    building_anomaly_max_ut: float = 15.0
    building_anomaly_radius_m: float = 20.0

    # Power line interference
    powerline_amplitude_ut: float = 0.8
    powerline_freq_hz: float = 50.0
    powerline_proximity_radius_m: float = 30.0

    # Temperature drift
    temp_drift_ut_per_c: float = 0.03
    temp_nominal_c: float = 25.0

    # Motor magnetic interference (scales with thrust)
    motor_interference_ut: float = 5.0

    seed: Optional[int] = None


@dataclass
class MagReading:
    mag_ut: Vector3
    heading_true_deg: float
    heading_magnetic_deg: float


class MagnetometerModel:
    def __init__(self, config: MagnetometerConfig | None = None):
        self.config = config or MagnetometerConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._time = 0.0

        # Convert IGRF from nT to µT (sensor output unit)
        self._igrf_ut = np.array([
            self.config.igrf_north_nt / 1000.0,
            self.config.igrf_east_nt / 1000.0,
            self.config.igrf_down_nt / 1000.0,
        ])

        self._hard_iron = np.array(self.config.hard_iron_ut)
        self._soft_iron = np.array(self.config.soft_iron_matrix)

    def _building_anomaly(
        self,
        drone_pos: Vector3,
        buildings: List[Tuple[Vector3, float]] | None,
    ) -> np.ndarray:
        """
        Magnetic anomaly from nearby buildings (rebar/steel content).
        Models as dipole-like falloff with random orientation per building.
        """
        if not buildings:
            return np.zeros(3)

        anomaly = np.zeros(3)
        cfg = self.config

        for center, width in buildings:
            dx = drone_pos.x - center.x
            dy = drone_pos.y - center.y
            dz = drone_pos.z - center.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < cfg.building_anomaly_radius_m and dist > 0.5:
                # Dipole falloff: 1/r³
                strength = cfg.building_anomaly_max_ut * (
                    (cfg.building_anomaly_radius_m / dist) ** 3
                )
                strength = min(strength, cfg.building_anomaly_max_ut * 3)

                # Direction: deterministic per building (hash of position)
                seed_val = int(abs(center.x * 1000 + center.y * 7 + center.z * 13)) % (2**31)
                bld_rng = np.random.default_rng(seed_val)
                direction = bld_rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction) + 1e-12

                anomaly += direction * strength

        return anomaly

    def _powerline_interference(self, drone_pos: Vector3) -> np.ndarray:
        """50Hz AC magnetic field from overhead power lines."""
        cfg = self.config
        phase = 2 * math.pi * cfg.powerline_freq_hz * self._time
        # Power lines typically run along roads — model as y-axis aligned
        amp = cfg.powerline_amplitude_ut * math.sin(phase)
        # Field is perpendicular to the wire (mostly x and z components)
        return np.array([amp * 0.7, 0.0, amp * 0.7])

    def _quantize(self, value: float) -> float:
        step = self.config.quantization_ut
        if step < 1e-12:
            return value
        return round(value / step) * step

    def apply_noise(
        self,
        attitude_roll_rad: float,
        attitude_pitch_rad: float,
        attitude_yaw_rad: float,
        drone_pos: Vector3,
        dt: float,
        ambient_temp_c: float = 32.0,
        thrust_fraction: float = 0.5,
        buildings: List[Tuple[Vector3, float]] | None = None,
    ) -> MagReading:
        """
        Generate noisy magnetometer reading in body frame.

        Takes drone attitude to rotate Earth field into body frame,
        then adds all noise sources.
        """
        self._time += dt
        cfg = self.config

        # --- Earth field in NED ---
        earth_field_ned = self._igrf_ut.copy()

        # Building anomaly (NED frame)
        earth_field_ned += self._building_anomaly(drone_pos, buildings)

        # --- Rotate NED → body frame ---
        cr, sr = math.cos(attitude_roll_rad), math.sin(attitude_roll_rad)
        cp, sp = math.cos(attitude_pitch_rad), math.sin(attitude_pitch_rad)
        cy, sy = math.cos(attitude_yaw_rad), math.sin(attitude_yaw_rad)

        R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        R_nb = R_roll @ R_pitch @ R_yaw

        field_body = R_nb @ earth_field_ned

        # --- Soft iron distortion ---
        field_body = self._soft_iron @ field_body

        # --- Hard iron offset ---
        field_body += self._hard_iron

        # --- Motor interference (scales with thrust) ---
        motor_noise = self._rng.normal(0, cfg.motor_interference_ut * thrust_fraction, 3)
        field_body += motor_noise

        # --- Power line interference ---
        field_body += self._powerline_interference(drone_pos)

        # --- Temperature drift ---
        temp_delta = ambient_temp_c - cfg.temp_nominal_c
        field_body += temp_delta * cfg.temp_drift_ut_per_c

        # --- White noise ---
        dt_sample = 1.0 / cfg.sample_rate_hz
        noise = self._rng.normal(0, cfg.noise_density_ut / math.sqrt(dt_sample), 3)
        field_body += noise

        # --- Quantization ---
        field_body = np.array([self._quantize(v) for v in field_body])

        # --- Compute headings ---
        # True heading from attitude
        heading_true = math.degrees(attitude_yaw_rad) % 360.0

        # Magnetic heading from sensor (tilt-compensated)
        # Undo tilt to get horizontal components
        Bx_h = field_body[0] * cp + field_body[1] * sp * sr + field_body[2] * sp * cr
        By_h = field_body[1] * cr - field_body[2] * sr
        heading_mag = math.degrees(math.atan2(-By_h, Bx_h)) % 360.0

        return MagReading(
            mag_ut=Vector3(x=float(field_body[0]), y=float(field_body[1]), z=float(field_body[2])),
            heading_true_deg=heading_true,
            heading_magnetic_deg=heading_mag,
        )
