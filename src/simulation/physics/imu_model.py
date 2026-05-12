"""
IMU noise model for cheap MEMS sensors (MPU-6050 class on Pixhawk Mini).

Models:
- Gyroscope: bias instability, angular random walk, quantization
- Accelerometer: bias, vibration noise from prop wash, quantization
- Temperature drift on both sensors
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.types.drone_types import Vector3


@dataclass
class IMUConfig:
    # Gyroscope (deg/s noise characteristics)
    gyro_bias_instability_dps: float = 0.04
    gyro_random_walk_dps_sqrt_hz: float = 0.005
    gyro_quantization_dps: float = 0.01
    gyro_temp_drift_dps_per_c: float = 0.002

    # Accelerometer (m/s² noise characteristics)
    accel_bias_instability_ms2: float = 0.02
    accel_noise_density_ms2_sqrt_hz: float = 0.004
    accel_quantization_ms2: float = 0.005
    accel_vibration_amplitude_ms2: float = 0.8
    accel_vibration_freq_hz: float = 120.0
    accel_temp_drift_ms2_per_c: float = 0.001

    sample_rate_hz: float = 100.0
    temp_nominal_c: float = 25.0
    seed: Optional[int] = None


@dataclass
class IMUReading:
    gyro: Vector3
    accel: Vector3
    temperature_c: float


class IMUNoiseModel:
    def __init__(self, config: IMUConfig | None = None):
        self.config = config or IMUConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._time = 0.0

        self._gyro_bias = self._rng.normal(0, self.config.gyro_bias_instability_dps, 3)
        self._accel_bias = self._rng.normal(0, self.config.accel_bias_instability_ms2, 3)

        self._gyro_bias_walk_sigma = self.config.gyro_bias_instability_dps * 0.01

    def _quantize(self, value: float, step: float) -> float:
        if step < 1e-12:
            return value
        return round(value / step) * step

    def apply_noise(
        self,
        true_angular_rate: Vector3,
        true_accel: Vector3,
        dt: float,
        ambient_temp_c: float = 32.0,
        thrust_fraction: float = 0.5,
    ) -> IMUReading:
        """
        Apply realistic IMU noise to true sensor values.
        true_angular_rate: degrees/s in body frame
        true_accel: m/s² in body frame (includes gravity)
        thrust_fraction: 0-1, affects vibration amplitude
        """
        self._time += dt
        dt_sample = 1.0 / self.config.sample_rate_hz
        temp_delta = ambient_temp_c - self.config.temp_nominal_c

        # --- Gyroscope ---
        self._gyro_bias += self._rng.normal(0, self._gyro_bias_walk_sigma, 3) * math.sqrt(dt)

        gyro_noise = self._rng.normal(
            0,
            self.config.gyro_random_walk_dps_sqrt_hz / math.sqrt(dt_sample),
            3,
        )
        temp_drift_g = self.config.gyro_temp_drift_dps_per_c * temp_delta

        noisy_gyro = np.array([
            true_angular_rate.x, true_angular_rate.y, true_angular_rate.z,
        ]) + self._gyro_bias + gyro_noise + temp_drift_g

        noisy_gyro = np.array([
            self._quantize(v, self.config.gyro_quantization_dps) for v in noisy_gyro
        ])

        # --- Accelerometer ---
        accel_noise = self._rng.normal(
            0,
            self.config.accel_noise_density_ms2_sqrt_hz / math.sqrt(dt_sample),
            3,
        )
        temp_drift_a = self.config.accel_temp_drift_ms2_per_c * temp_delta

        vib_amp = self.config.accel_vibration_amplitude_ms2 * thrust_fraction
        vib_phase = 2 * math.pi * self.config.accel_vibration_freq_hz * self._time
        vibration = np.array([
            vib_amp * math.sin(vib_phase + self._rng.uniform(0, math.pi)),
            vib_amp * math.sin(vib_phase * 1.1 + self._rng.uniform(0, math.pi)),
            vib_amp * math.sin(vib_phase * 0.9 + self._rng.uniform(0, math.pi)),
        ])

        noisy_accel = np.array([
            true_accel.x, true_accel.y, true_accel.z,
        ]) + self._accel_bias + accel_noise + temp_drift_a + vibration

        noisy_accel = np.array([
            self._quantize(v, self.config.accel_quantization_ms2) for v in noisy_accel
        ])

        return IMUReading(
            gyro=Vector3(x=float(noisy_gyro[0]), y=float(noisy_gyro[1]), z=float(noisy_gyro[2])),
            accel=Vector3(x=float(noisy_accel[0]), y=float(noisy_accel[1]), z=float(noisy_accel[2])),
            temperature_c=ambient_temp_c + self._rng.normal(0, 0.5),
        )

    def heading_drift_estimate(self, elapsed_sec: float) -> float:
        """
        Estimate worst-case heading drift in degrees after elapsed_sec
        without GPS correction (dead reckoning).
        """
        bias_drift = self.config.gyro_bias_instability_dps * elapsed_sec
        random_walk = self.config.gyro_random_walk_dps_sqrt_hz * math.sqrt(elapsed_sec)
        return abs(bias_drift) + 3 * random_walk
