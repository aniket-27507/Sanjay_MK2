"""
400Hz IMU oversampling pipeline.

The scenario executor runs at low rate (1-10 Hz). Real IMU hardware
outputs at 400Hz. This module bridges the gap:

1. Takes two consecutive DynamicsOutput states (from flight_dynamics.py)
2. Interpolates angular rates and specific force between them
3. Runs IMUNoiseModel at 400Hz substeps to generate realistic noise
4. Outputs a buffer of IMU+magnetometer readings at true sensor rate

The interpolation uses cubic Hermite splines for smooth, physically
plausible transitions between sim ticks. Noise is injected per-sample
(not interpolated), so the noise spectrum matches real hardware.

Output is directly compatible with EKF development, PX4 replay, and
hardware comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.core.types.drone_types import Vector3
from .imu_model import IMUNoiseModel, IMUConfig, IMUReading
from .magnetometer_model import MagnetometerModel, MagnetometerConfig, MagReading
from .flight_dynamics import DynamicsOutput, AttitudeState


@dataclass
class HighRateIMUSample:
    """Single 400Hz IMU sample with all sensor channels."""
    timestamp_us: int
    gyro_dps: Vector3
    accel_ms2: Vector3
    mag_ut: Vector3
    temperature_c: float
    # Ground truth for validation
    true_gyro_dps: Vector3
    true_accel_ms2: Vector3
    true_attitude: AttitudeState


@dataclass
class HighRateIMUConfig:
    imu_rate_hz: float = 400.0
    mag_rate_hz: float = 100.0
    imu: IMUConfig = field(default_factory=lambda: IMUConfig(sample_rate_hz=400.0))
    mag: MagnetometerConfig = field(default_factory=MagnetometerConfig)
    seed: Optional[int] = None


class HighRateIMUPipeline:
    """
    Generates 400Hz IMU + 100Hz magnetometer streams between sim ticks.

    Usage:
        pipeline = HighRateIMUPipeline()
        # Each sim tick:
        samples = pipeline.generate_samples(
            prev_dynamics, curr_dynamics,
            sim_dt, drone_pos, buildings, temp, thrust
        )
        # samples is a list of HighRateIMUSample at 400Hz
    """

    def __init__(self, config: HighRateIMUConfig | None = None):
        self.config = config or HighRateIMUConfig()
        self._imu = IMUNoiseModel(self.config.imu)
        self._mag = MagnetometerModel(self.config.mag)
        self._rng = np.random.default_rng(self.config.seed)
        self._time_us: int = 0
        self._mag_counter = 0
        self._last_mag = MagReading(
            mag_ut=Vector3(), heading_true_deg=0.0, heading_magnetic_deg=0.0,
        )

    def _hermite_interp(
        self, t: float,
        p0: float, p1: float,
        m0: float, m1: float,
    ) -> float:
        """Cubic Hermite interpolation for smooth transitions."""
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def _interp_vector3(
        self, t: float,
        v0: Vector3, v1: Vector3,
        rate0: Vector3, rate1: Vector3,
        dt: float,
    ) -> Vector3:
        """Interpolate Vector3 using cubic Hermite with rate tangents."""
        return Vector3(
            x=self._hermite_interp(t, v0.x, v1.x, rate0.x * dt, rate1.x * dt),
            y=self._hermite_interp(t, v0.y, v1.y, rate0.y * dt, rate1.y * dt),
            z=self._hermite_interp(t, v0.z, v1.z, rate0.z * dt, rate1.z * dt),
        )

    def _interp_attitude(
        self, t: float, a0: AttitudeState, a1: AttitudeState,
    ) -> AttitudeState:
        """Linear interpolation of attitude (good enough for small angles)."""
        return AttitudeState(
            roll_rad=a0.roll_rad + t * (a1.roll_rad - a0.roll_rad),
            pitch_rad=a0.pitch_rad + t * (a1.pitch_rad - a0.pitch_rad),
            yaw_rad=a0.yaw_rad + t * (a1.yaw_rad - a0.yaw_rad),
            roll_rate_rps=a0.roll_rate_rps + t * (a1.roll_rate_rps - a0.roll_rate_rps),
            pitch_rate_rps=a0.pitch_rate_rps + t * (a1.pitch_rate_rps - a0.pitch_rate_rps),
            yaw_rate_rps=a0.yaw_rate_rps + t * (a1.yaw_rate_rps - a0.yaw_rate_rps),
        )

    def generate_samples(
        self,
        prev_dynamics: DynamicsOutput,
        curr_dynamics: DynamicsOutput,
        sim_dt: float,
        drone_pos: Vector3,
        ambient_temp_c: float = 32.0,
        thrust_fraction: float = 0.5,
        buildings=None,
    ) -> List[HighRateIMUSample]:
        """
        Generate high-rate IMU samples between two consecutive sim ticks.

        Returns list of HighRateIMUSample at config.imu_rate_hz.
        For a 1-second sim tick at 400Hz, this returns 400 samples.
        """
        imu_dt = 1.0 / self.config.imu_rate_hz
        mag_ratio = int(self.config.imu_rate_hz / self.config.mag_rate_hz)
        num_samples = max(1, int(sim_dt * self.config.imu_rate_hz))

        # Compute rate-of-change for Hermite tangents (finite difference)
        gyro_rate_prev = Vector3()
        gyro_rate_curr = Vector3(
            x=(curr_dynamics.angular_rate_body_dps.x - prev_dynamics.angular_rate_body_dps.x) / sim_dt,
            y=(curr_dynamics.angular_rate_body_dps.y - prev_dynamics.angular_rate_body_dps.y) / sim_dt,
            z=(curr_dynamics.angular_rate_body_dps.z - prev_dynamics.angular_rate_body_dps.z) / sim_dt,
        )

        accel_rate_prev = Vector3()
        accel_rate_curr = Vector3(
            x=(curr_dynamics.specific_force_body_ms2.x - prev_dynamics.specific_force_body_ms2.x) / sim_dt,
            y=(curr_dynamics.specific_force_body_ms2.y - prev_dynamics.specific_force_body_ms2.y) / sim_dt,
            z=(curr_dynamics.specific_force_body_ms2.z - prev_dynamics.specific_force_body_ms2.z) / sim_dt,
        )

        samples: List[HighRateIMUSample] = []

        for i in range(num_samples):
            t = (i + 1) / num_samples

            # Interpolate ground truth
            true_gyro = self._interp_vector3(
                t,
                prev_dynamics.angular_rate_body_dps,
                curr_dynamics.angular_rate_body_dps,
                gyro_rate_prev, gyro_rate_curr,
                sim_dt,
            )
            true_accel = self._interp_vector3(
                t,
                prev_dynamics.specific_force_body_ms2,
                curr_dynamics.specific_force_body_ms2,
                accel_rate_prev, accel_rate_curr,
                sim_dt,
            )
            true_att = self._interp_attitude(
                t, prev_dynamics.attitude, curr_dynamics.attitude,
            )

            # Apply IMU noise at this sample
            imu_reading = self._imu.apply_noise(
                true_angular_rate=true_gyro,
                true_accel=true_accel,
                dt=imu_dt,
                ambient_temp_c=ambient_temp_c,
                thrust_fraction=thrust_fraction,
            )

            # Magnetometer at lower rate
            self._mag_counter += 1
            if self._mag_counter >= mag_ratio:
                self._mag_counter = 0
                self._last_mag = self._mag.apply_noise(
                    attitude_roll_rad=true_att.roll_rad,
                    attitude_pitch_rad=true_att.pitch_rad,
                    attitude_yaw_rad=true_att.yaw_rad,
                    drone_pos=drone_pos,
                    dt=1.0 / self.config.mag_rate_hz,
                    ambient_temp_c=ambient_temp_c,
                    thrust_fraction=thrust_fraction,
                    buildings=buildings,
                )

            self._time_us += int(imu_dt * 1e6)

            samples.append(HighRateIMUSample(
                timestamp_us=self._time_us,
                gyro_dps=imu_reading.gyro,
                accel_ms2=imu_reading.accel,
                mag_ut=self._last_mag.mag_ut,
                temperature_c=imu_reading.temperature_c,
                true_gyro_dps=true_gyro,
                true_accel_ms2=true_accel,
                true_attitude=true_att,
            ))

        return samples

    def reset(self) -> None:
        self._time_us = 0
        self._mag_counter = 0
