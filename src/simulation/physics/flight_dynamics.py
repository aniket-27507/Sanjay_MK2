"""
Simplified 6DOF flight dynamics for a cheap quadcopter (~500g, Pixhawk Mini).

Derives true body-frame angular rates and specific force from:
- Commanded velocity vs actual velocity (wind-perturbed)
- Attitude required to achieve thrust vector
- PD attitude controller response (models Pixhawk PX4 inner loop)
- Prop torque reaction and gyroscopic precession

Outputs feed directly into IMUNoiseModel.apply_noise() as ground truth.

Coordinate frames:
  NED (North-East-Down) — world frame, z positive down
  Body — x forward (heading), y right, z down through belly
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3


GRAVITY = 9.80665


@dataclass
class FlightDynamicsConfig:
    mass_kg: float = 0.50
    arm_length_m: float = 0.12
    Ixx: float = 0.0023
    Iyy: float = 0.0023
    Izz: float = 0.0046
    prop_inertia: float = 3.5e-5

    max_roll_deg: float = 35.0
    max_pitch_deg: float = 35.0
    max_yaw_rate_dps: float = 200.0

    # PD attitude controller gains (models PX4 MC_ROLL/PITCH/YAW_P)
    roll_p: float = 6.5
    roll_d: float = 0.35
    pitch_p: float = 6.5
    pitch_d: float = 0.35
    yaw_p: float = 2.8
    yaw_d: float = 0.15

    # Velocity → attitude mapping
    velocity_to_tilt_gain: float = 8.0

    seed: Optional[int] = None


@dataclass
class AttitudeState:
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    roll_rate_rps: float = 0.0
    pitch_rate_rps: float = 0.0
    yaw_rate_rps: float = 0.0


@dataclass
class DynamicsOutput:
    """Ground-truth values to feed into IMU noise model."""
    angular_rate_body_dps: Vector3
    specific_force_body_ms2: Vector3
    attitude: AttitudeState
    thrust_fraction: float


class FlightDynamicsModel:
    """
    Simplified quadcopter dynamics: velocity commands → attitude → angular rates.

    The real Pixhawk inner loop runs at 250Hz. We model its effect:
    1. Desired velocity → desired tilt angles (roll/pitch for horizontal, thrust for vertical)
    2. PD controller drives attitude toward desired → angular acceleration
    3. Integrate angular rates
    4. Compute body-frame specific force (thrust + gravity rotated into body)
    """

    def __init__(self, config: FlightDynamicsConfig | None = None):
        self.config = config or FlightDynamicsConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._state = AttitudeState()
        self._prop_speed_sum = 0.0

    @property
    def attitude(self) -> AttitudeState:
        return self._state

    def _desired_attitude(
        self,
        commanded_vel: Vector3,
        actual_vel: Vector3,
        heading_rad: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute desired roll, pitch, yaw_rate, and thrust_fraction
        from velocity error, mimicking PX4 position controller output.
        """
        cfg = self.config

        vel_err_n = commanded_vel.x - actual_vel.x
        vel_err_e = commanded_vel.y - actual_vel.y
        vel_err_d = commanded_vel.z - actual_vel.z

        # Rotate NED velocity error into heading-aligned frame
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        fwd_err = vel_err_n * cos_h + vel_err_e * sin_h
        right_err = -vel_err_n * sin_h + vel_err_e * cos_h

        # Velocity error → desired tilt (PX4 MPC_XY_VEL_P ≈ 1.8)
        gain = cfg.velocity_to_tilt_gain
        desired_pitch = -np.clip(
            math.radians(fwd_err * gain),
            -math.radians(cfg.max_pitch_deg),
            math.radians(cfg.max_pitch_deg),
        )
        desired_roll = np.clip(
            math.radians(right_err * gain),
            -math.radians(cfg.max_roll_deg),
            math.radians(cfg.max_roll_deg),
        )

        # Vertical: thrust fraction to counter gravity + climb
        hover_thrust = 0.5
        climb_cmd = -vel_err_d * 0.3
        thrust_fraction = np.clip(hover_thrust + climb_cmd, 0.1, 1.0)

        # Yaw rate: zero for now (heading maintained by autopilot)
        desired_yaw_rate = 0.0

        return float(desired_roll), float(desired_pitch), desired_yaw_rate, float(thrust_fraction)

    def _attitude_controller(
        self,
        desired_roll: float,
        desired_pitch: float,
        desired_yaw_rate: float,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        PD attitude controller → angular accelerations (rad/s²).
        Models PX4 mc_att_control inner loop.
        """
        cfg = self.config
        s = self._state

        roll_err = desired_roll - s.roll_rad
        pitch_err = desired_pitch - s.pitch_rad
        yaw_rate_err = desired_yaw_rate - s.yaw_rate_rps

        roll_acc = (cfg.roll_p * roll_err - cfg.roll_d * s.roll_rate_rps) / cfg.Ixx
        pitch_acc = (cfg.pitch_p * pitch_err - cfg.pitch_d * s.pitch_rate_rps) / cfg.Iyy
        yaw_acc = (cfg.yaw_p * yaw_rate_err - cfg.yaw_d * s.yaw_rate_rps) / cfg.Izz

        return roll_acc, pitch_acc, yaw_acc

    def _gyroscopic_precession(self) -> Tuple[float, float, float]:
        """
        Gyroscopic torque from spinning props.
        For a + config quad, net prop angular momentum is along body Z.
        Cross product with body angular velocity gives precession torque.
        """
        s = self._state
        Jp = self.config.prop_inertia
        omega_prop = self._prop_speed_sum

        # tau_gyro = J_prop * omega_prop × omega_body
        gx = -Jp * omega_prop * s.pitch_rate_rps / self.config.Ixx
        gy = Jp * omega_prop * s.roll_rate_rps / self.config.Iyy
        gz = 0.0
        return gx, gy, gz

    def step(
        self,
        commanded_vel: Vector3,
        actual_vel: Vector3,
        heading_rad: float,
        thrust_fraction: float,
        wind_accel: Vector3,
        dt: float,
    ) -> DynamicsOutput:
        """
        Advance one dynamics step. Returns ground-truth angular rates
        and specific force in body frame for IMU.

        Sub-steps internally at 200Hz to keep the PD controller stable
        regardless of sim tick rate (1-10Hz).
        """
        cfg = self.config
        s = self._state

        des_roll, des_pitch, des_yaw_rate, thrust_frac = self._desired_attitude(
            commanded_vel, actual_vel, heading_rad,
        )
        if thrust_fraction > 0:
            thrust_frac = thrust_fraction

        self._prop_speed_sum = thrust_frac * 4 * 800.0

        # Sub-step at 200Hz for controller stability
        substep_dt = 1.0 / 200.0
        num_substeps = max(1, int(dt / substep_dt))
        actual_substep_dt = dt / num_substeps

        for _ in range(num_substeps):
            roll_acc, pitch_acc, yaw_acc = self._attitude_controller(
                des_roll, des_pitch, des_yaw_rate, actual_substep_dt,
            )

            gx, gy, gz = self._gyroscopic_precession()
            roll_acc += gx
            pitch_acc += gy
            yaw_acc += gz

            s.roll_rate_rps += roll_acc * actual_substep_dt
            s.pitch_rate_rps += pitch_acc * actual_substep_dt
            s.yaw_rate_rps += yaw_acc * actual_substep_dt

            s.roll_rad += s.roll_rate_rps * actual_substep_dt
            s.pitch_rad += s.pitch_rate_rps * actual_substep_dt
            s.yaw_rad += s.yaw_rate_rps * actual_substep_dt

            max_r = math.radians(cfg.max_roll_deg * 1.2)
            max_p = math.radians(cfg.max_pitch_deg * 1.2)
            s.roll_rad = np.clip(s.roll_rad, -max_r, max_r)
            s.pitch_rad = np.clip(s.pitch_rad, -max_p, max_p)

        s.yaw_rad = s.yaw_rad % (2 * math.pi)

        # --- Specific force in body frame (what the accelerometer measures) ---
        # Accelerometer measures non-gravitational acceleration:
        #   a_sensed = (F_thrust + F_aero) / m
        # For hover at level attitude: a_sensed = [0, 0, -g] (thrust up = -Z in NED body)
        total_thrust_ms2 = thrust_frac * 2.0 * GRAVITY
        thrust_body = np.array([0.0, 0.0, -total_thrust_ms2])

        # Wind disturbance rotated into body frame
        cr, sr = math.cos(s.roll_rad), math.sin(s.roll_rad)
        cp, sp = math.cos(s.pitch_rad), math.sin(s.pitch_rad)
        cy, sy = math.cos(s.yaw_rad), math.sin(s.yaw_rad)
        wind_ned = np.array([wind_accel.x, wind_accel.y, wind_accel.z])
        R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
        R_nb = R_roll @ R_pitch @ R_yaw
        wind_body = R_nb @ wind_ned

        specific_force = thrust_body + wind_body

        # Vibration from motor imbalance (adds high-freq component)
        motor_vib = self._rng.normal(0, 0.15 * thrust_frac, 3)
        specific_force += motor_vib

        angular_rate_dps = Vector3(
            x=math.degrees(s.roll_rate_rps),
            y=math.degrees(s.pitch_rate_rps),
            z=math.degrees(s.yaw_rate_rps),
        )

        return DynamicsOutput(
            angular_rate_body_dps=angular_rate_dps,
            specific_force_body_ms2=Vector3.from_array(specific_force),
            attitude=AttitudeState(
                roll_rad=s.roll_rad,
                pitch_rad=s.pitch_rad,
                yaw_rad=s.yaw_rad,
                roll_rate_rps=s.roll_rate_rps,
                pitch_rate_rps=s.pitch_rate_rps,
                yaw_rate_rps=s.yaw_rate_rps,
            ),
            thrust_fraction=float(thrust_frac),
        )

    def reset(self, heading_rad: float = 0.0) -> None:
        self._state = AttitudeState(yaw_rad=heading_rad)
        self._prop_speed_sum = 0.0
