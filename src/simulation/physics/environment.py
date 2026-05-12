"""
Physics environment wrapper — single integration point for the scenario executor.

Manages all Tier 1 + Tier 2 physics models and provides a unified
`apply_physics()` call that takes a drone's commanded velocity and
returns the actual velocity after wind, GPS noise, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.types.drone_types import Vector3

from .wind_model import WindModel, WindConfig
from .gps_model import GPSNoiseModel, GPSConfig
from .battery_model import BatteryModel, BatteryConfig
from .imu_model import IMUNoiseModel, IMUConfig, IMUReading
from .atmosphere_model import AtmosphereModel, AtmosphereConfig
from .lidar_noise_model import LiDARNoiseModel, LiDARNoiseConfig
from .servo_lidar_model import ServoLiDARModel, ServoLiDARConfig


@dataclass
class PhysicsConfig:
    enabled: bool = True
    wind: WindConfig = field(default_factory=WindConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    lidar_noise: LiDARNoiseConfig = field(default_factory=LiDARNoiseConfig)
    servo_lidar: ServoLiDARConfig = field(default_factory=ServoLiDARConfig)


@dataclass
class PhysicsResult:
    """Output of apply_physics for a single drone tick."""
    actual_velocity: Vector3
    actual_position: Vector3
    gps_position: Vector3
    gps_velocity: Vector3
    battery_soc_pct: float
    battery_voltage: float
    thrust_fraction: float
    wind_acceleration: Vector3
    imu_reading: Optional[IMUReading] = None
    should_rtl: bool = False
    is_battery_critical: bool = False


class PhysicsEnvironment:
    """
    Per-simulation physics state. Create one per ScenarioExecutor.
    Manages per-drone models (battery, GPS, IMU each have independent state).
    Wind and atmosphere are shared (same weather for all drones).
    """

    def __init__(self, config: PhysicsConfig | None = None):
        self.config = config or PhysicsConfig()
        self._wind = WindModel(self.config.wind)
        self._atmosphere = AtmosphereModel(self.config.atmosphere)
        self._servo_lidar_config = self.config.servo_lidar

        self._per_drone: Dict[int, _DronePhysicsState] = {}

    def register_drone(self, drone_id: int) -> None:
        """Initialize per-drone physics models."""
        self._per_drone[drone_id] = _DronePhysicsState(
            gps=GPSNoiseModel(self.config.gps),
            battery=BatteryModel(self.config.battery),
            imu=IMUNoiseModel(self.config.imu),
            servo_lidar=ServoLiDARModel(self._servo_lidar_config),
        )

    @property
    def wind(self) -> WindModel:
        return self._wind

    @property
    def atmosphere(self) -> AtmosphereModel:
        return self._atmosphere

    def get_servo_lidar(self, drone_id: int) -> ServoLiDARModel:
        return self._per_drone[drone_id].servo_lidar

    def get_battery(self, drone_id: int) -> BatteryModel:
        return self._per_drone[drone_id].battery

    def apply_physics(
        self,
        drone_id: int,
        true_position: Vector3,
        commanded_velocity: Vector3,
        dt: float,
        buildings: List[Tuple[Vector3, float]] | None = None,
    ) -> PhysicsResult:
        """
        Apply all physics models to a drone's commanded velocity.

        Returns the actual position/velocity after wind forces, plus
        GPS-observed position (what the autopilot sees), battery state, etc.
        """
        if not self.config.enabled:
            new_pos = Vector3(
                true_position.x + commanded_velocity.x * dt,
                true_position.y + commanded_velocity.y * dt,
                true_position.z + commanded_velocity.z * dt,
            )
            return PhysicsResult(
                actual_velocity=commanded_velocity,
                actual_position=new_pos,
                gps_position=new_pos,
                gps_velocity=commanded_velocity,
                battery_soc_pct=100.0,
                battery_voltage=12.6,
                thrust_fraction=0.5,
                wind_acceleration=Vector3(),
            )

        state = self._per_drone[drone_id]
        altitude_agl = abs(true_position.z)

        # --- Atmosphere: thrust efficiency at current altitude ---
        power_mult = self._atmosphere.power_multiplier(altitude_agl_m=altitude_agl)
        thrust_eff = self._atmosphere.thrust_efficiency(altitude_agl_m=altitude_agl)

        # Thrust fraction needed to achieve commanded velocity
        # Hover ~= 0.5 thrust. Moving faster = more thrust.
        cmd_speed = commanded_velocity.magnitude()
        thrust_fraction = min(1.0, 0.45 / max(0.01, thrust_eff) + cmd_speed * 0.04)

        # --- Wind: force applied to drone ---
        wind_accel = self._wind.compute_acceleration(
            true_position, commanded_velocity, dt, buildings,
        )

        # Actual velocity = commanded + wind effect
        actual_velocity = Vector3(
            commanded_velocity.x + wind_accel.x * dt,
            commanded_velocity.y + wind_accel.y * dt,
            commanded_velocity.z + wind_accel.z * dt,
        )

        # Actual position integration
        actual_position = Vector3(
            true_position.x + actual_velocity.x * dt,
            true_position.y + actual_velocity.y * dt,
            true_position.z + actual_velocity.z * dt,
        )

        # --- GPS: what the autopilot observes ---
        gps_pos, gps_vel = state.gps.apply_noise(
            actual_position, actual_velocity, dt, buildings,
        )

        # --- Battery: temperature-adjusted drain ---
        temp_c = self._atmosphere.config.temperature_c
        soc = state.battery.tick(dt, thrust_fraction * power_mult, temp_c)
        voltage = state.battery.voltage(state.battery.current_draw(thrust_fraction))

        # --- IMU: noisy sensor readings ---
        # True angular rate from velocity change (simplified)
        true_accel = Vector3(
            wind_accel.x,
            wind_accel.y,
            wind_accel.z + 9.81,  # gravity in NED down
        )
        imu_reading = state.imu.apply_noise(
            true_angular_rate=Vector3(),  # would need attitude controller for real rates
            true_accel=true_accel,
            dt=dt,
            ambient_temp_c=temp_c,
            thrust_fraction=thrust_fraction,
        )

        return PhysicsResult(
            actual_velocity=actual_velocity,
            actual_position=actual_position,
            gps_position=gps_pos,
            gps_velocity=gps_vel,
            battery_soc_pct=soc,
            battery_voltage=voltage,
            thrust_fraction=thrust_fraction,
            wind_acceleration=wind_accel,
            imu_reading=imu_reading,
            should_rtl=state.battery.should_rtl,
            is_battery_critical=state.battery.is_critical,
        )


@dataclass
class _DronePhysicsState:
    gps: GPSNoiseModel
    battery: BatteryModel
    imu: IMUNoiseModel
    servo_lidar: ServoLiDARModel
