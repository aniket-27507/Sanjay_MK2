"""
Physics environment wrapper — single integration point for the scenario executor.

Manages all Tier 1 + Tier 2 + Tier 3 (Guwahati full-fidelity) physics models
and provides a unified `apply_physics()` call that takes a drone's commanded
velocity and returns the actual velocity after wind, GPS noise, etc.

Tier 3 additions (Guwahati full-fidelity):
- FlightDynamicsModel: true angular rates + body-frame specific force
- MagnetometerModel: IGRF field + building distortion + motor interference
- MonsoonModel: rain events with cascading sensor/flight effects
- RFEnvironmentModel: WiFi mesh + GPS degradation in urban RF
- HighRateIMUPipeline: 400Hz oversampled IMU+mag output
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
from .flight_dynamics import FlightDynamicsModel, FlightDynamicsConfig, DynamicsOutput
from .magnetometer_model import MagnetometerModel, MagnetometerConfig, MagReading
from .monsoon_model import MonsoonModel, MonsoonConfig, MonsoonState
from .rf_model import RFEnvironmentModel, RFConfig, RFState
from .imu_highrate import HighRateIMUPipeline, HighRateIMUConfig, HighRateIMUSample


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
    # Tier 3: Guwahati full-fidelity
    flight_dynamics: FlightDynamicsConfig = field(default_factory=FlightDynamicsConfig)
    magnetometer: MagnetometerConfig = field(default_factory=MagnetometerConfig)
    monsoon: MonsoonConfig = field(default_factory=MonsoonConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    highrate_imu: HighRateIMUConfig = field(default_factory=HighRateIMUConfig)
    enable_highrate_imu: bool = False
    enable_monsoon: bool = False
    enable_rf: bool = False


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
    # Tier 3 additions
    dynamics: Optional[DynamicsOutput] = None
    mag_reading: Optional[MagReading] = None
    monsoon_state: Optional[MonsoonState] = None
    rf_state: Optional[RFState] = None
    highrate_imu_samples: Optional[List[HighRateIMUSample]] = None


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

        # Shared environment models
        self._monsoon: Optional[MonsoonModel] = None
        if self.config.enable_monsoon:
            self._monsoon = MonsoonModel(self.config.monsoon)
        self._monsoon_state = MonsoonState()

        self._per_drone: Dict[int, _DronePhysicsState] = {}

    def register_drone(self, drone_id: int) -> None:
        """Initialize per-drone physics models."""
        self._per_drone[drone_id] = _DronePhysicsState(
            gps=GPSNoiseModel(self.config.gps),
            battery=BatteryModel(self.config.battery),
            imu=IMUNoiseModel(self.config.imu),
            servo_lidar=ServoLiDARModel(self._servo_lidar_config),
            dynamics=FlightDynamicsModel(self.config.flight_dynamics),
            magnetometer=MagnetometerModel(self.config.magnetometer),
            rf=RFEnvironmentModel(self.config.rf) if self.config.enable_rf else None,
            highrate_pipeline=HighRateIMUPipeline(self.config.highrate_imu) if self.config.enable_highrate_imu else None,
            prev_dynamics=None,
        )

    @property
    def wind(self) -> WindModel:
        return self._wind

    @property
    def atmosphere(self) -> AtmosphereModel:
        return self._atmosphere

    @property
    def monsoon_state(self) -> MonsoonState:
        return self._monsoon_state

    def get_servo_lidar(self, drone_id: int) -> ServoLiDARModel:
        return self._per_drone[drone_id].servo_lidar

    def get_battery(self, drone_id: int) -> BatteryModel:
        return self._per_drone[drone_id].battery

    def get_dynamics(self, drone_id: int) -> FlightDynamicsModel:
        return self._per_drone[drone_id].dynamics

    def tick_environment(self, dt: float) -> None:
        """Advance shared environment models (call once per sim tick, not per drone)."""
        if self._monsoon:
            self._monsoon_state = self._monsoon.tick(dt)

    def apply_physics(
        self,
        drone_id: int,
        true_position: Vector3,
        commanded_velocity: Vector3,
        dt: float,
        buildings: List[Tuple[Vector3, float]] | None = None,
        heading_rad: float = 0.0,
    ) -> PhysicsResult:
        """
        Apply all physics models to a drone's commanded velocity.

        Returns the actual position/velocity after wind forces, plus
        GPS-observed position (what the autopilot sees), battery state,
        flight dynamics, magnetometer, and optionally 400Hz IMU stream.
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

        # --- Monsoon effects on environment ---
        ms = self._monsoon_state
        temp_c = self._atmosphere.config.temperature_c + ms.temp_offset_c

        # --- Atmosphere: thrust efficiency at current altitude ---
        power_mult = self._atmosphere.power_multiplier(altitude_agl_m=altitude_agl)
        thrust_eff = self._atmosphere.thrust_efficiency(altitude_agl_m=altitude_agl)
        thrust_eff *= ms.prop_efficiency_multiplier

        # Thrust fraction needed to achieve commanded velocity
        cmd_speed = commanded_velocity.magnitude()
        thrust_fraction = min(1.0, 0.45 / max(0.01, thrust_eff) + cmd_speed * 0.04)

        # --- Wind: force applied to drone (monsoon intensified) ---
        wind_accel = self._wind.compute_acceleration(
            true_position, commanded_velocity, dt, buildings,
        )
        wind_accel = Vector3(
            x=wind_accel.x * ms.wind_gust_multiplier,
            y=wind_accel.y * ms.wind_gust_multiplier,
            z=wind_accel.z * ms.wind_gust_multiplier,
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

        # --- Flight dynamics: true angular rates and specific force ---
        dyn_output = state.dynamics.step(
            commanded_vel=commanded_velocity,
            actual_vel=actual_velocity,
            heading_rad=heading_rad,
            thrust_fraction=thrust_fraction,
            wind_accel=wind_accel,
            dt=dt,
        )

        # --- GPS: what the autopilot observes ---
        gps_pos, gps_vel = state.gps.apply_noise(
            actual_position, actual_velocity, dt, buildings,
        )

        # RF model adjusts GPS sigma dynamically
        rf_state = None
        if state.rf:
            rf_state = state.rf.compute_rf_state(
                true_position, dt, buildings, ms.current_intensity_mmhr,
            )

        # --- Battery: temperature-adjusted drain (monsoon affects drain) ---
        drain_mult = power_mult * ms.battery_drain_multiplier
        soc = state.battery.tick(dt, thrust_fraction * drain_mult, temp_c)
        voltage = state.battery.voltage(state.battery.current_draw(thrust_fraction))

        # --- IMU: noisy sensor readings from flight dynamics ground truth ---
        imu_reading = state.imu.apply_noise(
            true_angular_rate=dyn_output.angular_rate_body_dps,
            true_accel=dyn_output.specific_force_body_ms2,
            dt=dt,
            ambient_temp_c=temp_c,
            thrust_fraction=thrust_fraction,
        )

        # --- Magnetometer ---
        mag_reading = state.magnetometer.apply_noise(
            attitude_roll_rad=dyn_output.attitude.roll_rad,
            attitude_pitch_rad=dyn_output.attitude.pitch_rad,
            attitude_yaw_rad=dyn_output.attitude.yaw_rad,
            drone_pos=true_position,
            dt=dt,
            ambient_temp_c=temp_c,
            thrust_fraction=thrust_fraction,
            buildings=buildings,
        )

        # --- 400Hz high-rate IMU pipeline ---
        highrate_samples = None
        if state.highrate_pipeline and state.prev_dynamics:
            highrate_samples = state.highrate_pipeline.generate_samples(
                prev_dynamics=state.prev_dynamics,
                curr_dynamics=dyn_output,
                sim_dt=dt,
                drone_pos=true_position,
                ambient_temp_c=temp_c,
                thrust_fraction=thrust_fraction,
                buildings=buildings,
            )
        state.prev_dynamics = dyn_output

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
            dynamics=dyn_output,
            mag_reading=mag_reading,
            monsoon_state=ms if self.config.enable_monsoon else None,
            rf_state=rf_state,
            highrate_imu_samples=highrate_samples,
        )


@dataclass
class _DronePhysicsState:
    gps: GPSNoiseModel
    battery: BatteryModel
    imu: IMUNoiseModel
    servo_lidar: ServoLiDARModel
    dynamics: FlightDynamicsModel
    magnetometer: MagnetometerModel
    rf: Optional[RFEnvironmentModel] = None
    highrate_pipeline: Optional[HighRateIMUPipeline] = None
    prev_dynamics: Optional[DynamicsOutput] = None
