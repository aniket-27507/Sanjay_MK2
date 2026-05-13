"""
Physics simulation models for hardware-fidelity drone validation.

Tier 1: WindModel, GPSNoiseModel, BatteryModel
Tier 2: IMUNoiseModel, AtmosphereModel, LiDARNoiseModel, BuildingTurbulence (inside WindModel)
Tier 3: FlightDynamicsModel, MagnetometerModel, MonsoonModel, RFEnvironmentModel, HighRateIMUPipeline
"""

from .wind_model import WindModel, WindConfig
from .gps_model import GPSNoiseModel, GPSConfig
from .battery_model import BatteryModel, BatteryConfig
from .imu_model import IMUNoiseModel, IMUConfig, IMUReading
from .atmosphere_model import AtmosphereModel, AtmosphereConfig
from .lidar_noise_model import LiDARNoiseModel, LiDARNoiseConfig
from .servo_lidar_model import ServoLiDARModel, ServoLiDARConfig
from .flight_dynamics import FlightDynamicsModel, FlightDynamicsConfig, DynamicsOutput, AttitudeState
from .magnetometer_model import MagnetometerModel, MagnetometerConfig, MagReading
from .monsoon_model import MonsoonModel, MonsoonConfig, MonsoonState
from .rf_model import RFEnvironmentModel, RFConfig, RFState
from .imu_highrate import HighRateIMUPipeline, HighRateIMUConfig, HighRateIMUSample
from .environment import PhysicsEnvironment, PhysicsConfig, PhysicsResult

__all__ = [
    "WindModel", "WindConfig",
    "GPSNoiseModel", "GPSConfig",
    "BatteryModel", "BatteryConfig",
    "IMUNoiseModel", "IMUConfig", "IMUReading",
    "AtmosphereModel", "AtmosphereConfig",
    "LiDARNoiseModel", "LiDARNoiseConfig",
    "ServoLiDARModel", "ServoLiDARConfig",
    "FlightDynamicsModel", "FlightDynamicsConfig", "DynamicsOutput", "AttitudeState",
    "MagnetometerModel", "MagnetometerConfig", "MagReading",
    "MonsoonModel", "MonsoonConfig", "MonsoonState",
    "RFEnvironmentModel", "RFConfig", "RFState",
    "HighRateIMUPipeline", "HighRateIMUConfig", "HighRateIMUSample",
    "PhysicsEnvironment", "PhysicsConfig", "PhysicsResult",
]
