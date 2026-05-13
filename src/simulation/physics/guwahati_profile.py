"""
Guwahati environment profile — factory for PhysicsConfig tuned to
real-world conditions at the Ganeshguri demo site.

Ganeshguri, Guwahati, Assam, India:
  Latitude:  26.1445°N
  Longitude: 91.7362°E
  Elevation: ~55m ASL
  Climate:   Subtropical monsoon (Cwa)
  June:      Pre-monsoon → monsoon onset
             Temperature 25-35°C, humidity 80-95%
             Avg rainfall 315mm/month (intense bursts)
             SW monsoon wind 8-15 km/h, gusts to 40 km/h

Geomagnetic field (IGRF-13, epoch 2026):
  Total intensity: ~46,200 nT
  Declination:     ~0.1°W (nearly true north — unusual, most places have offset)
  Inclination:     ~39.5° (moderate dip — not as steep as higher latitudes)
  North (X):       ~35,600 nT
  East (Y):        ~-60 nT
  Down (Z):        ~29,400 nT

RF environment:
  Dense urban: Jio/Airtel/BSNL towers, concrete+rebar buildings
  2.4GHz WiFi: significant multipath from buildings
  GPS L1: marginal interference from dense cell infrastructure
"""

from __future__ import annotations

from .environment import PhysicsConfig
from .wind_model import WindConfig
from .gps_model import GPSConfig
from .battery_model import BatteryConfig
from .imu_model import IMUConfig
from .atmosphere_model import AtmosphereConfig
from .lidar_noise_model import LiDARNoiseConfig
from .servo_lidar_model import ServoLiDARConfig
from .flight_dynamics import FlightDynamicsConfig
from .magnetometer_model import MagnetometerConfig
from .monsoon_model import MonsoonConfig, RainIntensity
from .rf_model import RFConfig
from .imu_highrate import HighRateIMUConfig


def guwahati_june_clear() -> PhysicsConfig:
    """Clear pre-monsoon day — hot, humid, moderate wind, no rain."""
    return PhysicsConfig(
        enabled=True,
        wind=WindConfig(
            base_speed_ms=3.5,
            base_direction_deg=225.0,
            gust_max_ms=5.0,
            gust_probability_per_sec=0.1,
            turbulence_intensity=0.25,
            building_turbulence_multiplier=2.0,
            drone_mass_kg=0.5,
        ),
        gps=GPSConfig(
            horizontal_sigma_m=2.5,
            multipath_extra_sigma_m=5.0,
        ),
        battery=BatteryConfig(
            capacity_mah=2200,
            cell_count=3,
            internal_resistance_ohm=0.08,
        ),
        imu=IMUConfig(
            gyro_bias_instability_dps=0.04,
            accel_vibration_amplitude_ms2=0.8,
            sample_rate_hz=400.0,
        ),
        atmosphere=AtmosphereConfig(
            temperature_c=33.0,
            relative_humidity_pct=82.0,
            station_altitude_asl_m=55.0,
        ),
        flight_dynamics=FlightDynamicsConfig(mass_kg=0.5),
        magnetometer=MagnetometerConfig(
            igrf_north_nt=35600.0,
            igrf_east_nt=-60.0,
            igrf_down_nt=29400.0,
            hard_iron_ut=(12.0, -8.0, 5.0),
            building_anomaly_max_ut=15.0,
            powerline_amplitude_ut=0.8,
            motor_interference_ut=5.0,
        ),
        monsoon=MonsoonConfig(
            initial_intensity=RainIntensity.DRY,
            burst_probability_per_min=0.01,
        ),
        rf=RFConfig(
            path_loss_exponent=3.2,
            building_wall_loss_db=12.0,
            fading_sigma_db=4.0,
        ),
        highrate_imu=HighRateIMUConfig(
            imu_rate_hz=400.0,
            mag_rate_hz=100.0,
            imu=IMUConfig(sample_rate_hz=400.0),
        ),
        enable_highrate_imu=True,
        enable_monsoon=False,
        enable_rf=True,
    )


def guwahati_june_premonsoon_burst() -> PhysicsConfig:
    """Pre-monsoon thunderstorm — sudden heavy rain, strong wind, GPS degradation."""
    cfg = guwahati_june_clear()
    cfg.wind.base_speed_ms = 8.0
    cfg.wind.gust_max_ms = 12.0
    cfg.wind.gust_probability_per_sec = 0.3
    cfg.wind.turbulence_intensity = 0.5
    cfg.wind.building_turbulence_multiplier = 3.0
    cfg.atmosphere.temperature_c = 28.0
    cfg.atmosphere.relative_humidity_pct = 95.0
    cfg.monsoon = MonsoonConfig(
        initial_intensity=RainIntensity.HEAVY,
        burst_probability_per_min=0.05,
        burst_peak_mmhr=85.0,
        burst_duration_range_sec=(300, 1200),
    )
    cfg.enable_monsoon = True
    return cfg


def guwahati_june_monsoon_steady() -> PhysicsConfig:
    """Active monsoon — steady moderate rain, reduced visibility, wet props."""
    cfg = guwahati_june_clear()
    cfg.wind.base_speed_ms=6.0
    cfg.wind.gust_max_ms=8.0
    cfg.wind.turbulence_intensity=0.35
    cfg.atmosphere.temperature_c=26.0
    cfg.atmosphere.relative_humidity_pct=95.0
    cfg.monsoon = MonsoonConfig(
        initial_intensity=RainIntensity.MODERATE,
        burst_probability_per_min=0.03,
        burst_peak_mmhr=50.0,
        drizzle_background_mmhr=8.0,
    )
    cfg.enable_monsoon = True
    return cfg


def guwahati_night_clear() -> PhysicsConfig:
    """Clear night — cooler, calmer wind, thermal signature advantage."""
    cfg = guwahati_june_clear()
    cfg.wind.base_speed_ms = 2.0
    cfg.wind.gust_max_ms = 3.0
    cfg.wind.turbulence_intensity = 0.15
    cfg.atmosphere.temperature_c = 25.0
    cfg.atmosphere.relative_humidity_pct = 90.0
    return cfg


GUWAHATI_PROFILES = {
    "june_clear": guwahati_june_clear,
    "june_premonsoon_burst": guwahati_june_premonsoon_burst,
    "june_monsoon_steady": guwahati_june_monsoon_steady,
    "night_clear": guwahati_night_clear,
}
