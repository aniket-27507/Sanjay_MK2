"""
RF interference model for WiFi mesh comms and GPS signal quality.

Models the urban RF environment in Guwahati for cheap drone hardware:
- WiFi mesh signal strength (Pi WiFi, 2.4 GHz)
- Path loss + urban multipath fading
- Building shadowing (signal blocked by structures)
- GPS signal degradation zones (urban canyons, near cell towers)
- GPS constellation geometry (HDOP from visible satellite count)
- Comms dropout prediction

Guwahati urban RF environment:
  Dense cell tower deployment (Jio, Airtel, BSNL) — potential GPS L1 interference
  Concrete+rebar buildings — strong RF shadowing
  High humidity — marginal increase in 2.4GHz attenuation
  Power lines — 50Hz EMI on unshielded GPS antennas
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3


@dataclass
class RFConfig:
    # WiFi mesh (2.4 GHz, Pi onboard)
    wifi_tx_power_dbm: float = 20.0
    wifi_freq_ghz: float = 2.4
    wifi_sensitivity_dbm: float = -80.0
    wifi_gcs_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Path loss exponent (free space = 2.0, urban = 2.7-3.5)
    path_loss_exponent: float = 3.2
    reference_distance_m: float = 1.0

    # Building attenuation (per wall/floor)
    building_wall_loss_db: float = 12.0
    building_shadow_margin_m: float = 5.0

    # Rayleigh fading (urban multipath)
    fading_sigma_db: float = 4.0

    # Humidity attenuation (marginal at 2.4 GHz)
    humidity_loss_db_per_100m: float = 0.1

    # GPS quality parameters
    gps_base_satellites: int = 10
    gps_urban_canyon_sat_loss: int = 4
    gps_min_satellites: int = 4
    gps_hdop_per_sat: float = 0.3
    gps_base_hdop: float = 1.2

    # GPS interference
    gps_cell_tower_degradation_m: float = 1.5
    gps_cell_tower_radius_m: float = 50.0
    gps_powerline_noise_m: float = 0.5

    # Rain effect on RF
    rain_attenuation_db_per_mmhr: float = 0.01

    seed: Optional[int] = None


@dataclass
class RFState:
    wifi_rssi_dbm: float = -50.0
    wifi_link_quality_pct: float = 100.0
    wifi_connected: bool = True
    gps_visible_sats: int = 10
    gps_hdop: float = 1.2
    gps_position_sigma_m: float = 2.5
    gps_quality_pct: float = 100.0
    comms_dropout: bool = False


class RFEnvironmentModel:
    """
    Models RF signal propagation for WiFi mesh and GPS in urban Guwahati.
    """

    def __init__(self, config: RFConfig | None = None):
        self.config = config or RFConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._time = 0.0

    def _free_space_path_loss(self, distance_m: float) -> float:
        """Log-distance path loss model."""
        if distance_m < 0.1:
            distance_m = 0.1
        cfg = self.config
        # FSPL at reference distance
        fspl_ref = 20 * math.log10(cfg.reference_distance_m) + \
                   20 * math.log10(cfg.wifi_freq_ghz * 1e9) + \
                   20 * math.log10(4 * math.pi / 3e8)
        # Log-distance model
        pl = fspl_ref + 10 * cfg.path_loss_exponent * math.log10(
            distance_m / cfg.reference_distance_m
        )
        return pl

    def _building_obstruction_loss(
        self,
        drone_pos: Vector3,
        gcs_pos: Tuple[float, float, float],
        buildings: List[Tuple[Vector3, float]] | None,
    ) -> float:
        """
        Estimate signal loss from buildings between drone and GCS.
        Simple: count buildings whose bounding region intersects the LOS.
        """
        if not buildings:
            return 0.0

        loss = 0.0
        d_pos = np.array([drone_pos.x, drone_pos.y, drone_pos.z])
        g_pos = np.array(gcs_pos)
        los_vec = g_pos - d_pos
        los_len = np.linalg.norm(los_vec)
        if los_len < 1.0:
            return 0.0
        los_dir = los_vec / los_len

        for center, width in buildings:
            b_pos = np.array([center.x, center.y, center.z])
            # Project building center onto LOS line
            to_bld = b_pos - d_pos
            proj = np.dot(to_bld, los_dir)
            if proj < 0 or proj > los_len:
                continue
            closest = d_pos + los_dir * proj
            dist_to_los = np.linalg.norm(b_pos - closest)
            effective_radius = width / 2.0 + self.config.building_shadow_margin_m

            if dist_to_los < effective_radius:
                penetration = 1.0 - (dist_to_los / effective_radius)
                loss += self.config.building_wall_loss_db * penetration

        return loss

    def _rayleigh_fading(self) -> float:
        """Random multipath fading (log-normal shadowing)."""
        return self._rng.normal(0, self.config.fading_sigma_db)

    def _gps_urban_degradation(
        self,
        drone_pos: Vector3,
        buildings: List[Tuple[Vector3, float]] | None,
    ) -> Tuple[int, float]:
        """
        Compute GPS satellite visibility and HDOP degradation
        from urban canyon effects.
        """
        cfg = self.config
        sat_loss = 0
        hdop_penalty = 0.0

        if buildings:
            nearby_buildings = 0
            for center, width in buildings:
                dx = drone_pos.x - center.x
                dy = drone_pos.y - center.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < width * 2.0:
                    nearby_buildings += 1
                    # Taller buildings block more sky
                    height = abs(center.z) if center.z != 0 else 15.0
                    if height > abs(drone_pos.z):
                        elevation_block = math.atan2(
                            height - abs(drone_pos.z), max(dist, 1.0)
                        )
                        hdop_penalty += elevation_block * 0.5

            sat_loss = min(
                cfg.gps_urban_canyon_sat_loss,
                nearby_buildings,
            )

        visible_sats = max(
            cfg.gps_min_satellites,
            cfg.gps_base_satellites - sat_loss + int(self._rng.normal(0, 0.5)),
        )

        hdop = cfg.gps_base_hdop + \
               (cfg.gps_base_satellites - visible_sats) * cfg.gps_hdop_per_sat + \
               hdop_penalty

        return visible_sats, max(0.8, hdop)

    def compute_rf_state(
        self,
        drone_pos: Vector3,
        dt: float,
        buildings: List[Tuple[Vector3, float]] | None = None,
        rain_intensity_mmhr: float = 0.0,
    ) -> RFState:
        """Compute current RF environment state for a drone."""
        self._time += dt
        cfg = self.config

        # --- WiFi signal strength ---
        gcs = np.array(cfg.wifi_gcs_position)
        drone = np.array([drone_pos.x, drone_pos.y, drone_pos.z])
        distance = float(np.linalg.norm(drone - gcs))

        path_loss = self._free_space_path_loss(distance)
        bld_loss = self._building_obstruction_loss(drone_pos, cfg.wifi_gcs_position, buildings)
        rain_loss = rain_intensity_mmhr * cfg.rain_attenuation_db_per_mmhr * (distance / 100.0)
        humidity_loss = cfg.humidity_loss_db_per_100m * (distance / 100.0)
        fading = self._rayleigh_fading()

        rssi = cfg.wifi_tx_power_dbm - path_loss - bld_loss - rain_loss - humidity_loss + fading

        # Link quality (0-100%)
        range_db = cfg.wifi_tx_power_dbm - cfg.wifi_sensitivity_dbm
        quality = max(0.0, min(100.0, (rssi - cfg.wifi_sensitivity_dbm) / range_db * 100.0))
        connected = rssi > cfg.wifi_sensitivity_dbm

        # --- GPS quality ---
        visible_sats, hdop = self._gps_urban_degradation(drone_pos, buildings)

        # Position sigma from HDOP
        base_sigma = 2.5
        gps_sigma = base_sigma * (hdop / cfg.gps_base_hdop)

        # Rain degrades GPS marginally (ionospheric scintillation proxy)
        if rain_intensity_mmhr > 20.0:
            gps_sigma *= 1.0 + (rain_intensity_mmhr - 20.0) * 0.005

        gps_quality = max(0.0, min(100.0, (visible_sats / cfg.gps_base_satellites) * 100.0))

        # Comms dropout (transient loss)
        dropout = not connected or quality < 10.0

        return RFState(
            wifi_rssi_dbm=rssi,
            wifi_link_quality_pct=quality,
            wifi_connected=connected,
            gps_visible_sats=visible_sats,
            gps_hdop=hdop,
            gps_position_sigma_m=gps_sigma,
            gps_quality_pct=gps_quality,
            comms_dropout=dropout,
        )
