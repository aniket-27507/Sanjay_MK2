"""
Project Sanjay Mk2 - Hardware Protection Layer (HPL)
====================================================
Last-resort collision avoidance operating directly on raw sensor data.

The HPL sits below the APF tactical layer and has OVERRIDE authority
over all velocity commands.  When an obstacle breaches the critical
distance threshold, the HPL directly commands the FlightController
to perform an emergency avoidance manoeuvre.

Design Principles:
    - Zero SLAM dependency — uses raw LiDAR range arrays for minimum latency
    - Hard authority — suppresses any velocity command that would
      reduce clearance below the safety margin
    - Deterministic — no ML inference in the hot path
    - Auditable — every override is logged with timestamped telemetry

Architecture:
    Sensor (raw LiDAR) → HPL → (override?) → FlightController → Actuators

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


class HPLState(Enum):
    """HPL operating state."""
    PASSIVE = auto()        # Monitoring, no override
    SUPPRESSING = auto()    # Blocking unsafe commands
    OVERRIDE = auto()       # Commanding emergency manoeuvre
    HARD_STOP = auto()      # Zero-velocity emergency hold


@dataclass
class HPLConfig:
    """Configuration for the Hardware Protection Layer."""

    # ── Distance Thresholds (meters) ──
    critical_distance: float = 1.5      # Override commanded velocity
    hard_stop_distance: float = 0.6     # Zero-velocity hold
    vertical_clearance: float = 2.0     # Min ground / ceiling clearance

    # ── Timing ──
    scan_rate_hz: float = 50.0          # Expected LiDAR update rate
    override_hold_time: float = 0.5     # Seconds to hold override after clear

    # ── Emergency Manoeuvre ──
    escape_speed: float = 2.0           # m/s — emergency retreat speed
    vertical_escape_bias: float = 0.6   # Fraction of escape directed upward

    # ── Ray Configuration ──
    num_sectors: int = 12               # Directional sectors for 360° scan
    sector_angle_deg: float = 30.0      # Per-sector angular width

    # ── Logging ──
    log_overrides: bool = True


@dataclass
class HPLOverrideEvent:
    """Logged when HPL overrides a command."""
    timestamp: float
    state: HPLState
    closest_range: float
    closest_bearing_deg: float
    original_command: Vector3
    override_command: Vector3
    sector_ranges: List[float]


class HardwareProtectionLayer:
    """
    Hardware Protection Layer for Alpha Drones.

    This module acts as an "electronic bumper" — it reads raw LiDAR
    range data at the sensor frame rate and can override any velocity
    command output by the APF or tactical planner if the command would
    drive the drone into a detected obstacle.

    Usage:
        hpl = HardwareProtectionLayer()

        # Feed raw scan at sensor rate
        hpl.update_scan(lidar_ranges)       # np.ndarray of ranges

        # Gate every outgoing velocity command
        safe_vel, was_overridden = hpl.gate_command(
            desired_velocity,
            drone_position,
        )
    """

    def __init__(self, config: Optional[HPLConfig] = None):
        self.config = config or HPLConfig()

        # Raw scan storage
        self._raw_ranges: np.ndarray = np.full(360, 100.0)
        self._sector_mins: np.ndarray = np.full(self.config.num_sectors, 100.0)

        # State
        self._state = HPLState.PASSIVE
        self._last_override_time: float = 0.0

        # Event log (ring buffer of last 100 events)
        self._event_log: List[HPLOverrideEvent] = []
        self._max_log_size: int = 100

    # ── Public Interface ──────────────────────────────────────────

    @property
    def state(self) -> HPLState:
        return self._state

    @property
    def is_overriding(self) -> bool:
        return self._state in (HPLState.SUPPRESSING, HPLState.OVERRIDE, HPLState.HARD_STOP)

    def update_scan(self, ranges: np.ndarray):
        """
        Ingest raw LiDAR range array.

        Args:
            ranges: 1-D array of range measurements (meters).
                    Index 0 = forward (0°), incrementing clockwise.
        """
        self._raw_ranges = np.clip(ranges, 0.0, 200.0)
        self._update_sectors()

    def update_scan_3d(self, points: np.ndarray):
        """
        Ingest a 3D point cloud and project to radial sectors.

        Args:
            points: Nx3 array of (x, y, z) points in body frame.
        """
        if points.shape[0] == 0:
            return

        # Compute horizontal range and bearing for each point
        xy = points[:, :2]
        ranges = np.linalg.norm(xy, axis=1)
        bearings = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360

        # Bin into sectors
        sector_size = 360.0 / self.config.num_sectors
        for i in range(self.config.num_sectors):
            lo = i * sector_size
            hi = lo + sector_size
            mask = (bearings >= lo) & (bearings < hi)
            if np.any(mask):
                self._sector_mins[i] = float(np.min(ranges[mask]))
            else:
                self._sector_mins[i] = 100.0

    def gate_command(
        self,
        desired_velocity: Vector3,
        drone_position: Optional[Vector3] = None,
    ) -> Tuple[Vector3, bool]:
        """
        Gate an outgoing velocity command.

        If the command would drive the drone toward a critically close
        obstacle, it is suppressed or replaced with an emergency manoeuvre.

        Args:
            desired_velocity: Velocity from APF / tactical planner.
            drone_position: Optional for telemetry logging.

        Returns:
            (safe_velocity, was_overridden)
        """
        closest_range, closest_sector = self._get_closest()
        closest_bearing = closest_sector * (360.0 / self.config.num_sectors)
        now = time.time()

        # ── Hard stop ──
        if closest_range < self.config.hard_stop_distance:
            self._state = HPLState.HARD_STOP
            override_vel = Vector3(x=0.0, y=0.0, z=0.0)
            self._log_event(
                closest_range, closest_bearing,
                desired_velocity, override_vel,
            )
            self._last_override_time = now
            logger.critical(
                f"HPL HARD STOP — obstacle at {closest_range:.2f}m "
                f"bearing {closest_bearing:.0f}°"
            )
            return override_vel, True

        # ── Override: retreat away from obstacle ──
        if closest_range < self.config.critical_distance:
            self._state = HPLState.OVERRIDE
            override_vel = self._compute_escape(closest_bearing)
            self._log_event(
                closest_range, closest_bearing,
                desired_velocity, override_vel,
            )
            self._last_override_time = now
            logger.warning(
                f"HPL OVERRIDE — obstacle at {closest_range:.2f}m "
                f"bearing {closest_bearing:.0f}°"
            )
            return override_vel, True

        # ── Suppression: check if desired command heads toward threat ──
        if closest_range < self.config.critical_distance * 2.0:
            if self._command_heads_toward(desired_velocity, closest_bearing):
                self._state = HPLState.SUPPRESSING
                safe_vel = self._suppress_component(
                    desired_velocity, closest_bearing
                )
                self._log_event(
                    closest_range, closest_bearing,
                    desired_velocity, safe_vel,
                )
                self._last_override_time = now
                return safe_vel, True

        # ── Hold override for a short period after clearance ──
        if now - self._last_override_time < self.config.override_hold_time:
            return desired_velocity, False

        # ── All clear ──
        self._state = HPLState.PASSIVE
        return desired_velocity, False

    # ── Sector Analysis ───────────────────────────────────────────

    def _update_sectors(self):
        """Partition raw scan into angular sectors and take min range."""
        n_rays = len(self._raw_ranges)
        rays_per_sector = n_rays // self.config.num_sectors

        for i in range(self.config.num_sectors):
            start = i * rays_per_sector
            end = start + rays_per_sector
            self._sector_mins[i] = float(np.min(self._raw_ranges[start:end]))

    def _get_closest(self) -> Tuple[float, int]:
        """Return (min_range, sector_index)."""
        idx = int(np.argmin(self._sector_mins))
        return float(self._sector_mins[idx]), idx

    # ── Command Manipulation ──────────────────────────────────────

    def _command_heads_toward(
        self, vel: Vector3, obstacle_bearing_deg: float
    ) -> bool:
        """Check if velocity vector is directed toward the obstacle."""
        vel_bearing = math.degrees(math.atan2(vel.y, vel.x)) % 360
        angular_diff = abs(vel_bearing - obstacle_bearing_deg)
        angular_diff = min(angular_diff, 360 - angular_diff)
        return angular_diff < 60.0  # ±60° cone

    def _suppress_component(
        self, vel: Vector3, obstacle_bearing_deg: float
    ) -> Vector3:
        """Remove the velocity component directed toward the obstacle."""
        obs_rad = math.radians(obstacle_bearing_deg)
        obs_dir = np.array([math.cos(obs_rad), math.sin(obs_rad), 0.0])

        v = vel.to_array()
        toward = np.dot(v, obs_dir)

        if toward > 0:
            v -= toward * obs_dir  # Remove approaching component
        return Vector3.from_array(v)

    def _compute_escape(self, obstacle_bearing_deg: float) -> Vector3:
        """Compute an emergency escape velocity away from threat."""
        # Retreat direction = opposite of obstacle bearing
        retreat_rad = math.radians(obstacle_bearing_deg + 180.0)
        horizontal = self.config.escape_speed * (1.0 - self.config.vertical_escape_bias)
        vertical = -self.config.escape_speed * self.config.vertical_escape_bias  # NED up

        return Vector3(
            x=horizontal * math.cos(retreat_rad),
            y=horizontal * math.sin(retreat_rad),
            z=vertical,
        )

    # ── Logging ───────────────────────────────────────────────────

    def _log_event(
        self,
        closest_range: float,
        closest_bearing: float,
        original: Vector3,
        override: Vector3,
    ):
        if not self.config.log_overrides:
            return
        event = HPLOverrideEvent(
            timestamp=time.time(),
            state=self._state,
            closest_range=closest_range,
            closest_bearing_deg=closest_bearing,
            original_command=original,
            override_command=override,
            sector_ranges=self._sector_mins.tolist(),
        )
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log.pop(0)

    def get_event_log(self) -> List[HPLOverrideEvent]:
        return list(self._event_log)

    def get_telemetry(self) -> Dict:
        """Return HPL telemetry."""
        closest_range, closest_sector = self._get_closest()
        return {
            "state": self._state.name,
            "closest_range_m": round(closest_range, 2),
            "closest_sector": closest_sector,
            "sector_mins": [round(s, 2) for s in self._sector_mins.tolist()],
            "override_events_total": len(self._event_log),
            "timestamp": time.time(),
        }
