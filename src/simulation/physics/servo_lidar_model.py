"""
2D LiDAR on servo tilt — scan-then-move obstacle avoidance model.

Simulates an RPLiDAR A1 (~$100, 360° 2D, 12m range) mounted on a
servo that tilts through a range of angles to build a pseudo-3D
point cloud of the environment.

Workflow:
1. Drone approaches obstacle (detected via continuous 2D horizontal scan)
2. Drone stops and hovers
3. Servo sweeps LiDAR through tilt_min to tilt_max in steps
4. At each tilt angle, LiDAR does full 360° horizontal sweep
5. All returns accumulated into a 3D point cloud (body frame)
6. Point cloud passed to obstacle avoidance for path planning
7. Drone proceeds on avoidance path
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3
from .lidar_noise_model import LiDARNoiseModel, LiDARNoiseConfig, NoisyLiDARReturn


@dataclass
class ServoLiDARConfig:
    horizontal_rays: int = 360
    max_range_m: float = 12.0
    min_range_m: float = 0.15
    scan_rate_hz: float = 5.5
    servo_tilt_min_deg: float = -30.0
    servo_tilt_max_deg: float = 30.0
    servo_tilt_steps: int = 13
    servo_sweep_speed_deg_per_sec: float = 60.0
    obstacle_detection_threshold_m: float = 10.0
    num_sectors: int = 12
    noise_config: LiDARNoiseConfig = field(default_factory=LiDARNoiseConfig)


@dataclass
class ScanPoint:
    """Single point in the accumulated 3D point cloud (body frame)."""
    x: float
    y: float
    z: float
    range_m: float
    h_angle_deg: float
    tilt_angle_deg: float
    is_false: bool = False


@dataclass
class SectorRange:
    """Min range for a horizontal sector (for HPL compatibility)."""
    sector_id: int
    angle_center_deg: float
    min_range_m: float
    num_returns: int


class ServoLiDARModel:
    """
    Simulates the 2D LiDAR + servo for Blender raycasting validation.
    Can operate in two modes:
    - Continuous 2D: single horizontal scan plane (patrol mode)
    - Full servo sweep: pseudo-3D scan (obstacle investigation mode)
    """

    def __init__(self, config: ServoLiDARConfig | None = None):
        self.config = config or ServoLiDARConfig()
        self._noise = LiDARNoiseModel(self.config.noise_config)
        self._tilt_angles = np.linspace(
            self.config.servo_tilt_min_deg,
            self.config.servo_tilt_max_deg,
            self.config.servo_tilt_steps,
        ).tolist()
        self._h_angles = np.linspace(0, 360, self.config.horizontal_rays, endpoint=False).tolist()

    def sweep_duration_sec(self) -> float:
        """Time required for a full servo sweep."""
        total_arc = self.config.servo_tilt_max_deg - self.config.servo_tilt_min_deg
        return total_arc / self.config.servo_sweep_speed_deg_per_sec

    def generate_ray_directions(self, tilt_deg: float) -> List[Tuple[float, float, np.ndarray]]:
        """
        Generate ray direction vectors for one horizontal scan at a given tilt.
        Returns list of (h_angle_deg, tilt_deg, direction_vector_body_frame).
        """
        tilt_rad = math.radians(tilt_deg)
        rays = []
        for h_deg in self._h_angles:
            h_rad = math.radians(h_deg)
            dx = math.cos(tilt_rad) * math.cos(h_rad)
            dy = math.cos(tilt_rad) * math.sin(h_rad)
            dz = math.sin(tilt_rad)
            rays.append((h_deg, tilt_deg, np.array([dx, dy, dz])))
        return rays

    def generate_full_sweep_rays(self) -> List[Tuple[float, float, np.ndarray]]:
        """Generate all rays for a complete servo sweep (all tilt steps)."""
        all_rays = []
        for tilt in self._tilt_angles:
            actual_tilt = self._noise.apply_servo_noise(tilt)
            all_rays.extend(self.generate_ray_directions(actual_tilt))
        return all_rays

    def process_returns(
        self,
        raw_returns: List[Tuple[float, float, float, bool]],
    ) -> Tuple[List[ScanPoint], List[SectorRange]]:
        """
        Process raw raycast results into a point cloud + sector ranges.
        raw_returns: list of (h_angle_deg, tilt_deg, true_range_m, hit_obstacle)

        Returns:
            point_cloud: List of 3D scan points in body frame
            sector_ranges: Per-sector minimum ranges (for HPL/APF)
        """
        point_cloud: List[ScanPoint] = []
        sector_mins: dict[int, float] = {i: self.config.max_range_m for i in range(self.config.num_sectors)}
        sector_counts: dict[int, int] = {i: 0 for i in range(self.config.num_sectors)}

        for h_deg, tilt_deg, true_range, hit_obstacle in raw_returns:
            noisy_ret = self._noise.apply_range_noise(true_range if hit_obstacle else self.config.max_range_m + 1)

            if not noisy_ret.is_valid:
                continue

            r = noisy_ret.range_m
            h_rad = math.radians(h_deg)
            t_rad = math.radians(tilt_deg)

            px = r * math.cos(t_rad) * math.cos(h_rad)
            py = r * math.cos(t_rad) * math.sin(h_rad)
            pz = r * math.sin(t_rad)

            point_cloud.append(ScanPoint(
                x=px, y=py, z=pz,
                range_m=r, h_angle_deg=h_deg, tilt_angle_deg=tilt_deg,
                is_false=noisy_ret.is_false_return,
            ))

            sector_id = int(h_deg / (360.0 / self.config.num_sectors)) % self.config.num_sectors
            if r < sector_mins[sector_id]:
                sector_mins[sector_id] = r
            sector_counts[sector_id] += 1

        sector_ranges = []
        for sid in range(self.config.num_sectors):
            sector_ranges.append(SectorRange(
                sector_id=sid,
                angle_center_deg=(sid + 0.5) * (360.0 / self.config.num_sectors),
                min_range_m=sector_mins[sid],
                num_returns=sector_counts[sid],
            ))

        return point_cloud, sector_ranges

    def continuous_2d_scan_rays(self) -> List[Tuple[float, float, np.ndarray]]:
        """Generate rays for a single horizontal scan (patrol mode, tilt=0)."""
        return self.generate_ray_directions(tilt_deg=0.0)

    def should_investigate(self, sector_ranges: List[SectorRange], min_returns: int = 5) -> bool:
        """Check if any sector has a reliable obstacle detection within threshold.

        Requires min_returns in the sector to filter false returns from noise.
        A real wall at 10m with 30 rays/sector produces ~28 returns; a false
        return produces 1-2.
        """
        for sr in sector_ranges:
            if (sr.min_range_m < self.config.obstacle_detection_threshold_m
                    and sr.num_returns >= min_returns):
                return True
        return False
