"""
Project Sanjay Mk2 - 3D LiDAR Driver
=====================================
Interface for 3D LiDAR sensor (simulated via Isaac Sim RTX LiDAR
or real hardware via ROS 2 sensor_msgs/PointCloud2).

Provides:
    - Raw point cloud ingestion (from ROS or sim)
    - Ground plane removal
    - DBSCAN-based 3D obstacle clustering
    - Sector-based range computation for HPL

Isaac Sim Note:
    Isaac Sim's RTX LiDAR publishes to a PointCloud2 topic.
    This driver consumes that data either through the ROS 2 bridge
    or directly from the sim's NumPy output.

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import Vector3
from src.single_drone.obstacle_avoidance.apf_3d import Obstacle3D

logger = logging.getLogger(__name__)


@dataclass
class Lidar3DConfig:
    """Configuration for the 3D LiDAR sensor."""

    # ── Sensor Specifications ──
    max_range: float = 30.0         # Maximum sensor range (m)
    min_range: float = 0.3          # Minimum sensor range (m)
    num_channels: int = 16          # Vertical channels (e.g. VLP-16)
    horizontal_fov: float = 360.0   # Horizontal FOV (degrees)
    vertical_fov: float = 30.0      # Vertical FOV (degrees)
    scan_rate_hz: float = 10.0      # Scans per second

    # ── Ground Removal ──
    ground_height_threshold: float = 0.3   # Points below this are ground (m)
    ground_removal: bool = True

    # ── Clustering (DBSCAN-like) ──
    cluster_eps: float = 0.8        # Max point-to-point distance (m)
    cluster_min_points: int = 5     # Min points to form a cluster
    max_clusters: int = 50          # Cap cluster output

    # ── Sector Mapping (for HPL) ──
    num_sectors: int = 12           # Horizontal sectors for range map


class Lidar3DDriver:
    """
    3D LiDAR driver for Alpha Drones.

    Consumes raw 3D point clouds and produces:
        1. Clustered Obstacle3D objects (for APF)
        2. Sector-minimum range arrays (for HPL)

    Usage:
        lidar = Lidar3DDriver()

        # Feed point cloud (Nx3 numpy array in body frame)
        lidar.update_points(point_cloud)

        # Get clustered obstacles for APF
        obstacles = lidar.get_obstacles()

        # Get sector ranges for HPL
        sector_ranges = lidar.get_sector_ranges()
    """

    def __init__(self, config: Optional[Lidar3DConfig] = None):
        self.config = config or Lidar3DConfig()

        # Raw data
        self._raw_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._filtered_points: np.ndarray = np.empty((0, 3), dtype=np.float32)

        # Processed data
        self._obstacles: List[Obstacle3D] = []
        self._sector_ranges: np.ndarray = np.full(
            self.config.num_sectors, self.config.max_range
        )
        self._last_update: float = 0.0

    # ── Public Interface ──────────────────────────────────────────

    def update_points(self, points: np.ndarray, drone_position: Optional[Vector3] = None):
        """
        Ingest a raw 3D point cloud.

        Args:
            points: Nx3 array of (x, y, z) in the sensor/body frame.
                    x = forward, y = left, z = up.
            drone_position: Optional world position for obstacle world-frame output.
        """
        if points.shape[0] == 0:
            self._raw_points = points
            self._filtered_points = points
            self._obstacles = []
            return

        self._raw_points = points
        self._last_update = time.time()

        # Range filter
        ranges = np.linalg.norm(points, axis=1)
        valid = (ranges >= self.config.min_range) & (ranges <= self.config.max_range)
        filtered = points[valid]

        # Ground removal
        if self.config.ground_removal:
            non_ground = filtered[:, 2] > self.config.ground_height_threshold
            filtered = filtered[non_ground]

        self._filtered_points = filtered

        # Cluster obstacles
        self._obstacles = self._cluster_obstacles(filtered, drone_position)

        # Build sector range map
        self._sector_ranges = self._build_sector_ranges(filtered)

    def update_from_ros_pointcloud2(self, data: bytes, width: int, height: int):
        """
        Parse a ROS PointCloud2 message in XYZ32F format.

        Args:
            data: Raw byte buffer from the message.
            width, height: Dimensions of the point cloud.
        """
        n_points = width * height
        points = np.frombuffer(data, dtype=np.float32).reshape(n_points, -1)[:, :3]
        self.update_points(points)

    def get_obstacles(self) -> List[Obstacle3D]:
        """Get clustered obstacles for APF."""
        return list(self._obstacles)

    def get_sector_ranges(self) -> np.ndarray:
        """Get sector minimum ranges for HPL (copy)."""
        return self._sector_ranges.copy()

    def get_filtered_points(self) -> np.ndarray:
        """Get filtered point cloud."""
        return self._filtered_points.copy()

    @property
    def point_count(self) -> int:
        return self._filtered_points.shape[0]

    @property
    def obstacle_count(self) -> int:
        return len(self._obstacles)

    @property
    def last_update_time(self) -> float:
        return self._last_update

    # ── Obstacle Clustering ───────────────────────────────────────

    def _cluster_obstacles(
        self,
        points: np.ndarray,
        drone_position: Optional[Vector3] = None,
    ) -> List[Obstacle3D]:
        """
        Simple grid-based + connected-components clustering.

        Uses a voxel grid approach for O(n) performance instead of
        full DBSCAN O(n²).
        """
        if points.shape[0] < self.config.cluster_min_points:
            return []

        eps = self.config.cluster_eps
        labels = self._grid_cluster(points, eps)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # Noise points

        obstacles: List[Obstacle3D] = []
        for label in sorted(unique_labels):
            if len(obstacles) >= self.config.max_clusters:
                break

            mask = labels == label
            cluster_points = points[mask]

            if cluster_points.shape[0] < self.config.cluster_min_points:
                continue

            center = cluster_points.mean(axis=0)
            # Bounding sphere radius
            dists = np.linalg.norm(cluster_points - center, axis=1)
            radius = float(np.max(dists)) + 0.1  # Small padding

            # Convert center to world frame if drone position given
            if drone_position is not None:
                world_center = Vector3(
                    x=drone_position.x + center[0],
                    y=drone_position.y + center[1],
                    z=drone_position.z + center[2],
                )
            else:
                world_center = Vector3(x=float(center[0]), y=float(center[1]), z=float(center[2]))

            obstacles.append(Obstacle3D(
                position=world_center,
                radius=radius,
                confidence=min(cluster_points.shape[0] / 20.0, 1.0),
            ))

        return obstacles

    def _grid_cluster(self, points: np.ndarray, eps: float) -> np.ndarray:
        """
        Fast grid-based clustering.

        Maps each point to a voxel, then uses connected-component
        labeling on adjacent voxels.
        """
        # Quantize to grid
        grid_coords = np.floor(points / eps).astype(np.int32)

        # Build voxel → point index map
        voxel_map: dict = {}
        for i, gc in enumerate(grid_coords):
            key = (int(gc[0]), int(gc[1]), int(gc[2]))
            if key not in voxel_map:
                voxel_map[key] = []
            voxel_map[key].append(i)

        # Connected components via BFS
        labels = np.full(points.shape[0], -1, dtype=np.int32)
        current_label = 0
        visited_voxels: set = set()

        for voxel_key in voxel_map:
            if voxel_key in visited_voxels:
                continue

            # BFS flood fill
            queue = [voxel_key]
            visited_voxels.add(voxel_key)
            component_indices: List[int] = []

            while queue:
                current_voxel = queue.pop(0)
                component_indices.extend(voxel_map[current_voxel])

                # Check 26-connected neighbors
                cx, cy, cz = current_voxel
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            neighbor = (cx + dx, cy + dy, cz + dz)
                            if neighbor in voxel_map and neighbor not in visited_voxels:
                                visited_voxels.add(neighbor)
                                queue.append(neighbor)

            # Assign label
            for idx in component_indices:
                labels[idx] = current_label
            current_label += 1

        return labels

    # ── Sector Range Map ──────────────────────────────────────────

    def _build_sector_ranges(self, points: np.ndarray) -> np.ndarray:
        """Build sector-minimum ranges for HPL."""
        sector_ranges = np.full(self.config.num_sectors, self.config.max_range)

        if points.shape[0] == 0:
            return sector_ranges

        # Compute bearing and horizontal range
        xy = points[:, :2]
        ranges = np.linalg.norm(xy, axis=1)
        bearings = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360

        sector_size = 360.0 / self.config.num_sectors
        for i in range(self.config.num_sectors):
            lo = i * sector_size
            hi = lo + sector_size
            mask = (bearings >= lo) & (bearings < hi)
            if np.any(mask):
                sector_ranges[i] = float(np.min(ranges[mask]))

        return sector_ranges

    # ── Telemetry ─────────────────────────────────────────────────

    def get_telemetry(self) -> dict:
        return {
            "raw_points": self._raw_points.shape[0],
            "filtered_points": self._filtered_points.shape[0],
            "obstacle_count": len(self._obstacles),
            "sector_ranges": [round(s, 2) for s in self._sector_ranges.tolist()],
            "last_update": self._last_update,
        }
