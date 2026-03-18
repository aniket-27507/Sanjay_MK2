"""
Project Sanjay Mk2 - Crowd Density Estimator
==============================================
Grid-based crowd density estimation from drone sensor observations.

Dual-mode operation:
    Mode 1 (Detection-based): Counts individual YOLO person detections
        per grid cell. Effective at low/moderate density (< 4 persons/m2).
    Mode 2 (Model-based): Uses CSRNet/DM-Count density map inference
        when detection count saturates (> saturation_threshold persons
        in a cell's FOV). Falls back to Mode 1 if model unavailable.

Output:
    - density_grid: np.ndarray (rows x cols) of persons/m2
    - CrowdZone list: clustered high-density regions

Grid system matches WorldModel (default 1000x1000m, cell_size=5.0m).

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage

from src.core.types.drone_types import (
    Vector3,
    CrowdCell,
    CrowdZone,
    CrowdDensityLevel,
    FusedObservation,
    classify_density,
)
from src.surveillance.crowd_density_model import CrowdDensityModelInference

logger = logging.getLogger(__name__)

# Default grid parameters (match WorldModel)
DEFAULT_CELL_SIZE = 5.0         # metres per cell
DEFAULT_GRID_WIDTH = 1000.0     # metres
DEFAULT_GRID_HEIGHT = 1000.0    # metres

# Temporal smoothing coefficient (0=full carry, 1=no smoothing)
SMOOTHING_ALPHA = 0.7

# Detection saturation: switch to model when this many persons per cell
DETECTION_SATURATION_THRESHOLD = 20

# Minimum density to include a cell in a CrowdZone
ZONE_DENSITY_THRESHOLD = 2.0   # persons/m2 (MODERATE and above)

# Camera FOV defaults for computing ground footprint
ALPHA_CAMERA_FOV_DEG = 84.0
BETA_CAMERA_FOV_DEG = 50.0


def _fov_radius(altitude: float, fov_deg: float) -> float:
    """Compute ground-level FOV radius from altitude and camera FOV angle."""
    import math
    half_angle = math.radians(fov_deg / 2.0)
    return abs(altitude) * math.tan(half_angle)


class CrowdDensityEstimator:
    """
    Grid-based crowd density estimation.

    Usage:
        estimator = CrowdDensityEstimator()
        estimator.update(fused_obs, drone_pos, altitude)
        grid = estimator.get_density_grid()
        zones = estimator.get_crowd_zones()
    """

    def __init__(
        self,
        grid_width: float = DEFAULT_GRID_WIDTH,
        grid_height: float = DEFAULT_GRID_HEIGHT,
        cell_size: float = DEFAULT_CELL_SIZE,
        smoothing_alpha: float = SMOOTHING_ALPHA,
        model_weights_path: Optional[str] = None,
    ):
        self.cell_size = cell_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cols = int(grid_width / cell_size)
        self.rows = int(grid_height / cell_size)
        self.cell_area = cell_size * cell_size

        self._smoothing = smoothing_alpha

        # Origin offset — center the grid at world (0, 0)
        self._origin_x = -grid_width / 2.0
        self._origin_y = -grid_height / 2.0

        # Density grid: persons/m2
        self._density: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.float64)
        # Person count grid (raw, before density conversion)
        self._count: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.int32)
        # Timestamp of last update per cell
        self._last_update: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.float64)

        # Density model for high-density regions
        self._model = CrowdDensityModelInference(weights_path=model_weights_path)

        # Cached zones (recomputed on update)
        self._zones: List[CrowdZone] = []
        self._last_zone_update: float = 0.0

        logger.info(
            f"CrowdDensityEstimator: {self.cols}x{self.rows} grid, "
            f"cell_size={cell_size}m, model_available={self._model.available}"
        )

    # ==================== COORDINATE HELPERS ====================

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world (x=North, y=East) to grid (row, col)."""
        col = int((x - self._origin_x) / self.cell_size)
        row = int((y - self._origin_y) / self.cell_size)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) to world center (x, y)."""
        x = self._origin_x + (col + 0.5) * self.cell_size
        y = self._origin_y + (row + 0.5) * self.cell_size
        return x, y

    # ==================== CORE UPDATE ====================

    def update(
        self,
        observation: FusedObservation,
        drone_position: Vector3,
        altitude: float,
        raw_frame: Optional[np.ndarray] = None,
        fov_deg: float = ALPHA_CAMERA_FOV_DEG,
    ) -> None:
        """
        Ingest a fused observation and update the density grid.

        Args:
            observation: FusedObservation from sensor fusion pipeline
            drone_position: Current drone position (NED)
            altitude: Drone altitude in metres (positive, not NED z)
            raw_frame: Optional RGB frame for model-based density estimation
            fov_deg: Camera field-of-view in degrees
        """
        now = time.time()
        fov_radius = _fov_radius(altitude, fov_deg)

        # Determine which grid cells are in the drone's FOV
        center_row, center_col = self.world_to_grid(drone_position.x, drone_position.y)
        cell_radius = max(1, int(fov_radius / self.cell_size))

        # Count persons per cell from detections
        cell_counts: Dict[Tuple[int, int], int] = {}
        for det in observation.detected_objects:
            if det.object_type != "person":
                continue
            r, c = self.world_to_grid(det.position.x, det.position.y)
            # Only count if within FOV radius
            dist = ((r - center_row) ** 2 + (c - center_col) ** 2) ** 0.5 * self.cell_size
            if dist <= fov_radius:
                cell_counts[(r, c)] = cell_counts.get((r, c), 0) + 1

        # Check for detection saturation (high density)
        max_count = max(cell_counts.values()) if cell_counts else 0
        use_model = (
            max_count >= DETECTION_SATURATION_THRESHOLD
            and self._model.available
            and raw_frame is not None
        )

        if use_model:
            self._update_from_model(raw_frame, drone_position, altitude, fov_radius, now)
        else:
            self._update_from_detections(cell_counts, center_row, center_col, cell_radius, now)

    def _update_from_detections(
        self,
        cell_counts: Dict[Tuple[int, int], int],
        center_row: int,
        center_col: int,
        cell_radius: int,
        now: float,
    ) -> None:
        """Update density grid from per-cell person detection counts."""
        # Clear cells in FOV that have no detections (they've been observed as empty)
        r_min = max(0, center_row - cell_radius)
        r_max = min(self.rows, center_row + cell_radius + 1)
        c_min = max(0, center_col - cell_radius)
        c_max = min(self.cols, center_col + cell_radius + 1)

        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                dist = ((r - center_row) ** 2 + (c - center_col) ** 2) ** 0.5
                if dist > cell_radius:
                    continue

                new_count = cell_counts.get((r, c), 0)
                new_density = new_count / self.cell_area

                # Apply temporal smoothing
                old_density = self._density[r, c]
                smoothed = self._smoothing * new_density + (1.0 - self._smoothing) * old_density
                self._density[r, c] = smoothed
                self._count[r, c] = new_count
                self._last_update[r, c] = now

    def _update_from_model(
        self,
        raw_frame: np.ndarray,
        drone_position: Vector3,
        altitude: float,
        fov_radius: float,
        now: float,
    ) -> None:
        """Update density grid from CSRNet model density map."""
        density_map = self._model.infer(raw_frame)
        if density_map is None:
            return

        # density_map is (H', W') — map to grid cells in the drone's FOV
        map_h, map_w = density_map.shape
        center_row, center_col = self.world_to_grid(drone_position.x, drone_position.y)
        cell_radius = max(1, int(fov_radius / self.cell_size))

        r_min = max(0, center_row - cell_radius)
        r_max = min(self.rows, center_row + cell_radius + 1)
        c_min = max(0, center_col - cell_radius)
        c_max = min(self.cols, center_col + cell_radius + 1)

        fov_rows = r_max - r_min
        fov_cols = c_max - c_min

        if fov_rows <= 0 or fov_cols <= 0:
            return

        # Resize density map to match FOV grid cells
        from PIL import Image
        dm_img = Image.fromarray(density_map.astype(np.float32))
        dm_resized = np.array(dm_img.resize((fov_cols, fov_rows), Image.BILINEAR))

        # Scale: model output sums to person count; convert to persons/m2
        total_model_count = density_map.sum()
        total_area = fov_rows * fov_cols * self.cell_area
        if total_area > 0 and dm_resized.sum() > 0:
            scale = total_model_count / dm_resized.sum()
            dm_scaled = dm_resized * scale / self.cell_area
        else:
            dm_scaled = dm_resized

        for dr in range(fov_rows):
            for dc in range(fov_cols):
                r, c = r_min + dr, c_min + dc
                new_density = float(dm_scaled[dr, dc])
                old_density = self._density[r, c]
                smoothed = self._smoothing * new_density + (1.0 - self._smoothing) * old_density
                self._density[r, c] = max(0.0, smoothed)
                self._count[r, c] = int(round(smoothed * self.cell_area))
                self._last_update[r, c] = now

    # ==================== QUERIES ====================

    def get_density_grid(self) -> np.ndarray:
        """Return the full density grid (rows x cols, persons/m2)."""
        return self._density.copy()

    def get_density_at(self, position: Vector3) -> float:
        """Get density at a world position."""
        r, c = self.world_to_grid(position.x, position.y)
        return float(self._density[r, c])

    def get_count_grid(self) -> np.ndarray:
        """Return the raw person count grid."""
        return self._count.copy()

    def get_cell(self, row: int, col: int) -> CrowdCell:
        """Get a CrowdCell at (row, col)."""
        density = float(self._density[row, col])
        return CrowdCell(
            row=row,
            col=col,
            density=density,
            density_level=classify_density(density),
            person_count=int(self._count[row, col]),
            timestamp=float(self._last_update[row, col]),
        )

    # ==================== ZONE DETECTION ====================

    def get_crowd_zones(self, threshold: float = ZONE_DENSITY_THRESHOLD) -> List[CrowdZone]:
        """
        Cluster adjacent cells above density threshold into CrowdZones.

        Uses connected-component labeling (scipy.ndimage).
        """
        # Binary mask: cells above threshold
        mask = self._density >= threshold

        if not mask.any():
            self._zones = []
            return []

        # Label connected components (8-connectivity)
        structure = np.ones((3, 3), dtype=np.int32)
        labels, num_features = ndimage.label(mask, structure=structure)

        zones: List[CrowdZone] = []
        for label_id in range(1, num_features + 1):
            component = labels == label_id
            cells = list(zip(*np.where(component)))

            if not cells:
                continue

            # Compute zone statistics
            densities = [float(self._density[r, c]) for r, c in cells]
            counts = [int(self._count[r, c]) for r, c in cells]

            avg_density = float(np.mean(densities))
            peak_density = float(np.max(densities))
            total_persons = sum(counts)

            # Compute center (average position of all cells)
            avg_row = float(np.mean([r for r, c in cells]))
            avg_col = float(np.mean([c for r, c in cells]))
            cx, cy = self.grid_to_world(int(avg_row), int(avg_col))

            zone = CrowdZone(
                center=Vector3(x=cx, y=cy, z=0.0),
                bounding_cells=[(r, c) for r, c in cells],
                avg_density=avg_density,
                peak_density=peak_density,
                total_persons=total_persons,
            )
            zones.append(zone)

        self._zones = zones
        self._last_zone_update = time.time()
        return zones

    def get_total_crowd_count(self) -> int:
        """Estimated total persons across the entire grid."""
        return int(self._count.sum())

    def to_dict(self) -> Dict:
        """Serialize state for WebSocket/GCS transmission."""
        zones = self._zones or self.get_crowd_zones()
        return {
            'grid_rows': self.rows,
            'grid_cols': self.cols,
            'cell_size': self.cell_size,
            'total_persons': self.get_total_crowd_count(),
            'zone_count': len(zones),
            'zones': [z.to_dict() for z in zones],
            'timestamp': time.time(),
        }
