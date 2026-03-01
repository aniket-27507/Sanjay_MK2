"""
Project Sanjay Mk2 - Simulated Depth Estimator
================================================
Simulates AI monocular depth estimation (e.g. Depth Anything V2).

Takes "ground truth" elevation from the world model and adds
realistic noise that scales with altitude, simulating the accuracy
degradation of real monocular depth networks at higher altitudes.

Usage:
    estimator = SimulatedDepthEstimator()
    depth_grid = estimator.estimate(drone_pos, altitude, world_model)
"""

from __future__ import annotations

import math
import logging
from typing import List, Tuple

import numpy as np

from src.core.types.drone_types import Vector3
from src.surveillance.world_model import WorldModel

logger = logging.getLogger(__name__)


class SimulatedDepthEstimator:
    """
    Simulates AI monocular depth estimation.
    
    Returns a grid of estimated elevations for visible cells.
    Ground truth from WorldModel.get_elevation() plus Gaussian noise
    that scales with altitude.
    """

    def __init__(
        self,
        base_accuracy_stddev: float = 0.5,
        altitude_noise_scale: float = 0.05,
        fov_deg: float = 84.0,
    ):
        """
        Args:
            base_accuracy_stddev: Base noise standard deviation in meters
            altitude_noise_scale: Additional noise per meter of altitude
            fov_deg: FOV matching the RGB camera
        """
        self.base_accuracy_stddev = base_accuracy_stddev
        self.altitude_noise_scale = altitude_noise_scale
        self.fov_deg = fov_deg

    def estimate(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Estimate depth/elevation for visible cells.
        
        Args:
            drone_position: Drone XY position
            altitude: Drone altitude AGL
            world_model: World to query
            
        Returns:
            (depth_grid, cell_list) where depth_grid[i] is the estimated
            elevation for cell_list[i].
        """
        # Get visible cells
        footprint_radius = altitude * math.tan(math.radians(self.fov_deg / 2.0))
        cells = world_model._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint_radius
        )

        if not cells:
            return np.array([]), []

        # Get ground truth elevations
        ground_truth = np.array([
            world_model.elevation[r, c] for r, c in cells
        ], dtype=np.float32)

        # Add noise that scales with altitude
        noise_stddev = self.base_accuracy_stddev + altitude * self.altitude_noise_scale
        noise = np.random.normal(0.0, noise_stddev, size=ground_truth.shape).astype(np.float32)

        estimated = ground_truth + noise

        return estimated, cells

    def get_accuracy_at_altitude(self, altitude: float) -> float:
        """Get expected accuracy (stddev in meters) at a given altitude."""
        return self.base_accuracy_stddev + altitude * self.altitude_noise_scale
