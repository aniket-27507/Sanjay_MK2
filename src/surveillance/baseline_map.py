"""
Project Sanjay Mk2 - Baseline Map
===================================
Stores a reference terrain map for change detection.

The baseline captures a snapshot of the world at a point in time.
Subsequent sensor observations are compared against this baseline
to detect anomalies (new objects, removed objects, thermal anomalies).

Features full-build instantiation and incremental patching.

@author: Prathamesh Hiwarkar
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


class BaselineMap:
    """
    Reference terrain map for change detection.
    
    Each cell stores:
    - Whether it has been surveyed
    - Terrain elevation
    - Known objects at that cell
    """

    def __init__(self, rows: int, cols: int, cell_size: float = 5.0):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # Survey status
        self.surveyed: np.ndarray = np.zeros((rows, cols), dtype=bool)

        # Elevation snapshot
        self.elevation: np.ndarray = np.zeros((rows, cols), dtype=np.float32)

        # Known objects per cell: dict of (row, col) -> list of object_ids
        self._cell_objects: Dict[Tuple[int, int], Set[str]] = {}

        # Global set of known object IDs with their positions
        self._known_objects: Dict[str, dict] = {}

        self._total_cells = rows * cols

    def build_from_world_model(self, world_model) -> None:
        """
        Build a complete baseline from the world model (mapping flight).
        
        Snapshots all terrain elevation and marks all cells as surveyed.
        Records all current objects as "known baseline" objects.
        """
        from src.surveillance.world_model import WorldModel
        wm: WorldModel = world_model

        # Copy elevation
        self.elevation = wm.elevation.copy()

        # Mark all cells as surveyed
        self.surveyed[:] = True

        # Record all objects
        for obj in wm.get_all_objects():
            row, col = wm.world_to_grid(obj.position.x, obj.position.y)
            key = (row, col)
            if key not in self._cell_objects:
                self._cell_objects[key] = set()
            self._cell_objects[key].add(obj.object_id)

            self._known_objects[obj.object_id] = {
                'object_type': obj.object_type,
                'position': (obj.position.x, obj.position.y, obj.position.z),
                'thermal_signature': obj.thermal_signature,
            }

        logger.info("Baseline built: %d cells, %d known objects",
                     self._total_cells, len(self._known_objects))

    def update_from_observation(
        self,
        coverage_cells: List[Tuple[int, int]],
        elevation_data: Optional[np.ndarray] = None,
        detected_object_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Incrementally update baseline from a sensor observation.
        
        Used during patrol to expand the surveyed area.
        """
        for i, (row, col) in enumerate(coverage_cells):
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.surveyed[row, col] = True

                if elevation_data is not None and i < len(elevation_data):
                    # Weighted average with existing data
                    if self.elevation[row, col] != 0:
                        self.elevation[row, col] = (
                            0.7 * self.elevation[row, col] +
                            0.3 * elevation_data[i]
                        )
                    else:
                        self.elevation[row, col] = elevation_data[i]

    def is_surveyed(self, row: int, col: int) -> bool:
        """Check if a cell has been surveyed."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return bool(self.surveyed[row, col])
        return False

    def is_known_object(self, object_id: str) -> bool:
        """Check if an object was in the baseline."""
        return object_id in self._known_objects

    def get_known_objects(self) -> Dict[str, dict]:
        """Get all known baseline objects."""
        return self._known_objects.copy()

    def coverage_percentage(self) -> float:
        """Get percentage of map that has been surveyed."""
        if self._total_cells == 0:
            return 0.0
        return float(np.sum(self.surveyed)) / self._total_cells * 100.0

    def surveyed_cell_count(self) -> int:
        """Get number of surveyed cells."""
        return int(np.sum(self.surveyed))
