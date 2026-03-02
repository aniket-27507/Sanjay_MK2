"""
Project Sanjay Mk2 - World Model
=================================
Procedural 2D grid world representing the terrain and dynamic objects.

This is the "ground truth" that simulated sensors query against.

World Coordinate System:
    - Hexagonal bounding box generated around mathematical center.
    - Objects implement full lifecycle models (spawn, persist, purge).

@author: Archishman Paul
"""

from __future__ import annotations

import math
import random
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """Terrain classification for each grid cell."""
    OPEN_GROUND = auto()
    ROAD = auto()
    BUILDING = auto()
    VEGETATION = auto()
    WATER = auto()


@dataclass
class WorldObject:
    """
    A dynamic object in the world.
    
    Can represent people, vehicles, camps, equipment, etc.
    Objects may be threats or benign.
    """
    object_id: str
    object_type: str            # "person", "vehicle", "camp", "equipment"
    position: Vector3
    thermal_signature: float    # 0.0 (ambient) - 1.0 (very hot)
    is_threat: bool = False
    visible: bool = True
    size: float = 1.0           # meters, affects detection probability
    spawn_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            'object_id': self.object_id,
            'object_type': self.object_type,
            'position': {'x': self.position.x, 'y': self.position.y, 'z': self.position.z},
            'thermal_signature': self.thermal_signature,
            'is_threat': self.is_threat,
            'visible': self.visible,
            'size': self.size,
        }


# Default thermal signatures by object type
THERMAL_SIGNATURES = {
    'person': 0.85,
    'vehicle': 0.70,
    'camp': 0.60,
    'equipment': 0.40,
    'animal': 0.75,
}

# Default sizes by object type (meters)
OBJECT_SIZES = {
    'person': 1.8,
    'vehicle': 4.5,
    'camp': 8.0,
    'equipment': 2.0,
    'animal': 1.0,
}


class WorldModel:
    """
    Grid-based world model with terrain and dynamic objects.
    
    The world is a rectangular grid. Each cell has:
    - terrain type (open, road, building, vegetation, water)
    - elevation (height above datum)
    
    Dynamic objects are overlaid on the grid and can be spawned/removed.
    """

    def __init__(self, width: float = 1000.0, height: float = 1000.0, cell_size: float = 5.0):
        """
        Args:
            width: World width in meters
            height: World height in meters
            cell_size: Grid cell size in meters
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.cols = int(width / cell_size)
        self.rows = int(height / cell_size)

        # Terrain grid: terrain type per cell
        self.terrain: np.ndarray = np.full((self.rows, self.cols), TerrainType.OPEN_GROUND.value, dtype=np.int8)
        # Elevation grid: meters above datum
        self.elevation: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Dynamic objects
        self._objects: Dict[str, WorldObject] = {}
        self._object_counter = 0

        # World origin offset (center the world at 0,0)
        self._origin_x = -width / 2.0
        self._origin_y = -height / 2.0

        logger.info(f"WorldModel created: {self.cols}x{self.rows} grid, {cell_size}m cells")

    # ==================== COORDINATE CONVERSION ====================

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid (row, col)."""
        col = int((x - self._origin_x) / self.cell_size)
        row = int((y - self._origin_y) / self.cell_size)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) to world coordinates (center of cell)."""
        x = self._origin_x + (col + 0.5) * self.cell_size
        y = self._origin_y + (row + 0.5) * self.cell_size
        return x, y

    # ==================== TERRAIN GENERATION ====================

    def generate_terrain(self, seed: int = 42):
        """
        Generate procedural terrain with buildings, roads, vegetation, and water.
        
        Places features within the hex patrol area (~50m radius from center).
        """
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Base: slight elevation variation (0-5m)
        self.elevation = np_rng.uniform(0, 5, (self.rows, self.cols)).astype(np.float32)

        # --- Roads: two crossing roads through center ---
        center_row, center_col = self.rows // 2, self.cols // 2
        road_width = max(1, int(6.0 / self.cell_size))  # ~6m wide road

        # Horizontal road
        for c in range(self.cols):
            for dr in range(-road_width // 2, road_width // 2 + 1):
                r = center_row + dr
                if 0 <= r < self.rows:
                    self.terrain[r, c] = TerrainType.ROAD.value
                    self.elevation[r, c] = 0.0

        # Vertical road
        for r in range(self.rows):
            for dc in range(-road_width // 2, road_width // 2 + 1):
                c = center_col + dc
                if 0 <= c < self.cols:
                    self.terrain[r, c] = TerrainType.ROAD.value
                    self.elevation[r, c] = 0.0

        # --- Buildings: clusters near roads ---
        num_buildings = 20
        for _ in range(num_buildings):
            # Place near roads with some offset
            bx = rng.uniform(-200, 200)
            by = rng.uniform(-200, 200)
            bw = rng.randint(2, 6)  # cells wide
            bh = rng.randint(2, 6)  # cells tall
            b_elev = rng.uniform(5, 20)  # building height

            br, bc = self.world_to_grid(bx, by)
            for dr in range(bh):
                for dc in range(bw):
                    r, c = br + dr, bc + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        if self.terrain[r, c] != TerrainType.ROAD.value:
                            self.terrain[r, c] = TerrainType.BUILDING.value
                            self.elevation[r, c] = b_elev

        # --- Vegetation: scattered patches ---
        num_veg_patches = 15
        for _ in range(num_veg_patches):
            vx = rng.uniform(-300, 300)
            vy = rng.uniform(-300, 300)
            vr = rng.randint(3, 10)  # radius in cells

            center_r, center_c = self.world_to_grid(vx, vy)
            for dr in range(-vr, vr + 1):
                for dc in range(-vr, vr + 1):
                    if dr * dr + dc * dc <= vr * vr:
                        r, c = center_r + dr, center_c + dc
                        if 0 <= r < self.rows and 0 <= c < self.cols:
                            if self.terrain[r, c] == TerrainType.OPEN_GROUND.value:
                                self.terrain[r, c] = TerrainType.VEGETATION.value
                                self.elevation[r, c] += rng.uniform(2, 8)

        # --- Water: one pond ---
        wx, wy = rng.uniform(-150, 150), rng.uniform(-150, 150)
        water_r = rng.randint(4, 8)
        center_r, center_c = self.world_to_grid(wx, wy)
        for dr in range(-water_r, water_r + 1):
            for dc in range(-water_r, water_r + 1):
                if dr * dr + dc * dc <= water_r * water_r:
                    r, c = center_r + dr, center_c + dc
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        self.terrain[r, c] = TerrainType.WATER.value
                        self.elevation[r, c] = -1.0

        logger.info("Terrain generated: %d buildings, %d veg patches, roads, 1 pond",
                     num_buildings, num_veg_patches)

    # ==================== OBJECT MANAGEMENT ====================

    def spawn_object(
        self,
        object_type: str,
        position: Vector3,
        is_threat: bool = False,
        spawn_time: float = 0.0,
    ) -> str:
        """
        Spawn a dynamic object into the world.
        
        Returns:
            Object ID string.
        """
        self._object_counter += 1
        obj_id = f"obj_{self._object_counter:04d}"

        thermal = THERMAL_SIGNATURES.get(object_type, 0.3)
        size = OBJECT_SIZES.get(object_type, 1.0)

        obj = WorldObject(
            object_id=obj_id,
            object_type=object_type,
            position=position,
            thermal_signature=thermal,
            is_threat=is_threat,
            size=size,
            spawn_time=spawn_time,
        )

        self._objects[obj_id] = obj
        logger.info("Spawned %s '%s' at (%.0f, %.0f) threat=%s",
                     object_type, obj_id, position.x, position.y, is_threat)
        return obj_id

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the world."""
        if object_id in self._objects:
            del self._objects[object_id]
            return True
        return False

    def get_object(self, object_id: str) -> Optional[WorldObject]:
        """Get a world object by ID."""
        return self._objects.get(object_id)

    def get_all_objects(self) -> List[WorldObject]:
        """Get all dynamic objects."""
        return list(self._objects.values())

    def get_threats(self) -> List[WorldObject]:
        """Get all objects marked as threats."""
        return [o for o in self._objects.values() if o.is_threat]

    # ==================== SENSOR QUERIES ====================

    def query_fov(
        self,
        drone_position: Vector3,
        altitude: float,
        fov_deg: float = 84.0,
    ) -> Tuple[List[WorldObject], List[Tuple[int, int]]]:
        """
        Query what is visible from a drone's position and altitude.
        
        Args:
            drone_position: Drone XY position (z ignored, altitude used instead)
            altitude: Drone altitude in meters AGL
            fov_deg: Camera field of view in degrees
            
        Returns:
            (visible_objects, coverage_cells)
        """
        # Calculate ground footprint radius from altitude and FOV
        footprint_radius = altitude * math.tan(math.radians(fov_deg / 2.0))

        # Get coverage cells
        coverage_cells = self._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint_radius
        )

        # Find objects within footprint
        visible_objects = []
        for obj in self._objects.values():
            if not obj.visible:
                continue
            dx = obj.position.x - drone_position.x
            dy = obj.position.y - drone_position.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= footprint_radius:
                visible_objects.append(obj)

        return visible_objects, coverage_cells

    def query_thermal(
        self,
        drone_position: Vector3,
        altitude: float,
        fov_deg: float = 40.0,
        threshold: float = 0.3,
    ) -> List[WorldObject]:
        """
        Query thermal signatures visible from drone position.
        
        Args:
            drone_position: Drone XY position
            altitude: Altitude in meters AGL
            fov_deg: Thermal camera FOV (typically narrower than RGB)
            threshold: Minimum thermal signature to detect
            
        Returns:
            Objects with thermal signature above threshold within FOV.
        """
        footprint_radius = altitude * math.tan(math.radians(fov_deg / 2.0))

        thermal_objects = []
        for obj in self._objects.values():
            if not obj.visible:
                continue
            if obj.thermal_signature < threshold:
                continue
            dx = obj.position.x - drone_position.x
            dy = obj.position.y - drone_position.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= footprint_radius:
                thermal_objects.append(obj)

        return thermal_objects

    def get_elevation(self, x: float, y: float) -> float:
        """Get terrain elevation at world coordinates."""
        row, col = self.world_to_grid(x, y)
        return float(self.elevation[row, col])

    def get_terrain_type(self, x: float, y: float) -> TerrainType:
        """Get terrain type at world coordinates."""
        row, col = self.world_to_grid(x, y)
        return TerrainType(int(self.terrain[row, col]))

    def get_objects_in_radius(self, position: Vector3, radius: float) -> List[WorldObject]:
        """Get all objects within a radius of a position."""
        result = []
        for obj in self._objects.values():
            dx = obj.position.x - position.x
            dy = obj.position.y - position.y
            if math.sqrt(dx * dx + dy * dy) <= radius:
                result.append(obj)
        return result

    # ==================== INTERNAL HELPERS ====================

    def _get_cells_in_radius(
        self, cx: float, cy: float, radius: float
    ) -> List[Tuple[int, int]]:
        """Get grid cells within a radius of a world coordinate."""
        center_row, center_col = self.world_to_grid(cx, cy)
        cell_radius = int(math.ceil(radius / self.cell_size))

        cells = []
        for dr in range(-cell_radius, cell_radius + 1):
            for dc in range(-cell_radius, cell_radius + 1):
                r, c = center_row + dr, center_col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    # Check actual distance
                    wx, wy = self.grid_to_world(r, c)
                    dist = math.sqrt((wx - cx) ** 2 + (wy - cy) ** 2)
                    if dist <= radius:
                        cells.append((r, c))
        return cells

    def get_terrain_summary(self) -> Dict[str, int]:
        """Get count of each terrain type."""
        summary = {}
        for tt in TerrainType:
            summary[tt.name] = int(np.sum(self.terrain == tt.value))
        return summary

    def to_dict(self) -> dict:
        """Serialize world state for frontend."""
        return {
            'width': self.width,
            'height': self.height,
            'cell_size': self.cell_size,
            'rows': self.rows,
            'cols': self.cols,
            'objects': [o.to_dict() for o in self._objects.values()],
            'terrain_summary': self.get_terrain_summary(),
        }
