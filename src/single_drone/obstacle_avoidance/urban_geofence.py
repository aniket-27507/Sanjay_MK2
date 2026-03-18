"""
Project Sanjay Mk2 - Urban Geofence Manager
=============================================
3D building-aware geofencing for urban operations.

Maintains a set of BuildingGeometry exclusion zones and provides:
    - Position safety checks (is a position inside a building envelope?)
    - Altitude restrictions based on nearby buildings
    - Nearest safe position projection for geofence violations
    - Obstacle interface for integration with AvoidanceManager

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from src.core.types.drone_types import Vector3, BuildingGeometry

logger = logging.getLogger(__name__)

# Safety margin above buildings (metres)
BUILDING_TOP_MARGIN = 10.0

# Minimum altitude in urban areas (metres)
URBAN_MIN_ALTITUDE = 30.0


class UrbanGeofenceManager:
    """
    3D building-aware geofence manager.

    Usage:
        gm = UrbanGeofenceManager()
        gm.add_building(BuildingGeometry(center=Vector3(100, 200, 0), height=50))
        assert gm.check_position(Vector3(100, 200, -40)) == False  # inside building
        safe = gm.nearest_safe_position(Vector3(100, 200, -40))
    """

    def __init__(self, safety_margin: float = BUILDING_TOP_MARGIN):
        self._buildings: Dict[str, BuildingGeometry] = {}
        self._safety_margin = safety_margin

    # ==================== BUILDING MANAGEMENT ====================

    def add_building(self, building: BuildingGeometry) -> None:
        """Register a building as an exclusion zone."""
        self._buildings[building.building_id] = building
        logger.info(
            "Geofence: added building %s at (%.0f, %.0f) h=%.0fm",
            building.building_id, building.center.x, building.center.y, building.height,
        )

    def remove_building(self, building_id: str) -> bool:
        """Remove a building from the geofence. Returns True if found."""
        return self._buildings.pop(building_id, None) is not None

    def get_buildings(self) -> List[BuildingGeometry]:
        """Get all registered buildings."""
        return list(self._buildings.values())

    def clear(self) -> None:
        """Remove all buildings."""
        self._buildings.clear()

    # ==================== POSITION CHECKS ====================

    def check_position(self, position: Vector3) -> bool:
        """
        Check if a position is safe (not inside any building envelope).

        Args:
            position: NED position (z is negative for altitude)

        Returns:
            True if position is safe, False if inside a building zone.
        """
        altitude = -position.z  # Convert NED z to positive altitude

        for bldg in self._buildings.values():
            if bldg.contains_xy(position.x, position.y, margin=self._safety_margin):
                # Check altitude: unsafe if below building top + margin
                if altitude < bldg.height + self._safety_margin:
                    return False

        return True

    def get_altitude_restriction(self, x: float, y: float) -> float:
        """
        Get the minimum safe altitude at a ground position.

        Returns the height of the tallest building at (x, y) plus
        safety margin, or URBAN_MIN_ALTITUDE, whichever is higher.

        Args:
            x: North position (m)
            y: East position (m)

        Returns:
            Minimum safe altitude (m, positive).
        """
        min_alt = URBAN_MIN_ALTITUDE

        for bldg in self._buildings.values():
            if bldg.contains_xy(x, y, margin=self._safety_margin):
                required = bldg.height + self._safety_margin
                min_alt = max(min_alt, required)

        return min_alt

    def nearest_safe_position(self, position: Vector3) -> Vector3:
        """
        Project an unsafe position to the nearest safe position.

        Strategy: if inside a building's XY footprint, move to the
        nearest edge of the footprint + safety margin. If altitude is
        too low, raise to minimum safe altitude.

        Args:
            position: Unsafe NED position

        Returns:
            Nearest safe NED position.
        """
        if self.check_position(position):
            return position

        safe_x, safe_y = position.x, position.y
        altitude = -position.z

        for bldg in self._buildings.values():
            if not bldg.contains_xy(position.x, position.y, margin=self._safety_margin):
                continue

            # Project to nearest edge
            hw, hd = bldg.half_extents
            hw += self._safety_margin
            hd += self._safety_margin

            dx = position.x - bldg.center.x
            dy = position.y - bldg.center.y

            # Distance to each edge
            dist_to_edges = [
                (abs(dx - hw), bldg.center.x + hw, position.y),    # East edge
                (abs(dx + hw), bldg.center.x - hw, position.y),    # West edge
                (abs(dy - hd), position.x, bldg.center.y + hd),    # South edge
                (abs(dy + hd), position.x, bldg.center.y - hd),    # North edge
            ]

            nearest_edge = min(dist_to_edges, key=lambda e: e[0])
            safe_x, safe_y = nearest_edge[1], nearest_edge[2]

            # Also ensure altitude
            min_alt = bldg.height + self._safety_margin
            altitude = max(altitude, min_alt)

        return Vector3(x=safe_x, y=safe_y, z=-altitude)

    def get_obstacles_for_avoidance(self) -> List[Dict]:
        """
        Export buildings as obstacles for the AvoidanceManager.

        Returns a list of obstacle dicts compatible with the
        obstacle avoidance pipeline.
        """
        obstacles = []
        for bldg in self._buildings.values():
            hw, hd = bldg.half_extents
            obstacles.append({
                'id': bldg.building_id,
                'position': bldg.center,
                'radius': max(hw, hd) + self._safety_margin,
                'height': bldg.height + self._safety_margin,
                'type': 'building',
            })
        return obstacles

    def check_path_clear(
        self,
        start: Vector3,
        end: Vector3,
        num_samples: int = 10,
    ) -> bool:
        """
        Check if a straight-line path between two positions is clear
        of building zones.
        """
        for i in range(num_samples + 1):
            t = i / num_samples
            pos = Vector3(
                x=start.x + t * (end.x - start.x),
                y=start.y + t * (end.y - start.y),
                z=start.z + t * (end.z - start.z),
            )
            if not self.check_position(pos):
                return False
        return True
