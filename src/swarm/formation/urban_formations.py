"""
Project Sanjay Mk2 - Urban Formation Adapter
==============================================
Building-aware formation adjustments for urban environments.

Provides adapters that compute formation slot positions around
buildings, in tight urban canyons, and at varied altitudes.

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional

from src.core.types.drone_types import Vector3, BuildingGeometry
from src.swarm.formation.formation_controller import (
    FormationController, FormationConfig, FormationType,
)

logger = logging.getLogger(__name__)


class UrbanFormationAdapter:
    """
    Adapts formation geometry for urban environments.

    Usage:
        adapter = UrbanFormationAdapter(formation_controller)
        slots = adapter.compute_building_orbit(building, num_drones=6)
        config = adapter.adjust_for_urban_canyon(corridor_width=60.0, heading=1.57)
    """

    def __init__(self, controller: Optional[FormationController] = None):
        self._controller = controller

    def compute_building_orbit(
        self,
        building: BuildingGeometry,
        num_drones: int = 6,
        altitude: float = 65.0,
    ) -> List[Vector3]:
        """
        Compute orbital slot positions around a building perimeter.

        Drones are evenly distributed on a circle at building_radius + standoff.

        Args:
            building: BuildingGeometry to orbit
            num_drones: Number of drones in the orbit
            altitude: Patrol altitude (metres, positive)

        Returns:
            List of slot positions in NED world coordinates.
        """
        radius = max(building.width, building.depth) / 2.0 + building.standoff_distance
        positions = []

        for i in range(num_drones):
            angle = i * (2.0 * math.pi / num_drones)
            x = building.center.x + radius * math.cos(angle)
            y = building.center.y + radius * math.sin(angle)
            positions.append(Vector3(x=x, y=y, z=-altitude))

        return positions

    def compute_tight_formation(
        self,
        center: Vector3,
        num_drones: int = 6,
        spacing: float = 40.0,
    ) -> List[Vector3]:
        """
        Compute a tighter hexagonal formation for urban environments.

        Half the normal spacing (40m vs 80m default).

        Args:
            center: Formation center (NED)
            num_drones: Number of drones
            spacing: Inter-drone spacing (m)

        Returns:
            List of slot positions in NED world coordinates.
        """
        positions = [center]
        for i in range(min(num_drones - 1, 6)):
            angle = math.pi / 2.0 - i * (2.0 * math.pi / 6.0)
            positions.append(Vector3(
                x=center.x + spacing * math.cos(angle),
                y=center.y + spacing * math.sin(angle),
                z=center.z,
            ))
        return positions[:num_drones]

    def adjust_for_urban_canyon(
        self,
        corridor_width: float,
        corridor_heading: float,
        num_drones: int = 6,
        altitude: float = 65.0,
    ) -> FormationConfig:
        """
        Return a FormationConfig adapted for a narrow urban canyon/corridor.

        If corridor is narrow (< 100m), uses LINEAR formation aligned
        with the corridor. Otherwise uses URBAN_TIGHT.

        Args:
            corridor_width: Width of the corridor (m)
            corridor_heading: Heading angle of the corridor (rad)
            num_drones: Number of drones
            altitude: Patrol altitude

        Returns:
            Adapted FormationConfig.
        """
        if corridor_width < 100.0:
            # Linear formation along the corridor
            spacing = min(corridor_width * 0.8, 40.0)
            return FormationConfig(
                formation_type=FormationType.LINEAR,
                spacing=spacing,
                altitude=altitude,
            )
        else:
            return FormationConfig(
                formation_type=FormationType.URBAN_TIGHT,
                spacing=min(corridor_width * 0.4, 40.0),
                altitude=altitude,
            )

    def compute_multi_building_coverage(
        self,
        buildings: List[BuildingGeometry],
        num_drones: int = 6,
        altitude: float = 65.0,
    ) -> List[Vector3]:
        """
        Distribute drones across multiple buildings for area coverage.

        Assigns drones proportionally to buildings by size, with at least
        one drone per building (up to num_drones).
        """
        if not buildings:
            return []

        n_buildings = min(len(buildings), num_drones)
        drones_per_building = [1] * n_buildings
        remaining = num_drones - n_buildings

        # Allocate extra drones to larger buildings
        areas = [b.width * b.depth for b in buildings[:n_buildings]]
        total_area = sum(areas) or 1.0
        for _ in range(remaining):
            # Give to building with highest area/drone ratio
            ratios = [areas[i] / drones_per_building[i] for i in range(n_buildings)]
            best = ratios.index(max(ratios))
            drones_per_building[best] += 1

        positions = []
        for i in range(n_buildings):
            orbit = self.compute_building_orbit(
                buildings[i], drones_per_building[i], altitude
            )
            positions.extend(orbit)

        return positions[:num_drones]
