"""
Project Sanjay Mk2 - Urban Patrol Pattern Generator
=====================================================
Generates waypoint sequences for high-rise building surveillance,
crowd event monitoring, and exit corridor coverage.

Patterns:
    1. Building Perimeter   — 8-waypoint rectangular orbit
    2. Vertical Scan        — descending zigzag on building face (Beta task)
    3. Crowd Overhead       — loiter sectors over event area
    4. Exit Corridor Monitor — linear formation along corridors

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

from src.core.types.drone_types import Vector3, Waypoint, BuildingGeometry

logger = logging.getLogger(__name__)


class UrbanPatrolPatternGenerator:
    """
    Generates waypoint sequences for urban surveillance patterns.

    Usage:
        gen = UrbanPatrolPatternGenerator()
        waypoints = gen.building_perimeter(building, standoff=30.0)
        beta_wps = gen.vertical_scan(face_center, face_width, building_height)
    """

    # ==================== BUILDING PERIMETER ====================

    def building_perimeter(
        self,
        building: BuildingGeometry,
        altitude: float = 65.0,
        speed: float = 3.0,
        standoff: Optional[float] = None,
    ) -> List[Waypoint]:
        """
        Generate an 8-waypoint rectangular perimeter orbit around a building.

        Waypoints at each corner + midpoint of each face, yaw always
        pointing toward the building center.

        Args:
            building: Target building
            altitude: Patrol altitude (m, positive)
            speed: Patrol speed (m/s)
            standoff: Distance from building facade (m), uses building default if None

        Returns:
            List of 8 Waypoints forming a rectangular orbit.
        """
        standoff = standoff or building.standoff_distance
        hw = building.width / 2.0 + standoff
        hd = building.depth / 2.0 + standoff
        cx, cy = building.center.x, building.center.y
        z = -altitude  # NED

        # 8 points: corners and face midpoints, clockwise from NW
        points = [
            (cx - hw, cy - hd),  # NW corner
            (cx,      cy - hd),  # N mid
            (cx + hw, cy - hd),  # NE corner
            (cx + hw, cy),       # E mid
            (cx + hw, cy + hd),  # SE corner
            (cx,      cy + hd),  # S mid
            (cx - hw, cy + hd),  # SW corner
            (cx - hw, cy),       # W mid
        ]

        waypoints = []
        for px, py in points:
            # Yaw pointing toward building center
            yaw = math.atan2(cy - py, cx - px)
            waypoints.append(Waypoint(
                position=Vector3(x=px, y=py, z=z),
                speed=speed,
                acceptance_radius=2.0,
                hold_time=0.0,
                yaw=yaw,
            ))

        return waypoints

    # ==================== VERTICAL SCAN ====================

    def vertical_scan(
        self,
        face_center: Vector3,
        face_width: float,
        building_height: float,
        speed: float = 2.0,
        altitude_step: float = 10.0,
        standoff: float = 20.0,
    ) -> List[Waypoint]:
        """
        Generate a descending zigzag scan pattern on one building face.

        Intended for Beta drone close-inspection. Starts at rooftop level,
        descends in altitude_step increments, scanning left-to-right and
        right-to-left alternately across the face width.

        Args:
            face_center: Center of the building face at ground level (NED, z=0)
            face_width: Width of the face to scan (m)
            building_height: Building height (m)
            speed: Scan speed (m/s)
            altitude_step: Vertical step between scan lines (m)
            standoff: Distance from the face (m)

        Returns:
            List of Waypoints forming the zigzag descent.
        """
        waypoints = []
        half_w = face_width / 2.0

        # Determine face normal direction (assume face_center offset from building)
        # For simplicity, scan along the y-axis at the face_center x position
        scan_x = face_center.x
        base_y = face_center.y

        num_levels = max(1, int(building_height / altitude_step))
        left_to_right = True

        for level in range(num_levels + 1):
            alt = building_height - level * altitude_step
            if alt < altitude_step:
                alt = altitude_step  # Don't go below minimum
            z = -alt  # NED

            if left_to_right:
                y_start = base_y - half_w
                y_end = base_y + half_w
            else:
                y_start = base_y + half_w
                y_end = base_y - half_w

            # Yaw facing the building face
            yaw = math.atan2(0, -1) if scan_x > face_center.x else 0.0

            waypoints.append(Waypoint(
                position=Vector3(x=scan_x + standoff, y=y_start, z=z),
                speed=speed,
                acceptance_radius=1.5,
                yaw=yaw,
            ))
            waypoints.append(Waypoint(
                position=Vector3(x=scan_x + standoff, y=y_end, z=z),
                speed=speed,
                acceptance_radius=1.5,
                yaw=yaw,
            ))

            left_to_right = not left_to_right

        return waypoints

    # ==================== CROWD OVERHEAD ====================

    def crowd_overhead(
        self,
        area_center: Vector3,
        area_radius: float,
        num_drones: int = 6,
        altitude: float = 65.0,
        speed: float = 3.0,
    ) -> Dict[int, List[Waypoint]]:
        """
        Generate loiter-sector waypoints for crowd event monitoring.

        Divides the circular area into equal pie-slice sectors, one per drone.
        Each drone gets a circular loiter pattern within its sector.

        Args:
            area_center: Center of the crowd event area
            area_radius: Radius of the area to cover (m)
            num_drones: Number of drones
            altitude: Monitoring altitude (m, positive)
            speed: Loiter speed (m/s)

        Returns:
            Dict of drone_index -> waypoint list.
        """
        z = -altitude
        result: Dict[int, List[Waypoint]] = {}

        for i in range(num_drones):
            sector_angle = i * (2.0 * math.pi / num_drones)
            sector_center_x = area_center.x + (area_radius * 0.5) * math.cos(sector_angle)
            sector_center_y = area_center.y + (area_radius * 0.5) * math.sin(sector_angle)

            # Generate 4-point loiter within the sector
            loiter_radius = area_radius * 0.3
            waypoints = []
            for j in range(4):
                angle = sector_angle + j * (math.pi / 2.0)
                px = sector_center_x + loiter_radius * math.cos(angle)
                py = sector_center_y + loiter_radius * math.sin(angle)
                # Yaw pointing toward area center
                yaw = math.atan2(area_center.y - py, area_center.x - px)
                waypoints.append(Waypoint(
                    position=Vector3(x=px, y=py, z=z),
                    speed=speed,
                    acceptance_radius=3.0,
                    hold_time=2.0,
                    yaw=yaw,
                ))
            result[i] = waypoints

        return result

    # ==================== EXIT CORRIDOR ====================

    def exit_corridor(
        self,
        corridors: List[Tuple[Vector3, Vector3]],
        altitude: float = 65.0,
        speed: float = 3.0,
    ) -> Dict[int, List[Waypoint]]:
        """
        Position drones along exit corridors to monitor crowd flow.

        One drone per corridor, stationed at the corridor midpoint
        with a back-and-forth patrol between start and end.

        Args:
            corridors: List of (start, end) position tuples defining corridors
            altitude: Monitoring altitude (m, positive)
            speed: Patrol speed (m/s)

        Returns:
            Dict of drone_index -> waypoint list.
        """
        z = -altitude
        result: Dict[int, List[Waypoint]] = {}

        for i, (start, end) in enumerate(corridors):
            # Midpoint for hover
            mid_x = (start.x + end.x) / 2.0
            mid_y = (start.y + end.y) / 2.0

            # Yaw aligned with corridor direction
            yaw = math.atan2(end.y - start.y, end.x - start.x)

            waypoints = [
                Waypoint(
                    position=Vector3(x=start.x, y=start.y, z=z),
                    speed=speed, acceptance_radius=2.0, yaw=yaw,
                ),
                Waypoint(
                    position=Vector3(x=mid_x, y=mid_y, z=z),
                    speed=speed, acceptance_radius=2.0, hold_time=3.0, yaw=yaw,
                ),
                Waypoint(
                    position=Vector3(x=end.x, y=end.y, z=z),
                    speed=speed, acceptance_radius=2.0, yaw=yaw,
                ),
            ]
            result[i] = waypoints

        return result
