"""
Tests for UrbanGeofenceManager — building exclusion zones and altitude restrictions.
"""

import pytest

from src.core.types.drone_types import Vector3, BuildingGeometry
from src.single_drone.obstacle_avoidance.urban_geofence import (
    UrbanGeofenceManager, BUILDING_TOP_MARGIN, URBAN_MIN_ALTITUDE,
)


@pytest.fixture
def geofence():
    gm = UrbanGeofenceManager()
    gm.add_building(BuildingGeometry(
        building_id="bldg_1",
        center=Vector3(100, 200, 0),
        width=30,
        depth=30,
        height=50,
        standoff_distance=30.0,
    ))
    return gm


# ==================== POSITION CHECKS ====================

class TestPositionChecks:
    def test_safe_position_above_building(self, geofence):
        # Above building: altitude 70m > 50m + 10m margin
        assert geofence.check_position(Vector3(100, 200, -70)) is True

    def test_unsafe_inside_building_footprint(self, geofence):
        # Inside building footprint at low altitude
        assert geofence.check_position(Vector3(100, 200, -40)) is False

    def test_safe_outside_building_footprint(self, geofence):
        # Outside building footprint at any altitude
        assert geofence.check_position(Vector3(300, 300, -20)) is True

    def test_edge_of_margin(self, geofence):
        # At exact building top + margin altitude (50 + 10 = 60m)
        # Should be safe at or above
        assert geofence.check_position(Vector3(100, 200, -60)) is True

    def test_just_below_margin(self, geofence):
        # Just below the safe altitude
        assert geofence.check_position(Vector3(100, 200, -59)) is False

    def test_empty_geofence_always_safe(self):
        gm = UrbanGeofenceManager()
        assert gm.check_position(Vector3(0, 0, -10)) is True


# ==================== ALTITUDE RESTRICTIONS ====================

class TestAltitudeRestrictions:
    def test_altitude_above_building(self, geofence):
        alt = geofence.get_altitude_restriction(100.0, 200.0)
        assert alt >= 50.0 + BUILDING_TOP_MARGIN

    def test_altitude_away_from_building(self, geofence):
        alt = geofence.get_altitude_restriction(500.0, 500.0)
        assert alt == URBAN_MIN_ALTITUDE

    def test_multiple_buildings_uses_tallest(self):
        gm = UrbanGeofenceManager()
        gm.add_building(BuildingGeometry(
            center=Vector3(0, 0, 0), width=50, depth=50, height=30,
        ))
        gm.add_building(BuildingGeometry(
            center=Vector3(0, 0, 0), width=40, depth=40, height=80,
        ))
        alt = gm.get_altitude_restriction(0.0, 0.0)
        assert alt >= 80.0 + BUILDING_TOP_MARGIN


# ==================== NEAREST SAFE POSITION ====================

class TestNearestSafePosition:
    def test_safe_position_unchanged(self, geofence):
        safe = Vector3(300, 300, -65)
        result = geofence.nearest_safe_position(safe)
        assert result.x == safe.x
        assert result.y == safe.y

    def test_unsafe_position_projected_outside(self, geofence):
        unsafe = Vector3(100, 200, -40)
        safe = geofence.nearest_safe_position(unsafe)
        assert geofence.check_position(safe)

    def test_altitude_raised_if_too_low(self, geofence):
        unsafe = Vector3(100, 200, -30)
        safe = geofence.nearest_safe_position(unsafe)
        assert -safe.z >= 50.0 + BUILDING_TOP_MARGIN


# ==================== BUILDING MANAGEMENT ====================

class TestBuildingManagement:
    def test_add_and_remove(self):
        gm = UrbanGeofenceManager()
        bldg = BuildingGeometry(building_id="test_1", center=Vector3(0, 0, 0))
        gm.add_building(bldg)
        assert len(gm.get_buildings()) == 1

        gm.remove_building("test_1")
        assert len(gm.get_buildings()) == 0

    def test_clear(self):
        gm = UrbanGeofenceManager()
        gm.add_building(BuildingGeometry(center=Vector3(0, 0, 0)))
        gm.add_building(BuildingGeometry(center=Vector3(100, 100, 0)))
        gm.clear()
        assert len(gm.get_buildings()) == 0


# ==================== PATH CHECKING ====================

class TestPathChecking:
    def test_clear_path(self, geofence):
        start = Vector3(300, 300, -65)
        end = Vector3(400, 400, -65)
        assert geofence.check_path_clear(start, end) is True

    def test_blocked_path(self, geofence):
        # Path through the building
        start = Vector3(50, 200, -40)
        end = Vector3(150, 200, -40)
        assert geofence.check_path_clear(start, end) is False


# ==================== OBSTACLE EXPORT ====================

class TestObstacleExport:
    def test_exports_buildings_as_obstacles(self, geofence):
        obstacles = geofence.get_obstacles_for_avoidance()
        assert len(obstacles) == 1
        obs = obstacles[0]
        assert obs['type'] == 'building'
        assert obs['height'] > 0
        assert obs['radius'] > 0
