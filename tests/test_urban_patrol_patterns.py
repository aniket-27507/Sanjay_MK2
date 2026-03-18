"""
Tests for urban patrol pattern generation and urban formations.
"""

import math
import pytest

from src.core.types.drone_types import Vector3, Waypoint, BuildingGeometry
from src.swarm.formation.formation_controller import FormationType, FormationController, FormationConfig
from src.swarm.formation.urban_formations import UrbanFormationAdapter
from src.swarm.coordination.urban_patrol_patterns import UrbanPatrolPatternGenerator


# ==================== FORMATION TYPES ====================

class TestUrbanFormationTypes:
    def test_urban_tight_exists(self):
        assert FormationType.URBAN_TIGHT is not None

    def test_building_orbit_exists(self):
        assert FormationType.BUILDING_ORBIT is not None

    def test_vertical_stack_exists(self):
        assert FormationType.VERTICAL_STACK is not None

    def test_urban_tight_generates_slots(self):
        config = FormationConfig(
            formation_type=FormationType.URBAN_TIGHT,
            spacing=80.0,
        )
        fc = FormationController(num_drones=6, config=config)
        slots = fc.get_slot_positions()
        assert len(slots) == 6

        # URBAN_TIGHT uses half spacing, so drones should be closer
        # than standard hex (80m -> 40m effective spacing)
        dists = [slots[0].distance_to(s) for s in slots[1:]]
        assert all(d < 50.0 for d in dists)  # All within 50m (half of 80m + margin)

    def test_vertical_stack_generates_slots(self):
        config = FormationConfig(
            formation_type=FormationType.VERTICAL_STACK,
            spacing=80.0,
            altitude=65.0,
        )
        fc = FormationController(num_drones=6, config=config)
        slots = fc.get_slot_positions()
        assert len(slots) == 6

        # Should have varied z offsets (different altitudes)
        z_values = [s.z for s in slots]
        unique_z = set(round(z, 1) for z in z_values)
        assert len(unique_z) >= 2  # At least 2 altitude layers


# ==================== URBAN FORMATION ADAPTER ====================

class TestUrbanFormationAdapter:
    def test_building_orbit_positions(self):
        adapter = UrbanFormationAdapter()
        building = BuildingGeometry(
            center=Vector3(100, 200, 0),
            width=30, depth=30, height=50,
            standoff_distance=25.0,
        )
        positions = adapter.compute_building_orbit(building, num_drones=6)
        assert len(positions) == 6

        # All positions should be roughly equidistant from center
        expected_radius = 15.0 + 25.0  # half-width + standoff
        for pos in positions:
            dist_xy = math.sqrt(
                (pos.x - building.center.x) ** 2 +
                (pos.y - building.center.y) ** 2
            )
            assert abs(dist_xy - expected_radius) < 1.0

    def test_tight_formation_closer(self):
        adapter = UrbanFormationAdapter()
        center = Vector3(0, 0, -65)
        positions = adapter.compute_tight_formation(center, num_drones=6, spacing=40.0)
        assert len(positions) == 6

        # Spacing should be ~40m (not 80m default)
        for pos in positions[1:]:
            dist = center.distance_to(pos)
            assert dist < 50.0

    def test_urban_canyon_narrow_returns_linear(self):
        adapter = UrbanFormationAdapter()
        config = adapter.adjust_for_urban_canyon(
            corridor_width=60.0, corridor_heading=0.0,
        )
        assert config.formation_type == FormationType.LINEAR
        assert config.spacing <= 60.0

    def test_urban_canyon_wide_returns_urban_tight(self):
        adapter = UrbanFormationAdapter()
        config = adapter.adjust_for_urban_canyon(
            corridor_width=150.0, corridor_heading=0.0,
        )
        assert config.formation_type == FormationType.URBAN_TIGHT

    def test_multi_building_coverage(self):
        adapter = UrbanFormationAdapter()
        buildings = [
            BuildingGeometry(center=Vector3(0, 0, 0), width=20, depth=20, height=50),
            BuildingGeometry(center=Vector3(200, 200, 0), width=40, depth=40, height=80),
        ]
        positions = adapter.compute_multi_building_coverage(buildings, num_drones=6)
        assert len(positions) == 6


# ==================== PATROL PATTERNS ====================

class TestBuildingPerimeter:
    def test_generates_8_waypoints(self):
        gen = UrbanPatrolPatternGenerator()
        building = BuildingGeometry(
            center=Vector3(100, 100, 0), width=30, depth=30, height=50,
        )
        wps = gen.building_perimeter(building)
        assert len(wps) == 8

    def test_yaw_faces_building(self):
        gen = UrbanPatrolPatternGenerator()
        building = BuildingGeometry(
            center=Vector3(0, 0, 0), width=20, depth=20, height=50,
            standoff_distance=30.0,
        )
        wps = gen.building_perimeter(building)

        for wp in wps:
            # Yaw should point toward (0,0)
            expected_yaw = math.atan2(0 - wp.position.y, 0 - wp.position.x)
            assert abs(wp.yaw - expected_yaw) < 0.01

    def test_altitude_set_correctly(self):
        gen = UrbanPatrolPatternGenerator()
        building = BuildingGeometry(center=Vector3(0, 0, 0))
        wps = gen.building_perimeter(building, altitude=65.0)
        for wp in wps:
            assert wp.position.z == -65.0


class TestVerticalScan:
    def test_generates_descending_waypoints(self):
        gen = UrbanPatrolPatternGenerator()
        wps = gen.vertical_scan(
            face_center=Vector3(0, 0, 0),
            face_width=30.0,
            building_height=50.0,
            altitude_step=10.0,
        )
        assert len(wps) > 0
        # First waypoint should be at high altitude, last at low
        assert wps[0].position.z <= wps[-1].position.z  # NED: more negative = higher

    def test_zigzag_pattern(self):
        gen = UrbanPatrolPatternGenerator()
        wps = gen.vertical_scan(
            face_center=Vector3(0, 0, 0),
            face_width=30.0,
            building_height=30.0,
            altitude_step=10.0,
        )
        # Should have pairs of waypoints (left-right, right-left)
        assert len(wps) >= 4
        assert len(wps) % 2 == 0  # Even number (pairs)


class TestCrowdOverhead:
    def test_generates_per_drone_waypoints(self):
        gen = UrbanPatrolPatternGenerator()
        result = gen.crowd_overhead(
            area_center=Vector3(0, 0, 0),
            area_radius=200.0,
            num_drones=4,
        )
        assert len(result) == 4
        for drone_id, wps in result.items():
            assert len(wps) == 4  # 4-point loiter per drone


class TestExitCorridor:
    def test_generates_per_corridor_waypoints(self):
        gen = UrbanPatrolPatternGenerator()
        corridors = [
            (Vector3(0, 0, 0), Vector3(100, 0, 0)),
            (Vector3(0, 0, 0), Vector3(0, 100, 0)),
        ]
        result = gen.exit_corridor(corridors)
        assert len(result) == 2
        for drone_id, wps in result.items():
            assert len(wps) == 3  # start, mid, end
