import pytest

from src.core.types.drone_types import Vector3
from src.simulation.geofence_truth import (
    BuildingAABB,
    TrackSample,
    audit_footprint_crossings,
    audit_track_clearance,
    building_aabb_to_geometry,
    choose_open_validation_center,
    nearest_clear_ned_point,
    plan_building_aware_route,
    point_clearance_xy,
    required_patrol_altitude_m,
    start_positions_clear,
)


def blender_to_ned(bx, by, bz):
    return by, bx, -bz


def test_building_aabb_to_geometry_uses_world_bounds():
    building = BuildingAABB(
        name="Building_Test",
        min_xyz=(10.0, 20.0, 0.0),
        max_xyz=(30.0, 50.0, 40.0),
    )

    geometry = building_aabb_to_geometry(building, blender_to_ned)

    assert geometry.center.x == pytest.approx(35.0)
    assert geometry.center.y == pytest.approx(20.0)
    assert geometry.width == pytest.approx(30.0)
    assert geometry.depth == pytest.approx(20.0)
    assert geometry.height == pytest.approx(40.0)


def test_open_center_rejects_launch_positions_inside_building_margin():
    bounds = [
        BuildingAABB("Building_Block", (-10.0, -10.0, 0.0), (10.0, 10.0, 20.0)),
    ]

    assert point_clearance_xy((0.0, 0.0), bounds, margin_m=2.0) < 0.0
    assert start_positions_clear((0.0, 0.0), 3, bounds, margin_m=2.0) is False

    selection = choose_open_validation_center(
        dense_center_blender=(0.0, 0.0, 0.0),
        bounds=bounds,
        num_drones=3,
        safety_margin_m=2.0,
        min_radius_m=15,
        max_radius_m=30,
        radius_step_m=5,
        angle_step_deg=45,
    )

    assert point_clearance_xy(selection.blender_xy, bounds, margin_m=2.0) > 0.0
    assert start_positions_clear(selection.blender_xy, 3, bounds, margin_m=2.0)


def test_segment_audit_catches_path_crossing_building_between_frames():
    bounds = [BuildingAABB("Building_Block", (0.0, 0.0, 0.0), (10.0, 10.0, 20.0))]
    samples = {
        "0": [
            TrackSample(0.0, "0", (-5.0, 5.0, 5.0)),
            TrackSample(1.0, "0", (15.0, 5.0, 5.0)),
        ]
    }

    audit = audit_track_clearance(samples, bounds, roof_margin_m=1.0, max_segment_step_m=1.0)

    assert audit["segments_checked"] == 1
    assert audit["collision_or_near_collision_samples"] > 0
    assert audit["collision_or_near_collision_frames"] > 0
    assert audit["min_roof_clearance_m"] < 0.0


def test_segment_audit_allows_path_above_roof_clearance():
    bounds = [BuildingAABB("Building_Block", (0.0, 0.0, 0.0), (10.0, 10.0, 20.0))]
    samples = {
        "0": [
            TrackSample(0.0, "0", (-5.0, 5.0, 35.0)),
            TrackSample(1.0, "0", (15.0, 5.0, 35.0)),
        ]
    }

    audit = audit_track_clearance(samples, bounds, roof_margin_m=10.0, max_segment_step_m=1.0)

    assert audit["footprint_samples"] > 0
    assert audit["collision_or_near_collision_samples"] == 0
    assert audit["min_roof_clearance_m"] == pytest.approx(15.0)


def test_footprint_audit_catches_over_building_route_above_roof():
    bounds = [BuildingAABB("Building_Block", (0.0, 0.0, 0.0), (10.0, 10.0, 20.0))]
    samples = {
        "0": [
            TrackSample(0.0, "0", (-5.0, 5.0, 35.0)),
            TrackSample(1.0, "0", (15.0, 5.0, 35.0)),
        ]
    }

    audit = audit_footprint_crossings(
        samples,
        bounds,
        footprint_margin_m=0.0,
        max_segment_step_m=1.0,
    )

    assert audit["footprint_crossing_samples"] > 0
    assert audit["footprint_crossing_frames"] > 0
    assert audit["min_lateral_clearance_m"] < 0.0
    assert audit["first_crossings"][0]["building"] == "Building_Block"


def test_footprint_audit_passes_route_around_building():
    bounds = [BuildingAABB("Building_Block", (0.0, 0.0, 0.0), (10.0, 10.0, 20.0))]
    samples = {
        "0": [
            TrackSample(0.0, "0", (-5.0, -5.0, 35.0)),
            TrackSample(1.0, "0", (-5.0, 15.0, 35.0)),
            TrackSample(2.0, "0", (15.0, 15.0, 35.0)),
        ]
    }

    audit = audit_footprint_crossings(
        samples,
        bounds,
        footprint_margin_m=0.0,
        max_segment_step_m=1.0,
    )

    assert audit["footprint_crossing_samples"] == 0
    assert audit["footprint_crossing_frames"] == 0
    assert audit["min_lateral_clearance_m"] > 0.0


def test_required_patrol_altitude_uses_tallest_local_roof():
    bounds = [
        BuildingAABB("Local_Low", (-10.0, -10.0, 0.0), (10.0, 10.0, 20.0)),
        BuildingAABB("Local_Tall", (40.0, 0.0, 0.0), (50.0, 10.0, 55.0)),
        BuildingAABB("Far_Tall", (500.0, 500.0, 0.0), (510.0, 510.0, 90.0)),
    ]

    altitude = required_patrol_altitude_m(
        center_blender_xy=(0.0, 0.0),
        bounds=bounds,
        patrol_radius_m=80.0,
        roof_margin_m=12.0,
        min_altitude_m=30.0,
    )

    assert altitude == 67.0


def test_building_aware_route_returns_direct_goal_when_clear():
    bounds = [BuildingAABB("Building_Block", (20.0, 20.0, 0.0), (30.0, 30.0, 20.0))]
    start = blender_to_ned(-20.0, 0.0, 40.0)
    goal = blender_to_ned(-10.0, 0.0, 40.0)

    route = plan_building_aware_route(
        start=start,
        goal=goal,
        bounds=bounds,
        grid_resolution_m=5.0,
        safety_margin_m=2.0,
    )

    assert route == [route[-1]]
    assert route[0].x == pytest.approx(goal[0])
    assert route[0].y == pytest.approx(goal[1])
    assert route[0].z == pytest.approx(goal[2])


def test_building_aware_route_avoids_rectangular_obstacle():
    bounds = [BuildingAABB("Building_Block", (-5.0, -5.0, 0.0), (5.0, 5.0, 20.0))]
    start = blender_to_ned(-20.0, 0.0, 40.0)
    goal = blender_to_ned(20.0, 0.0, 40.0)

    route = plan_building_aware_route(
        start=start,
        goal=goal,
        bounds=bounds,
        grid_resolution_m=5.0,
        safety_margin_m=2.0,
    )

    assert len(route) > 1
    samples = {
        "0": [
            TrackSample(float(idx), "0", (point.y, point.x, -point.z))
            for idx, point in enumerate([Vector3(*start), *route])
        ]
    }
    audit = audit_footprint_crossings(
        samples,
        bounds,
        footprint_margin_m=2.0,
        max_segment_step_m=1.0,
    )
    assert audit["footprint_crossing_samples"] == 0


def test_building_aware_route_smoothing_preserves_obstacle_avoidance():
    bounds = [BuildingAABB("Building_Block", (-5.0, -5.0, 0.0), (5.0, 5.0, 20.0))]
    start = blender_to_ned(-20.0, 0.0, 40.0)
    goal = blender_to_ned(20.0, 0.0, 40.0)

    route = plan_building_aware_route(
        start=start,
        goal=goal,
        bounds=bounds,
        grid_resolution_m=2.5,
        safety_margin_m=2.0,
    )

    samples = {
        "0": [
            TrackSample(float(idx), "0", (point.y, point.x, -point.z))
            for idx, point in enumerate([Vector3(*start), *route])
        ]
    }
    audit = audit_footprint_crossings(
        samples,
        bounds,
        footprint_margin_m=2.0,
        max_segment_step_m=1.0,
    )
    assert audit["footprint_crossing_samples"] == 0


def test_nearest_clear_ned_point_moves_waypoint_outside_building_margin():
    bounds = [BuildingAABB("Building_Block", (-5.0, -5.0, 0.0), (5.0, 5.0, 20.0))]
    blocked = Vector3(x=0.0, y=0.0, z=-40.0)

    clear = nearest_clear_ned_point(
        blocked,
        bounds,
        safety_margin_m=2.0,
        search_step_m=2.5,
        max_radius_m=20.0,
    )

    assert clear.z == blocked.z
    samples = {"0": [TrackSample(0.0, "0", (clear.y, clear.x, -clear.z))]}
    audit = audit_footprint_crossings(samples, bounds, footprint_margin_m=2.0)
    assert audit["footprint_crossing_samples"] == 0
