"""
Project Sanjay Mk2 - Avoidance Stack Tests
============================================
Tests for AvoidanceManager, APF3D, HPL, and TacticalPlanner.
"""

import numpy as np
import pytest

from src.core.types.drone_types import Vector3
from src.single_drone.obstacle_avoidance.avoidance_manager import (
    AvoidanceManager,
    AvoidanceManagerConfig,
)
from src.single_drone.obstacle_avoidance.apf_3d import (
    APF3DAvoidance,
    APF3DConfig,
    AvoidanceState,
    Obstacle3D,
)
from src.single_drone.obstacle_avoidance.hardware_protection import (
    HardwareProtectionLayer,
    HPLConfig,
    HPLState,
)
from src.single_drone.obstacle_avoidance.tactical_planner import (
    TacticalPlanner,
    PlannerConfig,
)
from src.single_drone.sensors.lidar_3d import Lidar3DDriver, Lidar3DConfig


class TestAPF3DAvoidance:

    def test_clear_state_when_no_obstacles(self):
        apf = APF3DAvoidance()
        vel, state = apf.compute(
            my_position=Vector3(0, 0, -65),
            my_velocity=Vector3(1, 0, 0),
            goal_position=Vector3(100, 0, -65),
        )
        assert state in (AvoidanceState.CLEAR, AvoidanceState.MONITORING)
        assert vel.magnitude() > 0

    def test_repulsion_near_obstacle(self):
        apf = APF3DAvoidance()
        obs = Obstacle3D(
            position=Vector3(5, 0, -65), radius=2.0, confidence=1.0,
        )
        apf.update_obstacles([obs])
        vel, state = apf.compute(
            my_position=Vector3(0, 0, -65),
            my_velocity=Vector3(1, 0, 0),
            goal_position=Vector3(20, 0, -65),
        )
        assert vel.y != 0.0 or vel.z != 0.0 or vel.x < 1.0

    def test_closest_obstacle_distance_reported(self):
        apf = APF3DAvoidance()
        obs = Obstacle3D(
            position=Vector3(10, 0, -65), radius=1.0, confidence=1.0,
        )
        apf.update_obstacles([obs])
        apf.compute(
            my_position=Vector3(0, 0, -65),
            my_velocity=Vector3(),
            goal_position=Vector3(20, 0, -65),
        )
        assert apf.closest_obstacle_distance < 15.0


class TestHardwareProtectionLayer:

    def test_passive_state_without_scan(self):
        hpl = HardwareProtectionLayer()
        assert hpl.state == HPLState.PASSIVE

    def test_gate_command_passthrough(self):
        hpl = HardwareProtectionLayer()
        desired = Vector3(3.0, 0.0, 0.0)
        safe, overridden = hpl.gate_command(desired, Vector3(0, 0, -65))
        assert not overridden


class TestTacticalPlanner:

    def test_plan_without_obstacles(self):
        planner = TacticalPlanner()
        planner.update_costmap_origin(Vector3(0, 0, 0))
        waypoints = planner.plan(Vector3(0, 0, 0), Vector3(50, 50, 0))
        assert isinstance(waypoints, list)

    def test_plan_with_obstacles(self):
        planner = TacticalPlanner()
        planner.update_costmap_origin(Vector3(0, 0, 0))
        planner.update_obstacles(
            [Vector3(25, 25, 0)], [5.0],
        )
        waypoints = planner.plan(Vector3(0, 0, 0), Vector3(50, 50, 0))
        assert isinstance(waypoints, list)


class TestAvoidanceManager:

    def test_init_and_properties(self):
        mgr = AvoidanceManager(drone_id=0)
        assert mgr.drone_id == 0
        assert mgr.state == AvoidanceState.CLEAR
        assert not mgr.is_avoiding
        assert not mgr.is_hpl_overriding

    def test_set_goal_clears_sub_waypoints(self):
        mgr = AvoidanceManager(drone_id=0)
        mgr.set_goal(Vector3(100, 0, -65))
        assert mgr._goal is not None

    def test_boids_velocity_integration(self):
        mgr = AvoidanceManager(drone_id=0)
        mgr.set_boids_velocity(Vector3(3.0, 0.0, 0.0))
        vel = mgr.compute_avoidance(
            drone_position=Vector3(0, 0, -65),
            drone_velocity=Vector3(),
        )
        assert vel.magnitude() > 0

    def test_feed_lidar_points(self):
        mgr = AvoidanceManager(drone_id=0)
        points = np.random.randn(100, 3).astype(np.float32) * 10
        mgr.feed_lidar_points(points, drone_position=Vector3(0, 0, -65))
        telem = mgr.get_telemetry()
        assert "lidar" in telem

    def test_synthetic_obstacle_changes_command_path(self):
        mgr = AvoidanceManager(drone_id=0)
        points = np.array(
            [
                [4.0 + dx, dy, 1.0 + dz]
                for dx in (-0.4, -0.2, 0.0, 0.2, 0.4)
                for dy in (-0.4, -0.2, 0.0, 0.2, 0.4)
                for dz in (-0.2, 0.0, 0.2)
            ],
            dtype=np.float32,
        )

        position = Vector3(0, 0, -8)
        mgr.set_goal(Vector3(12, 0, -8))
        mgr.feed_lidar_points(points, drone_position=position)
        velocity = mgr.compute_avoidance(
            drone_position=position,
            drone_velocity=Vector3(),
        )
        telem = mgr.get_telemetry()

        assert telem["lidar"]["obstacle_count"] == 1
        assert telem["avoidance_state"] in {"MONITORING", "AVOIDING", "STUCK", "EMERGENCY"}
        assert abs(velocity.y) > 0.25 or abs(velocity.z) > 0.25 or velocity.x < 0.5

    def test_telemetry_output(self):
        mgr = AvoidanceManager(drone_id=0)
        telem = mgr.get_telemetry()
        assert "drone_id" in telem
        assert "avoidance_state" in telem
        assert "hpl_state" in telem
        assert "velocity" in telem
        assert "timestamp" in telem


class TestLidar3DDriver:

    def test_empty_update(self):
        driver = Lidar3DDriver()
        driver.update_points(np.empty((0, 3), dtype=np.float32))
        assert len(driver.get_obstacles()) == 0

    def test_update_with_points(self):
        driver = Lidar3DDriver()
        points = np.array([
            [5.0, 0.0, 0.0],
            [5.1, 0.1, 0.0],
            [5.0, -0.1, 0.1],
        ], dtype=np.float32)
        driver.update_points(points, drone_position=Vector3(0, 0, 0))
        telem = driver.get_telemetry()
        assert telem["filtered_points"] >= 0

    def test_empty_update_resets_sector_ranges(self):
        driver = Lidar3DDriver()
        points = np.array([
            [2.0, 0.0, 1.0],
            [2.1, 0.1, 1.0],
            [2.0, -0.1, 1.1],
            [2.2, 0.0, 1.0],
            [1.9, 0.1, 0.9],
        ], dtype=np.float32)
        driver.update_points(points, drone_position=Vector3(0, 0, 0))
        assert min(driver.get_sector_ranges()) < driver.config.max_range

        driver.update_points(np.empty((0, 3), dtype=np.float32))

        assert min(driver.get_sector_ranges()) == driver.config.max_range
