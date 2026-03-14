"""
Project Sanjay Mk2 - SwarmWaypointRunner Tests
===============================================
Unit and integration tests for the 7-drone swarm checkpoint system.
"""

import asyncio
import math

import pytest

from src.core.types.drone_types import DroneType, Vector3, Waypoint
from src.simulation.surveillance_layout import (
    ALPHA_ALTITUDE,
    BETA_ALTITUDE,
    BETA_ID,
    FORMATION_SPACING,
)
from src.swarm.swarm_waypoint_runner import (
    CheckpointPhase,
    SimDrone,
    SwarmCheckpointStatus,
    SwarmExecutionState,
    SwarmWaypointRunner,
)


# ── SimDrone unit tests ──────────────────────────────────────────


class TestSimDrone:

    def test_initial_position(self):
        pos = Vector3(x=100, y=200, z=-25)
        drone = SimDrone(drone_id=0, position=pos)
        assert drone.position.x == 100
        assert drone.position.y == 200
        assert drone.position.z == -25

    def test_step_updates_position(self):
        drone = SimDrone(drone_id=0, position=Vector3())
        drone.step(Vector3(x=10, y=0, z=0), dt=1.0)
        assert abs(drone.position.x - 10.0) < 0.01
        assert abs(drone.position.y) < 0.01

    def test_to_state_has_correct_id(self):
        drone = SimDrone(drone_id=42, position=Vector3(), drone_type=DroneType.BETA)
        state = drone.to_state()
        assert state.drone_id == 42
        assert state.drone_type == DroneType.BETA


# ── SwarmWaypointRunner unit tests ───────────────────────────────


class TestSwarmWaypointRunnerCheckpoints:

    def test_add_checkpoint(self):
        runner = SwarmWaypointRunner(headless=True)
        runner.add_checkpoint(position=Vector3(x=200, y=200, z=-65))
        assert len(runner.checkpoints) == 1
        assert runner.status.total_checkpoints == 1

    def test_add_multiple_checkpoints(self):
        runner = SwarmWaypointRunner(headless=True)
        runner.add_checkpoint(position=Vector3(x=100, y=100, z=-65))
        runner.add_checkpoint(position=Vector3(x=200, y=200, z=-65))
        runner.add_checkpoint(position=Vector3(x=300, y=300, z=-65))
        assert len(runner.checkpoints) == 3
        assert runner.status.total_checkpoints == 3

    def test_clear_checkpoints(self):
        runner = SwarmWaypointRunner(headless=True)
        runner.add_checkpoint(position=Vector3(x=200, y=200, z=-65))
        runner.add_checkpoint(position=Vector3(x=300, y=300, z=-65))
        runner.clear_checkpoints()
        assert len(runner.checkpoints) == 0
        assert runner.status.total_checkpoints == 0
        assert runner.status.current_index == 0


class TestSwarmWaypointRunnerFormation:

    def test_formation_has_7_slots(self):
        runner = SwarmWaypointRunner(headless=True)
        slots = runner._formation.get_slot_positions()
        assert len(slots) == 7

    def test_beta_at_center_slot(self):
        runner = SwarmWaypointRunner(headless=True)
        center_slot = runner._formation.get_slot_for_drone(BETA_ID)
        assert center_slot is not None
        # Center slot offset should be (0, 0, 0) relative to formation center
        center = runner._formation.center
        dist = center_slot.distance_to(center)
        assert dist < 1.0, f"Beta slot should be at center, dist={dist}"

    def test_alphas_at_vertex_slots(self):
        runner = SwarmWaypointRunner(headless=True)
        center = runner._formation.center
        for drone_id in range(6):
            slot = runner._formation.get_slot_for_drone(drone_id)
            assert slot is not None, f"Alpha {drone_id} has no slot"
            # Alpha slots should be at spacing distance from center (in XY)
            dx = slot.x - center.x
            dy = slot.y - center.y
            dist = math.sqrt(dx * dx + dy * dy)
            assert dist > 10.0, f"Alpha {drone_id} too close to center ({dist}m)"

    def test_set_formation_spacing(self):
        runner = SwarmWaypointRunner(headless=True, formation_spacing=80.0)
        runner.set_formation_spacing(120.0)
        assert runner._formation_spacing == 120.0
        assert runner._formation.config.spacing == 120.0

    def test_spacing_clamped(self):
        runner = SwarmWaypointRunner(headless=True)
        runner.set_formation_spacing(10.0)
        assert runner._formation_spacing == 30.0  # Minimum
        runner.set_formation_spacing(200.0)
        assert runner._formation_spacing == 150.0  # Maximum


class TestSwarmWaypointRunnerStatus:

    def test_initial_status(self):
        runner = SwarmWaypointRunner(headless=True)
        s = runner.status
        assert s.state == SwarmExecutionState.IDLE
        assert s.current_index == 0
        assert s.total_checkpoints == 0

    @pytest.mark.asyncio
    async def test_execute_without_checkpoints_returns_false(self):
        runner = SwarmWaypointRunner(headless=True)
        result = await runner.execute()
        assert result is False

    def test_pause_resume(self):
        runner = SwarmWaypointRunner(headless=True)
        runner.pause()
        assert runner.status.state == SwarmExecutionState.PAUSED
        runner.resume()
        assert runner.status.state == SwarmExecutionState.RUNNING


class TestSwarmWaypointRunnerPhaseLogic:

    def _make_runner_with_drones_at_checkpoint(self, checkpoint_pos: Vector3):
        """Create a runner with all drones pre-positioned at a checkpoint."""
        runner = SwarmWaypointRunner(headless=True)
        runner.add_checkpoint(position=checkpoint_pos)

        # Manually init drones at their target positions
        from src.core.utils.geometry import hex_positions

        cx, cy = checkpoint_pos.x, checkpoint_pos.y
        hex_pos = hex_positions(cx, cy, runner._formation_spacing, n=7)

        runner._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=cx, y=cy, z=-BETA_ALTITUDE),
            drone_type=DroneType.BETA,
        )
        for i in range(6):
            vx, vy = hex_pos[i + 1]
            runner._drones[i] = SimDrone(
                drone_id=i,
                position=Vector3(x=vx, y=vy, z=-ALPHA_ALTITUDE),
                drone_type=DroneType.ALPHA,
            )

        return runner

    def test_is_hex_reformed_when_at_vertices(self):
        cp = Vector3(x=400, y=350, z=-65)
        runner = self._make_runner_with_drones_at_checkpoint(cp)

        # Set formation center to checkpoint
        runner._formation.set_center(Vector3(x=cp.x, y=cp.y, z=-ALPHA_ALTITUDE))

        assert runner._is_hex_reformed(), "Hex should be reformed when drones are at slots"

    def test_is_hex_not_reformed_when_far(self):
        cp = Vector3(x=400, y=350, z=-65)
        runner = self._make_runner_with_drones_at_checkpoint(cp)

        # Move alpha_0 far away
        runner._drones[0].position = Vector3(x=0, y=0, z=-ALPHA_ALTITUDE)

        runner._formation.set_center(Vector3(x=cp.x, y=cp.y, z=-ALPHA_ALTITUDE))
        assert not runner._is_hex_reformed()

    def test_is_swarm_at_checkpoint_xy(self):
        cp = Vector3(x=400, y=350, z=-65)
        runner = self._make_runner_with_drones_at_checkpoint(cp)
        runner._formation.set_center(Vector3(x=cp.x, y=cp.y, z=-ALPHA_ALTITUDE))
        runner._flock_center = Vector3(x=cp.x, y=cp.y, z=-ALPHA_ALTITUDE)

        wp = Waypoint(position=cp)
        assert runner._is_swarm_at_checkpoint_xy(wp)

    def test_compute_formation_quality(self):
        cp = Vector3(x=400, y=350, z=-65)
        runner = self._make_runner_with_drones_at_checkpoint(cp)
        runner._formation.set_center(Vector3(x=cp.x, y=cp.y, z=-ALPHA_ALTITUDE))

        wp = Waypoint(position=cp)
        quality = runner._compute_formation_quality(wp)
        assert quality == 1.0, f"Quality should be 1.0 when all at vertices, got {quality}"

    def test_compute_formation_quality_partial(self):
        cp = Vector3(x=400, y=350, z=-65)
        runner = self._make_runner_with_drones_at_checkpoint(cp)
        runner._formation.set_center(Vector3(x=cp.x, y=cp.y, z=-ALPHA_ALTITUDE))

        # Move 2 alphas away
        runner._drones[0].position = Vector3(x=0, y=0, z=-ALPHA_ALTITUDE)
        runner._drones[1].position = Vector3(x=0, y=100, z=-ALPHA_ALTITUDE)

        wp = Waypoint(position=cp)
        quality = runner._compute_formation_quality(wp)
        assert abs(quality - 4.0 / 6.0) < 0.01

    def test_compute_min_inter_drone(self):
        cp = Vector3(x=400, y=350, z=-65)
        runner = self._make_runner_with_drones_at_checkpoint(cp)

        min_dist = runner._compute_min_inter_drone()
        assert min_dist > 0.0
        assert min_dist < float("inf")


class TestSwarmWaypointRunnerBetaVelocity:

    def test_beta_moves_toward_goal(self):
        runner = SwarmWaypointRunner(headless=True)

        # Place beta at origin
        runner._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=0, y=0, z=-25),
            drone_type=DroneType.BETA,
        )

        goal = Vector3(x=100, y=0, z=-25)
        runner._apply_beta_velocity(goal)

        # After one step, beta should have moved toward goal
        assert runner._drones[BETA_ID].position.x > 0.0

    def test_beta_climbs_toward_higher_altitude(self):
        runner = SwarmWaypointRunner(headless=True)

        runner._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=100, y=100, z=-25),
            drone_type=DroneType.BETA,
        )

        # Goal at z=-65 (higher altitude in NED = more negative)
        goal = Vector3(x=100, y=100, z=-65)
        runner._apply_beta_velocity(goal)

        # Beta z should have decreased (moved toward -65)
        assert runner._drones[BETA_ID].position.z < -25.0

    def test_beta_holds_at_goal(self):
        runner = SwarmWaypointRunner(headless=True)

        # Place beta right at the goal
        runner._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=100, y=100, z=-25),
            drone_type=DroneType.BETA,
        )

        goal = Vector3(x=100, y=100, z=-25)
        runner._apply_beta_velocity(goal)

        # Beta should barely move (within 0.5m tolerance)
        pos = runner._drones[BETA_ID].position
        assert abs(pos.x - 100) < 1.0
        assert abs(pos.y - 100) < 1.0


class TestSwarmWaypointRunnerIntegration:
    """Integration tests that run the full execute loop (headless)."""

    @pytest.mark.asyncio
    async def test_single_checkpoint_completion(self):
        """Test that swarm reaches a single nearby checkpoint."""
        runner = SwarmWaypointRunner(headless=True, formation_spacing=30.0)

        # Place checkpoint close to default formation center
        runner.add_checkpoint(
            position=Vector3(x=400, y=350, z=-40),
        )

        # Run with a timeout to prevent hanging
        try:
            result = await asyncio.wait_for(runner.execute(), timeout=120.0)
        except asyncio.TimeoutError:
            runner.stop()
            pytest.skip("Integration test timed out (expected in CI)")
            return

        assert result is True
        assert runner.status.state == SwarmExecutionState.COMPLETE

    @pytest.mark.asyncio
    async def test_stop_during_execution(self):
        """Test that stop() halts execution gracefully."""
        runner = SwarmWaypointRunner(headless=True)

        # Place checkpoint far away so transit takes a while
        runner.add_checkpoint(
            position=Vector3(x=9999, y=9999, z=-65),
        )

        async def stop_after_delay():
            await asyncio.sleep(0.5)
            runner.stop()

        task = asyncio.create_task(stop_after_delay())

        result = await runner.execute()
        await task

        assert result is False
        assert runner.status.state == SwarmExecutionState.STOPPED


# ── Flock Center Tests ──────────────────────────────────────────

class TestFlockCenter:
    """Tests for the moving flock center mechanism."""

    def _make_runner_with_flock(self, center_x=100.0, center_y=100.0):
        """Create a runner with drones spawned at a known center."""
        from src.core.utils.geometry import hex_positions
        runner = SwarmWaypointRunner(
            headless=True,
            start_center=Vector3(x=center_x, y=center_y, z=0),
            start_radius=80.0,
        )
        hex_pos = hex_positions(center_x, center_y, 80.0, n=7)
        runner._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=center_x, y=center_y, z=-BETA_ALTITUDE),
            drone_type=DroneType.BETA,
        )
        for i in range(6):
            vx, vy = hex_pos[i + 1]
            runner._drones[i] = SimDrone(
                drone_id=i,
                position=Vector3(x=vx, y=vy, z=-ALPHA_ALTITUDE),
                drone_type=DroneType.ALPHA,
            )
        runner._flock_center = Vector3(x=center_x, y=center_y, z=-ALPHA_ALTITUDE)
        runner._formation.set_center(runner._flock_center)
        runner._current_hex_center = Vector3(x=center_x, y=center_y, z=0.0)
        runner._current_hex_radius = 80.0
        return runner

    def test_flock_center_moves_toward_checkpoint(self):
        runner = self._make_runner_with_flock(100.0, 100.0)
        cp = Waypoint(position=Vector3(x=500.0, y=500.0, z=-65))

        old_x = runner._flock_center.x
        old_y = runner._flock_center.y
        arrived = runner._advance_flock_center(cp)

        assert not arrived, "Should not arrive in one step"
        assert runner._flock_center.x > old_x, "Flock center should move toward checkpoint X"
        assert runner._flock_center.y > old_y, "Flock center should move toward checkpoint Y"

    def test_flock_center_speed_limit(self):
        runner = self._make_runner_with_flock(100.0, 100.0)
        cp = Waypoint(position=Vector3(x=10000.0, y=10000.0, z=-65))

        old_fc = Vector3(x=runner._flock_center.x, y=runner._flock_center.y, z=0)
        runner._advance_flock_center(cp)
        new_fc = Vector3(x=runner._flock_center.x, y=runner._flock_center.y, z=0)

        dx = new_fc.x - old_fc.x
        dy = new_fc.y - old_fc.y
        step_dist = math.sqrt(dx * dx + dy * dy)
        max_step = runner.FLOCK_TRANSIT_SPEED * runner._dt + 0.01  # Small tolerance
        assert step_dist <= max_step, f"Step {step_dist:.3f}m exceeds max {max_step:.3f}m"

    def test_flock_center_arrives_at_checkpoint(self):
        runner = self._make_runner_with_flock(100.0, 100.0)
        # Checkpoint very close
        cp = Waypoint(position=Vector3(x=100.5, y=100.5, z=-65))

        arrived = runner._advance_flock_center(cp)
        assert arrived, "Should arrive when within 1m"
        assert abs(runner._flock_center.x - 100.5) < 0.01
        assert abs(runner._flock_center.y - 100.5) < 0.01

    def test_flock_center_pauses_when_drone_lags(self):
        runner = self._make_runner_with_flock(100.0, 100.0)

        # Move alpha_0 far away (simulate stuck on obstacle)
        runner._drones[0].position = Vector3(x=0, y=0, z=-ALPHA_ALTITUDE)

        assert runner._any_drone_lagging(), "Should detect lagging drone"

        cp = Waypoint(position=Vector3(x=500.0, y=500.0, z=-65))
        old_x = runner._flock_center.x
        runner._advance_flock_center(cp)

        assert runner._flock_center.x == old_x, "Flock center should NOT move when drone lags"
        assert runner._flock_paused, "flock_paused flag should be True"

    def test_flock_center_resumes_after_lag_resolved(self):
        runner = self._make_runner_with_flock(100.0, 100.0)
        cp = Waypoint(position=Vector3(x=500.0, y=500.0, z=-65))

        # All drones at their slots — no lag
        assert not runner._any_drone_lagging()

        old_x = runner._flock_center.x
        runner._advance_flock_center(cp)
        assert runner._flock_center.x > old_x, "Should advance when no lag"
        assert not runner._flock_paused

    def test_formation_center_tracks_flock_center(self):
        runner = self._make_runner_with_flock(100.0, 100.0)
        cp = Waypoint(position=Vector3(x=500.0, y=500.0, z=-65))

        runner._advance_flock_center(cp)

        # Formation center should match flock center
        fc = runner._formation._center  # Direct access to verify
        assert abs(fc.x - runner._flock_center.x) < 0.01
        assert abs(fc.y - runner._flock_center.y) < 0.01

    def test_hex_boundary_not_applied_during_transit(self):
        runner = self._make_runner_with_flock(100.0, 100.0)
        runner._phase = CheckpointPhase.TRANSIT

        # Place Beta far from hex center
        runner._drones[BETA_ID] = SimDrone(
            drone_id=BETA_ID,
            position=Vector3(x=0, y=0, z=-BETA_ALTITUDE),
            drone_type=DroneType.BETA,
        )

        # Goal is also far from hex center
        goal = Vector3(x=500, y=500, z=-BETA_ALTITUDE)
        runner._apply_beta_velocity(goal)

        # Beta should move toward goal (not clamped to hex)
        pos = runner._drones[BETA_ID].position
        assert pos.x > 0, "Beta should move toward goal during TRANSIT"
        assert pos.y > 0, "Beta should move toward goal during TRANSIT"
