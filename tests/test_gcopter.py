"""Unit tests for src.single_drone.planning.gcopter.

Phase 0 Task 0.5 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.2, §4.2).

The L-BFGS optimiser shapes (q_interior, T) to minimise
    w_T * sum(T) + w_energy * energy + w_corridor * outside(corridor) + w_velocity * over(v_max)
The tests verify that
    1. After optimisation, the trajectory stays inside its corridor polytopes
       (corridor residual <= small tolerance at quadrature points).
    2. The total cost does not increase relative to the initial guess.
    3. Velocity magnitude stays within / near v_max (soft constraint).
    4. With no obstacles, the optimiser tightens the total time toward the
       lower bound — i.e. time is actively being minimised.
    5. Returned object is a Trajectory hitting both endpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import (
    convex_cover,
    polytope_contains,
)
from src.single_drone.planning.gcopter import GCopterConfig, gcopter_optimize
from src.single_drone.planning.minco import Trajectory


def _zero_bc(s: int, D: int) -> np.ndarray:
    return np.zeros((s + 1, D), dtype=np.float64)


def _world_bounds_for(lo=(0.0, 0.0, 0.0), hi=(20.0, 20.0, 5.0)):
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


class TestOpenCorridor:
    @pytest.fixture
    def optimised(self) -> Trajectory:
        s = 3
        D = 3
        bc_start = _zero_bc(s, D)
        bc_start[0] = [1.0, 5.0, 2.0]
        bc_end = _zero_bc(s, D)
        bc_end[0] = [18.0, 5.0, 2.0]
        waypoints = np.array(
            [
                [1.0, 5.0, 2.0],
                [7.0, 5.0, 2.0],
                [13.0, 5.0, 2.0],
                [18.0, 5.0, 2.0],
            ]
        )
        durations = np.array([2.0, 2.0, 2.0])
        polys = convex_cover(
            route=list(waypoints),
            surface_points=np.zeros((0, 3)),
            world_bounds=_world_bounds_for(),
        )
        cfg = GCopterConfig(v_max=4.0)
        traj = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polys,
            config=cfg,
        )
        return traj

    def test_returns_trajectory(self, optimised: Trajectory) -> None:
        assert isinstance(optimised, Trajectory)

    def test_endpoints_preserved(self, optimised: Trajectory) -> None:
        np.testing.assert_allclose(
            optimised.evaluate(0.0), [1.0, 5.0, 2.0], atol=1e-6
        )
        np.testing.assert_allclose(
            optimised.evaluate(optimised.total_time), [18.0, 5.0, 2.0], atol=1e-5
        )

    def test_velocity_close_to_limit(self, optimised: Trajectory) -> None:
        # with no obstacles, optimiser drives time down; velocity should approach v_max
        ts = np.linspace(0.0, optimised.total_time, 200)
        v_norms = np.array(
            [np.linalg.norm(optimised.evaluate(t, 1)) for t in ts]
        )
        # soft penalty allows a small overshoot
        assert v_norms.max() <= 4.0 * 1.10


class TestCorridorContainment:
    def test_corridor_residual_small_after_optimisation(self) -> None:
        s = 3
        D = 3
        bc_start = _zero_bc(s, D)
        bc_start[0] = [1.0, 5.0, 2.0]
        bc_end = _zero_bc(s, D)
        bc_end[0] = [18.0, 5.0, 2.0]
        waypoints = np.array(
            [
                [1.0, 5.0, 2.0],
                [7.0, 5.0, 2.0],
                [13.0, 5.0, 2.0],
                [18.0, 5.0, 2.0],
            ]
        )
        # surface points clamping the corridor in y
        surface = np.array(
            [
                [4.0, 8.0, 2.0],
                [10.0, 8.0, 2.0],
                [16.0, 8.0, 2.0],
                [4.0, 2.0, 2.0],
                [10.0, 2.0, 2.0],
                [16.0, 2.0, 2.0],
            ]
        )
        polys = convex_cover(
            list(waypoints), surface, _world_bounds_for()
        )
        traj = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=np.array([2.0, 2.0, 2.0]),
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polys,
            config=GCopterConfig(v_max=4.0, n_quad=20),
        )
        # check trajectory at quadrature points within each segment
        for k in range(traj.M):
            A_k, b_k = polys[k].A, polys[k].b
            ts = np.linspace(traj.knot_times[k], traj.knot_times[k + 1], 25)
            for t in ts:
                p = traj.evaluate(t, 0)
                residual = float(np.max(A_k @ p - b_k))
                # small numerical leak permitted; soft penalty has finite gain
                assert residual <= 0.1, (
                    f"trajectory leaves segment {k} polytope by {residual} at t={t}"
                )


class TestCostMonotonicity:
    def test_cost_does_not_increase(self) -> None:
        from src.single_drone.planning.gcopter import _evaluate_cost  # internal helper

        s = 3
        D = 3
        bc_start = _zero_bc(s, D)
        bc_start[0] = [1.0, 5.0, 2.0]
        bc_end = _zero_bc(s, D)
        bc_end[0] = [18.0, 5.0, 2.0]
        waypoints = np.array(
            [
                [1.0, 5.0, 2.0],
                [10.0, 5.0, 2.0],
                [18.0, 5.0, 2.0],
            ]
        )
        polys = convex_cover(list(waypoints), np.zeros((0, 3)), _world_bounds_for())
        cfg = GCopterConfig(v_max=4.0)

        initial_traj = Trajectory(waypoints, np.array([3.0, 3.0]), bc_start, bc_end, s=s)
        initial_cost = _evaluate_cost(initial_traj, polys, cfg)

        optimised = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=np.array([3.0, 3.0]),
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polys,
            config=cfg,
        )
        final_cost = _evaluate_cost(optimised, polys, cfg)
        assert final_cost <= initial_cost + 1e-6


class TestTimeMinimisation:
    def test_total_time_decreases_with_no_obstacles(self) -> None:
        s = 3
        D = 3
        bc_start = _zero_bc(s, D)
        bc_start[0] = [1.0, 5.0, 2.0]
        bc_end = _zero_bc(s, D)
        bc_end[0] = [18.0, 5.0, 2.0]
        waypoints = np.array(
            [
                [1.0, 5.0, 2.0],
                [10.0, 5.0, 2.0],
                [18.0, 5.0, 2.0],
            ]
        )
        polys = convex_cover(list(waypoints), np.zeros((0, 3)), _world_bounds_for())
        # give it a lazy initial schedule so optimisation has room to tighten
        traj = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=np.array([5.0, 5.0]),
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polys,
            config=GCopterConfig(v_max=4.0),
        )
        assert traj.total_time < 10.0  # initial total was 10.0
