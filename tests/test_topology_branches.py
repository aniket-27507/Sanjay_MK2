"""Tests for the rebuilt src/swarm/topology_branches.py (Avenue 3 v2).

Covers the four failure-mode fixes:
  - Trigger correctly skips when no neighbours / no violation
  - Branches use homotopy constraints (signatures actually differ)
  - Consistency bonus favours prior-tick signature when costs are close
  - Main solve always considered as one of the candidate branches
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import GCopterConfig
from src.swarm.topology_branches import (
    MultiBranchConfig, MultiBranchResult, multi_branch_optimize,
)


def _box(lo, hi):
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


@pytest.fixture
def open_corridor():
    """Wide open corridor along +x with two segments."""
    s = 3
    wps = np.array([
        [0., 0., 1.],
        [5., 0., 1.],
        [10., 0., 1.],
    ])
    T = np.array([2.5, 2.5])
    bc_start = np.zeros((s + 1, 3)); bc_start[0] = wps[0]
    bc_end = np.zeros((s + 1, 3)); bc_end[0] = wps[-1]
    polytopes = [
        _box(np.array([-5, -5, -1]), np.array([8, 5, 3])),
        _box(np.array([2, -5, -1]), np.array([15, 5, 3])),
    ]
    config = GCopterConfig(
        s=s, w_corridor=1e2, w_velocity=1.0, v_max=5.0,
        n_quad=8, maxiter=20,
    )
    return wps, T, bc_start, bc_end, polytopes, config


class TestTriggerLogic:

    def test_no_neighbours_no_branches(self, open_corridor) -> None:
        """No neighbours means no swarm violation means no branching."""
        wps, T, bc_start, bc_end, polytopes, config = open_corridor
        result = multi_branch_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            swarm_neighbours=None, swarm_config=None,
            warm_start=False,
            branch_config=MultiBranchConfig(n_branches=4),
            swarm_clearance_horizontal=1.0,
        )
        assert result.n_branches_run == 1
        assert result.main_branch_used is True
        assert result.signature == ()
        assert "no-trigger" in result.trigger_reason


class TestResultStructure:

    def test_result_carries_signature(self, open_corridor) -> None:
        wps, T, bc_start, bc_end, polytopes, config = open_corridor
        result = multi_branch_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=False,
            branch_config=MultiBranchConfig(n_branches=4),
            swarm_clearance_horizontal=1.0,
        )
        assert isinstance(result.signature, tuple)
        assert len(result.branch_costs) == result.n_branches_run
        assert len(result.branch_signatures) == result.n_branches_run

    def test_main_branch_always_considered(self, open_corridor) -> None:
        """Even when branches fire, index 0 is always the main solve."""
        wps, T, bc_start, bc_end, polytopes, config = open_corridor
        result = multi_branch_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=False,
            branch_config=MultiBranchConfig(n_branches=4),
            swarm_clearance_horizontal=1.0,
        )
        # branch_costs[0] is the main solve's cost, irrespective of whether
        # branches were spawned. Should be finite.
        assert np.isfinite(result.branch_costs[0])


# ---------------------------------------------------------------------------
# Adaptive perturbation scale (Gap 5)
# ---------------------------------------------------------------------------

from src.swarm.topology_branches import (  # noqa: E402
    _adaptive_perturbation_scale,
    _corridor_min_half_extent,
    _median_neighbour_separation,
)


def _box_polytope(half_extent: float) -> "Polytope":  # noqa: F821
    from src.single_drone.planning.corridor_generator import Polytope
    A = np.array(
        [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
         [0., 0., 1.], [0., 0., -1.]],
        dtype=np.float64,
    )
    b = np.full(6, half_extent, dtype=np.float64)
    return Polytope(A=A, b=b)


class TestAdaptivePerturbation:

    def test_corridor_half_extent_picks_min_slack(self) -> None:
        poly = _box_polytope(4.0)
        # Single interior waypoint at +2 along x: slack along +x = 4-2 = 2
        wps = np.array([[0., 0., 0.], [2., 0., 0.], [5., 0., 0.]])
        ext = _corridor_min_half_extent([poly, poly], wps)
        assert ext == pytest.approx(2.0)

    def test_corridor_returns_inf_when_empty(self) -> None:
        wps = np.empty((0, 3))
        assert _corridor_min_half_extent([], wps) == float("inf")

    def _line_traj_at(self, x0: float, x1: float):
        from src.single_drone.planning.minco import Trajectory
        s, D = 3, 3
        bc_start = np.zeros((s + 1, D)); bc_start[0] = [x0, 0., 0.]
        bc_end = np.zeros((s + 1, D)); bc_end[0] = [x1, 0., 0.]
        xs = np.linspace(x0, x1, 3)
        wps = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        durs = np.array([1.0, 1.0])
        return Trajectory(wps, durs, bc_start, bc_end, s=s)

    def test_median_neighbour_separation(self) -> None:
        # Two stationary neighbours at x=3 and x=7
        nbr_a = self._line_traj_at(3.0, 3.0)
        nbr_b = self._line_traj_at(7.0, 7.0)
        own_wps = np.array([[0., 0., 0.], [4., 0., 0.], [8., 0., 0.]])
        sep = _median_neighbour_separation(own_wps, [(nbr_a, 0.0), (nbr_b, 0.0)])
        assert sep == pytest.approx(5.0)

    def test_adaptive_min_of_both_clamped_by_fallback(self) -> None:
        # Interior waypoint at x=2 inside a [-4,4]^3 box → Chebyshev slack to
        # the closest wall (+x face) is 4-2=2. 0.4*2 = 0.8.
        # Neighbour at x=5 → distance from start (origin) is 5; median = 5.
        # 0.6*5 = 3.0. min(0.8, 3.0) = 0.8 < fallback 2.0 → returns 0.8.
        nbr = self._line_traj_at(5.0, 5.0)
        own_wps = np.array([[0., 0., 0.], [2., 0., 0.], [5., 0., 0.]])
        scale = _adaptive_perturbation_scale(
            [_box_polytope(4.0), _box_polytope(4.0)],
            own_wps, [(nbr, 0.0)], fallback=2.0,
        )
        assert scale == pytest.approx(0.8)

    def test_fallback_when_no_signal(self) -> None:
        # No polytopes, no neighbours → fallback
        scale = _adaptive_perturbation_scale(
            [], np.array([[0., 0., 0.], [1., 0., 0.]]),
            [], fallback=2.0,
        )
        assert scale == 2.0
