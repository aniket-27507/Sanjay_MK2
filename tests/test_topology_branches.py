"""Tests for src/swarm/topology_branches.py (Avenue 3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import GCopterConfig
from src.swarm.topology_branches import (
    MultiBranchConfig,
    _generate_topology_perturbations,
    _is_better_branch,
    multi_branch_optimize,
)


def _box_polytope(lo, hi):
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


class TestPerturbationGeneration:

    def test_no_interior_waypoints_returns_empty(self) -> None:
        """With only start+end (no interior), nothing to perturb."""
        wps = np.array([[0., 0., 1.], [10., 0., 1.]])
        out = _generate_topology_perturbations(wps, 1.0, 4)
        assert out == []

    def test_zero_forward_returns_empty(self) -> None:
        """Start == end (no direction) → can't construct body frame."""
        wps = np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
        out = _generate_topology_perturbations(wps, 1.0, 4)
        assert out == []

    def test_lateral_perturbations_have_correct_shape(self) -> None:
        """4 branches; lateral perturbations move interior in y; verticals in z."""
        wps = np.array([
            [0., 0., 1.],
            [5., 0., 1.],
            [10., 0., 1.],
        ])
        out = _generate_topology_perturbations(wps, 2.0, 4)
        assert len(out) == 4
        # Branch 0 = lateral +y, branch 1 = lateral -y (forward is +x)
        # so interior waypoint should move ±2 in y
        assert out[0][1, 1] == pytest.approx(+2.0)
        assert out[1][1, 1] == pytest.approx(-2.0)
        # Branches 2, 3 = vertical ± (scale halved)
        assert out[2][1, 2] == pytest.approx(1.0 + 1.0)  # base + scale*0.5*1
        assert out[3][1, 2] == pytest.approx(1.0 - 1.0)
        # Start and end unchanged in every branch
        for branch in out:
            np.testing.assert_allclose(branch[0], wps[0])
            np.testing.assert_allclose(branch[-1], wps[-1])

    def test_n_branches_caps_output(self) -> None:
        wps = np.array([[0., 0., 1.], [5., 0., 1.], [10., 0., 1.]])
        out_2 = _generate_topology_perturbations(wps, 1.0, 2)
        out_4 = _generate_topology_perturbations(wps, 1.0, 4)
        assert len(out_2) == 2
        assert len(out_4) == 4


class TestBranchSelection:

    def test_safety_beats_unsafety(self) -> None:
        """Candidate above clearance, best below → candidate wins."""
        better = _is_better_branch(
            cand_dist=2.5, cand_cost=1e5,
            best_dist=1.0, best_cost=10.0,
            clearance=2.0,
        )
        assert better is True

    def test_dont_lose_safety(self) -> None:
        better = _is_better_branch(
            cand_dist=1.0, cand_cost=1.0,
            best_dist=2.5, best_cost=1e5,
            clearance=2.0,
        )
        assert better is False

    def test_larger_min_dist_wins_when_neither_safe(self) -> None:
        """Both colliding (below clearance), pick the one closer to safety."""
        better = _is_better_branch(
            cand_dist=1.5, cand_cost=1e5,
            best_dist=0.5, best_cost=10.0,
            clearance=2.0,
        )
        assert better is True

    def test_cost_tiebreak_within_band(self) -> None:
        """Distances within tie band → lower cost wins."""
        better = _is_better_branch(
            cand_dist=1.50, cand_cost=10.0,
            best_dist=1.52, best_cost=100.0,
            clearance=2.0,
            distance_tie_band=0.10,
        )
        assert better is True


class TestMultiBranchIntegration:

    @pytest.fixture
    def scenario_no_neighbours(self):
        """No neighbours → branches should not be triggered."""
        wps = np.array([[0., 0., 1.], [5., 0., 1.], [10., 0., 1.]])
        T = np.array([2.5, 2.5])
        bc_start = np.zeros((4, 3)); bc_start[0] = wps[0]
        bc_end = np.zeros((4, 3)); bc_end[0] = wps[-1]
        polytopes = [
            _box_polytope(np.array([-2, -2, 0]), np.array([6, 2, 3])),
            _box_polytope(np.array([4, -2, 0]), np.array([12, 2, 3])),
        ]
        config = GCopterConfig(
            s=3, w_corridor=1e2, w_velocity=1.0, v_max=5.0,
            n_quad=4, maxiter=20,
        )
        return wps, T, bc_start, bc_end, polytopes, config

    def test_no_neighbours_runs_only_main(self, scenario_no_neighbours) -> None:
        wps, T, bc_start, bc_end, polytopes, config = scenario_no_neighbours
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
        assert result.selected_branch_idx == 0
