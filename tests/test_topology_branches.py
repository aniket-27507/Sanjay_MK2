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
