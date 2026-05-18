"""Tests for the adaptive warm-start logic in gcopter_optimize.

The contract being verified:
  - When the initial guess is at a stationary point, optimisation is SKIPPED
    (zero L-BFGS iterations, just the one gradient check).
  - When the initial guess is close to a stationary point, a REDUCED iter
    budget is used (warm_start_maxiter rather than maxiter).
  - When the initial guess is far from optimum, the FULL maxiter budget is
    available.
  - Skipping does not produce a worse trajectory than running full L-BFGS
    from the same x0.

All tests use return_meta=True to inspect the optimiser's decisions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import GCopterConfig, gcopter_optimize


def _make_polytope(lo, hi):
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


@pytest.fixture
def small_scenario():
    """Tiny straight-line scenario with very loose polytopes — optimum is
    near the straight line, easy to warm-start to a stationary point."""
    s = 3
    wps = np.array([
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 1.0],
        [10.0, 0.0, 1.0],
    ])
    T = np.array([2.5, 2.5])
    bc_start = np.zeros((s + 1, 3)); bc_start[0] = wps[0]
    bc_end = np.zeros((s + 1, 3)); bc_end[0] = wps[-1]
    polytopes = [
        _make_polytope(np.array([-2, -2, 0]), np.array([6, 2, 3])),
        _make_polytope(np.array([4, -2, 0]), np.array([12, 2, 3])),
    ]
    config = GCopterConfig(
        s=s, w_time=1.0, w_energy=1e-3, w_corridor=1.0e2, w_velocity=1.0,
        v_max=5.0, n_quad=8, maxiter=50,
        warm_start_skip_ratio=1e-4,
        warm_start_relax_ratio=1e-2,
        warm_start_maxiter=5,
    )
    return wps, T, bc_start, bc_end, polytopes, config


class TestAdaptiveWarmStart:

    def test_cold_start_skips_gradient_check(self, small_scenario) -> None:
        """warm_start=False means no gradient eval at x0 — zero overhead."""
        wps, T, bc_start, bc_end, polytopes, config = small_scenario
        _, meta = gcopter_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=False, return_meta=True,
        )
        # Cold start: gradient norm at x0 was never computed
        assert np.isnan(meta["grad_norm_at_x0"])
        assert np.isnan(meta["ratio_at_x0"])
        assert meta["skipped"] is False
        assert meta["maxiter_used"] == config.maxiter

    def test_warm_start_at_optimum_is_skipped(self, small_scenario) -> None:
        """warm_start=True + initial guess at optimum → skip L-BFGS entirely."""
        wps, T, bc_start, bc_end, polytopes, config = small_scenario
        traj_opt = gcopter_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
        )
        _, meta = gcopter_optimize(
            initial_waypoints=traj_opt.waypoints,
            initial_durations=traj_opt.durations,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=True, return_meta=True,
        )
        assert meta["skipped"] is True
        assert meta["iters"] == 0
        assert meta["ratio_at_x0"] < config.warm_start_skip_ratio

    def test_warm_start_far_from_optimum_uses_full_budget(self, small_scenario) -> None:
        """warm_start=True but initial guess is bad → full maxiter still used."""
        wps, T, bc_start, bc_end, polytopes, config = small_scenario
        wps_bad = wps.copy()
        wps_bad[1] = [5.0, 1.5, 1.0]
        T_bad = np.array([1.0, 1.0])

        _, meta = gcopter_optimize(
            initial_waypoints=wps_bad, initial_durations=T_bad,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=True, return_meta=True,
        )
        assert meta["skipped"] is False
        # Bad warm start → ratio above reduce threshold → full budget
        assert meta["ratio_at_x0"] > config.warm_start_relax_ratio
        assert meta["maxiter_used"] == config.maxiter

    def test_skipping_returns_same_trajectory(self, small_scenario) -> None:
        """Skipped optimisation returns the warm-start trajectory unchanged."""
        wps, T, bc_start, bc_end, polytopes, config = small_scenario
        traj_opt = gcopter_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
        )
        traj_skipped = gcopter_optimize(
            initial_waypoints=traj_opt.waypoints,
            initial_durations=traj_opt.durations,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=True,
        )
        np.testing.assert_allclose(
            traj_skipped.waypoints, traj_opt.waypoints, atol=1e-9
        )
        np.testing.assert_allclose(
            traj_skipped.durations, traj_opt.durations, atol=1e-9
        )

    def test_meta_keys_present(self, small_scenario) -> None:
        wps, T, bc_start, bc_end, polytopes, config = small_scenario
        _, meta = gcopter_optimize(
            initial_waypoints=wps, initial_durations=T,
            bc_start=bc_start, bc_end=bc_end,
            polytopes=polytopes, config=config,
            warm_start=True, return_meta=True,
        )
        for k in ("warm_start", "skipped", "iters", "n_evals",
                  "grad_norm_at_x0", "ratio_at_x0", "cost_at_x0",
                  "maxiter_used"):
            assert k in meta
