"""Finite-difference verification of the COMPOSITE gcopter `_cost_and_grad`.

The individual pieces (energy in test_minco.py, swarm in test_swarm_penalty.py)
have FD tests. The assembled gcopter cost — the function that L-BFGS consumes
— does not, until this file. That's the gap that lets chain-rule bugs hide.

What we verify:
  1. Pure energy + corridor + velocity gradient (no swarm)
  2. Same plus swarm penalty (drone-pair scenario)

We compare analytical (q, T) gradients against centered finite differences on
the same composite cost the optimiser sees. Tolerances are tighter than the
existing per-component tests because chain-rule bugs typically produce O(1)
errors, not O(eps) errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import (
    GCopterConfig,
    _cost_and_grad,
    _evaluate_cost,
)
from src.single_drone.planning.minco import Trajectory


def _make_polytope(lo, hi):
    """Axis-aligned box polytope {x : A x <= b}."""
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


def _build_traj(wps, T, s=3):
    bc_start = np.zeros((s + 1, wps.shape[1]))
    bc_start[0] = wps[0]
    bc_end = np.zeros((s + 1, wps.shape[1]))
    bc_end[0] = wps[-1]
    return Trajectory(wps, T, bc_start, bc_end, s=s), bc_start, bc_end


def _composite_cost_at(wps, T, polytopes, config):
    """Pure-cost evaluator that mirrors what L-BFGS sees (no gradient)."""
    traj, _, _ = _build_traj(wps, T, s=config.s)
    return _evaluate_cost(traj, polytopes, config)


def _fd_grad_q_composite(wps, T, polytopes, config, eps=5e-6):
    M = T.size
    D = wps.shape[1]
    gq = np.zeros((M - 1, D))
    for k in range(M - 1):
        for d in range(D):
            wp_plus = wps.copy(); wp_plus[k + 1, d] += eps
            wp_minus = wps.copy(); wp_minus[k + 1, d] -= eps
            cp = _composite_cost_at(wp_plus, T, polytopes, config)
            cm = _composite_cost_at(wp_minus, T, polytopes, config)
            gq[k, d] = (cp - cm) / (2 * eps)
    return gq


def _fd_grad_T_composite(wps, T, polytopes, config, eps=5e-6):
    M = T.size
    gT = np.zeros(M)
    for k in range(M):
        T_plus = T.copy(); T_plus[k] += eps
        T_minus = T.copy(); T_minus[k] -= eps
        cp = _composite_cost_at(wps, T_plus, polytopes, config)
        cm = _composite_cost_at(wps, T_minus, polytopes, config)
        gT[k] = (cp - cm) / (2 * eps)
    return gT


# ---------------------------------------------------------------------------
# Scenarios — designed so each cost component is non-trivial at the test point.
# ---------------------------------------------------------------------------


@pytest.fixture
def composite_scenario():
    """3-segment trajectory crossing a polytope corridor at near-limit speed."""
    # Endpoints far apart, interior waypoints offset off the straight line to
    # ensure the corridor penalty is active.
    wps = np.array([
        [0.0,  0.0, 1.0],
        [3.0,  1.5, 1.0],  # interior 1 — offset in y
        [6.0, -0.5, 1.0],  # interior 2 — offset in y the other way
        [9.0,  0.0, 1.0],
    ])
    T = np.array([1.2, 1.0, 1.4])
    # Three polytopes, one per segment, each containing its sub-leg with
    # MODEST slack so corridor leaks are non-zero when the polynomial bulges.
    polytopes = [
        _make_polytope(np.array([-0.5, -0.5,  0.5]), np.array([3.5,  2.5, 2.0])),
        _make_polytope(np.array([ 2.5, -2.5,  0.5]), np.array([6.5,  2.5, 2.0])),
        _make_polytope(np.array([ 5.5, -2.5,  0.5]), np.array([9.5,  1.5, 2.0])),
    ]
    config = GCopterConfig(
        s=3,
        w_time=1.0,
        w_energy=1e-2,
        w_corridor=1.0e3,
        w_velocity=1.0e1,
        v_max=3.0,
        n_quad=8,
    )
    return wps, T, polytopes, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompositeGradient:
    """End-to-end FD vs analytical for what L-BFGS actually sees."""

    def test_grad_q_matches_fd(self, composite_scenario) -> None:
        wps, T, polytopes, config = composite_scenario
        traj, _, _ = _build_traj(wps, T, s=config.s)
        _, gq_ana, _ = _cost_and_grad(traj, polytopes, config)
        gq_fd = _fd_grad_q_composite(wps, T, polytopes, config)

        # Relative tolerance: 1e-3 is loose, but FD noise at eps=5e-6 has
        # its own floor (~1e-4 for these magnitudes).
        np.testing.assert_allclose(gq_ana, gq_fd, rtol=1e-3, atol=1e-3,
                                    err_msg=f"\nAnalytical:\n{gq_ana}\nFD:\n{gq_fd}")

    def test_grad_T_matches_fd(self, composite_scenario) -> None:
        wps, T, polytopes, config = composite_scenario
        traj, _, _ = _build_traj(wps, T, s=config.s)
        _, _, gT_ana = _cost_and_grad(traj, polytopes, config)
        gT_fd = _fd_grad_T_composite(wps, T, polytopes, config)

        np.testing.assert_allclose(gT_ana, gT_fd, rtol=1e-3, atol=1e-3,
                                    err_msg=f"\nAnalytical:\n{gT_ana}\nFD:\n{gT_fd}")

    def test_cost_value_matches(self, composite_scenario) -> None:
        """Sanity: _cost_and_grad and _evaluate_cost should return the same scalar."""
        wps, T, polytopes, config = composite_scenario
        traj, _, _ = _build_traj(wps, T, s=config.s)
        cost_with_grad, _, _ = _cost_and_grad(traj, polytopes, config)
        cost_eval = _evaluate_cost(traj, polytopes, config)
        np.testing.assert_allclose(cost_with_grad, cost_eval, rtol=1e-10)


class TestCompositeGradientWithSwarm:
    """Same FD oracle but with a neighbour broadcasting a trajectory."""

    @pytest.fixture
    def swarm_scenario(self, composite_scenario):
        from src.swarm.swarm_penalty import SwarmPenaltyConfig
        wps, T, polytopes, config = composite_scenario
        # Place a "neighbour" right in the path so the swarm penalty is active.
        nb_wps = np.array([
            [3.0,  0.0, 1.0],
            [4.5,  0.5, 1.0],
            [6.0,  0.0, 1.0],
        ])
        nb_T = np.array([1.5, 1.5])
        nb_traj, _, _ = _build_traj(nb_wps, nb_T, s=config.s)
        sw_cfg = SwarmPenaltyConfig(
            clearance_horizontal=2.0,
            clearance_vertical=1.0,
            weight=1.0e3,
            n_quad=8,
        )
        return wps, T, polytopes, config, [(nb_traj, 0.0)], sw_cfg

    def _composite_with_swarm(self, wps, T, polytopes, config, neighbours, sw_cfg):
        from src.swarm.swarm_penalty import compute_swarm_cost_and_grad
        traj, _, _ = _build_traj(wps, T, s=config.s)
        cost = _evaluate_cost(traj, polytopes, config)
        sc, _, _ = compute_swarm_cost_and_grad(traj, neighbours, sw_cfg)
        return cost + sc

    def test_grad_q_with_swarm_matches_fd(self, swarm_scenario) -> None:
        from src.swarm.swarm_penalty import compute_swarm_cost_and_grad
        wps, T, polytopes, config, neighbours, sw_cfg = swarm_scenario

        # Analytical = corridor/vel/energy grad + swarm grad
        traj, _, _ = _build_traj(wps, T, s=config.s)
        _, gq_base, _ = _cost_and_grad(traj, polytopes, config)
        _, gq_sw, _ = compute_swarm_cost_and_grad(traj, neighbours, sw_cfg)
        gq_ana = gq_base + gq_sw

        # FD
        eps = 5e-6
        M = T.size; D = wps.shape[1]
        gq_fd = np.zeros((M - 1, D))
        for k in range(M - 1):
            for d in range(D):
                wp_plus = wps.copy(); wp_plus[k + 1, d] += eps
                wp_minus = wps.copy(); wp_minus[k + 1, d] -= eps
                cp = self._composite_with_swarm(wp_plus, T, polytopes, config, neighbours, sw_cfg)
                cm = self._composite_with_swarm(wp_minus, T, polytopes, config, neighbours, sw_cfg)
                gq_fd[k, d] = (cp - cm) / (2 * eps)

        np.testing.assert_allclose(gq_ana, gq_fd, rtol=1e-3, atol=1e-3,
                                    err_msg=f"\nAnalytical:\n{gq_ana}\nFD:\n{gq_fd}")

    def test_grad_T_with_swarm_matches_fd(self, swarm_scenario) -> None:
        from src.swarm.swarm_penalty import compute_swarm_cost_and_grad
        wps, T, polytopes, config, neighbours, sw_cfg = swarm_scenario

        traj, _, _ = _build_traj(wps, T, s=config.s)
        _, _, gT_base = _cost_and_grad(traj, polytopes, config)
        _, _, gT_sw = compute_swarm_cost_and_grad(traj, neighbours, sw_cfg)
        gT_ana = gT_base + gT_sw

        eps = 5e-6
        M = T.size
        gT_fd = np.zeros(M)
        for k in range(M):
            T_plus = T.copy(); T_plus[k] += eps
            T_minus = T.copy(); T_minus[k] -= eps
            cp = self._composite_with_swarm(wps, T_plus, polytopes, config, neighbours, sw_cfg)
            cm = self._composite_with_swarm(wps, T_minus, polytopes, config, neighbours, sw_cfg)
            gT_fd[k] = (cp - cm) / (2 * eps)

        np.testing.assert_allclose(gT_ana, gT_fd, rtol=1e-3, atol=1e-3,
                                    err_msg=f"\nAnalytical:\n{gT_ana}\nFD:\n{gT_fd}")
