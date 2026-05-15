"""Tests for src.swarm.swarm_penalty — ellipsoidal inter-drone penalty + grads."""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.minco import Trajectory
from src.swarm.swarm_penalty import (
    SwarmPenaltyConfig,
    compute_swarm_cost_and_grad,
)


def _line_traj(
    start_x: float,
    end_x: float,
    y: float = 0.0,
    z: float = 2.0,
    durations=(1.0, 1.0),
) -> Trajectory:
    s, D = 3, 3
    bc_start = np.zeros((s + 1, D))
    bc_start[0] = [start_x, y, z]
    bc_end = np.zeros((s + 1, D))
    bc_end[0] = [end_x, y, z]
    wps = np.array(
        [[start_x, y, z], [(start_x + end_x) / 2, y, z], [end_x, y, z]]
    )
    durations = np.asarray(durations, dtype=np.float64)
    return Trajectory(wps, durations, bc_start, bc_end, s=s)


def _make_own(y_own: float = 0.0) -> Trajectory:
    """Own trajectory: moves from (-5, y_own, 2) to (+5, y_own, 2) over 2 s."""
    return _line_traj(-5.0, 5.0, y=y_own, z=2.0)


class TestPenaltyBasics:
    def test_zero_cost_when_far_apart(self) -> None:
        own = _make_own(y_own=0.0)
        nb = _line_traj(-5.0, 5.0, y=20.0)  # 20 m away in y, far outside clearance
        cost, gq, gT = compute_swarm_cost_and_grad(
            own, [(nb, 0.0)], SwarmPenaltyConfig(weight=1.0)
        )
        assert cost == pytest.approx(0.0)
        assert np.allclose(gq, 0.0)
        assert np.allclose(gT, 0.0)

    def test_nonzero_cost_when_overlapping(self) -> None:
        # offset slightly so we're inside the clearance ellipsoid but NOT at the
        # symmetry minimum (where the gradient legitimately vanishes).
        own = _make_own(y_own=0.3)
        nb = _line_traj(-5.0, 5.0, y=0.0)
        cost, gq, gT = compute_swarm_cost_and_grad(
            own, [(nb, 0.0)], SwarmPenaltyConfig(weight=1.0)
        )
        assert cost > 0.0
        assert not np.allclose(gq, 0.0)

    def test_no_neighbours_returns_zero(self) -> None:
        own = _make_own()
        cost, gq, gT = compute_swarm_cost_and_grad(
            own, [], SwarmPenaltyConfig()
        )
        assert cost == 0.0
        assert np.allclose(gq, 0.0)
        assert np.allclose(gT, 0.0)


class TestPenaltyGradient:
    @staticmethod
    def _swap_q(own: Trajectory, q_int: np.ndarray) -> Trajectory:
        wps = own.waypoints.copy()
        wps[1:-1] = q_int
        return Trajectory(wps, own.durations, own.bc_start, own.bc_end, s=own.s)

    @staticmethod
    def _swap_T(own: Trajectory, T: np.ndarray) -> Trajectory:
        return Trajectory(own.waypoints, T, own.bc_start, own.bc_end, s=own.s)

    @staticmethod
    def _cost_only(own, neighbours, cfg):
        c, _, _ = compute_swarm_cost_and_grad(own, neighbours, cfg)
        return c

    def test_fd_matches_grad_q(self) -> None:
        own = _make_own(y_own=0.5)
        nb = _line_traj(-5.0, 5.0, y=0.0)
        cfg = SwarmPenaltyConfig(weight=1.0, clearance_horizontal=2.0)
        _, gq, _ = compute_swarm_cost_and_grad(own, [(nb, 0.0)], cfg)
        eps = 1e-5
        q0 = own.waypoints[1:-1].copy()
        gq_fd = np.zeros_like(gq)
        for k in range(gq.shape[0]):
            for d in range(gq.shape[1]):
                qp = q0.copy(); qp[k, d] += eps
                qm = q0.copy(); qm[k, d] -= eps
                Cp = self._cost_only(self._swap_q(own, qp), [(nb, 0.0)], cfg)
                Cm = self._cost_only(self._swap_q(own, qm), [(nb, 0.0)], cfg)
                gq_fd[k, d] = (Cp - Cm) / (2 * eps)
        rel = np.linalg.norm(gq - gq_fd) / (np.linalg.norm(gq) + 1e-8)
        assert rel < 1e-2, f"swarm-grad q relative error = {rel}"

    def test_fd_matches_grad_T(self) -> None:
        own = _make_own(y_own=0.5)
        # Give nb longer duration and a negative t_offset so own's full quadrature
        # window sits strictly inside nb's defined range — avoids the cost-
        # function discontinuity at the trajectory boundary (overlap window
        # boundaries are not differentiable; the analytical gradient is correct
        # at interior points where the cost is smooth).
        nb = _line_traj(-7.0, 7.0, y=0.0, durations=(2.0, 2.0))
        t_offset = -1.0  # nb absolute window: [-1, 3], own's: [0, 2]
        cfg = SwarmPenaltyConfig(weight=1.0, clearance_horizontal=2.0)
        _, _, gT = compute_swarm_cost_and_grad(own, [(nb, t_offset)], cfg)
        eps = 1e-5
        T0 = own.durations.copy()
        gT_fd = np.zeros_like(gT)
        for k in range(T0.size):
            Tp = T0.copy(); Tp[k] += eps
            Tm = T0.copy(); Tm[k] -= eps
            Cp = self._cost_only(self._swap_T(own, Tp), [(nb, t_offset)], cfg)
            Cm = self._cost_only(self._swap_T(own, Tm), [(nb, t_offset)], cfg)
            gT_fd[k] = (Cp - Cm) / (2 * eps)
        rel = np.linalg.norm(gT - gT_fd) / (np.linalg.norm(gT) + 1e-8)
        assert rel < 1e-2, f"swarm-grad T relative error = {rel}"


class TestEllipsoidGeometry:
    def test_z_axis_compressed(self) -> None:
        """Same Δposition in x vs in z should yield more penalty in z when cz < cx."""
        own = _make_own(y_own=0.0)
        nb_x = _line_traj(-5.0, 5.0, y=1.5)  # 1.5 m offset in y (within x-clear 2 m)
        nb_z = _line_traj(-5.0, 5.0, y=0.0, z=2.0 + 0.7)  # 0.7 m offset in z (within z-clear 1.0)
        cfg = SwarmPenaltyConfig(clearance_horizontal=2.0, clearance_vertical=1.0)
        cost_y, _, _ = compute_swarm_cost_and_grad(own, [(nb_x, 0.0)], cfg)
        cost_z, _, _ = compute_swarm_cost_and_grad(own, [(nb_z, 0.0)], cfg)
        # 0.7 / 1.0 = 0.7 vs 1.5 / 2.0 = 0.75 — z-offset is *relatively* closer
        assert cost_z > cost_y
