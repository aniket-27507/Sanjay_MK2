"""Tests for src.swarm.ghost_obstacles — soft no-fly regions + analytical gradient.

Gap 2 part 1: standalone module, no rig integration yet. Tests cover:

- Cost is zero outside the ellipsoid, positive inside.
- Multi-ghost summation is linear in the ghost list.
- Analytical grad_q and grad_T match finite differences within 1 %.
- Ellipsoidal shape: same Δposition costs more in z (where clearance is
  smaller) than in xy.
- `gcopter_optimize(... ghost_obstacles=...)` actually detours the
  optimised trajectory around the ghost.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import GCopterConfig, gcopter_optimize
from src.single_drone.planning.minco import Trajectory
from src.swarm.ghost_obstacles import (
    GhostManager,
    GhostManagerConfig,
    GhostObstacle,
    GhostObstacleConfig,
    compute_ghost_cost_and_grad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _wide_box(
    start: np.ndarray, end: np.ndarray, half=(5.0, 5.0, 5.0)
) -> Polytope:
    """One fat axis-aligned bounding box around the straight segment."""
    lo = np.minimum(start, end) - np.asarray(half)
    hi = np.maximum(start, end) + np.asarray(half)
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


class TestCostBasics:
    def test_zero_cost_when_no_ghosts(self) -> None:
        own = _line_traj(-5.0, 5.0)
        cost, gq, gT = compute_ghost_cost_and_grad(
            own, [], GhostObstacleConfig()
        )
        assert cost == 0.0
        assert np.allclose(gq, 0.0)
        assert np.allclose(gT, 0.0)

    def test_zero_cost_when_ghost_is_far(self) -> None:
        own = _line_traj(-5.0, 5.0, y=0.0, z=2.0)
        ghost = GhostObstacle(
            center=np.array([0.0, 50.0, 2.0]),  # 50 m off the path in y
            clearance_horizontal=2.0,
            clearance_vertical=1.0,
            weight=1.0e3,
        )
        cost, gq, gT = compute_ghost_cost_and_grad(
            own, [ghost], GhostObstacleConfig()
        )
        assert cost == pytest.approx(0.0)
        assert np.allclose(gq, 0.0)
        assert np.allclose(gT, 0.0)

    def test_nonzero_cost_when_path_crosses_ghost(self) -> None:
        # Ghost sits on top of the path midpoint with positive y-offset
        # so the gradient is non-trivial (not at the symmetry minimum).
        own = _line_traj(-5.0, 5.0, y=0.3, z=2.0)
        ghost = GhostObstacle(
            center=np.array([0.0, 0.0, 2.0]),
            clearance_horizontal=2.0,
            clearance_vertical=1.0,
            weight=1.0e3,
        )
        cost, gq, gT = compute_ghost_cost_and_grad(
            own, [ghost], GhostObstacleConfig()
        )
        assert cost > 0.0
        # Gradient should be non-zero in y (the off-axis direction).
        assert not np.allclose(gq, 0.0)

    def test_zero_weight_ghost_skipped(self) -> None:
        """A ghost with weight=0 contributes nothing — useful for caller-
        side decay schemes that fade ghosts out rather than removing
        them."""
        own = _line_traj(-5.0, 5.0)
        ghost = GhostObstacle(
            center=np.array([0.0, 0.0, 2.0]),
            clearance_horizontal=2.0,
            clearance_vertical=1.0,
            weight=0.0,
        )
        cost, _, _ = compute_ghost_cost_and_grad(
            own, [ghost], GhostObstacleConfig()
        )
        assert cost == 0.0

    def test_multi_ghost_cost_is_sum_of_singles(self) -> None:
        own = _line_traj(-5.0, 5.0, y=0.3, z=2.0)
        g1 = GhostObstacle(center=np.array([-2.0, 0.0, 2.0]), weight=1.0e3)
        g2 = GhostObstacle(center=np.array([+2.0, 0.0, 2.0]), weight=1.0e3)
        c1, _, _ = compute_ghost_cost_and_grad(own, [g1], GhostObstacleConfig())
        c2, _, _ = compute_ghost_cost_and_grad(own, [g2], GhostObstacleConfig())
        cboth, _, _ = compute_ghost_cost_and_grad(
            own, [g1, g2], GhostObstacleConfig()
        )
        assert cboth == pytest.approx(c1 + c2, rel=1e-12)

    def test_invalid_clearance_raises(self) -> None:
        own = _line_traj(-5.0, 5.0)
        bad = GhostObstacle(
            center=np.zeros(3),
            clearance_horizontal=0.0,
            clearance_vertical=1.0,
            weight=1.0,
        )
        with pytest.raises(ValueError):
            compute_ghost_cost_and_grad(own, [bad], GhostObstacleConfig())


# ---------------------------------------------------------------------------
# Gradient (finite-difference checks)
# ---------------------------------------------------------------------------


class TestGradient:
    @staticmethod
    def _swap_q(own: Trajectory, q_int: np.ndarray) -> Trajectory:
        wps = own.waypoints.copy()
        wps[1:-1] = q_int
        return Trajectory(wps, own.durations, own.bc_start, own.bc_end, s=own.s)

    @staticmethod
    def _swap_T(own: Trajectory, T: np.ndarray) -> Trajectory:
        return Trajectory(own.waypoints, T, own.bc_start, own.bc_end, s=own.s)

    @staticmethod
    def _cost_only(own, ghosts, cfg):
        c, _, _ = compute_ghost_cost_and_grad(own, ghosts, cfg)
        return c

    def test_fd_matches_grad_q(self) -> None:
        own = _line_traj(-5.0, 5.0, y=0.5, z=2.0)
        ghost = GhostObstacle(
            center=np.array([0.0, 0.0, 2.0]),
            clearance_horizontal=2.0,
            clearance_vertical=1.0,
            weight=1.0,
        )
        cfg = GhostObstacleConfig()
        _, gq, _ = compute_ghost_cost_and_grad(own, [ghost], cfg)
        eps = 1e-5
        q0 = own.waypoints[1:-1].copy()
        gq_fd = np.zeros_like(gq)
        for k in range(gq.shape[0]):
            for d in range(gq.shape[1]):
                qp = q0.copy(); qp[k, d] += eps
                qm = q0.copy(); qm[k, d] -= eps
                Cp = self._cost_only(self._swap_q(own, qp), [ghost], cfg)
                Cm = self._cost_only(self._swap_q(own, qm), [ghost], cfg)
                gq_fd[k, d] = (Cp - Cm) / (2 * eps)
        rel = np.linalg.norm(gq - gq_fd) / (np.linalg.norm(gq) + 1e-8)
        assert rel < 1e-2, f"ghost grad_q relative error = {rel}"

    def test_fd_matches_grad_T(self) -> None:
        own = _line_traj(-5.0, 5.0, y=0.5, z=2.0)
        ghost = GhostObstacle(
            center=np.array([0.0, 0.0, 2.0]),
            clearance_horizontal=2.0,
            clearance_vertical=1.0,
            weight=1.0,
        )
        cfg = GhostObstacleConfig()
        _, _, gT = compute_ghost_cost_and_grad(own, [ghost], cfg)
        eps = 1e-5
        T0 = own.durations.copy()
        gT_fd = np.zeros_like(gT)
        for k in range(T0.size):
            Tp = T0.copy(); Tp[k] += eps
            Tm = T0.copy(); Tm[k] -= eps
            Cp = self._cost_only(self._swap_T(own, Tp), [ghost], cfg)
            Cm = self._cost_only(self._swap_T(own, Tm), [ghost], cfg)
            gT_fd[k] = (Cp - Cm) / (2 * eps)
        rel = np.linalg.norm(gT - gT_fd) / (np.linalg.norm(gT) + 1e-8)
        assert rel < 1e-2, f"ghost grad_T relative error = {rel}"


# ---------------------------------------------------------------------------
# Ellipsoid shape
# ---------------------------------------------------------------------------


class TestEllipsoidShape:
    def test_vertical_clearance_costs_more_than_horizontal(self) -> None:
        """Same Δposition in z hurts more than in y when cz < cx — matches
        the swarm penalty's downwash convention."""
        own_y = _line_traj(-5.0, 5.0, y=1.5, z=2.0)
        own_z = _line_traj(-5.0, 5.0, y=0.0, z=2.7)
        ghost = GhostObstacle(
            center=np.array([0.0, 0.0, 2.0]),
            clearance_horizontal=2.0,   # 1.5 / 2.0 = 0.75
            clearance_vertical=1.0,     # 0.7 / 1.0 = 0.70 — closer
            weight=1.0e3,
        )
        cy, _, _ = compute_ghost_cost_and_grad(
            own_y, [ghost], GhostObstacleConfig()
        )
        cz, _, _ = compute_ghost_cost_and_grad(
            own_z, [ghost], GhostObstacleConfig()
        )
        assert cz > cy


# ---------------------------------------------------------------------------
# gcopter integration
# ---------------------------------------------------------------------------


class TestGcopterIntegration:
    """A ghost obstacle planted on the straight-line initial path should
    cause `gcopter_optimize` to detour the interior waypoint sideways."""

    def _setup(self):
        start = np.array([-5.0, 0.0, 2.0])
        end = np.array([+5.0, 0.0, 2.0])
        # Nudge the interior waypoint slightly off-axis so the symmetric
        # ghost cost has a finite gradient direction from x_0 (a waypoint
        # AT the ghost centre would sit at a saddle). The optimiser still
        # has to do real work to escape; the ghost is much wider than the
        # 0.1 m initial offset.
        mid = (start + end) / 2.0 + np.array([0.0, 0.1, 0.0])
        waypoints = np.stack([start, mid, end])
        durations = np.array([1.5, 1.5], dtype=np.float64)
        bc_start = np.zeros((4, 3)); bc_start[0] = start
        bc_end = np.zeros((4, 3)); bc_end[0] = end
        polytopes = [
            _wide_box(start, mid),
            _wide_box(mid, end),
        ]
        return waypoints, durations, bc_start, bc_end, polytopes

    def test_optimiser_detours_around_ghost(self) -> None:
        waypoints, durations, bc_start, bc_end, polytopes = self._setup()
        cfg = GCopterConfig(s=3, v_max=3.0, n_quad=8, maxiter=40)
        # Baseline: no ghosts.
        traj_no_ghost = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polytopes,
            config=cfg,
        )
        mid_no_ghost = traj_no_ghost.waypoints[1]
        # With a ghost at the midpoint, the interior waypoint should move
        # off the y=0 line. The corridor box is fat enough that escape
        # is feasible.
        ghost = GhostObstacle(
            center=np.array([0.0, 0.0, 2.0]),
            clearance_horizontal=2.5,
            clearance_vertical=1.0,
            weight=1.0e4,
        )
        traj_ghost = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polytopes,
            config=cfg,
            ghost_obstacles=[ghost],
        )
        mid_ghost = traj_ghost.waypoints[1]
        # The interior waypoint should move noticeably away from y=0 (or z).
        delta_off_axis = float(
            np.linalg.norm(mid_ghost[1:] - mid_no_ghost[1:])
        )
        assert delta_off_axis > 0.3, (
            f"interior waypoint barely moved: |Δ_off_axis| = "
            f"{delta_off_axis:.3f} m (expected > 0.3 m)"
        )

    def test_default_behaviour_unchanged_without_ghost_kwarg(self) -> None:
        """Existing callers that don't pass ghost_obstacles see byte-for-
        byte identical results."""
        waypoints, durations, bc_start, bc_end, polytopes = self._setup()
        cfg = GCopterConfig(s=3, v_max=3.0, n_quad=8, maxiter=40)
        traj_default = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polytopes,
            config=cfg,
        )
        traj_none = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polytopes,
            config=cfg,
            ghost_obstacles=None,
        )
        np.testing.assert_allclose(
            traj_default.waypoints, traj_none.waypoints, atol=0.0
        )
        traj_empty = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polytopes,
            config=cfg,
            ghost_obstacles=[],
        )
        np.testing.assert_allclose(
            traj_default.waypoints, traj_empty.waypoints, atol=0.0
        )


# ---------------------------------------------------------------------------
# GhostManager.decay_by_factor — Avenue 4 ↔ Avenue 5 bridge support
# ---------------------------------------------------------------------------


class TestGhostManagerDecayByFactor:
    """`decay_by_factor` generalises `decay()` to arbitrary multipliers so
    `_install_post_mgr_trajectory` can apply orbit-duration cumulative
    decay in one shot."""

    @staticmethod
    def _seeded(weight: float = 1.0e3, threshold: float = 10.0) -> GhostManager:
        mgr = GhostManager(
            config=GhostManagerConfig(
                initial_weight=weight, weight_threshold=threshold
            )
        )
        mgr.seed_from_positions(
            [np.array([0.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0])],
            t_planted=0.0,
        )
        return mgr

    def test_identity_factor_preserves_weights(self) -> None:
        mgr = self._seeded()
        weights_before = [g.weight for g in mgr.active_ghosts()]
        mgr.decay_by_factor(1.0)
        weights_after = [g.weight for g in mgr.active_ghosts()]
        assert weights_before == weights_after
        assert len(mgr) == 2

    def test_partial_factor_scales_weights(self) -> None:
        mgr = self._seeded(weight=100.0, threshold=1.0)
        mgr.decay_by_factor(0.25)
        active = mgr.active_ghosts()
        assert len(active) == 2
        for g in active:
            assert g.weight == pytest.approx(25.0)

    def test_factor_below_threshold_prunes_all(self) -> None:
        mgr = self._seeded(weight=100.0, threshold=50.0)
        # 100 * 0.4 = 40 < threshold → both pruned.
        mgr.decay_by_factor(0.4)
        assert len(mgr) == 0
        assert mgr.n_pruned_total == 2

    def test_zero_factor_clears_all(self) -> None:
        mgr = self._seeded()
        mgr.decay_by_factor(0.0)
        assert len(mgr) == 0
        assert mgr.n_pruned_total == 2

    def test_negative_factor_clears_all(self) -> None:
        mgr = self._seeded()
        mgr.decay_by_factor(-0.5)
        assert len(mgr) == 0

    def test_equivalent_to_repeated_decay(self) -> None:
        """`decay_by_factor(d**n)` == n calls to `decay()` when no entries
        cross the threshold during the iterated path."""
        d = 0.6
        n = 4
        mgr_iter = GhostManager(
            config=GhostManagerConfig(
                initial_weight=1.0e3, decay_per_tick=d, weight_threshold=1.0
            )
        )
        mgr_bulk = GhostManager(
            config=GhostManagerConfig(
                initial_weight=1.0e3, decay_per_tick=d, weight_threshold=1.0
            )
        )
        mgr_iter.seed_from_positions(
            [np.array([1.0, 2.0, 3.0])], t_planted=0.0
        )
        mgr_bulk.seed_from_positions(
            [np.array([1.0, 2.0, 3.0])], t_planted=0.0
        )
        for _ in range(n):
            mgr_iter.decay()
        mgr_bulk.decay_by_factor(d ** n)
        assert mgr_iter.active_ghosts()[0].weight == pytest.approx(
            mgr_bulk.active_ghosts()[0].weight
        )
