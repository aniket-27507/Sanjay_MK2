"""Tests for src/swarm/cbf_safety_filter.py (Avenue 4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.swarm.cbf_safety_filter import (
    CBFConfig, CBFResult, _cbf_filter_one_drone, apply_cbf_filter,
)


class TestSingleDroneCBF:

    def test_no_neighbours_passthrough(self) -> None:
        """Zero neighbours → velocity unchanged, no intervention."""
        v_orig = np.array([1.0, 0.0, 0.0])
        v_new, intervened, mag, infeas = _cbf_filter_one_drone(
            x_self=np.array([0., 0., 0.]),
            v_self=v_orig,
            x_others=np.zeros((0, 3)),
            v_others=np.zeros((0, 3)),
            cfg=CBFConfig(clearance=1.0),
        )
        np.testing.assert_allclose(v_new, v_orig)
        assert intervened is False
        assert mag == 0.0
        assert infeas is False

    def test_far_neighbour_no_intervention(self) -> None:
        """Drone 5m apart, both stationary, clearance 1m → no intervention."""
        v_new, intervened, _, _ = _cbf_filter_one_drone(
            x_self=np.array([0., 0., 0.]),
            v_self=np.array([0., 0., 0.]),
            x_others=np.array([[5., 0., 0.]]),
            v_others=np.array([[0., 0., 0.]]),
            cfg=CBFConfig(clearance=1.0, alpha=2.0),
        )
        np.testing.assert_allclose(v_new, [0., 0., 0.])
        assert intervened is False

    def test_head_on_collision_filters_velocity(self) -> None:
        """Two drones approaching head-on: CBF should slow / reverse self."""
        # Self at (0,0,0) moving +x at 2 m/s
        # Other at (1.5,0,0) moving -x at 2 m/s
        # Clearance 1.0 → currently h = 1.5^2 - 1.0^2 = 1.25 (safe)
        # h_dot = 2 (rel_x) · (rel_v) = 2 * (-1.5) * (4) = -12  (closing fast)
        # cbf_lhs = -12 + 2 * 1.25 = -9.5 < 0 → VIOLATED
        v_orig = np.array([2., 0., 0.])
        v_new, intervened, mag, _ = _cbf_filter_one_drone(
            x_self=np.array([0., 0., 0.]),
            v_self=v_orig,
            x_others=np.array([[1.5, 0., 0.]]),
            v_others=np.array([[-2., 0., 0.]]),
            cfg=CBFConfig(clearance=1.0, alpha=2.0),
        )
        assert intervened is True
        assert mag > 0
        # Self should be slowed in +x (closer to 0 or going -x)
        assert v_new[0] < v_orig[0]

    def test_max_velocity_correction_clamps_huge_violations(self) -> None:
        """Catastrophic violation: correction should be clamped to max."""
        v_orig = np.array([10., 0., 0.])
        v_new, intervened, mag, infeas = _cbf_filter_one_drone(
            x_self=np.array([0., 0., 0.]),
            v_self=v_orig,
            # Neighbour very close, closing at 10 m/s
            x_others=np.array([[0.6, 0., 0.]]),
            v_others=np.array([[-10., 0., 0.]]),
            cfg=CBFConfig(
                clearance=1.0, alpha=2.0,
                max_velocity_correction=2.0,
            ),
        )
        assert intervened is True
        assert mag == pytest.approx(2.0, abs=1e-6)
        # The clamp should set infeasible=True
        assert infeas is True


class TestApplyFilterToTrajectory:

    def test_no_violations_returns_input(self) -> None:
        """Drones flying parallel, far apart → no filter activation."""
        T, N = 10, 3
        positions = np.zeros((T, N, 3))
        velocities = np.zeros((T, N, 3))
        # Three drones at y = 0, 5, 10; all moving +x at 1 m/s
        for t in range(T):
            for i in range(N):
                positions[t, i] = [float(t * 0.1), float(i * 5.0), 0.0]
                velocities[t, i] = [1.0, 0.0, 0.0]
        out = apply_cbf_filter(positions, velocities, dt=0.1,
                                cfg=CBFConfig(clearance=1.0, alpha=2.0))
        assert out.total_interventions == 0
        np.testing.assert_allclose(out.filtered_velocities, velocities)

    def test_head_on_triggers_interventions(self) -> None:
        """Two drones head-on at clearance distance → many interventions."""
        T = 50
        positions = np.zeros((T, 2, 3))
        velocities = np.zeros((T, 2, 3))
        # Initially 3m apart on x-axis, closing at 1 m/s each
        for t in range(T):
            positions[t, 0] = [0.0 + 1.0 * t * 0.1, 0.0, 0.0]
            positions[t, 1] = [3.0 - 1.0 * t * 0.1, 0.0, 0.0]
            velocities[t, 0] = [1.0, 0.0, 0.0]
            velocities[t, 1] = [-1.0, 0.0, 0.0]
        out = apply_cbf_filter(positions, velocities, dt=0.1,
                                cfg=CBFConfig(clearance=1.0, alpha=2.0))
        assert out.total_interventions > 0
        assert out.max_correction_magnitude > 0

    def test_filtered_positions_diverge_from_raw_during_intervention(self) -> None:
        """When CBF intervenes, filtered positions should differ from raw."""
        T = 30
        positions = np.zeros((T, 2, 3))
        velocities = np.zeros((T, 2, 3))
        for t in range(T):
            positions[t, 0] = [0.0 + 0.5 * t * 0.1, 0.0, 0.0]
            positions[t, 1] = [2.0 - 0.5 * t * 0.1, 0.0, 0.0]
            velocities[t, 0] = [0.5, 0.0, 0.0]
            velocities[t, 1] = [-0.5, 0.0, 0.0]
        out = apply_cbf_filter(positions, velocities, dt=0.1,
                                cfg=CBFConfig(clearance=1.5, alpha=2.0))
        if out.total_interventions > 0:
            # Filtered positions should diverge in some frame
            diffs = np.linalg.norm(out.filtered_positions - positions, axis=-1)
            assert diffs.max() > 1e-3
