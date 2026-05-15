"""Unit tests for src.single_drone.planning.minco.

Phase 0 Task 0.4 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.1, §2.2).

The Trajectory class implements MINCO with control order `s` (default 3,
i.e. minimum-snap). For one segment with rest-to-rest boundary conditions of
duration 1, the closed-form minimum-snap polynomial is

    p(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7

which we use as the ground truth in test_single_segment_rest_to_rest_snap.

For multi-segment trajectories we verify:
    - position at each interior knot equals the prescribed waypoint
    - derivatives 1..s are continuous across knots (smoothness)
    - the trajectory hits both boundary conditions
    - energy (integral of (p^{(s+1)})^2) is non-negative
    - tightening the durations changes the energy in the expected direction
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.single_drone.planning.minco import Trajectory, M_matrix, Q_matrix


# ---------------------------------------------------------------------------
# Polynomial-basis matrices
# ---------------------------------------------------------------------------
class TestMMatrix:
    def test_M_at_zero(self) -> None:
        # M(0) maps c -> [c_0, 1!*c_1, 2!*c_2, ..., s!*c_s]
        s = 3
        M = M_matrix(s, 0.0, deriv_max=s)
        assert M.shape == (s + 1, 2 * s + 2)
        expected = np.zeros((s + 1, 2 * s + 2))
        for j in range(s + 1):
            expected[j, j] = math.factorial(j)
        np.testing.assert_allclose(M, expected)

    def test_M_at_unit_time(self) -> None:
        s = 1
        M = M_matrix(s, 1.0, deriv_max=1)  # s=1 -> polynomial degree 3, so 4 coefs
        # p(1) = c_0 + c_1 + c_2 + c_3
        # p'(1) = c_1 + 2 c_2 + 3 c_3
        expected = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )
        np.testing.assert_allclose(M, expected)


class TestQMatrix:
    def test_Q_at_zero_is_zero(self) -> None:
        Q = Q_matrix(s=3, T=0.0)
        np.testing.assert_allclose(Q, np.zeros((8, 8)))

    def test_Q_is_symmetric_and_psd(self) -> None:
        for T in (0.5, 1.0, 3.0):
            Q = Q_matrix(s=3, T=T)
            np.testing.assert_allclose(Q, Q.T)
            # PSD: all eigenvalues nonneg
            w = np.linalg.eigvalsh(Q + Q.T)  # 2Q; same null space
            assert (w >= -1e-9).all()

    def test_Q_known_value_unit_segment(self) -> None:
        # For s=3, T=1, Q[4,4] = (4!/0!)^2 / 1 = 576
        Q = Q_matrix(s=3, T=1.0)
        assert Q[4, 4] == pytest.approx(576.0)
        # Q[5,5] = (5!/1!)^2 / 3 = 14400 / 3 = 4800
        assert Q[5, 5] == pytest.approx(4800.0)


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def _zero_bc(s: int, D: int) -> np.ndarray:
    return np.zeros((s + 1, D), dtype=np.float64)


class TestSingleSegment:
    def test_rest_to_rest_minimum_snap_polynomial(self) -> None:
        # canonical closed-form: p(t) = 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7
        s = 3
        D = 1
        bc_start = _zero_bc(s, D)
        bc_end = _zero_bc(s, D)
        bc_end[0, 0] = 1.0  # end position is 1
        waypoints = np.array([[0.0], [1.0]])
        durations = np.array([1.0])
        traj = Trajectory(waypoints, durations, bc_start, bc_end, s=s)
        # coefficients of piece 0
        c = traj.coeffs[0, :, 0]
        expected = np.array([0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0])
        np.testing.assert_allclose(c, expected, atol=1e-8)

    def test_endpoints_match_boundary_conditions(self) -> None:
        s = 2
        D = 3
        bc_start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        bc_end = np.array([[5.0, 2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        waypoints = np.array([bc_start[0], bc_end[0]])
        durations = np.array([2.0])
        traj = Trajectory(waypoints, durations, bc_start, bc_end, s=s)
        np.testing.assert_allclose(traj.evaluate(0.0), bc_start[0], atol=1e-8)
        np.testing.assert_allclose(traj.evaluate(0.0, 1), bc_start[1], atol=1e-8)
        np.testing.assert_allclose(traj.evaluate(0.0, 2), bc_start[2], atol=1e-8)
        T = float(durations[0])
        np.testing.assert_allclose(traj.evaluate(T), bc_end[0], atol=1e-7)
        np.testing.assert_allclose(traj.evaluate(T, 1), bc_end[1], atol=1e-7)
        np.testing.assert_allclose(traj.evaluate(T, 2), bc_end[2], atol=1e-7)


class TestMultiSegment:
    @pytest.fixture
    def traj(self) -> Trajectory:
        s = 3
        D = 3
        bc_start = _zero_bc(s, D)
        bc_end = _zero_bc(s, D)
        bc_end[0] = [10.0, 0.0, 0.0]
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [7.0, -1.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        durations = np.array([1.0, 1.0, 1.0])
        return Trajectory(waypoints, durations, bc_start, bc_end, s=s)

    def test_position_at_knots_matches_waypoints(self, traj: Trajectory) -> None:
        knot_times = traj.knot_times
        for i, t in enumerate(knot_times):
            np.testing.assert_allclose(traj.evaluate(t), traj.waypoints[i], atol=1e-7)

    def test_derivatives_continuous_at_interior_knots(self, traj: Trajectory) -> None:
        # Evaluate analytically from each side of the knot using the underlying
        # piece coefficients — the KKT system enforces continuity to machine
        # precision, finite differences would mask that with truncation error.
        import math

        for i in range(1, traj.M):  # interior knots: 1..M-1
            left_piece = traj.coeffs[i - 1]  # (deg+1, D)
            right_piece = traj.coeffs[i]     # (deg+1, D)
            Tleft = float(traj.durations[i - 1])
            deg = 2 * traj.s + 1
            for order in range(1, traj.s + 1):
                # left value at tau = T_{i-1}
                left_val = np.zeros(traj.D)
                for idx in range(order, deg + 1):
                    fac = math.factorial(idx) // math.factorial(idx - order)
                    left_val += fac * (Tleft ** (idx - order)) * left_piece[idx]
                # right value at tau = 0  =>  only the idx=order term survives
                right_val = math.factorial(order) * right_piece[order]
                np.testing.assert_allclose(left_val, right_val, atol=1e-9)

    def test_energy_nonnegative(self, traj: Trajectory) -> None:
        assert traj.energy() >= 0.0

    def test_evaluate_outside_bounds_clamps(self, traj: Trajectory) -> None:
        # before t=0
        np.testing.assert_allclose(traj.evaluate(-1.0), traj.waypoints[0])
        # after total_time
        np.testing.assert_allclose(
            traj.evaluate(traj.total_time + 5.0), traj.waypoints[-1], atol=1e-7
        )

    def test_evaluate_high_derivative_returns_zero(self, traj: Trajectory) -> None:
        # polynomial degree is 2s+1 = 7; derivative 8 is identically zero
        v = traj.evaluate(0.5, 8)
        np.testing.assert_allclose(v, np.zeros(traj.D), atol=1e-12)


class TestValidation:
    def test_rejects_waypoint_count_mismatch(self) -> None:
        s = 3
        D = 1
        bc = _zero_bc(s, D)
        with pytest.raises(ValueError):
            Trajectory(
                waypoints=np.zeros((5, D)),
                durations=np.array([1.0, 1.0]),  # M=2 -> needs 3 waypoints, not 5
                bc_start=bc,
                bc_end=bc,
                s=s,
            )

    def test_rejects_nonpositive_duration(self) -> None:
        s = 3
        D = 1
        bc = _zero_bc(s, D)
        bc[0, 0] = 0.0
        bc_e = _zero_bc(s, D)
        bc_e[0, 0] = 1.0
        with pytest.raises(ValueError):
            Trajectory(
                waypoints=np.array([[0.0], [1.0]]),
                durations=np.array([0.0]),
                bc_start=bc,
                bc_end=bc_e,
                s=s,
            )

    def test_rejects_bc_position_mismatch_with_endpoints(self) -> None:
        s = 3
        D = 1
        bc = _zero_bc(s, D)
        bc_e = _zero_bc(s, D)
        bc_e[0, 0] = 1.0
        with pytest.raises(ValueError):
            Trajectory(
                waypoints=np.array([[0.0], [2.0]]),  # endpoint 2 disagrees with bc_e=1
                durations=np.array([1.0]),
                bc_start=bc,
                bc_end=bc_e,
                s=s,
            )


class TestEnergyMonotonicity:
    """Longer durations should reduce the integral of squared (s+1)-th derivative."""

    def test_longer_total_time_lower_energy(self) -> None:
        s = 3
        D = 1
        bc = _zero_bc(s, D)
        bc_e = _zero_bc(s, D)
        bc_e[0, 0] = 1.0
        waypoints = np.array([[0.0], [0.5], [1.0]])

        t_short = Trajectory(waypoints, np.array([0.5, 0.5]), bc, bc_e, s=s)
        t_long = Trajectory(waypoints, np.array([2.0, 2.0]), bc, bc_e, s=s)
        assert t_long.energy() < t_short.energy()
