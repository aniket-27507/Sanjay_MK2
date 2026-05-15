"""Unit tests for src.single_drone.planning.flatness.

Phase 0 Task 0.6 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.5, §4.2).

Quadrotor differential flatness map:
    (p, v, a, j) -> (thrust_magnitude, body_quaternion, body_rate)

Verifies physical correctness on hover, vertical thrust, free-fall, and
drag-loaded flight.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.flatness import (
    flat_state,
    evaluate_trajectory_dynamics,
)
from src.single_drone.planning.minco import Trajectory


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


class TestHover:
    def test_thrust_equals_weight_at_rest(self) -> None:
        # at rest: a = 0, v = 0, j = 0 → thrust = m * g
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=_zeros(3), j=_zeros(3),
            mass=1.5, gravity=9.81,
        )
        assert T == pytest.approx(1.5 * 9.81)

    def test_identity_orientation_at_hover(self) -> None:
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=_zeros(3), j=_zeros(3),
            mass=1.0, gravity=9.81, yaw=0.0,
        )
        # hover with yaw=0 → body-z aligns with world-z → identity quaternion
        # w >= 0 convention, body z = world z, body x = world x
        np.testing.assert_allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_body_rate_zero_at_rest(self) -> None:
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=_zeros(3), j=_zeros(3),
            mass=1.0, gravity=9.81,
        )
        np.testing.assert_allclose(w, _zeros(3), atol=1e-9)


class TestVerticalManeuver:
    def test_thrust_during_vertical_accel(self) -> None:
        a = np.array([0.0, 0.0, 5.0])  # accelerate up at 5 m/s²
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=a, j=_zeros(3),
            mass=1.0, gravity=9.81,
        )
        assert T == pytest.approx(9.81 + 5.0)

    def test_thrust_during_free_fall(self) -> None:
        # a = -g → thrust = 0 (free fall)
        a = np.array([0.0, 0.0, -9.81])
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=a, j=_zeros(3),
            mass=2.0, gravity=9.81,
        )
        assert T == pytest.approx(0.0, abs=1e-9)

    def test_pure_vertical_thrust_keeps_body_z_up(self) -> None:
        a = np.array([0.0, 0.0, 2.0])
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=a, j=_zeros(3),
            mass=1.0, gravity=9.81,
        )
        # body z-axis points along world z (i.e. identity-ish quaternion)
        np.testing.assert_allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-9)


class TestLateralAccel:
    def test_horizontal_accel_tilts_body(self) -> None:
        # ax = 9.81 m/s² → tilt angle θ where tan(θ) = ax / g = 1 → θ = 45°
        a = np.array([9.81, 0.0, 0.0])
        T, q, w = flat_state(
            p=_zeros(3), v=_zeros(3), a=a, j=_zeros(3),
            mass=1.0, gravity=9.81, yaw=0.0,
        )
        # thrust magnitude = m * sqrt(g² + ax²) = sqrt(2) * g
        assert T == pytest.approx(np.sqrt(2.0) * 9.81)
        # b_z = (a + g_vec) / ||a + g_vec|| = ([9.81, 0, 9.81]) / (sqrt(2)*9.81)
        # = (1/sqrt(2)) [1, 0, 1]
        # Body z extracted via quaternion: rotate world z by q
        # tilt magnitude — compute by rotating world ẑ via quaternion
        from src.single_drone.planning.flatness import rotate_vector_by_quat
        b_z = rotate_vector_by_quat(np.array([0.0, 0.0, 1.0]), q)
        np.testing.assert_allclose(
            b_z, np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), atol=1e-9
        )


class TestDrag:
    def test_drag_adds_to_thrust_in_direction_of_motion(self) -> None:
        # Hover (a = 0), but moving with v != 0 and drag_coeff k > 0.
        # Newton: m a = F_thrust - m g_vec + F_drag, F_drag = -D v.
        # At a = 0: F_thrust = m g_vec + D v
        v = np.array([2.0, 0.0, 0.0])  # moving along +x
        T_no_drag, _, _ = flat_state(
            p=_zeros(3), v=v, a=_zeros(3), j=_zeros(3),
            mass=1.0, gravity=9.81, drag_coeffs=(0.0, 0.0, 0.0),
        )
        T_with_drag, _, _ = flat_state(
            p=_zeros(3), v=v, a=_zeros(3), j=_zeros(3),
            mass=1.0, gravity=9.81, drag_coeffs=(0.5, 0.0, 0.0),
        )
        # with drag, thrust must work harder (its lateral component grows)
        assert T_with_drag > T_no_drag


class TestQuaternion:
    def test_quat_unit_norm(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.uniform(-3, 3, size=3)
            v = rng.uniform(-2, 2, size=3)
            j = rng.uniform(-5, 5, size=3)
            T, q, w = flat_state(
                p=_zeros(3), v=v, a=a, j=j,
                mass=1.0, gravity=9.81,
            )
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-9)


class TestTrajectoryDynamics:
    def test_trajectory_thrust_within_bounds(self) -> None:
        s = 3
        D = 3
        bc_start = np.zeros((s + 1, D))
        bc_start[0] = [0.0, 0.0, 0.0]
        bc_end = np.zeros((s + 1, D))
        bc_end[0] = [5.0, 0.0, 0.0]
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )
        durations = np.array([1.5, 1.5])  # gentle trajectory
        traj = Trajectory(waypoints, durations, bc_start, bc_end, s=s)

        times, thrusts, quats, rates = evaluate_trajectory_dynamics(
            traj, dt=0.05, mass=1.0, gravity=9.81,
        )
        # 3 second gentle trajectory → thrust should be within [0, 2*m*g]
        assert thrusts.min() >= 0.0
        assert thrusts.max() < 2.0 * 9.81
        # quaternions normalized
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)
