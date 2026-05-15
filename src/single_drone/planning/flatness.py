"""Quadrotor differential flatness map.

Phase 0 Task 0.6 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.5, §4.2).

A quadrotor's full state can be recovered from a smooth position trajectory:

    p(t)            ->  position
    v(t) = p'(t)    ->  velocity
    a(t) = p''(t)   ->  acceleration
    j(t) = p'''(t)  ->  jerk

The Newton equation, with body thrust along the body z-axis and quadratic
drag along velocity, gives

    F_thrust = m * (a + g_vec) + D @ v

so that

    thrust_magnitude = ||F_thrust||
    body_z_axis      = F_thrust / thrust_magnitude

Yaw is an extra DOF (a quadrotor is differentially flat with output (p, ψ));
we expose it as a `yaw` argument. Body x is obtained by projecting the
world-frame heading onto the plane orthogonal to body z, body y closes the
triad.

Body rates come from time-differentiating the body-z direction:

    d/dt F_thrust = m * j + D @ a
    body_z_dot    = (I - b_z b_z^T) (F_thrust_dot) / thrust_magnitude
    omega_x       = - b_y . body_z_dot
    omega_y       =   b_x . body_z_dot
    omega_z       =   yaw_rate

With these, callers can analytically test
    - thrust <= f_max  (motor saturation)
    - thrust >= f_min  (need positive thrust)
    - body z tilt <= θ_max  (rollover limit)
    - ||omega|| <= ω_max    (gyro / control rate limit)
without invoking a physics simulator.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def flat_state(
    p: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    j: np.ndarray,
    mass: float = 1.0,
    gravity: float = 9.81,
    drag_coeffs: Sequence[float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    yaw_rate: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute (thrust, quaternion, body_rate) at one instant.

    Parameters
    ----------
    p, v, a, j : (3,) float arrays
        Position, velocity, acceleration, jerk at one instant.
    mass : float
        Vehicle mass in kg.
    gravity : float
        Magnitude of gravitational acceleration. Gravity points along -world-z.
    drag_coeffs : sequence of 3 floats
        Diagonal of the linear-drag matrix D. F_drag = -D v in the world frame.
        Set to all zeros to disable drag.
    yaw : float
        Yaw angle (rad).
    yaw_rate : float
        Yaw rate (rad/s). Passed straight through as omega_z.

    Returns
    -------
    thrust : float
        Magnitude of the commanded thrust (N).
    quaternion : (4,) float array
        Body orientation as [w, x, y, z] (Hamilton convention, w in [0, 1]).
    body_rate : (3,) float array
        [omega_x, omega_y, omega_z] in the body frame (rad/s).
    """
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    drag = np.asarray(drag_coeffs, dtype=np.float64)
    if drag.shape != (3,):
        raise ValueError("drag_coeffs must have length 3")

    g_vec = np.array([0.0, 0.0, gravity], dtype=np.float64)
    D = np.diag(drag)

    F_thrust = mass * a + mass * g_vec + D @ v
    thrust = float(np.linalg.norm(F_thrust))

    if thrust < 1e-9:
        # free-fall: orientation undefined; pick identity
        b_z = np.array([0.0, 0.0, 1.0])
    else:
        b_z = F_thrust / thrust

    # body x from desired yaw, projected orthogonal to b_z
    c_psi, s_psi = float(np.cos(yaw)), float(np.sin(yaw))
    x_yawed = np.array([c_psi, s_psi, 0.0])
    b_x = x_yawed - float(np.dot(x_yawed, b_z)) * b_z
    bx_norm = float(np.linalg.norm(b_x))
    if bx_norm < 1e-9:
        # gimbal-lock-like degeneracy (body z parallel to world x);
        # use world y as fallback
        b_x = np.array([0.0, 1.0, 0.0])
        b_x = b_x - float(np.dot(b_x, b_z)) * b_z
        b_x /= max(float(np.linalg.norm(b_x)), 1e-12)
    else:
        b_x = b_x / bx_norm
    b_y = np.cross(b_z, b_x)

    R = np.column_stack([b_x, b_y, b_z])
    quat = rotation_matrix_to_quat(R)

    # body rates from jerk
    if thrust < 1e-9:
        b_z_dot = np.zeros(3)
    else:
        F_thrust_dot = mass * j + D @ a
        b_z_dot = (F_thrust_dot - float(np.dot(b_z, F_thrust_dot)) * b_z) / thrust

    omega_x = -float(np.dot(b_y, b_z_dot))
    omega_y = float(np.dot(b_x, b_z_dot))
    omega_z = float(yaw_rate)
    body_rate = np.array([omega_x, omega_y, omega_z], dtype=np.float64)

    return thrust, quat, body_rate


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to [w, x, y, z] quaternion (Shepperd's method).

    Sign convention picks w >= 0 when possible.
    """
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q = [w, x, y, z] (Hamilton)."""
    w, x, y, z = q
    # v' = q * v * q^-1
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R @ np.asarray(v, dtype=np.float64)


def evaluate_trajectory_dynamics(
    trajectory,
    dt: float = 0.05,
    mass: float = 1.0,
    gravity: float = 9.81,
    drag_coeffs: Sequence[float] = (0.0, 0.0, 0.0),
    yaw_policy=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a Trajectory and return (times, thrust, quat, body_rate) arrays.

    `yaw_policy` is an optional callable yaw_policy(t) -> (yaw, yaw_rate). If
    None, yaw is held at 0.
    """
    times = np.arange(0.0, trajectory.total_time + 1e-12, dt)
    thrust = np.empty(times.shape, dtype=np.float64)
    quats = np.empty((times.shape[0], 4), dtype=np.float64)
    rates = np.empty((times.shape[0], 3), dtype=np.float64)
    for i, t in enumerate(times):
        p = trajectory.evaluate(t, 0)
        v = trajectory.evaluate(t, 1)
        a = trajectory.evaluate(t, 2)
        j = trajectory.evaluate(t, 3)
        if yaw_policy is None:
            yaw, yaw_rate = 0.0, 0.0
        else:
            yaw, yaw_rate = yaw_policy(t)
        T, q, w = flat_state(
            p=p, v=v, a=a, j=j,
            mass=mass, gravity=gravity, drag_coeffs=drag_coeffs,
            yaw=yaw, yaw_rate=yaw_rate,
        )
        thrust[i] = T
        quats[i] = q
        rates[i] = w
    return times, thrust, quats, rates


def is_dynamically_feasible(
    trajectory,
    thrust_range: Tuple[float, float] = (0.0, 30.0),
    tilt_max_rad: Optional[float] = None,
    body_rate_max: Optional[float] = None,
    **flat_kwargs,
) -> bool:
    """Return True if the trajectory respects all enabled dynamic limits.

    `tilt_max_rad` checks the angle between the body z-axis and world z-axis.
    `body_rate_max` checks the 2-norm of the body rate.
    """
    times, thrust, quats, rates = evaluate_trajectory_dynamics(
        trajectory, **flat_kwargs
    )
    if thrust.min() < thrust_range[0] or thrust.max() > thrust_range[1]:
        return False
    if tilt_max_rad is not None:
        # cos(tilt) = b_z . [0,0,1]; rotate world z by quat to get b_z
        cos_tilt = np.array(
            [rotate_vector_by_quat(np.array([0.0, 0.0, 1.0]), q)[2] for q in quats]
        )
        # numerical safety
        cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
        if np.arccos(cos_tilt).max() > tilt_max_rad:
            return False
    if body_rate_max is not None:
        if np.linalg.norm(rates, axis=1).max() > body_rate_max:
            return False
    return True
