"""MINCO trajectory representation.

Phase 0 Task 0.4 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.1, §2.2).

The MINCO (Minimum Control) parameterisation describes a piecewise-polynomial
trajectory by

    - M+1 waypoints q_0, q_1, ..., q_M     (positions in R^D)
    - M segment durations T_1, ..., T_M
    - 2(s+1) boundary conditions (position + derivatives up to order s
      at the start, same at the end)

For minimum-snap trajectories (s=3), each segment is a polynomial of degree
2s+1 = 7 in time. Given (q, T, BC), the polynomial coefficients that minimise
the integral of the squared (s+1)-th derivative are uniquely determined by
the KKT system

    [ 2 Q   A^T ] [ c ]   [ 0 ]
    [ A     0   ] [ λ ] = [ d ]

where
    - Q is block-diagonal across segments; block k is the Gram matrix of
      monomial derivatives that produces ∫_0^{T_k} (p_k^{(s+1)})^2 dt
      = c_k^T Q_k c_k.
    - A encodes (a) the 2(s+1) endpoint BCs, (b) position pinning at each
      interior knot, and (c) continuity of derivatives 1..s across each
      interior knot.

This module provides:

    M_matrix(s, T, deriv_max)    polynomial-basis-to-derivatives matrix
    Q_matrix(s, T)               control-effort cost matrix per segment
    Trajectory                   constructs, evaluates, computes energy

The optimiser (gcopter.py) treats Trajectory as a black box: it perturbs (q, T)
and rebuilds the Trajectory; the closed-form solve buys the cost gradient by
finite differences quickly enough for L-BFGS in Phase 0.

Reference: Wang & Gao, "Geometrically Constrained Trajectory Optimization
for Multicopters" (IEEE T-RO 2022). Clean-room Python port.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def M_matrix(s: int, T: float, deriv_max: int) -> np.ndarray:
    """Map polynomial coefficients [c_0, ..., c_{2s+1}] to derivatives at time T.

    Returns an (deriv_max+1, 2s+2) matrix M such that
        [p(T), p'(T), ..., p^{(deriv_max)}(T)]^T = M @ c.
    Concretely, M[j, i] = i! / (i-j)! * T^{i-j} for i >= j, else 0.
    """
    deg = 2 * s + 1
    out = np.zeros((deriv_max + 1, deg + 1), dtype=np.float64)
    for j in range(deriv_max + 1):
        for i in range(j, deg + 1):
            out[j, i] = (math.factorial(i) / math.factorial(i - j)) * (T ** (i - j))
    return out


def Q_matrix(s: int, T: float) -> np.ndarray:
    """Control-effort cost matrix: ∫_0^T (p^{(s+1)}(t))^2 dt = c^T Q c.

    For a polynomial p(t) = Σ c_i t^i of degree 2s+1, the (s+1)-th derivative is

        p^{(s+1)}(t) = Σ_{i=s+1}^{2s+1} i! / (i-s-1)! · c_i · t^{i-s-1}.

    Squaring and integrating from 0 to T yields the symmetric PSD matrix

        Q[i, j] = (i! / (i-s-1)!) · (j! / (j-s-1)!) / (i+j-2s-1) · T^{i+j-2s-1}

    for i, j in [s+1, 2s+1]; zero elsewhere.
    """
    deg = 2 * s + 1
    Q = np.zeros((deg + 1, deg + 1), dtype=np.float64)
    if T <= 0.0:
        return Q
    for i in range(s + 1, deg + 1):
        coef_i = math.factorial(i) / math.factorial(i - s - 1)
        for j in range(s + 1, deg + 1):
            coef_j = math.factorial(j) / math.factorial(j - s - 1)
            power = i + j - 2 * s - 1
            Q[i, j] = coef_i * coef_j * (T ** power) / power
    return Q


class Trajectory:
    """Piecewise-polynomial MINCO trajectory.

    Parameters
    ----------
    waypoints : (M+1, D) array
        Positions at the M+1 knots (start, interior knots, end).
    durations : (M,) array
        Strictly positive segment durations.
    bc_start, bc_end : (s+1, D) arrays
        Boundary conditions: row j is the j-th time derivative at the
        endpoint. Row 0 must equal waypoints[0] / waypoints[-1] respectively.
    s : int
        Control order. s=3 minimises snap (default).

    Attributes
    ----------
    coeffs : (M, 2s+2, D) ndarray
        Polynomial coefficients per segment per dimension.
    knot_times : (M+1,) ndarray
        Cumulative-sum knot times starting at 0.
    total_time : float
        Sum of durations.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        durations: np.ndarray,
        bc_start: np.ndarray,
        bc_end: np.ndarray,
        s: int = 3,
    ) -> None:
        waypoints = np.asarray(waypoints, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64).ravel()
        bc_start = np.asarray(bc_start, dtype=np.float64)
        bc_end = np.asarray(bc_end, dtype=np.float64)

        if waypoints.ndim != 2:
            raise ValueError("waypoints must be a 2D array of shape (M+1, D)")
        M = int(durations.size)
        if waypoints.shape[0] != M + 1:
            raise ValueError(
                f"waypoints must have M+1={M + 1} rows; got {waypoints.shape[0]}"
            )
        D = waypoints.shape[1]
        if bc_start.shape != (s + 1, D) or bc_end.shape != (s + 1, D):
            raise ValueError(
                f"bc_start and bc_end must have shape ({s + 1}, {D})"
            )
        if np.any(durations <= 0.0):
            raise ValueError("all durations must be strictly positive")
        if not np.allclose(bc_start[0], waypoints[0]):
            raise ValueError("bc_start[0] must equal waypoints[0]")
        if not np.allclose(bc_end[0], waypoints[-1]):
            raise ValueError("bc_end[0] must equal waypoints[-1]")

        self.s = int(s)
        self.M = M
        self.D = D
        self.waypoints = waypoints
        self.durations = durations
        self.bc_start = bc_start
        self.bc_end = bc_end
        self.knot_times = np.concatenate(([0.0], np.cumsum(durations)))
        self.total_time = float(self.knot_times[-1])

        self.coeffs = self._solve_coefficients()

    # ------------------------------------------------------------------
    # Coefficient solve (closed-form, per-call)
    # ------------------------------------------------------------------
    def _solve_coefficients(self) -> np.ndarray:
        s, M, D = self.s, self.M, self.D
        deg_plus_1 = 2 * s + 2  # number of coefficients per segment
        N = M * deg_plus_1

        # Constraint count:
        #   start BCs:       s+1
        #   end BCs:         s+1
        #   per interior knot (M-1 of them): pos-left + pos-right + s deriv continuity = s+2
        n_constraints = 2 * (s + 1) + (M - 1) * (s + 2)

        A_mat = np.zeros((n_constraints, N), dtype=np.float64)
        d_mat = np.zeros((n_constraints, D), dtype=np.float64)

        # Q block-diagonal
        Q_mat = np.zeros((N, N), dtype=np.float64)
        for k in range(M):
            Qk = Q_matrix(s, float(self.durations[k]))
            row = k * deg_plus_1
            Q_mat[row : row + deg_plus_1, row : row + deg_plus_1] = Qk

        row = 0

        # Start BCs
        M_start = M_matrix(s, 0.0, s)
        A_mat[row : row + s + 1, 0:deg_plus_1] = M_start
        d_mat[row : row + s + 1, :] = self.bc_start
        row += s + 1

        # End BCs
        M_end = M_matrix(s, float(self.durations[-1]), s)
        A_mat[row : row + s + 1, (M - 1) * deg_plus_1 : M * deg_plus_1] = M_end
        d_mat[row : row + s + 1, :] = self.bc_end
        row += s + 1

        # Interior knots
        for k in range(M - 1):
            Tk = float(self.durations[k])
            Ml = M_matrix(s, Tk, s)
            Mr = M_matrix(s, 0.0, s)
            col_l = k * deg_plus_1
            col_r = (k + 1) * deg_plus_1
            # pos pin from left segment
            A_mat[row, col_l : col_l + deg_plus_1] = Ml[0]
            d_mat[row, :] = self.waypoints[k + 1]
            row += 1
            # pos pin from right segment
            A_mat[row, col_r : col_r + deg_plus_1] = Mr[0]
            d_mat[row, :] = self.waypoints[k + 1]
            row += 1
            # derivative continuity j=1..s
            for j in range(1, s + 1):
                A_mat[row, col_l : col_l + deg_plus_1] = Ml[j]
                A_mat[row, col_r : col_r + deg_plus_1] = -Mr[j]
                d_mat[row, :] = 0.0
                row += 1

        assert row == n_constraints, f"row={row} != n_constraints={n_constraints}"

        # KKT system
        N_kkt = N + n_constraints
        K = np.zeros((N_kkt, N_kkt), dtype=np.float64)
        K[:N, :N] = 2.0 * Q_mat
        K[:N, N:] = A_mat.T
        K[N:, :N] = A_mat
        rhs = np.zeros((N_kkt, D), dtype=np.float64)
        rhs[N:, :] = d_mat

        sol = np.linalg.solve(K, rhs)
        c_flat = sol[:N, :]
        return c_flat.reshape(M, deg_plus_1, D)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, t: float, derivative_order: int = 0) -> np.ndarray:
        if derivative_order < 0:
            raise ValueError("derivative_order must be non-negative")
        t = float(t)
        # find segment and local time
        if t <= 0.0:
            k = 0
            tau = 0.0
        elif t >= self.total_time:
            k = self.M - 1
            tau = float(self.durations[-1])
        else:
            # knot_times[1:] is the upper boundary of each segment
            k = int(np.searchsorted(self.knot_times[1:], t, side="right"))
            tau = float(t - self.knot_times[k])

        deg = 2 * self.s + 1
        if derivative_order > deg:
            return np.zeros(self.D)

        c = self.coeffs[k]  # (deg+1, D)
        result = np.zeros(self.D, dtype=np.float64)
        for i in range(derivative_order, deg + 1):
            fac = math.factorial(i) // math.factorial(i - derivative_order)
            result += fac * (tau ** (i - derivative_order)) * c[i]
        return result

    # ------------------------------------------------------------------
    # Energy (control-effort integral)
    # ------------------------------------------------------------------
    def energy(self) -> float:
        total = 0.0
        for k in range(self.M):
            Qk = Q_matrix(self.s, float(self.durations[k]))
            for d in range(self.D):
                ck = self.coeffs[k, :, d]
                total += float(ck @ Qk @ ck)
        return total

    # ------------------------------------------------------------------
    # Decision-variable convenience
    # ------------------------------------------------------------------
    @property
    def interior_waypoints(self) -> np.ndarray:
        """Returns the M-1 interior waypoints (the free positional decision vars)."""
        if self.M <= 1:
            return np.zeros((0, self.D), dtype=np.float64)
        return self.waypoints[1:-1].copy()

    def sample(self, n: int = 100, derivative_order: int = 0) -> np.ndarray:
        """Convenience: sample `n` equally-spaced points along the trajectory."""
        ts = np.linspace(0.0, self.total_time, n)
        return np.stack([self.evaluate(t, derivative_order) for t in ts], axis=0)
