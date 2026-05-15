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
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve


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


def M_matrix_dT(s: int, T: float, deriv_max: int) -> np.ndarray:
    """Entry-wise derivative of M_matrix(s, T, deriv_max) with respect to T.

    M[j, i] = i! / (i-j)! · T^{i-j}  for i >= j
        ⇒  ∂M/∂T[j, i] = i! / (i-j-1)! · T^{i-j-1}  for i > j; else 0.

    Returns an (deriv_max+1, 2s+2) matrix.
    """
    deg = 2 * s + 1
    out = np.zeros((deriv_max + 1, deg + 1), dtype=np.float64)
    for j in range(deriv_max + 1):
        for i in range(j + 1, deg + 1):
            out[j, i] = (math.factorial(i) / math.factorial(i - j - 1)) * (
                T ** (i - j - 1)
            )
    return out


def Q_matrix_dT(s: int, T: float) -> np.ndarray:
    """Entry-wise derivative of Q_matrix(s, T) with respect to T.

    Q[i, j] = coef_ij · T^{p} / p  with p = i+j-2s-1 ≥ 1
        ⇒  ∂Q/∂T[i, j] = coef_ij · T^{p-1}.
    """
    deg = 2 * s + 1
    out = np.zeros((deg + 1, deg + 1), dtype=np.float64)
    if T < 0.0:
        return out
    for i in range(s + 1, deg + 1):
        coef_i = math.factorial(i) / math.factorial(i - s - 1)
        for j in range(s + 1, deg + 1):
            coef_j = math.factorial(j) / math.factorial(j - s - 1)
            power = i + j - 2 * s - 1
            out[i, j] = coef_i * coef_j * (T ** (power - 1))
    return out


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

        # caches populated by _solve_coefficients_and_cache for gradient methods
        self._N: int = M * (2 * self.s + 2)  # total coefficient count
        self._m_constraints: int = 2 * (self.s + 1) + (M - 1) * (self.s + 2)
        self._A_constraints: Optional[np.ndarray] = None  # (m, N)
        self._kkt_matrix: Optional[np.ndarray] = None  # (N+m, N+m)
        self._kkt_lu: Optional[tuple] = None  # scipy lu_factor result
        self._kkt_solution: Optional[np.ndarray] = None  # (N+m, D), stacked [c; λ]

        self.coeffs = self._solve_coefficients_and_cache()

    # ------------------------------------------------------------------
    # Coefficient solve + KKT caching (for gradient computations)
    # ------------------------------------------------------------------
    def _solve_coefficients_and_cache(self) -> np.ndarray:
        s, M, D = self.s, self.M, self.D
        deg_plus_1 = 2 * s + 2
        N = self._N
        n_constraints = self._m_constraints

        A_mat = np.zeros((n_constraints, N), dtype=np.float64)
        d_mat = np.zeros((n_constraints, D), dtype=np.float64)

        Q_mat = np.zeros((N, N), dtype=np.float64)
        for k in range(M):
            Qk = Q_matrix(s, float(self.durations[k]))
            row = k * deg_plus_1
            Q_mat[row : row + deg_plus_1, row : row + deg_plus_1] = Qk

        row = 0
        M_start = M_matrix(s, 0.0, s)
        A_mat[row : row + s + 1, 0:deg_plus_1] = M_start
        d_mat[row : row + s + 1, :] = self.bc_start
        row += s + 1

        M_end = M_matrix(s, float(self.durations[-1]), s)
        A_mat[row : row + s + 1, (M - 1) * deg_plus_1 : M * deg_plus_1] = M_end
        d_mat[row : row + s + 1, :] = self.bc_end
        row += s + 1

        for k in range(M - 1):
            Tk = float(self.durations[k])
            Ml = M_matrix(s, Tk, s)
            Mr = M_matrix(s, 0.0, s)
            col_l = k * deg_plus_1
            col_r = (k + 1) * deg_plus_1
            A_mat[row, col_l : col_l + deg_plus_1] = Ml[0]
            d_mat[row, :] = self.waypoints[k + 1]
            row += 1
            A_mat[row, col_r : col_r + deg_plus_1] = Mr[0]
            d_mat[row, :] = self.waypoints[k + 1]
            row += 1
            for j in range(1, s + 1):
                A_mat[row, col_l : col_l + deg_plus_1] = Ml[j]
                A_mat[row, col_r : col_r + deg_plus_1] = -Mr[j]
                d_mat[row, :] = 0.0
                row += 1

        assert row == n_constraints, f"row={row} != n_constraints={n_constraints}"

        N_kkt = N + n_constraints
        K = np.zeros((N_kkt, N_kkt), dtype=np.float64)
        K[:N, :N] = 2.0 * Q_mat
        K[:N, N:] = A_mat.T
        K[N:, :N] = A_mat
        rhs = np.zeros((N_kkt, D), dtype=np.float64)
        rhs[N:, :] = d_mat

        lu = lu_factor(K)
        sol = lu_solve(lu, rhs)

        # cache for gradients
        self._A_constraints = A_mat
        self._kkt_matrix = K
        self._kkt_lu = lu
        self._kkt_solution = sol

        c_flat = sol[:N, :]
        return c_flat.reshape(M, deg_plus_1, D)

    # ------------------------------------------------------------------
    # KKT perturbation builders (for energy_grad + evaluate_with_grad)
    # ------------------------------------------------------------------
    def _dKz_dT(self, k: int) -> np.ndarray:
        """Returns the (N+m, D) right-hand side `(∂K/∂T_k) · z`.

        Used by `dc_dT_k` via implicit-function differentiation
        K · ∂z/∂T_k = -(∂K/∂T_k) · z. Exploits sparsity rather than
        building the full ∂K/∂T_k explicitly.
        """
        s = self.s
        M = self.M
        deg_plus_1 = 2 * s + 2
        N = self._N
        m = self._m_constraints
        D = self.D
        z = self._kkt_solution
        c = z[:N, :]
        lam = z[N:, :]

        out = np.zeros((N + m, D), dtype=np.float64)

        # 1) top block: 2 · (∂Q/∂T_k) · c — only segment k's coeffs contribute
        dQ_seg = Q_matrix_dT(s, float(self.durations[k]))
        row = k * deg_plus_1
        out[row : row + deg_plus_1, :] += 2.0 * (dQ_seg @ c[row : row + deg_plus_1, :])

        # 2) top block: (∂A/∂T_k)^T · λ
        # 3) bottom block: (∂A/∂T_k) · c
        M_dT = M_matrix_dT(s, float(self.durations[k]), s)  # (s+1, deg_plus_1)
        if k == M - 1:
            row_block = s + 1  # constraint-row index of end BC
            col = (M - 1) * deg_plus_1
            # ∂A · c → bottom rows
            out[N + row_block : N + row_block + s + 1, :] += (
                M_dT @ c[col : col + deg_plus_1, :]
            )
            # (∂A)^T · λ → top rows
            out[col : col + deg_plus_1, :] += (
                M_dT.T @ lam[row_block : row_block + s + 1, :]
            )
        else:
            base = 2 * (s + 1) + k * (s + 2)
            col_l = k * deg_plus_1
            # deriv-0 row of M_dT → constraint row `base` (pos-pin-left)
            out[N + base, :] += M_dT[0] @ c[col_l : col_l + deg_plus_1, :]
            out[col_l : col_l + deg_plus_1, :] += np.outer(M_dT[0], lam[base, :])
            # deriv-j rows (j=1..s) → constraint rows base+1+j (continuity)
            for j in range(1, s + 1):
                cr = base + 1 + j
                out[N + cr, :] += M_dT[j] @ c[col_l : col_l + deg_plus_1, :]
                out[col_l : col_l + deg_plus_1, :] += np.outer(M_dT[j], lam[cr, :])

        return out

    def dc_dq_interior(self, k_knot: int) -> np.ndarray:
        """∂c/∂q_{k_knot, d}, returned as a flat (N,) vector independent of d.

        The same vector applies to every spatial dimension: ∂c[:, d]/∂q[k_knot, d]
        is this vector; cross-dim derivatives are zero.
        """
        s = self.s
        M = self.M
        N = self._N
        m = self._m_constraints
        if not (0 <= k_knot < M - 1):
            raise ValueError(f"k_knot must be in [0, {M - 1})")
        base = 2 * (s + 1) + k_knot * (s + 2)
        e = np.zeros(N + m, dtype=np.float64)
        e[N + base] = 1.0      # pos-pin-left
        e[N + base + 1] = 1.0  # pos-pin-right
        u = lu_solve(self._kkt_lu, e)
        return u[:N]

    def dc_dT_segment(self, k_seg: int) -> np.ndarray:
        """∂c/∂T_{k_seg} as an (N, D) matrix."""
        if not (0 <= k_seg < self.M):
            raise ValueError(f"k_seg must be in [0, {self.M})")
        rhs = -self._dKz_dT(k_seg)
        u = lu_solve(self._kkt_lu, rhs)
        return u[: self._N, :]

    def dc_dq_interior_all(self) -> list:
        """Cache + return [dc_dq_interior(k) for k in range(M-1)]."""
        if getattr(self, "_dc_dq_cache", None) is None:
            self._dc_dq_cache = [self.dc_dq_interior(k) for k in range(self.M - 1)]
        return self._dc_dq_cache

    def dc_dT_segment_all(self) -> list:
        """Cache + return [dc_dT_segment(k) for k in range(M)]."""
        if getattr(self, "_dc_dT_cache", None) is None:
            self._dc_dT_cache = [self.dc_dT_segment(k) for k in range(self.M)]
        return self._dc_dT_cache

    def _monomial_basis(self, tau: float, deriv_order: int) -> np.ndarray:
        """Returns (2s+2,) vector b such that c_seg^T @ b = p^(deriv)(tau)."""
        deg = 2 * self.s + 1
        b = np.zeros(deg + 1, dtype=np.float64)
        if deriv_order > deg:
            return b
        for i in range(deriv_order, deg + 1):
            fac = math.factorial(i) // math.factorial(i - deriv_order)
            b[i] = fac * (tau ** (i - deriv_order))
        return b

    def evaluate_segment_with_grad(
        self,
        k_seg: int,
        tau: float,
        deriv_order: int,
        dc_dq_list: Optional[list] = None,
        dc_dT_list: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate p^(deriv)(tau) at local time tau within segment k_seg.

        Returns
        -------
        value : (D,) ndarray
            Polynomial value (or derivative).
        grad_q : (M-1, D) ndarray
            grad_q[k_int, d] = ∂value[d]/∂q[k_int, d].
            (Off-diagonal in dimension is zero — value[d] depends only on q[:, d].)
        grad_T : (M, D) ndarray
            grad_T[k_T, d] = ∂value[d]/∂T_{k_T} at fixed tau.
            The caller adds the ∂tau/∂T_k_seg = s_frac chain term separately.
        p_deriv_next : (D,) ndarray
            Next-derivative value at the same point — used by the caller to
            evaluate the ∂/∂tau chain for T_k_seg.
        """
        s = self.s
        M = self.M
        D = self.D
        deg_plus_1 = 2 * s + 2
        b = self._monomial_basis(tau, deriv_order)
        b_next = self._monomial_basis(tau, deriv_order + 1)
        c_seg = self.coeffs[k_seg]  # (deg+1, D)
        value = c_seg.T @ b
        p_deriv_next = c_seg.T @ b_next

        if dc_dq_list is None:
            dc_dq_list = self.dc_dq_interior_all()
        if dc_dT_list is None:
            dc_dT_list = self.dc_dT_segment_all()

        seg_slice = slice(k_seg * deg_plus_1, (k_seg + 1) * deg_plus_1)

        grad_q = np.zeros((max(M - 1, 0), D), dtype=np.float64)
        for k_int in range(M - 1):
            slice_dot = float(b @ dc_dq_list[k_int][seg_slice])
            grad_q[k_int, :] = slice_dot  # diagonal in d

        grad_T = np.zeros((M, D), dtype=np.float64)
        for k_T in range(M):
            grad_T[k_T, :] = b @ dc_dT_list[k_T][seg_slice, :]

        return value, grad_q, grad_T, p_deriv_next

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
    # Energy (control-effort integral) + analytical gradient
    # ------------------------------------------------------------------
    def energy(self) -> float:
        total = 0.0
        for k in range(self.M):
            Qk = Q_matrix(self.s, float(self.durations[k]))
            for d in range(self.D):
                ck = self.coeffs[k, :, d]
                total += float(ck @ Qk @ ck)
        return total

    def energy_grad(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (∂E/∂q_interior, ∂E/∂T) for the energy integral.

        Shapes: (M-1, D) and (M,).

        Math
        ----
        Let E = c^T Q c with c, Q stacked across segments (Q block-diagonal).
        With M segments and waypoints q (interior + fixed end-points):
            ∂E/∂q  = 2 (Q c)^T (∂c/∂q)
            ∂E/∂T_k = 2 (Q c)^T (∂c/∂T_k)  +  c^T (∂Q/∂T_k) c
        where ∂c/∂q comes from K(T) z = b(q): only b depends on q.
        ∂c/∂T_k comes from K(T) z = b: K · ∂z/∂T_k = -(∂K/∂T_k) · z.
        K's LU factor is cached on the trajectory.
        """
        s, M, D = self.s, self.M, self.D
        N = self._N

        # Q c (segment-block evaluation)
        deg_plus_1 = 2 * s + 2
        c_flat = self.coeffs.reshape(N, D)
        Qc = np.zeros_like(c_flat)
        for k in range(M):
            Qk = Q_matrix(s, float(self.durations[k]))
            row = k * deg_plus_1
            Qc[row : row + deg_plus_1, :] = Qk @ c_flat[row : row + deg_plus_1, :]

        # ∂E/∂q_int
        grad_q = np.zeros((max(M - 1, 0), D), dtype=np.float64)
        for k in range(M - 1):
            dcdq = self.dc_dq_interior(k)  # (N,)
            for d in range(D):
                grad_q[k, d] = 2.0 * float(Qc[:, d] @ dcdq)

        # ∂E/∂T
        grad_T = np.zeros(M, dtype=np.float64)
        for kk in range(M):
            dcdT = self.dc_dT_segment(kk)  # (N, D)
            piece1 = 2.0 * float(np.sum(Qc * dcdT))
            dQ_seg = Q_matrix_dT(s, float(self.durations[kk]))
            row = kk * deg_plus_1
            c_seg = c_flat[row : row + deg_plus_1, :]
            piece2 = 0.0
            for d in range(D):
                piece2 += float(c_seg[:, d] @ dQ_seg @ c_seg[:, d])
            grad_T[kk] = piece1 + piece2
        return grad_q, grad_T

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
