"""Bayesian-filter warm-start estimator (Avenue 2, simplified).

Inspired by Yuan & Yu (2025), arXiv 2508.14299, "Sequential Convex Programming
with Filtering-Based Warm-Starting for Continuous-Time Multiagent Quadrotor
Trajectory Optimization." The paper's full contribution is a two-part
framework: (1) reformulating continuous-time constraints as auxiliary
dynamics so SCP can satisfy them between discretisation points, (2) a
Bayesian-filter warm-start over the SCP iteration space that yields
up to 100x speedup vs benchmark methods.

THIS MODULE IMPLEMENTS PART (2) ONLY — the Bayesian-filter warm start
applied to the existing L-BFGS-on-MINCO formulation. Part (1) would
require a deeper SCP rewrite of gcopter.py and is deferred.

WHAT THE FILTER DOES
====================
After each L-BFGS call, we have an observed optimum x_obs in the
(q_int.ravel(), T) flat-vector space. The filter maintains a running
mean estimate x_hat and a diagonal covariance P:

    Initialization (first observation):
        x_hat = x_obs
        P     = P0       (high initial uncertainty)

    On each new observation x_obs:
        innovation: y = x_obs - x_hat
        gain:       K = P / (P + R)
        update:     x_hat <- x_hat + K * y
        cov:        P     <- (I - K) * P + Q

Where R is observation noise (how much we trust each L-BFGS result)
and Q is process noise (how much the underlying optimum drifts).

WHEN THIS HELPS
===============
The filter contributes value in two situations:

1. **Stable scenarios (crossing N=3, patrol N=3 sparse).** The optimum
   varies smoothly over time. The filter's smoothed estimate is closer
   to the true future optimum than the raw previous solution. Used as
   warm start, L-BFGS converges in even fewer line-search steps.

2. **Noisy / multi-modal scenarios — DOES NOT HELP.** When the optimum
   switches between homotopy classes (e.g. converge collapse), the
   filter's smoothed average lies between the modes and is a WORSE
   initial guess than either mode. Detect this via innovation magnitude;
   when it spikes, reset filter or fall back to raw warm start.

LIMITATIONS RELATIVE TO YUAN & YU
=================================
- We use a diagonal P, not full covariance.
- No process model: the underlying assumption is the optimum is locally
  stationary plus drift. The paper uses the SCP iteration's structural
  prior (the optimum is a fixed point of a linearised problem).
- Single-trajectory filter; no joint multi-agent state estimation.
- No theoretical convergence guarantees: this is a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BayesianWarmStartFilter:
    """Kalman-filter-style estimator of the L-BFGS optimum.

    Parameters
    ----------
    process_noise : float
        Q in the Kalman recursion. Larger → faster forgetting of past
        observations (filter tracks moving optima better but is noisier).
    observation_noise : float
        R in the Kalman recursion. Larger → less weight on each new
        observation (filter is smoother but slower to adapt).
    initial_variance : float
        P_0 diagonal entries on first observation. Should be large
        relative to process_noise so the first observation effectively
        sets the state.
    innovation_reset_threshold : float
        If the innovation norm exceeds this multiple of the current
        state norm, treat as a regime change and reset the filter to
        the new observation. Disables the smoothing for sudden jumps.
    """
    process_noise: float = 1.0e-2
    observation_noise: float = 1.0e-1
    initial_variance: float = 1.0
    innovation_reset_threshold: float = 2.0

    x_hat: Optional[np.ndarray] = field(default=None, repr=False)
    P: Optional[np.ndarray] = field(default=None, repr=False)
    n_updates: int = 0
    n_resets: int = 0
    last_innovation_norm: float = float("nan")

    def update(self, x_obs: np.ndarray) -> None:
        """Incorporate a new L-BFGS-converged x_obs."""
        x_obs = np.asarray(x_obs, dtype=float).ravel()
        if self.x_hat is None:
            self.x_hat = x_obs.copy()
            self.P = np.full(x_obs.shape, self.initial_variance, dtype=float)
            self.n_updates = 1
            return

        if self.x_hat.shape != x_obs.shape:
            # Dimensionality change (e.g. M, the number of segments,
            # changed). Re-initialise.
            self.x_hat = x_obs.copy()
            self.P = np.full(x_obs.shape, self.initial_variance, dtype=float)
            self.n_resets += 1
            self.n_updates = 1
            return

        innovation = x_obs - self.x_hat
        innov_norm = float(np.linalg.norm(innovation))
        x_norm = max(float(np.linalg.norm(self.x_hat)), 1.0)
        self.last_innovation_norm = innov_norm

        # Regime-change detection: if the innovation is large compared
        # to the state, the previous estimate is stale (homotopy switch,
        # neighbour reconfiguration). Reset rather than smooth across it.
        if innov_norm > self.innovation_reset_threshold * x_norm:
            self.x_hat = x_obs.copy()
            self.P = np.full(x_obs.shape, self.initial_variance, dtype=float)
            self.n_resets += 1
            self.n_updates += 1
            return

        # Standard Kalman update with diagonal P, scalar-per-dim R/Q.
        R = self.observation_noise
        Q = self.process_noise
        K = self.P / (self.P + R)
        self.x_hat = self.x_hat + K * innovation
        self.P = (1.0 - K) * self.P + Q
        self.n_updates += 1

    def predict(self) -> Optional[np.ndarray]:
        """Return the current state estimate (best guess of next optimum)."""
        return None if self.x_hat is None else self.x_hat.copy()

    def confidence(self) -> float:
        """Scalar in [0, 1] measuring filter confidence.

        Higher means lower normalised covariance and at least 2 updates
        with no recent reset. Caller uses this to gate aggressive warm-
        start tuning (e.g. only when confidence > 0.5).
        """
        if self.x_hat is None or self.n_updates < 2:
            return 0.0
        # Map mean covariance through a saturating function. With
        # process_noise=1e-2, P converges to ~Q after a few updates;
        # confidence ~exp(-Q) = 0.99. With initial_variance=1, single-
        # update P ~= 1, confidence = exp(-1) = 0.37.
        mean_P = float(np.mean(self.P))
        return float(np.exp(-mean_P))


# ---------------------------------------------------------------------------
# Glue: pack/unpack between Trajectory waypoints+durations and flat vectors
# ---------------------------------------------------------------------------

def pack_state(waypoints: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Pack (interior waypoints, durations) into a single flat vector.

    Boundary waypoints (index 0 and -1) are NOT in the optimisation
    variable — they are bc_start/bc_end. Only interior waypoints are
    tracked by the filter.
    """
    if waypoints.shape[0] <= 2:
        return durations.copy().ravel()
    interior = waypoints[1:-1].ravel()
    return np.concatenate([interior, durations.ravel()])


def unpack_state(
    x: np.ndarray,
    template_waypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_state. Uses template_waypoints to know shape."""
    M_plus_1 = template_waypoints.shape[0]
    D = template_waypoints.shape[1]
    M = M_plus_1 - 1  # number of segments
    if M_plus_1 <= 2:
        return template_waypoints.copy(), x.copy().ravel()
    n_interior = (M_plus_1 - 2) * D
    interior = x[:n_interior].reshape(M_plus_1 - 2, D)
    durations = x[n_interior:].copy()
    new_waypoints = template_waypoints.copy()
    new_waypoints[1:-1] = interior
    return new_waypoints, durations
