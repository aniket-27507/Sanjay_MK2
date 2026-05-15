"""Visual-Inertial Odometry drift injection model.

Phase 1 Stage B.4 of the rigs plan (see docs/MINCO_PIVOT.md §5.4).

Models three kinds of error in a VIO position estimate:

    1. Random walk: σ_walk per √s, integrated independently per axis.
    2. Systematic bias: deterministic drift `bias_rate` (m/s) along an axis
       (axis chosen at construction).
    3. Occasional jumps: Bernoulli per tick with probability
       `jump_prob_per_sec * dt`, magnitude drawn uniformly from a sphere of
       radius `jump_magnitude`.

The drift accumulates into a 3-vector that the rig adds to the drone's
TRUE position to produce its self-reported "estimated" position. Inter-
agent corrections then close the loop.

Default parameters match MINCO_PIVOT.md §5.4 ("standard drift rate").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VIODriftConfig:
    sigma_walk: float = 0.02            # m per sqrt(s) random walk per axis
    bias_rate: float = 0.01             # m/s systematic drift
    bias_axis: tuple = (1.0, 0.0, 0.0)  # direction of systematic drift (unit-norm)
    jump_prob_per_sec: float = 0.005    # Bernoulli per second
    jump_magnitude: float = 0.3         # m, uniform on a sphere


class VIODrift:
    """Accumulates a 3-vector drift over time.

    Use `step(dt)` once per simulation tick to grow the drift, and `correct`
    to feed back a measurement (e.g. inter-agent observation residual). The
    `value` property reads the current accumulated drift.
    """

    def __init__(
        self,
        config: VIODriftConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        axis = np.asarray(config.bias_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise ValueError("bias_axis must be non-zero")
        self._bias_unit = axis / norm
        self._drift = np.zeros(3, dtype=np.float64)
        self._elapsed = 0.0

    @property
    def value(self) -> np.ndarray:
        return self._drift.copy()

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def step(self, dt: float) -> np.ndarray:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        # random walk
        if self.config.sigma_walk > 0.0:
            self._drift += self.rng.normal(
                0.0, self.config.sigma_walk * np.sqrt(dt), size=3
            )
        # systematic bias
        if self.config.bias_rate != 0.0:
            self._drift += self._bias_unit * (self.config.bias_rate * dt)
        # occasional jumps
        p_jump = self.config.jump_prob_per_sec * dt
        if p_jump > 0.0 and self.rng.random() < p_jump:
            # uniform on sphere
            direction = self.rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            self._drift += direction * self.config.jump_magnitude
        self._elapsed += dt
        return self._drift.copy()

    def correct(self, observed_residual: np.ndarray, gain: float = 0.5) -> None:
        """Pull the drift toward (drift - residual) by `gain`.

        `observed_residual` is the estimator's best guess of the current drift
        (e.g. derived from inter-agent depth observations). gain=1.0 zeros
        the drift fully; gain=0.5 is a typical Kalman-style update.
        """
        if not (0.0 <= gain <= 1.0):
            raise ValueError("gain must be in [0, 1]")
        residual = np.asarray(observed_residual, dtype=np.float64)
        self._drift -= gain * residual

    def reset(self) -> None:
        self._drift[:] = 0.0
        self._elapsed = 0.0
