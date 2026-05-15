"""Stereo depth-camera noise + range limits.

Phase 1 Stage B.6 of the rigs plan (see docs/MINCO_PIVOT.md §3.2, §5.7).

OAK-D-Lite-class stereo depth noise has two known features:

    1. Quadratic-in-range stdev:  σ(r) ≈ k · r² for stereo baselines on
       the order of 7 cm. We model σ = noise_coeff · r² with
       noise_coeff defaulting to 0.005 (≈ 5 mm at 1 m, 5 cm at 3 m, 50 cm
       at 10 m — matching published OAK-D bench results).
    2. Hard range cap: depth values beyond `max_range_m` (default 10 m for
       OAK-D Lite, dropped to 3 m in fog scenarios) become invalid /
       infinity.

Additional knob:
    - `dropout_pct`: Bernoulli per-pixel drop, mimicking occlusion-boundary
      mismatches that the stereo block discards. Dropped values become NaN.

Real fog/rain / sensor failure is composed by adjusting `max_range_m` and
scaling `noise_coeff`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthNoiseConfig:
    max_range_m: float = 10.0           # OAK-D Lite reliable depth (outdoor)
    noise_coeff: float = 0.005          # σ(r) = noise_coeff · r²
    dropout_pct: float = 5.0            # 0..100; Bernoulli per pixel
    invalid_sentinel: float = float("inf")  # value used for >max_range and dropouts


def apply(
    true_depth: np.ndarray,
    config: DepthNoiseConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply stereo noise + range cap + dropout to a true depth array.

    Parameters
    ----------
    true_depth : array of shape (H, W) (or 1D)
        Per-pixel true distance in metres. Must be non-negative.
    config : DepthNoiseConfig
    rng : numpy Generator (default: fresh)

    Returns
    -------
    array of same shape: noisy depth, with invalids set to `invalid_sentinel`.
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(true_depth, dtype=np.float64)
    out = arr.copy()
    # Gaussian noise with σ = noise_coeff · r²
    if config.noise_coeff > 0.0:
        sigma = config.noise_coeff * arr * arr
        out += rng.normal(0.0, np.maximum(sigma, 0.0))
    # range cap (operates on the NOISY value — what the sensor delivers)
    out = np.where(out > config.max_range_m, config.invalid_sentinel, out)
    # dropout
    if config.dropout_pct > 0.0:
        p = config.dropout_pct / 100.0
        mask = rng.random(out.shape) < p
        out = np.where(mask, config.invalid_sentinel, out)
    # clip negatives back to invalid (sensor can't return negative depth)
    out = np.where(out < 0.0, config.invalid_sentinel, out)
    return out


def valid_fraction(noisy_depth: np.ndarray, config: DepthNoiseConfig) -> float:
    """Fraction of pixels with valid (finite, in-range) depth."""
    arr = np.asarray(noisy_depth, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    valid = np.isfinite(arr) & (arr > 0.0) & (arr <= config.max_range_m)
    return float(valid.mean())
