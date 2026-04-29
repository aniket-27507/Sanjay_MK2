"""Evaluation metrics for the LiDAR predictive-occupancy world model.

All metrics operate on logits / targets of shape ``[N, F, H, S]`` where N
is total samples, F is the number of future horizons, H is height bands,
and S is sectors.

Metrics produced:

- ``f1_per_horizon``       : per-horizon F1 at fixed threshold
- ``iou_per_horizon``      : per-horizon IoU at fixed threshold
- ``ece_per_horizon``      : 10-bin Expected Calibration Error
- ``front_recall_per_horizon``,
  ``side_recall_per_horizon``,
  ``rear_recall_per_horizon`` : per-horizon recall split by sector cone
- ``fn_in_tube_per_horizon``: false-negative rate restricted to a front
  cone whose half-angle defaults to 30 degrees (a safety-critical
  approximation of the planned trajectory tube; the velocity-aware tube
  will be a v2 follow-up).

The polar-grid sector convention: ``s = n_sectors // 2`` is "directly
forward" (theta = 0); ``s = 0`` and ``s = n_sectors - 1`` are at
theta = ±pi (rear).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sector_angles(n_sectors: int) -> np.ndarray:
    """Return the centre azimuth (radians, in (-pi, pi]) of each sector."""
    s = np.arange(n_sectors, dtype=np.float64)
    return (s / n_sectors) * 2.0 * math.pi - math.pi


def _cone_mask(n_sectors: int, half_angle_deg: float) -> np.ndarray:
    """Return a boolean ``(n_sectors,)`` mask for sectors within
    ``half_angle_deg`` of theta=0 (front)."""
    angles = _sector_angles(n_sectors)
    half = math.radians(float(half_angle_deg))
    # Front is centred on theta=0
    return np.abs(angles) <= half


def _per_horizon_f1_iou(
    pred: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """pred and target are bool ``[N, F, H, S]``; returns (f1, iou) of shape ``(F,)``."""
    eps = 1e-9
    tp = (pred & target).sum(axis=(0, 2, 3)).astype(np.float64)
    fp = (pred & ~target).sum(axis=(0, 2, 3)).astype(np.float64)
    fn = (~pred & target).sum(axis=(0, 2, 3)).astype(np.float64)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou


def _per_horizon_ece(
    probs: np.ndarray, target: np.ndarray, n_bins: int = 10
) -> np.ndarray:
    """Compute 10-bin ECE per horizon."""
    F = probs.shape[1]
    out = np.zeros(F, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for f in range(F):
        p = probs[:, f].reshape(-1)
        y = target[:, f].reshape(-1).astype(np.float64)
        n = p.size
        if n == 0:
            out[f] = 0.0
            continue
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                idx = (p >= lo) & (p <= hi)
            else:
                idx = (p >= lo) & (p < hi)
            if not np.any(idx):
                continue
            conf = p[idx].mean()
            acc = y[idx].mean()
            out[f] += (idx.sum() / n) * abs(conf - acc)
    return out


def _recall_in_mask(
    pred: np.ndarray, target: np.ndarray, sector_mask: np.ndarray
) -> np.ndarray:
    """Recall over the cells whose sector index is in ``sector_mask``."""
    eps = 1e-9
    masked_pred = pred[..., sector_mask]
    masked_target = target[..., sector_mask]
    tp = (masked_pred & masked_target).sum(axis=(0, 2, 3)).astype(np.float64)
    fn = (~masked_pred & masked_target).sum(axis=(0, 2, 3)).astype(np.float64)
    return tp / (tp + fn + eps)


def _fn_rate_in_mask(
    pred: np.ndarray, target: np.ndarray, sector_mask: np.ndarray
) -> np.ndarray:
    """False-negative rate (FN / total positives) inside the sector mask."""
    eps = 1e-9
    masked_pred = pred[..., sector_mask]
    masked_target = target[..., sector_mask]
    fn = (~masked_pred & masked_target).sum(axis=(0, 2, 3)).astype(np.float64)
    pos = masked_target.sum(axis=(0, 2, 3)).astype(np.float64)
    return fn / (pos + eps)


def compute_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = 0.5,
    n_ece_bins: int = 10,
    front_half_angle_deg: float = 30.0,
    rear_half_angle_deg: float = 30.0,
    tube_half_angle_deg: float = 30.0,
) -> Dict[str, List[float]]:
    """Run every per-horizon metric on a flat (N, F, H, S) tensor pair.

    Targets must be 0/1.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits/targets shape mismatch: {logits.shape} vs {targets.shape}"
        )
    if logits.ndim != 4:
        raise ValueError(f"Expected (N, F, H, S); got ndim={logits.ndim}")

    probs = _sigmoid(logits.astype(np.float64))
    pred = probs > float(threshold)
    target = targets.astype(bool)

    f1, iou = _per_horizon_f1_iou(pred, target)
    ece = _per_horizon_ece(probs, target, n_bins=n_ece_bins)

    n_sectors = logits.shape[-1]
    front_mask = _cone_mask(n_sectors, front_half_angle_deg)
    rear_angles = _sector_angles(n_sectors)
    rear_mask = (np.abs(rear_angles) >= math.pi - math.radians(rear_half_angle_deg))
    side_mask = ~(front_mask | rear_mask)
    tube_mask = _cone_mask(n_sectors, tube_half_angle_deg)

    front_recall = _recall_in_mask(pred, target, front_mask)
    side_recall = _recall_in_mask(pred, target, side_mask)
    rear_recall = _recall_in_mask(pred, target, rear_mask)
    fn_in_tube = _fn_rate_in_mask(pred, target, tube_mask)

    return {
        "f1_per_horizon": f1.tolist(),
        "iou_per_horizon": iou.tolist(),
        "ece_per_horizon": ece.tolist(),
        "front_recall_per_horizon": front_recall.tolist(),
        "side_recall_per_horizon": side_recall.tolist(),
        "rear_recall_per_horizon": rear_recall.tolist(),
        "fn_in_tube_per_horizon": fn_in_tube.tolist(),
    }


def check_acceptance(
    metrics: Dict[str, List[float]],
    horizons: Sequence[float],
    acceptance: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """Return ``(passed, failures)`` against the YAML acceptance block.

    Supported keys (each may be missing):
        f1_at_<H>s_min, iou_at_<H>s_min, front_recall_at_<H>s_min,
        ece_at_<H>s_max, fn_in_tube_at_<H>s_max
    Where <H> matches one of the floats in ``horizons`` (e.g. 0.5).
    """
    failures: List[str] = []
    horizons = list(horizons)
    for key, threshold in acceptance.items():
        # Parse "<metric>_at_<H>s_min" or "<metric>_at_<H>s_max"
        try:
            metric_name, _, rest = key.partition("_at_")
            assert rest.endswith("_min") or rest.endswith("_max")
            direction = rest[-3:]   # "min" or "max"
            horizon_s = float(rest[:-len("_") - 3].rstrip("s"))
        except Exception:
            failures.append(f"Could not parse acceptance key {key!r}")
            continue
        try:
            h_idx = horizons.index(horizon_s)
        except ValueError:
            failures.append(f"Horizon {horizon_s} not in {horizons!r} for key {key!r}")
            continue
        metric_key_map = {
            "f1": "f1_per_horizon",
            "iou": "iou_per_horizon",
            "ece": "ece_per_horizon",
            "front_recall": "front_recall_per_horizon",
            "side_recall": "side_recall_per_horizon",
            "rear_recall": "rear_recall_per_horizon",
            "fn_in_tube": "fn_in_tube_per_horizon",
        }
        metric_key = metric_key_map.get(metric_name)
        if metric_key is None or metric_key not in metrics:
            failures.append(f"Unknown metric for key {key!r}")
            continue
        observed = float(metrics[metric_key][h_idx])
        threshold = float(threshold)
        ok = (observed >= threshold) if direction == "min" else (observed <= threshold)
        if not ok:
            failures.append(
                f"{key}: observed={observed:.4f}, threshold={threshold:.4f}, dir={direction}"
            )
    return (not failures), failures
