"""Tests for the LiDAR world-model metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.single_drone.world_model.metrics import (
    check_acceptance,
    compute_metrics,
)


F = 4
H = 6
S = 72


def _logits_for(targets: np.ndarray, conf: float = 6.0) -> np.ndarray:
    return np.where(targets > 0.5, conf, -conf).astype(np.float32)


def test_perfect_predictions_yield_f1_one():
    rng = np.random.default_rng(0)
    targets = (rng.uniform(0, 1, size=(2, F, H, S)) > 0.95).astype(np.uint8)
    logits = _logits_for(targets)
    m = compute_metrics(logits, targets)
    for f in range(F):
        # Skip horizons that happen to have no positives (F1 undefined)
        if targets[:, f].sum() == 0:
            continue
        assert m["f1_per_horizon"][f] == pytest.approx(1.0, abs=1e-6)
        assert m["iou_per_horizon"][f] == pytest.approx(1.0, abs=1e-6)


def test_all_zero_predictions_yield_zero_f1():
    rng = np.random.default_rng(1)
    targets = (rng.uniform(0, 1, size=(2, F, H, S)) > 0.9).astype(np.uint8)
    logits = np.full_like(targets, fill_value=-10, dtype=np.float32)
    m = compute_metrics(logits, targets)
    for f in range(F):
        assert m["f1_per_horizon"][f] == pytest.approx(0.0, abs=1e-6)


def test_iou_handcrafted():
    targets = np.zeros((1, 1, H, S), dtype=np.uint8)
    targets[0, 0, H // 2, 36] = 1
    targets[0, 0, H // 2, 37] = 1

    pred = np.zeros((1, 1, H, S), dtype=np.float32)
    pred[0, 0, H // 2, 36] = 1.0  # only sector 36 predicted
    pred[0, 0, H // 2, 38] = 1.0  # false positive

    logits = np.where(pred > 0.5, 6.0, -6.0).astype(np.float32)
    m = compute_metrics(logits, targets)
    # TP=1, FP=1, FN=1 → IoU = 1/3
    assert m["iou_per_horizon"][0] == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_per_sector_recall_split_for_handcrafted_truth():
    # Front truth at sector 36 (theta=0); rear truth at sector 0 (theta=-pi); side at 18.
    targets = np.zeros((1, 1, H, S), dtype=np.uint8)
    targets[0, 0, H // 2, 36] = 1   # front
    targets[0, 0, H // 2, 18] = 1   # side
    targets[0, 0, H // 2, 0] = 1    # rear

    # Only the front cell is predicted
    pred = np.zeros((1, 1, H, S), dtype=np.float32)
    pred[0, 0, H // 2, 36] = 1.0
    logits = np.where(pred > 0.5, 6.0, -6.0).astype(np.float32)
    m = compute_metrics(logits, targets, front_half_angle_deg=30.0, rear_half_angle_deg=30.0)
    assert m["front_recall_per_horizon"][0] == pytest.approx(1.0)
    assert m["side_recall_per_horizon"][0] == pytest.approx(0.0)
    assert m["rear_recall_per_horizon"][0] == pytest.approx(0.0)


def test_fn_in_tube_within_front_cone():
    # Truth with one positive in front, one outside the tube; predict nothing.
    targets = np.zeros((1, 1, H, S), dtype=np.uint8)
    targets[0, 0, H // 2, 36] = 1   # front (in tube)
    targets[0, 0, H // 2, 0] = 1    # rear (out of tube)

    logits = np.full(targets.shape, -10.0, dtype=np.float32)
    m = compute_metrics(logits, targets, tube_half_angle_deg=30.0)
    # FN-in-tube = 1 missed / 1 in-tube positive → 1.0
    assert m["fn_in_tube_per_horizon"][0] == pytest.approx(1.0)


def test_ece_for_uniform_predictions_with_balanced_truth():
    rng = np.random.default_rng(0)
    n = 4096
    # Predictions all = 0.5; truth balanced → ECE should be near 0.
    logits = np.zeros((n, 1, 1, 1), dtype=np.float32)
    targets = (rng.uniform(0, 1, size=(n, 1, 1, 1)) > 0.5).astype(np.uint8)
    m = compute_metrics(logits, targets, n_ece_bins=10)
    assert m["ece_per_horizon"][0] < 0.05


def test_check_acceptance_pass_and_fail():
    metrics = {
        "f1_per_horizon": [0.7, 0.6, 0.5, 0.4],
        "front_recall_per_horizon": [0.9, 0.8, 0.7, 0.6],
        "fn_in_tube_per_horizon": [0.03, 0.05, 0.07, 0.10],
        "iou_per_horizon": [0.5, 0.4, 0.3, 0.2],
        "ece_per_horizon": [0.02, 0.03, 0.04, 0.05],
        "side_recall_per_horizon": [0.6, 0.5, 0.4, 0.3],
        "rear_recall_per_horizon": [0.4, 0.3, 0.2, 0.1],
    }
    horizons = [0.5, 1.0, 1.5, 2.0]
    acceptance = {
        "f1_at_0.5s_min": 0.65,
        "front_recall_at_0.5s_min": 0.85,
        "fn_in_tube_at_0.5s_max": 0.05,
    }
    ok, failures = check_acceptance(metrics, horizons, acceptance)
    assert ok and not failures

    # Failure case
    failing = dict(acceptance)
    failing["f1_at_0.5s_min"] = 0.99
    ok2, failures2 = check_acceptance(metrics, horizons, failing)
    assert not ok2
    assert any("f1_at_0.5s_min" in f for f in failures2)
