"""Tests for src.validation.depth_noise_model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.validation.depth_noise_model import DepthNoiseConfig, apply, valid_fraction


class TestNoiseProfile:
    def test_zero_noise_passthrough(self) -> None:
        cfg = DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.0, dropout_pct=0.0)
        d = np.array([0.5, 1.0, 2.0, 5.0])
        out = apply(d, cfg, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out, d)

    def test_range_cap_invalidates(self) -> None:
        cfg = DepthNoiseConfig(max_range_m=3.0, noise_coeff=0.0, dropout_pct=0.0)
        out = apply(np.array([2.0, 5.0]), cfg, rng=np.random.default_rng(0))
        assert out[0] == 2.0
        assert math.isinf(out[1])

    def test_quadratic_noise_scales_with_range_squared(self) -> None:
        # at r=1, σ = 0.005; at r=4, σ = 0.005 * 16 = 0.08
        # Sample many times and compare empirical std
        rng = np.random.default_rng(0)
        cfg = DepthNoiseConfig(max_range_m=100.0, noise_coeff=0.005, dropout_pct=0.0)
        n = 5000
        d1 = apply(np.full(n, 1.0), cfg, rng=rng)
        d4 = apply(np.full(n, 4.0), cfg, rng=rng)
        std1 = float(np.std(d1))
        std4 = float(np.std(d4))
        # ratio should be ~16, with some sampling slack
        ratio = std4 / std1
        assert 12.0 < ratio < 20.0

    def test_dropout_pct_roughly_correct(self) -> None:
        rng = np.random.default_rng(0)
        cfg = DepthNoiseConfig(max_range_m=100.0, noise_coeff=0.0, dropout_pct=30.0)
        n = 5000
        out = apply(np.full(n, 1.0), cfg, rng=rng)
        dropped = int(np.sum(np.isinf(out)))
        # binomial(n=5000, p=0.3) → mean 1500, sigma ~33
        assert 1400 < dropped < 1600


class TestValidFraction:
    def test_all_valid_when_inside_range_no_dropout(self) -> None:
        cfg = DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.0, dropout_pct=0.0)
        d = np.array([0.5, 1.0, 2.0])
        out = apply(d, cfg, rng=np.random.default_rng(0))
        assert valid_fraction(out, cfg) == 1.0

    def test_drops_inf(self) -> None:
        cfg = DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.0, dropout_pct=0.0)
        arr = np.array([1.0, float("inf"), 2.0])
        assert valid_fraction(arr, cfg) == pytest.approx(2 / 3)
