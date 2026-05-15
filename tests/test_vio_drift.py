"""Tests for src.validation.vio_drift_model."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.vio_drift_model import VIODrift, VIODriftConfig


class TestRandomWalk:
    def test_zero_dt_is_noop(self) -> None:
        rng = np.random.default_rng(0)
        v = VIODrift(VIODriftConfig(sigma_walk=1.0, bias_rate=0.0, jump_prob_per_sec=0.0), rng)
        v.step(0.0)
        np.testing.assert_allclose(v.value, np.zeros(3))

    def test_random_walk_grows_with_sqrt_t(self) -> None:
        # mean(||drift||) ~ sqrt(3 * sigma^2 * t)
        runs = []
        for seed in range(64):
            rng = np.random.default_rng(seed)
            v = VIODrift(VIODriftConfig(sigma_walk=0.1, bias_rate=0.0, jump_prob_per_sec=0.0), rng)
            for _ in range(100):
                v.step(0.1)  # total 10 s
            runs.append(float(np.linalg.norm(v.value)))
        mean = float(np.mean(runs))
        # expected drift magnitude ~ sigma * sqrt(D * t) = 0.1 * sqrt(30) ≈ 0.55
        # E[|chi_3|] ≈ 1.595 * sigma; for sigma=sigma_walk*sqrt(t)=0.1*sqrt(10)=0.316
        # → E[||drift||] ≈ 1.595 * 0.316 ≈ 0.50.  Allow 0.3..0.8 window.
        assert 0.3 < mean < 0.8


class TestSystematicBias:
    def test_pure_bias_accumulates_linearly(self) -> None:
        v = VIODrift(
            VIODriftConfig(sigma_walk=0.0, bias_rate=0.01, jump_prob_per_sec=0.0, bias_axis=(1, 0, 0)),
            rng=np.random.default_rng(0),
        )
        for _ in range(100):
            v.step(1.0)  # 100 s
        np.testing.assert_allclose(v.value, [1.0, 0.0, 0.0], atol=1e-9)

    def test_unit_norms_bias_axis(self) -> None:
        v = VIODrift(
            VIODriftConfig(sigma_walk=0.0, bias_rate=1.0, jump_prob_per_sec=0.0, bias_axis=(3, 0, 0)),
            rng=np.random.default_rng(0),
        )
        v.step(1.0)
        # axis (3,0,0) normalized to (1,0,0) → drift = (1,0,0) after 1 s
        np.testing.assert_allclose(v.value, [1.0, 0.0, 0.0], atol=1e-9)


class TestJumps:
    def test_jumps_change_drift(self) -> None:
        # very high jump probability ensures one happens in this window
        rng = np.random.default_rng(0)
        v = VIODrift(
            VIODriftConfig(sigma_walk=0.0, bias_rate=0.0, jump_prob_per_sec=1.0, jump_magnitude=0.3),
            rng,
        )
        for _ in range(50):
            v.step(0.5)
        # should have jumped multiple times
        assert float(np.linalg.norm(v.value)) > 0.1


class TestCorrection:
    def test_full_gain_zeros_drift(self) -> None:
        v = VIODrift(VIODriftConfig(sigma_walk=0.0, bias_rate=1.0, jump_prob_per_sec=0.0), rng=np.random.default_rng(0))
        v.step(2.0)
        residual = v.value.copy()  # this IS the drift
        v.correct(residual, gain=1.0)
        np.testing.assert_allclose(v.value, np.zeros(3), atol=1e-9)

    def test_half_gain_halves_drift(self) -> None:
        v = VIODrift(VIODriftConfig(sigma_walk=0.0, bias_rate=1.0, jump_prob_per_sec=0.0), rng=np.random.default_rng(0))
        v.step(2.0)
        d0 = v.value.copy()
        v.correct(d0, gain=0.5)
        np.testing.assert_allclose(v.value, 0.5 * d0, atol=1e-9)


class TestValidation:
    def test_rejects_zero_axis(self) -> None:
        with pytest.raises(ValueError):
            VIODrift(VIODriftConfig(bias_axis=(0, 0, 0)), rng=np.random.default_rng(0))

    def test_rejects_negative_dt(self) -> None:
        v = VIODrift(VIODriftConfig(), rng=np.random.default_rng(0))
        with pytest.raises(ValueError):
            v.step(-1.0)

    def test_rejects_bad_gain(self) -> None:
        v = VIODrift(VIODriftConfig(), rng=np.random.default_rng(0))
        with pytest.raises(ValueError):
            v.correct(np.zeros(3), gain=1.5)
