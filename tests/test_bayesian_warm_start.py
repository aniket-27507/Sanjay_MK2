"""Tests for src/single_drone/planning/bayesian_warm_start.py (Avenue 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.bayesian_warm_start import (
    BayesianWarmStartFilter,
    pack_state,
    unpack_state,
)


class TestPackUnpack:

    def test_pack_unpack_roundtrip(self) -> None:
        """pack_state followed by unpack_state should recover original."""
        wps = np.array([
            [0., 0., 1.],
            [3., 1., 1.5],
            [7., -1., 1.5],
            [10., 0., 1.],
        ])
        T = np.array([2.0, 3.0, 2.5])
        x = pack_state(wps, T)
        wps_back, T_back = unpack_state(x, wps)
        np.testing.assert_allclose(wps, wps_back)
        np.testing.assert_allclose(T, T_back)

    def test_no_interior_handled(self) -> None:
        """Two-waypoint case (no interior): pack returns just durations."""
        wps = np.array([[0., 0., 1.], [10., 0., 1.]])
        T = np.array([5.0])
        x = pack_state(wps, T)
        np.testing.assert_allclose(x, T)
        wps_back, T_back = unpack_state(x, wps)
        np.testing.assert_allclose(wps_back, wps)
        np.testing.assert_allclose(T_back, T)


class TestBayesianFilter:

    def test_first_observation_is_state(self) -> None:
        """First update should set state to the observation exactly."""
        f = BayesianWarmStartFilter()
        x0 = np.array([1.0, 2.0, 3.0])
        f.update(x0)
        np.testing.assert_allclose(f.x_hat, x0)
        assert f.n_updates == 1

    def test_repeated_same_observation_converges(self) -> None:
        """Feeding the same x repeatedly → x_hat stays at x, P shrinks."""
        f = BayesianWarmStartFilter(process_noise=1e-3, observation_noise=1e-1)
        x = np.array([5.0, -2.0, 1.5])
        for _ in range(20):
            f.update(x)
        np.testing.assert_allclose(f.x_hat, x, atol=1e-6)
        # P should be << initial_variance after many updates
        assert np.mean(f.P) < 0.05

    def test_confidence_grows_with_consistent_updates(self) -> None:
        f = BayesianWarmStartFilter()
        x = np.array([1.0, 1.0, 1.0])
        c0 = f.confidence()
        f.update(x); c1 = f.confidence()
        for _ in range(10):
            f.update(x)
        c_late = f.confidence()
        assert c0 == 0.0
        assert c1 == 0.0  # only 1 update; confidence requires >=2
        assert c_late > 0.5

    def test_innovation_reset_on_regime_change(self) -> None:
        """A huge jump should reset the filter to the new observation."""
        f = BayesianWarmStartFilter(innovation_reset_threshold=2.0)
        for _ in range(10):
            f.update(np.array([1.0, 1.0]))
        prev_resets = f.n_resets
        # Huge jump: 100x state magnitude
        f.update(np.array([100.0, 100.0]))
        assert f.n_resets > prev_resets
        # State should track the new observation
        np.testing.assert_allclose(f.x_hat, [100.0, 100.0])

    def test_small_innovation_no_reset(self) -> None:
        """Small noise around the same value should NOT trigger a reset."""
        f = BayesianWarmStartFilter(
            innovation_reset_threshold=2.0,
            observation_noise=1.0,
        )
        rng = np.random.default_rng(0)
        target = np.array([5.0, 5.0])
        for _ in range(20):
            f.update(target + rng.normal(scale=0.1, size=2))
        assert f.n_resets == 0
        np.testing.assert_allclose(f.x_hat, target, atol=0.3)

    def test_shape_change_triggers_reset(self) -> None:
        """If the state dimensionality changes (e.g. M changes), reset."""
        f = BayesianWarmStartFilter()
        f.update(np.zeros(5))
        f.update(np.zeros(5))
        prev_resets = f.n_resets
        f.update(np.zeros(7))
        assert f.n_resets > prev_resets
        assert f.x_hat.shape == (7,)

    def test_predict_returns_copy(self) -> None:
        f = BayesianWarmStartFilter()
        x = np.array([1.0, 2.0])
        f.update(x)
        p = f.predict()
        p[0] = 99.0  # mutating prediction should not affect state
        np.testing.assert_allclose(f.x_hat, [1.0, 2.0])

    def test_predict_before_update_is_none(self) -> None:
        f = BayesianWarmStartFilter()
        assert f.predict() is None
        assert f.confidence() == 0.0
