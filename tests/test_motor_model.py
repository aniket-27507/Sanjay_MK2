"""Tests for src.validation.motor_model."""

from __future__ import annotations

import math

import pytest

from src.validation.motor_model import MotorWear, MotorWearConfig


class TestThrustScaling:
    def test_initial_efficiency(self) -> None:
        w = MotorWear(MotorWearConfig(initial_efficiency=0.95, degradation_rate_per_hour=0.02))
        assert w.thrust_scaling(0.0) == pytest.approx(0.95)

    def test_decay_after_one_hour(self) -> None:
        w = MotorWear(MotorWearConfig(initial_efficiency=1.0, degradation_rate_per_hour=0.05))
        assert w.thrust_scaling(1.0) == pytest.approx(0.95)

    def test_floored_at_min(self) -> None:
        w = MotorWear(MotorWearConfig(initial_efficiency=1.0, degradation_rate_per_hour=0.5, min_efficiency=0.6))
        assert w.thrust_scaling(100.0) == pytest.approx(0.6)

    def test_no_degradation(self) -> None:
        w = MotorWear(MotorWearConfig(initial_efficiency=1.0, degradation_rate_per_hour=0.0))
        assert w.thrust_scaling(1000.0) == pytest.approx(1.0)


class TestMissionAbort:
    def test_finite_when_degrading(self) -> None:
        w = MotorWear(MotorWearConfig(initial_efficiency=1.0, degradation_rate_per_hour=0.1, min_efficiency=0.6))
        assert w.mission_abort_hours() == pytest.approx(4.0)

    def test_infinite_when_no_degradation(self) -> None:
        w = MotorWear(MotorWearConfig(initial_efficiency=1.0, degradation_rate_per_hour=0.0))
        assert math.isinf(w.mission_abort_hours())


class TestValidation:
    def test_rejects_zero_initial(self) -> None:
        with pytest.raises(ValueError):
            MotorWear(MotorWearConfig(initial_efficiency=0.0))

    def test_rejects_min_above_initial(self) -> None:
        with pytest.raises(ValueError):
            MotorWear(MotorWearConfig(initial_efficiency=0.5, min_efficiency=0.7))

    def test_rejects_negative_hours(self) -> None:
        w = MotorWear(MotorWearConfig())
        with pytest.raises(ValueError):
            w.thrust_scaling(-1.0)
