"""
Tests for the fast-mode SensorScheduler Gym environment.

Verifies:
  - State varies across resets (lux, weather, mission state, threats)
  - Reward is computed on POST-rails FPS values (no rails-gaming)
  - Detection probability responds to lux, sensor FPS, weather
  - Episode termination at episode_steps

Skips entirely if gymnasium isn't installed (training infra is Colab-side).
"""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

import numpy as np

from src.core.types.drone_types import DroneMissionState
from src.single_drone.sensor_scheduler_fast_env import SensorSchedulerFastEnv
from src.single_drone.sensor_scheduler_rl import (
    ACTION_SPACE_SIZE,
    STATE_VECTOR_SIZE,
    encode_action,
)


# ────────────────────────────────────────────────────────────────────
#  Lifecycle
# ────────────────────────────────────────────────────────────────────


def test_observation_action_spaces():
    env = SensorSchedulerFastEnv(seed=0)
    assert env.observation_space.shape == (STATE_VECTOR_SIZE,)
    assert env.action_space.n == ACTION_SPACE_SIZE


def test_reset_returns_valid_obs():
    env = SensorSchedulerFastEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (STATE_VECTOR_SIZE,)
    assert obs.dtype == np.float32
    assert (obs >= 0).all() and (obs <= 1).all()
    assert "lux" in info and "weather" in info


def test_step_returns_valid_tuple():
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (STATE_VECTOR_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "rgb_fps" in info and "thermal_fps" in info


def test_episode_terminates_at_step_limit():
    env = SensorSchedulerFastEnv(episode_steps=5, seed=0)
    env.reset()
    for i in range(5):
        _, _, terminated, _, _ = env.step(0)
    assert terminated, "episode should terminate after episode_steps"


# ────────────────────────────────────────────────────────────────────
#  State actually varies (the bug we fixed)
# ────────────────────────────────────────────────────────────────────


def test_lux_varies_across_resets():
    env = SensorSchedulerFastEnv(seed=42)
    luxes = []
    for _ in range(50):
        env.reset()
        luxes.append(env._lux)
    assert min(luxes) < 100, "should sample some night episodes (lux<100)"
    assert max(luxes) > 10000, "should sample some daytime episodes (lux>10k)"


def test_mission_state_varies_across_resets():
    env = SensorSchedulerFastEnv(seed=42)
    states = set()
    for _ in range(80):
        env.reset()
        states.add(env._mission_state)
    assert len(states) >= 3, "should sample multiple mission states"
    assert DroneMissionState.PATROL_HIGH in states


def test_weather_varies_across_resets():
    env = SensorSchedulerFastEnv(seed=42)
    weathers = set()
    for _ in range(80):
        env.reset()
        weathers.add(env._weather)
    assert len(weathers) >= 2, "should sample multiple weather categories"


def test_threat_density_varies():
    env = SensorSchedulerFastEnv(seed=42)
    counts = []
    for _ in range(80):
        env.reset()
        counts.append(len(env._objects))
    assert min(counts) == 0, "should have empty episodes"
    assert max(counts) >= 2, "should have busy episodes"


# ────────────────────────────────────────────────────────────────────
#  Rails are applied; reward uses POST-rails FPS (no gaming)
# ────────────────────────────────────────────────────────────────────


def test_rails_force_thermal_at_low_lux():
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    env._lux = 5.0   # force night
    # Action = (rgb=15, thermal=0). R1 should force thermal_fps>=5.
    action = encode_action(15, 0)
    _, _, _, _, info = env.step(action)
    assert info["thermal_fps"] >= 5, "R1 should have raised thermal_fps in low lux"
    assert "R1_night_thermal" in info["rails_triggered"]


def test_rails_force_rgb_minimum():
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    # Action = (rgb=0, thermal=15). R2 should force rgb_fps>=2.
    action = encode_action(0, 15)
    _, _, _, _, info = env.step(action)
    assert info["rgb_fps"] >= 2, "R2 should have raised rgb_fps to evidence minimum"
    assert "R2_rgb_evidence_min" in info["rails_triggered"]


def test_reward_uses_post_rails_fps():
    """If the proposed action has rgb=0 but rails force rgb=2, the reward's
    compute cost must reflect rgb=2, not rgb=0. Otherwise PPO games rails."""
    env = SensorSchedulerFastEnv(episode_steps=1, seed=0)
    env.reset()
    env._objects = []  # no detections so reward is purely compute cost + switch
    # Tick 1: propose (rgb=0, thermal=15). Rails force rgb to 2.
    action = encode_action(0, 15)
    _, reward_post_rails, _, _, info = env.step(action)
    # post-rails compute cost = 0.3 * (2 + 15) / 60 = 0.085; minus initial-tick has no switch penalty
    expected_cost = -0.3 * (info["rgb_fps"] + info["thermal_fps"]) / 60.0
    # Allow tiny rounding; the key check is that we used post-rails FPS not the proposed 0+15
    assert abs(reward_post_rails - expected_cost) < 1e-3, (
        f"reward {reward_post_rails} should match post-rails compute cost {expected_cost}, "
        "implying rails-gaming is closed"
    )


# ────────────────────────────────────────────────────────────────────
#  Detection model responds to conditions
# ────────────────────────────────────────────────────────────────────


def test_thermal_detects_in_dark_when_rgb_cant():
    """In near-zero lux, RGB P(detect) should crater while thermal stays usable."""
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    env._lux = 5.0
    env._weather = "clear"
    from src.single_drone.sensor_scheduler_fast_env import _SimObject
    obj = _SimObject(object_type="person", thermal_signature=0.85, size=0.6)
    p_rgb = env._detect_prob_rgb(obj, fps=15)
    p_thermal = env._detect_prob_thermal(obj, fps=15)
    assert p_thermal > p_rgb, f"thermal {p_thermal:.3f} should beat RGB {p_rgb:.3f} at night"


def test_rgb_better_in_daylight_for_camo_target():
    """A weapon_person (low RGB visibility) at high lux should still be detectable
    by both, but RGB P should be in a usable range, not zero."""
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    env._lux = 50000.0
    env._weather = "clear"
    from src.single_drone.sensor_scheduler_fast_env import _SimObject
    obj = _SimObject(object_type="weapon_person", thermal_signature=0.85, size=0.6)
    p_rgb = env._detect_prob_rgb(obj, fps=15)
    assert p_rgb > 0.1, "RGB should still see weapon_person in daylight at usable rate"


def test_no_capture_no_detections():
    """If both fps are 0 (rails will catch this in reality, but test the model
    in isolation): P(detect) should be 0."""
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    from src.single_drone.sensor_scheduler_fast_env import _SimObject
    obj = _SimObject(object_type="person", thermal_signature=0.85, size=0.6)
    assert env._detect_prob_rgb(obj, fps=0) == 0.0
    assert env._detect_prob_thermal(obj, fps=0) == 0.0


# ────────────────────────────────────────────────────────────────────
#  End-to-end PPO compatibility smoke test
# ────────────────────────────────────────────────────────────────────


def test_random_policy_full_episode_smoke():
    """Run an entire random-policy episode without errors."""
    env = SensorSchedulerFastEnv(seed=0)
    env.reset()
    total_reward = 0.0
    for _ in range(env._episode_steps):
        action = env.action_space.sample()
        _, r, terminated, truncated, _ = env.step(action)
        total_reward += r
        if terminated or truncated:
            break
    # Sanity: reward should be a finite float
    assert np.isfinite(total_reward)
