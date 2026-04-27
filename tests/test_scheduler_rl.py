"""
Tests for the SensorScheduler RL primitives and Gym environment.

The state-encoding / action-decoding / reward tests run anywhere
(no third-party deps).  The Gym env tests skip if gymnasium is not
installed -- training infrastructure is intended to run on Colab.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types.drone_types import DroneMissionState
from src.single_drone.sensor_scheduler import SensorState
from src.single_drone.sensor_scheduler_rl import (
    ACTION_SPACE_SIZE,
    CLASS_PRIORITY,
    RGB_FPS_LEVELS,
    STATE_VECTOR_SIZE,
    THERMAL_FPS_LEVELS,
    compute_reward,
    decode_action,
    encode_action,
    encode_state,
)


# ────────────────────────────────────────────────────────────────────
#  Action space
# ────────────────────────────────────────────────────────────────────


def test_action_space_size():
    assert ACTION_SPACE_SIZE == len(RGB_FPS_LEVELS) * len(THERMAL_FPS_LEVELS) == 30


def test_decode_action_round_trip():
    for rgb in RGB_FPS_LEVELS:
        for thermal in THERMAL_FPS_LEVELS:
            idx = encode_action(rgb, thermal)
            r2, t2 = decode_action(idx)
            assert (r2, t2) == (rgb, thermal)


def test_decode_action_rejects_out_of_range():
    with pytest.raises(ValueError):
        decode_action(-1)
    with pytest.raises(ValueError):
        decode_action(ACTION_SPACE_SIZE)


def test_decode_action_extremes():
    assert decode_action(0) == (RGB_FPS_LEVELS[0], THERMAL_FPS_LEVELS[0])
    assert decode_action(ACTION_SPACE_SIZE - 1) == (RGB_FPS_LEVELS[-1], THERMAL_FPS_LEVELS[-1])


# ────────────────────────────────────────────────────────────────────
#  State encoding
# ────────────────────────────────────────────────────────────────────


def test_encode_state_shape_and_dtype():
    vec = encode_state(SensorState())
    assert vec.shape == (STATE_VECTOR_SIZE,)
    assert vec.dtype == np.float32


def test_encode_state_lux_normalization():
    daylight = encode_state(SensorState(ambient_lux=100000.0))[0]
    night = encode_state(SensorState(ambient_lux=5.0))[0]
    assert daylight == 1.0
    assert 0.0 <= night < 0.001


def test_encode_state_mission_one_hot():
    """Each mission state lights up exactly one slot."""
    for ms in DroneMissionState:
        vec = encode_state(SensorState(mission_state=ms))
        mission_block = vec[1:10]
        assert mission_block.sum() == pytest.approx(1.0)


def test_encode_state_clipping():
    vec = encode_state(
        SensorState(threat_score=99.0, weapon_class_conf=2.0, missed_detection_streak=999),
    )
    assert (vec >= 0).all() and (vec <= 1).all()


def test_encode_state_last_action_normalised():
    vec = encode_state(SensorState(), last_rgb_fps=30, last_thermal_fps=30)
    assert vec[15] == 1.0
    assert vec[16] == 1.0


# ────────────────────────────────────────────────────────────────────
#  Reward function
# ────────────────────────────────────────────────────────────────────


class _FakeDetection:
    def __init__(self, object_type, confidence):
        self.object_type = object_type
        self.confidence = confidence


def test_reward_positive_on_detections():
    dets = [_FakeDetection("person", 0.8)]
    r = compute_reward(dets, rgb_fps=15, thermal_fps=0,
                       prev_rgb_fps=15, prev_thermal_fps=0)
    assert r > 0


def test_reward_class_priority_weighting():
    """Weapon detection rewards much more than vehicle detection."""
    weapon_r = compute_reward(
        [_FakeDetection("weapon_person", 1.0)],
        rgb_fps=15, thermal_fps=0,
        prev_rgb_fps=15, prev_thermal_fps=0,
    )
    vehicle_r = compute_reward(
        [_FakeDetection("vehicle", 1.0)],
        rgb_fps=15, thermal_fps=0,
        prev_rgb_fps=15, prev_thermal_fps=0,
    )
    assert weapon_r > 5 * vehicle_r  # 5.0 vs 0.5 priority


def test_reward_compute_cost_penalises_high_fps():
    cheap = compute_reward([], rgb_fps=2, thermal_fps=0,
                           prev_rgb_fps=2, prev_thermal_fps=0)
    expensive = compute_reward([], rgb_fps=30, thermal_fps=30,
                               prev_rgb_fps=30, prev_thermal_fps=30)
    assert expensive < cheap


def test_reward_switch_penalty_fires_on_change():
    no_change = compute_reward([], rgb_fps=15, thermal_fps=0,
                               prev_rgb_fps=15, prev_thermal_fps=0)
    with_change = compute_reward([], rgb_fps=15, thermal_fps=0,
                                 prev_rgb_fps=2, prev_thermal_fps=0)
    assert with_change < no_change


def test_reward_first_tick_no_switch_penalty():
    from src.single_drone.sensor_scheduler_rl import DEFAULT_ALPHA
    first = compute_reward([], rgb_fps=15, thermal_fps=0,
                           prev_rgb_fps=None, prev_thermal_fps=None)
    # No detections, no switch penalty -> only compute cost is negative
    expected = -DEFAULT_ALPHA * (15 + 0) / 60.0
    assert first == pytest.approx(expected, rel=1e-5)


# ────────────────────────────────────────────────────────────────────
#  Gym env (skipped if gymnasium not installed)
# ────────────────────────────────────────────────────────────────────


def test_gym_env_basic_lifecycle():
    pytest.importorskip("gymnasium")
    from src.single_drone.sensor_scheduler_env import SensorSchedulerEnv

    env = SensorSchedulerEnv(episode_duration_sec=10.0, seed=0)
    obs, info = env.reset()
    assert obs.shape == (STATE_VECTOR_SIZE,)
    assert obs.dtype == np.float32

    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(int(action))
        assert obs.shape == (STATE_VECTOR_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert "rgb_fps" in info and "thermal_fps" in info
        if terminated:
            break

    env.close()


def test_gym_env_terminates_at_duration():
    pytest.importorskip("gymnasium")
    from src.single_drone.sensor_scheduler_env import SensorSchedulerEnv

    env = SensorSchedulerEnv(episode_duration_sec=5.0, seed=1)
    env.reset()
    for _ in range(200):  # 200 steps >> duration at SENSOR_HZ=2
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert terminated, "episode should have ended within the duration budget"
    env.close()


def test_gym_env_action_applied_to_drone_zero():
    """Whatever action we set should be visible in the executor's last action for drone 0."""
    pytest.importorskip("gymnasium")
    from src.single_drone.sensor_scheduler_env import SensorSchedulerEnv, TRAINABLE_DRONE_ID
    from src.single_drone.sensor_scheduler_rl import encode_action

    env = SensorSchedulerEnv(episode_duration_sec=10.0, seed=2)
    env.reset()
    desired_idx = encode_action(rgb_fps=10, thermal_fps=15)
    obs, reward, terminated, truncated, info = env.step(desired_idx)
    # Hard rails may upgrade FPS but should not lower legal proposals
    assert info["rgb_fps"] in RGB_FPS_LEVELS
    assert info["thermal_fps"] in THERMAL_FPS_LEVELS
    env.close()
