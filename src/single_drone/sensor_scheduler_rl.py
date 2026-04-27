"""
Project Sanjay Mk2 - Sensor Scheduler RL Primitives
====================================================
Pure-Python helpers used by the RL training loop. Intentionally has NO
dependency on gymnasium or stable-baselines3 so it can be imported and
unit-tested without those packages installed.

  - encode_state(SensorState)            -> np.ndarray, fixed-shape state vector
  - decode_action(action_idx)            -> (rgb_fps, thermal_fps)
  - compute_reward(fused, action, prev)  -> float

Reward shape and class priorities follow docs/ARCHITECTURE.md:251-280.

@author: Archishman Paul
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import DroneMissionState
from src.single_drone.sensor_scheduler import SensorState


# ════════════════════════════════════════════════════════════════════
#  Action space (Discrete(30) = 6 RGB levels x 5 thermal levels)
# ════════════════════════════════════════════════════════════════════

RGB_FPS_LEVELS: List[int] = [0, 2, 5, 10, 15, 30]
THERMAL_FPS_LEVELS: List[int] = [0, 5, 10, 15, 30]
ACTION_SPACE_SIZE: int = len(RGB_FPS_LEVELS) * len(THERMAL_FPS_LEVELS)   # 30


def decode_action(action_idx: int) -> Tuple[int, int]:
    """Discrete index -> (rgb_fps, thermal_fps)."""
    if not 0 <= action_idx < ACTION_SPACE_SIZE:
        raise ValueError(f"action_idx must be in [0, {ACTION_SPACE_SIZE}), got {action_idx}")
    rgb_i = action_idx // len(THERMAL_FPS_LEVELS)
    th_i = action_idx % len(THERMAL_FPS_LEVELS)
    return RGB_FPS_LEVELS[rgb_i], THERMAL_FPS_LEVELS[th_i]


def encode_action(rgb_fps: int, thermal_fps: int) -> int:
    """Inverse of decode_action; rounds to nearest legal level."""
    rgb_i = min(range(len(RGB_FPS_LEVELS)), key=lambda i: abs(RGB_FPS_LEVELS[i] - rgb_fps))
    th_i = min(range(len(THERMAL_FPS_LEVELS)), key=lambda i: abs(THERMAL_FPS_LEVELS[i] - thermal_fps))
    return rgb_i * len(THERMAL_FPS_LEVELS) + th_i


# ════════════════════════════════════════════════════════════════════
#  State vector (17 dimensions, all in [0, 1] except as noted)
# ════════════════════════════════════════════════════════════════════
#  Layout:
#   [0]      ambient_lux normalized (lux / 100000, clipped)
#   [1..9]   DroneMissionState one-hot (9 categories)
#   [10]     threat_score
#   [11]     rgb_max_conf
#   [12]     thermal_max_conf
#   [13]     weapon_class_conf
#   [14]     missed_detection_streak / 10, clipped
#   [15]     last rgb_fps / 30
#   [16]     last thermal_fps / 30
# ════════════════════════════════════════════════════════════════════

STATE_VECTOR_SIZE: int = 17

_MISSION_ORDER: List[DroneMissionState] = [
    DroneMissionState.PATROL_HIGH,
    DroneMissionState.TRACK_HIGH,
    DroneMissionState.INSPECTION_PENDING,
    DroneMissionState.DESCEND_CONFIRM,
    DroneMissionState.FACADE_SCAN,
    DroneMissionState.TARGET_CONFIRM,
    DroneMissionState.REASCEND_REJOIN,
    DroneMissionState.CROWD_OVERWATCH,
    DroneMissionState.DEGRADED_SAFE,
]


def encode_state(
    state: SensorState,
    last_rgb_fps: int = 0,
    last_thermal_fps: int = 0,
) -> np.ndarray:
    """Pack a SensorState (+ last action) into a fixed-shape float32 vector."""
    vec = np.zeros(STATE_VECTOR_SIZE, dtype=np.float32)

    vec[0] = float(np.clip(state.ambient_lux / 100000.0, 0.0, 1.0))

    if state.mission_state in _MISSION_ORDER:
        vec[1 + _MISSION_ORDER.index(state.mission_state)] = 1.0

    vec[10] = float(np.clip(state.threat_score, 0.0, 1.0))
    vec[11] = float(np.clip(state.rgb_max_conf, 0.0, 1.0))
    vec[12] = float(np.clip(state.thermal_max_conf, 0.0, 1.0))
    vec[13] = float(np.clip(state.weapon_class_conf, 0.0, 1.0))
    vec[14] = float(np.clip(state.missed_detection_streak / 10.0, 0.0, 1.0))
    vec[15] = float(np.clip(last_rgb_fps / 30.0, 0.0, 1.0))
    vec[16] = float(np.clip(last_thermal_fps / 30.0, 0.0, 1.0))

    return vec


# ════════════════════════════════════════════════════════════════════
#  Reward function
# ════════════════════════════════════════════════════════════════════
#  R = detection_reward - alpha * compute_cost - beta * switch_penalty
#
#  detection_reward = sum( class_priority[obj.type] * obj.confidence )
#  compute_cost     = (rgb_fps + thermal_fps) / 60          (max FPS pair)
#  switch_penalty   = 1 if rgb_fps or thermal_fps changed since last tick
#
#  The reward intentionally does NOT penalise missed objects directly --
#  the absence of positive reward when objects are present is signal
#  enough, and hard rails (R1/R2) prevent the degenerate "everything off"
#  policy from being legal anyway.
# ════════════════════════════════════════════════════════════════════

CLASS_PRIORITY = {
    "person":           1.0,
    "weapon_person":    5.0,
    "vehicle":          0.5,
    "fire":             3.0,
    "explosive_device": 5.0,
    "crowd":            2.0,
}

DEFAULT_ALPHA: float = 0.6   # compute-cost weight (raised from 0.3 after a
                             # 300k run collapsed to always-on. Stronger penalty
                             # forces the policy to discriminate via state.)
DEFAULT_BETA: float = 0.05   # switch-penalty weight


def compute_reward(
    detected_objects: List,
    rgb_fps: int,
    thermal_fps: int,
    prev_rgb_fps: Optional[int],
    prev_thermal_fps: Optional[int],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """One-tick reward.  Designed for clarity, not micro-optimisation."""

    # 1. Detection reward (positive)
    detection_reward = 0.0
    for obj in detected_objects:
        priority = CLASS_PRIORITY.get(obj.object_type, 1.0)
        detection_reward += priority * float(getattr(obj, "confidence", 0.0))

    # 2. Compute cost (negative)
    compute_cost = (rgb_fps + thermal_fps) / 60.0

    # 3. Switch penalty (negative)
    if prev_rgb_fps is None and prev_thermal_fps is None:
        switch_penalty = 0.0
    else:
        changed = (rgb_fps != prev_rgb_fps) or (thermal_fps != prev_thermal_fps)
        switch_penalty = 1.0 if changed else 0.0

    return detection_reward - alpha * compute_cost - beta * switch_penalty
