"""
Project Sanjay Mk2 - SensorScheduler Fast-Mode Gym Environment
================================================================
Lightweight pseudo-environment for PPO training of the SensorScheduler
policy.  Designed for *truth-based upscaling*: every synthetic distribution
is explicitly marked TIER 1 SYNTHETIC and is meant to be replaced with
measured data over time without changing the env's interface.

Why this exists:
  - The scenario-mode env (sensor_scheduler_env.py) trains on the real
    ScenarioExecutor but only 2 of 17 state features actually vary --
    most importantly, ambient_lux is hard-coded.  Policies trained there
    collapse to degenerate optima.
  - Fast mode samples every state feature at reset() so the policy sees
    the variation it needs to learn (day vs night, threat density,
    mission phase, weather placeholder).
  - Trained policy validates against the real ScenarioExecutor in 5C.

Upscaling tiers:
  TIER 1 (this file)  -- synthetic distributions, physics-inspired
  TIER 2 (future)     -- measured distributions from deployment logs
                          (replace _sample_lux, _sample_weather, etc.)
  TIER 3 (future)     -- measured detection P-curves from real RGB/thermal
                          payloads (replace _detect_prob_*)

The state shape, action space, reward formula stay stable across tiers.

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from src.core.types.drone_types import DroneMissionState
from src.single_drone.sensor_scheduler import (
    HardRails,
    HeuristicPolicy,
    SensorAction,
    SensorMode,
    SensorState,
)
from src.single_drone.sensor_scheduler_rl import (
    ACTION_SPACE_SIZE,
    CLASS_PRIORITY,
    STATE_VECTOR_SIZE,
    compute_reward,
    decode_action,
    encode_state,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Synthetic objects placed in the world each episode
# ════════════════════════════════════════════════════════════════════


@dataclass
class _SimObject:
    """A ground-truth object the synthetic detector tries to find."""
    object_type: str
    thermal_signature: float       # 0-1 (cold-hot)
    size: float                    # metres (rough scale)


# Class spawn weights -- TIER 1 SYNTHETIC.
# Replace with measured spawn distributions from real-deployment logs.
_CLASS_SPAWN_WEIGHTS = {
    "person":           4.0,
    "vehicle":          3.0,
    "fire":             0.7,
    "crowd":            0.5,
    "weapon_person":    0.3,
    "explosive_device": 0.1,
}

# Per-class size + thermal signature priors -- TIER 1 SYNTHETIC.
_CLASS_PRIORS = {
    "person":           dict(size=0.6, thermal=0.85),
    "vehicle":          dict(size=2.0, thermal=0.55),
    "fire":             dict(size=1.2, thermal=1.00),
    "crowd":            dict(size=3.0, thermal=0.80),
    "weapon_person":    dict(size=0.6, thermal=0.85),
    "explosive_device": dict(size=0.4, thermal=0.50),
}

WEATHER_CATEGORIES = ["clear", "cloudy", "rain", "fog", "smoke"]

# Weather attenuation for detection -- TIER 1 SYNTHETIC.
# Replace with measured payload performance vs weather sensor logs.
_WEATHER_ATTENUATION = {
    "clear":  1.00,
    "cloudy": 0.92,
    "rain":   0.70,
    "fog":    0.50,
    "smoke":  0.40,
}


# ════════════════════════════════════════════════════════════════════
#  Fast Gym environment
# ════════════════════════════════════════════════════════════════════


class SensorSchedulerFastEnv(gym.Env):
    """Pseudo-physics Gym env for PPO training of the SensorScheduler.

    No ScenarioExecutor dependency.  Every state feature varies per episode.

    Args:
        episode_steps: number of scheduler ticks per episode (matches
            roughly what scenario_executor would produce in 60s @ 2 Hz).
        seed: RNG seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = 120, seed: Optional[int] = None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_VECTOR_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self._episode_steps = episode_steps
        self._rng = random.Random(seed)

        # Per-episode state, populated in reset()
        self._step_idx: int = 0
        self._lux: float = 50000.0
        self._weather: str = "clear"
        self._mission_state: DroneMissionState = DroneMissionState.PATROL_HIGH
        self._objects: List[_SimObject] = []
        self._missed_streak: int = 0
        self._last_rgb_fps: int = 0
        self._last_thermal_fps: int = 0
        self._last_rgb_conf: float = 0.0
        self._last_thermal_conf: float = 0.0
        self._last_weapon_conf: float = 0.0

    # ----------------------------------------------------------------
    #  Per-episode samplers (each is a Tier 1/2/3 swap point)
    # ----------------------------------------------------------------

    def _sample_lux(self) -> float:
        """Sample ambient lux for this episode.

        TIER 1 SYNTHETIC: log-uniform in [1, 100000] covers moonlit night
        through full noon. Roughly equal probability mass per decade.
        Replace with measured deployment-time-of-day lux logs in TIER 2.
        """
        return float(10 ** self._rng.uniform(0.0, 5.0))

    def _sample_weather(self) -> str:
        """Pick a weather category. TIER 1 SYNTHETIC.
        Weights skewed toward clear (most operations are good-weather).
        Replace with measured deployment weather frequency in TIER 2.
        """
        weights = [0.55, 0.20, 0.10, 0.10, 0.05]   # clear/cloudy/rain/fog/smoke
        return self._rng.choices(WEATHER_CATEGORIES, weights=weights, k=1)[0]

    def _sample_mission_state(self) -> DroneMissionState:
        """Pick the drone's mission phase at episode start. TIER 1 SYNTHETIC.

        Skewed toward PATROL_HIGH (most ticks are routine patrol) but with
        meaningful weight on inspection phases so the rails fire and the
        policy learns that those states require dual sensors.
        """
        choices = [
            (DroneMissionState.PATROL_HIGH,        0.55),
            (DroneMissionState.TRACK_HIGH,         0.10),
            (DroneMissionState.INSPECTION_PENDING, 0.08),
            (DroneMissionState.DESCEND_CONFIRM,    0.07),
            (DroneMissionState.FACADE_SCAN,        0.05),
            (DroneMissionState.TARGET_CONFIRM,     0.05),
            (DroneMissionState.CROWD_OVERWATCH,    0.05),
            (DroneMissionState.REASCEND_REJOIN,    0.03),
            (DroneMissionState.DEGRADED_SAFE,      0.02),
        ]
        states, weights = zip(*choices)
        return self._rng.choices(states, weights=weights, k=1)[0]

    def _sample_objects(self) -> List[_SimObject]:
        """Spawn 0-N ground-truth objects for this episode. TIER 1 SYNTHETIC.

        Episode object density is itself randomized so the policy sees
        both quiet and busy scenes.  Replace with measured per-scenario
        threat density when real deployment logs are available.
        """
        n = self._rng.choices([0, 1, 2, 3, 4], weights=[0.20, 0.30, 0.25, 0.15, 0.10], k=1)[0]
        names = list(_CLASS_SPAWN_WEIGHTS.keys())
        weights = list(_CLASS_SPAWN_WEIGHTS.values())
        objects: List[_SimObject] = []
        for _ in range(n):
            cls = self._rng.choices(names, weights=weights, k=1)[0]
            prior = _CLASS_PRIORS[cls]
            objects.append(_SimObject(
                object_type=cls,
                thermal_signature=max(0.0, min(1.0, prior["thermal"] + self._rng.gauss(0.0, 0.05))),
                size=max(0.1, prior["size"] + self._rng.gauss(0.0, 0.15)),
            ))
        return objects

    # ----------------------------------------------------------------
    #  Synthetic detection model (TIER 1)
    # ----------------------------------------------------------------

    def _fps_factor(self, fps: int) -> float:
        """More frames -> better detection, saturating. TIER 1 SYNTHETIC."""
        if fps <= 0:
            return 0.0
        return min(1.0, math.sqrt(fps / 30.0))   # 30 FPS = 1.0, 5 FPS = 0.41

    def _lux_factor_rgb(self, lux: float) -> float:
        """RGB falls off below 100 lux. TIER 1 SYNTHETIC.
        Replace with measured RGB sensor SNR-vs-lux curves in TIER 3.
        """
        return max(0.05, min(1.0, math.log10(max(lux, 1.0)) / 3.0))   # 1 lux=0.05, 1000 lux=1.0

    def _detect_prob_rgb(self, obj: _SimObject, fps: int) -> float:
        """P(detect | RGB sensor at fps, lux, weather, object). TIER 1 SYNTHETIC."""
        if fps <= 0:
            return 0.0
        weather_atten = _WEATHER_ATTENUATION[self._weather]
        size_factor = min(1.0, obj.size / 2.0)
        # Some classes are inherently harder for RGB (tiny / camouflaged):
        class_base = {
            "person":           0.85,
            "weapon_person":    0.55,   # weapon hidden -> harder
            "vehicle":          0.95,
            "fire":             0.90,
            "explosive_device": 0.40,
            "crowd":            0.95,
        }.get(obj.object_type, 0.7)
        return class_base * self._lux_factor_rgb(self._lux) * self._fps_factor(fps) * weather_atten * size_factor

    def _detect_prob_thermal(self, obj: _SimObject, fps: int) -> float:
        """P(detect | thermal sensor at fps, weather, object). TIER 1 SYNTHETIC.
        Thermal is mostly lux-independent but contrast (signature) matters.
        """
        if fps <= 0:
            return 0.0
        weather_atten = _WEATHER_ATTENUATION[self._weather]
        # Hot ambient (high lux ~ daylight ~ warm surfaces) reduces contrast
        ambient_warmth = self._lux_factor_rgb(self._lux)   # 0-1 proxy for daytime warmth
        contrast = obj.thermal_signature * (1.0 - 0.4 * ambient_warmth)
        contrast = max(0.05, min(1.0, contrast))
        class_base = {
            "person":           0.85,
            "weapon_person":    0.85,   # body still hot, regardless of weapon
            "vehicle":          0.80,
            "fire":             1.00,   # giveaway thermal target
            "explosive_device": 0.30,   # often near-ambient
            "crowd":            0.90,
        }.get(obj.object_type, 0.7)
        return class_base * contrast * self._fps_factor(fps) * weather_atten

    # ----------------------------------------------------------------
    #  Gym API
    # ----------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self._step_idx = 0
        self._lux = self._sample_lux()
        self._weather = self._sample_weather()
        self._mission_state = self._sample_mission_state()
        self._objects = self._sample_objects()
        self._missed_streak = 0
        self._last_rgb_fps = 0
        self._last_thermal_fps = 0
        self._last_rgb_conf = 0.0
        self._last_thermal_conf = 0.0
        self._last_weapon_conf = 0.0

        return self._build_obs(), {
            "lux": self._lux,
            "weather": self._weather,
            "mission_state": self._mission_state.name,
            "n_objects": len(self._objects),
        }

    def step(self, action_idx: int):
        rgb_fps_proposed, thermal_fps_proposed = decode_action(int(action_idx))

        # Apply rails -- crucially, reward is computed on POST-RAILS values
        # so PPO can't game the rails by proposing illegal actions.
        sensor_state = self._current_sensor_state()
        proposed_action = SensorAction(
            rgb_fps=rgb_fps_proposed,
            thermal_fps=thermal_fps_proposed,
            mode=SensorMode.DAY_PATROL,
        )
        final_action = HardRails.apply(sensor_state, proposed_action)
        rgb_fps = final_action.rgb_fps
        thermal_fps = final_action.thermal_fps

        # Synthetic detection: per-object Bernoulli with class-specific P
        detected = []
        for obj in self._objects:
            p_rgb = self._detect_prob_rgb(obj, rgb_fps)
            p_thermal = self._detect_prob_thermal(obj, thermal_fps)
            # Cross-modal fusion: P(detected) = 1 - P(missed_rgb) * P(missed_thermal)
            p_combined = 1.0 - (1.0 - p_rgb) * (1.0 - p_thermal)
            if self._rng.random() < p_combined:
                # Confidence biased high when either single sensor is strong
                conf = max(p_rgb, p_thermal)
                detected.append(_DetectedObj(
                    object_type=obj.object_type,
                    confidence=conf,
                ))

        reward = compute_reward(
            detected,
            rgb_fps=rgb_fps,
            thermal_fps=thermal_fps,
            prev_rgb_fps=self._last_rgb_fps if self._step_idx > 0 else None,
            prev_thermal_fps=self._last_thermal_fps if self._step_idx > 0 else None,
        )

        # Update derived state for next tick's observation
        self._last_rgb_fps = rgb_fps
        self._last_thermal_fps = thermal_fps
        self._last_rgb_conf = max((d.confidence for d in detected if rgb_fps > 0), default=0.0)
        self._last_thermal_conf = max((d.confidence for d in detected if thermal_fps > 0), default=0.0)
        self._last_weapon_conf = max(
            (d.confidence for d in detected if d.object_type == "weapon_person"),
            default=0.0,
        )
        if detected:
            self._missed_streak = 0
        else:
            self._missed_streak += 1
        # Mission state may transition: a high-conf weapon detection escalates to inspection
        if self._last_weapon_conf > 0.5 and self._mission_state == DroneMissionState.PATROL_HIGH:
            self._mission_state = DroneMissionState.INSPECTION_PENDING

        self._step_idx += 1
        terminated = self._step_idx >= self._episode_steps
        truncated = False
        info = {
            "rgb_fps": rgb_fps,
            "thermal_fps": thermal_fps,
            "rgb_fps_proposed": rgb_fps_proposed,
            "thermal_fps_proposed": thermal_fps_proposed,
            "rails_triggered": final_action.rails_triggered,
            "n_detections": len(detected),
            "lux": self._lux,
            "weather": self._weather,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    # ----------------------------------------------------------------
    #  Internals
    # ----------------------------------------------------------------

    def _current_sensor_state(self) -> SensorState:
        return SensorState(
            ambient_lux=self._lux,
            mission_state=self._mission_state,
            threat_score=min(1.0, self._last_weapon_conf + 0.5 * self._last_rgb_conf),
            rgb_max_conf=self._last_rgb_conf,
            thermal_max_conf=self._last_thermal_conf,
            weapon_class_conf=self._last_weapon_conf,
            missed_detection_streak=self._missed_streak,
        )

    def _build_obs(self) -> np.ndarray:
        return encode_state(
            self._current_sensor_state(),
            last_rgb_fps=self._last_rgb_fps,
            last_thermal_fps=self._last_thermal_fps,
        )


@dataclass
class _DetectedObj:
    object_type: str
    confidence: float
