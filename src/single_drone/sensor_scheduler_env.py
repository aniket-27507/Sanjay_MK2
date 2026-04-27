"""
Project Sanjay Mk2 - SensorScheduler Gymnasium Environment
============================================================
Gym env wrapping ScenarioExecutor for PPO training of the sensor-scheduling
policy network.  Phase B of the scheduler architecture.

  - reset()  picks a random scenario, resets the executor.
  - step(a)  buffers action on drone 0's scheduler, advances one sensor
             tick, returns (obs, reward, terminated, truncated, info).
  - drones 1..N keep using the heuristic policy so the simulated scene
    is realistic.  Only drone 0 trains.

Imports gymnasium; will raise ImportError if not installed.

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from src.simulation.scenario_executor import ScenarioExecutor
from src.simulation.scenario_loader import ScenarioLoader
from src.single_drone.sensor_scheduler import (
    HeuristicPolicy,
    SensorAction,
    SensorMode,
    SensorScheduler,
    SensorState,
)
from src.single_drone.sensor_scheduler_rl import (
    ACTION_SPACE_SIZE,
    STATE_VECTOR_SIZE,
    compute_reward,
    decode_action,
    encode_state,
)

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_DIR = Path("config/scenarios")
TRAINABLE_DRONE_ID = 0   # drone 0 trains; other drones run heuristic


# ════════════════════════════════════════════════════════════════════
#  Buffered policy: returns whatever the env most recently set
# ════════════════════════════════════════════════════════════════════


class _BufferedPolicy:
    """Scheduler policy that defers to an externally-provided action.

    On the very first tick (before any action has been buffered) it
    falls back to HeuristicPolicy so the executor doesn't crash.
    """

    def __init__(self):
        self._next_action: Optional[SensorAction] = None
        self._last_state: Optional[SensorState] = None
        self._heuristic = HeuristicPolicy()

    def set_action(self, action: SensorAction) -> None:
        self._next_action = action

    def decide(self, state: SensorState) -> SensorAction:
        self._last_state = state
        if self._next_action is None:
            return self._heuristic.decide(state)
        return self._next_action

    @property
    def last_state(self) -> Optional[SensorState]:
        return self._last_state


# ════════════════════════════════════════════════════════════════════
#  Gym environment
# ════════════════════════════════════════════════════════════════════


class SensorSchedulerEnv(gym.Env):
    """Train a policy network on the real ScenarioExecutor."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_paths: Optional[List[Path]] = None,
        scenarios_dir: Path = DEFAULT_SCENARIO_DIR,
        episode_duration_sec: float = 60.0,
        gcs_port_base: int = 19600,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_VECTOR_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        if scenario_paths is None:
            scenario_paths = sorted(Path(scenarios_dir).glob("*.yaml"))
            if not scenario_paths:
                raise FileNotFoundError(f"no scenarios in {scenarios_dir}")
        self._scenario_paths = scenario_paths
        self._episode_duration = episode_duration_sec
        self._gcs_port_base = gcs_port_base
        self._gcs_port = gcs_port_base
        self._rng = random.Random(seed)

        self._executor: Optional[ScenarioExecutor] = None
        self._policy: Optional[_BufferedPolicy] = None
        self._last_rgb_fps: int = 0
        self._last_thermal_fps: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        scenario = ScenarioLoader.load(self._rng.choice(self._scenario_paths))
        scenario.duration_sec = self._episode_duration

        # Unique GCS port per episode to avoid bind collisions across resets
        self._gcs_port += 1
        ex = ScenarioExecutor(scenario, gcs_port=self._gcs_port)
        ex._gcs = None  # never start a real GCS during training

        # Inject buffered policy into drone 0's scheduler. Other drones
        # keep their default heuristic schedulers.
        self._policy = _BufferedPolicy()
        ex._sensor_schedulers[TRAINABLE_DRONE_ID] = SensorScheduler(policy=self._policy)

        self._executor = ex
        self._last_rgb_fps = 0
        self._last_thermal_fps = 0

        # Advance until drone 0's scheduler has been queried at least once
        # so we have a real state to return.
        if not self._advance_until_first_decision():
            return np.zeros(STATE_VECTOR_SIZE, dtype=np.float32), {}

        obs = encode_state(
            self._policy.last_state, self._last_rgb_fps, self._last_thermal_fps,
        )
        return obs, {"scenario_id": scenario.id}

    def step(self, action_idx: int):
        assert self._executor is not None and self._policy is not None, "call reset() first"

        rgb_fps, thermal_fps = decode_action(int(action_idx))
        sensor_action = SensorAction(
            rgb_fps=rgb_fps,
            thermal_fps=thermal_fps,
            mode=SensorMode.DAY_PATROL,  # mode is informational; rails will mask
        )
        self._policy.set_action(sensor_action)

        still_running = self._executor.step_one_tick()

        # Reward from drone 0's fused observation produced by this tick.
        fused = self._executor._last_fused_obs.get(TRAINABLE_DRONE_ID)
        detected = fused.detected_objects if fused is not None else []
        reward = compute_reward(
            detected,
            rgb_fps=rgb_fps,
            thermal_fps=thermal_fps,
            prev_rgb_fps=self._last_rgb_fps,
            prev_thermal_fps=self._last_thermal_fps,
        )

        self._last_rgb_fps = rgb_fps
        self._last_thermal_fps = thermal_fps

        terminated = not still_running
        truncated = False
        obs = encode_state(
            self._policy.last_state if self._policy.last_state else SensorState(),
            self._last_rgb_fps,
            self._last_thermal_fps,
        )
        info = {
            "rgb_fps": rgb_fps,
            "thermal_fps": thermal_fps,
            "n_detections": len(detected),
            "rails_triggered": (
                self._executor._last_scheduler_action[TRAINABLE_DRONE_ID].rails_triggered
                if self._executor._last_scheduler_action[TRAINABLE_DRONE_ID] else []
            ),
        }
        return obs, float(reward), terminated, truncated, info

    def _advance_until_first_decision(self) -> bool:
        """Run executor until drone 0's scheduler has been queried at least once."""
        if self._executor is None or self._policy is None:
            return False
        for _ in range(50):  # cap to avoid infinite loop on degenerate scenarios
            if self._policy.last_state is not None:
                return True
            if not self._executor.step_one_tick():
                return False
        return self._policy.last_state is not None

    def close(self):
        self._executor = None
        self._policy = None
