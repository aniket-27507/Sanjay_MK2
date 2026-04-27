"""
Project Sanjay Mk2 - Sensor Scheduler (Phase A)
================================================
Decides which sensors run and at what FPS based on current drone context.

Architecture: three-layer (see docs/ARCHITECTURE.md:119-187).
  Layer 1 - HardRails: deterministic rules that cannot be overridden.
  Layer 2 - HeuristicPolicy: state-machine fallback (this file).
  Layer 3 - Learned policy (future: PPO-trained, Phase B).

Phase A = rails + heuristic only, no RL. Safe to deploy as the default
scheduler until a learned policy is trained and validated.

Inputs (SensorState):   ambient lux, mission state, threat / confidence
                        signals, operator overrides.
Outputs (SensorAction): RGB FPS, thermal FPS, operational mode,
                        list of triggered rails (audit trail).

@author: Archishman Paul
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from src.core.types.drone_types import DroneMissionState

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  FPS presets and thresholds (single source of truth)
# ════════════════════════════════════════════════════════════════════

FPS_OFF: int = 0
FPS_LOW: int = 2        # Evidence-chain minimum for RGB
FPS_MED: int = 5        # Dim-light patrol
FPS_HIGH: int = 15      # Default active patrol
FPS_BURST: int = 30     # Short-duration emergency

NIGHT_LUX_THRESHOLD: float = 10.0    # Below this, thermal becomes mandatory
WEAPON_CONF_TRIGGER: float = 0.3     # Cross-modal verify above this confidence
MISSED_DETECTION_BURST: int = 3      # Frames without detection -> emergency burst

INSPECTION_STATES = frozenset({
    DroneMissionState.INSPECTION_PENDING,
    DroneMissionState.DESCEND_CONFIRM,
    DroneMissionState.FACADE_SCAN,
    DroneMissionState.TARGET_CONFIRM,
})


# ════════════════════════════════════════════════════════════════════
#  Enums and dataclasses
# ════════════════════════════════════════════════════════════════════


class SensorMode(Enum):
    """Operational sensor mode (distinct from mission state)."""
    STARTUP = auto()
    DAY_PATROL = auto()
    NIGHT_PATROL = auto()
    INSPECT_DUAL = auto()
    EMERGENCY_BURST = auto()


@dataclass
class OperatorOverride:
    """Human-in-the-loop override. None fields mean 'no override'."""
    rgb_fps: Optional[int] = None
    thermal_fps: Optional[int] = None


@dataclass
class SensorState:
    """Scheduler input: everything needed to pick a sensor configuration."""
    ambient_lux: float = 10000.0                                    # noon sun ~100k, dusk ~100, moonlight ~0.1
    mission_state: DroneMissionState = DroneMissionState.PATROL_HIGH
    threat_score: float = 0.0                                       # 0-1
    rgb_max_conf: float = 0.0                                       # highest detection conf this tick
    thermal_max_conf: float = 0.0
    weapon_class_conf: float = 0.0                                  # specifically weapon_person conf
    missed_detection_streak: int = 0                                # ticks since last high-conf detection
    operator_override: OperatorOverride = field(default_factory=OperatorOverride)


@dataclass
class SensorAction:
    """Scheduler output: the final sensor configuration for this tick."""
    rgb_fps: int
    thermal_fps: int
    mode: SensorMode
    rails_triggered: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
#  Layer 1 - Hard Rails (inviolable)
# ════════════════════════════════════════════════════════════════════


class HardRails:
    """Deterministic safety rules. Applied after policy decides, before
    the action leaves the scheduler. These rules exist for auditability
    (law-enforcement evidence chain) and cannot be learned away."""

    @staticmethod
    def apply(state: SensorState, action: SensorAction) -> SensorAction:
        """Enforce every rail in order. Later rails may strengthen but
        never weaken earlier ones (rail composition is monotonic)."""
        rgb = action.rgb_fps
        thermal = action.thermal_fps
        triggered: List[str] = []

        # R1: thermal must be active in low-light
        if state.ambient_lux < NIGHT_LUX_THRESHOLD and thermal < FPS_MED:
            thermal = FPS_MED
            triggered.append("R1_night_thermal")

        # R2: RGB minimum for evidence chain (audit / court admissibility)
        if rgb < FPS_LOW:
            rgb = FPS_LOW
            triggered.append("R2_rgb_evidence_min")

        # R3: inspection phases require both sensors at active rate
        if state.mission_state in INSPECTION_STATES:
            if rgb < FPS_HIGH:
                rgb = FPS_HIGH
            if thermal < FPS_HIGH:
                thermal = FPS_HIGH
            triggered.append("R3_inspection_dual")

        # R4: possible armed threat -> cross-modal verify at active rate
        possible_weapon = state.weapon_class_conf > WEAPON_CONF_TRIGGER
        if possible_weapon:
            if rgb < FPS_HIGH:
                rgb = FPS_HIGH
            if thermal < FPS_HIGH:
                thermal = FPS_HIGH
            triggered.append("R4_weapon_cross_modal")

        # R5: operator override wins (final word). Honored even if it
        # lowers FPS below other rails -- human has legal authority.
        if state.operator_override.rgb_fps is not None:
            rgb = state.operator_override.rgb_fps
            triggered.append("R5_operator_rgb_override")
        if state.operator_override.thermal_fps is not None:
            thermal = state.operator_override.thermal_fps
            triggered.append("R5_operator_thermal_override")

        return SensorAction(
            rgb_fps=rgb,
            thermal_fps=thermal,
            mode=action.mode,
            rails_triggered=triggered,
        )


# ════════════════════════════════════════════════════════════════════
#  Layer 2 - Heuristic Policy (state-machine fallback, no learning)
# ════════════════════════════════════════════════════════════════════


class HeuristicPolicy:
    """Deterministic state-machine from docs/ARCHITECTURE.md:162-187.
    Used as the default until a PPO-trained policy network is validated."""

    @staticmethod
    def decide(state: SensorState) -> SensorAction:
        """Pick a mode + FPS pair from the scheduling state machine."""
        # Priority order matches the state-machine diagram:
        # EMERGENCY_BURST > INSPECT_DUAL > NIGHT_PATROL > DAY_PATROL

        if state.missed_detection_streak >= MISSED_DETECTION_BURST:
            return SensorAction(
                rgb_fps=FPS_BURST,
                thermal_fps=FPS_BURST,
                mode=SensorMode.EMERGENCY_BURST,
            )

        if state.mission_state in INSPECTION_STATES:
            return SensorAction(
                rgb_fps=FPS_HIGH,
                thermal_fps=FPS_HIGH,
                mode=SensorMode.INSPECT_DUAL,
            )

        if state.ambient_lux < NIGHT_LUX_THRESHOLD:
            return SensorAction(
                rgb_fps=FPS_MED,
                thermal_fps=FPS_HIGH,
                mode=SensorMode.NIGHT_PATROL,
            )

        return SensorAction(
            rgb_fps=FPS_HIGH,
            thermal_fps=FPS_OFF,
            mode=SensorMode.DAY_PATROL,
        )


# ════════════════════════════════════════════════════════════════════
#  Public scheduler (composes policy + rails)
# ════════════════════════════════════════════════════════════════════


class SensorScheduler:
    """Phase-A scheduler: heuristic policy gated by hard rails.

    Usage::

        scheduler = SensorScheduler()
        action = scheduler.tick(sensor_state)
        rgb_camera.set_fps(action.rgb_fps)
        thermal_camera.set_fps(action.thermal_fps)
        audit_log.record(action.mode, action.rails_triggered)
    """

    def __init__(self, policy: Optional[HeuristicPolicy] = None):
        self._policy = policy or HeuristicPolicy()
        self._last_action: Optional[SensorAction] = None

    def tick(self, state: SensorState) -> SensorAction:
        """Single scheduling step. Call at fixed cadence (e.g. 2 Hz)."""
        proposed = self._policy.decide(state)
        final = HardRails.apply(state, proposed)
        self._last_action = final

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "scheduler: mode=%s rgb=%d thermal=%d rails=%s",
                final.mode.name, final.rgb_fps, final.thermal_fps,
                ",".join(final.rails_triggered) or "none",
            )
        return final

    @property
    def last_action(self) -> Optional[SensorAction]:
        return self._last_action
