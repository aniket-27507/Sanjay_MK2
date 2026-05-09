"""
Tests for src/single_drone/sensor_scheduler.py (Phase A).

One test per hard rail, one test per heuristic mode transition.
"""

from __future__ import annotations

import pytest

from src.core.types.drone_types import DroneMissionState
from src.single_drone.sensor_scheduler import (
    FPS_BURST,
    FPS_HIGH,
    FPS_LOW,
    FPS_MED,
    FPS_OFF,
    HardRails,
    HeuristicPolicy,
    OperatorOverride,
    SensorAction,
    SensorMode,
    SensorScheduler,
    SensorState,
)


# ────────────────────────────────────────────────────────────────────
#  Hard rails
# ────────────────────────────────────────────────────────────────────


def test_rail_R1_low_lux_forces_thermal_on():
    """Ambient lux below 10 must force thermal to at least FPS_MED."""
    state = SensorState(ambient_lux=5.0)
    proposed = SensorAction(rgb_fps=FPS_HIGH, thermal_fps=FPS_OFF, mode=SensorMode.DAY_PATROL)

    final = HardRails.apply(state, proposed)

    assert final.thermal_fps >= FPS_MED
    assert "R1_night_thermal" in final.rails_triggered


def test_rail_R2_rgb_evidence_minimum():
    """RGB FPS must never drop below FPS_LOW (evidence chain)."""
    state = SensorState(ambient_lux=50000.0)
    proposed = SensorAction(rgb_fps=FPS_OFF, thermal_fps=FPS_HIGH, mode=SensorMode.NIGHT_PATROL)

    final = HardRails.apply(state, proposed)

    assert final.rgb_fps >= FPS_LOW
    assert "R2_rgb_evidence_min" in final.rails_triggered


@pytest.mark.parametrize("mission", [
    DroneMissionState.INSPECTION_PENDING,
    DroneMissionState.DESCEND_CONFIRM,
    DroneMissionState.FACADE_SCAN,
    DroneMissionState.TARGET_CONFIRM,
])
def test_rail_R3_inspection_forces_dual_sensors(mission):
    """Inspection phases demand both RGB and thermal at FPS_HIGH."""
    state = SensorState(ambient_lux=50000.0, mission_state=mission)
    proposed = SensorAction(rgb_fps=FPS_LOW, thermal_fps=FPS_OFF, mode=SensorMode.DAY_PATROL)

    final = HardRails.apply(state, proposed)

    assert final.rgb_fps >= FPS_HIGH
    assert final.thermal_fps >= FPS_HIGH
    assert "R3_inspection_dual" in final.rails_triggered


def test_rail_R4_weapon_confidence_forces_cross_modal():
    """weapon_person confidence > 0.3 on either sensor -> both on at FPS_HIGH."""
    state = SensorState(ambient_lux=50000.0, weapon_class_conf=0.5)
    proposed = SensorAction(rgb_fps=FPS_HIGH, thermal_fps=FPS_OFF, mode=SensorMode.DAY_PATROL)

    final = HardRails.apply(state, proposed)

    assert final.rgb_fps >= FPS_HIGH
    assert final.thermal_fps >= FPS_HIGH
    assert "R4_weapon_cross_modal" in final.rails_triggered


def test_rail_R4_weapon_low_conf_does_not_trigger():
    """Below the 0.3 threshold, no cross-modal forcing."""
    state = SensorState(ambient_lux=50000.0, weapon_class_conf=0.1)
    proposed = SensorAction(rgb_fps=FPS_HIGH, thermal_fps=FPS_OFF, mode=SensorMode.DAY_PATROL)

    final = HardRails.apply(state, proposed)

    assert "R4_weapon_cross_modal" not in final.rails_triggered
    assert final.thermal_fps == FPS_OFF  # stays off


def test_rail_R5_operator_override_wins():
    """Explicit operator override replaces whatever policy+rails chose."""
    state = SensorState(
        ambient_lux=5.0,  # would force thermal on via R1
        operator_override=OperatorOverride(rgb_fps=5, thermal_fps=0),
    )
    proposed = SensorAction(rgb_fps=FPS_HIGH, thermal_fps=FPS_HIGH, mode=SensorMode.NIGHT_PATROL)

    final = HardRails.apply(state, proposed)

    assert final.rgb_fps == 5
    assert final.thermal_fps == 0          # override beats R1
    assert "R5_operator_rgb_override" in final.rails_triggered
    assert "R5_operator_thermal_override" in final.rails_triggered


# ────────────────────────────────────────────────────────────────────
#  Heuristic policy state machine
# ────────────────────────────────────────────────────────────────────


def test_heuristic_day_patrol_default():
    state = SensorState(ambient_lux=50000.0)
    action = HeuristicPolicy.decide(state)
    assert action.mode == SensorMode.DAY_PATROL
    assert action.rgb_fps == FPS_HIGH
    assert action.thermal_fps == FPS_OFF


def test_heuristic_night_patrol_on_low_lux():
    state = SensorState(ambient_lux=5.0)
    action = HeuristicPolicy.decide(state)
    assert action.mode == SensorMode.NIGHT_PATROL
    assert action.thermal_fps == FPS_HIGH


def test_heuristic_inspect_dual_on_inspection_state():
    state = SensorState(
        ambient_lux=50000.0,
        mission_state=DroneMissionState.DESCEND_CONFIRM,
    )
    action = HeuristicPolicy.decide(state)
    assert action.mode == SensorMode.INSPECT_DUAL
    assert action.rgb_fps == FPS_HIGH
    assert action.thermal_fps == FPS_HIGH


def test_heuristic_emergency_burst_requires_track_high_and_missed_streak():
    """EMERGENCY_BURST gated on (mission_state == TRACK_HIGH) AND (missed_streak >= 3).
    Both conditions necessary -- the burst is the response to losing sight of an
    actively-tracked threat, not to dry patrol generally."""
    state = SensorState(
        ambient_lux=50000.0,
        mission_state=DroneMissionState.TRACK_HIGH,
        missed_detection_streak=5,
    )
    action = HeuristicPolicy.decide(state)
    assert action.mode == SensorMode.EMERGENCY_BURST
    assert action.rgb_fps == FPS_BURST
    assert action.thermal_fps == FPS_BURST


def test_heuristic_no_burst_on_quiet_patrol():
    """Pure PATROL_HIGH with high missed_streak must NOT trigger burst.

    This is the regression that motivated the gating change: once
    missed_detection_streak got wired into scenario_executor, the old
    rule ('missed_streak >= 3 -> burst') fired on every quiet patrol
    after ~1.5s of no detections. New rule requires TRACK_HIGH too.
    """
    state = SensorState(
        ambient_lux=50000.0,
        mission_state=DroneMissionState.PATROL_HIGH,
        missed_detection_streak=20,   # very long dry patrol
    )
    action = HeuristicPolicy.decide(state)
    assert action.mode == SensorMode.DAY_PATROL, (
        f"missed_streak alone must not trigger burst on PATROL_HIGH; got {action.mode}"
    )
    assert action.thermal_fps == FPS_OFF


def test_heuristic_track_high_without_missed_streak_no_burst():
    """TRACK_HIGH alone (no missed streak) is just active tracking, not burst."""
    state = SensorState(
        ambient_lux=50000.0,
        mission_state=DroneMissionState.TRACK_HIGH,
        missed_detection_streak=0,
    )
    action = HeuristicPolicy.decide(state)
    assert action.mode != SensorMode.EMERGENCY_BURST


def test_heuristic_inspection_beats_emergency_when_not_track_high():
    """Under the new gating, INSPECT_DUAL fires for inspection states regardless
    of missed_streak (since they're not TRACK_HIGH).  This documents that the
    priority order is now EMERGENCY_BURST > INSPECT_DUAL > NIGHT > DAY only
    when EMERGENCY_BURST's gating conditions are met."""
    state = SensorState(
        ambient_lux=50000.0,
        mission_state=DroneMissionState.DESCEND_CONFIRM,
        missed_detection_streak=10,
    )
    action = HeuristicPolicy.decide(state)
    assert action.mode == SensorMode.INSPECT_DUAL


# ────────────────────────────────────────────────────────────────────
#  Scheduler end-to-end (policy + rails composed)
# ────────────────────────────────────────────────────────────────────


def test_scheduler_tick_applies_both_layers():
    """tick() should run policy then rails, producing the audited action."""
    scheduler = SensorScheduler()
    state = SensorState(ambient_lux=5.0)  # triggers NIGHT_PATROL + R1 already satisfied

    action = scheduler.tick(state)

    assert action.mode == SensorMode.NIGHT_PATROL
    assert action.rgb_fps >= FPS_LOW       # R2 satisfied
    assert action.thermal_fps >= FPS_MED   # R1 satisfied
    assert scheduler.last_action is action


def test_scheduler_tick_persists_last_action():
    scheduler = SensorScheduler()
    assert scheduler.last_action is None
    scheduler.tick(SensorState())
    assert scheduler.last_action is not None


def test_scheduler_rails_fire_on_bad_policy_output():
    """If a (future) policy proposes an unsafe action, rails still correct it."""

    class EvilPolicy:
        @staticmethod
        def decide(state):
            # Intentionally unsafe: everything off at night.
            return SensorAction(rgb_fps=FPS_OFF, thermal_fps=FPS_OFF, mode=SensorMode.DAY_PATROL)

    scheduler = SensorScheduler(policy=EvilPolicy())
    state = SensorState(ambient_lux=5.0)

    action = scheduler.tick(state)

    assert action.rgb_fps >= FPS_LOW       # R2 fixed it
    assert action.thermal_fps >= FPS_MED   # R1 fixed it
    assert "R1_night_thermal" in action.rails_triggered
    assert "R2_rgb_evidence_min" in action.rails_triggered
