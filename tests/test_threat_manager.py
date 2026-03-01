"""Tests for the Threat Manager."""
import pytest
from src.surveillance.threat_manager import ThreatManager
from src.surveillance.change_detection import ChangeEvent
from src.core.types.drone_types import Vector3, ThreatLevel, ThreatStatus


def _make_change_event(event_id="chg_001", obj_type="person", confidence=0.7):
    return ChangeEvent(
        event_id=event_id,
        position=Vector3(x=50, y=50, z=0),
        change_type="new_object",
        object_type=obj_type,
        description="Test threat",
        threat_level=ThreatLevel.HIGH,
        confidence=confidence,
        detected_by=0,
    )


class TestThreatCreation:
    def test_report_creates_threat(self):
        tm = ThreatManager()
        event = _make_change_event()
        threat = tm.report_change(event, current_time=1.0)
        assert threat is not None
        assert threat.threat_level == ThreatLevel.HIGH
        assert threat.detected_by == 0

    def test_high_confidence_auto_promotes(self):
        tm = ThreatManager(confirmation_threshold=0.5)
        event = _make_change_event(confidence=0.8)
        threat = tm.report_change(event, current_time=1.0)
        assert threat.status == ThreatStatus.PENDING_CONFIRMATION

    def test_low_confidence_stays_detected(self):
        tm = ThreatManager(confirmation_threshold=0.5)
        event = _make_change_event(confidence=0.3)
        threat = tm.report_change(event, current_time=1.0)
        assert threat.status == ThreatStatus.DETECTED


class TestBetaDispatch:
    def test_request_confirmation_selects_nearest(self):
        tm = ThreatManager(confirmation_threshold=0.5)
        event = _make_change_event(confidence=0.8)
        threat = tm.report_change(event, current_time=1.0)

        betas = [
            (3, Vector3(100, 100, 0)),  # farther
            (4, Vector3(55, 55, 0)),     # closer
        ]
        selected = tm.request_confirmation(threat.threat_id, betas)
        assert selected == 4
        assert threat.status == ThreatStatus.CONFIRMING
        assert threat.assigned_beta == 4

    def test_no_betas_available(self):
        tm = ThreatManager(confirmation_threshold=0.5)
        event = _make_change_event(confidence=0.8)
        threat = tm.report_change(event, current_time=1.0)
        selected = tm.request_confirmation(threat.threat_id, [])
        assert selected is None


class TestThreatLifecycle:
    def test_confirm_threat(self):
        tm = ThreatManager(confirmation_threshold=0.5)
        event = _make_change_event(confidence=0.8)
        threat = tm.report_change(event, current_time=1.0)
        tm.request_confirmation(threat.threat_id, [(4, Vector3(55, 55, 0))])

        result = tm.confirm_threat(threat.threat_id, is_confirmed=True, current_time=10.0)
        assert result.status == ThreatStatus.CONFIRMED
        assert result.confirmation_time == 10.0

    def test_clear_false_positive(self):
        tm = ThreatManager(confirmation_threshold=0.5)
        event = _make_change_event(confidence=0.8)
        threat = tm.report_change(event, current_time=1.0)
        tm.request_confirmation(threat.threat_id, [(4, Vector3(55, 55, 0))])

        result = tm.confirm_threat(threat.threat_id, is_confirmed=False, current_time=10.0)
        assert result.status == ThreatStatus.CLEARED

    def test_resolve_threat(self):
        tm = ThreatManager()
        event = _make_change_event()
        threat = tm.report_change(event, current_time=1.0)
        tm.resolve_threat(threat.threat_id, current_time=50.0)
        assert threat.status == ThreatStatus.RESOLVED

    def test_aging(self):
        tm = ThreatManager(threat_timeout=60.0)
        event = _make_change_event(confidence=0.3)
        threat = tm.report_change(event, current_time=1.0)
        assert threat.status == ThreatStatus.DETECTED

        tm.update(current_time=100.0)  # 99 seconds later > 60s timeout
        assert threat.status == ThreatStatus.RESOLVED

    def test_get_active_excludes_resolved(self):
        tm = ThreatManager()
        e1 = _make_change_event(event_id="a")
        e2 = _make_change_event(event_id="b")
        t1 = tm.report_change(e1, current_time=1.0)
        t2 = tm.report_change(e2, current_time=2.0)
        tm.resolve_threat(t1.threat_id)
        active = tm.get_active_threats()
        assert len(active) == 1
        assert active[0].threat_id == t2.threat_id
