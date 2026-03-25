"""Tests for the Alpha-only mission-policy layer."""

from src.core.types.drone_types import (
    InspectionRecommendation,
    SensorType,
    Threat,
    ThreatLevel,
    Vector3,
)
from src.response.mission_policy import MissionPolicyConfig, MissionPolicyEngine


class TestMissionPolicyEngine:
    def setup_method(self):
        self.engine = MissionPolicyEngine(
            MissionPolicyConfig(
                critical_threat_threshold=0.75,
                multi_sensor_min_count=2,
                max_active_inspectors=1,
            )
        )

    def test_critical_multi_sensor_threat_descends(self):
        threat = Threat(
            threat_id="thr_001",
            position=Vector3(10.0, 10.0, -10.0),
            threat_level=ThreatLevel.CRITICAL,
            object_type="weapon_person",
            confidence=0.9,
            threat_score=0.92,
            detected_by=0,
        )
        vector = self.engine.build_threat_vector(
            threat,
            [SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
        )
        decision = self.engine.evaluate_threat(
            vector,
            active_inspectors=0,
            sector_coverage_pct=90.0,
            corridor_safe=True,
            swarm_coverage_ready=True,
        )
        assert decision.should_descend is True
        assert decision.recommendation in {
            InspectionRecommendation.DESCEND_CONFIRM,
            InspectionRecommendation.FACADE_SCAN,
        }

    def test_crowd_risk_stays_high(self):
        threat = Threat(
            threat_id="thr_002",
            position=Vector3(50.0, 30.0, 0.0),
            threat_level=ThreatLevel.HIGH,
            object_type="crowd_risk",
            confidence=0.8,
            threat_score=0.84,
            detected_by=-1,
        )
        vector = self.engine.build_threat_vector(
            threat,
            [SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
        )
        decision = self.engine.evaluate_threat(vector)
        assert decision.should_descend is False
        assert decision.recommendation == InspectionRecommendation.CROWD_OVERWATCH

    def test_single_sensor_threat_is_held_high(self):
        threat = Threat(
            threat_id="thr_003",
            position=Vector3(0.0, 0.0, -5.0),
            threat_level=ThreatLevel.CRITICAL,
            object_type="weapon_person",
            confidence=0.9,
            threat_score=0.91,
            detected_by=0,
        )
        vector = self.engine.build_threat_vector(threat, [SensorType.WIDE_RGB_CAMERA])
        decision = self.engine.evaluate_threat(
            vector,
            active_inspectors=0,
            sector_coverage_pct=95.0,
            corridor_safe=True,
            swarm_coverage_ready=True,
        )
        assert decision.should_descend is False
        assert decision.reason == "insufficient_sensor_evidence"
