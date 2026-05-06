"""Tests for the Alpha-only mission-policy layer."""

from src.core.types.drone_types import (
    AutonomyDecisionType,
    DroneMissionState,
    InspectionRecommendation,
    SensorType,
    Threat,
    ThreatLevel,
    Vector3,
)
from src.response.mission_policy import (
    CommsState,
    MissionPolicyConfig,
    MissionPolicyContext,
    MissionPolicyEngine,
    OperatorOverride,
    OperatorOverrideAction,
)


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

    def test_simultaneous_threats_are_prioritized_deterministically(self):
        crowd = self.engine.build_threat_vector(
            Threat(
                threat_id="thr_crowd",
                position=Vector3(50.0, 0.0, 0.0),
                threat_level=ThreatLevel.HIGH,
                object_type="crowd_risk",
                confidence=0.95,
                threat_score=0.98,
                detected_by=1,
            ),
            [SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
        )
        weapon = self.engine.build_threat_vector(
            Threat(
                threat_id="thr_weapon",
                position=Vector3(10.0, 0.0, -5.0),
                threat_level=ThreatLevel.CRITICAL,
                object_type="weapon_person",
                confidence=0.82,
                threat_score=0.82,
                detected_by=0,
            ),
            [SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
        )
        fire = self.engine.build_threat_vector(
            Threat(
                threat_id="thr_fire",
                position=Vector3(30.0, 0.0, -5.0),
                threat_level=ThreatLevel.CRITICAL,
                object_type="fire",
                confidence=0.9,
                threat_score=0.9,
                detected_by=2,
            ),
            [SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
        )

        results = self.engine.evaluate_threats(
            [crowd, fire, weapon],
            MissionPolicyContext(
                active_inspectors=0,
                sector_coverage_pct=95.0,
                corridor_safe=True,
                swarm_coverage_ready=True,
            ),
        )

        assert [vector.threat_id for vector, _ in results] == [
            "thr_weapon",
            "thr_fire",
            "thr_crowd",
        ]
        assert results[0][1].decision in {
            AutonomyDecisionType.DESCEND,
            AutonomyDecisionType.EXECUTE_FACADE_SCAN,
        }
        assert results[0][1].reason_details["priority_rank"] == 1
        assert results[1][1].reason == "active_inspector_limit_reached"

    def test_gcs_degraded_blocks_new_descent_with_audit_reason(self):
        threat = Threat(
            threat_id="thr_comms",
            position=Vector3(0.0, 0.0, -5.0),
            threat_level=ThreatLevel.CRITICAL,
            object_type="explosive_device",
            confidence=0.9,
            threat_score=0.93,
            detected_by=0,
        )
        vector = self.engine.build_threat_vector(
            threat,
            [SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
        )

        decision = self.engine.evaluate_threat(
            vector,
            context=MissionPolicyContext(comms_state=CommsState.GCS_DEGRADED),
        )

        assert decision.decision == AutonomyDecisionType.HOLD
        assert decision.should_descend is False
        assert decision.reason == "gcs_disconnected_new_descent_blocked"
        assert decision.reason_details["comms_state"] == "GCS_DEGRADED"

    def test_operator_force_inspect_bypasses_score_but_not_lidar_safety(self):
        threat = Threat(
            threat_id="thr_force",
            position=Vector3(0.0, 0.0, -5.0),
            threat_level=ThreatLevel.HIGH,
            object_type="vehicle",
            confidence=0.6,
            threat_score=0.4,
            detected_by=0,
        )
        vector = self.engine.build_threat_vector(threat, [SensorType.WIDE_RGB_CAMERA])

        forced = self.engine.evaluate_threat(
            vector,
            context=MissionPolicyContext(
                corridor_safe=True,
                operator_override=OperatorOverride(
                    action=OperatorOverrideAction.FORCE_INSPECT,
                    operator_id="officer_1",
                    reason="suspicious parked vehicle",
                    target_threat_id="thr_force",
                ),
            ),
        )
        blocked = self.engine.evaluate_threat(
            vector,
            context=MissionPolicyContext(
                corridor_safe=False,
                operator_override=OperatorOverride(
                    action=OperatorOverrideAction.FORCE_INSPECT,
                    target_threat_id="thr_force",
                ),
            ),
        )

        assert forced.should_descend is True
        assert forced.reason == "operator_force_inspect"
        assert forced.reason_details["operator_id"] == "officer_1"
        assert blocked.decision == AutonomyDecisionType.ABORT
        assert blocked.reason == "operator_force_inspect_blocked_lidar_corridor_unsafe"

    def test_rejoin_decisions_are_explicit_and_auditable(self):
        hold = self.engine.evaluate_rejoin(
            threat_id="thr_001",
            inspector_id=2,
            confirmation_complete=False,
        )
        rejoin = self.engine.evaluate_rejoin(
            threat_id="thr_001",
            inspector_id=2,
            confirmation_complete=True,
        )
        degraded = self.engine.evaluate_rejoin(
            threat_id="thr_001",
            inspector_id=2,
            confirmation_complete=False,
            comms_state=CommsState.DRONE_ISOLATED,
        )

        assert hold.decision == AutonomyDecisionType.HOLD
        assert hold.mission_state == DroneMissionState.TARGET_CONFIRM
        assert hold.reason == "confirmation_not_complete"
        assert rejoin.decision == AutonomyDecisionType.REASCEND
        assert rejoin.reason == "confirmation_complete_rejoin"
        assert degraded.decision == AutonomyDecisionType.REASCEND
        assert degraded.reason == "comms_degraded_rejoin_to_safe_altitude"
