"""
Project Sanjay Mk2 - Mission Policy Engine
==========================================
Deterministic mission-policy layer for the Alpha-only police swarm.

This module sits above perception/threat tracking and below swarm/flight
control. It does not command motors directly; it scores threats, gates
inspection descent, and chooses safe high-level actions for the existing
coordination stack to execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from src.core.types.drone_types import (
    AutonomyDecision,
    AutonomyDecisionType,
    CrowdRiskState,
    DroneMissionState,
    InspectionRecommendation,
    SensorType,
    Threat,
    ThreatLevel,
    ThreatVector,
    Vector3,
)


@dataclass
class MissionPolicyConfig:
    """Thresholds and gates for Alpha-only inspection autonomy."""

    critical_threat_threshold: float = 0.75
    multi_sensor_min_count: int = 2
    max_active_inspectors: int = 1
    min_sector_coverage_pct: float = 75.0
    allow_crowd_descent: bool = False
    facade_scan_standoff: float = 30.0
    patrol_altitude: float = 65.0
    inspection_altitude: float = 35.0
    max_confirmation_distance: float = 12.0
    disconnected_allows_new_descent: bool = False


FACADE_OBJECT_TYPES = {
    "weapon_person",
    "person_window",
    "window_intruder",
    "unauthorized_access",
    "thermal_hotspot",
    "fire",
}

CROWD_OBJECT_TYPES = {"crowd_risk"}


class MissionPolicyEngine:
    """Score threats and select deterministic autonomy actions."""

    def __init__(self, config: Optional[MissionPolicyConfig] = None):
        self.config = config or MissionPolicyConfig()

    def build_threat_vector(
        self,
        threat: Threat,
        sensor_evidence: Iterable[SensorType],
        mission_profile: str = "crowd_event",
        crowd_risk: Optional[CrowdRiskState] = None,
    ) -> ThreatVector:
        evidence = self._normalize_sensor_evidence(sensor_evidence)
        recommendation = self._recommendation_for(threat, evidence, crowd_risk)
        spatial_urgency = min(1.0, 0.4 + threat.confidence * 0.6)
        behavioral_urgency = min(1.0, threat.threat_score)
        return ThreatVector(
            threat_id=threat.threat_id,
            threat_level=threat.threat_level,
            position=threat.position,
            object_type=threat.object_type,
            confidence=threat.confidence,
            threat_score=threat.threat_score,
            sensor_evidence=evidence,
            crowd_risk_score=crowd_risk.stampede_risk if crowd_risk else 0.0,
            spatial_urgency=spatial_urgency,
            behavioral_urgency=behavioral_urgency,
            inspection_recommendation=recommendation,
            mission_profile=mission_profile,
        )

    def evaluate_threat(
        self,
        vector: ThreatVector,
        active_inspectors: int = 0,
        sector_coverage_pct: float = 100.0,
        corridor_safe: bool = True,
        swarm_coverage_ready: bool = True,
        operator_hold: bool = False,
        gcs_connected: bool = True,
    ) -> AutonomyDecision:
        """Return the deterministic action for a fused threat vector."""
        if operator_hold:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HOLD,
                mission_state=DroneMissionState.TRACK_HIGH,
                recommendation=InspectionRecommendation.HOLD_POSITION,
                should_descend=False,
                score=vector.threat_score,
                reason="operator_hold",
            )

        if vector.object_type in CROWD_OBJECT_TYPES:
            return AutonomyDecision(
                decision=AutonomyDecisionType.CROWD_RETASK,
                mission_state=DroneMissionState.CROWD_OVERWATCH,
                recommendation=InspectionRecommendation.CROWD_OVERWATCH,
                should_descend=False,
                score=vector.crowd_risk_score or vector.threat_score,
                reason="crowd_overwatch_high_altitude_only",
            )

        if active_inspectors >= self.config.max_active_inspectors:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HOLD,
                mission_state=DroneMissionState.TRACK_HIGH,
                recommendation=InspectionRecommendation.TRACK_HIGH,
                should_descend=False,
                score=vector.threat_score,
                reason="active_inspector_limit_reached",
            )

        if len(vector.sensor_evidence) < self.config.multi_sensor_min_count:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HOLD,
                mission_state=DroneMissionState.TRACK_HIGH,
                recommendation=InspectionRecommendation.TRACK_HIGH,
                should_descend=False,
                score=vector.threat_score,
                reason="insufficient_sensor_evidence",
            )

        if vector.threat_score < self.config.critical_threat_threshold:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HOLD,
                mission_state=DroneMissionState.TRACK_HIGH,
                recommendation=InspectionRecommendation.TRACK_HIGH,
                should_descend=False,
                score=vector.threat_score,
                reason="below_critical_threshold",
            )

        if sector_coverage_pct < self.config.min_sector_coverage_pct or not swarm_coverage_ready:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HOLD,
                mission_state=DroneMissionState.INSPECTION_PENDING,
                recommendation=InspectionRecommendation.TRACK_HIGH,
                should_descend=False,
                score=vector.threat_score,
                reason="coverage_repair_pending",
                requires_swarm_approval=True,
            )

        if not corridor_safe:
            return AutonomyDecision(
                decision=AutonomyDecisionType.ABORT,
                mission_state=DroneMissionState.DEGRADED_SAFE,
                recommendation=InspectionRecommendation.ABORT,
                should_descend=False,
                score=vector.threat_score,
                reason="lidar_corridor_unsafe",
            )

        if not gcs_connected and not self.config.disconnected_allows_new_descent:
            return AutonomyDecision(
                decision=AutonomyDecisionType.HOLD,
                mission_state=DroneMissionState.TRACK_HIGH,
                recommendation=InspectionRecommendation.TRACK_HIGH,
                should_descend=False,
                score=vector.threat_score,
                reason="gcs_disconnected_new_descent_blocked",
            )

        decision_type = AutonomyDecisionType.DESCEND
        mission_state = DroneMissionState.DESCEND_CONFIRM
        recommendation = InspectionRecommendation.DESCEND_CONFIRM
        reason = "critical_multi_sensor_inspection"

        if vector.inspection_recommendation == InspectionRecommendation.FACADE_SCAN:
            decision_type = AutonomyDecisionType.EXECUTE_FACADE_SCAN
            mission_state = DroneMissionState.FACADE_SCAN
            recommendation = InspectionRecommendation.FACADE_SCAN
            reason = "facade_scan_required"

        return AutonomyDecision(
            decision=decision_type,
            mission_state=mission_state,
            recommendation=recommendation,
            should_descend=True,
            score=vector.threat_score,
            reason=reason,
            requires_swarm_approval=True,
        )

    def select_inspector(
        self,
        threat_position: Vector3,
        drone_positions: Sequence[tuple[int, Vector3]],
        unavailable: Optional[set[int]] = None,
    ) -> Optional[int]:
        """Pick the nearest Alpha for inspection."""
        unavailable = unavailable or set()
        best_id: Optional[int] = None
        best_dist = float("inf")
        for drone_id, pos in drone_positions:
            if drone_id in unavailable:
                continue
            dist = pos.distance_to(threat_position)
            if dist < best_dist:
                best_dist = dist
                best_id = drone_id
        return best_id

    def _recommendation_for(
        self,
        threat: Threat,
        evidence: Sequence[SensorType],
        crowd_risk: Optional[CrowdRiskState],
    ) -> InspectionRecommendation:
        if threat.object_type in CROWD_OBJECT_TYPES or crowd_risk is not None:
            return InspectionRecommendation.CROWD_OVERWATCH
        if threat.object_type in FACADE_OBJECT_TYPES:
            return InspectionRecommendation.FACADE_SCAN
        if threat.threat_level == ThreatLevel.CRITICAL and len(evidence) >= 2:
            return InspectionRecommendation.DESCEND_CONFIRM
        return InspectionRecommendation.TRACK_HIGH

    def _normalize_sensor_evidence(
        self,
        evidence: Iterable[SensorType],
    ) -> list[SensorType]:
        deduped = []
        seen = set()
        for sensor in evidence:
            normalized = SensorType.WIDE_RGB_CAMERA if sensor == SensorType.RGB_CAMERA else sensor
            if normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped
