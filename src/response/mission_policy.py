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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterable, Optional, Sequence

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


class CommsState(Enum):
    """Operational communication state visible to mission policy."""

    CONNECTED = auto()
    GCS_DEGRADED = auto()
    SWARM_DEGRADED = auto()
    DRONE_ISOLATED = auto()


class OperatorOverrideAction(Enum):
    """Explicit operator commands that can override normal policy scoring."""

    NONE = auto()
    HOLD = auto()
    FORCE_INSPECT = auto()
    FORCE_REJOIN = auto()
    ABORT = auto()


@dataclass
class OperatorOverride:
    """Human-in-the-loop override attached to a policy evaluation."""

    action: OperatorOverrideAction = OperatorOverrideAction.NONE
    operator_id: str = "operator"
    reason: str = ""
    target_threat_id: Optional[str] = None
    target_drone_id: Optional[int] = None

    def applies_to(self, threat_id: str) -> bool:
        return self.target_threat_id in {None, threat_id}


@dataclass
class MissionPolicyContext:
    """Runtime gates used by mission policy for a single evaluation step."""

    active_inspectors: int = 0
    sector_coverage_pct: float = 100.0
    corridor_safe: bool = True
    swarm_coverage_ready: bool = True
    comms_state: CommsState = CommsState.CONNECTED
    operator_override: OperatorOverride = field(default_factory=OperatorOverride)

    @property
    def gcs_connected(self) -> bool:
        return self.comms_state != CommsState.GCS_DEGRADED


FACADE_OBJECT_TYPES = {
    "weapon_person",
    "person_window",
    "window_intruder",
    "unauthorized_access",
    "thermal_hotspot",
    "fire",
}

CROWD_OBJECT_TYPES = {"crowd_risk"}


OBJECT_PRIORITY = {
    "weapon_person": 500,
    "explosive_device": 500,
    "fire": 300,
    "thermal_hotspot": 250,
    "person_window": 240,
    "window_intruder": 240,
    "unauthorized_access": 220,
    "crowd_risk": 200,
    "crowd": 180,
    "person": 100,
    "vehicle": 50,
}

THREAT_LEVEL_PRIORITY = {
    ThreatLevel.CRITICAL: 4,
    ThreatLevel.HIGH: 3,
    ThreatLevel.MEDIUM: 2,
    ThreatLevel.LOW: 1,
    ThreatLevel.UNKNOWN: 0,
}


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
        context: Optional[MissionPolicyContext] = None,
        operator_override: Optional[OperatorOverride] = None,
    ) -> AutonomyDecision:
        """Return the deterministic action for a fused threat vector."""
        ctx = context or self._context_from_legacy_args(
            active_inspectors=active_inspectors,
            sector_coverage_pct=sector_coverage_pct,
            corridor_safe=corridor_safe,
            swarm_coverage_ready=swarm_coverage_ready,
            operator_hold=operator_hold,
            gcs_connected=gcs_connected,
            operator_override=operator_override,
        )
        override = ctx.operator_override
        force_inspect = (
            override.action == OperatorOverrideAction.FORCE_INSPECT
            and override.applies_to(vector.threat_id)
        )

        if override.applies_to(vector.threat_id):
            if override.action == OperatorOverrideAction.ABORT:
                return self._decision(
                    AutonomyDecisionType.ABORT,
                    DroneMissionState.DEGRADED_SAFE,
                    InspectionRecommendation.ABORT,
                    False,
                    vector.threat_score,
                    "operator_abort",
                    self._audit_details(vector, ctx, operator_override=override),
                )
            if override.action == OperatorOverrideAction.HOLD:
                return self._decision(
                    AutonomyDecisionType.HOLD,
                    DroneMissionState.TRACK_HIGH,
                    InspectionRecommendation.HOLD_POSITION,
                    False,
                    vector.threat_score,
                    "operator_hold",
                    self._audit_details(vector, ctx, operator_override=override),
                )
            if override.action == OperatorOverrideAction.FORCE_REJOIN:
                return self._decision(
                    AutonomyDecisionType.REASCEND,
                    DroneMissionState.REASCEND_REJOIN,
                    InspectionRecommendation.REASCEND_REJOIN,
                    False,
                    vector.threat_score,
                    "operator_force_rejoin",
                    self._audit_details(vector, ctx, operator_override=override),
                )

        if vector.object_type in CROWD_OBJECT_TYPES and not self.config.allow_crowd_descent:
            reason = (
                "operator_force_inspect_blocked_crowd_descent_disabled"
                if force_inspect
                else "crowd_overwatch_high_altitude_only"
            )
            return self._decision(
                AutonomyDecisionType.CROWD_RETASK,
                DroneMissionState.CROWD_OVERWATCH,
                InspectionRecommendation.CROWD_OVERWATCH,
                False,
                vector.crowd_risk_score or vector.threat_score,
                reason,
                self._audit_details(vector, ctx, operator_override=override),
            )

        if ctx.active_inspectors >= self.config.max_active_inspectors:
            reason = (
                "operator_force_inspect_blocked_active_inspector_limit"
                if force_inspect
                else "active_inspector_limit_reached"
            )
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.TRACK_HIGH,
                InspectionRecommendation.TRACK_HIGH,
                False,
                vector.threat_score,
                reason,
                self._audit_details(vector, ctx, operator_override=override),
            )

        if len(vector.sensor_evidence) < self.config.multi_sensor_min_count and not force_inspect:
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.TRACK_HIGH,
                InspectionRecommendation.TRACK_HIGH,
                False,
                vector.threat_score,
                "insufficient_sensor_evidence",
                self._audit_details(vector, ctx, operator_override=override),
            )

        if vector.threat_score < self.config.critical_threat_threshold and not force_inspect:
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.TRACK_HIGH,
                InspectionRecommendation.TRACK_HIGH,
                False,
                vector.threat_score,
                "below_critical_threshold",
                self._audit_details(vector, ctx, operator_override=override),
            )

        if ctx.sector_coverage_pct < self.config.min_sector_coverage_pct or not ctx.swarm_coverage_ready:
            reason = (
                "operator_force_inspect_blocked_coverage_repair_pending"
                if force_inspect
                else "coverage_repair_pending"
            )
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.INSPECTION_PENDING,
                InspectionRecommendation.TRACK_HIGH,
                False,
                vector.threat_score,
                reason,
                self._audit_details(vector, ctx, operator_override=override),
                requires_swarm_approval=True,
            )

        if not ctx.corridor_safe:
            reason = (
                "operator_force_inspect_blocked_lidar_corridor_unsafe"
                if force_inspect
                else "lidar_corridor_unsafe"
            )
            return self._decision(
                AutonomyDecisionType.ABORT,
                DroneMissionState.DEGRADED_SAFE,
                InspectionRecommendation.ABORT,
                False,
                vector.threat_score,
                reason,
                self._audit_details(vector, ctx, operator_override=override),
            )

        if self._new_descent_blocked_by_comms(ctx) and not force_inspect:
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.TRACK_HIGH,
                InspectionRecommendation.TRACK_HIGH,
                False,
                vector.threat_score,
                self._comms_block_reason(ctx),
                self._audit_details(vector, ctx, operator_override=override),
            )

        decision_type = AutonomyDecisionType.DESCEND
        mission_state = DroneMissionState.DESCEND_CONFIRM
        recommendation = InspectionRecommendation.DESCEND_CONFIRM
        reason = "operator_force_inspect" if force_inspect else "critical_multi_sensor_inspection"

        if vector.inspection_recommendation == InspectionRecommendation.FACADE_SCAN:
            decision_type = AutonomyDecisionType.EXECUTE_FACADE_SCAN
            mission_state = DroneMissionState.FACADE_SCAN
            recommendation = InspectionRecommendation.FACADE_SCAN
            reason = "operator_force_facade_scan" if force_inspect else "facade_scan_required"

        return self._decision(
            decision_type,
            mission_state,
            recommendation,
            True,
            vector.threat_score,
            reason,
            self._audit_details(vector, ctx, operator_override=override),
            requires_swarm_approval=True,
        )

    def prioritize_threats(self, vectors: Sequence[ThreatVector]) -> list[ThreatVector]:
        """Return threats in deterministic mission-policy priority order."""
        return sorted(vectors, key=self._priority_sort_key)

    def evaluate_threats(
        self,
        vectors: Sequence[ThreatVector],
        context: Optional[MissionPolicyContext] = None,
    ) -> list[tuple[ThreatVector, AutonomyDecision]]:
        """Evaluate simultaneous threats with deterministic arbitration."""
        ctx = context or MissionPolicyContext()
        results: list[tuple[ThreatVector, AutonomyDecision]] = []
        active_inspectors = ctx.active_inspectors

        for rank, vector in enumerate(self.prioritize_threats(vectors), start=1):
            local_ctx = MissionPolicyContext(
                active_inspectors=active_inspectors,
                sector_coverage_pct=ctx.sector_coverage_pct,
                corridor_safe=ctx.corridor_safe,
                swarm_coverage_ready=ctx.swarm_coverage_ready,
                comms_state=ctx.comms_state,
                operator_override=ctx.operator_override,
            )
            decision = self.evaluate_threat(vector, context=local_ctx)
            decision.reason_details["priority_rank"] = rank
            decision.reason_details["priority_key"] = self._priority_tuple(vector)
            results.append((vector, decision))
            if decision.decision in {
                AutonomyDecisionType.DESCEND,
                AutonomyDecisionType.EXECUTE_FACADE_SCAN,
            }:
                active_inspectors += 1
        return results

    def evaluate_rejoin(
        self,
        threat_id: str,
        inspector_id: int,
        confirmation_complete: bool,
        corridor_safe: bool = True,
        sector_rejoin_ready: bool = True,
        comms_state: CommsState = CommsState.CONNECTED,
        operator_override: Optional[OperatorOverride] = None,
    ) -> AutonomyDecision:
        """Return the explicit, auditable rejoin/hold decision for an inspector."""
        override = operator_override or OperatorOverride()
        details = {
            "threat_id": threat_id,
            "inspector_id": inspector_id,
            "confirmation_complete": confirmation_complete,
            "corridor_safe": corridor_safe,
            "sector_rejoin_ready": sector_rejoin_ready,
            "comms_state": comms_state.name,
            "operator_override": override.action.name,
            "operator_id": override.operator_id,
            "operator_reason": override.reason,
        }

        if override.action == OperatorOverrideAction.ABORT:
            return self._decision(
                AutonomyDecisionType.ABORT,
                DroneMissionState.DEGRADED_SAFE,
                InspectionRecommendation.ABORT,
                False,
                1.0,
                "operator_abort_rejoin",
                details,
            )

        if not corridor_safe:
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.DEGRADED_SAFE,
                InspectionRecommendation.HOLD_POSITION,
                False,
                1.0,
                "rejoin_blocked_lidar_corridor_unsafe",
                details,
            )

        if comms_state != CommsState.CONNECTED:
            return self._decision(
                AutonomyDecisionType.REASCEND,
                DroneMissionState.REASCEND_REJOIN,
                InspectionRecommendation.REASCEND_REJOIN,
                False,
                1.0,
                "comms_degraded_rejoin_to_safe_altitude",
                details,
            )

        if override.action == OperatorOverrideAction.FORCE_REJOIN:
            return self._decision(
                AutonomyDecisionType.REASCEND,
                DroneMissionState.REASCEND_REJOIN,
                InspectionRecommendation.REASCEND_REJOIN,
                False,
                1.0,
                "operator_force_rejoin",
                details,
            )

        if not confirmation_complete:
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.TARGET_CONFIRM,
                InspectionRecommendation.TARGET_CONFIRM,
                False,
                0.0,
                "confirmation_not_complete",
                details,
            )

        if not sector_rejoin_ready:
            return self._decision(
                AutonomyDecisionType.HOLD,
                DroneMissionState.TARGET_CONFIRM,
                InspectionRecommendation.HOLD_POSITION,
                False,
                0.0,
                "sector_rejoin_not_ready",
                details,
            )

        return self._decision(
            AutonomyDecisionType.REASCEND,
            DroneMissionState.REASCEND_REJOIN,
            InspectionRecommendation.REASCEND_REJOIN,
            False,
            1.0,
            "confirmation_complete_rejoin",
            details,
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

    def _context_from_legacy_args(
        self,
        active_inspectors: int,
        sector_coverage_pct: float,
        corridor_safe: bool,
        swarm_coverage_ready: bool,
        operator_hold: bool,
        gcs_connected: bool,
        operator_override: Optional[OperatorOverride],
    ) -> MissionPolicyContext:
        if operator_override is None and operator_hold:
            operator_override = OperatorOverride(action=OperatorOverrideAction.HOLD)
        comms_state = CommsState.CONNECTED if gcs_connected else CommsState.GCS_DEGRADED
        return MissionPolicyContext(
            active_inspectors=active_inspectors,
            sector_coverage_pct=sector_coverage_pct,
            corridor_safe=corridor_safe,
            swarm_coverage_ready=swarm_coverage_ready,
            comms_state=comms_state,
            operator_override=operator_override or OperatorOverride(),
        )

    def _new_descent_blocked_by_comms(self, ctx: MissionPolicyContext) -> bool:
        if ctx.comms_state in {CommsState.SWARM_DEGRADED, CommsState.DRONE_ISOLATED}:
            return True
        return ctx.comms_state == CommsState.GCS_DEGRADED and not self.config.disconnected_allows_new_descent

    def _comms_block_reason(self, ctx: MissionPolicyContext) -> str:
        if ctx.comms_state == CommsState.SWARM_DEGRADED:
            return "swarm_comms_degraded_new_descent_blocked"
        if ctx.comms_state == CommsState.DRONE_ISOLATED:
            return "drone_isolated_new_descent_blocked"
        return "gcs_disconnected_new_descent_blocked"

    def _priority_tuple(self, vector: ThreatVector) -> tuple[int, int, float, float, float, str]:
        return (
            OBJECT_PRIORITY.get(vector.object_type, 0),
            THREAT_LEVEL_PRIORITY.get(vector.threat_level, 0),
            round(vector.threat_score, 6),
            round(vector.confidence, 6),
            round(vector.crowd_risk_score, 6),
            vector.threat_id,
        )

    def _priority_sort_key(self, vector: ThreatVector) -> tuple[int, int, float, float, float, str]:
        priority, level, score, confidence, crowd_risk, threat_id = self._priority_tuple(vector)
        return (-priority, -level, -score, -confidence, -crowd_risk, threat_id)

    def _audit_details(
        self,
        vector: ThreatVector,
        ctx: MissionPolicyContext,
        operator_override: Optional[OperatorOverride] = None,
    ) -> dict[str, Any]:
        override = operator_override or OperatorOverride()
        return {
            "threat_id": vector.threat_id,
            "object_type": vector.object_type,
            "threat_level": vector.threat_level.name,
            "threat_score": round(vector.threat_score, 3),
            "confidence": round(vector.confidence, 3),
            "sensor_evidence_count": len(vector.sensor_evidence),
            "sensor_evidence": [sensor.name for sensor in vector.sensor_evidence],
            "active_inspectors": ctx.active_inspectors,
            "max_active_inspectors": self.config.max_active_inspectors,
            "sector_coverage_pct": round(ctx.sector_coverage_pct, 2),
            "min_sector_coverage_pct": self.config.min_sector_coverage_pct,
            "corridor_safe": ctx.corridor_safe,
            "swarm_coverage_ready": ctx.swarm_coverage_ready,
            "comms_state": ctx.comms_state.name,
            "operator_override": override.action.name,
            "operator_id": override.operator_id,
            "operator_reason": override.reason,
        }

    def _decision(
        self,
        decision: AutonomyDecisionType,
        mission_state: DroneMissionState,
        recommendation: InspectionRecommendation,
        should_descend: bool,
        score: float,
        reason: str,
        reason_details: dict[str, Any],
        requires_swarm_approval: bool = False,
    ) -> AutonomyDecision:
        return AutonomyDecision(
            decision=decision,
            mission_state=mission_state,
            recommendation=recommendation,
            should_descend=should_descend,
            score=score,
            reason=reason,
            requires_swarm_approval=requires_swarm_approval,
            reason_details=reason_details,
        )
