"""
Project Sanjay Mk2 - Threat Manager
=====================================
Manages the lifecycle of detected threats and coordinates
Beta drone dispatch for visual confirmation.

Threat Lifecycle:
    DETECTED -> PENDING -> CONFIRMING -> CONFIRMED | CLEARED -> RESOLVED
    
    - DETECTED: New anomaly from ChangeDetector
    - PENDING: Queued for Beta inspection
    - CONFIRMING: Beta is en route
    - CONFIRMED: Beta verified threat
    - CLEARED: Beta confirmed false positive

@author: Prathamesh Hiwarkar
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple, Set

from src.core.types.drone_types import (
    Vector3, ThreatLevel, ThreatStatus, Threat,
    CrowdZone, StampedeIndicator, StampedeRiskLevel,
)
from src.surveillance.beta_shepherd import BetaShepherdProtocol
from src.surveillance.change_detection import ChangeEvent

logger = logging.getLogger(__name__)

# Confidence threshold to request Beta confirmation
CONFIRMATION_THRESHOLD = 0.50

# Time before a threat ages out (seconds)
THREAT_TIMEOUT = 120.0

# Distance threshold for Beta to be "on scene"
BETA_ARRIVAL_RADIUS = 15.0


class ThreatScorer:
    """
    Weighted threat scoring per spec §5.3.
    Final score = 0.30*Spatial + 0.20*Temporal + 0.35*Behavioural + 0.15*Classification
    """
    WEIGHT_SPATIAL = 0.30
    WEIGHT_TEMPORAL = 0.20
    WEIGHT_BEHAVIOURAL = 0.35
    WEIGHT_CLASSIFICATION = 0.15

    def compute(
        self,
        spatial_score: float,
        temporal_score: float,
        behavioural_score: float,
        classification_score: float,
    ) -> float:
        return (
            self.WEIGHT_SPATIAL * spatial_score
            + self.WEIGHT_TEMPORAL * temporal_score
            + self.WEIGHT_BEHAVIOURAL * behavioural_score
            + self.WEIGHT_CLASSIFICATION * classification_score
        )


class ThreatManager:
    """
    Manages threat lifecycle and coordinates Beta drone dispatch.
    
    Receives change events from the ChangeDetector, tracks threats
    through their lifecycle, and assigns Beta drones for confirmation.
    """

    def __init__(
        self,
        confirmation_threshold: float = CONFIRMATION_THRESHOLD,
        threat_timeout: float = THREAT_TIMEOUT,
        threat_score_threshold: float = 0.65,
        hex_center: Optional[Vector3] = None,
        hex_radius: float = 80.0,
    ):
        self.confirmation_threshold = confirmation_threshold
        self.threat_timeout = threat_timeout
        self.threat_score_threshold = threat_score_threshold
        self.scorer = ThreatScorer()
        self._hex_center = hex_center or Vector3(x=500.0, y=500.0, z=-25.0)
        self._hex_radius = hex_radius

        self._threats: Dict[str, Threat] = {}
        self._threat_counter = 0

        # Map: change_event object_id -> threat_id (prevent duplicate threats)
        self._object_to_threat: Dict[str, str] = {}

        # Active shepherd protocols (threat_id -> BetaShepherdProtocol)
        self._shepherds: Dict[str, BetaShepherdProtocol] = {}

        # Map: crowd zone_id -> threat_id (prevent duplicate crowd threats)
        self._crowd_zone_to_threat: Dict[str, str] = {}

    # ==================== CROWD RISK REPORTING ====================

    # StampedeRiskLevel -> ThreatLevel mapping
    _RISK_TO_THREAT_LEVEL = {
        StampedeRiskLevel.WATCH: ThreatLevel.MEDIUM,
        StampedeRiskLevel.WARNING: ThreatLevel.MEDIUM,
        StampedeRiskLevel.ALERT: ThreatLevel.HIGH,
        StampedeRiskLevel.ACTIVE: ThreatLevel.CRITICAL,
    }

    def report_crowd_risk(
        self,
        zone: CrowdZone,
        indicators: List[StampedeIndicator],
        current_time: Optional[float] = None,
    ) -> Optional[Threat]:
        """
        Report a crowd zone as a threat based on stampede risk analysis.

        Creates or updates a Threat for the zone. Only zones at WARNING
        or above should be reported.

        Args:
            zone: CrowdZone with stampede_risk and risk_level populated
            indicators: StampedeIndicators associated with this zone
            current_time: Current simulation time

        Returns:
            Created/updated Threat, or None if risk level too low.
        """
        current_time = current_time or time.time()

        # Only report if risk level warrants a threat
        if zone.risk_level in (StampedeRiskLevel.NONE, StampedeRiskLevel.WATCH):
            return None

        threat_level = self._RISK_TO_THREAT_LEVEL.get(
            zone.risk_level, ThreatLevel.MEDIUM
        )

        # Check if we already have a threat for this zone
        existing_id = self._crowd_zone_to_threat.get(zone.zone_id)
        if existing_id and existing_id in self._threats:
            threat = self._threats[existing_id]
            threat.threat_level = threat_level
            threat.confidence = min(1.0, zone.stampede_risk)
            threat.threat_score = zone.stampede_risk
            threat.position = zone.center
            # Upgrade status if risk escalated
            if (zone.risk_level == StampedeRiskLevel.ACTIVE
                    and threat.status == ThreatStatus.DETECTED):
                threat.status = ThreatStatus.PENDING_CONFIRMATION
            return threat

        # Create new crowd threat
        self._threat_counter += 1
        threat_id = f"thr_{self._threat_counter:04d}"

        # Use stampede risk as the behavioural dimension (weight=0.35)
        # and derive a composite score
        max_indicator_sev = max(
            (i.severity for i in indicators), default=0.0
        )
        score = self.scorer.compute(
            spatial_score=0.7,  # crowds near infrastructure = high spatial
            temporal_score=0.5,
            behavioural_score=zone.stampede_risk,
            classification_score=0.7,
        )

        status = ThreatStatus.DETECTED
        if score >= self.threat_score_threshold:
            status = ThreatStatus.PENDING_CONFIRMATION

        threat = Threat(
            threat_id=threat_id,
            position=zone.center,
            threat_level=threat_level,
            status=status,
            object_type="crowd_risk",
            confidence=min(1.0, zone.stampede_risk),
            detected_by=-1,  # crowd detection is multi-drone
            detection_time=current_time,
        )
        threat.threat_score = score

        self._threats[threat_id] = threat
        self._crowd_zone_to_threat[zone.zone_id] = threat_id

        indicator_summary = ", ".join(
            f"{i.indicator_type}({i.severity:.2f})" for i in indicators[:3]
        )
        logger.warning(
            "CROWD THREAT created: %s [%s] risk=%.2f density=%.1f/m2 "
            "persons=%d indicators=[%s]",
            threat_id, threat_level.name, zone.stampede_risk,
            zone.peak_density, zone.total_persons, indicator_summary,
        )

        return threat

    def report_change(self, event: ChangeEvent, current_time: Optional[float] = None) -> Threat:
        """
        Report a change event and create/update a threat.
        
        Args:
            event: ChangeEvent from the ChangeDetector
            current_time: Current simulation time
            
        Returns:
            The created or updated Threat.
        """
        current_time = current_time or time.time()

        # Check if we already have a threat for this object
        existing_id = self._object_to_threat.get(event.event_id)
        if existing_id and existing_id in self._threats:
            # Update existing threat
            threat = self._threats[existing_id]
            threat.confidence = max(threat.confidence, event.confidence)
            threat.threat_level = event.threat_level
            return threat

        # Create new threat
        self._threat_counter += 1
        threat_id = f"thr_{self._threat_counter:04d}"

        # Compute composite threat score (spec §5.3)
        score = self.scorer.compute(
            spatial_score=event.spatial_score,
            temporal_score=event.temporal_score,
            behavioural_score=event.behavioural_score,
            classification_score=event.classification_score,
        )

        status = ThreatStatus.DETECTED
        if score >= self.threat_score_threshold:
            status = ThreatStatus.PENDING_CONFIRMATION

        threat = Threat(
            threat_id=threat_id,
            position=Vector3(x=event.position.x, y=event.position.y, z=event.position.z),
            threat_level=event.threat_level,
            status=status,
            object_type=event.object_type,
            confidence=event.confidence,
            detected_by=event.detected_by,
            detection_time=current_time,
        )
        threat.threat_score = score

        self._threats[threat_id] = threat
        self._object_to_threat[event.event_id] = threat_id

        logger.info("Threat created: %s [%s] %s confidence=%.2f at (%.0f, %.0f)",
                     threat_id, threat.threat_level.name, status.name,
                     threat.confidence, threat.position.x, threat.position.y)

        return threat

    def request_confirmation(
        self,
        threat_id: str,
        available_betas: List[Tuple[int, Vector3]],
    ) -> Optional[int]:
        """
        Request Beta drone confirmation for a threat.
        
        Args:
            threat_id: Threat to confirm
            available_betas: List of (drone_id, position) for available Beta drones
            
        Returns:
            Selected Beta drone_id, or None if no Beta available.
        """
        threat = self._threats.get(threat_id)
        if threat is None:
            return None

        if threat.status not in (ThreatStatus.DETECTED, ThreatStatus.PENDING_CONFIRMATION):
            return None

        if not available_betas:
            return None

        # Select nearest Beta drone
        best_id = None
        best_dist = float('inf')
        for beta_id, beta_pos in available_betas:
            dist = threat.position.distance_to(beta_pos)
            if dist < best_dist:
                best_dist = dist
                best_id = beta_id

        if best_id is not None:
            threat.assigned_beta = best_id
            threat.status = ThreatStatus.CONFIRMING

            # Find the Beta's position for shepherd initialisation
            beta_pos = None
            for bid, bpos in available_betas:
                if bid == best_id:
                    beta_pos = bpos
                    break

            # Start shepherd guidance (spec §7.2)
            shepherd = BetaShepherdProtocol(
                threat_id=threat_id,
                beta_id=best_id,
                detecting_alpha_id=threat.detected_by if isinstance(threat.detected_by, int) else 0,
                hex_center=self._hex_center,
                hex_radius=self._hex_radius,
            )
            shepherd.start_guidance(
                initial_threat_pos=threat.position,
                initial_beta_pos=beta_pos or self._hex_center,
            )
            self._shepherds[threat_id] = shepherd

            logger.info("Beta %d dispatched to %s (dist=%.0fm) — shepherd started",
                         best_id, threat_id, best_dist)

        return best_id

    def beta_arrived(self, threat_id: str, beta_drone_id: int) -> bool:
        """
        Notify that a Beta drone has arrived at the threat location.
        
        Returns True if the Beta is on the correct threat.
        """
        threat = self._threats.get(threat_id)
        if threat is None:
            return False
        return threat.assigned_beta == beta_drone_id

    def confirm_threat(
        self,
        threat_id: str,
        is_confirmed: bool,
        current_time: Optional[float] = None,
    ) -> Optional[Threat]:
        """
        Record Beta drone's confirmation or clearing of a threat.
        
        Args:
            threat_id: Threat to confirm/clear
            is_confirmed: True if threat is real, False if false positive
            current_time: Current simulation time
            
        Returns:
            Updated Threat, or None if not found.
        """
        current_time = current_time or time.time()
        threat = self._threats.get(threat_id)
        if threat is None:
            return None

        if is_confirmed:
            threat.status = ThreatStatus.CONFIRMED
            threat.confidence = min(1.0, threat.confidence + 0.3)
            threat.confirmed_by = threat.assigned_beta
            logger.warning("THREAT CONFIRMED: %s [%s] at (%.0f, %.0f)",
                          threat_id, threat.threat_level.name,
                          threat.position.x, threat.position.y)
        else:
            threat.status = ThreatStatus.CLEARED
            threat.confidence *= 0.3
            logger.info("Threat CLEARED: %s (false positive)", threat_id)

        threat.confirmation_time = current_time

        # Stop shepherd guidance (spec §7.2)
        shepherd = self._shepherds.pop(threat_id, None)
        if shepherd and shepherd.is_active:
            shepherd.stop_guidance(confirmed=is_confirmed)

        return threat

    def resolve_threat(self, threat_id: str, current_time: Optional[float] = None):
        """Mark a threat as resolved."""
        current_time = current_time or time.time()
        threat = self._threats.get(threat_id)
        if threat:
            threat.status = ThreatStatus.RESOLVED
            threat.resolution_time = current_time
            logger.info("Threat resolved: %s", threat_id)

    def update(self, current_time: Optional[float] = None):
        """
        Update threat states — age out stale threats.
        
        Args:
            current_time: Current simulation time
        """
        current_time = current_time or time.time()

        for threat in list(self._threats.values()):
            # Age out old detected threats that never got confirmed
            if threat.status in (ThreatStatus.DETECTED, ThreatStatus.PENDING_CONFIRMATION):
                if current_time - threat.detection_time > self.threat_timeout:
                    threat.status = ThreatStatus.RESOLVED
                    threat.resolution_time = current_time
                    logger.info("Threat aged out: %s", threat.threat_id)

            # Age out cleared threats
            if threat.status == ThreatStatus.CLEARED:
                if threat.confirmation_time and current_time - threat.confirmation_time > 30.0:
                    threat.status = ThreatStatus.RESOLVED
                    threat.resolution_time = current_time

        # Prune resolved threats older than 60s to prevent unbounded growth
        stale_ids = [
            tid for tid, t in self._threats.items()
            if t.status == ThreatStatus.RESOLVED
            and getattr(t, "resolution_time", None)
            and current_time - t.resolution_time > 60.0
        ]
        for tid in stale_ids:
            del self._threats[tid]
        if stale_ids:
            self._object_to_threat = {
                k: v for k, v in self._object_to_threat.items()
                if v not in stale_ids
            }

    def get_active_threats(self) -> List[Threat]:
        """Get all threats that are not resolved."""
        return [
            t for t in self._threats.values()
            if t.status != ThreatStatus.RESOLVED
        ]

    def get_threats_needing_confirmation(self) -> List[Threat]:
        """Get threats that need Beta drone confirmation."""
        return [
            t for t in self._threats.values()
            if t.status == ThreatStatus.PENDING_CONFIRMATION
        ]

    def get_threat(self, threat_id: str) -> Optional[Threat]:
        """Get a specific threat."""
        return self._threats.get(threat_id)

    def get_all_threats(self) -> List[Threat]:
        """Get all threats including resolved."""
        return list(self._threats.values())

    def tick_shepherds(
        self,
        dt: float,
        threat_positions: Dict[str, Vector3],
        threat_velocities: Dict[str, Vector3],
        beta_positions: Dict[int, Vector3],
    ) -> Dict[int, Tuple[Vector3, float]]:
        """
        Advance all active shepherd protocols one time step.

        Args:
            dt: Simulation time step (s).
            threat_positions: threat_id → current threat world position.
            threat_velocities: threat_id → current threat velocity estimate.
            beta_positions: beta_drone_id → current Beta position.

        Returns:
            Dict of beta_drone_id → (target_position, target_speed) for each
            Beta currently under shepherd guidance.
        """
        targets: Dict[int, Tuple[Vector3, float]] = {}
        finished: List[str] = []

        for tid, shepherd in self._shepherds.items():
            if not shepherd.is_active:
                finished.append(tid)
                continue

            t_pos = threat_positions.get(tid, shepherd.target_position)
            t_vel = threat_velocities.get(tid, Vector3())
            b_pos = beta_positions.get(shepherd.beta_id)
            if b_pos is None:
                continue

            target, speed = shepherd.tick(dt, t_pos, t_vel, b_pos)
            targets[shepherd.beta_id] = (target, speed)

        for tid in finished:
            self._shepherds.pop(tid, None)

        return targets

    def get_active_shepherds(self) -> Dict[str, BetaShepherdProtocol]:
        """Get all active shepherd protocols."""
        return {tid: s for tid, s in self._shepherds.items() if s.is_active}

    def set_hex_center(self, center: Vector3):
        """Update the hex center (used for Beta RTL after confirmation)."""
        self._hex_center = center

    def set_hex_radius(self, radius: float):
        """Update the hex radius (used for Beta boundary enforcement)."""
        self._hex_radius = radius

    def has_active_threat_response(self) -> bool:
        """Return True if any shepherd is actively guiding Beta to a threat."""
        return bool(self._shepherds)

    def reset(self):
        """Reset all threats."""
        self._threats.clear()
        self._object_to_threat.clear()
        self._shepherds.clear()
        self._threat_counter = 0
