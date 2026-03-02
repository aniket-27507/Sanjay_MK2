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
)
from src.surveillance.change_detection import ChangeEvent

logger = logging.getLogger(__name__)

# Confidence threshold to request Beta confirmation
CONFIRMATION_THRESHOLD = 0.50

# Time before a threat ages out (seconds)
THREAT_TIMEOUT = 120.0

# Distance threshold for Beta to be "on scene"
BETA_ARRIVAL_RADIUS = 15.0


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
    ):
        self.confirmation_threshold = confirmation_threshold
        self.threat_timeout = threat_timeout

        self._threats: Dict[str, Threat] = {}
        self._threat_counter = 0

        # Map: change_event object_id -> threat_id (prevent duplicate threats)
        self._object_to_threat: Dict[str, str] = {}

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

        status = ThreatStatus.DETECTED
        if event.confidence >= self.confirmation_threshold:
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
            logger.info("Beta %d dispatched to %s (dist=%.0fm)",
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

    def reset(self):
        """Reset all threats."""
        self._threats.clear()
        self._object_to_threat.clear()
        self._threat_counter = 0
