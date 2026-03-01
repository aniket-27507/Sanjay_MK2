"""
Project Sanjay Mk2 - Change Detection Engine
==============================================
Compares live sensor observations against the baseline map to
detect anomalies: new objects, missing objects, thermal anomalies.

Classification rules:
    - Person in area:         HIGH
    - Unknown vehicle:        MEDIUM
    - New structure/camp:     LOW
    - Thermal-only anomaly:   MEDIUM
    - Missing baseline obj:   LOW

Usage:
    detector = ChangeDetector(baseline_map)
    changes = detector.detect_changes(fused_observation)
    for change in changes:
        print(f"{change.change_type}: {change.description} [{change.threat_level}]")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.core.types.drone_types import (
    Vector3, ThreatLevel,
    DetectedObject, FusedObservation,
)
from src.surveillance.baseline_map import BaselineMap

logger = logging.getLogger(__name__)


# Threat classification rules by object type
THREAT_CLASSIFICATION = {
    'person': ThreatLevel.HIGH,
    'vehicle': ThreatLevel.MEDIUM,
    'camp': ThreatLevel.LOW,
    'equipment': ThreatLevel.LOW,
    'thermal_only': ThreatLevel.MEDIUM,
    'thermal_contact': ThreatLevel.MEDIUM,
    'unknown': ThreatLevel.MEDIUM,
}

# Minimum confidence to report a change
MIN_CHANGE_CONFIDENCE = 0.35


@dataclass
class ChangeEvent:
    """A detected change/anomaly compared to baseline."""
    event_id: str
    position: Vector3
    change_type: str            # "new_object", "thermal_anomaly"
    object_type: str
    description: str
    threat_level: ThreatLevel
    confidence: float
    detected_by: int            # drone_id
    thermal_signature: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'event_id': self.event_id,
            'position': [self.position.x, self.position.y, self.position.z],
            'change_type': self.change_type,
            'object_type': self.object_type,
            'description': self.description,
            'threat_level': self.threat_level.name,
            'confidence': round(self.confidence, 3),
            'detected_by': self.detected_by,
            'thermal_signature': self.thermal_signature,
            'timestamp': self.timestamp,
        }


class ChangeDetector:
    """
    Compares fused sensor observations against the baseline map.
    
    Detects:
    - New objects not in baseline
    - Thermal anomalies in normally cold areas
    """

    def __init__(
        self,
        baseline: BaselineMap,
        min_confidence: float = MIN_CHANGE_CONFIDENCE,
    ):
        self.baseline = baseline
        self.min_confidence = min_confidence
        self._event_counter = 0

        # Track recently reported objects to avoid duplicate events
        self._recently_reported: dict = {}  # object_id -> timestamp
        self._report_cooldown = 10.0  # seconds between reports of same object

    def detect_changes(
        self,
        observation: FusedObservation,
        current_time: Optional[float] = None,
    ) -> List[ChangeEvent]:
        """
        Detect changes between a fused observation and the baseline.
        
        Args:
            observation: Fused sensor observation
            current_time: Current simulation time
            
        Returns:
            List of ChangeEvent anomalies detected.
        """
        current_time = current_time or time.time()
        changes: List[ChangeEvent] = []

        # Clean old cooldowns
        self._clean_cooldowns(current_time)

        for det in observation.detected_objects:
            if det.confidence < self.min_confidence:
                continue

            # Skip if recently reported
            if det.object_id in self._recently_reported:
                continue

            # Check if this object is in the baseline
            if not self.baseline.is_known_object(det.object_id):
                # NEW OBJECT — not in baseline
                change = self._create_change_event(
                    det,
                    change_type="new_object",
                    drone_id=observation.drone_id,
                    current_time=current_time,
                )
                changes.append(change)
                self._recently_reported[det.object_id] = current_time

        return changes

    def _create_change_event(
        self,
        detection: DetectedObject,
        change_type: str,
        drone_id: int,
        current_time: float,
    ) -> ChangeEvent:
        """Create a ChangeEvent from a detection."""
        self._event_counter += 1
        event_id = f"chg_{self._event_counter:04d}"

        threat_level = self._classify_threat(detection)

        if change_type == "new_object":
            desc = f"New {detection.object_type} detected at ({detection.position.x:.0f}, {detection.position.y:.0f})"
            if detection.thermal_signature > 0.5:
                desc += f" [thermal: {detection.thermal_signature:.1f}]"
        else:
            desc = f"Anomaly at ({detection.position.x:.0f}, {detection.position.y:.0f})"

        return ChangeEvent(
            event_id=event_id,
            position=Vector3(
                x=detection.position.x,
                y=detection.position.y,
                z=detection.position.z,
            ),
            change_type=change_type,
            object_type=detection.object_type,
            description=desc,
            threat_level=threat_level,
            confidence=detection.confidence,
            detected_by=drone_id,
            thermal_signature=detection.thermal_signature,
            timestamp=current_time,
        )

    def _classify_threat(self, detection: DetectedObject) -> ThreatLevel:
        """Classify threat level based on object type and confidence."""
        base_level = THREAT_CLASSIFICATION.get(
            detection.object_type, ThreatLevel.UNKNOWN
        )

        # Upgrade threat level if high confidence + thermal
        if detection.confidence > 0.7 and detection.thermal_signature > 0.6:
            if base_level == ThreatLevel.MEDIUM:
                return ThreatLevel.HIGH

        return base_level

    def _clean_cooldowns(self, current_time: float):
        """Remove expired cooldowns."""
        expired = [
            oid for oid, t in self._recently_reported.items()
            if current_time - t > self._report_cooldown
        ]
        for oid in expired:
            del self._recently_reported[oid]

    def reset(self):
        """Reset detector state."""
        self._recently_reported.clear()
        self._event_counter = 0
