"""
Project Sanjay Mk2 - Sensor Fusion Pipeline
=============================================
Fuses observations from RGB and thermal cameras into unified detections
with boosted confidence.

Confidence boosting rules:
    - RGB-only detection:              0.3 - 0.6
    - RGB + thermal corroboration:     0.6 - 0.85

Implements temporal buffering to cross reference proximal sensors.

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from src.core.types.drone_types import (
    Vector3, SensorType,
    DetectedObject, SensorObservation, FusedObservation,
)

logger = logging.getLogger(__name__)

# Distance threshold for matching objects across sensors (meters)
CROSS_SENSOR_MATCH_RADIUS = 15.0

# Confidence boost factors
THERMAL_CORROBORATION_BOOST = 0.25


class SensorFusionPipeline:
    """
    Fuses multi-sensor observations into unified detections.
    
    Collects observations from RGB and thermal sensors, then
    cross-references detections to boost confidence and enrich
    object data.
    """

    def __init__(self, match_radius: float = CROSS_SENSOR_MATCH_RADIUS):
        """
        Args:
            match_radius: Max distance (m) to consider two detections
                         from different sensors as the same object.
        """
        self.match_radius = match_radius
        self._pending_observations: List[SensorObservation] = []

    def add_observation(self, observation: SensorObservation):
        """Add a sensor observation to the pending fusion buffer."""
        self._pending_observations.append(observation)

    def fuse(self) -> Optional[FusedObservation]:
        """
        Fuse all pending observations into a single FusedObservation.
        
        Returns:
            FusedObservation with cross-referenced detections, or None
            if no observations are pending.
        """
        if not self._pending_observations:
            return None

        # Separate by sensor type
        rgb_obs = [o for o in self._pending_observations if o.sensor_type == SensorType.RGB_CAMERA]
        thermal_obs = [o for o in self._pending_observations if o.sensor_type == SensorType.THERMAL_CAMERA]

        # Collect all RGB detections as the primary set
        primary_detections: List[DetectedObject] = []
        for obs in rgb_obs:
            primary_detections.extend(obs.detected_objects)

        # Collect thermal detections
        thermal_detections: List[DetectedObject] = []
        for obs in thermal_obs:
            thermal_detections.extend(obs.detected_objects)

        # Cross-reference: boost RGB detections that have thermal corroboration
        fused_objects: List[DetectedObject] = []
        matched_thermal_ids = set()

        for det in primary_detections:
            thermal_match = self._find_thermal_match(det, thermal_detections)
            boosted_confidence = det.confidence

            if thermal_match is not None:
                # Thermal corroboration found — boost confidence
                boosted_confidence = min(0.95, det.confidence + THERMAL_CORROBORATION_BOOST)
                matched_thermal_ids.add(thermal_match.object_id)
                thermal_sig = thermal_match.thermal_signature
            else:
                thermal_sig = 0.0

            fused_objects.append(DetectedObject(
                object_id=det.object_id,
                object_type=det.object_type,
                position=det.position,
                confidence=boosted_confidence,
                thermal_signature=thermal_sig,
                sensor_type=SensorType.RGB_CAMERA,  # Primary sensor
            ))

        # Add thermal-only detections (not matched to any RGB detection)
        for t_det in thermal_detections:
            if t_det.object_id not in matched_thermal_ids:
                fused_objects.append(DetectedObject(
                    object_id=t_det.object_id,
                    object_type="thermal_only",
                    position=t_det.position,
                    confidence=t_det.confidence * 0.8,  # Lower confidence without RGB
                    thermal_signature=t_det.thermal_signature,
                    sensor_type=SensorType.THERMAL_CAMERA,
                ))

        # Merge coverage cells
        all_coverage = set()
        for obs in self._pending_observations:
            all_coverage.update(obs.coverage_cells)

        # Get reference position from first observation
        ref_obs = self._pending_observations[0]

        sensor_count = len(set(o.sensor_type for o in self._pending_observations))

        # Clear buffer
        self._pending_observations.clear()

        return FusedObservation(
            drone_id=ref_obs.drone_id,
            position=ref_obs.drone_position,
            detected_objects=fused_objects,
            coverage_cells=list(all_coverage),
            sensor_count=sensor_count,
        )

    def _find_thermal_match(
        self,
        rgb_detection: DetectedObject,
        thermal_detections: List[DetectedObject],
    ) -> Optional[DetectedObject]:
        """
        Find a thermal detection that matches an RGB detection by proximity.
        """
        best_match = None
        best_dist = self.match_radius

        for t_det in thermal_detections:
            dist = rgb_detection.position.distance_to(t_det.position)
            if dist < best_dist:
                best_dist = dist
                best_match = t_det

        return best_match

    def clear(self):
        """Clear pending observations."""
        self._pending_observations.clear()
