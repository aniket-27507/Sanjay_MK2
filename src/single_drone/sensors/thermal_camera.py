"""
Project Sanjay Mk2 - Simulated Thermal Camera
===============================================
Simulates an LWIR (Long-Wave Infrared) thermal imaging sensor.

Detection Model:
    - Queries `WorldModel` against objects that emit heat profiles.
    - Calculates a thermal contrast vs. ambient environment curve.
    - Narrower FOV than RGB.

@author: Archishman Paul
"""

from __future__ import annotations

import math
import random
import logging
from typing import List

from src.core.types.drone_types import (
    Vector3, SensorType,
    DetectedObject, SensorObservation,
)
from src.surveillance.world_model import WorldModel, WorldObject

logger = logging.getLogger(__name__)


class SimulatedThermalCamera:
    """
    Simulates an LWIR thermal camera (640×512, 8-14μm).
    
    Detects objects by their thermal signature. Objects with higher
    thermal contrast against ambient are detected more reliably.
    """

    def __init__(
        self,
        fov_deg: float = 40.0,
        thermal_threshold: float = 0.3,
        max_detection_range: float = 120.0,
    ):
        """
        Args:
            fov_deg: Field of view in degrees (narrower than RGB)
            thermal_threshold: Minimum thermal signature to detect (0-1)
            max_detection_range: Maximum range for detection in meters
        """
        self.fov_deg = fov_deg
        self.thermal_threshold = thermal_threshold
        self.max_detection_range = max_detection_range

    def capture(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int = 0,
    ) -> SensorObservation:
        """
        Capture a simulated thermal observation.
        
        Args:
            drone_position: Drone XY position
            altitude: Altitude AGL in meters
            world_model: The world to observe
            drone_id: ID of the observing drone
            
        Returns:
            SensorObservation with thermally-detected objects.
        """
        # Query world for objects with thermal signatures
        thermal_objects = world_model.query_thermal(
            drone_position, altitude, self.fov_deg, self.thermal_threshold
        )

        # Get coverage cells
        _, coverage_cells = world_model.query_fov(
            drone_position, altitude, self.fov_deg
        )

        # Apply thermal detection model
        detected: List[DetectedObject] = []
        for obj in thermal_objects:
            det_prob = self._thermal_detection_probability(obj, altitude)
            if random.random() < det_prob:
                # Thermal cameras detect presence but can't identify type well
                confidence = self._thermal_confidence(obj, altitude)
                detected.append(DetectedObject(
                    object_id=obj.object_id,
                    object_type="thermal_contact",  # Thermal can't classify type
                    position=Vector3(x=obj.position.x, y=obj.position.y, z=obj.position.z),
                    confidence=confidence,
                    thermal_signature=obj.thermal_signature,
                    sensor_type=SensorType.THERMAL_CAMERA,
                ))

        return SensorObservation(
            sensor_type=SensorType.THERMAL_CAMERA,
            drone_id=drone_id,
            drone_position=Vector3(x=drone_position.x, y=drone_position.y, z=drone_position.z),
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    def _thermal_detection_probability(self, obj: WorldObject, altitude: float) -> float:
        """
        Detection probability based on thermal contrast and range.
        
        Higher thermal signature = easier to detect.
        """
        # Thermal contrast factor (higher signature = much easier to detect)
        thermal_factor = obj.thermal_signature ** 0.5  # Square root for softer curve

        # Range factor
        range_factor = max(0.0, 1.0 - altitude / self.max_detection_range)

        prob = 0.9 * thermal_factor * (0.3 + 0.7 * range_factor)
        return min(1.0, max(0.0, prob))

    def _thermal_confidence(self, obj: WorldObject, altitude: float) -> float:
        """
        Confidence in thermal detection.
        
        Thermal gives moderate confidence — confirms something warm is there,
        but can't identify what it is.
        """
        thermal_factor = min(1.0, obj.thermal_signature * 1.2)
        range_factor = max(0.3, 1.0 - altitude / self.max_detection_range)
        confidence = 0.5 * thermal_factor * range_factor + 0.2
        return min(0.85, max(0.2, confidence))

    def get_footprint_radius(self, altitude: float) -> float:
        """Get the ground footprint radius at a given altitude."""
        return altitude * math.tan(math.radians(self.fov_deg / 2.0))
