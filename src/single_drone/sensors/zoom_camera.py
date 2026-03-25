"""
Project Sanjay Mk2 - Simulated Zoom EO Camera
=============================================
Simulates the narrow-FOV electro-optical confirmation camera carried by
each Alpha for close inspection and facade scanning.
"""

from __future__ import annotations

import math
import random
from typing import List

from src.core.types.drone_types import (
    DetectedObject,
    SensorObservation,
    SensorType,
    Vector3,
)
from src.surveillance.world_model import WorldModel, WorldObject


class SimulatedZoomEOCamera:
    """Narrow-FOV EO confirmation camera with higher close-range confidence."""

    def __init__(
        self,
        fov_deg: float = 24.0,
        base_detection_prob: float = 0.95,
        base_confidence: float = 0.82,
        max_detection_range: float = 45.0,
    ):
        self.fov_deg = fov_deg
        self.base_detection_prob = base_detection_prob
        self.base_confidence = base_confidence
        self.max_detection_range = max_detection_range

    def capture(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int = 0,
    ) -> SensorObservation:
        visible_objects, coverage_cells = world_model.query_fov(
            drone_position,
            altitude,
            self.fov_deg,
        )

        detected: List[DetectedObject] = []
        for obj in visible_objects:
            det_prob = self._detection_probability(obj, drone_position)
            if random.random() < det_prob:
                confidence = self._confidence(obj, drone_position)
                detected.append(
                    DetectedObject(
                        object_id=obj.object_id,
                        object_type=obj.object_type,
                        position=Vector3(
                            x=obj.position.x,
                            y=obj.position.y,
                            z=obj.position.z,
                        ),
                        confidence=confidence,
                        thermal_signature=obj.thermal_signature,
                        sensor_type=SensorType.ZOOM_EO_CAMERA,
                    )
                )

        return SensorObservation(
            sensor_type=SensorType.ZOOM_EO_CAMERA,
            drone_id=drone_id,
            drone_position=Vector3(
                x=drone_position.x,
                y=drone_position.y,
                z=drone_position.z,
            ),
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    def get_footprint_radius(self, altitude: float) -> float:
        return altitude * math.tan(math.radians(self.fov_deg / 2.0))

    def _detection_probability(self, obj: WorldObject, drone_position: Vector3) -> float:
        distance = max(1.0, drone_position.distance_to(obj.position))
        range_factor = max(0.0, 1.0 - distance / self.max_detection_range)
        type_bonus = 0.12 if obj.object_type in {"weapon_person", "person", "vehicle"} else 0.0
        return min(1.0, self.base_detection_prob * (0.45 + 0.55 * range_factor) + type_bonus)

    def _confidence(self, obj: WorldObject, drone_position: Vector3) -> float:
        distance = max(1.0, drone_position.distance_to(obj.position))
        range_factor = max(0.25, 1.0 - distance / self.max_detection_range)
        return min(0.99, self.base_confidence * (0.5 + 0.5 * range_factor))
