"""
Project Sanjay Mk2 - Simulated RGB Camera
==========================================
Simulates a visual camera sensor for drone surveillance.

Alpha drones carry a 4K wide-angle camera (84° FOV) at 65m altitude.
Beta drones carry a 1080p narrow-FOV camera (50° FOV) at 25m altitude.

Detection Model:
    - Detection probability decreases with altitude
    - Alpha: detects but may not identify (lower confidence)
    - Beta: high-detail identification (higher confidence)
    - Objects with larger size are easier to detect

Usage:
    camera = SimulatedRGBCamera(drone_type=DroneType.ALPHA)
    observation = camera.capture(drone_pos, altitude, world_model)
"""

from __future__ import annotations

import math
import random
import logging
from typing import List, Set, Tuple

from src.core.types.drone_types import (
    Vector3, DroneType, SensorType,
    DetectedObject, SensorObservation,
)
from src.surveillance.world_model import WorldModel, WorldObject

logger = logging.getLogger(__name__)


# Sensor profiles by drone type
CAMERA_PROFILES = {
    DroneType.ALPHA: {
        'fov_deg': 84.0,           # Wide-angle for area coverage
        'base_detection_prob': 0.75,
        'max_detection_range': 100.0,  # meters (at altitude)
        'base_confidence': 0.45,   # Lower confidence at 65m
        'size_factor': 0.05,       # How much object size helps detection
    },
    DroneType.BETA: {
        'fov_deg': 50.0,           # Narrower, higher detail
        'base_detection_prob': 0.95,
        'max_detection_range': 60.0,
        'base_confidence': 0.80,   # Higher confidence at 25m
        'size_factor': 0.08,
    },
}


class SimulatedRGBCamera:
    """
    Simulates an RGB camera sensor.
    
    Queries the world model from the drone's position and returns
    detected objects with realistic detection probabilities.
    """

    def __init__(self, drone_type: DroneType = DroneType.ALPHA):
        profile = CAMERA_PROFILES[drone_type]
        self.drone_type = drone_type
        self.fov_deg: float = profile['fov_deg']
        self.base_detection_prob: float = profile['base_detection_prob']
        self.max_detection_range: float = profile['max_detection_range']
        self.base_confidence: float = profile['base_confidence']
        self.size_factor: float = profile['size_factor']

    def capture(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int = 0,
    ) -> SensorObservation:
        """
        Capture a simulated RGB observation.
        
        Args:
            drone_position: Drone XY position
            altitude: Altitude AGL in meters
            world_model: The world to observe
            drone_id: ID of the observing drone
            
        Returns:
            SensorObservation with detected objects and coverage cells.
        """
        # Query world model for objects in FOV
        visible_objects, coverage_cells = world_model.query_fov(
            drone_position, altitude, self.fov_deg
        )

        # Apply detection model
        detected: List[DetectedObject] = []
        for obj in visible_objects:
            det_prob = self._detection_probability(obj, altitude)
            if random.random() < det_prob:
                confidence = self._calculate_confidence(obj, altitude)
                detected.append(DetectedObject(
                    object_id=obj.object_id,
                    object_type=obj.object_type if confidence > 0.6 else "unknown",
                    position=Vector3(x=obj.position.x, y=obj.position.y, z=obj.position.z),
                    confidence=confidence,
                    thermal_signature=0.0,  # RGB doesn't measure thermal
                    sensor_type=SensorType.RGB_CAMERA,
                ))

        return SensorObservation(
            sensor_type=SensorType.RGB_CAMERA,
            drone_id=drone_id,
            drone_position=Vector3(x=drone_position.x, y=drone_position.y, z=drone_position.z),
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    def _detection_probability(self, obj: WorldObject, altitude: float) -> float:
        """Calculate probability of detecting an object."""
        # Base probability, reduced by altitude
        alt_factor = max(0.0, 1.0 - altitude / self.max_detection_range)
        size_bonus = obj.size * self.size_factor
        prob = self.base_detection_prob * alt_factor + size_bonus
        return min(1.0, max(0.0, prob))

    def _calculate_confidence(self, obj: WorldObject, altitude: float) -> float:
        """Calculate identification confidence for a detected object."""
        alt_factor = max(0.0, 1.0 - altitude / self.max_detection_range)
        size_bonus = obj.size * self.size_factor * 0.5
        confidence = self.base_confidence * (0.5 + 0.5 * alt_factor) + size_bonus
        return min(1.0, max(0.1, confidence))

    def get_footprint_radius(self, altitude: float) -> float:
        """Get the ground footprint radius at a given altitude."""
        return altitude * math.tan(math.radians(self.fov_deg / 2.0))
