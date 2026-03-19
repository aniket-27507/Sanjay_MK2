"""
Project Sanjay Mk2 — Sensor Degradation Model
===============================================
Wraps existing sensor classes with fault injection modes
for degraded operation scenarios.

Modes:
    normal:       passthrough (no modification)
    noisy:        inject N random false detections per capture
    intermittent: skip captures with probability P
    failed:       return empty observations

@author: Claude Code
"""

from __future__ import annotations

import random
import logging
from typing import Optional

from src.core.types.drone_types import (
    DetectedObject, SensorObservation, SensorType, Vector3,
)
from src.surveillance.world_model import WorldModel

logger = logging.getLogger(__name__)


class DegradedSensorWrapper:
    """Wraps a sensor (RGB or thermal camera) with degradation modes."""

    def __init__(self, inner_sensor, mode: str = "normal", **kwargs):
        """
        Args:
            inner_sensor: The actual sensor instance (SimulatedRGBCamera or SimulatedThermalCamera)
            mode: "normal", "noisy", "intermittent", "failed"
            kwargs:
                noise_count: (int) number of false detections per capture (noisy mode)
                skip_prob: (float) probability of skipping a capture (intermittent mode)
        """
        self._inner = inner_sensor
        self._mode = mode
        self._noise_count = kwargs.get("noise_count", 3)
        self._skip_prob = kwargs.get("skip_prob", 0.3)
        self._capture_count = 0

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        self._mode = value
        logger.info("Sensor degradation mode set to: %s", value)

    def capture(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int = 0,
    ) -> SensorObservation:
        """Capture with degradation applied."""
        self._capture_count += 1

        if self._mode == "failed":
            # Return empty observation
            return SensorObservation(
                sensor_type=getattr(self._inner, "sensor_type", SensorType.RGB_CAMERA),
                drone_id=drone_id,
                drone_position=drone_position,
                drone_altitude=altitude,
                detected_objects=[],
                coverage_cells=[],
            )

        if self._mode == "intermittent":
            if random.random() < self._skip_prob:
                return SensorObservation(
                    sensor_type=getattr(self._inner, "sensor_type", SensorType.RGB_CAMERA),
                    drone_id=drone_id,
                    drone_position=drone_position,
                    drone_altitude=altitude,
                    detected_objects=[],
                    coverage_cells=[],
                )

        # Normal capture from inner sensor
        obs = self._inner.capture(drone_position, altitude, world_model, drone_id)

        if self._mode == "noisy":
            # Inject random false detections
            for i in range(self._noise_count):
                fake_x = drone_position.x + random.uniform(-50, 50)
                fake_y = drone_position.y + random.uniform(-50, 50)
                obs.detected_objects.append(DetectedObject(
                    object_id=f"noise_{drone_id}_{self._capture_count}_{i}",
                    object_type="unknown",
                    position=Vector3(fake_x, fake_y, 0),
                    confidence=random.uniform(0.15, 0.40),
                    thermal_signature=random.uniform(0.1, 0.3),
                    sensor_type=SensorType.RGB_CAMERA,
                ))

        return obs
