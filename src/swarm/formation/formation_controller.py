"""
Project Sanjay Mk2 - Formation Controller
==========================================
Maintains geographic formation geometry for the Alpha Regiment.

Handles real-time formation keeping: given a desired formation
pattern and the current positions of all drones, computes the
velocity correction each drone should apply to converge on its
assigned slot while respecting:
    - Minimum inter-drone distance (anti-collision)
    - Maximum convergence speed
    - Smooth deceleration near target slot

This module works cooperatively with the AvoidanceManager —
formation corrections are treated as an additive velocity bias
that the APF can override if obstacles are detected.

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import DroneState, Vector3

logger = logging.getLogger(__name__)


class FormationType(Enum):
    """Available formation geometries."""
    HEXAGONAL = auto()
    LINEAR = auto()
    WEDGE = auto()
    RING = auto()
    DIAMOND = auto()
    CUSTOM = auto()


@dataclass
class FormationSlot:
    """A slot within a formation pattern."""
    slot_id: int
    offset: Vector3              # Offset from formation center
    assigned_drone_id: int = -1  # -1 = unassigned


@dataclass
class FormationConfig:
    """Configuration for formation keeping."""
    formation_type: FormationType = FormationType.HEXAGONAL
    spacing: float = 80.0                # Inter-drone spacing (m)
    altitude: float = 65.0               # Formation altitude (m)
    convergence_gain: float = 0.7        # P-gain for position correction
    max_correction_speed: float = 4.0    # Max formation correction speed (m/s)
    deceleration_radius: float = 10.0    # Start slowing within this radius (m)
    min_separation: float = 50.0         # Anti-collision minimum distance (m)
    separation_gain: float = 2.0         # Repulsive gain for anti-collision
    center_altitude_offset: float = 0.0  # Z-offset for center slot (e.g., Beta at different alt)


class FormationController:
    """
    Manages formation geometry and slot assignment for the Alpha Regiment.

    Usage:
        fc = FormationController(num_drones=6)
        fc.set_center(Vector3(500, 500, -65))

        # Each tick, get formation correction velocity for each drone
        corrections = fc.compute_corrections(drone_states)
        my_correction = corrections.get(my_drone_id, Vector3())
    """

    def __init__(
        self,
        num_drones: int = 6,
        config: Optional[FormationConfig] = None,
    ):
        self.config = config or FormationConfig()
        self._num_drones = num_drones
        self._center = Vector3(x=500.0, y=500.0, z=-self.config.altitude)
        self._heading: float = 0.0  # Formation heading (rad)
        self._slots: List[FormationSlot] = []

        self._generate_slots()

    def set_center(self, center: Vector3):
        """Set the formation center position."""
        self._center = center

    def set_heading(self, heading_rad: float):
        """Set the formation heading (rotation)."""
        self._heading = heading_rad
        self._generate_slots()

    def assign_drones(self, drone_ids: List[int]):
        """Assign drones to formation slots (nearest-first)."""
        for i, slot in enumerate(self._slots):
            if i < len(drone_ids):
                slot.assigned_drone_id = drone_ids[i]

    def compute_corrections(
        self,
        drone_states: Dict[int, DroneState],
    ) -> Dict[int, Vector3]:
        """
        Compute formation-keeping velocity corrections for all drones.

        Args:
            drone_states: Map of drone_id → DroneState.

        Returns:
            Map of drone_id → velocity correction (NED).
        """
        corrections: Dict[int, Vector3] = {}

        for slot in self._slots:
            drone_id = slot.assigned_drone_id
            if drone_id < 0 or drone_id not in drone_states:
                continue

            state = drone_states[drone_id]
            target = self._slot_world_position(slot)

            # Position error
            error = target - state.position
            dist = error.magnitude()

            if dist < 1.0:
                corrections[drone_id] = Vector3()
                continue

            # P-control with deceleration
            direction = error.normalized()
            if dist < self.config.deceleration_radius:
                speed = self.config.max_correction_speed * (
                    dist / self.config.deceleration_radius
                )
            else:
                speed = self.config.max_correction_speed

            correction = direction * (speed * self.config.convergence_gain)

            # Anti-collision separation from other drones
            separation = self._compute_separation(
                drone_id, state.position, drone_states
            )
            correction = correction + separation

            # Clamp
            mag = correction.magnitude()
            if mag > self.config.max_correction_speed:
                correction = correction * (self.config.max_correction_speed / mag)

            corrections[drone_id] = correction

        return corrections

    def get_slot_positions(self) -> List[Vector3]:
        """Get world positions of all formation slots."""
        return [self._slot_world_position(s) for s in self._slots]

    def get_slot_for_drone(self, drone_id: int) -> Optional[Vector3]:
        """Get world slot position for a specific drone, if assigned."""
        for slot in self._slots:
            if slot.assigned_drone_id == drone_id:
                return self._slot_world_position(slot)
        return None

    # ── Formation Generation ──────────────────────────────────────

    def _generate_slots(self):
        """Generate slot offsets based on formation type."""
        n = self._num_drones
        spacing = self.config.spacing
        ft = self.config.formation_type

        offsets: List[Vector3] = []

        if ft == FormationType.HEXAGONAL:
            offsets = self._hex_offsets(n, spacing)
        elif ft == FormationType.LINEAR:
            offsets = self._linear_offsets(n, spacing)
        elif ft == FormationType.WEDGE:
            offsets = self._wedge_offsets(n, spacing)
        elif ft == FormationType.RING:
            offsets = self._ring_offsets(n, spacing)
        elif ft == FormationType.DIAMOND:
            offsets = self._diamond_offsets(n, spacing)
        else:
            offsets = self._hex_offsets(n, spacing)

        # Rotate by heading
        cos_h = math.cos(self._heading)
        sin_h = math.sin(self._heading)

        self._slots = []
        for i, offset in enumerate(offsets):
            rotated = Vector3(
                x=offset.x * cos_h - offset.y * sin_h,
                y=offset.x * sin_h + offset.y * cos_h,
                z=offset.z,
            )
            self._slots.append(FormationSlot(
                slot_id=i,
                offset=rotated,
                assigned_drone_id=i if i < self._num_drones else -1,
            ))

    def _hex_offsets(self, n: int, spacing: float) -> List[Vector3]:
        offsets: List[Vector3] = [Vector3(0, 0, self.config.center_altitude_offset)]  # Center drone (Beta)
        for i in range(min(n - 1, 6)):
            # Start at top vertex (90 deg) to align with Alpha_0 slot.
            angle = (math.pi / 2) + i * (2 * math.pi / 6)
            offsets.append(Vector3(
                x=spacing * math.cos(angle),
                y=spacing * math.sin(angle),
                z=0,
            ))
        return offsets[:n]

    def _linear_offsets(self, n: int, spacing: float) -> List[Vector3]:
        total = (n - 1) * spacing
        return [
            Vector3(x=0, y=-total / 2 + i * spacing, z=0)
            for i in range(n)
        ]

    def _wedge_offsets(self, n: int, spacing: float) -> List[Vector3]:
        offsets = [Vector3(0, 0, 0)]
        for i in range(1, n):
            side = 1 if i % 2 == 1 else -1
            row = (i + 1) // 2
            offsets.append(Vector3(
                x=-row * spacing * 0.7,
                y=side * row * spacing * 0.5,
                z=0,
            ))
        return offsets

    def _ring_offsets(self, n: int, spacing: float) -> List[Vector3]:
        radius = spacing * n / (2 * math.pi)
        return [
            Vector3(
                x=radius * math.cos(i * 2 * math.pi / n),
                y=radius * math.sin(i * 2 * math.pi / n),
                z=0,
            )
            for i in range(n)
        ]

    def _diamond_offsets(self, n: int, spacing: float) -> List[Vector3]:
        offsets = [Vector3(spacing, 0, 0)]  # Lead
        offsets.append(Vector3(0, -spacing / 2, 0))
        offsets.append(Vector3(0, spacing / 2, 0))
        offsets.append(Vector3(-spacing, 0, 0))
        offsets.append(Vector3(-spacing / 2, -spacing, 0))
        offsets.append(Vector3(-spacing / 2, spacing, 0))
        return offsets[:n]

    # ── Helpers ───────────────────────────────────────────────────

    def _slot_world_position(self, slot: FormationSlot) -> Vector3:
        """Compute world position for a formation slot."""
        return self._center + slot.offset

    def _compute_separation(
        self,
        my_id: int,
        my_pos: Vector3,
        states: Dict[int, DroneState],
    ) -> Vector3:
        """Compute anti-collision separation force."""
        total = Vector3()

        for did, state in states.items():
            if did == my_id:
                continue

            to_me = my_pos - state.position
            dist = to_me.magnitude()

            if dist < self.config.min_separation and dist > 0.1:
                strength = self.config.separation_gain * (
                    (self.config.min_separation - dist) / self.config.min_separation
                )
                total = total + to_me.normalized() * strength

        return total
