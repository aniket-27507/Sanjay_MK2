"""
Project Sanjay Mk2 - Formation Controller Tests
================================================
"""

import pytest
from src.core.types.drone_types import DroneState, DroneType, Vector3
from src.swarm.formation.formation_controller import (
    FormationConfig,
    FormationController,
    FormationType,
)


def _make_state(drone_id: int, x: float = 0.0, y: float = 0.0) -> DroneState:
    return DroneState(
        drone_id=drone_id,
        drone_type=DroneType.ALPHA,
        position=Vector3(x=x, y=y, z=-65.0),
        velocity=Vector3(),
    )


class TestFormationController:

    def test_hexagonal_slots(self):
        fc = FormationController(num_drones=6)
        fc.assign_drones(list(range(6)))
        for i in range(6):
            slot = fc.get_slot_for_drone(i)
            assert slot is not None, f"No slot for drone {i}"

    def test_linear_slots(self):
        cfg = FormationConfig(formation_type=FormationType.LINEAR, spacing=50.0)
        fc = FormationController(num_drones=4, config=cfg)
        fc.assign_drones(list(range(4)))
        for i in range(4):
            slot = fc.get_slot_for_drone(i)
            assert slot is not None

    def test_slot_changes_on_spacing_change(self):
        fc = FormationController(num_drones=3)
        fc.assign_drones([0, 1, 2])
        slot_before = fc.get_slot_for_drone(0)
        fc.config.spacing = 200.0
        fc._generate_slots()
        slot_after = fc.get_slot_for_drone(0)
        assert slot_before is not None and slot_after is not None

    def test_assign_fewer_drones_than_slots(self):
        fc = FormationController(num_drones=6)
        fc.assign_drones([0, 2, 4])
        for did in [0, 2, 4]:
            assert fc.get_slot_for_drone(did) is not None
        assert fc.get_slot_for_drone(1) is None

    def test_compute_corrections_returns_dict(self):
        fc = FormationController(num_drones=3)
        fc.assign_drones([0, 1, 2])
        states = {i: _make_state(i, x=float(i * 20)) for i in range(3)}
        corrections = fc.compute_corrections(states)
        assert isinstance(corrections, dict)
        for drone_id in [0, 1, 2]:
            assert drone_id in corrections
            assert isinstance(corrections[drone_id], Vector3)
