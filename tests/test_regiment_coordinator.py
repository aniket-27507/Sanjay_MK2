"""
Project Sanjay Mk2 - Regiment Coordinator Tests
================================================
"""

import pytest
import asyncio

from src.core.types.drone_types import DroneState, DroneType, Vector3
from src.swarm.coordination import AlphaRegimentCoordinator, RegimentConfig


def _make_state(drone_id: int, x: float = 0.0, y: float = 0.0) -> DroneState:
    return DroneState(
        drone_id=drone_id,
        drone_type=DroneType.ALPHA,
        position=Vector3(x=x, y=y, z=-65.0),
        velocity=Vector3(),
        battery=95.0,
    )


class TestAlphaRegimentCoordinator:

    @pytest.mark.asyncio
    async def test_init_and_register_drones(self):
        cfg = RegimentConfig(
            formation_spacing=80.0,
            formation_altitude=65.0,
            total_coverage_area=1000.0,
            use_boids_flocking=True,
        )
        coord = AlphaRegimentCoordinator(my_drone_id=0, config=cfg)
        await coord.initialize()
        for i in range(6):
            coord.register_drone(i)
        coord.update_member_state(0, _make_state(0, x=0, y=0))
        coord.update_member_state(1, _make_state(1, x=80, y=0))

    @pytest.mark.asyncio
    async def test_coordination_step_produces_velocity(self):
        cfg = RegimentConfig(
            formation_spacing=80.0,
            formation_altitude=65.0,
            total_coverage_area=1000.0,
            use_boids_flocking=True,
        )
        coord = AlphaRegimentCoordinator(my_drone_id=0, config=cfg)
        await coord.initialize()
        for i in range(3):
            coord.register_drone(i)
            coord.update_member_state(i, _make_state(i, x=float(i * 80)))

        coord.coordination_step()
        vel = coord.get_desired_velocity(0)
        assert isinstance(vel, Vector3)

    @pytest.mark.asyncio
    async def test_gossip_roundtrip(self):
        configs = [
            RegimentConfig(
                formation_spacing=80.0,
                formation_altitude=65.0,
                total_coverage_area=1000.0,
                use_boids_flocking=True,
            )
            for _ in range(3)
        ]
        coords = []
        for i in range(3):
            c = AlphaRegimentCoordinator(my_drone_id=i, config=configs[i])
            await c.initialize()
            for j in range(3):
                c.register_drone(j)
                c.update_member_state(j, _make_state(j, x=float(j * 80)))
            coords.append(c)

        payloads = [c.prepare_gossip_payload() for c in coords]
        for i, c in enumerate(coords):
            for j, p in enumerate(payloads):
                if i != j and p:
                    c.ingest_gossip_payload(p)

        for c in coords:
            c.coordination_step()

        for c in coords:
            vel = c.get_desired_velocity(c.my_drone_id)
            assert isinstance(vel, Vector3)
