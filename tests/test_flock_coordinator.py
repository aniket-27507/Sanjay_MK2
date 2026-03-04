from dataclasses import dataclass

from src.core.types.drone_types import DroneState, Vector3
from src.swarm.flock_coordinator import FlockCoordinator


@dataclass
class SectorStub:
    drone_id: int
    center: Vector3
    radius: float


def test_tick_returns_bounded_velocity_and_goal():
    flock = FlockCoordinator(drone_id=0, num_drones=3)
    flock.update_membership([0, 1, 2])

    my_state = DroneState(drone_id=0, position=Vector3(0, 0, -65), battery=100.0)
    peer_states = {
        1: DroneState(drone_id=1, position=Vector3(30, 0, -65), battery=100.0),
        2: DroneState(drone_id=2, position=Vector3(-30, 0, -65), battery=100.0),
    }
    sectors = [
        SectorStub(0, Vector3(100, 0, -65), 60.0),
        SectorStub(1, Vector3(120, 50, -65), 60.0),
        SectorStub(2, Vector3(120, -50, -65), 60.0),
    ]

    v = flock.tick(
        my_state=my_state,
        peer_states=peer_states,
        obstacles=[],
        sector_assignments=sectors,
        home_position=Vector3(0, 0, -65),
    )

    assert v.magnitude() <= flock.boids.config.max_speed + 1e-6
    assert flock.current_goal is not None


def test_gossip_consensus_and_reassignment_after_member_loss():
    f0 = FlockCoordinator(drone_id=0, num_drones=2)
    f1 = FlockCoordinator(drone_id=1, num_drones=2)

    f0.update_membership([0, 1])
    f1.update_membership([0, 1])

    sectors = [
        SectorStub(0, Vector3(100, 0, -65), 60.0),
        SectorStub(1, Vector3(-100, 0, -65), 60.0),
    ]

    s0 = DroneState(drone_id=0, position=Vector3(95, 0, -65), battery=100.0)
    s1 = DroneState(drone_id=1, position=Vector3(-95, 0, -65), battery=100.0)

    # Initial bidding + gossip.
    f0.tick(s0, {1: s1}, obstacles=[], sector_assignments=sectors)
    f1.tick(s1, {0: s0}, obstacles=[], sector_assignments=sectors)
    f0.ingest_gossip_payload(1, f1.prepare_gossip_payload(s1))
    f1.ingest_gossip_payload(0, f0.prepare_gossip_payload(s0))

    # Drone 0 drops out; drone 1 should reclaim released tasks quickly.
    f1.update_membership([1])
    for _ in range(2):
        f1.tick(s1, {}, obstacles=[], sector_assignments=sectors)

    current = f1.cbba.get_current_task()
    assert current is not None
    assert current.assigned_to == 1
