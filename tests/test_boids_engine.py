from src.core.types.drone_types import DroneState, Vector3
from src.swarm.boids import BoidsConfig, BoidsEngine


def _state(drone_id: int, pos: Vector3, vel: Vector3 = Vector3()) -> DroneState:
    return DroneState(drone_id=drone_id, position=pos, velocity=vel)


def test_separation_repels_close_neighbor():
    cfg = BoidsConfig(
        w_separation=1.0,
        w_alignment=0.0,
        w_cohesion=0.0,
        w_goal_seeking=0.0,
        w_obstacle_avoidance=0.0,
        w_formation_bias=0.0,
        w_energy_saving=0.0,
    )
    engine = BoidsEngine(cfg)

    states = {
        0: _state(0, Vector3(0, 0, 0)),
        1: _state(1, Vector3(10, 0, 0)),
    }
    v = engine.compute(drone_id=0, states=states, goal=None, obstacles=[])
    assert v.x < 0


def test_obstacle_repulsion_activates_inside_range():
    cfg = BoidsConfig(
        w_separation=0.0,
        w_alignment=0.0,
        w_cohesion=0.0,
        w_goal_seeking=0.0,
        w_obstacle_avoidance=1.0,
        w_formation_bias=0.0,
        w_energy_saving=0.0,
    )
    engine = BoidsEngine(cfg)

    states = {0: _state(0, Vector3(0, 0, 0))}
    obstacles = [{"position": [5.0, 0.0, 0.0], "radius": 1.0}]
    v = engine.compute(drone_id=0, states=states, goal=None, obstacles=obstacles)
    assert v.x < 0


def test_velocity_clamping_respects_limits():
    cfg = BoidsConfig(
        w_separation=0.0,
        w_alignment=0.0,
        w_cohesion=0.0,
        w_goal_seeking=10.0,
        w_obstacle_avoidance=0.0,
        w_formation_bias=0.0,
        w_energy_saving=0.0,
        max_speed=8.0,
        max_vertical_speed=3.0,
    )
    engine = BoidsEngine(cfg)

    states = {0: _state(0, Vector3(0, 0, 0))}
    v = engine.compute(
        drone_id=0,
        states=states,
        goal=Vector3(1000, 1000, 1000),
        obstacles=[],
    )
    assert v.magnitude() <= 8.0 + 1e-6
    assert abs(v.z) <= 3.0 + 1e-6


def test_energy_rule_biases_to_cruise_speed():
    cfg = BoidsConfig(
        w_separation=0.0,
        w_alignment=0.0,
        w_cohesion=0.0,
        w_goal_seeking=0.0,
        w_obstacle_avoidance=0.0,
        w_formation_bias=0.0,
        w_energy_saving=1.0,
        acceleration_penalty=0.0,
        speed_convergence_rate=0.3,
        cruise_speed=5.0,
    )
    engine = BoidsEngine(cfg)

    states = {0: _state(0, Vector3(0, 0, 0), Vector3(8, 0, 0))}
    v = engine.compute(drone_id=0, states=states, goal=Vector3(100, 0, 0), obstacles=[])
    assert v.x < 0  # above cruise speed => negative acceleration bias
