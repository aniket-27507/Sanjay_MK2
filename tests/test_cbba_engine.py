from src.core.types.drone_types import DroneState, Vector3
from src.swarm.cbba import CBBAConfig, CBBAEngine, SwarmTask, TaskType


def test_battery_infeasible_task_rejected():
    engine = CBBAEngine(drone_id=0, config=CBBAConfig())
    state = DroneState(drone_id=0, position=Vector3(0, 0, 0), battery=16.0)
    task = SwarmTask(
        task_id="far",
        task_type=TaskType.SECTOR_COVERAGE,
        position=Vector3(500, 0, 0),
        radius=30,
        estimated_energy=10.0,
    )

    score = engine.score_task(state, task, current_bundle=[])
    assert score < 0.0


def test_tie_break_prefers_lower_drone_id_and_prunes_bundle_suffix():
    engine = CBBAEngine(drone_id=2)
    state = DroneState(drone_id=2, position=Vector3(0, 0, 0), battery=100.0)

    t1 = SwarmTask("t1", TaskType.SECTOR_COVERAGE, Vector3(20, 0, 0), 20)
    t2 = SwarmTask("t2", TaskType.SECTOR_COVERAGE, Vector3(25, 0, 0), 20)
    engine.upsert_tasks([t1, t2])
    engine.bundle_phase(state)

    # Remote agent outbids/tie-breaks on first task.
    first = engine.get_bundle_ids()[0]
    engine.consensus_phase(
        remote_bids={first: engine.winning_bids[first]},
        remote_agents={first: 1},
        remote_id=1,
        remote_timestamps={first: engine.bid_timestamps[first] + 0.1},
    )

    assert engine.winning_agents[first] == 1
    assert first not in engine.get_bundle_ids()


def test_cbba_converges_across_six_drones_with_gossip_rounds():
    tasks = [
        SwarmTask(f"task_{i}", TaskType.SECTOR_COVERAGE, Vector3(i * 30.0, 0.0, 0.0), 20)
        for i in range(4)
    ]
    engines = [CBBAEngine(drone_id=i, config=CBBAConfig(max_bundle_size=2)) for i in range(6)]
    states = [
        DroneState(drone_id=i, position=Vector3(i * 25.0, 0.0, 0.0), battery=100.0)
        for i in range(6)
    ]

    for engine in engines:
        engine.upsert_tasks(tasks)

    for _ in range(10):
        for engine, state in zip(engines, states):
            engine.bundle_phase(state)

        payloads = [engine.get_bids_payload() for engine in engines]
        for i, receiver in enumerate(engines):
            for j, payload in enumerate(payloads):
                if i == j:
                    continue
                receiver.ingest_remote_payload(j, payload)

    baseline = engines[0].winning_agents
    for engine in engines[1:]:
        assert engine.winning_agents == baseline


def test_rtl_task_stays_assigned_to_target_drone():
    rtl = SwarmTask(
        task_id="rtl_0",
        task_type=TaskType.RTL,
        position=Vector3(0, 0, 0),
        radius=5,
        priority=10.0,
        assigned_to=0,
    )

    e0 = CBBAEngine(drone_id=0)
    e1 = CBBAEngine(drone_id=1)
    e0.upsert_task(rtl)
    e1.upsert_task(rtl)

    s0 = DroneState(drone_id=0, position=Vector3(50, 0, 0), battery=50.0)
    s1 = DroneState(drone_id=1, position=Vector3(10, 0, 0), battery=50.0)
    e0.bundle_phase(s0)
    e1.bundle_phase(s1)

    assert "rtl_0" in e0.get_bundle_ids()
    assert "rtl_0" not in e1.get_bundle_ids()
