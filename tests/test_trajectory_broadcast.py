"""Tests for src.swarm.trajectory_broadcast."""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.minco import Trajectory
from src.swarm.trajectory_broadcast import (
    SwarmBroadcaster,
    deserialize,
    serialize,
)
from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig


def _example_traj() -> Trajectory:
    s, D = 3, 3
    bc_start = np.zeros((s + 1, D))
    bc_end = np.zeros((s + 1, D))
    bc_end[0] = [10.0, 0.0, 0.0]
    waypoints = np.array(
        [[0.0, 0.0, 0.0], [3.0, 1.0, 0.0], [7.0, -1.0, 0.0], [10.0, 0.0, 0.0]]
    )
    durations = np.array([1.0, 1.0, 1.0])
    return Trajectory(waypoints, durations, bc_start, bc_end, s=s)


class TestSerialize:
    def test_roundtrip_preserves_evaluation(self) -> None:
        traj = _example_traj()
        buf = serialize(traj)
        rebuilt = deserialize(buf)
        for t in np.linspace(0.0, traj.total_time, 11):
            np.testing.assert_allclose(traj.evaluate(t), rebuilt.evaluate(t), atol=1e-10)
            np.testing.assert_allclose(traj.evaluate(t, 1), rebuilt.evaluate(t, 1), atol=1e-10)

    def test_size_under_one_kb(self) -> None:
        # MINCO_PIVOT.md §4.4: ~0.5 KB target
        traj = _example_traj()
        buf = serialize(traj)
        assert len(buf) < 1024, f"payload too large: {len(buf)}"

    def test_bad_magic_raises(self) -> None:
        with pytest.raises(ValueError):
            deserialize(b"BADMAGIC1234")

    def test_truncated_raises(self) -> None:
        traj = _example_traj()
        buf = serialize(traj)
        with pytest.raises(ValueError):
            deserialize(buf[:-10])


class TestSwarmBroadcaster:
    def test_neighbour_visible_after_latency(self) -> None:
        ch = BroadcastChannel(
            ChannelConfig(latency_ms_mean=10.0, latency_ms_jitter=0.0),
            n_agents=3,
            rng=np.random.default_rng(0),
        )
        bcast_0 = SwarmBroadcaster(0, ch)
        bcast_1 = SwarmBroadcaster(1, ch)
        traj = _example_traj()
        bcast_0.broadcast(traj, t_now=0.0)

        assert bcast_1.poll(t_now=0.005) == {}  # before latency
        nb = bcast_1.poll(t_now=0.02)            # after latency
        assert 0 in nb
        np.testing.assert_allclose(
            nb[0].trajectory.evaluate(0.5), traj.evaluate(0.5), atol=1e-10
        )

    def test_latest_only_keeps_most_recent(self) -> None:
        ch = BroadcastChannel(
            ChannelConfig(latency_ms_mean=0.0, latency_ms_jitter=0.0),
            n_agents=3,
            rng=np.random.default_rng(0),
        )
        bcast_0 = SwarmBroadcaster(0, ch)
        bcast_1 = SwarmBroadcaster(1, ch)

        # broadcast two distinct trajectories from agent 0
        t1 = _example_traj()
        t2 = _example_traj()
        t2.waypoints[1] = [3.5, 1.5, 0.0]  # mutate for diff
        t2 = Trajectory(
            t2.waypoints, t2.durations, t2.bc_start, t2.bc_end, s=t2.s
        )

        bcast_0.broadcast(t1, t_now=0.0)
        bcast_0.broadcast(t2, t_now=1.0)
        bcast_1.poll(t_now=2.0)
        nb = bcast_1.latest()
        # the most-recent (t2) should win
        np.testing.assert_allclose(nb[0].trajectory.waypoints[1], [3.5, 1.5, 0.0])
        assert nb[0].t_sent == pytest.approx(1.0)
