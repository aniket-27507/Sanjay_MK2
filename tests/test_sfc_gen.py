"""Unit tests for src.single_drone.planning.sfc_gen.plan_path_rrt.

Phase 0 Task 0.2 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.2).

RRT must produce a sequence of 3D waypoints that:
    - starts at `start`, ends at (or within tolerance of) `goal`
    - has every waypoint in voxel-free space
    - has every consecutive segment collision-free

Determinism is preserved by feeding a seeded numpy Generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.sfc_gen import (
    plan_path_rrt,
    plan_path_rrt_connect,
)
from src.single_drone.planning.voxel_map import VoxelMap


def _empty_map(size=(40, 40, 10), voxel_size=0.5, origin=(0.0, 0.0, 0.0)) -> VoxelMap:
    return VoxelMap(origin=np.asarray(origin), size=size, voxel_size=voxel_size)


def _wall_map(voxel_size=0.5) -> VoxelMap:
    """40x40x10 voxel map with a vertical wall at x=10m with a single hole.

    Voxel coords: wall at i=20, full j sweep, full k sweep, except a 4-voxel
    gap at j in [12..15] and k in [2..5] (i.e. a window for the drone to fly through).
    """
    m = _empty_map(voxel_size=voxel_size)
    for j in range(40):
        for k in range(10):
            if 12 <= j <= 15 and 2 <= k <= 5:
                continue  # hole
            m.set_occupied_voxel((20, j, k))
    return m


def _segment_is_free(p0, p1, voxel_map, step=None) -> bool:
    if step is None:
        step = voxel_map.voxel_size * 0.5
    d = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    n = max(1, int(np.ceil(d / step)))
    for i in range(n + 1):
        t = i / n
        p = p0 + t * (p1 - p0)
        if voxel_map.query(p) == 1:
            return False
    return True


class TestBasicPath:
    def test_open_space_path_found(self) -> None:
        m = _empty_map()
        rng = np.random.default_rng(42)
        path = plan_path_rrt(
            start=np.array([2.0, 2.0, 2.0]),
            goal=np.array([15.0, 15.0, 2.0]),
            voxel_map=m,
            timeout=2.0,
            rng=rng,
        )
        assert path  # non-empty
        np.testing.assert_allclose(path[0], [2.0, 2.0, 2.0])

    def test_path_ends_at_goal(self) -> None:
        m = _empty_map()
        rng = np.random.default_rng(7)
        goal = np.array([15.0, 15.0, 2.0])
        path = plan_path_rrt(
            start=np.array([2.0, 2.0, 2.0]),
            goal=goal,
            voxel_map=m,
            timeout=2.0,
            rng=rng,
        )
        assert path
        np.testing.assert_allclose(path[-1], goal)

    def test_all_waypoints_in_free_space(self) -> None:
        m = _empty_map()
        rng = np.random.default_rng(7)
        path = plan_path_rrt(
            np.array([2.0, 2.0, 2.0]),
            np.array([15.0, 15.0, 2.0]),
            m,
            timeout=2.0,
            rng=rng,
        )
        for p in path:
            assert m.query(p) == 0

    def test_segments_are_collision_free(self) -> None:
        m = _wall_map()
        rng = np.random.default_rng(123)
        path = plan_path_rrt(
            np.array([2.0, 7.0, 2.0]),
            np.array([18.0, 7.0, 2.0]),
            m,
            timeout=5.0,
            rng=rng,
        )
        assert path, "should find a path through the window"
        for a, b in zip(path[:-1], path[1:]):
            assert _segment_is_free(a, b, m), f"segment {a} -> {b} hits an obstacle"


class TestEdgeCases:
    def test_start_blocked_returns_empty(self) -> None:
        m = _empty_map()
        m.set_occupied(np.array([2.0, 2.0, 2.0]))
        rng = np.random.default_rng(0)
        path = plan_path_rrt(
            np.array([2.0, 2.0, 2.0]),
            np.array([10.0, 10.0, 2.0]),
            m,
            timeout=0.2,
            rng=rng,
        )
        assert path == []

    def test_goal_blocked_returns_empty(self) -> None:
        m = _empty_map()
        m.set_occupied(np.array([10.0, 10.0, 2.0]))
        rng = np.random.default_rng(0)
        path = plan_path_rrt(
            np.array([2.0, 2.0, 2.0]),
            np.array([10.0, 10.0, 2.0]),
            m,
            timeout=0.2,
            rng=rng,
        )
        assert path == []

    def test_timeout_respected_when_unreachable(self) -> None:
        # surround the start with obstacles within the same map bounds → unreachable
        m = _empty_map(size=(20, 20, 10), voxel_size=0.5)
        # fill an O-shape around start
        for i in range(2, 7):
            for j in range(2, 7):
                for k in range(0, 10):
                    if (i, j) == (4, 4):
                        continue
                    m.set_occupied_voxel((i, j, k))
        rng = np.random.default_rng(0)
        # start sits inside the cavity, goal outside the cavity
        start = m.voxel_to_world((4, 4, 4))
        goal = np.array([8.0, 8.0, 2.0])
        import time

        t0 = time.perf_counter()
        path = plan_path_rrt(start, goal, m, timeout=0.2, rng=rng)
        elapsed = time.perf_counter() - t0
        assert path == []
        # generous slack: numpy alloc / clock granularity. Should finish
        # well under 1.5 s even on a loaded machine.
        assert elapsed < 1.5

    def test_rejects_nonvector_inputs(self) -> None:
        m = _empty_map()
        with pytest.raises(ValueError):
            plan_path_rrt(
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
                m,
            )


class TestDeterminism:
    def test_seeded_rng_reproducible(self) -> None:
        m = _empty_map()
        start = np.array([2.0, 2.0, 2.0])
        goal = np.array([15.0, 15.0, 2.0])

        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = plan_path_rrt(start, goal, m, timeout=2.0, rng=rng1)
        p2 = plan_path_rrt(start, goal, m, timeout=2.0, rng=rng2)
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            np.testing.assert_allclose(a, b)


class TestWallScenario:
    def test_finds_window_through_wall(self) -> None:
        m = _wall_map()
        # window center in world: voxel (20, 13.5, 3.5) -> (10.25, 6.75, 1.75)
        # start before wall (x<10), goal after wall (x>10)
        rng = np.random.default_rng(2024)
        start = np.array([3.0, 7.0, 2.0])
        goal = np.array([17.0, 7.0, 2.0])
        path = plan_path_rrt(start, goal, m, timeout=10.0, rng=rng)
        assert path, "RRT should find a route through the window"
        # the path should cross x = 10 m (the wall plane) — i.e. some point has x > 10
        xs = [p[0] for p in path]
        assert max(xs) > 10.0 and min(xs) < 10.0


class TestRRTConnect:
    """Tests for `plan_path_rrt_connect`.

    The bidirectional planner has the same correctness contract as
    `plan_path_rrt` — every waypoint in free space, every segment
    collision-free, endpoints preserved.
    """

    def test_open_field_path(self) -> None:
        m = _empty_map()
        rng = np.random.default_rng(7)
        start = np.array([2.0, 2.0, 2.0])
        goal = np.array([18.0, 18.0, 2.0])
        path = plan_path_rrt_connect(start, goal, m, timeout=2.0, rng=rng)
        assert path, "RRT-Connect should find a path in open field"
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], goal)
        # every interior waypoint in free space
        for p in path:
            assert m.query(p) == 0

    def test_finds_window_through_wall(self) -> None:
        m = _wall_map()
        rng = np.random.default_rng(2024)
        start = np.array([3.0, 7.0, 2.0])
        goal = np.array([17.0, 7.0, 2.0])
        path = plan_path_rrt_connect(start, goal, m, timeout=5.0, rng=rng)
        assert path, "RRT-Connect should find the wall window"
        xs = [p[0] for p in path]
        assert max(xs) > 10.0 and min(xs) < 10.0

    def test_blocked_start_returns_empty(self) -> None:
        m = _empty_map()
        m.set_occupied_voxel((4, 4, 4))   # blocks the start voxel
        rng = np.random.default_rng(1)
        start = np.array([2.0, 2.0, 2.0])
        goal = np.array([10.0, 10.0, 2.0])
        path = plan_path_rrt_connect(start, goal, m, timeout=1.0, rng=rng)
        assert path == []

    def test_typically_faster_than_single_tree_on_clutter(self) -> None:
        # Build a moderately cluttered field — 30% obstacle voxels.
        rng_obs = np.random.default_rng(13)
        m = _empty_map(size=(30, 30, 6), voxel_size=0.5)
        # carve start/goal cylinders so endpoints stay free
        n_obstacles = int(0.30 * 30 * 30 * 6)
        for _ in range(n_obstacles):
            i, j, k = (
                int(rng_obs.integers(0, 30)),
                int(rng_obs.integers(0, 30)),
                int(rng_obs.integers(0, 6)),
            )
            # keep a corridor at j around 15
            if 13 <= j <= 16 and 2 <= k <= 4:
                continue
            m.set_occupied_voxel((i, j, k))
        start = np.array([1.5, 7.5, 1.5])
        goal = np.array([13.5, 7.5, 1.5])

        import time as _time
        rng_a = np.random.default_rng(11)
        t0 = _time.perf_counter()
        path_a = plan_path_rrt_connect(start, goal, m, timeout=3.0, rng=rng_a)
        t_connect = _time.perf_counter() - t0

        rng_b = np.random.default_rng(11)
        t0 = _time.perf_counter()
        path_b = plan_path_rrt(start, goal, m, timeout=3.0, rng=rng_b)
        t_single = _time.perf_counter() - t0

        # RRT-Connect should find a path within the budget
        assert path_a, (
            "RRT-Connect failed to find a path inside 3 s on a 30%-density "
            "30×30×6 map with a clear corridor"
        )
        # The single-tree RRT may or may not find a path — that's the
        # whole point. If both succeed, Connect should be no slower.
        if path_b:
            assert t_connect <= t_single * 1.5
