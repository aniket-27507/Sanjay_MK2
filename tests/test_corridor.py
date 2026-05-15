"""Unit tests for src.single_drone.planning.corridor_generator.

Phase 0 Task 0.3 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.2, §2.3).

Correctness invariants the FIRI corridors must satisfy:
    1. Polytope is a convex region {x : A x <= b}, A is (m, 3), b is (m,).
    2. The polytope contains its seed segment endpoints (and the chord between
       them, since polytopes are convex).
    3. Every obstacle (surface) point is excluded — for each obstacle, at
       least one halfplane satisfies A_i . obs >= b_i (on or outside the
       boundary).
    4. With no obstacles, the polytope equals the world bounding box (6 rows).
    5. Consecutive polytopes overlap — they share the route waypoint at their
       boundary by construction, so their intersection is non-empty.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.corridor_generator import (
    Polytope,
    convex_cover,
    inflate_segment_polytope,
    polytope_contains,
)
from src.single_drone.planning.voxel_map import VoxelMap


def _world_bounds(size=(40, 40, 10), voxel_size=0.5, origin=(0.0, 0.0, 0.0)):
    m = VoxelMap(origin=np.asarray(origin), size=size, voxel_size=voxel_size)
    return m.world_bounds


class TestPolytopeBasics:
    def test_empty_world_returns_box(self) -> None:
        lo, hi = _world_bounds()
        p = inflate_segment_polytope(
            p0=np.array([5.0, 5.0, 2.0]),
            p1=np.array([10.0, 5.0, 2.0]),
            surface_points=np.zeros((0, 3)),
            world_bounds=(lo, hi),
        )
        assert isinstance(p, Polytope)
        assert p.A.shape == (6, 3)
        assert p.b.shape == (6,)

    def test_polytope_contains_endpoints(self) -> None:
        lo, hi = _world_bounds()
        p0 = np.array([5.0, 5.0, 2.0])
        p1 = np.array([10.0, 5.0, 2.0])
        obstacles = np.array(
            [
                [7.5, 8.0, 2.0],
                [7.5, 2.0, 2.0],
                [7.5, 5.0, 5.0],
            ]
        )
        p = inflate_segment_polytope(p0, p1, obstacles, (lo, hi))
        assert polytope_contains(p, p0)
        assert polytope_contains(p, p1)
        # midpoint too (convex)
        assert polytope_contains(p, 0.5 * (p0 + p1))

    def test_obstacle_is_excluded(self) -> None:
        lo, hi = _world_bounds()
        p0 = np.array([5.0, 5.0, 2.0])
        p1 = np.array([10.0, 5.0, 2.0])
        # one obstacle right next to the segment
        obs = np.array([[7.5, 7.0, 2.0]])
        p = inflate_segment_polytope(p0, p1, obs, (lo, hi))
        # obstacle should be on the boundary or excluded (max residual >= 0)
        assert np.max(p.A @ obs[0] - p.b) >= -1e-6

    def test_all_obstacles_excluded(self) -> None:
        lo, hi = _world_bounds()
        p0 = np.array([5.0, 5.0, 2.0])
        p1 = np.array([10.0, 5.0, 2.0])
        rng = np.random.default_rng(0)
        # 50 random obstacles in the world, not on the segment
        obstacles = []
        while len(obstacles) < 50:
            q = rng.uniform(lo + 0.5, hi - 0.5)
            # don't put obstacles right on the segment line (y=5, z=2)
            if abs(q[1] - 5.0) > 1.5 or abs(q[2] - 2.0) > 1.5 or q[0] < 5.0 or q[0] > 10.0:
                obstacles.append(q)
        obstacles = np.asarray(obstacles)
        p = inflate_segment_polytope(p0, p1, obstacles, (lo, hi))

        for o in obstacles:
            residual = np.max(p.A @ o - p.b)
            assert residual >= -1e-6, (
                f"obstacle {o} sits strictly inside polytope (residual={residual})"
            )

    def test_polytope_normals_unit_length(self) -> None:
        lo, hi = _world_bounds()
        p0 = np.array([5.0, 5.0, 2.0])
        p1 = np.array([10.0, 5.0, 2.0])
        obs = np.array([[7.5, 7.0, 2.0], [7.5, 3.0, 2.0]])
        p = inflate_segment_polytope(p0, p1, obs, (lo, hi))
        norms = np.linalg.norm(p.A, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)


class TestPolytopeContains:
    def test_contains_basic(self) -> None:
        # unit cube [0,1]^3 as a polytope
        A = np.array(
            [
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
            ],
            dtype=np.float64,
        )
        b = np.array([1, 0, 1, 0, 1, 0], dtype=np.float64)
        p = Polytope(A=A, b=b)
        assert polytope_contains(p, np.array([0.5, 0.5, 0.5]))
        assert not polytope_contains(p, np.array([1.5, 0.5, 0.5]))
        assert not polytope_contains(p, np.array([-0.1, 0.5, 0.5]))


class TestConvexCover:
    def test_cover_count_one_less_than_waypoints(self) -> None:
        lo, hi = _world_bounds()
        route = [
            np.array([2.0, 5.0, 2.0]),
            np.array([8.0, 5.0, 2.0]),
            np.array([15.0, 5.0, 2.0]),
        ]
        ps = convex_cover(route, np.zeros((0, 3)), (lo, hi))
        assert len(ps) == 2

    def test_short_route_yields_empty_cover(self) -> None:
        lo, hi = _world_bounds()
        ps = convex_cover([np.array([1.0, 1.0, 1.0])], np.zeros((0, 3)), (lo, hi))
        assert ps == []

    def test_cover_polytopes_contain_their_endpoints(self) -> None:
        lo, hi = _world_bounds()
        route = [
            np.array([2.0, 5.0, 2.0]),
            np.array([8.0, 5.0, 2.0]),
            np.array([15.0, 5.0, 2.0]),
        ]
        rng = np.random.default_rng(0)
        obstacles = rng.uniform(lo + 0.5, hi - 0.5, size=(30, 3))
        # remove obstacles too close to route (within 1.0m)
        keep = []
        for o in obstacles:
            for w in route:
                if np.linalg.norm(o - w) < 1.0:
                    break
            else:
                keep.append(o)
        obstacles = np.asarray(keep)

        ps = convex_cover(route, obstacles, (lo, hi))
        for i, p in enumerate(ps):
            assert polytope_contains(p, route[i])
            assert polytope_contains(p, route[i + 1])

    def test_consecutive_polytopes_overlap(self) -> None:
        lo, hi = _world_bounds()
        route = [
            np.array([2.0, 5.0, 2.0]),
            np.array([8.0, 5.0, 2.0]),
            np.array([15.0, 5.0, 2.0]),
        ]
        # shared waypoint is route[1]; it must lie in both ps[0] and ps[1]
        ps = convex_cover(route, np.zeros((0, 3)), (lo, hi))
        shared = route[1]
        assert polytope_contains(ps[0], shared)
        assert polytope_contains(ps[1], shared)

    def test_obstacle_outside_every_polytope(self) -> None:
        lo, hi = _world_bounds()
        route = [
            np.array([2.0, 5.0, 2.0]),
            np.array([8.0, 5.0, 2.0]),
            np.array([15.0, 5.0, 2.0]),
        ]
        obstacles = np.array(
            [
                [5.0, 8.0, 2.0],
                [11.0, 8.0, 2.0],
                [5.0, 2.0, 2.0],
                [11.0, 2.0, 2.0],
            ]
        )
        ps = convex_cover(route, obstacles, (lo, hi))
        for poly in ps:
            for o in obstacles:
                assert np.max(poly.A @ o - poly.b) >= -1e-6
