"""Unit tests for src.single_drone.planning.voxel_map.

Phase 0 Task 0.1 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.2).
Voxel map is the foundation: every later stage (RRT, FIRI, MINCO penalty
evaluation) collision-checks through this module.

Test design follows the spec's API contract:
    - set_occupied(point) / set_occupied_points(points)
    - query(point) -> 0 (free) or 1 (occupied)
    - dilate(radius_voxels)
    - get_surface_points()
And the safety convention that out-of-bounds queries return 1 (occupied),
since a path planner must treat the unknown beyond the world as unsafe.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.single_drone.planning.voxel_map import VoxelMap


def make_map(size=(20, 20, 10), voxel_size=0.5, origin=(0.0, 0.0, 0.0)) -> VoxelMap:
    return VoxelMap(origin=np.asarray(origin), size=size, voxel_size=voxel_size)


class TestConstruction:
    def test_default_grid_is_empty(self) -> None:
        m = make_map()
        assert m.num_occupied == 0

    def test_rejects_non3_origin(self) -> None:
        with pytest.raises(ValueError):
            VoxelMap(origin=np.array([0.0, 0.0]), size=(10, 10, 10), voxel_size=0.5)

    def test_rejects_nonpositive_voxel_size(self) -> None:
        with pytest.raises(ValueError):
            VoxelMap(origin=np.zeros(3), size=(10, 10, 10), voxel_size=0.0)
        with pytest.raises(ValueError):
            VoxelMap(origin=np.zeros(3), size=(10, 10, 10), voxel_size=-1.0)

    def test_rejects_bad_size(self) -> None:
        with pytest.raises(ValueError):
            VoxelMap(origin=np.zeros(3), size=(10, 10), voxel_size=0.5)
        with pytest.raises(ValueError):
            VoxelMap(origin=np.zeros(3), size=(10, 10, 0), voxel_size=0.5)


class TestCoordinates:
    def test_world_to_voxel_at_origin(self) -> None:
        m = make_map()
        # origin sits at the corner of voxel (0,0,0); a point just inside maps to (0,0,0)
        assert m.world_to_voxel(np.array([0.01, 0.01, 0.01])) == (0, 0, 0)

    def test_world_to_voxel_step(self) -> None:
        m = make_map(voxel_size=0.5)
        # 0.5 m step puts us in (1,0,0)
        assert m.world_to_voxel(np.array([0.6, 0.0, 0.0])) == (1, 0, 0)

    def test_voxel_to_world_centers(self) -> None:
        m = make_map(voxel_size=0.5)
        center = m.voxel_to_world((0, 0, 0))
        # center of voxel (0,0,0) is at origin + 0.5 * voxel_size in each axis
        np.testing.assert_allclose(center, [0.25, 0.25, 0.25])

    def test_world_voxel_roundtrip(self) -> None:
        m = make_map(voxel_size=0.5, origin=(-5.0, -5.0, 0.0))
        idx = (3, 7, 2)
        center = m.voxel_to_world(idx)
        assert m.world_to_voxel(center) == idx

    def test_in_bounds(self) -> None:
        m = make_map(size=(10, 10, 5))
        assert m.in_bounds((0, 0, 0))
        assert m.in_bounds((9, 9, 4))
        assert not m.in_bounds((10, 0, 0))
        assert not m.in_bounds((0, -1, 0))
        assert not m.in_bounds((0, 0, 5))


class TestSetAndQuery:
    def test_empty_grid_queries_free(self) -> None:
        m = make_map()
        rng = np.random.default_rng(0)
        # 20 random in-bounds points
        for _ in range(20):
            p = rng.uniform(0.0, 5.0, size=3)
            assert m.query(p) == 0

    def test_set_then_query_returns_occupied(self) -> None:
        m = make_map()
        p = np.array([1.0, 1.0, 1.0])
        m.set_occupied(p)
        assert m.query(p) == 1

    def test_neighbor_voxel_remains_free(self) -> None:
        m = make_map(voxel_size=0.5)
        m.set_occupied(np.array([1.0, 1.0, 1.0]))
        # one voxel over in +x
        assert m.query(np.array([1.6, 1.0, 1.0])) == 0

    def test_out_of_bounds_query_is_occupied(self) -> None:
        m = make_map(size=(10, 10, 10), voxel_size=0.5)
        # negative side
        assert m.query(np.array([-0.5, 1.0, 1.0])) == 1
        # past the upper side: size 10 * voxel 0.5 = 5.0 m
        assert m.query(np.array([5.5, 1.0, 1.0])) == 1
        assert m.query(np.array([1.0, 1.0, 6.0])) == 1

    def test_set_occupied_out_of_bounds_is_silent(self) -> None:
        m = make_map(size=(10, 10, 10), voxel_size=0.5)
        m.set_occupied(np.array([-1.0, 1.0, 1.0]))
        assert m.num_occupied == 0

    def test_set_occupied_points_batch(self) -> None:
        m = make_map(size=(20, 20, 10), voxel_size=0.5)
        pts = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],  # duplicate, same voxel — should not double count
                [2.0, 2.0, 2.0],
                [-5.0, 0.0, 0.0],  # out of bounds — drop
            ]
        )
        m.set_occupied_points(pts)
        assert m.num_occupied == 2

    def test_set_occupied_points_empty(self) -> None:
        m = make_map()
        m.set_occupied_points(np.zeros((0, 3)))
        assert m.num_occupied == 0


class TestDilate:
    def test_dilate_zero_is_noop(self) -> None:
        m = make_map(voxel_size=0.5)
        m.set_occupied(np.array([2.0, 2.0, 2.0]))
        before = m.num_occupied
        m.dilate(0)
        assert m.num_occupied == before

    def test_dilate_one_chebyshev_inflates_cube(self) -> None:
        m = make_map(size=(20, 20, 20), voxel_size=0.5)
        # single voxel in the middle so no boundary clipping
        m.set_occupied(np.array([5.0, 5.0, 5.0]))
        m.dilate(1)
        # 3x3x3 cube = 27 voxels
        assert m.num_occupied == 27

    def test_dilate_two_chebyshev_inflates_to_5cube(self) -> None:
        m = make_map(size=(20, 20, 20), voxel_size=0.5)
        m.set_occupied(np.array([5.0, 5.0, 5.0]))
        m.dilate(2)
        # 5x5x5 = 125 voxels
        assert m.num_occupied == 125

    def test_dilate_preserves_original(self) -> None:
        m = make_map(voxel_size=0.5)
        center = np.array([2.0, 2.0, 2.0])
        m.set_occupied(center)
        m.dilate(2)
        assert m.query(center) == 1

    def test_dilate_clamps_to_bounds(self) -> None:
        m = make_map(size=(5, 5, 5), voxel_size=0.5)
        # set a voxel in the corner
        corner_world = m.voxel_to_world((0, 0, 0))
        m.set_occupied(corner_world)
        m.dilate(1)
        # corner expansion of 3x3x3 normally = 27 voxels, but 19 fall in bounds
        # (the 8 outside-the-octant neighbors are clipped). Specifically: the
        # corner has 1 in-octant slot — only the (i,j,k) with i,j,k >= 0 are valid.
        # That's 2x2x2 = 8 voxels in-bounds.
        assert m.num_occupied == 8

    def test_dilate_idempotent_after_full_coverage(self) -> None:
        m = make_map(size=(3, 3, 3), voxel_size=0.5)
        center = m.voxel_to_world((1, 1, 1))
        m.set_occupied(center)
        m.dilate(2)
        # everything is occupied (3x3x3 = 27)
        full = m.num_occupied
        assert full == 27
        m.dilate(5)  # should not exceed bounds
        assert m.num_occupied == 27


class TestSurfacePoints:
    def test_single_voxel_is_surface(self) -> None:
        m = make_map(size=(10, 10, 10), voxel_size=0.5)
        m.set_occupied(np.array([2.0, 2.0, 2.0]))
        pts = m.get_surface_points()
        assert pts.shape == (1, 3)

    def test_empty_grid_has_no_surface(self) -> None:
        m = make_map()
        pts = m.get_surface_points()
        assert pts.shape == (0, 3)

    def test_solid_cube_only_returns_shell(self) -> None:
        m = make_map(size=(10, 10, 10), voxel_size=0.5)
        # fill a 3x3x3 solid block at indices (3..5, 3..5, 3..5)
        for i in range(3, 6):
            for j in range(3, 6):
                for k in range(3, 6):
                    m.set_occupied(m.voxel_to_world((i, j, k)))
        pts = m.get_surface_points()
        # 27 total, 1 interior (the center voxel at (4,4,4)) → 26 surface voxels
        assert pts.shape == (26, 3)

    def test_boundary_voxel_counts_as_surface(self) -> None:
        m = make_map(size=(5, 5, 5), voxel_size=0.5)
        corner_world = m.voxel_to_world((0, 0, 0))
        m.set_occupied(corner_world)
        pts = m.get_surface_points()
        # a corner voxel has out-of-bounds neighbors, which count as "free" for
        # surface extraction; the voxel itself must be reported.
        assert pts.shape == (1, 3)
        np.testing.assert_allclose(pts[0], corner_world)


class TestSparseStorage:
    def test_large_grid_small_memory(self) -> None:
        # 1000x1000x100 = 1e8 voxels would be hundreds of MB dense, but
        # sparse storage should be O(num_occupied) only.
        m = VoxelMap(origin=np.zeros(3), size=(1000, 1000, 100), voxel_size=0.1)
        m.set_occupied(np.array([10.0, 10.0, 5.0]))
        m.set_occupied(np.array([20.0, 20.0, 5.0]))
        assert m.num_occupied == 2

    def test_iterating_occupied_voxels(self) -> None:
        m = make_map()
        m.set_occupied(np.array([1.0, 1.0, 1.0]))
        m.set_occupied(np.array([2.0, 2.0, 2.0]))
        voxels = list(m.occupied_voxels())
        assert len(voxels) == 2
        # each entry is a tuple of three ints
        for v in voxels:
            assert isinstance(v, tuple) and len(v) == 3
            assert all(isinstance(c, int) for c in v)
