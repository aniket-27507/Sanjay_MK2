"""3D binary occupancy grid with hash-based sparse storage.

Phase 0 Task 0.1 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.2).

The voxel map sits at the bottom of the planning stack:

    depth camera -> point cloud -> VoxelMap.set_occupied_points()
    VoxelMap.dilate(drone_radius)   # inflate by drone half-extent
    VoxelMap.query(point)           # collision-check (used by RRT, FIRI, MINCO penalty)
    VoxelMap.get_surface_points()   # FIRI seed points for corridor inflation

Design choices:
    - Sparse storage: a set of (i, j, k) integer tuples. Trades per-query
      hash-lookup overhead for O(num_occupied) memory rather than O(grid_volume).
      For drone-scale maps (e.g. 1000^3 voxels with <1% occupancy) this is the
      only practical option.
    - Safety convention: out-of-bounds queries return 1 (occupied). A planner
      that treats the unknown beyond the world as free will happily fly into
      it; the unknown-as-blocked rule is the safe default.
    - set_occupied of an out-of-bounds point is silently dropped (convenient
      for batch loading of point clouds that may include edge noise).

Reference: GCOPTER voxel_map.hpp (ZJU-FAST-Lab). This is a clean-room Python
port — algorithm only, no source code copied.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

VoxelIndex = Tuple[int, int, int]


class VoxelMap:
    """Sparse 3D binary occupancy grid.

    Voxel (i, j, k) covers the half-open box
        [origin + (i, j, k) * voxel_size, origin + (i+1, j+1, k+1) * voxel_size)
    Its center is at origin + (i + 0.5, j + 0.5, k + 0.5) * voxel_size.

    Parameters
    ----------
    origin : (3,) array
        World coordinates of the (0, 0, 0) voxel's minimum corner.
    size : tuple of three ints
        Grid dimensions in voxels along (x, y, z).
    voxel_size : float
        Edge length of one voxel in metres. Must be positive.
    """

    __slots__ = ("origin", "size", "voxel_size", "_occupied")

    def __init__(
        self,
        origin: np.ndarray,
        size: Tuple[int, int, int],
        voxel_size: float,
    ) -> None:
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError("origin must be a length-3 vector")
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        size_tuple = tuple(int(s) for s in size)
        if len(size_tuple) != 3 or any(s <= 0 for s in size_tuple):
            raise ValueError("size must be a length-3 tuple of positive ints")

        self.origin = origin
        self.size = size_tuple
        self.voxel_size = float(voxel_size)
        self._occupied: set[VoxelIndex] = set()

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def world_to_voxel(self, point) -> VoxelIndex:
        p = np.asarray(point, dtype=np.float64)
        idx = np.floor((p - self.origin) / self.voxel_size).astype(np.int64)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def voxel_to_world(self, idx) -> np.ndarray:
        i, j, k = idx
        return self.origin + (np.array([i, j, k], dtype=np.float64) + 0.5) * self.voxel_size

    def in_bounds(self, idx) -> bool:
        i, j, k = idx
        return (
            0 <= i < self.size[0]
            and 0 <= j < self.size[1]
            and 0 <= k < self.size[2]
        )

    # ------------------------------------------------------------------
    # Occupancy mutation
    # ------------------------------------------------------------------
    def set_occupied(self, point) -> None:
        idx = self.world_to_voxel(point)
        if self.in_bounds(idx):
            self._occupied.add(idx)

    def set_occupied_points(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return
        idx = np.floor((pts - self.origin) / self.voxel_size).astype(np.int64)
        lo = np.array([0, 0, 0], dtype=np.int64)
        hi = np.array(self.size, dtype=np.int64)
        mask = np.all((idx >= lo) & (idx < hi), axis=1)
        for i, j, k in idx[mask]:
            self._occupied.add((int(i), int(j), int(k)))

    def set_occupied_voxel(self, idx: VoxelIndex) -> None:
        if self.in_bounds(idx):
            self._occupied.add((int(idx[0]), int(idx[1]), int(idx[2])))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, point) -> int:
        idx = self.world_to_voxel(point)
        if not self.in_bounds(idx):
            return 1
        return 1 if idx in self._occupied else 0

    def query_voxel(self, idx: VoxelIndex) -> int:
        if not self.in_bounds(idx):
            return 1
        return 1 if tuple(idx) in self._occupied else 0

    # ------------------------------------------------------------------
    # Dilation (Chebyshev / cube neighbourhood, GCOPTER stride-style)
    # ------------------------------------------------------------------
    def dilate(self, radius_voxels: int) -> None:
        r = int(radius_voxels)
        if r <= 0:
            return
        sx, sy, sz = self.size
        new_occupied: set[VoxelIndex] = set()
        for i0, j0, k0 in self._occupied:
            i_lo = max(0, i0 - r)
            i_hi = min(sx - 1, i0 + r)
            j_lo = max(0, j0 - r)
            j_hi = min(sy - 1, j0 + r)
            k_lo = max(0, k0 - r)
            k_hi = min(sz - 1, k0 + r)
            for i in range(i_lo, i_hi + 1):
                for j in range(j_lo, j_hi + 1):
                    for k in range(k_lo, k_hi + 1):
                        new_occupied.add((i, j, k))
        self._occupied = new_occupied

    # ------------------------------------------------------------------
    # Surface extraction (FIRI seed points)
    # ------------------------------------------------------------------
    _FACE_NEIGHBORS = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    def get_surface_points(self) -> np.ndarray:
        """Return world-coordinate centers of occupied voxels that touch free space.

        A voxel is on the surface if at least one of its 6 face-neighbours is
        either out-of-bounds or not occupied. Out-of-bounds neighbours count as
        free for this purpose (i.e. boundary voxels are surface), since FIRI
        wants the outer envelope of obstacles even when they touch the world
        boundary.
        """
        if not self._occupied:
            return np.zeros((0, 3), dtype=np.float64)

        sx, sy, sz = self.size
        surface: list[VoxelIndex] = []
        for i, j, k in self._occupied:
            for di, dj, dk in self._FACE_NEIGHBORS:
                ni, nj, nk = i + di, j + dj, k + dk
                if not (0 <= ni < sx and 0 <= nj < sy and 0 <= nk < sz):
                    surface.append((i, j, k))
                    break
                if (ni, nj, nk) not in self._occupied:
                    surface.append((i, j, k))
                    break

        if not surface:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([self.voxel_to_world(idx) for idx in surface], dtype=np.float64)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def num_occupied(self) -> int:
        return len(self._occupied)

    def occupied_voxels(self) -> Iterable[VoxelIndex]:
        return iter(self._occupied)

    @property
    def world_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (lower_corner, upper_corner) in world coordinates."""
        lo = self.origin.copy()
        hi = self.origin + np.array(self.size, dtype=np.float64) * self.voxel_size
        return lo, hi
