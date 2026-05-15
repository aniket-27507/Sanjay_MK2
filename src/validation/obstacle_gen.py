"""Procedural obstacle generators for the validation rigs.

See docs/MINCO_PIVOT.md §4.5.

Provides:
    random_obstacle_field(rng, voxel_size, size, density, ...) -> VoxelMap
        Uniformly-sampled obstacles, hitting an exact target density. The
        cheapest generator and the spec's reference test scenario.

    random_pillars(rng, voxel_map, n_pillars, radius_range, height_range)
        Vertical cylindrical pillars discretised into the voxel grid. Closer
        to GCOPTER's `mockamap` forest scenarios; more spatially correlated
        than uniform random voxels.

    clear_around(point, radius)
        Helper returning a callable that selects points outside a sphere — used
        to keep start / goal corridors obstacle-free.

The two primary generators both return a populated VoxelMap (not just a point
cloud), so the rigs can dilate by drone radius before querying / planning.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from src.single_drone.planning.voxel_map import VoxelMap


def _ensure_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def clear_around(
    point: np.ndarray, radius: float
) -> Tuple[np.ndarray, float]:
    """Return a (center, radius) pair suitable for the `clear_zones` argument
    of the generators below.
    """
    return (np.asarray(point, dtype=np.float64), float(radius))


def random_obstacle_field(
    rng: Optional[np.random.Generator],
    size: Tuple[int, int, int],
    voxel_size: float,
    density: float,
    origin: np.ndarray = np.zeros(3),
    clear_zones: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    max_attempts: int = 8,
) -> VoxelMap:
    """Uniformly-sampled obstacle voxels at a given density.

    Parameters
    ----------
    rng : numpy Generator
        Random source.
    size : (Lx, Ly, Lz) ints
        Grid dimensions in voxels.
    voxel_size : float
        Edge length per voxel (m).
    density : float in [0, 1]
        Target fraction of voxels marked occupied. The function samples with
        rejection until the count is reached (or `max_attempts` is exhausted).
    origin : (3,) array
        World coordinates of the grid origin corner.
    clear_zones : list of (center, radius)
        Spheres (in world coords) that must remain obstacle-free.
    max_attempts : int
        Rejection-sampling cap.

    Returns
    -------
    VoxelMap
        Populated occupancy grid.
    """
    rng = _ensure_rng(rng)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    m = VoxelMap(origin=np.asarray(origin, dtype=np.float64), size=size, voxel_size=voxel_size)

    n_target = int(round(density * size[0] * size[1] * size[2]))
    if n_target == 0:
        return m

    lo, hi = m.world_bounds
    sample_lo = lo + 0.5 * voxel_size
    sample_hi = hi - 0.5 * voxel_size

    attempts = 0
    while m.num_occupied < n_target and attempts < max_attempts:
        attempts += 1
        n_needed = n_target - m.num_occupied
        # over-sample by 2x to compensate for collisions / cleared zones, but
        # add one at a time so we stop exactly at the target.
        pts = rng.uniform(sample_lo, sample_hi, size=(max(n_needed * 2, 64), 3))
        if clear_zones:
            mask = np.ones(pts.shape[0], dtype=bool)
            for center, radius in clear_zones:
                d = np.linalg.norm(pts - np.asarray(center, dtype=np.float64), axis=1)
                mask &= d > radius
            pts = pts[mask]
        for p in pts:
            if m.num_occupied >= n_target:
                break
            m.set_occupied(p)

    return m


def random_pillars(
    rng: Optional[np.random.Generator],
    size: Tuple[int, int, int],
    voxel_size: float,
    n_pillars: int,
    radius_range: Tuple[float, float] = (0.3, 0.8),
    origin: np.ndarray = np.zeros(3),
    clear_zones: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    height_fraction_range: Tuple[float, float] = (0.5, 1.0),
) -> VoxelMap:
    """Vertical cylindrical pillars discretised into the voxel grid.

    Each pillar has a uniformly-sampled centre (x, y), radius from
    `radius_range`, and vertical extent from `height_fraction_range` * world
    height. Centres falling inside any `clear_zones` sphere are re-sampled
    (up to 5 retries each).
    """
    rng = _ensure_rng(rng)
    m = VoxelMap(origin=np.asarray(origin, dtype=np.float64), size=size, voxel_size=voxel_size)
    lo, hi = m.world_bounds
    h_total = float(hi[2] - lo[2])

    for _ in range(n_pillars):
        for retry in range(6):
            cx = float(rng.uniform(lo[0] + 1.0, hi[0] - 1.0))
            cy = float(rng.uniform(lo[1] + 1.0, hi[1] - 1.0))
            if clear_zones:
                p_xy = np.array([cx, cy])
                ok = True
                for center, radius in clear_zones:
                    c2 = np.asarray(center)[:2]
                    if np.linalg.norm(p_xy - c2) < radius:
                        ok = False
                        break
                if not ok:
                    continue
            break
        r = float(rng.uniform(*radius_range))
        h_frac_lo, h_frac_hi = height_fraction_range
        h_lo = lo[2]
        h_hi = lo[2] + float(rng.uniform(h_frac_lo, h_frac_hi)) * h_total

        nv = int(np.ceil(r / voxel_size)) + 1
        # sample the disk
        pts = []
        for di in range(-nv, nv + 1):
            for dj in range(-nv, nv + 1):
                if (di * voxel_size) ** 2 + (dj * voxel_size) ** 2 > r * r:
                    continue
                xx = cx + di * voxel_size
                yy = cy + dj * voxel_size
                z = h_lo + 0.5 * voxel_size
                while z < h_hi:
                    pts.append([xx, yy, z])
                    z += voxel_size
        if pts:
            m.set_occupied_points(np.asarray(pts, dtype=np.float64))

    return m


def measured_density(voxel_map: VoxelMap) -> float:
    """The fraction of the grid that is currently occupied."""
    total = voxel_map.size[0] * voxel_map.size[1] * voxel_map.size[2]
    return voxel_map.num_occupied / total if total > 0 else 0.0
