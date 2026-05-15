"""FIRI safe-flight-corridor generation.

Phase 0 Task 0.3 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.3, §4.2).

A safe flight corridor is a sequence of convex polytopes H_1, ..., H_K, each
of the form

    H_i = {x in R^3 : A_i x <= b_i}

such that:
    - the corresponding route segment lies entirely inside H_i
    - H_i is obstacle-free
    - consecutive H_i, H_{i+1} overlap (they share the route waypoint between
      them by construction, since this implementation is segment-seeded)

FIRI (Fast Iterative Region Inflation), as adopted here, is segment-seeded:
each polytope is built around the segment [w_i, w_{i+1}] of the RRT route.
For each obstacle point (sorted by distance to the segment), we test whether
it is already excluded by an existing halfplane; if not, we add a new
halfplane through the obstacle, normal to the line from its closest segment
point. Because that line is by construction perpendicular to the segment at
the closest point, the new halfplane never cuts the segment.

This guarantees the route stays inside the corridor and every obstacle ends up
on the boundary or outside, even though we never explicitly enforce either
property via a constraint solver.

Reference: GCOPTER firi.hpp / sfc_gen.hpp (ZJU-FAST-Lab). Clean-room Python
port — algorithm only.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Sequence, Tuple

import numpy as np

Polytope = namedtuple("Polytope", ["A", "b"])

WorldBounds = Tuple[np.ndarray, np.ndarray]


def polytope_contains(p: Polytope, point: np.ndarray, tol: float = 1e-9) -> bool:
    """Test whether `point` lies inside polytope `p` (Ax <= b within tolerance)."""
    return bool(np.all(p.A @ np.asarray(point, dtype=np.float64) <= p.b + tol))


def _world_box(world_bounds: WorldBounds) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = world_bounds
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    b = np.array(
        [hi[0], -lo[0], hi[1], -lo[1], hi[2], -lo[2]], dtype=np.float64
    )
    return A, b


def inflate_segment_polytope(
    p0: np.ndarray,
    p1: np.ndarray,
    surface_points: np.ndarray,
    world_bounds: WorldBounds,
    max_extent: float | None = None,
    skip_tol: float = 1e-9,
) -> Polytope:
    """Inflate a convex polytope around the line segment [p0, p1].

    Parameters
    ----------
    p0, p1 : (3,) float arrays
        Segment endpoints. Both must already lie in free space.
    surface_points : (N, 3) float array
        Obstacle surface points (typically `voxel_map.get_surface_points()`).
        May be empty.
    world_bounds : (lower, upper)
        World bounding box. Always part of the polytope's halfspaces.
    max_extent : float | None
        Optional tighter axis-aligned crop around the segment midpoint, in
        metres. Speeds up inflation when the world is much larger than the
        local corridor of interest.
    skip_tol : float
        Numerical tolerance used to decide whether an obstacle is already
        excluded by the current polytope.

    Returns
    -------
    Polytope
        With shape (m, 3) for A and (m,) for b. All rows have ||A_i||_2 = 1
        (except the 6 axis-aligned world bound rows, which already have unit
        norm). Each obstacle point satisfies max(A obs - b) >= 0.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    if p0.shape != (3,) or p1.shape != (3,):
        raise ValueError("p0 and p1 must be length-3 vectors")

    lo, hi = world_bounds
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()

    if max_extent is not None:
        center = 0.5 * (p0 + p1)
        lo = np.maximum(lo, center - max_extent)
        hi = np.minimum(hi, center + max_extent)

    A, b = _world_box((lo, hi))

    surface = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    if surface.shape[0] == 0:
        return Polytope(A=A, b=b)

    # closest point on segment to each surface point
    d = p1 - p0
    d2 = float(np.dot(d, d))
    if d2 < 1e-12:
        closest_pts = np.tile(p0, (surface.shape[0], 1))
        dists = np.linalg.norm(surface - p0, axis=1)
    else:
        diffs = surface - p0
        t = np.clip((diffs @ d) / d2, 0.0, 1.0)
        closest_pts = p0[None, :] + t[:, None] * d[None, :]
        dists = np.linalg.norm(surface - closest_pts, axis=1)

    order = np.argsort(dists)

    A_rows: list[np.ndarray] = [A]
    b_rows: list[np.ndarray] = [b]
    A_cur = A
    b_cur = b

    for idx in order:
        obs = surface[idx]
        residual = A_cur @ obs - b_cur
        if np.any(residual > skip_tol):
            continue  # already excluded by some existing halfplane

        normal_dir = obs - closest_pts[idx]
        norm = float(np.linalg.norm(normal_dir))
        if norm < 1e-12:
            # obstacle coincides with the segment — segment is in collision;
            # caller violated the precondition. Skip this obstacle defensively.
            continue
        n_hat = normal_dir / norm
        b_new = float(np.dot(n_hat, obs))

        # numerical safety: segment endpoints must be on the safe side.
        if (
            float(np.dot(n_hat, p0)) > b_new + 1e-6
            or float(np.dot(n_hat, p1)) > b_new + 1e-6
        ):
            continue

        A_cur = np.vstack([A_cur, n_hat[None, :]])
        b_cur = np.append(b_cur, b_new)

    return Polytope(A=A_cur, b=b_cur)


def convex_cover(
    route: Sequence[np.ndarray],
    surface_points: np.ndarray,
    world_bounds: WorldBounds,
    max_extent: float | None = None,
) -> List[Polytope]:
    """Build a polytope per segment of a route.

    Parameters
    ----------
    route : sequence of (3,) float arrays
        Waypoints from path planner. Length 0 or 1 yields no polytopes.
    surface_points : (N, 3) float array
        Obstacle surface points shared across all segments.
    world_bounds : (lower, upper)
        World bounding box.
    max_extent : float | None
        Optional crop around each segment's midpoint.

    Returns
    -------
    list of Polytope
        Length = max(0, len(route) - 1). Consecutive polytopes share the
        waypoint between them (each polytope contains both endpoints of its
        segment), so their intersection is non-empty.
    """
    if len(route) < 2:
        return []
    return [
        inflate_segment_polytope(
            p0=np.asarray(route[i], dtype=np.float64),
            p1=np.asarray(route[i + 1], dtype=np.float64),
            surface_points=surface_points,
            world_bounds=world_bounds,
            max_extent=max_extent,
        )
        for i in range(len(route) - 1)
    ]


def shortcut(polytopes: Sequence[Polytope]) -> List[Polytope]:
    """Drop polytopes whose corresponding segment endpoint is already inside
    both neighbours, keeping only those that contribute new free space.

    For v0 this is a no-op pass-through; the MINCO optimiser tolerates extra
    polytopes. The hook is preserved so that a future implementation can apply
    a tighter pruning rule (e.g. drop P_i if the route segment that seeded it
    is fully contained in P_{i-1} or P_{i+1}).
    """
    return list(polytopes)
