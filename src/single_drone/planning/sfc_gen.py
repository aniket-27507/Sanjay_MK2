"""Path search utilities for the safe-flight-corridor pipeline.

Phase 0 Task 0.2 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.2).

Contains:
    plan_path_rrt(start, goal, voxel_map, timeout) -> List[np.ndarray]
        Standard RRT on a VoxelMap. Returns a list of 3D waypoints from start
        to goal, where every waypoint is in free space and every consecutive
        segment is collision-free. Returns [] on failure / timeout / blocked
        endpoints.

The corridor inflater (corridor_generator.py) seeds its FIRI inflation from
each waypoint of this route.

Notes on the algorithm:
    - Goal biasing (goal_bias probability) keeps RRT from wandering on long
      open-space problems.
    - Tree is stored as a pre-allocated (max_iter+2, 3) numpy buffer so that
      nearest-neighbour queries vectorise.
    - Wall-clock timeout is checked every 64 iterations to avoid syscall
      overhead.
    - Segment collision checks sub-sample the line at half-voxel intervals,
      so the only way to miss an obstacle is to undersample inside a single
      voxel — which the half-voxel step guarantees against.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.single_drone.planning.voxel_map import VoxelMap


def _collision_free_segment(
    p0: np.ndarray, p1: np.ndarray, voxel_map: VoxelMap, step: float
) -> bool:
    delta = p1 - p0
    d = float(np.linalg.norm(delta))
    n = max(1, int(np.ceil(d / step)))
    for i in range(n + 1):
        t = i / n
        if voxel_map.query(p0 + t * delta) == 1:
            return False
    return True


def plan_path_rrt(
    start: np.ndarray,
    goal: np.ndarray,
    voxel_map: VoxelMap,
    timeout: float = 1.0,
    step_size: Optional[float] = None,
    goal_bias: float = 0.1,
    goal_tolerance: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    max_iter: int = 50_000,
) -> List[np.ndarray]:
    """RRT path search over a VoxelMap.

    Parameters
    ----------
    start, goal : (3,) array_like
        Endpoint world coordinates. Both must be in free space, otherwise the
        function returns [].
    voxel_map : VoxelMap
        Occupancy grid. Should already be dilated by the drone's collision
        radius / voxel_size; this function does no further inflation.
    timeout : float
        Wall-clock budget in seconds.
    step_size : float | None
        RRT extension length in metres. Defaults to 4 * voxel_size.
    goal_bias : float
        Probability of sampling the goal directly each iteration.
    goal_tolerance : float | None
        Required proximity for success; defaults to step_size.
    rng : np.random.Generator | None
        Seeded for deterministic tests. New generator if None.
    max_iter : int
        Hard iteration cap independent of the wall-clock budget.

    Returns
    -------
    list of (3,) ndarray
        Waypoints from start to goal. Each consecutive pair is collision-free.
        Empty list on failure.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start and goal must be length-3 vectors")

    if step_size is None:
        step_size = 4.0 * voxel_map.voxel_size
    if goal_tolerance is None:
        goal_tolerance = step_size
    if rng is None:
        rng = np.random.default_rng()

    if voxel_map.query(start) == 1 or voxel_map.query(goal) == 1:
        return []

    lo, hi = voxel_map.world_bounds
    seg_step = max(voxel_map.voxel_size * 0.5, 1e-3)

    capacity = max_iter + 2
    node_pts = np.empty((capacity, 3), dtype=np.float64)
    parents = np.full(capacity, -1, dtype=np.int64)
    node_pts[0] = start
    n = 1
    goal_idx = -1

    t0 = time.perf_counter()
    iters = 0
    while iters < max_iter:
        iters += 1
        # check wall clock every 64 iters to amortise syscall cost
        if (iters & 0x3F) == 0 and time.perf_counter() - t0 >= timeout:
            break

        # sample
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = rng.uniform(lo, hi)

        # nearest existing node
        dists = np.linalg.norm(node_pts[:n] - sample, axis=1)
        nearest = int(np.argmin(dists))
        nearest_pt = node_pts[nearest]

        # extend toward sample by step_size
        delta = sample - nearest_pt
        d = float(np.linalg.norm(delta))
        if d < 1e-9:
            continue
        new_pt = nearest_pt + delta * (min(step_size, d) / d)

        if voxel_map.query(new_pt) == 1:
            continue
        if not _collision_free_segment(nearest_pt, new_pt, voxel_map, seg_step):
            continue

        node_pts[n] = new_pt
        parents[n] = nearest
        n += 1

        # success if we can reach the goal from the new node
        if np.linalg.norm(new_pt - goal) <= goal_tolerance:
            if _collision_free_segment(new_pt, goal, voxel_map, seg_step):
                node_pts[n] = goal
                parents[n] = n - 1
                goal_idx = n
                n += 1
                break

        if n >= capacity:
            break

    if goal_idx == -1:
        return []

    path: list[np.ndarray] = []
    i = goal_idx
    while i != -1:
        path.append(node_pts[i].copy())
        i = int(parents[i])
    path.reverse()
    return path


def shortcut_path(
    path: List[np.ndarray],
    voxel_map: VoxelMap,
    seg_step: Optional[float] = None,
) -> List[np.ndarray]:
    """Greedy shortcut: drop intermediate waypoints when the chord between two
    surviving waypoints is collision-free.

    Used to reduce the number of FIRI corridors generated by the inflater.
    Preserves the first and last waypoints.
    """
    if len(path) <= 2:
        return [p.copy() for p in path]
    if seg_step is None:
        seg_step = max(voxel_map.voxel_size * 0.5, 1e-3)

    out = [path[0].copy()]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _collision_free_segment(path[i], path[j], voxel_map, seg_step):
                break
            j -= 1
        out.append(path[j].copy())
        i = j
    return out
