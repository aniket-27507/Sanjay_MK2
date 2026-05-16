"""Path search utilities for the safe-flight-corridor pipeline.

Phase 0 Task 0.2 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.2).

Contains:
    plan_path_rrt(start, goal, voxel_map, timeout) -> List[np.ndarray]
        Standard goal-biased RRT.
    plan_path_rrt_connect(start, goal, voxel_map, timeout) -> List[np.ndarray]
        Bidirectional RRT-Connect (Kuffner & LaValle 1999). Two trees,
        rooted at start and goal, alternately extend and try to greedily
        connect to the other tree's newest node. Empirically 10-100× faster
        than vanilla RRT on cluttered maps because both trees grow into the
        free-space corridor between them simultaneously.

The corridor inflater (corridor_generator.py) seeds its FIRI inflation from
each waypoint of this route.

Notes on the algorithms:
    - Goal biasing (goal_bias probability) keeps single-tree RRT from
      wandering on long open-space problems. RRT-Connect doesn't need
      goal bias because its second tree IS the goal bias.
    - Trees are stored as pre-allocated (max_iter+2, 3) numpy buffers so
      that nearest-neighbour queries vectorise.
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


def plan_path_rrt_connect(
    start: np.ndarray,
    goal: np.ndarray,
    voxel_map: VoxelMap,
    timeout: float = 1.0,
    step_size: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    max_iter: int = 50_000,
) -> List[np.ndarray]:
    """Bidirectional RRT-Connect path search over a VoxelMap.

    Parameters
    ----------
    start, goal : (3,) array_like
        Endpoint world coordinates. Both must be in free space.
    voxel_map : VoxelMap
        Occupancy grid, already dilated by the drone's collision radius.
    timeout : float
        Wall-clock budget in seconds.
    step_size : float | None
        Tree extension length in metres. Defaults to 2 × voxel_size — half
        of the single-tree RRT default because RRT-Connect benefits more
        from finer steps in cluttered regions.
    rng : np.random.Generator | None
    max_iter : int
        Hard iteration cap; each iteration grows one tree by one step.

    Returns
    -------
    list of (3,) ndarray
        Waypoints from start to goal. Every consecutive pair is
        collision-free. Empty list on failure / timeout / blocked endpoints.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start and goal must be length-3 vectors")

    if step_size is None:
        step_size = 2.0 * voxel_map.voxel_size
    if rng is None:
        rng = np.random.default_rng()

    if voxel_map.query(start) == 1 or voxel_map.query(goal) == 1:
        return []

    lo, hi = voxel_map.world_bounds
    seg_step = max(voxel_map.voxel_size * 0.5, 1e-3)
    capacity = max_iter + 2

    # Two trees: A starts at `start`, B at `goal`. Each tracked separately.
    tree_a_pts = np.empty((capacity, 3), dtype=np.float64)
    tree_a_par = np.full(capacity, -1, dtype=np.int64)
    tree_a_pts[0] = start
    n_a = 1

    tree_b_pts = np.empty((capacity, 3), dtype=np.float64)
    tree_b_par = np.full(capacity, -1, dtype=np.int64)
    tree_b_pts[0] = goal
    n_b = 1

    def _extend(
        tree_pts: np.ndarray,
        tree_par: np.ndarray,
        n_nodes: int,
        target: np.ndarray,
    ) -> tuple[int, int]:
        """Try to extend `tree` one step toward `target`. Returns
        (new_count, status):
            status =  1  new node placed AT target  (REACHED)
            status =  0  new node placed but not at target  (ADVANCED)
            status = -1  collision or no progress  (TRAPPED)
        """
        dists = np.linalg.norm(tree_pts[:n_nodes] - target, axis=1)
        nearest = int(np.argmin(dists))
        nearest_pt = tree_pts[nearest]
        delta = target - nearest_pt
        d = float(np.linalg.norm(delta))
        if d < 1e-9:
            return n_nodes, -1
        if d <= step_size:
            new_pt = target.copy()
            reached = True
        else:
            new_pt = nearest_pt + delta * (step_size / d)
            reached = False
        if voxel_map.query(new_pt) == 1:
            return n_nodes, -1
        if not _collision_free_segment(nearest_pt, new_pt, voxel_map, seg_step):
            return n_nodes, -1
        tree_pts[n_nodes] = new_pt
        tree_par[n_nodes] = nearest
        return n_nodes + 1, (1 if reached else 0)

    def _connect(
        tree_pts: np.ndarray,
        tree_par: np.ndarray,
        n_nodes: int,
        target: np.ndarray,
    ) -> tuple[int, int]:
        """Greedy extend repeatedly toward `target` until REACHED or TRAPPED."""
        status = 0
        while status == 0:
            n_nodes, status = _extend(tree_pts, tree_par, n_nodes, target)
            if n_nodes >= capacity:
                break
        return n_nodes, status

    t0 = time.perf_counter()
    iters = 0
    connected = False
    # The newest node of whichever tree just extended ends up at
    # tree_a_pts[n_a - 1] before the swap below.
    swap_count = 0
    while iters < max_iter:
        iters += 1
        if (iters & 0x3F) == 0 and time.perf_counter() - t0 >= timeout:
            break

        sample = rng.uniform(lo, hi)
        n_a_new, status = _extend(tree_a_pts, tree_a_par, n_a, sample)
        if status == -1:
            # Tree A trapped on this sample; swap and try again.
            tree_a_pts, tree_b_pts = tree_b_pts, tree_a_pts
            tree_a_par, tree_b_par = tree_b_par, tree_a_par
            n_a, n_b = n_b, n_a
            swap_count += 1
            continue
        n_a = n_a_new
        new_pt_a = tree_a_pts[n_a - 1].copy()

        # Try to connect tree B all the way to new_pt_a
        n_b, status_b = _connect(tree_b_pts, tree_b_par, n_b, new_pt_a)
        if status_b == 1:
            connected = True
            break

        # swap so trees alternate growth — balances exploration
        tree_a_pts, tree_b_pts = tree_b_pts, tree_a_pts
        tree_a_par, tree_b_par = tree_b_par, tree_a_par
        n_a, n_b = n_b, n_a
        swap_count += 1

        if n_a >= capacity or n_b >= capacity:
            break

    if not connected:
        return []

    # After connection: tree_a_pts[n_a-1] and tree_b_pts[n_b-1] both refer
    # to the same coordinate. Walk parents in each tree back to its root.
    path_a: list[np.ndarray] = []
    i = n_a - 1
    while i != -1:
        path_a.append(tree_a_pts[i].copy())
        i = int(tree_a_par[i])
    path_a.reverse()  # tree_a root first

    path_b: list[np.ndarray] = []
    i = n_b - 1
    while i != -1:
        path_b.append(tree_b_pts[i].copy())
        i = int(tree_b_par[i])
    # path_b starts at the connection point (same as path_a[-1]) and walks
    # to tree_b's root. Skip the duplicate first element.
    path_b = path_b[1:]

    # If swap_count is odd, tree A and tree B have swapped roles — A is
    # rooted at goal, B at start. Detect by checking which root matches.
    a_at_start = bool(np.allclose(path_a[0], start, atol=1e-9))
    if a_at_start:
        # A=start, B=goal → A forward + B forward gives start → goal
        return path_a + path_b
    else:
        # A=goal, B=start → reverse everything: B reverse + A reverse
        full = path_a + path_b
        full.reverse()
        return full


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
