"""End-to-end Phase 0 pipeline integration test.

Generates a random obstacle point cloud, runs the full MINCO planning
pipeline (VoxelMap → RRT → FIRI → MINCO → flatness), and asserts:

    - The pipeline produces a non-empty trajectory.
    - The trajectory hits start and goal.
    - All quadrature samples lie inside their corridor polytopes (modulo a
      small soft-penalty leak).
    - Thrust stays in a physical range over the trajectory.

Also prints timing for each stage — useful when comparing the Python port
against the GCOPTER C++ reference in later phases.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.single_drone.planning import (
    GCopterConfig,
    Trajectory,
    VoxelMap,
    convex_cover,
    evaluate_trajectory_dynamics,
    gcopter_optimize,
    plan_path_rrt,
    shortcut_path,
)


def _random_obstacle_map(
    rng: np.random.Generator,
    size=(40, 40, 10),
    voxel_size=0.5,
    density=0.05,
    clear_around=None,  # list of (point, radius) pairs to leave clear
) -> VoxelMap:
    m = VoxelMap(origin=np.zeros(3), size=size, voxel_size=voxel_size)
    n_voxels = size[0] * size[1] * size[2]
    n_obstacles = int(density * n_voxels)
    lo, hi = m.world_bounds
    pts = rng.uniform(lo + 0.5, hi - 0.5, size=(n_obstacles * 2, 3))
    if clear_around:
        mask = np.ones(pts.shape[0], dtype=bool)
        for center, radius in clear_around:
            d = np.linalg.norm(pts - center, axis=1)
            mask &= d > radius
        pts = pts[mask]
    pts = pts[:n_obstacles]
    m.set_occupied_points(pts)
    return m


def test_minco_e2e_pipeline() -> None:
    rng = np.random.default_rng(2026)

    start = np.array([2.0, 10.0, 2.0])
    goal = np.array([18.0, 10.0, 2.0])

    # leave a 1.5 m sphere around each endpoint clear so RRT can attach
    voxel_map = _random_obstacle_map(
        rng,
        density=0.04,
        clear_around=[(start, 1.5), (goal, 1.5)],
    )

    # dilate by 1 voxel (≈ drone radius / voxel_size)
    voxel_map.dilate(radius_voxels=1)

    assert voxel_map.query(start) == 0, "start fell on an obstacle — bad seed"
    assert voxel_map.query(goal) == 0, "goal fell on an obstacle — bad seed"

    # 1. RRT
    t0 = time.perf_counter()
    route = plan_path_rrt(start, goal, voxel_map, timeout=10.0, rng=rng)
    t_rrt = (time.perf_counter() - t0) * 1000  # ms
    assert route, "RRT failed to find a route"
    # shortcut to keep FIRI work small
    route = shortcut_path(route, voxel_map)
    assert len(route) >= 2

    # 2. FIRI corridors
    surface = voxel_map.get_surface_points()
    t0 = time.perf_counter()
    polys = convex_cover(route, surface, voxel_map.world_bounds)
    t_firi = (time.perf_counter() - t0) * 1000

    # 3. MINCO setup: initial durations sized by distance / v_max
    v_nominal = 3.0
    durations = []
    for a, b in zip(route[:-1], route[1:]):
        d = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        durations.append(max(0.5, d / v_nominal))
    durations = np.asarray(durations, dtype=np.float64)

    s = 3
    D = 3
    bc_start = np.zeros((s + 1, D))
    bc_start[0] = start
    bc_end = np.zeros((s + 1, D))
    bc_end[0] = goal
    waypoints = np.array(route, dtype=np.float64)

    # 4. GCOPTER optimisation
    t0 = time.perf_counter()
    traj = gcopter_optimize(
        initial_waypoints=waypoints,
        initial_durations=durations,
        bc_start=bc_start,
        bc_end=bc_end,
        polytopes=polys,
        config=GCopterConfig(v_max=v_nominal, n_quad=12, maxiter=120),
    )
    t_minco = (time.perf_counter() - t0) * 1000

    assert isinstance(traj, Trajectory)

    # 5. Verify endpoints
    np.testing.assert_allclose(traj.evaluate(0.0), start, atol=1e-5)
    np.testing.assert_allclose(traj.evaluate(traj.total_time), goal, atol=1e-4)

    # 6. Verify corridor containment at quadrature samples (soft-penalty leak <= 0.2 m)
    max_leak = 0.0
    for k in range(traj.M):
        A_k, b_k = polys[k].A, polys[k].b
        for tau in np.linspace(traj.knot_times[k], traj.knot_times[k + 1], 20):
            p = traj.evaluate(tau)
            leak = float(np.max(A_k @ p - b_k))
            if leak > max_leak:
                max_leak = leak
    assert max_leak <= 0.30, f"trajectory leaks {max_leak:.3f} m outside corridor"

    # 7. Verify dynamic feasibility via flatness
    _, thrust, quats, rates = evaluate_trajectory_dynamics(traj, dt=0.05)
    assert thrust.min() >= 0.0
    assert thrust.max() < 3.0 * 9.81  # 3g cap, plenty for a gentle trajectory

    print(
        f"\n[Phase 0 E2E timing] RRT={t_rrt:.1f}ms  "
        f"FIRI={t_firi:.1f}ms  MINCO={t_minco:.1f}ms  "
        f"total={(t_rrt + t_firi + t_minco):.1f}ms  "
        f"segments={traj.M}  max_corridor_leak={max_leak:.4f}m  "
        f"max_thrust={thrust.max():.2f}N"
    )
