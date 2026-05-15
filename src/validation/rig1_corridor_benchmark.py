"""Rig 1: corridor escape benchmark.

See docs/MINCO_PIVOT.md §5.2.

Question
--------
Given random obstacles of increasing density, can the MINCO pipeline find
a path, and how fast?

Pipeline under test
-------------------
    obstacle point cloud
        -> VoxelMap (sparse occupancy + drone-radius dilation)
        -> RRT route
        -> FIRI convex cover (one polytope per route segment)
        -> MINCO + L-BFGS optimisation
        -> flatness-based dynamic feasibility check

Per-trial metrics
-----------------
    t_setup_ms, t_rrt_ms, t_firi_ms, t_minco_ms, t_total_ms
    n_waypoints, n_segments
    total_time_s, energy_J
    thrust_max_N, thrust_min_N
    v_max_observed, tilt_max_rad
    max_corridor_leak_m
    success (bool: leak <= tolerance and RRT found a route)

Output is a JSON file with two top-level keys:
    runs    : list of per-trial dicts
    summary : per-density aggregate (median/mean/std/min/max/count)

CLI
---
    python -m src.validation.rig1_corridor_benchmark \
        --densities 0.05,0.15,0.30,0.45 --runs 50 --output rig1.json

Notes
-----
- Each MINCO run uses scipy L-BFGS-B with finite-difference gradients (Phase 0
  baseline). That dominates the wall clock — typically ~20 s per run on a Mac
  with M=7 segments. The Phase 1 exit criterion of t_total < 50 ms at 0.30
  density needs analytical gradients (queued for a follow-on phase).
- "Potato test" Docker invocation in `docs/MINCO_PIVOT.md` §5.2 should be
  scriptable by re-running this module under an ARM/CPU-pinned container.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.single_drone.planning import (
    GCopterConfig,
    Trajectory,
    convex_cover,
    evaluate_trajectory_dynamics,
    gcopter_optimize,
    plan_path_rrt,
    rotate_vector_by_quat,
    shortcut_path,
)
from src.validation.metrics import MetricsCollector, summarise
from src.validation.obstacle_gen import clear_around, random_obstacle_field


@dataclass
class Rig1Config:
    map_size: Tuple[int, int, int] = (40, 40, 10)
    voxel_size: float = 0.5
    drone_radius_voxels: int = 1
    v_max: float = 4.0
    drone_mass: float = 1.0
    gravity: float = 9.81
    drag_coeffs: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rrt_timeout_s: float = 5.0
    gcopter_maxiter: int = 80
    gcopter_n_quad: int = 12
    start: Tuple[float, float, float] = (2.0, 10.0, 2.0)
    goal: Tuple[float, float, float] = (18.0, 10.0, 2.0)
    clear_radius: float = 1.5
    leak_tolerance_m: float = 0.30  # max acceptable corridor leak for success


def run_one_trial(
    seed: int,
    density: float,
    config: Optional[Rig1Config] = None,
) -> Dict[str, float]:
    """Execute one full RRT → FIRI → MINCO → flatness pass and return metrics."""
    if config is None:
        config = Rig1Config()
    rng = np.random.default_rng(seed)
    result: Dict[str, float] = {"seed": seed, "density": density, "success": False}

    start = np.asarray(config.start, dtype=np.float64)
    goal = np.asarray(config.goal, dtype=np.float64)

    # ---- 1. obstacle field
    t0 = time.perf_counter()
    voxel_map = random_obstacle_field(
        rng,
        size=config.map_size,
        voxel_size=config.voxel_size,
        density=density,
        clear_zones=[
            clear_around(start, config.clear_radius),
            clear_around(goal, config.clear_radius),
        ],
    )
    voxel_map.dilate(config.drone_radius_voxels)
    result["t_setup_ms"] = (time.perf_counter() - t0) * 1000.0

    if voxel_map.query(start) == 1 or voxel_map.query(goal) == 1:
        result["error"] = "endpoint_blocked_after_dilation"
        return result

    # ---- 2. RRT
    t0 = time.perf_counter()
    route = plan_path_rrt(
        start, goal, voxel_map, timeout=config.rrt_timeout_s, rng=rng
    )
    result["t_rrt_ms"] = (time.perf_counter() - t0) * 1000.0
    if not route:
        result["error"] = "rrt_failed"
        result["t_total_ms"] = result["t_rrt_ms"]
        return result
    route = shortcut_path(route, voxel_map)
    result["n_waypoints"] = len(route)

    # ---- 3. FIRI corridors
    surface = voxel_map.get_surface_points()
    t0 = time.perf_counter()
    polytopes = convex_cover(route, surface, voxel_map.world_bounds)
    result["t_firi_ms"] = (time.perf_counter() - t0) * 1000.0

    # ---- 4. MINCO + L-BFGS
    durations = [
        max(0.5, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / config.v_max)
        for a, b in zip(route[:-1], route[1:])
    ]
    durations = np.asarray(durations, dtype=np.float64)
    bc_start = np.zeros((4, 3), dtype=np.float64)  # s=3 → 4 BCs
    bc_start[0] = start
    bc_end = np.zeros((4, 3), dtype=np.float64)
    bc_end[0] = goal
    waypoints = np.asarray(route, dtype=np.float64)

    t0 = time.perf_counter()
    try:
        traj: Trajectory = gcopter_optimize(
            initial_waypoints=waypoints,
            initial_durations=durations,
            bc_start=bc_start,
            bc_end=bc_end,
            polytopes=polytopes,
            config=GCopterConfig(
                v_max=config.v_max,
                n_quad=config.gcopter_n_quad,
                maxiter=config.gcopter_maxiter,
            ),
        )
    except Exception as e:  # pragma: no cover — guarded for robustness
        result["t_minco_ms"] = (time.perf_counter() - t0) * 1000.0
        result["error"] = f"minco:{type(e).__name__}:{e}"
        result["t_total_ms"] = (
            result.get("t_rrt_ms", 0.0)
            + result.get("t_firi_ms", 0.0)
            + result["t_minco_ms"]
        )
        return result
    result["t_minco_ms"] = (time.perf_counter() - t0) * 1000.0
    result["n_segments"] = int(traj.M)
    result["total_time_s"] = float(traj.total_time)
    result["energy_J"] = float(traj.energy())

    # ---- 5. flatness + corridor diagnostics
    _, thrust, quats, _ = evaluate_trajectory_dynamics(
        traj,
        dt=0.05,
        mass=config.drone_mass,
        gravity=config.gravity,
        drag_coeffs=config.drag_coeffs,
    )
    result["thrust_max_N"] = float(thrust.max())
    result["thrust_min_N"] = float(thrust.min())

    n_samples = 200
    ts = np.linspace(0.0, traj.total_time, n_samples)
    v_norms = np.array([np.linalg.norm(traj.evaluate(t, 1)) for t in ts])
    result["v_max_observed"] = float(v_norms.max())

    z_world = np.array([0.0, 0.0, 1.0])
    cos_tilt = np.clip(
        np.array([rotate_vector_by_quat(z_world, q)[2] for q in quats]),
        -1.0, 1.0,
    )
    result["tilt_max_rad"] = float(np.arccos(cos_tilt.min()))

    max_leak = -np.inf
    for k in range(traj.M):
        A_k, b_k = polytopes[k].A, polytopes[k].b
        for t in np.linspace(traj.knot_times[k], traj.knot_times[k + 1], 20):
            p = traj.evaluate(t)
            leak = float(np.max(A_k @ p - b_k))
            if leak > max_leak:
                max_leak = leak
    result["max_corridor_leak_m"] = max_leak
    result["success"] = max_leak <= config.leak_tolerance_m

    result["t_total_ms"] = (
        result["t_rrt_ms"] + result["t_firi_ms"] + result["t_minco_ms"]
    )
    return result


def run_benchmark(
    densities: List[float],
    runs_per_density: int,
    config: Optional[Rig1Config] = None,
    base_seed: int = 1000,
    verbose: bool = True,
) -> MetricsCollector:
    if config is None:
        config = Rig1Config()
    mc = MetricsCollector()
    for d_idx, density in enumerate(densities):
        if verbose:
            print(f"\n--- density = {density:.2f} ---")
        for run_idx in range(runs_per_density):
            seed = base_seed + d_idx * 10_000 + run_idx
            row = run_one_trial(seed, density, config)
            mc.start_run(density=density, seed=seed)
            for k, v in row.items():
                if k in ("density", "seed"):
                    continue
                mc.record(k, v)
            mc.finish_run()
            if verbose:
                ok = row.get("success", False)
                t_total = row.get("t_total_ms", float("nan"))
                err = row.get("error", "")
                line = f"  run {run_idx + 1}/{runs_per_density}: "
                line += f"success={ok}  t_total={t_total:7.1f} ms"
                if err:
                    line += f"  [{err}]"
                print(line, flush=True)
    return mc


def _format_summary(summary: Dict[str, dict]) -> str:
    rows = []
    rows.append(
        f"{'group':16s}  {'runs':>5s}  {'succ%':>7s}  {'t_total median':>15s}  "
        f"{'t_rrt med':>10s}  {'t_firi med':>11s}  {'t_minco med':>12s}  "
        f"{'leak_m max':>11s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        t_total = agg.get("t_total_ms", {}).get("median", float("nan"))
        t_rrt = agg.get("t_rrt_ms", {}).get("median", float("nan"))
        t_firi = agg.get("t_firi_ms", {}).get("median", float("nan"))
        t_minco = agg.get("t_minco_ms", {}).get("median", float("nan"))
        leak = agg.get("max_corridor_leak_m", {}).get("max", float("nan"))
        rows.append(
            f"{group:16s}  {n:5d}  {sr:6.1f}%  {t_total:15.1f}  "
            f"{t_rrt:10.1f}  {t_firi:11.1f}  {t_minco:12.1f}  {leak:11.4f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rig 1: corridor escape benchmark")
    parser.add_argument(
        "--densities",
        type=str,
        default="0.05,0.15,0.30,0.45",
        help="Comma-separated obstacle densities to sweep (default: 0.05,0.15,0.30,0.45)",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Runs per density (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rig1_results.json",
        help="JSON output path (default: rig1_results.json)",
    )
    parser.add_argument(
        "--map-size",
        type=str,
        default="40,40,10",
        help="Voxel grid dimensions Lx,Ly,Lz (default: 40,40,10)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.5, help="Voxel edge (m)"
    )
    parser.add_argument(
        "--maxiter", type=int, default=80, help="L-BFGS max iterations"
    )
    parser.add_argument(
        "--v-max", type=float, default=4.0, help="Velocity limit (m/s)"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    densities = [float(x) for x in args.densities.split(",")]
    map_size = tuple(int(x) for x in args.map_size.split(","))
    if len(map_size) != 3:
        print("map-size must be Lx,Ly,Lz (three ints)", file=sys.stderr)
        return 2

    config = Rig1Config(
        map_size=map_size,
        voxel_size=args.voxel_size,
        gcopter_maxiter=args.maxiter,
        v_max=args.v_max,
    )

    print(
        f"Rig 1 — densities {densities}, {args.runs} runs each, "
        f"map={map_size}, voxel_size={args.voxel_size}m, "
        f"v_max={args.v_max}m/s, gcopter_maxiter={args.maxiter}"
    )

    mc = run_benchmark(
        densities=densities,
        runs_per_density=args.runs,
        config=config,
        verbose=not args.quiet,
    )

    mc.export_json(args.output, label_keys=["density"])
    print(f"\nResults written to {args.output}")

    summary = summarise(mc.runs, label_keys=["density"])
    print("\n=== Summary ===")
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
