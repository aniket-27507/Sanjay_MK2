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
- The MINCO solver uses scipy L-BFGS-B with analytical gradients
  (energy + corridor + velocity assembled in `gcopter._cost_and_grad`;
  swarm penalty in `src.swarm.swarm_penalty`). The finite-difference path
  the original Phase 0 plan described is no longer in use; the FD oracle
  in `tests/test_minco_gradients_e2e.py` only verifies correctness.
  Wall-clock per trial at density 0.30, `gcopter_maxiter=30`, voxel-grid
  20³ is ~50-250 ms — dominated by the L-BFGS line search, not the
  per-evaluation cost.
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
    plan_path_rrt_connect,
    rotate_vector_by_quat,
    shortcut_path,
)
from src.validation.metrics import MetricsCollector, summarise
from src.validation.obstacle_gen import (
    clear_around,
    measured_density,
    random_obstacle_field,
)


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
    rrt_step_size: Optional[float] = None    # None → planner default
    rrt_planner: str = "rrt_connect"          # "rrt" | "rrt_connect"
    gcopter_maxiter: int = 80
    gcopter_n_quad: int = 12
    start: Tuple[float, float, float] = (2.0, 10.0, 2.0)
    goal: Tuple[float, float, float] = (18.0, 10.0, 2.0)
    clear_radius: float = 1.5
    leak_tolerance_m: float = 0.30  # max acceptable corridor leak for success

    # The MINCO_PIVOT.md §5.2 density table refers to the obstacle fraction
    # of the *dilated* voxel map (the planner's view), not raw obstacle
    # placement. With drone_radius_voxels=1 a single placed obstacle
    # becomes a 3×3×3 occupied block, so raw density 0.30 produces ~99 %
    # post-dilation occupancy and no path exists. `density_is_post_dilation`
    # toggles a bisection that picks the raw obstacle fraction to hit the
    # requested *dilated* density. Set to False to use raw density directly
    # (for legacy comparisons).
    density_is_post_dilation: bool = True
    density_bisect_tol: float = 0.02
    density_bisect_max_iter: int = 8


def _build_voxel_map_for_target_density(
    rng: np.random.Generator,
    target_density: float,
    config: Rig1Config,
    start: np.ndarray,
    goal: np.ndarray,
) -> Tuple[object, float]:
    """Return a dilated VoxelMap whose post-dilation occupancy matches
    `target_density` (within `config.density_bisect_tol`).

    Bisects on the raw obstacle fraction passed to `random_obstacle_field`.
    Each candidate map is built with the same rng state (cloned) so the
    bisection itself is reproducible.
    """
    clear_zones = [
        clear_around(start, config.clear_radius),
        clear_around(goal, config.clear_radius),
    ]
    # Each random_obstacle_field call consumes rng. To make bisection
    # behave like "we tried different raw densities on the same problem",
    # snapshot the seed and rebuild a fresh rng each iteration.
    snapshot_seed = int(rng.integers(1 << 31))

    def _build(raw: float):
        local_rng = np.random.default_rng(snapshot_seed)
        m = random_obstacle_field(
            local_rng,
            size=config.map_size,
            voxel_size=config.voxel_size,
            density=raw,
            clear_zones=clear_zones,
        )
        m.dilate(config.drone_radius_voxels)
        return m, measured_density(m)

    if not config.density_is_post_dilation or config.drone_radius_voxels <= 0:
        m, achieved = _build(target_density)
        return m, achieved

    lo, hi = 0.0, min(1.0, target_density)
    best_map, best_achieved = _build(hi)
    for _ in range(config.density_bisect_max_iter):
        if abs(best_achieved - target_density) <= config.density_bisect_tol:
            break
        if best_achieved > target_density:
            hi = (lo + hi) / 2.0
            best_map, best_achieved = _build(hi)
        else:
            lo = hi
            hi = min(1.0, hi * 1.5 + 0.01)
            best_map, best_achieved = _build(hi)
    return best_map, best_achieved


def run_one_trial(
    seed: int,
    density: float,
    config: Optional[Rig1Config] = None,
    keep_record: bool = False,
) -> Dict[str, float]:
    """Execute one full RRT → FIRI → MINCO → flatness pass and return metrics.

    When `keep_record=True`, `result["viz_record"]` carries the data the
    Plotly visualiser needs (obstacles, RRT route, polytope AABBs,
    trajectory samples).
    """
    if config is None:
        config = Rig1Config()
    rng = np.random.default_rng(seed)
    result: Dict[str, float] = {"seed": seed, "density": density, "success": False}
    viz: Optional[Dict] = (
        {"density": density, "seed": seed, "start": None, "goal": None}
        if keep_record else None
    )

    start = np.asarray(config.start, dtype=np.float64)
    goal = np.asarray(config.goal, dtype=np.float64)

    # ---- 1. obstacle field
    t0 = time.perf_counter()
    voxel_map, achieved_density = _build_voxel_map_for_target_density(
        rng, density, config, start, goal
    )
    result["achieved_dilated_density"] = achieved_density
    result["t_setup_ms"] = (time.perf_counter() - t0) * 1000.0

    if viz is not None:
        viz["start"] = start.tolist()
        viz["goal"] = goal.tolist()
        viz["achieved_dilated_density"] = float(achieved_density)
        # surface voxels for the obstacle scatter
        surf = voxel_map.get_surface_points()
        if surf is not None and len(surf) > 0:
            arr = np.asarray(surf, dtype=np.float64)
            # cap point cloud size so HTML stays light
            if arr.shape[0] > 4000:
                idx = np.random.default_rng(seed).choice(
                    arr.shape[0], size=4000, replace=False
                )
                arr = arr[idx]
            viz["obstacle_points"] = arr.tolist()
        else:
            viz["obstacle_points"] = []

    if voxel_map.query(start) == 1 or voxel_map.query(goal) == 1:
        result["error"] = "endpoint_blocked_after_dilation"
        return result

    # ---- 2. RRT
    t0 = time.perf_counter()
    if config.rrt_planner == "rrt_connect":
        route = plan_path_rrt_connect(
            start,
            goal,
            voxel_map,
            timeout=config.rrt_timeout_s,
            step_size=config.rrt_step_size,
            rng=rng,
        )
    elif config.rrt_planner == "rrt":
        route = plan_path_rrt(
            start,
            goal,
            voxel_map,
            timeout=config.rrt_timeout_s,
            step_size=config.rrt_step_size,
            rng=rng,
        )
    else:
        raise ValueError(
            f"unknown rrt_planner {config.rrt_planner!r}; choose 'rrt' or 'rrt_connect'"
        )
    result["t_rrt_ms"] = (time.perf_counter() - t0) * 1000.0
    if not route:
        result["error"] = "rrt_failed"
        result["t_total_ms"] = result["t_rrt_ms"]
        if viz is not None:
            result["viz_record"] = viz
        return result
    if viz is not None:
        viz["rrt_route"] = [list(map(float, p)) for p in route]
    route = shortcut_path(route, voxel_map)
    if viz is not None:
        viz["shortcut_route"] = [list(map(float, p)) for p in route]
    result["n_waypoints"] = len(route)

    # ---- 3. FIRI corridors
    surface = voxel_map.get_surface_points()
    t0 = time.perf_counter()
    polytopes = convex_cover(route, surface, voxel_map.world_bounds)
    result["t_firi_ms"] = (time.perf_counter() - t0) * 1000.0

    if viz is not None:
        # AABB of each polytope: from the half-space inequalities A x <= b
        # the smallest enclosing axis-aligned box. We extract by solving
        # min/max on each axis subject to A x <= b. Fast approximation:
        # use the convex_cover seed segment plus a uniform margin of
        # ~half the diagonal of the bounding box of the polytope's vertices.
        # We just bound by the segment endpoints + a 2m halo here for
        # visualization purposes.
        boxes = []
        for k, poly in enumerate(polytopes):
            seg_lo = np.asarray(route[k], dtype=np.float64)
            seg_hi = np.asarray(route[k + 1], dtype=np.float64)
            lo = np.minimum(seg_lo, seg_hi) - 1.5
            hi = np.maximum(seg_lo, seg_hi) + 1.5
            boxes.append({"min": lo.tolist(), "max": hi.tolist()})
        viz["polytope_boxes"] = boxes

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

    if viz is not None:
        n_viz_samples = 120
        ts_viz = np.linspace(0.0, traj.total_time, n_viz_samples)
        samples = []
        for t in ts_viz:
            p = traj.evaluate(t, 0)
            v = traj.evaluate(t, 1)
            samples.append(
                {
                    "t": float(t),
                    "p": [float(p[0]), float(p[1]), float(p[2])],
                    "v": float(np.linalg.norm(v)),
                }
            )
        viz["trajectory_samples"] = samples
        viz["success"] = bool(result["success"])
        result["viz_record"] = viz

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
    parser.add_argument(
        "--planner",
        type=str,
        default="rrt_connect",
        choices=["rrt", "rrt_connect"],
        help="Path planner (default: rrt_connect — much faster on clutter).",
    )
    parser.add_argument(
        "--rrt-step-size",
        type=float,
        default=None,
        help="RRT extension length in metres (default: planner-specific).",
    )
    parser.add_argument(
        "--rrt-timeout",
        type=float,
        default=5.0,
        help="RRT wall-clock budget (s).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="If set, write a PNG headline chart at this path next to the JSON.",
    )
    parser.add_argument(
        "--viz",
        type=str,
        default="",
        help="If set, run one extra detailed trial and write an interactive "
        "Plotly HTML at this path. Uses --viz-density (default 0.30) and "
        "--viz-seed (default 12345).",
    )
    parser.add_argument(
        "--viz-density", type=float, default=0.30,
        help="Density to use for the viz trial (default: 0.30).",
    )
    parser.add_argument(
        "--viz-seed", type=int, default=12345,
        help="Seed to use for the viz trial (default: 12345).",
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
        rrt_planner=args.planner,
        rrt_step_size=args.rrt_step_size,
        rrt_timeout_s=args.rrt_timeout,
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

    if args.plot:
        from src.validation.plots import emit_plot
        emit_plot("rig1", mc.runs, args.plot)
        print(f"Plot written to {args.plot}")

    if args.viz:
        from src.validation.visualize import emit_viz
        row = run_one_trial(
            args.viz_seed, args.viz_density, config, keep_record=True,
        )
        record = row.get("viz_record")
        if record is None:
            print(
                f"Viz trial failed (no record produced; error="
                f"{row.get('error', '?')})",
                file=sys.stderr,
            )
        else:
            emit_viz("rig1", record, args.viz)
            print(f"Viz written to {args.viz}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
