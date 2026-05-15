"""Rig 6: environmental disturbance.

See docs/MINCO_PIVOT.md §5.7.

Question
--------
How robust is the system to wind gusts, fog, and sensor failure?

Pipeline under test
-------------------
    Per scenario, one drone flying a straight corridor between `start`
    and `goal`:

    1. Generate a MINCO trajectory (single segment, straight, low
       optimisation budget — corridor is fat enough to absorb tracking
       error).
    2. Per tick:
         - desired_pos, desired_vel = trajectory.evaluate(t, derivative)
         - wind_accel = WindModel.compute_acceleration(pos, vel, dt)
         - tracker correction: a = k_p (desired_pos - pos)
                                  + k_d (desired_vel - vel)
                                  + wind_accel
         - integrate pos and vel
         - sample noisy depth from `apply()` with scenario range / noise
         - compute corridor clearance and sensor valid-fraction

Scenarios
---------
    calm        base 0.5 m/s wind
    breezy      base 3 m/s + 5 m/s gust max
    windy       base 5 m/s + 8 m/s gust max
    foggy       depth max range 3 m
    rain        depth max range 5 m + 2× noise coefficient
    sensor_fail depth max range 0 m → forced sensor failure

Metrics
-------
    tracking_error_mean_m, tracking_error_max_m
    corridor_clearance_min_m, corridor_clearance_mean_m
    corridor_breached (bool)
    depth_valid_fraction_mean (0..1)
    sensor_failed (bool: valid fraction dropped below threshold any tick)
    wind_speed_max_observed (m/s)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.types.drone_types import Vector3
from src.simulation.physics.wind_model import WindConfig, WindModel
from src.single_drone.planning import (
    GCopterConfig,
    Polytope,
    Trajectory,
    gcopter_optimize,
)
from src.validation.depth_noise_model import DepthNoiseConfig, apply, valid_fraction
from src.validation.metrics import MetricsCollector, summarise


SCENARIOS = ("calm", "breezy", "windy", "foggy", "rain", "sensor_fail")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Rig6Config:
    # geometry
    start: Tuple[float, float, float] = (-15.0, 0.0, 5.0)
    goal: Tuple[float, float, float] = (15.0, 0.0, 5.0)
    corridor_half_extent: Tuple[float, float, float] = (3.0, 3.0, 2.0)

    # trajectory
    v_max: float = 4.0
    gcopter_maxiter: int = 12

    # tracker (PD controller)
    k_p: float = 4.0
    k_d: float = 2.0
    drone_mass_kg: float = 0.5

    # simulation
    dt: float = 0.1
    sample_depth_pixels: int = 32 * 32

    # depth sensor failure threshold
    sensor_valid_threshold: float = 0.3   # below this → failed
    sensor_failure_max_range_m: float = 0.05  # treated as "camera dead"


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------


def scenario_to_models(
    scenario: str, config: Rig6Config, rng: np.random.Generator
) -> Tuple[WindConfig, DepthNoiseConfig, str]:
    """Return (WindConfig, DepthNoiseConfig, label) for the named scenario.

    Each scenario maps directly to MINCO_PIVOT.md §5.7. Seeds are derived
    from `rng` so the same scenario at different seeds picks different
    gust timing / dropout patterns.
    """
    seed = int(rng.integers(1 << 31))
    if scenario == "calm":
        return (
            WindConfig(base_speed_ms=0.5, gust_max_ms=1.0,
                       gust_probability_per_sec=0.05,
                       drone_mass_kg=config.drone_mass_kg, seed=seed),
            DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.005, dropout_pct=2.0),
            "calm",
        )
    if scenario == "breezy":
        return (
            WindConfig(base_speed_ms=3.0, gust_max_ms=5.0,
                       gust_probability_per_sec=0.10,
                       drone_mass_kg=config.drone_mass_kg, seed=seed),
            DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.005, dropout_pct=3.0),
            "breezy",
        )
    if scenario == "windy":
        return (
            WindConfig(base_speed_ms=5.0, gust_max_ms=8.0,
                       gust_probability_per_sec=0.15,
                       drone_mass_kg=config.drone_mass_kg, seed=seed),
            DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.005, dropout_pct=5.0),
            "windy",
        )
    if scenario == "foggy":
        return (
            WindConfig(base_speed_ms=1.0, gust_max_ms=2.0,
                       drone_mass_kg=config.drone_mass_kg, seed=seed),
            DepthNoiseConfig(max_range_m=3.0, noise_coeff=0.005, dropout_pct=10.0),
            "foggy",
        )
    if scenario == "rain":
        return (
            WindConfig(base_speed_ms=2.0, gust_max_ms=4.0,
                       drone_mass_kg=config.drone_mass_kg, seed=seed),
            DepthNoiseConfig(max_range_m=5.0, noise_coeff=0.010, dropout_pct=15.0),
            "rain",
        )
    if scenario == "sensor_fail":
        return (
            WindConfig(base_speed_ms=1.0, gust_max_ms=2.0,
                       drone_mass_kg=config.drone_mass_kg, seed=seed),
            DepthNoiseConfig(
                max_range_m=config.sensor_failure_max_range_m,
                noise_coeff=0.020,
                dropout_pct=80.0,
            ),
            "sensor_fail",
        )
    raise ValueError(f"unknown scenario {scenario!r}")


# ---------------------------------------------------------------------------
# Trajectory + corridor setup
# ---------------------------------------------------------------------------


def _corridor_box(
    start: np.ndarray, goal: np.ndarray, half_extent: Sequence[float]
) -> Polytope:
    lo = np.minimum(start, goal) - np.asarray(half_extent, dtype=np.float64)
    hi = np.maximum(start, goal) + np.asarray(half_extent, dtype=np.float64)
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


def _make_trajectory(
    start: np.ndarray, goal: np.ndarray, config: Rig6Config
) -> Tuple[Trajectory, Polytope]:
    s = 3
    D = 3
    M = 1
    durations = np.array(
        [max(0.5, float(np.linalg.norm(goal - start)) / config.v_max)],
        dtype=np.float64,
    )
    waypoints = np.stack([start, goal], axis=0)
    bc_start = np.zeros((s + 1, D), dtype=np.float64)
    bc_start[0] = start
    bc_end = np.zeros((s + 1, D), dtype=np.float64)
    bc_end[0] = goal
    poly = _corridor_box(start, goal, config.corridor_half_extent)
    traj = gcopter_optimize(
        initial_waypoints=waypoints,
        initial_durations=durations,
        bc_start=bc_start,
        bc_end=bc_end,
        polytopes=[poly],
        config=GCopterConfig(
            v_max=config.v_max,
            n_quad=6,
            maxiter=config.gcopter_maxiter,
        ),
    )
    return traj, poly


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def _signed_corridor_clearance(p: np.ndarray, poly: Polytope) -> float:
    """Positive when inside (= margin to nearest face); negative when
    outside the polytope. Same sign convention as Rig 1's leak metric
    but inverted to read as 'clearance'."""
    residual = poly.A @ p - poly.b
    leak = float(np.max(residual))
    return -leak


def _ground_truth_depth_field(
    pos: np.ndarray, goal: np.ndarray, n_pixels: int, rng: np.random.Generator
) -> np.ndarray:
    """Synthesise a true depth array — distance to the goal forward face,
    with mild per-pixel variation. Enough structure for the noise model
    to produce a realistic valid-fraction.
    """
    base = float(np.linalg.norm(goal - pos))
    return np.clip(rng.normal(base, base * 0.05, size=n_pixels), 0.1, 30.0)


def run_one_trial(
    seed: int,
    scenario: str,
    config: Optional[Rig6Config] = None,
) -> Dict[str, float]:
    if config is None:
        config = Rig6Config()
    rng = np.random.default_rng(seed)

    start = np.asarray(config.start, dtype=np.float64)
    goal = np.asarray(config.goal, dtype=np.float64)
    traj, poly = _make_trajectory(start, goal, config)

    wind_cfg, depth_cfg, label = scenario_to_models(scenario, config, rng)
    wind = WindModel(wind_cfg)

    pos = start.copy()
    vel = np.zeros(3, dtype=np.float64)

    n_steps = int(np.ceil(traj.total_time / config.dt)) + 1
    track_errs: List[float] = []
    clearances: List[float] = []
    depth_valid_fracs: List[float] = []
    wind_speeds: List[float] = []

    sensor_failed = False
    corridor_breached = False

    for step in range(n_steps):
        t = min(step * config.dt, traj.total_time)
        desired_pos = traj.evaluate(t, 0)
        desired_vel = traj.evaluate(t, 1)

        # PD tracker + wind acceleration
        accel = (
            config.k_p * (desired_pos - pos)
            + config.k_d * (desired_vel - vel)
        )
        v3_pos = Vector3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        v3_vel = Vector3(x=float(vel[0]), y=float(vel[1]), z=float(vel[2]))
        w_acc_v3 = wind.compute_acceleration(v3_pos, v3_vel, config.dt)
        w_acc = np.array([w_acc_v3.x, w_acc_v3.y, w_acc_v3.z], dtype=np.float64)
        accel = accel + w_acc

        # integrate
        vel = vel + accel * config.dt
        pos = pos + vel * config.dt

        # metrics
        track_errs.append(float(np.linalg.norm(pos - desired_pos)))
        clr = _signed_corridor_clearance(pos, poly)
        clearances.append(clr)
        if clr < 0.0:
            corridor_breached = True
        wind_speeds.append(float(np.linalg.norm(w_acc) * config.drone_mass_kg))  # ≈ wind force / m

        # depth sensor
        true_depth = _ground_truth_depth_field(
            pos, goal, config.sample_depth_pixels, rng
        )
        noisy = apply(true_depth, depth_cfg, rng=rng)
        vf = valid_fraction(noisy, depth_cfg)
        depth_valid_fracs.append(vf)
        if vf < config.sensor_valid_threshold:
            sensor_failed = True

    track_arr = np.asarray(track_errs)
    clr_arr = np.asarray(clearances)
    vf_arr = np.asarray(depth_valid_fracs)

    result: Dict[str, float] = {
        "seed": seed,
        "scenario": label,
        "tracking_error_mean_m": float(track_arr.mean()) if track_arr.size else 0.0,
        "tracking_error_max_m": float(track_arr.max()) if track_arr.size else 0.0,
        "corridor_clearance_min_m": float(clr_arr.min()) if clr_arr.size else 0.0,
        "corridor_clearance_mean_m": float(clr_arr.mean()) if clr_arr.size else 0.0,
        "corridor_breached": bool(corridor_breached),
        "depth_valid_fraction_mean": float(vf_arr.mean()) if vf_arr.size else 0.0,
        "depth_valid_fraction_min": float(vf_arr.min()) if vf_arr.size else 0.0,
        "sensor_failed": bool(sensor_failed),
        "wind_speed_max_observed_ms": (
            float(max(wind_speeds)) if wind_speeds else 0.0
        ),
        "trajectory_time_s": float(traj.total_time),
        "success": (not corridor_breached) and (not sensor_failed),
    }
    return result


def run_benchmark(
    scenarios: Sequence[str],
    runs_per_scenario: int,
    config: Optional[Rig6Config] = None,
    base_seed: int = 6000,
    verbose: bool = True,
) -> MetricsCollector:
    if config is None:
        config = Rig6Config()
    mc = MetricsCollector()
    for idx_s, scenario in enumerate(scenarios):
        if verbose:
            print(f"\n--- scenario={scenario} ---")
        for run_idx in range(runs_per_scenario):
            seed = base_seed + idx_s * 1000 + run_idx
            row = run_one_trial(seed, scenario, config)
            mc.start_run(scenario=scenario, seed=seed)
            for k, v in row.items():
                if k in ("scenario", "seed"):
                    continue
                mc.record(k, v)
            mc.finish_run()
            if verbose:
                te = row["tracking_error_mean_m"]
                clr = row["corridor_clearance_min_m"]
                vf = row["depth_valid_fraction_mean"]
                print(
                    f"  run {run_idx + 1}/{runs_per_scenario}: "
                    f"track_err={te:5.3f}m  clr_min={clr:+5.2f}m  "
                    f"vf={vf:5.3f}  breached={row['corridor_breached']}  "
                    f"sensor_failed={row['sensor_failed']}",
                    flush=True,
                )
    return mc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_summary(summary: Dict[str, dict]) -> str:
    rows = []
    rows.append(
        f"{'group':28s}  {'runs':>5s}  {'succ%':>7s}  "
        f"{'track med':>10s}  {'clr_min min':>12s}  "
        f"{'vf med':>8s}  {'sensor_fail%':>13s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        te = agg.get("tracking_error_mean_m", {}).get("median", float("nan"))
        clr = agg.get("corridor_clearance_min_m", {}).get("min", float("nan"))
        vf = agg.get("depth_valid_fraction_mean", {}).get("median", float("nan"))
        # sensor_failed is a bool — its mean over runs = failure rate
        sf_mean = 0.0
        if "sensor_failed" in agg:
            sf_mean = agg["sensor_failed"].get("mean", 0.0) * 100.0
        rows.append(
            f"{group:28s}  {n:5d}  {sr:6.1f}%  "
            f"{te:10.3f}  {clr:+12.3f}  {vf:8.3f}  {sf_mean:13.1f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rig 6: environmental disturbance")
    parser.add_argument(
        "--scenarios", type=str,
        default="calm,breezy,windy,foggy,rain,sensor_fail",
        help="Comma-separated scenarios from " + ",".join(SCENARIOS),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--v-max", type=float, default=4.0)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="rig6_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            print(f"unknown scenario {s!r}; choose from {SCENARIOS}", file=sys.stderr)
            return 2

    config = Rig6Config(
        v_max=args.v_max,
        gcopter_maxiter=args.maxiter,
        dt=args.dt,
    )

    print(
        f"Rig 6 — scenarios={scenarios}, {args.runs} runs each, "
        f"v_max={args.v_max}m/s, maxiter={args.maxiter}, dt={args.dt}s"
    )

    mc = run_benchmark(
        scenarios=scenarios,
        runs_per_scenario=args.runs,
        config=config,
        verbose=not args.quiet,
    )

    mc.export_json(args.output, label_keys=["scenario"])
    print(f"\nResults written to {args.output}")

    summary = summarise(mc.runs, label_keys=["scenario"])
    print("\n=== Summary ===")
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
