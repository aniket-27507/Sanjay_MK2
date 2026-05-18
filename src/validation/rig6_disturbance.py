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

    # GPS-only RTL behaviour on persistent sensor failure
    sensor_fail_persistence_ticks: int = 3   # how many bad ticks before RTL
    rtl_speed_ms: float = 2.0                 # conservative cruise back to start


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
    """Synthesise a true depth array for the OAK-D Lite stereo camera.

    Real-world outdoor depth scenes are dominated by **near-field
    structure** (ground, walls, foliage at 0.5-8 m). Using "distance to
    goal" would put every pixel at 30+ m, beyond *every* configured
    scenario's max_range (10 m calm, 3 m fog, 0.05 m sensor_fail), so all
    scenarios would falsely show vf≈0.

    Mix a near-field band (most pixels, what the camera reliably sees)
    with a far-field tail (a few pixels looking down the corridor toward
    the goal). The reliable range against each scenario's max_range is
    what makes calm pass and fog/sensor_fail fail.
    """
    near = rng.uniform(0.5, 8.0, size=int(n_pixels * 0.8))
    far_base = float(np.linalg.norm(goal - pos))
    far = np.clip(
        rng.normal(far_base, max(far_base * 0.05, 0.1), size=n_pixels - near.size),
        0.5, 30.0,
    )
    return np.concatenate([near, far])


def run_one_trial(
    seed: int,
    scenario: str,
    config: Optional[Rig6Config] = None,
    wind_cfg_override: Optional[WindConfig] = None,
    depth_cfg_override: Optional[DepthNoiseConfig] = None,
    label_override: Optional[str] = None,
    keep_record: bool = False,
) -> Dict[str, float]:
    if config is None:
        config = Rig6Config()
    rng = np.random.default_rng(seed)

    start = np.asarray(config.start, dtype=np.float64)
    goal = np.asarray(config.goal, dtype=np.float64)
    traj, poly = _make_trajectory(start, goal, config)

    if wind_cfg_override is not None or depth_cfg_override is not None:
        # Sweep mode — caller has full control. Fall back to scenario for
        # anything not overridden, then apply patches.
        base_wind, base_depth, base_label = scenario_to_models(scenario, config, rng)
        wind_cfg = wind_cfg_override if wind_cfg_override is not None else base_wind
        depth_cfg = depth_cfg_override if depth_cfg_override is not None else base_depth
        label = label_override if label_override is not None else base_label
    else:
        wind_cfg, depth_cfg, label = scenario_to_models(scenario, config, rng)
    wind = WindModel(wind_cfg)

    pos = start.copy()
    vel = np.zeros(3, dtype=np.float64)

    n_steps = int(np.ceil(traj.total_time / config.dt)) + 1
    track_errs: List[float] = []
    clearances: List[float] = []
    depth_valid_fracs: List[float] = []
    wind_speeds: List[float] = []
    viz_samples: Optional[List[Dict]] = [] if keep_record else None

    sensor_failed = False
    corridor_breached = False

    # GPS-only RTL state machine — when sensor stays bad for N consecutive
    # ticks, switch to a straight-line return to `start` at conservative
    # `rtl_speed_ms`. The MINCO trajectory is abandoned for the rest of
    # the run.
    bad_streak = 0
    rtl_triggered = False
    rtl_trigger_t: float = float("nan")
    rtl_completed = False

    for step in range(n_steps):
        t = min(step * config.dt, traj.total_time)

        if not rtl_triggered:
            # PD tracker on the MINCO trajectory + wind acceleration
            desired_pos = traj.evaluate(t, 0)
            desired_vel = traj.evaluate(t, 1)
            accel = (
                config.k_p * (desired_pos - pos)
                + config.k_d * (desired_vel - vel)
            )
        else:
            # GPS-only RTL: head straight for `start` at `rtl_speed_ms`
            direction = start - pos
            dist = float(np.linalg.norm(direction))
            if dist < 0.3:
                rtl_completed = True
                desired_pos = start
                desired_vel = np.zeros(3)
            else:
                desired_pos = pos + direction / max(dist, 1e-6) * 0.5
                desired_vel = direction / max(dist, 1e-6) * config.rtl_speed_ms
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

        if viz_samples is not None:
            viz_samples.append({
                "t": float(t),
                "p": [float(pos[0]), float(pos[1]), float(pos[2])],
                "desired_p": [
                    float(desired_pos[0]),
                    float(desired_pos[1]),
                    float(desired_pos[2]),
                ],
                "wind": [float(w_acc[0]), float(w_acc[1]), float(w_acc[2])],
            })

        # metrics — only count corridor clearance against the original
        # plan; once we're in RTL, corridor breach is expected and not a
        # failure of the disturbance pipeline.
        if not rtl_triggered:
            track_errs.append(float(np.linalg.norm(pos - desired_pos)))
            clr = _signed_corridor_clearance(pos, poly)
            clearances.append(clr)
            if clr < 0.0:
                corridor_breached = True
        wind_speeds.append(float(np.linalg.norm(w_acc) * config.drone_mass_kg))

        # depth sensor
        true_depth = _ground_truth_depth_field(
            pos, goal, config.sample_depth_pixels, rng
        )
        noisy = apply(true_depth, depth_cfg, rng=rng)
        vf = valid_fraction(noisy, depth_cfg)
        depth_valid_fracs.append(vf)
        if vf < config.sensor_valid_threshold:
            sensor_failed = True
            bad_streak += 1
            if (
                not rtl_triggered
                and bad_streak >= config.sensor_fail_persistence_ticks
            ):
                rtl_triggered = True
                rtl_trigger_t = float(t)
        else:
            bad_streak = 0

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
        "rtl_triggered": bool(rtl_triggered),
        "rtl_trigger_time_s": rtl_trigger_t,
        "rtl_completed": bool(rtl_completed),
        "wind_speed_max_observed_ms": (
            float(max(wind_speeds)) if wind_speeds else 0.0
        ),
        "trajectory_time_s": float(traj.total_time),
        # success: clean run is no breach and no sensor failure; a sensor
        # failure that triggers a *completed* RTL is treated as a safe
        # graceful degradation, not a pipeline failure.
        "success": (
            (not corridor_breached) and
            (
                (not sensor_failed)
                or (rtl_triggered and rtl_completed)
            )
        ),
    }

    if viz_samples is not None:
        he = np.asarray(config.corridor_half_extent, dtype=np.float64)
        result["viz_record"] = {
            "scenario": label,
            "sample_dt_s": float(config.dt),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "corridor_min": (np.minimum(start, goal) - he).tolist(),
            "corridor_max": (np.maximum(start, goal) + he).tolist(),
            "trajectory_samples": viz_samples,
            "rtl_trigger_time_s": (
                float(rtl_trigger_t)
                if np.isfinite(rtl_trigger_t)
                else None
            ),
            "tracking_error_max_m": result["tracking_error_max_m"],
            "corridor_clearance_min_m": result["corridor_clearance_min_m"],
            "corridor_breached": bool(result["corridor_breached"]),
            "depth_valid_fraction_mean": result["depth_valid_fraction_mean"],
            "depth_valid_fraction_min": result["depth_valid_fraction_min"],
            "sensor_failed": bool(result["sensor_failed"]),
            "rtl_triggered": bool(result["rtl_triggered"]),
            "wind_speed_max_observed_ms": result["wind_speed_max_observed_ms"],
            "success": bool(result["success"]),
        }
    return result


def _failure_rate(rows: Sequence[Dict[str, float]], key: str) -> float:
    """Fraction of rows where boolean `key` is True. NaN if no rows."""
    if not rows:
        return float("nan")
    fails = sum(1 for r in rows if bool(r.get(key, False)))
    return fails / len(rows)


def sweep_wind(
    wind_speeds_ms: Sequence[float],
    runs_per_step: int,
    config: Optional[Rig6Config] = None,
    base_seed: int = 6500,
    failure_rate_threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[float, MetricsCollector]:
    """Sweep wind base_speed_ms — return (safe_wind_limit_ms, collector).

    `safe_wind_limit_ms` is the largest wind speed at which the corridor
    breach failure rate stays below `failure_rate_threshold` (default 50%).
    Returns NaN if even the lowest tested speed already breaks the
    pipeline.
    """
    if config is None:
        config = Rig6Config()
    mc = MetricsCollector()
    safe_limit = float("nan")
    for ws in wind_speeds_ms:
        # build a wind config tuned for the swept speed; keep depth at OAK-D
        # defaults so we isolate the wind variable
        wind_cfg = WindConfig(
            base_speed_ms=float(ws),
            gust_max_ms=max(1.0, 1.5 * float(ws)),
            gust_probability_per_sec=0.15,
            drone_mass_kg=config.drone_mass_kg,
            seed=0,  # placeholder; per-run seed below
        )
        depth_cfg = DepthNoiseConfig(max_range_m=10.0, noise_coeff=0.005, dropout_pct=2.0)
        per_step_rows: List[Dict[str, float]] = []
        for run_idx in range(runs_per_step):
            seed = base_seed + int(ws * 1000) + run_idx
            wind_cfg.seed = seed
            row = run_one_trial(
                seed,
                "calm",
                config,
                wind_cfg_override=wind_cfg,
                depth_cfg_override=depth_cfg,
                label_override=f"wind_{ws:.0f}",
            )
            row["wind_speed_ms"] = float(ws)
            per_step_rows.append(row)
            mc.start_run(wind_speed_ms=float(ws), seed=seed)
            for k, v in row.items():
                if k in ("seed",):
                    continue
                mc.record(k, v)
            mc.finish_run()
        fr = _failure_rate(per_step_rows, "corridor_breached")
        if verbose:
            print(
                f"  wind={ws:5.1f} m/s  breach_rate={fr*100:5.1f}%  "
                f"track_med={np.median([r['tracking_error_mean_m'] for r in per_step_rows]):5.3f}m"
            )
        if fr < failure_rate_threshold:
            safe_limit = float(ws)
        # we don't early-exit — the full curve is useful in the JSON
    return safe_limit, mc


def sweep_depth(
    depth_ranges_m: Sequence[float],
    runs_per_step: int,
    config: Optional[Rig6Config] = None,
    base_seed: int = 6600,
    failure_rate_threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[float, MetricsCollector]:
    """Sweep depth max_range_m — return (depth_range_threshold_m, mc).

    `depth_range_threshold_m` is the smallest range at which the sensor
    failure rate stays below `failure_rate_threshold`. Below that range
    the camera no longer reliably reaches the goal-distance silhouette.
    NaN if even the largest tested range still fails (shouldn't happen
    on the spec's 1..10 m sweep).
    """
    if config is None:
        config = Rig6Config()
    mc = MetricsCollector()
    threshold = float("nan")
    sorted_ranges = sorted(depth_ranges_m)  # increasing
    for r_m in sorted_ranges:
        wind_cfg = WindConfig(
            base_speed_ms=1.0, gust_max_ms=2.0,
            drone_mass_kg=config.drone_mass_kg, seed=0,
        )
        depth_cfg = DepthNoiseConfig(
            max_range_m=float(r_m), noise_coeff=0.005, dropout_pct=2.0,
        )
        per_step_rows: List[Dict[str, float]] = []
        for run_idx in range(runs_per_step):
            seed = base_seed + int(r_m * 1000) + run_idx
            wind_cfg.seed = seed
            row = run_one_trial(
                seed,
                "calm",
                config,
                wind_cfg_override=wind_cfg,
                depth_cfg_override=depth_cfg,
                label_override=f"depth_{r_m:.0f}",
            )
            row["depth_range_m"] = float(r_m)
            per_step_rows.append(row)
            mc.start_run(depth_range_m=float(r_m), seed=seed)
            for k, v in row.items():
                if k in ("seed",):
                    continue
                mc.record(k, v)
            mc.finish_run()
        fr = _failure_rate(per_step_rows, "sensor_failed")
        if verbose:
            print(
                f"  depth={r_m:5.1f} m  sensor_fail_rate={fr*100:5.1f}%  "
                f"vf_med={np.median([r['depth_valid_fraction_mean'] for r in per_step_rows]):5.3f}"
            )
        if fr < failure_rate_threshold and np.isnan(threshold):
            # first range where sensor stays healthy = threshold
            threshold = float(r_m)
    return threshold, mc


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


def _format_summary(summary: Dict[str, dict], runs: Sequence[Dict] = ()) -> str:
    """Render the summary. `runs` is the raw record list — needed because
    `summarise()` filters bool fields out of its numeric aggregates, so
    sensor_failed / rtl_triggered have to be tallied separately.
    """
    # group raw runs by the same group key summarise used (scenario label)
    bool_tally: Dict[str, Dict[str, int]] = {}
    for r in runs:
        scen = r.get("scenario", "?")
        group_key = f"scenario={scen}"
        d = bool_tally.setdefault(group_key, {"n": 0, "sensor_failed": 0, "rtl": 0})
        d["n"] += 1
        if bool(r.get("sensor_failed")):
            d["sensor_failed"] += 1
        if bool(r.get("rtl_triggered")):
            d["rtl"] += 1

    rows = []
    rows.append(
        f"{'group':28s}  {'runs':>5s}  {'succ%':>7s}  "
        f"{'track med':>10s}  {'clr_min min':>12s}  "
        f"{'vf med':>8s}  {'sensor_fail%':>13s}  {'rtl%':>6s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        te = agg.get("tracking_error_mean_m", {}).get("median", float("nan"))
        clr = agg.get("corridor_clearance_min_m", {}).get("min", float("nan"))
        vf = agg.get("depth_valid_fraction_mean", {}).get("median", float("nan"))
        tally = bool_tally.get(group, {"n": n, "sensor_failed": 0, "rtl": 0})
        sf_pct = 100.0 * tally["sensor_failed"] / max(tally["n"], 1)
        rtl_pct = 100.0 * tally["rtl"] / max(tally["n"], 1)
        rows.append(
            f"{group:28s}  {n:5d}  {sr:6.1f}%  "
            f"{te:10.3f}  {clr:+12.3f}  {vf:8.3f}  {sf_pct:13.1f}  {rtl_pct:5.1f}"
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
    parser.add_argument(
        "--sweep",
        type=str,
        default="",
        choices=["", "wind", "depth", "both"],
        help="Run a parameter sweep instead of the fixed-scenario benchmark.",
    )
    parser.add_argument(
        "--wind-range",
        type=str,
        default="0,15,1",
        help="Sweep wind base_speed_ms: lo,hi,step (default 0,15,1).",
    )
    parser.add_argument(
        "--depth-range",
        type=str,
        default="1,10,1",
        help="Sweep depth max_range_m: lo,hi,step (default 1,10,1).",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="Failure-rate threshold for the sweep verdicts (default 0.5).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="",
        help="If set, write a PNG headline chart at this path.",
    )
    parser.add_argument(
        "--viz",
        type=str,
        default="",
        help="If set, run one extra detailed trial of --viz-scenario and "
        "write an interactive Plotly HTML there.",
    )
    parser.add_argument(
        "--viz-scenario", type=str, default="windy",
        help="Scenario for the viz trial (default: windy).",
    )
    parser.add_argument("--viz-seed", type=int, default=6262)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = Rig6Config(
        v_max=args.v_max,
        gcopter_maxiter=args.maxiter,
        dt=args.dt,
    )

    if args.sweep:
        results: Dict[str, float] = {}
        all_runs: List[Dict[str, float]] = []
        if args.sweep in ("wind", "both"):
            lo, hi, step = (float(x) for x in args.wind_range.split(","))
            speeds = list(np.arange(lo, hi + step / 2.0, step))
            print(
                f"Rig 6 SWEEP wind — speeds={speeds} m/s, "
                f"{args.runs} runs/step, threshold={args.failure_threshold:.0%}"
            )
            safe_wind, mc_wind = sweep_wind(
                speeds,
                runs_per_step=args.runs,
                config=config,
                failure_rate_threshold=args.failure_threshold,
                verbose=not args.quiet,
            )
            results["safe_wind_limit_ms"] = safe_wind
            all_runs.extend(mc_wind.runs)
            print(f"  → safe_wind_limit_ms = {safe_wind}")

        if args.sweep in ("depth", "both"):
            lo, hi, step = (float(x) for x in args.depth_range.split(","))
            ranges = list(np.arange(lo, hi + step / 2.0, step))
            print(
                f"Rig 6 SWEEP depth — ranges={ranges} m, "
                f"{args.runs} runs/step, threshold={args.failure_threshold:.0%}"
            )
            threshold, mc_depth = sweep_depth(
                ranges,
                runs_per_step=args.runs,
                config=config,
                failure_rate_threshold=args.failure_threshold,
                verbose=not args.quiet,
            )
            results["depth_range_threshold_m"] = threshold
            all_runs.extend(mc_depth.runs)
            print(f"  → depth_range_threshold_m = {threshold}")

        # write combined sweep JSON
        with open(args.output, "w") as f:
            json.dump(
                {"summary": results, "runs": all_runs},
                f,
                indent=2,
                default=str,
            )
        print(f"\nResults written to {args.output}")

        if args.plot:
            from src.validation.plots import emit_plot
            emit_plot("rig6", all_runs, args.plot)
            print(f"Plot written to {args.plot}")
        return 0

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            print(f"unknown scenario {s!r}; choose from {SCENARIOS}", file=sys.stderr)
            return 2

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
    print(_format_summary(summary, mc.runs))

    if args.plot:
        from src.validation.plots import emit_plot
        emit_plot("rig6", mc.runs, args.plot)
        print(f"Plot written to {args.plot}")

    if args.viz:
        from src.validation.visualize import emit_viz
        row = run_one_trial(
            args.viz_seed, args.viz_scenario, config, keep_record=True,
        )
        record = row.get("viz_record")
        if record is None:
            print("Viz trial produced no record", file=sys.stderr)
        else:
            emit_viz("rig6", record, args.viz)
            print(f"Viz written to {args.viz}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
