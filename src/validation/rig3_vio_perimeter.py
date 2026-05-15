"""Rig 3: VIO drift + perimeter fencing.

See docs/MINCO_PIVOT.md §5.4.

Question
--------
With VIO drift injected, does the swarm maintain its hex patrol perimeter?
At what drift rate does it break, and how much does inter-agent correction
help?

Pipeline under test
-------------------
    - N drones on a regular polygon perimeter (hex by default), patrolling
      their assigned arcs at constant speed (truth-state model — no
      physics engine, per the GCOPTER methodology in MINCO_PIVOT.md §5.1).
    - Each drone runs a VIODrift instance that accumulates random-walk +
      systematic-bias + jump errors over time.
    - Estimated position = truth + drift.
    - Optional inter-agent drift correction: each drone observes its
      neighbours' true relative positions via depth camera, compares them
      to the neighbours' broadcast (estimated) positions, and uses the
      residual to pull its own drift toward zero (a swarm-consensus filter).

Per-run metrics
---------------
    drift_magnitude_max_m, drift_magnitude_mean_m
    drift_corrected_max_m  (drift after correction, max over drones/ticks)
    perimeter_deviation_max_m, perimeter_deviation_mean_m
    sector_coverage_pct
    true_inter_drone_dist_min_m
    time_to_failure_s   (first tick where perimeter deviation exceeds the
                         configured tolerance, NaN if never)
    success (bool: time_to_failure is NaN)

CLI
---
    python -m src.validation.rig3_vio_perimeter \
        --drones 3 --drift-rate 0.02 --correction on,off --output rig3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.validation.metrics import MetricsCollector, summarise
from src.validation.vio_drift_model import VIODrift, VIODriftConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Rig3Config:
    # geometry
    perimeter_radius: float = 30.0     # circumscribed radius of the hex
    altitude: float = 5.0
    patrol_speed: float = 3.0          # m/s tangential

    # drift
    sigma_walk: float = 0.02           # m / sqrt(s)
    bias_rate: float = 0.01            # m/s
    bias_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    jump_prob_per_sec: float = 0.005
    jump_magnitude: float = 0.3
    drift_rate_multiplier: float = 1.0  # 1.0 = baseline, 5.0 = aggressive

    # correction
    correction_gain: float = 0.4       # 0..1; applied each second
    correction_period_s: float = 1.0   # how often inter-agent obs is processed

    # simulation
    sim_duration_s: float = 60.0
    dt: float = 0.1
    perimeter_tolerance_m: float = 2.0
    sector_coverage_bucket_deg: float = 5.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _hex_perimeter_position(
    drone_id: int, n_drones: int, t: float, config: Rig3Config
) -> np.ndarray:
    """Intended TRUE position of `drone_id` at time `t`, patrolling an
    arc on a circle of radius `perimeter_radius`.

    Each drone owns a sector of the circle of angular width 2π/n_drones
    centred on its station angle. It patrols within that sector at constant
    angular speed in a sawtooth pattern.
    """
    R = config.perimeter_radius
    station = 2.0 * np.pi * drone_id / n_drones
    arc_half = np.pi / n_drones            # half sector width in rad
    # tangential speed → angular speed
    omega = config.patrol_speed / max(R, 1e-6)
    # sawtooth over [-arc_half, +arc_half]
    phase = (omega * t) % (4.0 * arc_half)
    if phase < 2.0 * arc_half:
        offset = -arc_half + phase
    else:
        offset = +arc_half - (phase - 2.0 * arc_half)
    theta = station + offset
    return np.array(
        [R * np.cos(theta), R * np.sin(theta), config.altitude],
        dtype=np.float64,
    )


def _angle_of(p: np.ndarray) -> float:
    """Angle in [0, 2π) of the XY-projection of `p`."""
    a = float(np.arctan2(p[1], p[0]))
    if a < 0.0:
        a += 2.0 * np.pi
    return a


# ---------------------------------------------------------------------------
# Per-drone state
# ---------------------------------------------------------------------------


@dataclass
class DroneState:
    drone_id: int
    truth: np.ndarray = field(default_factory=lambda: np.zeros(3))
    drift_model: Optional[VIODrift] = None
    estimated: np.ndarray = field(default_factory=lambda: np.zeros(3))

    drift_max: float = 0.0
    drift_sum: float = 0.0
    drift_samples: int = 0
    perimeter_dev_max: float = 0.0
    perimeter_dev_sum: float = 0.0
    perimeter_dev_samples: int = 0
    angle_buckets: set = field(default_factory=set)


def _drone_drift_config(config: Rig3Config) -> VIODriftConfig:
    m = float(config.drift_rate_multiplier)
    return VIODriftConfig(
        sigma_walk=config.sigma_walk * m,
        bias_rate=config.bias_rate * m,
        bias_axis=config.bias_axis,
        jump_prob_per_sec=config.jump_prob_per_sec * m,
        jump_magnitude=config.jump_magnitude * m,
    )


# ---------------------------------------------------------------------------
# Inter-agent correction
# ---------------------------------------------------------------------------


def _apply_correction(
    drones: Sequence[DroneState],
    gain: float,
) -> None:
    """Use truth-relative observations of neighbours to compute each
    drone's drift residual, then pull its drift toward zero.

    Residual for drone i, observed via neighbour j:
        residual_i ≈ estimated_i − truth_i
                    = (estimated_i − estimated_j) − (truth_i − truth_j)
        observed = (estimated_i − estimated_j) is what drone i knows
        true_relative = (truth_i − truth_j) is what its depth camera sees
        residual_obs   = observed - true_relative

    We average over neighbours and apply with `gain`.
    """
    n = len(drones)
    if n < 2 or gain <= 0.0:
        return
    for i, di in enumerate(drones):
        if di.drift_model is None:
            continue
        accum = np.zeros(3, dtype=np.float64)
        count = 0
        for j, dj in enumerate(drones):
            if i == j:
                continue
            observed = di.estimated - dj.estimated
            true_rel = di.truth - dj.truth
            residual = observed - true_rel
            accum += residual
            count += 1
        if count == 0:
            continue
        residual_i = accum / count
        di.drift_model.correct(residual_i, gain=gain)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_one_trial(
    seed: int,
    n_drones: int,
    correction_enabled: bool,
    config: Optional[Rig3Config] = None,
) -> Dict[str, float]:
    if config is None:
        config = Rig3Config()
    if n_drones < 1:
        raise ValueError("n_drones must be >= 1")
    rng = np.random.default_rng(seed)

    drift_cfg = _drone_drift_config(config)
    drones: List[DroneState] = []
    for i in range(n_drones):
        ds = DroneState(drone_id=i)
        # one rng per drone, deterministic from outer seed
        ds.drift_model = VIODrift(drift_cfg, rng=np.random.default_rng(rng.integers(1 << 31)))
        ds.truth = _hex_perimeter_position(i, n_drones, 0.0, config)
        ds.estimated = ds.truth.copy()
        drones.append(ds)

    # ---- main loop
    n_steps = int(np.ceil(config.sim_duration_s / config.dt))
    bucket_size_rad = np.deg2rad(config.sector_coverage_bucket_deg)
    failure_time = float("nan")
    min_inter_drone = float("inf")

    t = 0.0
    last_correction_t = 0.0
    for step in range(n_steps):
        t = step * config.dt

        # advance truth
        for d in drones:
            d.truth = _hex_perimeter_position(d.drone_id, n_drones, t, config)

        # advance drift
        for d in drones:
            assert d.drift_model is not None
            d.drift_model.step(config.dt)
            d.estimated = d.truth + d.drift_model.value

        # periodic inter-agent correction
        if correction_enabled and (t - last_correction_t) >= config.correction_period_s:
            _apply_correction(drones, gain=config.correction_gain)
            # refresh estimated after correction (drift was nudged)
            for d in drones:
                assert d.drift_model is not None
                d.estimated = d.truth + d.drift_model.value
            last_correction_t = t

        # metrics per tick
        for d in drones:
            assert d.drift_model is not None
            drift_norm = float(np.linalg.norm(d.drift_model.value))
            d.drift_max = max(d.drift_max, drift_norm)
            d.drift_sum += drift_norm
            d.drift_samples += 1

            # perimeter deviation: how far is the ESTIMATED position from the
            # circle of radius `perimeter_radius` at altitude?
            radial = float(np.linalg.norm(d.estimated[:2]))
            dev = abs(radial - config.perimeter_radius) + abs(
                d.estimated[2] - config.altitude
            )
            d.perimeter_dev_max = max(d.perimeter_dev_max, dev)
            d.perimeter_dev_sum += dev
            d.perimeter_dev_samples += 1

            if np.isnan(failure_time) and dev > config.perimeter_tolerance_m:
                failure_time = t

            # sector coverage bucket on TRUTH angle
            ang = _angle_of(d.truth)
            bucket = int(ang / bucket_size_rad)
            d.angle_buckets.add(bucket)

        # pairwise true inter-drone distance
        if n_drones >= 2:
            pts = np.array([d.truth for d in drones])
            diff = pts[:, None, :] - pts[None, :, :]
            d_pair = np.linalg.norm(diff, axis=-1)
            iu = np.triu_indices(n_drones, k=1)
            tick_min = float(d_pair[iu].min())
            if tick_min < min_inter_drone:
                min_inter_drone = tick_min

    # ---- aggregate per-trial metrics
    drift_max = max((d.drift_max for d in drones), default=0.0)
    drift_mean = float(
        np.mean([d.drift_sum / max(1, d.drift_samples) for d in drones])
    )
    perim_max = max((d.perimeter_dev_max for d in drones), default=0.0)
    perim_mean = float(
        np.mean([d.perimeter_dev_sum / max(1, d.perimeter_dev_samples) for d in drones])
    )
    # coverage: what fraction of buckets in own sector did each drone visit?
    total_buckets_per_sector = int(round((2.0 * np.pi / n_drones) / bucket_size_rad))
    visited = np.mean([len(d.angle_buckets) for d in drones])
    coverage_pct = 100.0 * float(visited) / max(1, total_buckets_per_sector)
    coverage_pct = min(100.0, coverage_pct)

    result: Dict[str, float] = {
        "seed": seed,
        "n_drones": n_drones,
        "correction": "on" if correction_enabled else "off",
        "drift_magnitude_max_m": drift_max,
        "drift_magnitude_mean_m": drift_mean,
        "drift_corrected_max_m": drift_max if correction_enabled else float("nan"),
        "perimeter_deviation_max_m": perim_max,
        "perimeter_deviation_mean_m": perim_mean,
        "sector_coverage_pct": coverage_pct,
        "true_inter_drone_dist_min_m": (
            min_inter_drone if np.isfinite(min_inter_drone) else float("nan")
        ),
        "time_to_failure_s": failure_time,
        "sim_duration_s": config.sim_duration_s,
        "drift_rate_multiplier": config.drift_rate_multiplier,
        "success": bool(np.isnan(failure_time)),
    }
    return result


def run_benchmark(
    drones_list: Sequence[int],
    correction_modes: Sequence[str],
    runs: int,
    config: Optional[Rig3Config] = None,
    base_seed: int = 3000,
    verbose: bool = True,
) -> MetricsCollector:
    if config is None:
        config = Rig3Config()
    mc = MetricsCollector()
    for idx_n, n in enumerate(drones_list):
        for idx_c, mode in enumerate(correction_modes):
            corr = mode.strip().lower()
            if corr not in ("on", "off"):
                raise ValueError(f"correction must be on/off, got {mode!r}")
            corr_enabled = corr == "on"
            if verbose:
                print(f"\n--- drones={n}, correction={corr} ---")
            for run_idx in range(runs):
                seed = base_seed + idx_n * 100_000 + idx_c * 1_000 + run_idx
                row = run_one_trial(seed, n, corr_enabled, config)
                mc.start_run(n_drones=n, correction=corr, seed=seed)
                for k, v in row.items():
                    if k in ("n_drones", "correction", "seed"):
                        continue
                    mc.record(k, v)
                mc.finish_run()
                if verbose:
                    ttf = row["time_to_failure_s"]
                    ttf_str = "—" if np.isnan(ttf) else f"{ttf:6.1f}s"
                    print(
                        f"  run {run_idx + 1}/{runs}: "
                        f"perim_max={row['perimeter_deviation_max_m']:5.2f}m  "
                        f"drift_max={row['drift_magnitude_max_m']:5.2f}m  "
                        f"ttf={ttf_str}  "
                        f"success={row['success']}",
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
        f"{'drift_max med':>14s}  {'perim_max med':>14s}  "
        f"{'coverage med':>13s}  {'ttf min':>10s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        dmax = agg.get("drift_magnitude_max_m", {}).get("median", float("nan"))
        pmax = agg.get("perimeter_deviation_max_m", {}).get("median", float("nan"))
        cov = agg.get("sector_coverage_pct", {}).get("median", float("nan"))
        ttf = agg.get("time_to_failure_s", {}).get("min", float("nan"))
        rows.append(
            f"{group:28s}  {n:5d}  {sr:6.1f}%  "
            f"{dmax:14.3f}  {pmax:14.3f}  {cov:13.1f}  {ttf:10.2f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rig 3: VIO drift + perimeter fencing")
    parser.add_argument("--drones", type=str, default="3",
                        help="Comma-separated drone counts (default: 3)")
    parser.add_argument("--correction", type=str, default="on,off",
                        help="Comma-separated correction modes (on/off)")
    parser.add_argument("--drift-rate", type=float, default=0.02,
                        help="Baseline sigma_walk in m/sqrt(s) (default: 0.02)")
    parser.add_argument("--drift-mult", type=float, default=1.0,
                        help="Multiplier on all drift sources (5.0 = aggressive)")
    parser.add_argument("--sim-duration", type=float, default=60.0,
                        help="Sim duration s (default: 60)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Perimeter deviation tolerance m (default: 2.0)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="rig3_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    drones_list = [int(x) for x in args.drones.split(",")]
    correction_modes = [m.strip() for m in args.correction.split(",")]

    config = Rig3Config(
        sigma_walk=args.drift_rate,
        drift_rate_multiplier=args.drift_mult,
        sim_duration_s=args.sim_duration,
        perimeter_tolerance_m=args.tolerance,
    )

    print(
        f"Rig 3 — drones={drones_list}, correction={correction_modes}, "
        f"sigma_walk={args.drift_rate}m/√s, mult={args.drift_mult}, "
        f"sim={args.sim_duration}s, tolerance={args.tolerance}m"
    )

    mc = run_benchmark(
        drones_list=drones_list,
        correction_modes=correction_modes,
        runs=args.runs,
        config=config,
        verbose=not args.quiet,
    )

    mc.export_json(args.output, label_keys=["n_drones", "correction"])
    print(f"\nResults written to {args.output}")

    summary = summarise(mc.runs, label_keys=["n_drones", "correction"])
    print("\n=== Summary ===")
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
