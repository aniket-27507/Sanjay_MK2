"""Rig 4: mission response time.

See docs/MINCO_PIVOT.md §5.5.

Question
--------
When a threat is detected, how fast can one drone break off to inspect while
the others close the coverage gap?

Pipeline under test
-------------------
    - 3-drone hex patrol (truth-state, like Rig 3).
    - At a chosen `threat_time_s`, a threat appears at `threat_position`.
    - A distance-based bid (the same scoring rule used by the CBBA engine in
      `src.swarm.cbba.cbba_engine`) picks the closest drone as INSPECTOR.
    - The inspector breaks off and flies a straight line to the threat at
      `inspect_speed`. On arrival it hovers for `inspect_dwell_s`, then flies
      back to its sector.
    - During the inspector's absence, the two remaining drones widen their
      sectors to maintain coverage.

Metrics
-------
    t_detect_to_replan_ms : wall-clock to compute the new assignment
    t_coverage_gap_s      : seconds during which < 100% of perimeter was
                            covered by the surviving patrol fleet
    coverage_pct_during   : mean coverage % across the inspection window
    t_regroup_s           : sim time from threat → inspector back on station
    inspector_arrival_s   : sim time from threat → inspector reaches threat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.validation.metrics import MetricsCollector, summarise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Rig4Config:
    n_drones: int = 3
    perimeter_radius: float = 30.0
    altitude: float = 5.0
    patrol_speed: float = 3.0
    inspect_speed: float = 5.0
    inspect_dwell_s: float = 10.0

    threat_time_s: float = 30.0
    threat_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    sim_duration_s: float = 120.0
    dt: float = 0.1

    # Coverage model: each drone covers an arc of half-width
    # `coverage_arc_half_rad` rad of perimeter, centred on its current
    # angular position. For surviving drones during the inspection window,
    # widen this by `coverage_widen_factor`.
    coverage_arc_half_rad: float = np.pi / 3.0   # 60° — i.e. 3-drone full
    coverage_widen_factor: float = 1.5
    coverage_bucket_deg: float = 5.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _patrol_position(
    drone_id: int, n_drones: int, t: float, config: Rig4Config
) -> np.ndarray:
    """Sawtooth patrol of one sector — same model as Rig 3."""
    R = config.perimeter_radius
    station = 2.0 * np.pi * drone_id / n_drones
    arc_half = np.pi / n_drones
    omega = config.patrol_speed / max(R, 1e-6)
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
    a = float(np.arctan2(p[1], p[0]))
    if a < 0.0:
        a += 2.0 * np.pi
    return a


# ---------------------------------------------------------------------------
# Inspector trajectory (linear segments: patrol → threat → patrol)
# ---------------------------------------------------------------------------


@dataclass
class InspectorPlan:
    drone_id: int
    t_break: float          # sim time when assignment was made
    p_break: np.ndarray     # patrol position at break time
    threat_pos: np.ndarray
    t_arrival: float        # sim time when threat is reached
    t_depart: float         # sim time when inspector leaves threat
    t_return: float         # sim time when inspector regains its sector
    p_return: np.ndarray    # patrol position rejoined


def _inspector_plan(
    drone_id: int,
    p_break: np.ndarray,
    threat_pos: np.ndarray,
    t_break: float,
    config: Rig4Config,
    n_drones: int,
) -> InspectorPlan:
    speed = max(config.inspect_speed, 1e-6)
    d_to_threat = float(np.linalg.norm(threat_pos - p_break))
    t_arrival = t_break + d_to_threat / speed
    t_depart = t_arrival + config.inspect_dwell_s
    # We rejoin patrol at the patrol position at the time we'd arrive flying
    # at `inspect_speed` straight back. Solve fixed-point-ish — easier: pick
    # the patrol position at t_depart + d_return/speed where d_return is
    # computed against position at t_depart. One-shot estimate is adequate
    # for this rig's metric.
    p_at_depart = _patrol_position(drone_id, n_drones, t_depart, config)
    d_return_guess = float(np.linalg.norm(p_at_depart - threat_pos))
    t_return = t_depart + d_return_guess / speed
    p_return = _patrol_position(drone_id, n_drones, t_return, config)
    return InspectorPlan(
        drone_id=drone_id,
        t_break=t_break,
        p_break=p_break.copy(),
        threat_pos=threat_pos.copy(),
        t_arrival=t_arrival,
        t_depart=t_depart,
        t_return=t_return,
        p_return=p_return.copy(),
    )


def _inspector_position(
    t: float, plan: InspectorPlan, config: Rig4Config, n_drones: int
) -> np.ndarray:
    """Where is the inspector at sim time `t`?

    Phases:
        t < t_break              → patrol (sawtooth)
        [t_break, t_arrival)     → linear flight, p_break → threat_pos
        [t_arrival, t_depart)    → hovering at threat_pos
        [t_depart, t_return)     → linear flight, threat_pos → p_return
        t >= t_return            → patrol resumed (sawtooth)
    """
    if t < plan.t_break:
        return _patrol_position(plan.drone_id, n_drones, t, config)
    if t < plan.t_arrival:
        u = (t - plan.t_break) / max(plan.t_arrival - plan.t_break, 1e-6)
        return plan.p_break + u * (plan.threat_pos - plan.p_break)
    if t < plan.t_depart:
        return plan.threat_pos
    if t < plan.t_return:
        u = (t - plan.t_depart) / max(plan.t_return - plan.t_depart, 1e-6)
        return plan.threat_pos + u * (plan.p_return - plan.threat_pos)
    return _patrol_position(plan.drone_id, n_drones, t, config)


# ---------------------------------------------------------------------------
# Bid: same distance-weighted score as CBBA
# ---------------------------------------------------------------------------


def _select_inspector(
    positions: Sequence[np.ndarray], threat: np.ndarray
) -> int:
    """Closest drone wins.

    The CBBA threat bid in `src.swarm.cbba.cbba_engine` mixes distance,
    battery, sensor heading, and load. With identical drones on a circular
    patrol the dominant term is distance — use it here so the rig isolates
    the decision-latency metric.
    """
    d = [float(np.linalg.norm(p - threat)) for p in positions]
    return int(np.argmin(d))


# ---------------------------------------------------------------------------
# Coverage model
# ---------------------------------------------------------------------------


def _coverage_pct(
    positions: Sequence[np.ndarray],
    inspector_id: Optional[int],
    config: Rig4Config,
) -> float:
    """Fraction of the perimeter buckets covered by patrolling drones.

    A patrolling drone covers angular buckets within `coverage_arc_half_rad`
    of its current angular position. The inspector covers nothing while
    off-perimeter. Surviving drones widen their coverage by
    `coverage_widen_factor` if an inspector is currently absent.
    """
    bucket_rad = np.deg2rad(config.coverage_bucket_deg)
    n_buckets = int(round(2.0 * np.pi / bucket_rad))
    covered = np.zeros(n_buckets, dtype=bool)
    widen = config.coverage_widen_factor if inspector_id is not None else 1.0
    arc_half = config.coverage_arc_half_rad * widen
    for i, p in enumerate(positions):
        if i == inspector_id:
            continue
        # is this drone roughly on the perimeter? if it's been pulled off
        # (e.g. inspector phase), skip — but normal patrol drones always are.
        radial = float(np.linalg.norm(p[:2]))
        if abs(radial - config.perimeter_radius) > 2.0:
            continue
        ang = _angle_of(p)
        for b in range(n_buckets):
            centre = (b + 0.5) * bucket_rad
            diff = ((centre - ang + np.pi) % (2.0 * np.pi)) - np.pi
            if abs(diff) <= arc_half:
                covered[b] = True
    return 100.0 * float(np.count_nonzero(covered)) / max(1, n_buckets)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_one_trial(
    seed: int,
    config: Optional[Rig4Config] = None,
) -> Dict[str, float]:
    if config is None:
        config = Rig4Config()
    rng = np.random.default_rng(seed)  # reserved for future jitter
    n = config.n_drones
    if n < 2:
        raise ValueError("n_drones must be >= 2")

    threat = np.asarray(config.threat_position, dtype=np.float64)
    n_steps = int(np.ceil(config.sim_duration_s / config.dt))
    threat_step = int(round(config.threat_time_s / config.dt))

    plan: Optional[InspectorPlan] = None
    t_detect_to_replan_ms = float("nan")
    coverage_during_sum = 0.0
    coverage_during_count = 0
    t_coverage_gap_s = 0.0

    for step in range(n_steps):
        t = step * config.dt

        # at the threat step, run the bid and build the inspector plan
        if step == threat_step and plan is None:
            patrol_positions = [
                _patrol_position(i, n, t, config) for i in range(n)
            ]
            t0 = time.perf_counter()
            inspector_id = _select_inspector(patrol_positions, threat)
            plan = _inspector_plan(
                inspector_id,
                patrol_positions[inspector_id],
                threat,
                t,
                config,
                n,
            )
            t_detect_to_replan_ms = (time.perf_counter() - t0) * 1000.0

        # compute current positions
        positions: List[np.ndarray] = []
        for i in range(n):
            if plan is not None and i == plan.drone_id:
                positions.append(_inspector_position(t, plan, config, n))
            else:
                positions.append(_patrol_position(i, n, t, config))

        # coverage during inspection window
        if plan is not None and t >= plan.t_break and t <= plan.t_return:
            cov_pct = _coverage_pct(positions, plan.drone_id, config)
            coverage_during_sum += cov_pct
            coverage_during_count += 1
            if cov_pct < 100.0 - 1e-6:
                t_coverage_gap_s += config.dt

    inspector_arrival_s = (
        plan.t_arrival - plan.t_break if plan is not None else float("nan")
    )
    t_regroup_s = plan.t_return - plan.t_break if plan is not None else float("nan")
    coverage_pct_during = (
        coverage_during_sum / coverage_during_count
        if coverage_during_count > 0
        else float("nan")
    )

    result: Dict[str, float] = {
        "seed": seed,
        "n_drones": n,
        "threat_time_s": config.threat_time_s,
        "inspector_id": float(plan.drone_id) if plan is not None else float("nan"),
        "t_detect_to_replan_ms": t_detect_to_replan_ms,
        "inspector_arrival_s": inspector_arrival_s,
        "t_coverage_gap_s": t_coverage_gap_s,
        "coverage_pct_during": coverage_pct_during,
        "t_regroup_s": t_regroup_s,
        "success": bool(
            plan is not None and t_regroup_s <= config.sim_duration_s - config.threat_time_s
        ),
    }
    return result


def run_benchmark(
    threat_positions: Sequence[Tuple[float, float, float]],
    runs_per_threat: int,
    config: Optional[Rig4Config] = None,
    base_seed: int = 4000,
    verbose: bool = True,
) -> MetricsCollector:
    if config is None:
        config = Rig4Config()
    mc = MetricsCollector()
    for idx_t, tp in enumerate(threat_positions):
        cfg_t = Rig4Config(**{**config.__dict__, "threat_position": tuple(tp)})
        if verbose:
            print(f"\n--- threat_position={tp} ---")
        for run_idx in range(runs_per_threat):
            seed = base_seed + idx_t * 1000 + run_idx
            row = run_one_trial(seed, cfg_t)
            mc.start_run(
                threat_x=tp[0], threat_y=tp[1], threat_z=tp[2], seed=seed
            )
            for k, v in row.items():
                if k == "seed":
                    continue
                mc.record(k, v)
            mc.finish_run()
            if verbose:
                ttr = row["t_detect_to_replan_ms"]
                cov = row["coverage_pct_during"]
                rgr = row["t_regroup_s"]
                print(
                    f"  run {run_idx + 1}/{runs_per_threat}: "
                    f"inspector={int(row['inspector_id'])}  "
                    f"t_replan={ttr:6.3f}ms  cov={cov:5.1f}%  "
                    f"regroup={rgr:6.1f}s  success={row['success']}",
                    flush=True,
                )
    return mc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_summary(summary: Dict[str, dict]) -> str:
    rows = []
    rows.append(
        f"{'group':30s}  {'runs':>5s}  {'succ%':>7s}  "
        f"{'t_replan med':>14s}  {'cov med':>9s}  "
        f"{'gap med':>9s}  {'regroup med':>12s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        tr = agg.get("t_detect_to_replan_ms", {}).get("median", float("nan"))
        cov = agg.get("coverage_pct_during", {}).get("median", float("nan"))
        gap = agg.get("t_coverage_gap_s", {}).get("median", float("nan"))
        rgr = agg.get("t_regroup_s", {}).get("median", float("nan"))
        rows.append(
            f"{group:30s}  {n:5d}  {sr:6.1f}%  "
            f"{tr:14.3f}  {cov:9.1f}  {gap:9.2f}  {rgr:12.1f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rig 4: mission response time")
    parser.add_argument(
        "--threats",
        type=str,
        default="0,0,5;15,0,5;-10,15,5",
        help="Semicolon-separated threat positions, each x,y,z (m)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-drones", type=int, default=3)
    parser.add_argument("--threat-time", type=float, default=30.0)
    parser.add_argument("--sim-duration", type=float, default=120.0)
    parser.add_argument("--inspect-dwell", type=float, default=10.0)
    parser.add_argument("--inspect-speed", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="rig4_results.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    threats = []
    for raw in args.threats.split(";"):
        parts = [float(x) for x in raw.split(",")]
        if len(parts) != 3:
            print(f"bad threat token {raw!r}: need x,y,z", file=sys.stderr)
            return 2
        threats.append(tuple(parts))

    config = Rig4Config(
        n_drones=args.n_drones,
        threat_time_s=args.threat_time,
        sim_duration_s=args.sim_duration,
        inspect_dwell_s=args.inspect_dwell,
        inspect_speed=args.inspect_speed,
    )

    print(
        f"Rig 4 — drones={args.n_drones}, threats={threats}, "
        f"threat_time={args.threat_time}s, dwell={args.inspect_dwell}s, "
        f"speed={args.inspect_speed}m/s"
    )

    mc = run_benchmark(
        threat_positions=threats,
        runs_per_threat=args.runs,
        config=config,
        verbose=not args.quiet,
    )

    mc.export_json(args.output, label_keys=["threat_x", "threat_y", "threat_z"])
    print(f"\nResults written to {args.output}")

    summary = summarise(mc.runs, label_keys=["threat_x", "threat_y", "threat_z"])
    print("\n=== Summary ===")
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
