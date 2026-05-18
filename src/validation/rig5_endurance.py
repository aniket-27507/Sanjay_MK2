"""Rig 5: endurance + attrition.

See docs/MINCO_PIVOT.md §5.6.

Question
--------
Over a 30-minute mission with battery drain and component degradation,
does coverage persist?

Pipeline under test
-------------------
    - N drones in hex patrol, each with:
        - BatteryModel (`src.simulation.physics.battery_model`)
        - MotorWear (`src.validation.motor_model`)
    - Per tick:
        - advance truth position (patrol)
        - drain battery proportional to commanded thrust;
        - thrust_fraction = hover_thrust_fraction / motor_thrust_scaling(flight_hours)
          → degraded motors burn more battery for the same lift
        - if battery hits RTL threshold, drone leaves patrol and a STANDBY
          spare takes over its sector (if any standby is available)
        - scenario-specific events (drone_down, cascading failure)
    - Coverage model identical to Rig 4 (sector arcs + bucket sweep).

Scenarios
---------
    normal              N=3, baseline
    battery_relay       N=3 + 1 standby (4 total)
    drone_down          N=3, one hard-fails mid-mission
    graceful_degrade    N=3, all start at 80% motor efficiency
    cascading_failure   N=3, two drones fail at staggered times

Metrics
-------
    mission_duration_s
    coverage_pct_timeline_mean
    coverage_gap_max_s
    coverage_loss_at_end_pct
    battery_consumed_wh
    relay_handoff_time_s   (NaN if no relay event)
    redistribution_time_s  (recovery time after first loss)
    degraded_thrust_ratio  (final mean motor_thrust_scaling)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.simulation.physics.battery_model import BatteryConfig, BatteryModel
from src.validation.metrics import MetricsCollector, summarise
from src.validation.motor_model import MotorWear, MotorWearConfig


SCENARIOS = (
    "normal",
    "battery_relay",
    "drone_down",
    "graceful_degrade",
    "cascading_failure",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Rig5Config:
    n_active: int = 3              # patrolling drones at t=0
    n_standby: int = 0             # spares (start in STANDBY status)
    perimeter_radius: float = 30.0
    altitude: float = 5.0
    patrol_speed: float = 3.0

    # battery + motor
    hover_thrust_fraction: float = 0.5
    ambient_temp_c: float = 32.0
    motor_initial_efficiency: float = 1.0
    motor_degradation_per_hour: float = 0.02
    motor_min_efficiency: float = 0.5

    # mission
    sim_duration_s: float = 1800.0   # 30 min default
    dt: float = 0.5
    coverage_arc_half_rad: float = np.pi / 3.0   # 60°, 3-drone full
    coverage_bucket_deg: float = 5.0

    # scenario hooks
    failure_times_s: Tuple[float, ...] = ()   # ids 0..len-1 fail at these times
    nominal_voltage_per_cell: float = 3.7

    # battery overrides
    capacity_mah: float = 2200.0


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------


ACTIVE = "active"
STANDBY = "standby"
RETURNING = "returning"
FAILED = "failed"


@dataclass
class DroneState5:
    drone_id: int
    sector_id: int = -1          # which sector are we currently covering? (−1 = none)
    status: str = ACTIVE
    battery: BatteryModel = field(default_factory=BatteryModel)
    motor: MotorWear = field(default_factory=lambda: MotorWear(MotorWearConfig()))
    flight_hours: float = 0.0
    energy_wh: float = 0.0       # cumulative consumed energy
    last_position: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _sector_position(sector_id: int, n_sectors: int, t: float, config: Rig5Config) -> np.ndarray:
    R = config.perimeter_radius
    station = 2.0 * np.pi * sector_id / n_sectors
    arc_half = np.pi / n_sectors
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
# Coverage
# ---------------------------------------------------------------------------


def _coverage_pct(
    positions_active: Sequence[Tuple[int, np.ndarray]],
    n_sectors: int,
    config: Rig5Config,
) -> float:
    """Fraction of perimeter buckets covered by currently-active drones.

    `positions_active` is a list of (sector_id, position) tuples for drones
    actively contributing to coverage. The arc each covers widens
    proportional to (n_sectors / n_active) so two surviving drones widen
    by 1.5×, etc.
    """
    bucket_rad = np.deg2rad(config.coverage_bucket_deg)
    n_buckets = int(round(2.0 * np.pi / bucket_rad))
    if not positions_active:
        return 0.0
    widen = n_sectors / max(1, len(positions_active))
    arc_half = config.coverage_arc_half_rad * widen
    covered = np.zeros(n_buckets, dtype=bool)
    for _sid, p in positions_active:
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
# Scenario setup
# ---------------------------------------------------------------------------


def _build_drones(seed: int, config: Rig5Config) -> List[DroneState5]:
    rng = np.random.default_rng(seed)
    drones: List[DroneState5] = []
    n_total = config.n_active + config.n_standby
    for i in range(n_total):
        bat = BatteryModel(
            BatteryConfig(
                capacity_mah=config.capacity_mah,
                seed=int(rng.integers(1 << 31)),
            )
        )
        motor = MotorWear(
            MotorWearConfig(
                initial_efficiency=config.motor_initial_efficiency,
                degradation_rate_per_hour=config.motor_degradation_per_hour,
                min_efficiency=config.motor_min_efficiency,
            )
        )
        status = ACTIVE if i < config.n_active else STANDBY
        sector = i if i < config.n_active else -1
        d = DroneState5(
            drone_id=i,
            sector_id=sector,
            status=status,
            battery=bat,
            motor=motor,
        )
        drones.append(d)
    return drones


def _apply_scenario_events(
    drones: List[DroneState5],
    t: float,
    config: Rig5Config,
    failed_events: List[float],
) -> None:
    """Apply scenario-specific hard failures at the configured times."""
    dt = config.dt
    for event_idx, t_fail in enumerate(config.failure_times_s):
        if event_idx >= len(drones):
            continue
        if t - dt < t_fail <= t and drones[event_idx].status not in (FAILED,):
            drones[event_idx].status = FAILED
            drones[event_idx].sector_id = -1
            failed_events.append(t)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_one_trial(
    seed: int,
    scenario: str,
    config: Optional[Rig5Config] = None,
    keep_record: bool = False,
) -> Dict[str, float]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if config is None:
        config = Rig5Config()

    # scenario-specific config patches
    if scenario == "battery_relay":
        config = Rig5Config(**{**config.__dict__, "n_standby": max(1, config.n_standby)})
    elif scenario == "drone_down":
        if not config.failure_times_s:
            config = Rig5Config(**{**config.__dict__, "failure_times_s": (config.sim_duration_s / 2.0,)})
    elif scenario == "graceful_degrade":
        config = Rig5Config(**{**config.__dict__, "motor_initial_efficiency": 0.8})
    elif scenario == "cascading_failure":
        if not config.failure_times_s:
            t1 = config.sim_duration_s / 3.0
            t2 = 2.0 * config.sim_duration_s / 3.0
            config = Rig5Config(**{**config.__dict__, "failure_times_s": (t1, t2)})

    drones = _build_drones(seed, config)
    n_sectors = config.n_active
    n_total = len(drones)

    n_steps = int(np.ceil(config.sim_duration_s / config.dt))
    coverage_timeline: List[float] = []
    coverage_timeline_with_t: List[Tuple[float, float]] = []
    coverage_gap_total = 0.0
    handoff_time = float("nan")
    redistribution_time = float("nan")
    first_failure_time = float("nan")
    failed_events: List[float] = []

    viz_pos: Optional[List[List[List[float]]]] = None
    viz_status: Optional[List[List[str]]] = None
    viz_batt: Optional[List[List[float]]] = None
    if keep_record:
        viz_pos = [[] for _ in range(n_total)]
        viz_status = [[] for _ in range(n_total)]
        viz_batt = [[] for _ in range(n_total)]

    for step in range(n_steps):
        t = step * config.dt

        # scenario events
        _apply_scenario_events(drones, t, config, failed_events)

        # advance each drone
        for d in drones:
            if d.status == FAILED:
                continue
            d.flight_hours += config.dt / 3600.0
            motor_scale = d.motor.thrust_scaling(d.flight_hours)
            # degraded motors need more thrust for the same lift
            thrust_frac = min(1.0, config.hover_thrust_fraction / max(motor_scale, 1e-6))
            soc_before = d.battery.soc_pct
            d.battery.tick(config.dt, thrust_frac, ambient_temp_c=config.ambient_temp_c)
            # energy consumed (Wh) = V * I * dt / 3600
            current = d.battery.current_draw(thrust_frac)
            voltage = d.battery.voltage(current)
            d.energy_wh += voltage * current * config.dt / 3600.0

            if d.status == ACTIVE and d.battery.should_rtl:
                # leave patrol — try to promote a standby into our sector
                vacated = d.sector_id
                d.status = RETURNING
                d.sector_id = -1
                if np.isnan(first_failure_time):
                    first_failure_time = t
                # find standby
                promoted = False
                for spare in drones:
                    if spare.status == STANDBY:
                        spare.status = ACTIVE
                        spare.sector_id = vacated
                        if np.isnan(handoff_time):
                            handoff_time = t
                        promoted = True
                        break
                if not promoted:
                    # surviving drones must widen — track that we lost a sector
                    pass

            if d.status == ACTIVE:
                d.last_position = _sector_position(d.sector_id, n_sectors, t, config)

        # if a drone hard-failed via scenario event, record failure time
        if failed_events and np.isnan(first_failure_time):
            first_failure_time = failed_events[0]

        # coverage
        active_for_coverage: List[Tuple[int, np.ndarray]] = []
        for d in drones:
            if d.status == ACTIVE and d.sector_id >= 0:
                active_for_coverage.append((d.sector_id, d.last_position))
        cov_pct = _coverage_pct(active_for_coverage, n_sectors, config)
        coverage_timeline.append(cov_pct)
        coverage_timeline_with_t.append((float(t), float(cov_pct)))
        if cov_pct < 100.0 - 1e-6:
            coverage_gap_total += config.dt

        # redistribution time: after first failure, how long until coverage
        # returns to 100%? (only set once)
        if (
            not np.isnan(first_failure_time)
            and np.isnan(redistribution_time)
            and t >= first_failure_time
            and cov_pct >= 100.0 - 1e-6
        ):
            redistribution_time = t - first_failure_time

        # viz capture
        if viz_pos is not None:
            for i, d in enumerate(drones):
                # For STANDBY drones, park them just outside the perimeter
                # near sector 0. For FAILED ones, leave the last known
                # position. For RETURNING, drift toward origin so the
                # animation reads "going home".
                if d.status == ACTIVE and d.sector_id >= 0:
                    p = _sector_position(d.sector_id, n_sectors, t, config)
                elif d.status == RETURNING:
                    p = d.last_position * 0.95  # slow drift inward
                elif d.status == STANDBY:
                    p = np.array(
                        [config.perimeter_radius * 1.2, 0.0, config.altitude]
                    )
                else:  # FAILED
                    p = d.last_position
                d.last_position = p if d.status != FAILED else d.last_position
                viz_pos[i].append([float(p[0]), float(p[1]), float(p[2])])
                viz_status[i].append(d.status)
                viz_batt[i].append(float(d.battery.soc_pct))

    # aggregate metrics
    coverage_mean = float(np.mean(coverage_timeline)) if coverage_timeline else 0.0
    coverage_end = coverage_timeline[-1] if coverage_timeline else 0.0
    energy_total = float(sum(d.energy_wh for d in drones))
    motor_scales = [d.motor.thrust_scaling(d.flight_hours) for d in drones]
    degraded_ratio = float(np.mean(motor_scales)) if motor_scales else 1.0

    n_alive = sum(1 for d in drones if d.status not in (FAILED,))
    result: Dict[str, float] = {
        "seed": seed,
        "scenario": scenario,
        "n_active": config.n_active,
        "n_standby": config.n_standby,
        "mission_duration_s": config.sim_duration_s,
        "coverage_pct_timeline_mean": coverage_mean,
        "coverage_pct_timeline": coverage_timeline_with_t,
        "coverage_loss_at_end_pct": 100.0 - coverage_end,
        "coverage_gap_max_s": coverage_gap_total,
        "battery_consumed_wh": energy_total,
        "relay_handoff_time_s": handoff_time,
        "redistribution_time_s": redistribution_time,
        "degraded_thrust_ratio": degraded_ratio,
        "drones_alive_at_end": n_alive,
        "success": coverage_mean >= 60.0,   # weak but useful sanity gate
    }
    if viz_pos is not None:
        result["viz_record"] = {
            "scenario": scenario,
            "n_active": config.n_active,
            "n_standby": config.n_standby,
            "perimeter_radius": float(config.perimeter_radius),
            "altitude": float(config.altitude),
            "sample_dt_s": float(config.dt),
            "positions_per_drone": viz_pos,
            "status_per_drone": viz_status,
            "battery_per_drone": viz_batt,
            "coverage_timeline": coverage_timeline_with_t,
            "coverage_pct_timeline_mean": coverage_mean,
            "coverage_gap_max_s": coverage_gap_total,
            "coverage_loss_at_end_pct": 100.0 - coverage_end,
            "drones_alive_at_end": n_alive,
            "success": bool(result["success"]),
        }
    return result


def run_benchmark(
    scenarios: Sequence[str],
    runs_per_scenario: int,
    config: Optional[Rig5Config] = None,
    base_seed: int = 5000,
    verbose: bool = True,
) -> MetricsCollector:
    if config is None:
        config = Rig5Config()
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
                cov = row["coverage_pct_timeline_mean"]
                gap = row["coverage_gap_max_s"]
                rt = row["redistribution_time_s"]
                rt_s = "—" if np.isnan(rt) else f"{rt:6.1f}s"
                print(
                    f"  run {run_idx + 1}/{runs_per_scenario}: "
                    f"cov_mean={cov:5.1f}%  gap={gap:6.1f}s  "
                    f"redistrib={rt_s}  success={row['success']}",
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
        f"{'cov med':>9s}  {'gap med':>9s}  "
        f"{'redistrib med':>14s}  {'energy med':>11s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        cov = agg.get("coverage_pct_timeline_mean", {}).get("median", float("nan"))
        gap = agg.get("coverage_gap_max_s", {}).get("median", float("nan"))
        rt = agg.get("redistribution_time_s", {}).get("median", float("nan"))
        en = agg.get("battery_consumed_wh", {}).get("median", float("nan"))
        rows.append(
            f"{group:28s}  {n:5d}  {sr:6.1f}%  "
            f"{cov:9.1f}  {gap:9.1f}  {rt:14.1f}  {en:11.2f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rig 5: endurance + attrition")
    parser.add_argument(
        "--scenarios", type=str, default="normal,battery_relay,drone_down",
        help=f"Comma-separated scenarios from {SCENARIOS}",
    )
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--n-active", type=int, default=3)
    parser.add_argument("--n-standby", type=int, default=0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument(
        "--failures", type=str, default="",
        help="Comma-separated seconds; drone_i fails at failures[i]",
    )
    parser.add_argument("--output", type=str, default="rig5_results.json")
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
        "--viz-scenario", type=str, default="drone_down",
        help="Scenario for the viz trial (default: drone_down).",
    )
    parser.add_argument("--viz-seed", type=int, default=5151)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            print(f"unknown scenario {s!r}; choose from {SCENARIOS}", file=sys.stderr)
            return 2

    failure_times: Tuple[float, ...] = ()
    if args.failures:
        failure_times = tuple(float(x) for x in args.failures.split(","))

    config = Rig5Config(
        n_active=args.n_active,
        n_standby=args.n_standby,
        sim_duration_s=args.duration,
        dt=args.dt,
        failure_times_s=failure_times,
    )

    print(
        f"Rig 5 — scenarios={scenarios}, n_active={args.n_active}, "
        f"n_standby={args.n_standby}, duration={args.duration}s, dt={args.dt}s"
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

    if args.plot:
        from src.validation.plots import emit_plot
        emit_plot("rig5", mc.runs, args.plot)
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
            emit_viz("rig5", record, args.viz)
            print(f"Viz written to {args.viz}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
