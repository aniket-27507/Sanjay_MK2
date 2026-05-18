"""Rig 2: swarm collision avoidance + scaling benchmark.

See docs/MINCO_PIVOT.md §5.3.

Question
--------
With N drones broadcasting MINCO trajectories over the simulated mesh, how
close do they get, and how does per-agent replan time scale with N?

Pipeline under test
-------------------
    per-drone initial MINCO (one corridor, straight-line waypoint)
        + SwarmBroadcaster (over BroadcastChannel)
        + every replan tick:
            - poll inbox → list of (neighbour_traj, t_offset)
            - re-optimise own trajectory with the ellipsoidal swarm penalty
              folded into GCopter's L-BFGS via gcopter_optimize(...,
              swarm_neighbours=...)
            - broadcast new trajectory

Scenarios
---------
    head_on  : N=2 drones aimed at each other
    crossing : N=3 paths crossing at the origin
    converge : N=3 drones aimed at the same goal
    patrol   : N drones equally spaced on a circle, swapping antipodal goals

Per-run metrics
---------------
    d_min_inter, d_mean_inter, near_misses, collisions
    t_replan_mean_ms, t_replan_max_ms, t_replan_per_agent_mean_ms
    broadcast_bandwidth_kbps, network_congestion_pct
    n_drones, scenario, comms_latency_ms, comms_loss_pct

CLI
---
    python -m src.validation.rig2_swarm_avoidance \
        --drones 3,6,12,25,50 --scenario patrol --runs 3 --output rig2.json

Notes
-----
- No obstacles — Rig 2 isolates inter-drone coupling. Rig 6 stacks
  obstacles + swarm on top.
- Each drone uses a single fat corridor polytope around its straight path.
  M=2 trajectory segments give the swarm penalty an interior waypoint to
  push around.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.single_drone.planning import (
    GCopterConfig,
    Polytope,
    Trajectory,
    gcopter_optimize,
)
from src.swarm.swarm_penalty import SwarmPenaltyConfig
from src.swarm.trajectory_broadcast import SwarmBroadcaster
from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig
from src.validation.metrics import MetricsCollector, summarise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS = ("head_on", "crossing", "converge", "patrol")


@dataclass
class Rig2Config:
    """Tunable parameters for one Rig 2 trial.

    Distances are in metres; times in seconds.
    """

    # geometry
    field_radius: float = 25.0          # patrol-circle radius
    altitude: float = 5.0
    corridor_half_extent: Tuple[float, float, float] = (3.0, 3.0, 2.0)

    # trajectory + optimiser
    v_max: float = 4.0
    minco_segments: int = 2             # M; M-1 free interior waypoints
    gcopter_maxiter: int = 25
    gcopter_n_quad: int = 8

    # swarm penalty
    clearance_horizontal: float = 2.0
    clearance_vertical: float = 1.0
    swarm_weight: float = 1.0e3
    # Linear staleness window for inter-drone broadcasts. A trajectory
    # whose t_sent is older than this contributes zero penalty; freshness
    # within the window decays linearly. Squared in compute_swarm_cost_and_grad.
    swarm_freshness_max_age_s: float = 0.5

    # Avenue 3: topology-guided multi-branch (TRUST-Planner pattern).
    # When enabled, the warm-started main solve is first checked for
    # predicted swarm violations; if any neighbour is within
    # `multi_branch_trigger_dist_fraction * clearance_horizontal`, k
    # additional branches with perturbed initial guesses are run and
    # the best (collision-free first, then lowest cost) is selected.
    enable_multi_branch: bool = False
    multi_branch_n: int = 4
    multi_branch_perturbation_scale: float = 1.5
    multi_branch_trigger_dist_fraction: float = 2.0

    # Avenue 2: Bayesian-filter warm start (simplified from Yuan & Yu 2025).
    # When enabled, each Drone maintains a Kalman-style estimator over its
    # (interior waypoints, durations) flat state vector. After a confidence
    # threshold is reached (default 0.5), subsequent reoptimise calls use
    # the filter's smoothed prediction as the warm-start initial guess
    # rather than the raw previous solution. Helps in stable scenarios
    # where the optimum drifts smoothly; auto-resets on regime change.
    enable_bayesian_warm_start: bool = False
    bayesian_process_noise: float = 1.0e-2
    bayesian_observation_noise: float = 1.0e-1
    bayesian_confidence_threshold: float = 0.5
    bayesian_innovation_reset_threshold: float = 2.0

    # Avenue 4: CBF safety filter applied as a POST-MINCO layer.
    # When enabled, sampled trajectory positions/velocities are passed
    # through a Control Barrier Function filter that enforces pairwise
    # h_dot + alpha h >= 0 by projecting onto the safe half-space. The
    # rig reports both raw and CBF-filtered collision counts. The filter
    # does NOT modify the underlying MINCO trajectories — it acts on the
    # sampled output as a safety overlay. This is the simplified version
    # of FECBF (arXiv 2603.13103); the full version replaces the swarm
    # penalty entirely with a per-tick QP.
    enable_cbf_filter: bool = False
    cbf_alpha: float = 2.0
    cbf_max_velocity_correction: float = 3.0

    # comms channel
    comms_latency_ms_mean: float = 50.0
    comms_latency_ms_jitter: float = 20.0
    comms_loss_pct: float = 0.0
    comms_bandwidth_kbps: Optional[float] = 1024.0

    # simulation loop
    replan_period_s: float = 1.0
    sim_duration_s: float = 8.0
    sample_dt_s: float = 0.1
    near_miss_radius: float = 1.5
    collision_radius: float = 0.5


# ---------------------------------------------------------------------------
# Scenario setup
# ---------------------------------------------------------------------------


def _patrol_endpoints(
    n_drones: int, radius: float, altitude: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """N drones evenly spaced on a circle of `radius`, each aimed at the
    antipodal slot. Generates well-defined inter-drone conflicts."""
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_drones):
        theta = 2.0 * np.pi * i / n_drones
        start = np.array(
            [radius * np.cos(theta), radius * np.sin(theta), altitude],
            dtype=np.float64,
        )
        # antipodal goal — drones cross near the centre
        goal = -start.copy()
        goal[2] = altitude
        pairs.append((start, goal))
    return pairs


def endpoints_for_scenario(
    scenario: str, n_drones: int, config: Rig2Config
) -> List[Tuple[np.ndarray, np.ndarray]]:
    alt = config.altitude
    r = config.field_radius

    if scenario == "head_on":
        if n_drones != 2:
            raise ValueError("head_on requires exactly 2 drones")
        a = np.array([-r, 0.0, alt])
        b = np.array([+r, 0.0, alt])
        return [(a, b), (b, a)]

    if scenario == "crossing":
        if n_drones != 3:
            raise ValueError("crossing requires exactly 3 drones")
        endpoints = []
        for i in range(3):
            theta = np.pi * i / 3.0  # 0°, 60°, 120°
            start = np.array([r * np.cos(theta), r * np.sin(theta), alt])
            goal = -start.copy()
            goal[2] = alt
            endpoints.append((start, goal))
        return endpoints

    if scenario == "converge":
        if n_drones != 3:
            raise ValueError("converge requires exactly 3 drones")
        goal = np.array([0.0, 0.0, alt])
        endpoints = []
        for i in range(3):
            theta = 2.0 * np.pi * i / 3.0
            start = np.array([r * np.cos(theta), r * np.sin(theta), alt])
            endpoints.append((start, goal.copy()))
        return endpoints

    if scenario == "patrol":
        return _patrol_endpoints(n_drones, r, alt)

    raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")


# ---------------------------------------------------------------------------
# Per-drone state
# ---------------------------------------------------------------------------


def _corridor_box(
    start: np.ndarray,
    goal: np.ndarray,
    half_extent: Sequence[float],
) -> Polytope:
    """One axis-aligned bounding box wide enough to wrap the straight path
    plus a slack of `half_extent` on every face.

    Each row of A is an outward face normal; b is the offset. The polytope
    is defined as {x : A x <= b}.
    """
    lo = np.minimum(start, goal) - np.asarray(half_extent, dtype=np.float64)
    hi = np.maximum(start, goal) + np.asarray(half_extent, dtype=np.float64)
    A = np.vstack([+np.eye(3), -np.eye(3)])
    b = np.concatenate([hi, -lo])
    return Polytope(A=A, b=b)


def _initial_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    config: Rig2Config,
    drone_id: int = 0,
    n_drones: int = 1,
) -> Tuple[Trajectory, List[Polytope]]:
    """Straight-line MINCO with `minco_segments` segments, one corridor
    per segment (each segment gets its own fat box around its sub-leg).

    Symmetric multi-drone scenarios (e.g. N-drone patrol with antipodal
    goals) put the L-BFGS optimiser in a saddle point: by symmetry the
    swarm-penalty gradient cancels at the centre, and the drones never
    learn to break the meeting. We offset each drone's interior
    waypoints by a small deterministic vector keyed by `drone_id` —
    enough to seed a non-zero gradient without violating the corridor.
    """
    M = max(1, int(config.minco_segments))
    s = 3
    D = 3
    fracs = np.linspace(0.0, 1.0, M + 1)
    waypoints = np.stack(
        [start + f * (goal - start) for f in fracs], axis=0
    )

    # Symmetry breaker: rotate the offset direction with drone_id around
    # the leg's perpendicular plane so no two drones nudge the same way.
    if n_drones > 1 and M > 1:
        leg = goal - start
        leg_norm = float(np.linalg.norm(leg))
        if leg_norm > 1e-9:
            leg_hat = leg / leg_norm
            world_up = np.array([0.0, 0.0, 1.0])
            perp1 = np.cross(leg_hat, world_up)
            p1n = float(np.linalg.norm(perp1))
            if p1n > 1e-6:
                perp1 = perp1 / p1n
            else:
                perp1 = np.array([1.0, 0.0, 0.0])
            perp2 = np.cross(leg_hat, perp1)
            theta = 2.0 * np.pi * drone_id / n_drones
            offset_dir = np.cos(theta) * perp1 + np.sin(theta) * perp2
            # cap at half the smaller corridor face so we never start out
            # of corridor
            hx, hy, hz = config.corridor_half_extent
            max_offset = 0.5 * min(hx, hy, hz)
            offset = offset_dir * max_offset
            # only the interior waypoints get the offset; endpoints stay fixed
            for k in range(1, M):
                waypoints[k] = waypoints[k] + offset

    leg_length = float(np.linalg.norm(goal - start))
    seg_length = leg_length / M
    seg_time = max(0.5, seg_length / config.v_max)
    durations = np.full(M, seg_time, dtype=np.float64)
    bc_start = np.zeros((s + 1, D), dtype=np.float64)
    bc_start[0] = start
    bc_end = np.zeros((s + 1, D), dtype=np.float64)
    bc_end[0] = goal

    traj = Trajectory(waypoints, durations, bc_start, bc_end, s=s)
    polytopes = [
        _corridor_box(waypoints[k], waypoints[k + 1], config.corridor_half_extent)
        for k in range(M)
    ]
    return traj, polytopes


@dataclass
class Drone:
    drone_id: int
    start: np.ndarray
    goal: np.ndarray
    trajectory: Trajectory
    polytopes: List[Polytope]
    broadcaster: SwarmBroadcaster
    t_broadcast: float = 0.0    # when the current trajectory was sent
    bytes_sent: int = 0
    # Avenue 1: warm-start state. Tracks whether we have a previous OPTIMISED
    # solution to seed the next L-BFGS call. Initial trajectory from
    # _initial_trajectory does NOT count as warm — it's a straight-line guess.
    _has_warm_start: bool = False
    # Aggregate stats for instrumentation (read in run_one_trial → metrics)
    n_skipped: int = 0
    n_reduced: int = 0
    n_full: int = 0
    # Avenue 3: multi-branch stats
    n_multibranch_triggered: int = 0
    n_multibranch_branch_won: int = 0
    _prev_signature: Optional[Tuple[int, ...]] = None
    n_signature_held: int = 0
    n_signature_switched: int = 0
    # Silent-exception canaries (catch the next time something quietly breaks)
    n_optim_exceptions: int = 0
    last_optim_exception: Optional[str] = None
    # Avenue 2: Bayesian filter
    _bayesian_filter: Optional[object] = None  # BayesianWarmStartFilter
    n_filter_predicted: int = 0
    n_filter_resets: int = 0

    def reoptimise(
        self,
        t_now: float,
        config: Rig2Config,
    ) -> float:
        """Re-optimise own trajectory against currently-known neighbours.

        Returns wall-clock elapsed milliseconds.
        """
        snapshots = self.broadcaster.latest()
        sw_cfg = SwarmPenaltyConfig(
            clearance_horizontal=config.clearance_horizontal,
            clearance_vertical=config.clearance_vertical,
            weight=config.swarm_weight,
            n_quad=config.gcopter_n_quad,
            freshness_max_age_s=config.swarm_freshness_max_age_s,
        )
        from src.swarm.swarm_penalty import freshness_from_staleness
        neighbours = [
            (snap.trajectory, snap.t_sent - t_now)
            for snap in snapshots.values()
        ]
        freshnesses = [
            freshness_from_staleness(
                t_now - snap.t_sent, sw_cfg.freshness_max_age_s
            )
            for snap in snapshots.values()
        ]

        gc_cfg = GCopterConfig(
            s=self.trajectory.s,
            v_max=config.v_max,
            n_quad=config.gcopter_n_quad,
            maxiter=config.gcopter_maxiter,
        )

        bc_start = np.zeros((self.trajectory.s + 1, self.trajectory.D))
        bc_start[0] = self.start
        bc_end = np.zeros((self.trajectory.s + 1, self.trajectory.D))
        bc_end[0] = self.goal

        # Avenue 2: Bayesian-filter initial guess prediction.
        # If the filter has enough updates and is confident, replace the
        # raw previous-trajectory initial guess with the filter's smoothed
        # estimate. On unconfident or first call, fall through to the
        # default (self.trajectory.waypoints/durations as-is).
        init_waypoints = self.trajectory.waypoints.copy()
        init_durations = self.trajectory.durations.copy()
        if config.enable_bayesian_warm_start:
            from src.single_drone.planning.bayesian_warm_start import (
                BayesianWarmStartFilter, pack_state, unpack_state,
            )
            if self._bayesian_filter is None:
                self._bayesian_filter = BayesianWarmStartFilter(
                    process_noise=config.bayesian_process_noise,
                    observation_noise=config.bayesian_observation_noise,
                    innovation_reset_threshold=(
                        config.bayesian_innovation_reset_threshold
                    ),
                )
            f = self._bayesian_filter
            if f.confidence() >= config.bayesian_confidence_threshold:
                x_pred = f.predict()
                if x_pred is not None:
                    pred_wps, pred_T = unpack_state(x_pred, init_waypoints)
                    # Sanity: durations must be positive and within bounds
                    if np.all(pred_T > 0.05) and np.all(pred_T < 30.0):
                        init_waypoints = pred_wps
                        init_durations = pred_T
                        self.n_filter_predicted += 1

        t0 = time.perf_counter()
        try:
            if config.enable_multi_branch:
                from src.swarm.topology_branches import (
                    MultiBranchConfig, multi_branch_optimize,
                )
                mb_cfg = MultiBranchConfig(
                    n_branches=config.multi_branch_n,
                    perturbation_scale=config.multi_branch_perturbation_scale,
                    trigger_dist_fraction=config.multi_branch_trigger_dist_fraction,
                )
                mb_result = multi_branch_optimize(
                    initial_waypoints=init_waypoints,
                    initial_durations=init_durations,
                    bc_start=bc_start, bc_end=bc_end,
                    polytopes=self.polytopes, config=gc_cfg,
                    swarm_neighbours=neighbours, swarm_config=sw_cfg,
                    swarm_freshnesses=freshnesses,
                    warm_start=self._has_warm_start,
                    branch_config=mb_cfg,
                    swarm_clearance_horizontal=config.clearance_horizontal,
                    prev_signature=self._prev_signature,
                )
                self.trajectory = mb_result.trajectory
                # Track signature continuity for diagnostics
                if (self._prev_signature is not None
                        and mb_result.signature == self._prev_signature):
                    self.n_signature_held += 1
                else:
                    self.n_signature_switched += 1
                self._prev_signature = mb_result.signature
                # Stats: did multi-branch even get triggered (>1 branch run)?
                if mb_result.n_branches_run > 1:
                    self.n_multibranch_triggered += 1
                    if not mb_result.main_branch_used:
                        self.n_multibranch_branch_won += 1
                # Approximate skipped/reduced/full from single-branch path
                self.n_full += 1
            else:
                traj, meta = gcopter_optimize(
                    initial_waypoints=init_waypoints,
                    initial_durations=init_durations,
                    bc_start=bc_start,
                    bc_end=bc_end,
                    polytopes=self.polytopes,
                    config=gc_cfg,
                    swarm_neighbours=neighbours,
                    swarm_config=sw_cfg,
                    swarm_freshnesses=freshnesses,
                    warm_start=self._has_warm_start,
                    return_meta=True,
                )
                self.trajectory = traj
                if meta["skipped"]:
                    self.n_skipped += 1
                elif meta["maxiter_used"] < gc_cfg.maxiter:
                    self.n_reduced += 1
                else:
                    self.n_full += 1
            # After the first successful optimise, future calls are warm.
            self._has_warm_start = True
            # Avenue 2: feed the new optimum into the Bayesian filter so
            # subsequent calls have an updated state estimate. Track reset
            # count for diagnostics.
            if config.enable_bayesian_warm_start and self._bayesian_filter is not None:
                from src.single_drone.planning.bayesian_warm_start import pack_state
                prev_resets = self._bayesian_filter.n_resets
                self._bayesian_filter.update(
                    pack_state(self.trajectory.waypoints, self.trajectory.durations)
                )
                if self._bayesian_filter.n_resets > prev_resets:
                    self.n_filter_resets += 1
        except Exception as _exc:  # pragma: no cover — defensive
            # Record so silent swallowing never again hides a real bug.
            # Diagnostics field is added on the Drone dataclass.
            self.last_optim_exception = repr(_exc)
            self.n_optim_exceptions += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms

    def broadcast(self, t_now: float) -> int:
        n_bytes = self.broadcaster.broadcast(self.trajectory, t_now)
        self.t_broadcast = t_now
        self.bytes_sent += n_bytes
        return n_bytes


# ---------------------------------------------------------------------------
# Distance / collision analytics
# ---------------------------------------------------------------------------


def _sample_positions(
    drones: Sequence[Drone],
    t_grid: np.ndarray,
) -> np.ndarray:
    """Return positions of shape (n_drones, n_steps, 3)."""
    n = len(drones)
    m = t_grid.size
    out = np.zeros((n, m, 3), dtype=np.float64)
    for i, dr in enumerate(drones):
        T = float(dr.trajectory.total_time)
        for j, t in enumerate(t_grid):
            tc = min(max(0.0, float(t)), T)
            out[i, j] = dr.trajectory.evaluate(tc, 0)
    return out


def _sample_velocities(
    drones: Sequence[Drone],
    t_grid: np.ndarray,
) -> np.ndarray:
    """Return velocities of shape (n_drones, n_steps, 3).

    Reads the first time-derivative of the MINCO trajectory at each t.
    Out-of-range t values return zero velocity (drone has reached goal).
    """
    n = len(drones)
    m = t_grid.size
    out = np.zeros((n, m, 3), dtype=np.float64)
    for i, dr in enumerate(drones):
        T = float(dr.trajectory.total_time)
        for j, t in enumerate(t_grid):
            tf = float(t)
            if tf < 0.0 or tf > T:
                continue  # zero velocity at goal / before start
            out[i, j] = dr.trajectory.evaluate(tf, 1)
    return out


def _pairwise_min_distance_metrics(
    positions: np.ndarray,
    near_miss_radius: float,
    collision_radius: float,
) -> Dict[str, float]:
    """Compute per-frame inter-drone minimum distances.

    positions : (N, M, 3)
    """
    N, M, _ = positions.shape
    if N < 2:
        return {
            "d_min_inter_m": float("inf"),
            "d_mean_inter_m": float("inf"),
            "near_misses": 0,
            "collisions": 0,
        }
    all_min = np.inf
    sum_min = 0.0
    nm = 0
    coll = 0
    for j in range(M):
        # pairwise distance
        diff = positions[:, j, :][:, None, :] - positions[None, :, j, :]
        d = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(N, k=1)
        pair_d = d[iu]
        frame_min = float(pair_d.min())
        if frame_min < all_min:
            all_min = frame_min
        sum_min += frame_min
        nm += int(np.sum(pair_d < near_miss_radius))
        coll += int(np.sum(pair_d < collision_radius))
    return {
        "d_min_inter_m": all_min,
        "d_mean_inter_m": sum_min / max(1, M),
        "near_misses": nm,
        "collisions": coll,
    }


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_one_trial(
    seed: int,
    n_drones: int,
    scenario: str,
    config: Optional[Rig2Config] = None,
    keep_record: bool = False,
) -> Dict[str, float]:
    if config is None:
        config = Rig2Config()
    rng = np.random.default_rng(seed)

    result: Dict[str, float] = {
        "seed": seed,
        "n_drones": n_drones,
        "scenario": scenario,
        "success": False,
    }
    viz: Optional[Dict] = (
        {
            "n_drones": n_drones,
            "scenario": scenario,
            "sample_dt_s": float(config.sample_dt_s),
            "near_miss_radius_m": float(config.near_miss_radius),
        }
        if keep_record else None
    )

    # ---- 1. endpoints
    try:
        endpoints = endpoints_for_scenario(scenario, n_drones, config)
    except ValueError as e:
        result["error"] = f"scenario:{e}"
        return result
    if viz is not None:
        viz["endpoints"] = [
            {"start": s.tolist(), "goal": g.tolist()} for s, g in endpoints
        ]

    # ---- 2. channel + per-drone setup
    channel = BroadcastChannel(
        config=ChannelConfig(
            latency_ms_mean=config.comms_latency_ms_mean,
            latency_ms_jitter=config.comms_latency_ms_jitter,
            packet_loss_pct=config.comms_loss_pct,
            bandwidth_kbps=config.comms_bandwidth_kbps,
        ),
        n_agents=n_drones,
        rng=np.random.default_rng(rng.integers(1 << 31)),
    )

    drones: List[Drone] = []
    for idx, (start, goal) in enumerate(endpoints):
        traj0, polys = _initial_trajectory(
            start, goal, config, drone_id=idx, n_drones=n_drones
        )
        broadcaster = SwarmBroadcaster(idx, channel)
        drone = Drone(
            drone_id=idx,
            start=start,
            goal=goal,
            trajectory=traj0,
            polytopes=polys,
            broadcaster=broadcaster,
        )
        drone.broadcast(t_now=0.0)
        drones.append(drone)

    # ---- 3. replan loop
    t = 0.0
    replan_ticks = 0
    sum_t_replan_ms = 0.0
    max_t_replan_ms = 0.0
    sum_per_agent_ms = 0.0

    while t < config.sim_duration_s:
        t += config.replan_period_s
        for dr in drones:
            dr.broadcaster.poll(t_now=t)
        tick_max = 0.0
        tick_sum = 0.0
        for dr in drones:
            elapsed = dr.reoptimise(t_now=t, config=config)
            tick_sum += elapsed
            if elapsed > tick_max:
                tick_max = elapsed
            dr.broadcast(t_now=t)
        replan_ticks += 1
        sum_t_replan_ms += tick_sum
        sum_per_agent_ms += tick_sum / max(1, n_drones)
        if tick_max > max_t_replan_ms:
            max_t_replan_ms = tick_max

    # ---- 4. sample positions over the union trajectory window
    horizon = max(dr.trajectory.total_time for dr in drones)
    n_samples = int(np.ceil(horizon / config.sample_dt_s)) + 1
    t_grid = np.linspace(0.0, horizon, n_samples)
    positions = _sample_positions(drones, t_grid)
    dist_metrics = _pairwise_min_distance_metrics(
        positions, config.near_miss_radius, config.collision_radius
    )

    # ---- 4b. Avenue 4: CBF safety filter as post-MINCO layer ---------------
    # Sample velocities, run the filter, and report both raw and CBF-filtered
    # collision metrics so the rig can show the safety-overlay impact.
    cbf_metrics: Dict[str, float] = {}
    if config.enable_cbf_filter:
        from src.swarm.cbf_safety_filter import (
            CBFConfig, apply_cbf_filter,
        )
        velocities = _sample_velocities(drones, t_grid)
        # cbf module expects (T, N, 3); rig stores as (N, T, 3)
        pos_T_first = np.transpose(positions, (1, 0, 2))
        vel_T_first = np.transpose(velocities, (1, 0, 2))
        dt = config.sample_dt_s
        cbf_cfg = CBFConfig(
            clearance=config.clearance_horizontal,
            alpha=config.cbf_alpha,
            max_velocity_correction=config.cbf_max_velocity_correction,
            apply_to_positions=True,
        )
        cbf_result = apply_cbf_filter(pos_T_first, vel_T_first, dt, cbf_cfg)
        # Re-transpose to rig's (N, T, 3) convention
        filtered_positions = np.transpose(cbf_result.filtered_positions, (1, 0, 2))
        cbf_dist_metrics = _pairwise_min_distance_metrics(
            filtered_positions, config.near_miss_radius, config.collision_radius
        )
        cbf_metrics = {
            "cbf_interventions": int(cbf_result.total_interventions),
            "cbf_max_correction_m_s": float(cbf_result.max_correction_magnitude),
            "cbf_n_infeasible": int(cbf_result.n_infeasible),
            "cbf_d_min_inter_m": float(cbf_dist_metrics["d_min_inter_m"]),
            "cbf_d_mean_inter_m": float(cbf_dist_metrics["d_mean_inter_m"]),
            "cbf_near_misses": int(cbf_dist_metrics["near_misses"]),
            "cbf_collisions": int(cbf_dist_metrics["collisions"]),
        }

    # ---- 5. comms / bandwidth
    ch_stats = channel.stats()
    total_bytes = sum(dr.bytes_sent for dr in drones)
    total_kbps = total_bytes * 8.0 / 1024.0 / max(t, 1e-6)
    pkts_sent_per_rx = ch_stats["sent"] * max(1, n_drones - 1)
    network_congestion_pct = 0.0
    if ch_stats["sent"] > 0:
        # crude proxy for congestion: dropped / attempted-deliveries
        delivered_or_dropped = max(1, ch_stats["delivered"] + ch_stats["dropped"])
        network_congestion_pct = (
            100.0 * ch_stats["dropped"] / delivered_or_dropped
        )

    result.update(
        {
            "t_replan_total_ms": sum_t_replan_ms,
            "t_replan_mean_ms": sum_t_replan_ms / max(1, replan_ticks),
            "t_replan_max_ms": max_t_replan_ms,
            "t_replan_per_agent_mean_ms": sum_per_agent_ms / max(1, replan_ticks),
            "broadcast_bandwidth_kbps": total_kbps,
            "network_congestion_pct": network_congestion_pct,
            "packets_sent": float(ch_stats["sent"]),
            "packets_delivered": float(ch_stats["delivered"]),
            "packets_dropped": float(ch_stats["dropped"]),
            **dist_metrics,
            **cbf_metrics,
        }
    )
    # When CBF is the deployed safety layer, the user-facing "collisions"
    # and "d_min_inter_m" should be the CBF-FILTERED counts — that's what
    # the rig will produce in the air. The raw (pre-filter) values remain
    # available as `raw_collisions` / `raw_d_min_inter_m` for diagnosis.
    if config.enable_cbf_filter and cbf_metrics:
        result["raw_collisions"] = result["collisions"]
        result["raw_d_min_inter_m"] = result.get("d_min_inter_m", float("nan"))
        result["collisions"] = cbf_metrics["cbf_collisions"]
        result["d_min_inter_m"] = cbf_metrics["cbf_d_min_inter_m"]
        result["near_misses"] = cbf_metrics["cbf_near_misses"]
    result["success"] = result["collisions"] == 0

    if viz is not None:
        # `positions` is shape (N, M, 3) — sampled at sample_dt over the
        # union-trajectory horizon. Convert to JSON-able lists.
        viz["positions_per_drone"] = positions.tolist()
        viz["d_min_inter_m"] = result.get("d_min_inter_m", float("nan"))
        viz["collisions"] = int(result.get("collisions", 0))
        result["viz_record"] = viz

    return result


def run_stress_matrix(
    drones_list: Sequence[int],
    scenario: str,
    latencies_ms: Sequence[float],
    losses_pct: Sequence[float],
    runs_per_combo: int,
    config: Optional[Rig2Config] = None,
    base_seed: int = 2500,
    verbose: bool = True,
) -> MetricsCollector:
    """Cartesian sweep over (n_drones, latency_ms, loss_pct).

    Adds two label keys to each row, `comms_latency_ms` and `comms_loss_pct`,
    so the summary breaks results down by communications stress as well as
    fleet size.
    """
    base_cfg = config if config is not None else Rig2Config()
    mc = MetricsCollector()
    combo_idx = 0
    for n in drones_list:
        for lat in latencies_ms:
            for loss in losses_pct:
                cfg = Rig2Config(
                    **{
                        **base_cfg.__dict__,
                        "comms_latency_ms_mean": float(lat),
                        "comms_loss_pct": float(loss),
                    }
                )
                if verbose:
                    print(
                        f"\n--- n={n}, latency={lat:.0f}ms, loss={loss:.0f}% ---"
                    )
                for run_idx in range(runs_per_combo):
                    seed = base_seed + combo_idx * 10_000 + run_idx
                    row = run_one_trial(seed, n, scenario, cfg)
                    mc.start_run(
                        n_drones=n,
                        comms_latency_ms=float(lat),
                        comms_loss_pct=float(loss),
                        scenario=scenario,
                        seed=seed,
                    )
                    for k, v in row.items():
                        if k in (
                            "n_drones", "scenario", "seed",
                        ):
                            continue
                        mc.record(k, v)
                    mc.finish_run()
                    if verbose:
                        ok = row.get("success", False)
                        dmin = row.get("d_min_inter_m", float("nan"))
                        tpa = row.get("t_replan_per_agent_mean_ms", float("nan"))
                        cong = row.get("network_congestion_pct", 0.0)
                        print(
                            f"  run {run_idx + 1}/{runs_per_combo}: "
                            f"success={ok}  d_min={dmin:5.2f}m  "
                            f"t/agent={tpa:6.1f}ms  congestion={cong:5.1f}%",
                            flush=True,
                        )
                combo_idx += 1
    return mc


def assert_scaling_is_flat(
    mc: MetricsCollector,
    *,
    small_n: int,
    large_n: int,
    factor: float = 2.0,
) -> Tuple[bool, float, float]:
    """Verify per-agent replan time stays within `factor` × from `small_n`
    drones to `large_n` (the spec's O(k) scaling claim).

    Pulls medians from `mc.runs`. Returns (passed, t_small, t_large).
    """
    rows = mc.to_records()

    def median_per_agent(n: int) -> float:
        vals = [
            float(r["t_replan_per_agent_mean_ms"])
            for r in rows
            if int(r.get("n_drones", -1)) == n
            and isinstance(r.get("t_replan_per_agent_mean_ms"), (int, float))
        ]
        if not vals:
            return float("nan")
        return float(np.median(vals))

    t_small = median_per_agent(small_n)
    t_large = median_per_agent(large_n)
    if not (np.isfinite(t_small) and np.isfinite(t_large) and t_small > 0):
        return False, t_small, t_large
    return bool(t_large <= factor * t_small), t_small, t_large


def run_benchmark(
    drones_list: Sequence[int],
    scenario: str,
    runs_per_size: int,
    config: Optional[Rig2Config] = None,
    base_seed: int = 2000,
    verbose: bool = True,
) -> MetricsCollector:
    if config is None:
        config = Rig2Config()
    mc = MetricsCollector()
    for idx_n, n in enumerate(drones_list):
        if verbose:
            print(f"\n--- n_drones = {n}, scenario = {scenario} ---")
        for run_idx in range(runs_per_size):
            seed = base_seed + idx_n * 10_000 + run_idx
            row = run_one_trial(seed, n, scenario, config)
            mc.start_run(n_drones=n, scenario=scenario, seed=seed)
            for k, v in row.items():
                if k in ("n_drones", "scenario", "seed"):
                    continue
                mc.record(k, v)
            mc.finish_run()
            if verbose:
                ok = row.get("success", False)
                dmin = row.get("d_min_inter_m", float("nan"))
                tpa = row.get("t_replan_per_agent_mean_ms", float("nan"))
                err = row.get("error", "")
                line = (
                    f"  run {run_idx + 1}/{runs_per_size}: success={ok}  "
                    f"d_min={dmin:5.2f}m  t/agent={tpa:6.1f}ms"
                )
                if err:
                    line += f"  [{err}]"
                print(line, flush=True)
    return mc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_summary(summary: Dict[str, dict]) -> str:
    rows = []
    rows.append(
        f"{'group':28s}  {'runs':>5s}  {'succ%':>7s}  {'d_min med':>10s}  "
        f"{'near_m sum':>11s}  {'coll sum':>9s}  {'t/agent med':>12s}  "
        f"{'bw med':>9s}"
    )
    rows.append("-" * len(rows[0]))
    for group, agg in summary.items():
        n = agg.get("n_runs", 0)
        sr = agg.get("success_rate", 0.0) * 100
        dmin = agg.get("d_min_inter_m", {}).get("median", float("nan"))
        nm = agg.get("near_misses", {}).get("mean", 0.0) * n
        coll = agg.get("collisions", {}).get("mean", 0.0) * n
        tpa = agg.get("t_replan_per_agent_mean_ms", {}).get("median", float("nan"))
        bw = agg.get("broadcast_bandwidth_kbps", {}).get("median", float("nan"))
        rows.append(
            f"{group:28s}  {n:5d}  {sr:6.1f}%  {dmin:10.3f}  "
            f"{nm:11.1f}  {coll:9.1f}  {tpa:12.1f}  {bw:9.2f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rig 2: swarm avoidance scaling benchmark"
    )
    parser.add_argument(
        "--drones",
        type=str,
        default="3,6,12",
        help="Comma-separated drone counts (default: 3,6,12)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="patrol",
        choices=list(SCENARIOS),
        help="Conflict scenario (default: patrol)",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Runs per drone-count (default: 3)"
    )
    parser.add_argument(
        "--output", type=str, default="rig2_results.json",
        help="JSON output path",
    )
    parser.add_argument("--sim-duration", type=float, default=8.0)
    parser.add_argument("--replan-period", type=float, default=1.0)
    parser.add_argument("--comms-latency-ms", type=float, default=50.0)
    parser.add_argument("--comms-loss-pct", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--v-max", type=float, default=4.0)
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Sweep the full latency × loss × N matrix (spec §5.3).",
    )
    parser.add_argument(
        "--latencies",
        type=str,
        default="50,100,200",
        help="Stress-mode: comma-separated latency_ms_mean values",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="0,10,30",
        help="Stress-mode: comma-separated packet_loss_pct values",
    )
    parser.add_argument(
        "--scaling-check",
        action="store_true",
        help="In stress mode, assert per-agent replan time stays within 2× "
        "from min(drones) to max(drones) and exit non-zero on failure.",
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
        help="If set, run one extra detailed trial at --viz-drones and write "
        "an interactive Plotly HTML there.",
    )
    parser.add_argument("--viz-drones", type=int, default=3,
                        help="Drone count for the viz trial (default: 3).")
    parser.add_argument("--viz-seed", type=int, default=2024,
                        help="Seed for the viz trial (default: 2024).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    drones_list = [int(x) for x in args.drones.split(",")]
    if args.scenario in ("head_on",):
        if any(n != 2 for n in drones_list):
            print("head_on requires drones=2", file=sys.stderr)
            return 2
    if args.scenario in ("crossing", "converge"):
        if any(n != 3 for n in drones_list):
            print(f"{args.scenario} requires drones=3", file=sys.stderr)
            return 2

    config = Rig2Config(
        v_max=args.v_max,
        gcopter_maxiter=args.maxiter,
        comms_latency_ms_mean=args.comms_latency_ms,
        comms_loss_pct=args.comms_loss_pct,
        replan_period_s=args.replan_period,
        sim_duration_s=args.sim_duration,
    )

    if args.stress:
        latencies = [float(x) for x in args.latencies.split(",")]
        losses = [float(x) for x in args.losses.split(",")]
        print(
            f"Rig 2 STRESS — scenario={args.scenario}, drones={drones_list}, "
            f"latencies={latencies}ms, losses={losses}%, {args.runs} runs/combo, "
            f"replan={args.replan_period}s, sim={args.sim_duration}s"
        )
        mc = run_stress_matrix(
            drones_list=drones_list,
            scenario=args.scenario,
            latencies_ms=latencies,
            losses_pct=losses,
            runs_per_combo=args.runs,
            config=config,
            verbose=not args.quiet,
        )
        mc.export_json(
            args.output,
            label_keys=["n_drones", "comms_latency_ms", "comms_loss_pct"],
        )
        print(f"\nResults written to {args.output}")
        summary = summarise(
            mc.runs,
            label_keys=["n_drones", "comms_latency_ms", "comms_loss_pct"],
        )
        print("\n=== Summary ===")
        print(_format_summary(summary))

        if args.scaling_check:
            ok, t_small, t_large = assert_scaling_is_flat(
                mc,
                small_n=min(drones_list),
                large_n=max(drones_list),
                factor=2.0,
            )
            print(
                f"\nScaling check: t_replan/agent at N={min(drones_list)} = "
                f"{t_small:.1f} ms, at N={max(drones_list)} = "
                f"{t_large:.1f} ms"
            )
            if ok:
                print("  PASS (within 2× target)")
            else:
                print("  FAIL (exceeded 2× target)")
                return 1

        if args.plot:
            from src.validation.plots import emit_plot
            emit_plot("rig2", mc.runs, args.plot)
            print(f"Plot written to {args.plot}")
        return 0

    print(
        f"Rig 2 — scenario={args.scenario}, drones={drones_list}, "
        f"{args.runs} runs each, replan={args.replan_period}s, "
        f"sim={args.sim_duration}s, comms_latency={args.comms_latency_ms}ms, "
        f"loss={args.comms_loss_pct}%"
    )

    mc = run_benchmark(
        drones_list=drones_list,
        scenario=args.scenario,
        runs_per_size=args.runs,
        config=config,
        verbose=not args.quiet,
    )

    mc.export_json(args.output, label_keys=["n_drones", "scenario"])
    print(f"\nResults written to {args.output}")

    summary = summarise(mc.runs, label_keys=["n_drones", "scenario"])
    print("\n=== Summary ===")
    print(_format_summary(summary))

    if args.plot:
        from src.validation.plots import emit_plot
        emit_plot("rig2", mc.runs, args.plot)
        print(f"Plot written to {args.plot}")

    if args.viz:
        from src.validation.visualize import emit_viz
        row = run_one_trial(
            args.viz_seed, args.viz_drones, args.scenario, config,
            keep_record=True,
        )
        record = row.get("viz_record")
        if record is None:
            print(
                f"Viz trial failed (error={row.get('error', '?')})",
                file=sys.stderr,
            )
        else:
            emit_viz("rig2", record, args.viz)
            print(f"Viz written to {args.viz}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
