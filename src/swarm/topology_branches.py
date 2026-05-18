"""Topology-guided multi-branch trajectory optimisation (Avenue 3).

Inspired by TRUST-Planner (arXiv 2508.14610, Aug 2025) and the broader
homotopy-class enumeration literature. The motivation: in symmetric or
dense multi-agent scenarios — Rig 2 converge with N drones aimed at the
same point, patrol N≥6 with crossings at the centre — L-BFGS gets stuck
at saddle points because multiple equally-good homotopy classes exist
(go left vs right vs over vs under each neighbour) and no local gradient
information distinguishes them.

The fix: enumerate `k` topology hints by perturbing the interior waypoints
along directions that span the homotopy space (lateral and vertical from
the drone's forward axis), run each as a separate L-BFGS branch, pick the
lowest-cost feasible result.

Key design decisions
--------------------
1. **Adaptive triggering.** Naive multi-branch is k× more expensive. We
   first run a single warm-started branch (cheap, ~20ms in Rig 2) and
   only escalate to multi-branch if the result shows predicted swarm
   violations (min inter-agent distance below threshold). Easy scenarios
   pay 1× cost; hard scenarios pay (k+1)×.

2. **Perturbation generation, not random restarts.** The branches are
   structured (lateral ±, vertical ±) rather than random. This is what
   makes them "topology hints" — each branch represents a different
   homotopy class of how the drone passes its neighbours.

3. **Cost-based tie-break, collision-first.** When picking the winner
   among branches, prefer ones with zero predicted swarm violations.
   Among those, pick lowest total cost. This avoids picking a
   "smoother but colliding" trajectory over a "rougher but safe" one.

What's simplified from TRUST-Planner
------------------------------------
The reference paper enumerates topologies via a dynamic-enhanced visible
PRM front-end and incrementally manages a multi-branch tree across
replan ticks. We do a per-replan k-branch enumeration without the PRM
or the cross-tick tree management. This captures the saddle-escape
mechanic without the full topology-tracking machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import (
    GCopterConfig, _evaluate_cost, gcopter_optimize,
)
from src.single_drone.planning.minco import Trajectory


@dataclass
class MultiBranchConfig:
    """Multi-branch optimisation hyperparameters."""
    # How many extra branches beyond the main warm-started solve.
    # 4 = lateral ±, vertical ± (covers the main homotopy classes).
    n_branches: int = 4
    # Magnitude of waypoint perturbation in meters. Should be comparable
    # to clearance_horizontal so the branches really land in different
    # homotopy classes rather than just nudging within the same.
    perturbation_scale: float = 1.0
    # Adaptive trigger: if min predicted inter-agent distance from the
    # main solve is below this threshold (relative to clearance), fall
    # back to multi-branch. Default 1.0 means "trigger if predicted
    # distance is below the swarm clearance" — i.e. an actual violation.
    trigger_dist_fraction: float = 1.0
    # Cap branch count even when adaptive trigger fires. Useful to bound
    # worst-case wall time. 0 means no extra cap.
    max_branches_when_triggered: int = 4
    # When evaluating branches for "best", prefer collision-free even if
    # the cost is slightly higher.
    prefer_collision_free: bool = True


@dataclass
class MultiBranchResult:
    trajectory: Trajectory
    total_cost: float
    main_branch_used: bool       # True if the warm-started main solve won
    n_branches_run: int          # incl. the main solve
    branch_costs: List[float] = field(default_factory=list)
    branch_min_dists: List[float] = field(default_factory=list)
    selected_branch_idx: int = 0


def _generate_topology_perturbations(
    initial_waypoints: np.ndarray,
    perturbation_scale: float,
    n_branches: int,
) -> List[np.ndarray]:
    """Generate up to `n_branches` perturbed copies of initial_waypoints.

    Returns a list of (M+1, D) arrays. Each perturbation offsets the
    interior waypoints (rows 1..M-1) along a different direction in the
    drone's body frame (built from start→goal as forward axis).

    Order: lateral +, lateral -, vertical +, vertical -, then bias-toward
    each cardinal direction in order. Cropped to `n_branches`.
    """
    if initial_waypoints.shape[0] < 3:
        # No interior waypoints to perturb.
        return []

    forward = initial_waypoints[-1] - initial_waypoints[0]
    fwd_norm = np.linalg.norm(forward)
    if fwd_norm < 1e-6:
        return []
    forward = forward / fwd_norm

    # Lateral (horizontal-perpendicular) direction. If forward is mostly
    # vertical, fall back to world +x.
    if abs(forward[2]) > 0.99:
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral = np.array([-forward[1], forward[0], 0.0])
        lateral /= max(np.linalg.norm(lateral), 1e-9)
    vertical = np.array([0.0, 0.0, 1.0])

    # Reduced perturbation in z so vertical branches don't fly into the
    # ceiling/floor of the corridor polytopes.
    z_scale = 0.5 * perturbation_scale

    directions: List[np.ndarray] = [
        +perturbation_scale * lateral,
        -perturbation_scale * lateral,
        +z_scale * vertical,
        -z_scale * vertical,
    ]

    out: List[np.ndarray] = []
    for d in directions[:n_branches]:
        perturbed = initial_waypoints.copy()
        perturbed[1:-1] += d  # offset interior waypoints
        out.append(perturbed)
    return out


def _min_predicted_swarm_distance(
    trajectory: Trajectory,
    swarm_neighbours: Sequence[Tuple[Trajectory, float]],
    n_samples: int = 16,
) -> float:
    """Sample own + neighbour trajectories on a common time grid, return
    minimum pairwise distance over the sample window."""
    if not swarm_neighbours:
        return float("inf")
    horizon = trajectory.total_time
    ts = np.linspace(0.0, horizon, n_samples)
    own_xyz = np.stack([trajectory.evaluate(t) for t in ts], axis=0)
    min_d = float("inf")
    for (nb_traj, t_offset) in swarm_neighbours:
        nb_horizon = nb_traj.total_time
        nb_xyz = np.stack(
            [nb_traj.evaluate(min(max(t + t_offset, 0.0), nb_horizon))
             for t in ts],
            axis=0,
        )
        diffs = own_xyz - nb_xyz
        dists = np.linalg.norm(diffs, axis=1)
        min_d = min(min_d, float(np.min(dists)))
    return min_d


def _total_cost_with_swarm(
    trajectory: Trajectory,
    polytopes: Sequence[Polytope],
    gc_config: GCopterConfig,
    swarm_neighbours: Sequence,
    swarm_config: object,
) -> float:
    """Single-drone cost + swarm cost (matches what L-BFGS sees)."""
    cost = _evaluate_cost(trajectory, polytopes, gc_config)
    if swarm_neighbours and swarm_config is not None:
        from src.swarm.swarm_penalty import compute_swarm_cost_and_grad
        sc, _, _ = compute_swarm_cost_and_grad(
            trajectory, list(swarm_neighbours), swarm_config
        )
        cost += sc
    return float(cost)


def _is_better_branch(
    cand_dist: float, cand_cost: float,
    best_dist: float, best_cost: float,
    clearance: float,
    distance_tie_band: float = 0.10,
) -> bool:
    """Lexicographic comparison for branch selection.

    Cost values vary by orders of magnitude across homotopy classes —
    a trajectory that escapes the corridor entirely can have lower
    swarm cost simply because it never approaches a neighbour. So we
    rank PRIMARILY by predicted min inter-agent distance (monotonic
    in safety), and only use cost as a fine tiebreak.

    Rules (return True iff candidate is better than best):
      1. If one is above clearance and the other is below: above wins.
      2. Otherwise: larger min_dist wins, with `distance_tie_band` slack.
      3. If distances are within the tie band: lower cost wins.
    """
    cand_safe = cand_dist >= clearance
    best_safe = best_dist >= clearance
    if cand_safe and not best_safe:
        return True
    if best_safe and not cand_safe:
        return False
    # Same safety class: prefer farther from collision
    if cand_dist > best_dist + distance_tie_band:
        return True
    if cand_dist < best_dist - distance_tie_band:
        return False
    # Distances within tie band: use cost
    return cand_cost < best_cost


def multi_branch_optimize(
    initial_waypoints: np.ndarray,
    initial_durations: np.ndarray,
    bc_start: np.ndarray,
    bc_end: np.ndarray,
    polytopes: Sequence[Polytope],
    config: GCopterConfig,
    swarm_neighbours: Optional[Sequence] = None,
    swarm_config: Optional[object] = None,
    warm_start: bool = False,
    branch_config: Optional[MultiBranchConfig] = None,
    swarm_clearance_horizontal: float = 1.0,
) -> MultiBranchResult:
    """Run a warm-started main branch, then escalate to topology branches
    if the main result shows predicted swarm violations.

    Parameters
    ----------
    initial_waypoints, initial_durations, bc_start, bc_end, polytopes :
        Same semantics as gcopter_optimize.
    config :
        GCopterConfig.
    swarm_neighbours, swarm_config :
        Same semantics as gcopter_optimize.
    warm_start :
        Whether the main solve should use the warm-start fast path. The
        topology branches are always cold (their initial guesses are
        perturbed copies, not actual previous solutions).
    branch_config :
        MultiBranchConfig. Defaults applied if None.
    swarm_clearance_horizontal :
        Reference clearance distance used by the trigger heuristic. The
        trigger fires if main-branch predicted min distance is below
        `trigger_dist_fraction * swarm_clearance_horizontal`.
    """
    bc = branch_config if branch_config is not None else MultiBranchConfig()

    # ---- Main warm-started branch ------------------------------------------
    main_traj, _meta = gcopter_optimize(
        initial_waypoints=initial_waypoints,
        initial_durations=initial_durations,
        bc_start=bc_start, bc_end=bc_end,
        polytopes=polytopes, config=config,
        swarm_neighbours=swarm_neighbours,
        swarm_config=swarm_config,
        warm_start=warm_start,
        return_meta=True,
    )
    main_cost = _total_cost_with_swarm(
        main_traj, polytopes, config, swarm_neighbours or [], swarm_config
    )
    main_min_dist = _min_predicted_swarm_distance(
        main_traj, swarm_neighbours or []
    )

    # ---- Trigger check: do we need branches? -------------------------------
    threshold = bc.trigger_dist_fraction * swarm_clearance_horizontal
    needs_branches = (
        bool(swarm_neighbours)
        and bc.n_branches > 0
        and main_min_dist < threshold
    )

    if not needs_branches:
        return MultiBranchResult(
            trajectory=main_traj,
            total_cost=main_cost,
            main_branch_used=True,
            n_branches_run=1,
            branch_costs=[main_cost],
            branch_min_dists=[main_min_dist],
            selected_branch_idx=0,
        )

    # ---- Multi-branch escalation -------------------------------------------
    n_extra = min(bc.n_branches, bc.max_branches_when_triggered or bc.n_branches)
    perturbations = _generate_topology_perturbations(
        initial_waypoints, bc.perturbation_scale, n_extra
    )

    # Score: (collision_free_first, cost). Lower is better on both axes.
    best_traj = main_traj
    best_cost = main_cost
    best_min_dist = main_min_dist
    selected = 0
    branch_costs = [main_cost]
    branch_min_dists = [main_min_dist]

    for i, perturbed_wps in enumerate(perturbations, start=1):
        try:
            cand_traj, _ = gcopter_optimize(
                initial_waypoints=perturbed_wps,
                initial_durations=initial_durations,
                bc_start=bc_start, bc_end=bc_end,
                polytopes=polytopes, config=config,
                swarm_neighbours=swarm_neighbours,
                swarm_config=swarm_config,
                warm_start=False,  # perturbed initial guess: cold
                return_meta=True,
            )
        except Exception:
            branch_costs.append(float("inf"))
            branch_min_dists.append(float("-inf"))
            continue

        cand_cost = _total_cost_with_swarm(
            cand_traj, polytopes, config, swarm_neighbours or [], swarm_config
        )
        cand_min_dist = _min_predicted_swarm_distance(
            cand_traj, swarm_neighbours or []
        )
        branch_costs.append(cand_cost)
        branch_min_dists.append(cand_min_dist)

        # Pick best via lexicographic (safety, distance, cost) rule.
        if _is_better_branch(
            cand_min_dist, cand_cost,
            best_min_dist, best_cost,
            clearance=swarm_clearance_horizontal,
        ):
            best_traj, best_cost, best_min_dist = cand_traj, cand_cost, cand_min_dist
            selected = i

    return MultiBranchResult(
        trajectory=best_traj,
        total_cost=best_cost,
        main_branch_used=(selected == 0),
        n_branches_run=1 + len(perturbations),
        branch_costs=branch_costs,
        branch_min_dists=branch_min_dists,
        selected_branch_idx=selected,
    )
