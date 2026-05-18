"""Topology-guided multi-branch trajectory optimisation (Avenue 3, rebuilt).

Inspired by T-MPC (de Groot et al., arXiv 2401.06021v2). This rebuild
fixes the four failure modes identified after the first attempt:

  Failure 1 — Branches used pure cold-start L-BFGS (~127 evals each).
              FIX: intermediate warm-start config (maxiter=10, ftol=1e-4,
              maxls=10), cutting per-branch cost roughly in half.

  Failure 2 — Branches collapsed to the same homotopy class because the
              swarm/corridor penalty pulled perturbed initial guesses
              back to the same local minimum.
              FIX: pass a HomotopyPenaltyContext per branch. The penalty
              enforces target homotopy signature via analytical gradient
              and survives the L-BFGS optimisation.

  Failure 3 — Broadcast instability: when all branches were colliding,
              cost was noise-dominated and the winning branch index
              changed each tick. Neighbours saw oscillating broadcasts.
              FIX: consistency bonus on the prior-tick signature, plus a
              "main_solve always considered" rule so single-branch
              equilibria remain available.

  Failure 4 — Trigger fired on "predicted collision," which is true even
              for unsolvable scenarios (converge). A3 would then keep
              trying to find a better homotopy class when none existed.
              FIX: trigger only when L-BFGS did NOT converge (hit
              maxiter) AND main solve has predicted violations. A
              converged-but-colliding solution is the best the local
              optimiser can do; A3 should respect it.

The interface
=============
The caller is the rig's Drone.reoptimise. Each Drone tracks its
previous-tick signature; that is passed in as `prev_signature` here.
On return, the caller stores `result.signature` for the next tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.gcopter import (
    GCopterConfig, _evaluate_cost, gcopter_optimize,
)
from src.single_drone.planning.minco import Trajectory
from src.swarm.homotopy import (
    HomotopyPenaltyContext,
    build_penalty_context,
    full_signature,
    generate_target_signatures,
)


@dataclass
class MultiBranchConfig:
    n_branches: int = 4
    perturbation_scale: float = 2.0
    trigger_dist_fraction: float = 2.0
    # Requiring non-convergence makes the trigger too restrictive in
    # warm-started flows where L-BFGS converges quickly even to
    # colliding optima. Set False by default; the "no flippable
    # signatures" guard in multi_branch_optimize prevents wasted work
    # in converge-style unsolvable scenarios.
    require_non_convergence: bool = False
    homotopy_penalty_weight: float = 5.0e2
    homotopy_epsilon: float = 0.3
    consistency_bonus: float = 0.15
    branch_maxiter: int = 10
    branch_ftol: float = 1.0e-4
    branch_maxls: int = 10


@dataclass
class MultiBranchResult:
    trajectory: Trajectory
    total_cost: float
    main_branch_used: bool
    n_branches_run: int
    signature: Tuple[int, ...]
    branch_costs: List[float] = field(default_factory=list)
    branch_signatures: List[Tuple[int, ...]] = field(default_factory=list)
    branch_min_dists: List[float] = field(default_factory=list)
    selected_branch_idx: int = 0
    trigger_reason: str = ""


def _sample_traj_on_grid(traj: Trajectory, ts: np.ndarray) -> np.ndarray:
    T = float(traj.total_time)
    out = np.zeros((len(ts), 3))
    for i, t in enumerate(ts):
        out[i] = traj.evaluate(min(max(0.0, float(t)), T), 0)
    return out


def _neighbours_for_signature(
    swarm_neighbours: Sequence,
    horizon: float,
    n_samples: int = 32,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    if not swarm_neighbours:
        return out
    ts = np.linspace(0.0, horizon, n_samples)
    for (nb_traj, t_offset) in swarm_neighbours:
        T_nb = float(nb_traj.total_time)
        ts_local = np.clip(ts + t_offset, 0.0, T_nb)
        xyz = np.stack([nb_traj.evaluate(float(t), 0) for t in ts_local], axis=0)
        out.append((xyz, ts.copy()))
    return out


def _min_predicted_swarm_distance(
    trajectory: Trajectory,
    swarm_neighbours: Sequence[Tuple[Trajectory, float]],
    n_samples: int = 16,
) -> float:
    if not swarm_neighbours:
        return float("inf")
    horizon = trajectory.total_time
    ts = np.linspace(0.0, horizon, n_samples)
    own_xyz = np.stack([trajectory.evaluate(float(t)) for t in ts], axis=0)
    min_d = float("inf")
    for (nb_traj, t_offset) in swarm_neighbours:
        nb_h = nb_traj.total_time
        nb_xyz = np.stack(
            [nb_traj.evaluate(min(max(float(t) + t_offset, 0.0), nb_h))
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
    cost = _evaluate_cost(trajectory, polytopes, gc_config)
    if swarm_neighbours and swarm_config is not None:
        from src.swarm.swarm_penalty import compute_swarm_cost_and_grad
        sc, _, _ = compute_swarm_cost_and_grad(
            trajectory, list(swarm_neighbours), swarm_config
        )
        cost += sc
    return float(cost)


def _perturb_for_target(
    initial_waypoints: np.ndarray,
    ctx: HomotopyPenaltyContext,
    scale: float,
) -> np.ndarray:
    """Generate an initial guess in the target homotopy class.

    For each interior waypoint k, sum nudges from each neighbour: if the
    target sign for neighbour j is +1, push along +lateral_j. The
    homotopy penalty then prevents L-BFGS from walking the perturbation
    back to the original minimum.
    """
    if initial_waypoints.shape[0] < 3:
        return initial_waypoints.copy()
    out = initial_waypoints.copy()
    n_int = out.shape[0] - 2
    sig = ctx.target_signature
    n_nbr = ctx.nbr_laterals_xy.shape[1] if ctx.nbr_laterals_xy.size > 0 else 0
    for k in range(n_int):
        nudge = np.zeros(3)
        for j in range(min(n_nbr, len(sig))):
            s_j = sig[j]
            if s_j == 0:
                continue
            lat = ctx.nbr_laterals_xy[k, j]
            nudge[0] += s_j * scale * lat[0]
            nudge[1] += s_j * scale * lat[1]
        out[k + 1] += nudge
    return out


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
    prev_signature: Optional[Tuple[int, ...]] = None,
) -> MultiBranchResult:
    bc = branch_config if branch_config is not None else MultiBranchConfig()

    # Step 1: Main warm-started branch (always runs)
    main_traj, main_meta = gcopter_optimize(
        initial_waypoints=initial_waypoints,
        initial_durations=initial_durations,
        bc_start=bc_start, bc_end=bc_end,
        polytopes=polytopes, config=config,
        swarm_neighbours=swarm_neighbours, swarm_config=swarm_config,
        warm_start=warm_start, return_meta=True,
    )
    main_cost = _total_cost_with_swarm(
        main_traj, polytopes, config, swarm_neighbours or [], swarm_config
    )
    main_min_dist = _min_predicted_swarm_distance(
        main_traj, swarm_neighbours or []
    )

    # Step 2: Compute main's signature
    nbr_samples = _neighbours_for_signature(
        swarm_neighbours or [], horizon=main_traj.total_time
    )
    if nbr_samples:
        own_ts = nbr_samples[0][1]
        own_xyz = _sample_traj_on_grid(main_traj, own_ts)
        main_signature = full_signature(
            own_xyz, own_ts, nbr_samples,
            interaction_radius=swarm_clearance_horizontal * 2.0,
        )
    else:
        main_signature = ()

    # Step 3: Adaptive trigger
    threshold = bc.trigger_dist_fraction * swarm_clearance_horizontal
    violation_likely = (
        bool(swarm_neighbours)
        and bc.n_branches > 0
        and main_min_dist < threshold
    )
    main_iters = int(main_meta.get("iters", 0))
    main_cap = int(main_meta.get("maxiter_used", config.maxiter))
    non_converged = main_iters >= main_cap
    if bc.require_non_convergence:
        needs_branches = violation_likely and non_converged
        trigger_reason = f"violation={violation_likely} non_conv={non_converged}"
    else:
        needs_branches = violation_likely
        trigger_reason = f"violation={violation_likely}"

    if not needs_branches:
        return MultiBranchResult(
            trajectory=main_traj,
            total_cost=main_cost,
            main_branch_used=True,
            n_branches_run=1,
            signature=main_signature,
            branch_costs=[main_cost],
            branch_signatures=[main_signature],
            branch_min_dists=[main_min_dist],
            selected_branch_idx=0,
            trigger_reason=f"no-trigger ({trigger_reason})",
        )

    # Step 4: Generate target signatures
    targets = generate_target_signatures(main_signature, bc.n_branches)
    if not targets:
        return MultiBranchResult(
            trajectory=main_traj,
            total_cost=main_cost,
            main_branch_used=True,
            n_branches_run=1,
            signature=main_signature,
            branch_costs=[main_cost],
            branch_signatures=[main_signature],
            branch_min_dists=[main_min_dist],
            selected_branch_idx=0,
            trigger_reason="triggered but no flippable signatures",
        )

    # Step 5: Run constrained branches
    interior_times = np.cumsum(initial_durations)[:-1]

    branch_gc_config = GCopterConfig(
        s=config.s,
        w_time=config.w_time,
        w_energy=config.w_energy,
        w_corridor=config.w_corridor,
        w_velocity=config.w_velocity,
        v_max=config.v_max,
        n_quad=config.n_quad,
        min_duration=config.min_duration,
        max_duration=config.max_duration,
        maxiter=bc.branch_maxiter,
        ftol=bc.branch_ftol,
        warm_start_skip_ratio=config.warm_start_skip_ratio,
        warm_start_relax_ratio=config.warm_start_relax_ratio,
        warm_start_maxiter=bc.branch_maxiter,
        warm_start_ftol=bc.branch_ftol,
        warm_start_maxls=bc.branch_maxls,
    )

    branch_costs: List[float] = [main_cost]
    branch_signatures: List[Tuple[int, ...]] = [main_signature]
    branch_min_dists: List[float] = [main_min_dist]
    branch_trajs: List[Trajectory] = [main_traj]

    for target_sig in targets:
        try:
            ctx = build_penalty_context(
                interior_waypoint_times=interior_times,
                neighbours=nbr_samples,
                target_signature=target_sig,
                weight=bc.homotopy_penalty_weight,
                epsilon=bc.homotopy_epsilon,
            )
        except Exception:
            continue

        perturbed_wps = _perturb_for_target(
            initial_waypoints, ctx, bc.perturbation_scale
        )

        try:
            cand_traj, _meta = gcopter_optimize(
                initial_waypoints=perturbed_wps,
                initial_durations=initial_durations.copy(),
                bc_start=bc_start, bc_end=bc_end,
                polytopes=polytopes, config=branch_gc_config,
                swarm_neighbours=swarm_neighbours,
                swarm_config=swarm_config,
                warm_start=True,
                return_meta=True,
                homotopy_context=ctx,
            )
        except Exception:
            branch_costs.append(float("inf"))
            branch_signatures.append(target_sig)
            branch_min_dists.append(float("-inf"))
            branch_trajs.append(main_traj)
            continue

        cand_xyz = _sample_traj_on_grid(cand_traj, nbr_samples[0][1])
        cand_sig = full_signature(
            cand_xyz, nbr_samples[0][1], nbr_samples,
            interaction_radius=swarm_clearance_horizontal * 2.0,
        )
        cand_cost = _total_cost_with_swarm(
            cand_traj, polytopes, branch_gc_config,
            swarm_neighbours or [], swarm_config,
        )
        cand_min_d = _min_predicted_swarm_distance(
            cand_traj, swarm_neighbours or []
        )
        branch_costs.append(cand_cost)
        branch_signatures.append(cand_sig)
        branch_min_dists.append(cand_min_d)
        branch_trajs.append(cand_traj)

    # Step 6: Select with consistency bonus
    best_idx = 0
    best_score = float("inf")
    best_safe = False
    for i, (cost, sig, md) in enumerate(zip(
        branch_costs, branch_signatures, branch_min_dists
    )):
        if not np.isfinite(cost):
            continue
        safe = md >= swarm_clearance_horizontal
        adj_cost = cost
        if prev_signature is not None and sig == prev_signature:
            adj_cost *= (1.0 - bc.consistency_bonus)
        if safe and not best_safe:
            best_idx, best_score, best_safe = i, adj_cost, True
        elif safe == best_safe and adj_cost < best_score:
            best_idx, best_score, best_safe = i, adj_cost, safe

    return MultiBranchResult(
        trajectory=branch_trajs[best_idx],
        total_cost=branch_costs[best_idx],
        main_branch_used=(best_idx == 0),
        n_branches_run=len(branch_trajs),
        signature=branch_signatures[best_idx],
        branch_costs=branch_costs,
        branch_signatures=branch_signatures,
        branch_min_dists=branch_min_dists,
        selected_branch_idx=best_idx,
        trigger_reason=f"triggered ({trigger_reason})",
    )
