"""Goal-region right-of-way arbitration for converging-goal scenarios.

Background
==========
The MINCO + FIRI + swarm-penalty + CBF + MGR stack is a *traffic*
solver: drones deflect away from neighbours' predicted paths. That
works when drones are crossing each other's paths, but it cannot
resolve a *queue* — N drones converging on the same goal point have
no deflection geometry available; the goal is the goal, and only
sequencing them ("you go first, others hold") fixes it.

This module adds a right-of-way layer above MINCO: when ≥2 drones
have goals within `proximity_radius_m` of each other, they enter a
goal cluster. The cluster gets N approach slots — slot 0 is the
shared goal, slots 1..N-1 are holding positions on rings of
increasing radius around it. Each drone bids on each slot:

    bid(drone, slot) = w_mission   · drone.mission_priority
                     - w_distance  · ||drone.position - slot.position||
                     - w_id_tiebreak · (drone.id / max_id)

A standard assignment problem solver (Hungarian) maps each drone to
exactly one slot, maximising total bid. The drone with slot 0
proceeds toward the goal at normal speed; drones with slot k>0 have
their effective goal overridden to the holding position and they
loiter there (still using MINCO / MGR / CBF for collision avoidance).

When the slot-0 drone arrives at the goal (within
`goal_arrived_radius_m`), it is marked *satisfied* and removed from
the cluster; the next auction round reassigns and a new winner gets
slot 0.

Determinism & decentralisation
==============================
The Hungarian solver is deterministic given a bid matrix. As long as
every drone observes the same broadcasts and produces the same bid
matrix, every drone arrives at the same assignment with no consensus
rounds needed (this is what CBBA would compute in the no-bundle
limit anyway — implementation-wise Hungarian is simpler).

In the rig simulation we run it centrally on the full drone list.
In deployment each drone broadcasts its bid vector (~64 bytes) and
runs Hungarian locally on the received matrix; the network protocol
detail is out of scope here.

Mission priority semantics
==========================
`mission_priority ∈ [0, 1]`:
- 0.0  training / low-stake patrol
- 0.5  normal patrol  (default)
- 1.0  active threat / medevac / pursuit

`w_mission` is sized so a 0.1 priority gap dominates a 1 m distance
gap. Tweak in tests if scenario calls for it.

Audit trail
===========
Police-context requirement: every slot assignment must be traceable
to a specific bid. `arbitrate()` returns an `ArbitrationResult` with
the full bid matrix, slot positions, and assignments — the caller is
expected to log this for post-hoc review (the GCS layer wires this
into the audit channel).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# Audit event type emitted to the GCS audit channel. One event per
# cluster per arbitration round. Matches the free-string convention
# used elsewhere in GCSServer (see `src/gcs/gcs_server.py:emit_audit`).
AUDIT_EVENT_TYPE = "goal_arbitration"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoalArbitrationConfig:
    """Tuning knobs for the right-of-way auction.

    Attributes
    ----------
    proximity_radius_m : float
        Two drones share a cluster if their goals are within this
        radius (Euclidean, 3D). For exact-shared-goal scenarios
        (converge_dense) any small positive value works; for fuzzy
        operational goals (patrol waypoints with ε noise) bump it up.
    holding_ring_radii_m : tuple[float, ...]
        Radii (m) of holding rings around the goal centroid. Slot 1
        sits on radii[0], slot 2 on radii[1], etc. The number of
        radii must be ≥ (max_cluster_size - 1).
    holding_ring_phase_rad : float
        Angular offset (rad) applied to the first holding position on
        each ring. Lets the operator stagger rings if downwash
        becomes a concern; default 0 places slot 1 along +x.
    goal_arrived_radius_m : float
        A drone is "satisfied" (and exits the cluster) when its
        position is within this radius of its assigned slot-0 goal.
    w_mission : float
        Weight on `mission_priority` term in the bid. Default sized
        so a 0.1 priority gap = 10 m of distance gap, i.e. a slightly
        higher-priority drone outbids a same-distance lower-priority
        drone by enough margin to survive ID tiebreaks.
    w_distance : float
        Weight on `||drone - slot||` term (negative — closer is
        better). Default 1.0 → 1 m extra distance = 1 unit bid loss.
    w_id_tiebreak : float
        Weight on the deterministic ID-based tiebreaker. Must be
        small enough to never dominate a real priority or distance
        gap; 0.001 means at most 0.001 bid swing across all drones.
    """

    proximity_radius_m: float = 0.5
    # Holding rings live OUTSIDE the typical operating radius so they
    # don't cross the slot-0 winner's approach corridor. The previous
    # tight defaults (3, 5, 7 m) put holding drones directly on the
    # line between a start position at radius 8 and the goal at origin,
    # so the winner's MINCO trajectory passed straight through a peer's
    # holding slot — MGR fired, no one arrived. Default radii now sit
    # at or beyond a typical Rig 2 start radius.
    holding_ring_radii_m: Tuple[float, ...] = (12.0, 14.0, 16.0, 18.0, 20.0)
    holding_ring_phase_rad: float = 0.0
    goal_arrived_radius_m: float = 1.5
    w_mission: float = 100.0
    w_distance: float = 1.0
    w_id_tiebreak: float = 1.0e-3


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DroneBidState:
    """Per-drone state needed to compute bids.

    All position vectors are 3-D ndarrays in the rig frame.
    """

    drone_id: int
    position: np.ndarray
    goal: np.ndarray
    mission_priority: float = 0.5
    satisfied: bool = False


@dataclass(frozen=True)
class ApproachSlot:
    """One slot in a goal cluster.

    `slot_id == 0` is always the goal itself; `slot_id > 0` are
    holding ring positions in order of increasing radius.
    """

    slot_id: int
    position: np.ndarray
    is_goal: bool


@dataclass
class ArbitrationResult:
    """Output of one `arbitrate()` call.

    Attributes
    ----------
    effective_goals : dict[int, np.ndarray]
        Maps drone_id → the goal MINCO should plan toward this tick.
        Drones not in any cluster get their original goal; clustered
        drones get their assigned slot position.
    assignments : dict[int, Optional[int]]
        Maps drone_id → assigned slot_id (0 means "go to goal", k>0
        means "hold at ring slot k", None means "not in any cluster").
    clusters : list[list[int]]
        For diagnostics — each entry is the list of drone_ids in one
        goal cluster. Disjoint.
    slots_per_cluster : dict[int, list[ApproachSlot]]
        For diagnostics — slot list per cluster, indexed by the
        cluster's lowest drone_id.
    bid_matrices : dict[int, np.ndarray]
        For audit — per cluster (indexed by lowest drone_id), a
        (N_drones × N_slots) array of bids. Caller logs this.
    """

    effective_goals: Dict[int, np.ndarray]
    assignments: Dict[int, Optional[int]]
    clusters: List[List[int]] = field(default_factory=list)
    slots_per_cluster: Dict[int, List[ApproachSlot]] = field(default_factory=dict)
    bid_matrices: Dict[int, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def detect_shared_goal_clusters(
    drones: Sequence[DroneBidState],
    proximity_radius_m: float,
) -> List[List[int]]:
    """Cluster drone IDs whose goals are within `proximity_radius_m`.

    Uses union-find on a goal-distance graph. Satisfied drones are
    excluded from clustering entirely (they've already won, they're
    not bidding for anything).

    Returns clusters of size ≥ 2 only — a single drone heading to a
    unique goal has no one to arbitrate with and short-circuits to
    its original goal. Clusters are returned in deterministic order
    (lowest member ID first; clusters sorted by their lowest member).
    """
    active = [d for d in drones if not d.satisfied]
    n = len(active)
    if n < 2:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    r2 = float(proximity_radius_m) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            d2 = float(np.sum((active[i].goal - active[j].goal) ** 2))
            if d2 <= r2:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(active[i].drone_id)

    clusters = [sorted(g) for g in groups.values() if len(g) >= 2]
    clusters.sort(key=lambda c: c[0])
    return clusters


def generate_approach_slots(
    cluster_drones: Sequence[DroneBidState],
    config: GoalArbitrationConfig,
) -> List[ApproachSlot]:
    """Produce one slot per drone in the cluster: slot 0 = goal,
    slot 1..N-1 = holding ring positions of increasing radius.

    Goal centroid is the mean of the cluster's drone goals (handles
    fuzzy clusters; for exact-shared-goal scenarios this equals the
    common goal). Holding rings have positions arranged evenly around
    the centroid in the xy plane at the goal's altitude.
    """
    n = len(cluster_drones)
    if n == 0:
        return []

    goal_centroid = np.mean(
        np.stack([d.goal for d in cluster_drones]), axis=0
    )

    slots: List[ApproachSlot] = [
        ApproachSlot(slot_id=0, position=goal_centroid.copy(), is_goal=True)
    ]

    radii = list(config.holding_ring_radii_m)
    if len(radii) < n - 1:
        # Extrapolate linearly with the same spacing.
        last = radii[-1] if radii else 3.0
        spacing = radii[-1] - radii[-2] if len(radii) >= 2 else 2.0
        while len(radii) < n - 1:
            last += spacing
            radii.append(last)

    # Holding positions: one per ring (1 drone per ring keeps spacing
    # generous). Phase per cluster is fixed so the assignment is
    # deterministic regardless of cluster traversal order.
    phase = float(config.holding_ring_phase_rad)
    for k in range(1, n):
        r = radii[k - 1]
        # Each ring slot offset by k * golden angle so they don't
        # stack on a line in xy.
        angle = phase + (k - 1) * 2.39996  # ≈ golden angle in radians
        offset = np.array(
            [r * np.cos(angle), r * np.sin(angle), 0.0], dtype=np.float64
        )
        slots.append(
            ApproachSlot(
                slot_id=k, position=goal_centroid + offset, is_goal=False
            )
        )

    return slots


def compute_bid_matrix(
    cluster_drones: Sequence[DroneBidState],
    slots: Sequence[ApproachSlot],
    config: GoalArbitrationConfig,
    max_drone_id: int,
) -> np.ndarray:
    """Compute the N×N bid matrix for one cluster.

    bid[i, j] = (w_mission · priority_i  if slot j is the goal else 0)
              - w_distance · ||pos_i - slot_j||
              - w_id_tiebreak · (id_i / max_id)

    The priority term is a **goal-slot bonus**, not a universal scale.
    If it applied to every slot, a high-priority drone could win a
    nearby *holding* ring instead of the goal — defeating the point.
    Restricting the bonus to slot 0 means "priority = how much you
    want the goal specifically," which is the operational semantic
    we want (medevac drone outbids patrol drone for the goal; both
    are equally fine taking holding rings).

    Higher is better. `max_drone_id` is used to normalise the
    tiebreaker; passing 0 disables it.
    """
    n = len(cluster_drones)
    m = len(slots)
    M = np.zeros((n, m), dtype=np.float64)
    norm = float(max_drone_id) if max_drone_id > 0 else 1.0
    for i, d in enumerate(cluster_drones):
        id_term = config.w_id_tiebreak * (float(d.drone_id) / norm)
        for j, s in enumerate(slots):
            dist = float(np.linalg.norm(d.position - s.position))
            priority_term = (
                config.w_mission * float(d.mission_priority)
                if s.is_goal else 0.0
            )
            M[i, j] = priority_term - config.w_distance * dist - id_term
    return M


def assign_slots(bid_matrix: np.ndarray) -> np.ndarray:
    """Two-stage assignment: deterministic slot-0 auction, then
    Hungarian on the remaining holding rings.

    Stage 1 — slot 0 (the goal). Whoever has the highest bid for
    column 0 wins it. This is the only stage that matters for the
    audit narrative ("drone X had the highest goal-slot bid, here
    is the broadcast log"), so we resolve it explicitly rather than
    delegating to a global optimiser. Ties are already broken inside
    the bid via `w_id_tiebreak`.

    Stage 2 — holding rings. The remaining N-1 drones map to the
    remaining N-1 slots via `scipy.optimize.linear_sum_assignment`
    on the sub-matrix. Holding-slot assignment is operationally
    fungible ("you're in a ring, you loiter, you wait for re-auction")
    so global optimality is fine — it just minimises total holding
    travel, which is a free win.

    Returns a length-N int array `assign` where `assign[i] = slot_id`
    for the i-th cluster drone. Bid matrix must be square.
    """
    if bid_matrix.size == 0:
        return np.empty(0, dtype=np.int64)

    n = bid_matrix.shape[0]
    assert bid_matrix.shape == (n, n)

    # Stage 1: slot-0 winner is argmax of column 0.
    winner_idx = int(np.argmax(bid_matrix[:, 0]))

    result = np.full(n, -1, dtype=np.int64)
    result[winner_idx] = 0

    if n == 1:
        return result

    # Stage 2: Hungarian on remaining (N-1) drones × (N-1) slots.
    other_drones = np.array(
        [i for i in range(n) if i != winner_idx], dtype=np.int64
    )
    other_slots = np.arange(1, n, dtype=np.int64)
    sub_M = bid_matrix[np.ix_(other_drones, other_slots)]
    row_ind, col_ind = linear_sum_assignment(-sub_M)
    assert np.array_equal(row_ind, np.arange(sub_M.shape[0]))
    for i, drone_idx in enumerate(other_drones):
        result[drone_idx] = int(other_slots[col_ind[i]])
    return result


def arbitrate(
    drones: Sequence[DroneBidState],
    config: GoalArbitrationConfig,
) -> ArbitrationResult:
    """Top-level: cluster, slot, bid, assign. Returns the per-drone
    effective goal plus full audit data.

    Drones marked `satisfied=True` keep their current goal and do
    not enter any cluster. Single-drone (no shared) goals pass
    through unchanged.
    """
    # Default: everyone gets their original goal.
    effective_goals: Dict[int, np.ndarray] = {
        d.drone_id: d.goal.copy() for d in drones
    }
    assignments: Dict[int, Optional[int]] = {d.drone_id: None for d in drones}

    clusters = detect_shared_goal_clusters(drones, config.proximity_radius_m)
    slots_per_cluster: Dict[int, List[ApproachSlot]] = {}
    bid_matrices: Dict[int, np.ndarray] = {}

    if not clusters:
        return ArbitrationResult(
            effective_goals=effective_goals,
            assignments=assignments,
            clusters=clusters,
            slots_per_cluster=slots_per_cluster,
            bid_matrices=bid_matrices,
        )

    by_id = {d.drone_id: d for d in drones}
    max_drone_id = max((d.drone_id for d in drones), default=0)

    for cluster_ids in clusters:
        cluster_drones = [by_id[i] for i in cluster_ids]
        slots = generate_approach_slots(cluster_drones, config)
        M = compute_bid_matrix(cluster_drones, slots, config, max_drone_id)
        assign = assign_slots(M)

        cluster_key = cluster_ids[0]
        slots_per_cluster[cluster_key] = slots
        bid_matrices[cluster_key] = M

        for idx, drone in enumerate(cluster_drones):
            slot_id = int(assign[idx])
            assignments[drone.drone_id] = slot_id
            effective_goals[drone.drone_id] = slots[slot_id].position.copy()

    return ArbitrationResult(
        effective_goals=effective_goals,
        assignments=assignments,
        clusters=clusters,
        slots_per_cluster=slots_per_cluster,
        bid_matrices=bid_matrices,
    )


def has_arrived(
    drone_position: np.ndarray,
    target: np.ndarray,
    goal_arrived_radius_m: float,
) -> bool:
    """Return True if `drone_position` is within
    `goal_arrived_radius_m` of `target`. Used by the rig to flip
    `satisfied=True` once a slot-0 winner reaches the goal."""
    d2 = float(np.sum((np.asarray(drone_position) - np.asarray(target)) ** 2))
    return d2 <= float(goal_arrived_radius_m) ** 2


# ---------------------------------------------------------------------------
# Audit serialisation — police-context post-hoc review
# ---------------------------------------------------------------------------


def format_arbitration_audit(
    t_now: float,
    result: ArbitrationResult,
) -> List[Tuple[str, str]]:
    """Serialise an `ArbitrationResult` into one audit event per
    cluster, ready to hand to `GCSServer.emit_audit(event_type, detail)`.

    Each entry is `(event_type, detail_json_str)` where `event_type` is
    the constant `AUDIT_EVENT_TYPE` ("goal_arbitration") and
    `detail_json_str` is a JSON object with this schema:

        {
            "t": float,                  # rig-global time of the round
            "cluster": [int, ...],       # drone_ids in this cluster
            "winner": int,               # drone_id assigned slot 0
            "assignments": {             # drone_id (str) → slot_id
                "0": 0, "1": 4, ...
            },
            "slots": [                   # one entry per slot, in order
                {"id": int, "pos": [x, y, z], "is_goal": bool},
                ...
            ],
            "bids": [[float, ...], ...]  # N×N row-major bid matrix
        }

    Police-context requirement: every slot-0 assignment must be
    defensible. The full bid matrix + slot positions in the JSON lets
    a reviewer recompute the assignment offline and verify it matches
    the broadcast — no hidden state, no opaque optimiser.

    Empty result (no clusters) returns an empty list — no audit
    spam when nothing is being arbitrated.
    """
    out: List[Tuple[str, str]] = []
    for cluster_ids in result.clusters:
        cluster_key = cluster_ids[0]
        slots = result.slots_per_cluster.get(cluster_key, [])
        bids = result.bid_matrices.get(cluster_key)
        # Find the slot-0 winner inside this cluster.
        winner = next(
            (
                drone_id for drone_id in cluster_ids
                if result.assignments.get(drone_id) == 0
            ),
            -1,
        )
        detail = {
            "t": float(t_now),
            "cluster": [int(d) for d in cluster_ids],
            "winner": int(winner),
            "assignments": {
                str(d): int(result.assignments[d])
                for d in cluster_ids
                if result.assignments.get(d) is not None
            },
            "slots": [
                {
                    "id": int(s.slot_id),
                    "pos": [
                        float(s.position[0]),
                        float(s.position[1]),
                        float(s.position[2]),
                    ],
                    "is_goal": bool(s.is_goal),
                }
                for s in slots
            ],
            "bids": (
                [[float(v) for v in row] for row in bids]
                if bids is not None else []
            ),
        }
        out.append((AUDIT_EVENT_TYPE, json.dumps(detail, separators=(",", ":"))))
    return out
