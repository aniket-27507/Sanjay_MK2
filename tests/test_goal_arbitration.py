"""Tests for src.swarm.goal_arbitration — right-of-way auction for
converging-goal scenarios.

Coverage:
  - Cluster detection: shared, separate, fuzzy proximity, satisfied
    drones excluded
  - Slot generation: slot 0 at goal centroid, rings at correct radii
  - Bid computation: priority dominates, distance penalises, id
    tiebreak is small but deterministic
  - Assignment: square matrix, optimal under Hungarian, no double-
    booking, slot 0 goes to highest bidder
  - Top-level arbitrate(): single-drone passthrough, mixed clusters,
    satisfied drones keep original goal
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.swarm.goal_arbitration import (
    AUDIT_EVENT_TYPE,
    ApproachSlot,
    ArbitrationResult,
    DroneBidState,
    GoalArbitrationConfig,
    arbitrate,
    assign_slots,
    compute_bid_matrix,
    detect_shared_goal_clusters,
    format_arbitration_audit,
    generate_approach_slots,
    has_arrived,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bid(
    drone_id: int,
    position=(0.0, 0.0, 5.0),
    goal=(0.0, 0.0, 5.0),
    mission_priority: float = 0.5,
    satisfied: bool = False,
) -> DroneBidState:
    return DroneBidState(
        drone_id=drone_id,
        position=np.asarray(position, dtype=np.float64),
        goal=np.asarray(goal, dtype=np.float64),
        mission_priority=mission_priority,
        satisfied=satisfied,
    )


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------


class TestClusterDetection:
    def test_no_drones_returns_empty(self) -> None:
        assert detect_shared_goal_clusters([], 1.0) == []

    def test_single_drone_returns_empty(self) -> None:
        # No one to arbitrate with.
        assert detect_shared_goal_clusters([_bid(0)], 1.0) == []

    def test_shared_goal_clusters_n_drones(self) -> None:
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(6)]
        clusters = detect_shared_goal_clusters(ds, 0.5)
        assert clusters == [[0, 1, 2, 3, 4, 5]]

    def test_two_separate_goals_form_two_clusters(self) -> None:
        ds = [
            _bid(0, goal=(0, 0, 5)),
            _bid(1, goal=(0, 0, 5)),
            _bid(2, goal=(100, 100, 5)),
            _bid(3, goal=(100, 100, 5)),
        ]
        clusters = detect_shared_goal_clusters(ds, 0.5)
        assert clusters == [[0, 1], [2, 3]]

    def test_lone_drone_with_unique_goal_excluded(self) -> None:
        ds = [
            _bid(0, goal=(0, 0, 5)),
            _bid(1, goal=(0, 0, 5)),
            _bid(2, goal=(100, 100, 5)),
        ]
        clusters = detect_shared_goal_clusters(ds, 0.5)
        # drone 2's goal is unique → not in any cluster.
        assert clusters == [[0, 1]]

    def test_fuzzy_proximity_groups_nearby_goals(self) -> None:
        ds = [
            _bid(0, goal=(0.0, 0.0, 5)),
            _bid(1, goal=(0.3, 0.0, 5)),  # within 0.5
            _bid(2, goal=(10.0, 0.0, 5)),  # far
        ]
        clusters = detect_shared_goal_clusters(ds, 0.5)
        assert clusters == [[0, 1]]

    def test_satisfied_drones_excluded(self) -> None:
        ds = [
            _bid(0, goal=(0, 0, 5), satisfied=True),
            _bid(1, goal=(0, 0, 5)),
            _bid(2, goal=(0, 0, 5)),
        ]
        clusters = detect_shared_goal_clusters(ds, 0.5)
        assert clusters == [[1, 2]]

    def test_transitive_clustering(self) -> None:
        # A-B within radius, B-C within radius, A-C just beyond. Union-
        # find should still merge all three.
        ds = [
            _bid(0, goal=(0.0, 0.0, 5)),
            _bid(1, goal=(0.4, 0.0, 5)),
            _bid(2, goal=(0.8, 0.0, 5)),
        ]
        clusters = detect_shared_goal_clusters(ds, 0.5)
        assert clusters == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# Slot generation
# ---------------------------------------------------------------------------


class TestSlotGeneration:
    def test_slot_count_matches_cluster_size(self) -> None:
        cfg = GoalArbitrationConfig()
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(4)]
        slots = generate_approach_slots(ds, cfg)
        assert len(slots) == 4

    def test_slot_zero_is_goal_centroid(self) -> None:
        cfg = GoalArbitrationConfig()
        ds = [
            _bid(0, goal=(0, 0, 5)),
            _bid(1, goal=(2, 0, 5)),  # centroid x = 1
        ]
        slots = generate_approach_slots(ds, cfg)
        assert slots[0].slot_id == 0
        assert slots[0].is_goal is True
        np.testing.assert_allclose(slots[0].position, [1.0, 0.0, 5.0])

    def test_holding_rings_at_correct_radii(self) -> None:
        cfg = GoalArbitrationConfig(
            holding_ring_radii_m=(3.0, 5.0, 7.0),
            holding_ring_phase_rad=0.0,
        )
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(4)]
        slots = generate_approach_slots(ds, cfg)
        # Slot 1 on ring r=3, slot 2 on r=5, slot 3 on r=7.
        for k in (1, 2, 3):
            xy_dist = np.linalg.norm(slots[k].position[:2])
            assert xy_dist == pytest.approx(cfg.holding_ring_radii_m[k - 1])
            assert slots[k].position[2] == pytest.approx(5.0)

    def test_holding_rings_share_goal_altitude(self) -> None:
        cfg = GoalArbitrationConfig()
        ds = [_bid(i, goal=(0, 0, 7.5)) for i in range(3)]
        slots = generate_approach_slots(ds, cfg)
        for s in slots:
            assert s.position[2] == pytest.approx(7.5)

    def test_radii_extrapolate_if_too_few(self) -> None:
        # 8 drones, but only 3 ring radii configured → extrapolate.
        cfg = GoalArbitrationConfig(holding_ring_radii_m=(3.0, 5.0, 7.0))
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(8)]
        slots = generate_approach_slots(ds, cfg)
        # Should not crash; outer rings at 9, 11, 13, 15 (spacing 2).
        radii = [np.linalg.norm(slots[k].position[:2]) for k in range(1, 8)]
        assert radii == pytest.approx([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])

    def test_holding_slots_not_collinear(self) -> None:
        # Golden-angle phasing should give distinct angular positions.
        cfg = GoalArbitrationConfig()
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(4)]
        slots = generate_approach_slots(ds, cfg)
        angles = [
            np.arctan2(s.position[1], s.position[0]) for s in slots[1:]
        ]
        # No two angles equal within 0.1 rad.
        for i, a in enumerate(angles):
            for b in angles[i + 1:]:
                assert abs(((a - b + np.pi) % (2 * np.pi)) - np.pi) > 0.1


# ---------------------------------------------------------------------------
# Bid computation
# ---------------------------------------------------------------------------


class TestBidComputation:
    def test_higher_priority_outbids_lower_for_slot_0(self) -> None:
        cfg = GoalArbitrationConfig()
        ds = [
            _bid(0, position=(2, 0, 5), goal=(0, 0, 5), mission_priority=0.5),
            _bid(1, position=(2, 0, 5), goal=(0, 0, 5), mission_priority=1.0),
        ]
        slots = generate_approach_slots(ds, cfg)
        M = compute_bid_matrix(ds, slots, cfg, max_drone_id=1)
        # Drone 1 (priority 1.0) > Drone 0 (priority 0.5) on slot 0.
        assert M[1, 0] > M[0, 0]

    def test_closer_outbids_farther_at_equal_priority(self) -> None:
        cfg = GoalArbitrationConfig()
        ds = [
            _bid(0, position=(10, 0, 5), goal=(0, 0, 5)),  # 10 m away
            _bid(1, position=(2, 0, 5), goal=(0, 0, 5)),  # 2 m away
        ]
        slots = generate_approach_slots(ds, cfg)
        M = compute_bid_matrix(ds, slots, cfg, max_drone_id=1)
        # Same priority, drone 1 is closer → outbids on slot 0.
        assert M[1, 0] > M[0, 0]

    def test_id_tiebreak_is_tiny_but_deterministic(self) -> None:
        cfg = GoalArbitrationConfig()
        # Two identical drones except for id.
        ds = [
            _bid(0, position=(1, 0, 5), goal=(0, 0, 5)),
            _bid(1, position=(1, 0, 5), goal=(0, 0, 5)),
        ]
        slots = generate_approach_slots(ds, cfg)
        M = compute_bid_matrix(ds, slots, cfg, max_drone_id=1)
        # Lower-id drone has slightly higher bid (id_term subtracted is
        # smaller).
        assert M[0, 0] > M[1, 0]
        # ...but the gap is tiny (within w_id_tiebreak).
        assert M[0, 0] - M[1, 0] <= cfg.w_id_tiebreak + 1e-9

    def test_priority_gap_dominates_distance_gap(self) -> None:
        # At default w_mission=100, a 0.1 priority gap = 10 units; far
        # more than a 1 m distance gap (1 unit).
        cfg = GoalArbitrationConfig()
        ds = [
            _bid(
                0, position=(0, 0, 5), goal=(0, 0, 5),
                mission_priority=0.5,
            ),  # at goal
            _bid(
                1, position=(10, 0, 5), goal=(0, 0, 5),
                mission_priority=0.7,
            ),  # 10 m away but +0.2 priority
        ]
        slots = generate_approach_slots(ds, cfg)
        M = compute_bid_matrix(ds, slots, cfg, max_drone_id=1)
        # Drone 1 wins on slot 0 because 0.2 priority gap (20 units) >
        # 10 m distance gap (10 units).
        assert M[1, 0] > M[0, 0]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


class TestAssignment:
    def test_each_drone_gets_unique_slot(self) -> None:
        # 4 drones, 4 slots — must be a permutation.
        np.random.seed(0)
        M = np.random.rand(4, 4)
        assign = assign_slots(M)
        assert sorted(assign.tolist()) == [0, 1, 2, 3]

    def test_highest_bidder_for_slot_0_wins_slot_0(self) -> None:
        # Construct a matrix where drone 2 has the clear max for slot
        # 0, and others are penalised for slot 0.
        M = np.array(
            [
                [0.0, 10.0, 5.0, 1.0],
                [0.0, 5.0, 10.0, 1.0],
                [100.0, 1.0, 1.0, 1.0],  # drone 2 dominates slot 0
                [0.0, 1.0, 1.0, 10.0],
            ]
        )
        assign = assign_slots(M)
        # Drone 2 → slot 0.
        assert assign[2] == 0

    def test_empty_matrix_handled(self) -> None:
        assign = assign_slots(np.empty((0, 0)))
        assert assign.size == 0

    def test_deterministic_across_calls(self) -> None:
        np.random.seed(7)
        M = np.random.rand(5, 5)
        a1 = assign_slots(M)
        a2 = assign_slots(M)
        np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# Top-level arbitrate()
# ---------------------------------------------------------------------------


class TestArbitrate:
    def test_no_drones_returns_empty(self) -> None:
        out = arbitrate([], GoalArbitrationConfig())
        assert out.effective_goals == {}
        assert out.assignments == {}
        assert out.clusters == []

    def test_single_drone_passthrough(self) -> None:
        ds = [_bid(0, goal=(5, 5, 5))]
        out = arbitrate(ds, GoalArbitrationConfig())
        np.testing.assert_array_equal(out.effective_goals[0], [5, 5, 5])
        assert out.assignments[0] is None
        assert out.clusters == []

    def test_unique_goals_passthrough(self) -> None:
        ds = [
            _bid(0, goal=(0, 0, 5)),
            _bid(1, goal=(100, 0, 5)),
            _bid(2, goal=(0, 100, 5)),
        ]
        out = arbitrate(ds, GoalArbitrationConfig())
        for d in ds:
            np.testing.assert_array_equal(out.effective_goals[d.drone_id], d.goal)
            assert out.assignments[d.drone_id] is None

    def test_6_drone_shared_goal_assigns_distinct_slots(self) -> None:
        # Symmetric ring around the goal, equal priority.
        ds = []
        r = 8.0
        for i in range(6):
            theta = 2.0 * np.pi * i / 6
            ds.append(
                _bid(
                    i,
                    position=(r * np.cos(theta), r * np.sin(theta), 5.0),
                    goal=(0, 0, 5),
                    mission_priority=0.5,
                )
            )
        out = arbitrate(ds, GoalArbitrationConfig())
        assert out.clusters == [[0, 1, 2, 3, 4, 5]]
        # Every drone gets a slot, all slots are distinct.
        slot_ids = [out.assignments[i] for i in range(6)]
        assert sorted(slot_ids) == [0, 1, 2, 3, 4, 5]
        # Slot 0 went to the lowest drone_id (ID tiebreak wins ties).
        assert out.assignments[0] == 0

    def test_priority_wins_over_distance(self) -> None:
        # 3 drones, all heading to same goal. Drone 2 is farthest but
        # has highest priority → should still get slot 0.
        ds = [
            _bid(0, position=(1, 0, 5), goal=(0, 0, 5), mission_priority=0.5),
            _bid(1, position=(2, 0, 5), goal=(0, 0, 5), mission_priority=0.5),
            _bid(2, position=(20, 0, 5), goal=(0, 0, 5), mission_priority=1.0),
        ]
        out = arbitrate(ds, GoalArbitrationConfig())
        assert out.assignments[2] == 0  # priority dominates 20 m distance

    def test_satisfied_drone_keeps_original_goal(self) -> None:
        ds = [
            _bid(0, goal=(0, 0, 5), satisfied=True),
            _bid(1, goal=(0, 0, 5)),
            _bid(2, goal=(0, 0, 5)),
        ]
        out = arbitrate(ds, GoalArbitrationConfig())
        # Drone 0 is satisfied → not in any cluster, keeps original goal.
        np.testing.assert_array_equal(out.effective_goals[0], [0, 0, 5])
        assert out.assignments[0] is None
        # Drones 1, 2 form a cluster.
        assert out.clusters == [[1, 2]]

    def test_two_separate_clusters_arbitrated_independently(self) -> None:
        ds = [
            _bid(0, position=(1, 0, 5), goal=(0, 0, 5)),
            _bid(1, position=(2, 0, 5), goal=(0, 0, 5)),
            _bid(2, position=(101, 0, 5), goal=(100, 0, 5)),
            _bid(3, position=(102, 0, 5), goal=(100, 0, 5)),
        ]
        out = arbitrate(ds, GoalArbitrationConfig())
        assert out.clusters == [[0, 1], [2, 3]]
        # Each cluster assigns slots 0 and 1.
        assert sorted([out.assignments[0], out.assignments[1]]) == [0, 1]
        assert sorted([out.assignments[2], out.assignments[3]]) == [0, 1]

    def test_audit_data_populated(self) -> None:
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(3)]
        out = arbitrate(ds, GoalArbitrationConfig())
        # Cluster key is min(drone_ids) in that cluster.
        assert 0 in out.slots_per_cluster
        assert 0 in out.bid_matrices
        assert out.bid_matrices[0].shape == (3, 3)
        assert len(out.slots_per_cluster[0]) == 3

    def test_effective_goal_matches_assigned_slot(self) -> None:
        ds = [_bid(i, position=(i * 2.0, 0, 5), goal=(0, 0, 5)) for i in range(3)]
        out = arbitrate(ds, GoalArbitrationConfig())
        for drone_id, slot_id in out.assignments.items():
            slot_pos = out.slots_per_cluster[0][slot_id].position
            np.testing.assert_array_equal(out.effective_goals[drone_id], slot_pos)


# ---------------------------------------------------------------------------
# has_arrived
# ---------------------------------------------------------------------------


class TestHasArrived:
    def test_at_target_is_arrived(self) -> None:
        assert has_arrived(np.array([0, 0, 5]), np.array([0, 0, 5]), 1.0)

    def test_within_radius_is_arrived(self) -> None:
        assert has_arrived(np.array([0.5, 0, 5]), np.array([0, 0, 5]), 1.0)

    def test_just_outside_radius_is_not_arrived(self) -> None:
        assert not has_arrived(
            np.array([1.5, 0, 5]), np.array([0, 0, 5]), 1.0
        )

    def test_3d_distance(self) -> None:
        # Vertical offset counts too.
        assert not has_arrived(
            np.array([0, 0, 7]), np.array([0, 0, 5]), 1.0
        )


# ---------------------------------------------------------------------------
# Audit serialisation — police-context post-hoc review
# ---------------------------------------------------------------------------


class TestAuditFormatter:
    """`format_arbitration_audit` must produce events compatible with
    `GCSServer.emit_audit(event_type: str, detail: str)`. The detail
    string is a JSON object that a reviewer can use to recompute the
    assignment offline and verify it matches the broadcast — police
    audit context demands no hidden state."""

    def test_no_clusters_emits_no_events(self) -> None:
        ds = [_bid(0, goal=(0, 0, 5)), _bid(1, goal=(100, 100, 5))]
        result = arbitrate(ds, GoalArbitrationConfig())
        events = format_arbitration_audit(t_now=1.0, result=result)
        assert events == []

    def test_one_cluster_one_event(self) -> None:
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(4)]
        result = arbitrate(ds, GoalArbitrationConfig())
        events = format_arbitration_audit(t_now=2.5, result=result)
        assert len(events) == 1
        event_type, detail_json = events[0]
        assert event_type == AUDIT_EVENT_TYPE
        # The detail field is what GCSServer.emit_audit wants — a string.
        assert isinstance(detail_json, str)

    def test_two_clusters_two_events(self) -> None:
        ds = [
            _bid(0, goal=(0, 0, 5)),
            _bid(1, goal=(0, 0, 5)),
            _bid(2, goal=(100, 0, 5)),
            _bid(3, goal=(100, 0, 5)),
        ]
        result = arbitrate(ds, GoalArbitrationConfig())
        events = format_arbitration_audit(t_now=3.0, result=result)
        assert len(events) == 2

    def test_detail_schema_complete(self) -> None:
        """The JSON must include every field the audit narrative needs:
        time, cluster, winner, assignments, slots (with positions),
        and the full bid matrix."""
        ds = [_bid(i, position=(i, 0, 5), goal=(0, 0, 5)) for i in range(3)]
        result = arbitrate(ds, GoalArbitrationConfig())
        events = format_arbitration_audit(t_now=7.25, result=result)
        detail = json.loads(events[0][1])
        assert set(detail.keys()) == {
            "t", "cluster", "winner", "assignments", "slots", "bids"
        }
        assert detail["t"] == pytest.approx(7.25)
        assert detail["cluster"] == [0, 1, 2]
        # Winner must be the drone assigned slot 0.
        assert detail["assignments"][str(detail["winner"])] == 0
        # Slots: 3 entries, one with is_goal=True.
        assert len(detail["slots"]) == 3
        n_goal = sum(1 for s in detail["slots"] if s["is_goal"])
        assert n_goal == 1
        # Each slot position has 3 coords.
        for s in detail["slots"]:
            assert len(s["pos"]) == 3
        # Bid matrix is N×N row-major.
        assert len(detail["bids"]) == 3
        for row in detail["bids"]:
            assert len(row) == 3

    def test_winner_recomputable_from_bids(self) -> None:
        """A reviewer with the audit detail alone must be able to
        verify the winner — argmax of column 0 of the bid matrix."""
        ds = [
            _bid(0, position=(5, 0, 5), goal=(0, 0, 5), mission_priority=0.5),
            _bid(1, position=(5, 0, 5), goal=(0, 0, 5), mission_priority=0.8),
            _bid(2, position=(5, 0, 5), goal=(0, 0, 5), mission_priority=0.3),
        ]
        result = arbitrate(ds, GoalArbitrationConfig())
        events = format_arbitration_audit(t_now=0.0, result=result)
        detail = json.loads(events[0][1])
        # Recompute the slot-0 winner from the bid matrix alone.
        bids = np.asarray(detail["bids"])
        recomputed_winner_idx = int(np.argmax(bids[:, 0]))
        recomputed_winner_id = detail["cluster"][recomputed_winner_idx]
        assert recomputed_winner_id == detail["winner"]
        # The drone with highest priority should win.
        assert detail["winner"] == 1

    def test_detail_is_compact_json(self) -> None:
        """No whitespace — the GCS audit channel is bandwidth-bounded
        on real swarms, and the audit log is capped at 500 entries."""
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(3)]
        result = arbitrate(ds, GoalArbitrationConfig())
        events = format_arbitration_audit(t_now=1.0, result=result)
        # Compact JSON has no spaces between separators.
        assert ", " not in events[0][1]
        assert ": " not in events[0][1]


# ---------------------------------------------------------------------------
# Live-GCS round-trip — does GCSServer.emit_audit actually accept these?
# ---------------------------------------------------------------------------


class TestGCSRoundTrip:
    """Sanity-check that the audit events produced by
    `format_arbitration_audit` flow through a real `GCSServer.emit_audit`
    call and are retrievable via `get_audit_log`. This is the contract
    we promised in the goal_arbitration module docstring."""

    def test_emit_audit_accepts_arbitration_events(self) -> None:
        from src.gcs.gcs_server import GCSServer
        gcs = GCSServer()
        ds = [_bid(i, goal=(0, 0, 5)) for i in range(3)]
        result = arbitrate(ds, GoalArbitrationConfig())
        for event_type, detail in format_arbitration_audit(
            t_now=5.0, result=result
        ):
            gcs.emit_audit(event_type, detail)
        log = gcs.get_audit_log(limit=10)
        assert len(log) == 1
        entry = log[0]
        # GCSServer.AuditEntry.to_dict() returns {ts, event, detail}.
        assert entry["event"] == AUDIT_EVENT_TYPE
        # detail is the JSON-stringified payload.
        parsed = json.loads(entry["detail"])
        assert parsed["cluster"] == [0, 1, 2]
        assert parsed["t"] == pytest.approx(5.0)
