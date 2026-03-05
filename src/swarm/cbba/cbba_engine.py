"""
Project Sanjay Mk2 - CBBA Engine
================================
Consensus-Based Bundle Algorithm for decentralized task allocation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.core.types.drone_types import DroneState, Vector3

from .task_types import SwarmTask


@dataclass
class CBBAConfig:
    max_bundle_size: int = 3
    max_task_range: float = 1500.0
    battery_reserve: float = 15.0
    formation_spacing: float = 80.0
    cruise_speed: float = 5.0

    weight_distance: float = 0.30
    weight_battery: float = 0.25
    weight_priority: float = 0.20
    weight_synergy: float = 0.10
    weight_urgency: float = 0.10
    weight_load_penalty: float = 0.05


class CBBAEngine:
    """Local CBBA state machine running per drone."""

    def __init__(
        self,
        drone_id: int,
        config: Optional[CBBAConfig] = None,
        home_position: Optional[Vector3] = None,
    ):
        self.drone_id = drone_id
        self.config = config or CBBAConfig()
        self.home_position = home_position or Vector3()

        self._known_tasks: Dict[str, SwarmTask] = {}
        self._bundle: List[str] = []
        self.winning_bids: Dict[str, float] = {}
        self.winning_agents: Dict[str, int] = {}
        self.bid_timestamps: Dict[str, float] = {}

        self._tasks_dirty = True
        self._cached_tasks_payload: Optional[List[Dict]] = None

    def upsert_tasks(self, tasks: List[SwarmTask]):
        for task in tasks:
            self._known_tasks[task.task_id] = task
        if tasks:
            self._tasks_dirty = True

    def upsert_task(self, task: SwarmTask):
        self._known_tasks[task.task_id] = task
        self._tasks_dirty = True

    def remove_task(self, task_id: str):
        self._known_tasks.pop(task_id, None)
        self.winning_bids.pop(task_id, None)
        self.winning_agents.pop(task_id, None)
        self.bid_timestamps.pop(task_id, None)
        if task_id in self._bundle:
            self._remove_from_bundle(task_id)
        self._tasks_dirty = True

    def get_bundle_ids(self) -> List[str]:
        return list(self._bundle)

    def get_current_task(self) -> Optional[SwarmTask]:
        for task_id in self._bundle:
            if self.winning_agents.get(task_id) == self.drone_id:
                return self._known_tasks.get(task_id)
        return None

    def score_task(
        self,
        drone_state: DroneState,
        task: SwarmTask,
        current_bundle: List[SwarmTask],
    ) -> float:
        dist = drone_state.position.distance_to(task.position)
        dist_score = max(0.0, 1.0 - dist / self.config.max_task_range)

        energy_to_task = self._estimate_energy(dist)
        energy_for_task = task.estimated_energy
        energy_to_home = self._estimate_energy(task.position.distance_to(self.home_position))
        remaining = drone_state.battery - energy_to_task - energy_for_task - energy_to_home

        if remaining < self.config.battery_reserve:
            return -1.0

        battery_score = max(0.0, min(remaining / 100.0, 1.0))
        priority_score = max(0.0, min(task.priority / 10.0, 1.0))

        synergy = 0.0
        for existing in current_bundle:
            if existing.position.distance_to(task.position) < self.config.formation_spacing * 2.0:
                synergy += 0.2

        load_penalty = len(current_bundle) * 0.15

        urgency = 0.0
        if task.deadline is not None:
            time_left = task.deadline - time.time()
            if time_left < task.estimated_duration * 2.0:
                urgency = 0.5

        return (
            self.config.weight_distance * dist_score
            + self.config.weight_battery * battery_score
            + self.config.weight_priority * priority_score
            + self.config.weight_synergy * synergy
            + self.config.weight_urgency * urgency
            - self.config.weight_load_penalty * load_penalty
        )

    def bundle_phase(self, drone_state: DroneState):
        """Greedy local bundle construction."""
        while len(self._bundle) < self.config.max_bundle_size:
            best_task: Optional[SwarmTask] = None
            best_score = -math.inf

            current_bundle = [
                self._known_tasks[task_id]
                for task_id in self._bundle
                if task_id in self._known_tasks
            ]

            for task in self._known_tasks.values():
                if task.task_id in self._bundle:
                    continue
                if task.assigned_to is not None and task.assigned_to != self.drone_id:
                    continue

                score = self.score_task(drone_state, task, current_bundle)
                if score < 0.0:
                    continue

                current_winner_score = self.winning_bids.get(task.task_id, -math.inf)
                current_winner_agent = self.winning_agents.get(task.task_id, math.inf)

                can_outbid = score > current_winner_score or (
                    score == current_winner_score and self.drone_id < current_winner_agent
                )
                if can_outbid and score > best_score:
                    best_task = task
                    best_score = score

            if best_task is None:
                break

            self._bundle.append(best_task.task_id)
            now = time.time()
            self.winning_bids[best_task.task_id] = best_score
            self.winning_agents[best_task.task_id] = self.drone_id
            self.bid_timestamps[best_task.task_id] = now

        self._sync_assignments()

    def consensus_phase(
        self,
        remote_bids: Dict[str, float],
        remote_agents: Dict[str, int],
        remote_id: int,
        remote_timestamps: Optional[Dict[str, float]] = None,
    ):
        """Apply CBBA conflict resolution against a remote peer state."""
        remote_timestamps = remote_timestamps or {}

        for task_id, remote_score in remote_bids.items():
            remote_agent = remote_agents.get(task_id, remote_id)
            remote_ts = remote_timestamps.get(task_id, 0.0)

            local_score = self.winning_bids.get(task_id, -math.inf)
            local_agent = self.winning_agents.get(task_id, math.inf)
            local_ts = self.bid_timestamps.get(task_id, 0.0)

            if remote_ts < local_ts and local_agent == self.drone_id:
                continue

            remote_wins = remote_score > local_score or (
                remote_score == local_score and remote_agent < local_agent
            )

            if remote_wins:
                self.winning_bids[task_id] = remote_score
                self.winning_agents[task_id] = remote_agent
                self.bid_timestamps[task_id] = remote_ts
                if task_id in self._bundle and remote_agent != self.drone_id:
                    self._remove_from_bundle(task_id)

        self._sync_assignments()

    def ingest_remote_payload(self, remote_id: int, payload: Dict):
        """Convenience wrapper for payload from `get_bids_payload`."""
        bids: Dict[str, float] = {}
        agents: Dict[str, int] = {}
        timestamps: Dict[str, float] = {}

        for task_id, entry in payload.items():
            bids[task_id] = float(entry.get("score", -math.inf))
            agents[task_id] = int(entry.get("agent_id", remote_id))
            timestamps[task_id] = float(entry.get("ts", 0.0))

        self.consensus_phase(bids, agents, remote_id=remote_id, remote_timestamps=timestamps)

    def get_bids_payload(self) -> Dict[str, Dict[str, float | int]]:
        payload: Dict[str, Dict[str, float | int]] = {}
        for task_id in self._bundle:
            payload[task_id] = {
                "score": float(self.winning_bids.get(task_id, -math.inf)),
                "agent_id": int(self.winning_agents.get(task_id, self.drone_id)),
                "ts": float(self.bid_timestamps.get(task_id, 0.0)),
            }
        return payload

    def get_known_tasks_payload(self) -> List[Dict]:
        if self._tasks_dirty or self._cached_tasks_payload is None:
            self._cached_tasks_payload = [task.to_dict() for task in self._known_tasks.values()]
            self._tasks_dirty = False
        return self._cached_tasks_payload

    def _remove_from_bundle(self, task_id: str):
        if task_id not in self._bundle:
            return

        cut_index = self._bundle.index(task_id)
        removed_suffix = self._bundle[cut_index:]
        self._bundle = self._bundle[:cut_index]

        for removed in removed_suffix:
            if self.winning_agents.get(removed) == self.drone_id:
                self.winning_bids.pop(removed, None)
                self.winning_agents.pop(removed, None)
                self.bid_timestamps.pop(removed, None)

    def _sync_assignments(self):
        changed = False
        for task in self._known_tasks.values():
            new_owner = self.winning_agents.get(task.task_id, task.assigned_to)
            if task.assigned_to != new_owner:
                task.assigned_to = new_owner
                changed = True
        if changed:
            self._tasks_dirty = True

    def _estimate_energy(self, distance: float) -> float:
        # Coarse linear model tuned for bundle feasibility pruning.
        return max(0.0, (distance / max(self.config.cruise_speed, 0.1)) * 0.02)

    def get_task(self, task_id: str) -> Optional[SwarmTask]:
        return self._known_tasks.get(task_id)

    def all_tasks(self) -> List[SwarmTask]:
        return list(self._known_tasks.values())

    def clear_agent_claims(self, agent_id: int):
        """Release tasks won by a departed/failed agent."""
        to_release = [
            task_id
            for task_id, winner in self.winning_agents.items()
            if winner == agent_id
        ]
        for task_id in to_release:
            self.winning_agents.pop(task_id, None)
            self.winning_bids.pop(task_id, None)
            self.bid_timestamps.pop(task_id, None)
            if task_id in self._bundle:
                self._remove_from_bundle(task_id)
        self._sync_assignments()
