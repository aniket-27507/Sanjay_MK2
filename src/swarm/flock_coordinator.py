"""
Project Sanjay Mk2 - Flock Coordinator
======================================
Integrates CBBA task selection and Boids motion generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from src.core.types.drone_types import DroneState, Vector3
from src.swarm.boids import BoidsConfig, BoidsEngine
from src.swarm.boids.dynamic_behaviors import (
    EnvironmentContext,
    FlockState,
    recommend_formation,
    scale_formation,
    should_merge,
    should_split,
)
from src.swarm.cbba import (
    CBBAConfig,
    CBBAEngine,
    SwarmTask,
    TaskGenerator,
    TaskGeneratorConfig,
)
from src.swarm.cbba.task_types import TaskType
from src.swarm.formation import FormationConfig, FormationController


@dataclass
class FlockCoordinatorConfig:
    boids: BoidsConfig = field(default_factory=BoidsConfig)
    cbba: CBBAConfig = field(default_factory=CBBAConfig)
    task_gen: TaskGeneratorConfig = field(default_factory=TaskGeneratorConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)


class FlockCoordinator:
    """Per-drone decentralized coordinator for WHAT (CBBA) + HOW (Boids)."""

    def __init__(
        self,
        drone_id: int,
        config: Optional[FlockCoordinatorConfig] = None,
        num_drones: int = 6,
    ):
        self.drone_id = drone_id
        self.config = config or FlockCoordinatorConfig()

        self.boids = BoidsEngine(self.config.boids)
        self.cbba = CBBAEngine(drone_id=drone_id, config=self.config.cbba)
        self.task_gen = TaskGenerator(self.config.task_gen)
        self.formation = FormationController(num_drones=num_drones, config=self.config.formation)

        self._active_drone_ids: List[int] = []
        self._known_members: set[int] = set()
        self._original_spacing = self.config.formation.spacing
        self._current_goal: Optional[Vector3] = None
        self._last_cbba_sync = 0.0
        self._boids_enabled = True
        self._cbba_enabled = True
        self._formation_enabled = True

    @property
    def current_goal(self) -> Optional[Vector3]:
        return self._current_goal

    def update_membership(self, drone_ids: Iterable[int]):
        new_members = set(drone_ids)
        removed = self._known_members - new_members
        for removed_id in removed:
            self.cbba.clear_agent_claims(removed_id)

        self._known_members = new_members
        self._active_drone_ids = sorted(new_members)
        if self._active_drone_ids:
            self.formation.assign_drones(self._active_drone_ids)

    def upsert_tasks(self, tasks: List[SwarmTask]):
        self.task_gen.upsert_tasks(tasks)
        self.cbba.upsert_tasks(tasks)

    def ingest_gossip_payload(self, sender_id: int, cbba_payload: Dict):
        for raw_task in cbba_payload.get("known_tasks", []):
            task = SwarmTask.from_dict(raw_task)
            self.task_gen.upsert_task(task)
            self.cbba.upsert_task(task)

        self.cbba.ingest_remote_payload(sender_id, cbba_payload.get("bids", {}))

    def prepare_gossip_payload(self, my_state: DroneState) -> Dict:
        return {
            "bids": self.cbba.get_bids_payload(),
            "bundle": self.cbba.get_bundle_ids(),
            "known_tasks": self.cbba.get_known_tasks_payload(),
            "timestamp": my_state.timestamp,
        }

    @property
    def boids_enabled(self) -> bool:
        return self._boids_enabled

    @property
    def cbba_enabled(self) -> bool:
        return self._cbba_enabled

    @property
    def formation_enabled(self) -> bool:
        return self._formation_enabled

    def enable_boids(self, enabled: bool = True):
        self._boids_enabled = enabled

    def disable_boids(self):
        self._boids_enabled = False

    def enable_cbba(self, enabled: bool = True):
        self._cbba_enabled = enabled

    def disable_cbba(self):
        self._cbba_enabled = False

    def enable_formation(self, enabled: bool = True):
        self._formation_enabled = enabled

    def disable_formation(self):
        self._formation_enabled = False

    def tick(
        self,
        my_state: DroneState,
        peer_states: Dict[int, DroneState],
        obstacles: Optional[List[dict]] = None,
        sector_assignments: Optional[List[object]] = None,
        home_position: Optional[Vector3] = None,
    ) -> Vector3:
        """Run one decentralized coordination tick and return desired velocity."""
        sectors = sector_assignments or []
        if not self.task_gen._startup_generated and sectors:
            startup = self.task_gen.generate_startup_tasks(sectors)
            if startup:
                self.upsert_tasks(startup)

        if my_state.battery <= self.task_gen.config.battery_low_threshold:
            self.upsert_tasks([
                self.task_gen.generate_rtl_task(
                    drone_id=my_state.drone_id,
                    home_position=home_position or Vector3(),
                )
            ])

        # Lightweight periodic perimeter tasking in degraded coverage mode.
        if len(self._active_drone_ids) < 6 and sectors:
            center = sectors[0].center if hasattr(sectors[0], "center") else Vector3()
            radius = float(getattr(sectors[0], "radius", 100.0))
            self.upsert_tasks(self.task_gen.generate_perimeter_tasks(center, radius, segments=4))

        current_task = None
        if self._cbba_enabled:
            self.cbba.bundle_phase(my_state)
            current_task = self.cbba.get_current_task()

        slot = self.formation.get_slot_for_drone(my_state.drone_id) if self._formation_enabled else None

        if current_task:
            task_dist = (current_task.position - self.formation._center).magnitude()
            if slot and task_dist < self.formation.config.spacing * 2:
                # Task is near formation — let formation bias handle positioning
                goal = slot
            else:
                # Task is far — suppress formation to avoid conflicting pulls
                goal = current_task.position
                slot = None
        else:
            goal = slot if slot else my_state.position

        if goal is None:
            goal = my_state.position
        self._current_goal = goal

        all_states = {my_state.drone_id: my_state, **peer_states}

        if self._formation_enabled:
            self._apply_dynamic_behaviors(all_states)

        if self._boids_enabled:
            velocity = self.boids.compute(
                drone_id=my_state.drone_id,
                states=all_states,
                goal=goal,
                obstacles=obstacles or [],
                formation_slot=slot,
            )
        else:
            velocity = self._direct_goal_velocity(my_state.position, goal)

        # For RTL tasks, bias strongly toward home.
        if current_task and current_task.task_type == TaskType.RTL:
            rtl_bias = (current_task.position - my_state.position).normalized() * 1.2
            velocity = self.boids._clamp_velocity(velocity + rtl_bias)  # intentional internal clamp reuse

        self._last_cbba_sync = time.time()
        return velocity

    def _apply_dynamic_behaviors(self, states: Dict[int, DroneState]):
        if not states:
            return

        task_positions = [
            task.position for task in self.cbba.all_tasks()
            if task.assigned_to is not None
        ]
        groups = should_split(
            drone_ids=list(states.keys()),
            task_positions=task_positions,
            formation_spacing=self.formation.config.spacing,
        )

        sub_flocks: List[FlockState] = []
        for group in groups:
            if not group:
                continue
            centroid = Vector3()
            for drone_id in group:
                centroid = centroid + states[drone_id].position
            centroid = centroid / float(len(group))
            sub_flocks.append(FlockState(drone_ids=group, centroid=centroid))

        if should_merge(sub_flocks, formation_spacing=self.formation.config.spacing):
            threat_density = 0.0
        else:
            # Spread-out tasks imply multi-front mission pressure.
            threat_density = 1.0

        env = EnvironmentContext(
            corridor_width=max(80.0, self.formation.config.spacing * len(groups)),
            threat_density=threat_density,
            perimeter_mode=len(self._active_drone_ids) < 6,
        )

        recommended = recommend_formation(env)
        if self.formation.config.formation_type != recommended:
            self.formation.config.formation_type = recommended
            self.formation._generate_slots()

        scaled_spacing = scale_formation(len(self._active_drone_ids), self._original_spacing)
        if abs(self.formation.config.spacing - scaled_spacing) > 1e-6:
            self.formation.config.spacing = scaled_spacing
            self.formation._generate_slots()

    def _direct_goal_velocity(self, current: Vector3, goal: Vector3) -> Vector3:
        direction = goal - current
        if direction.magnitude() < 1e-6:
            return Vector3()
        # Use boids max speed config as common command cap.
        max_speed = max(0.1, float(self.config.boids.max_speed))
        return direction.normalized() * min(direction.magnitude(), max_speed)
