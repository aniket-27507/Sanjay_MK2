"""
Project Sanjay Mk2 - CBBA Task Generator
=========================================
Creates decentralized CBBA tasks from mission context events.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.core.types.drone_types import Vector3

from .task_types import SwarmTask, TaskType


@dataclass
class TaskGeneratorConfig:
    battery_low_threshold: float = 30.0
    perimeter_interval_s: float = 20.0


class TaskGenerator:
    """Deterministic task factory to avoid duplicate task churn."""

    def __init__(self, config: Optional[TaskGeneratorConfig] = None):
        self.config = config or TaskGeneratorConfig()
        self._tasks: Dict[str, SwarmTask] = {}
        self._startup_generated = False
        self._last_perimeter_gen = 0.0

    def upsert_task(self, task: SwarmTask):
        self._tasks[task.task_id] = task

    def upsert_tasks(self, tasks: Sequence[SwarmTask]):
        for task in tasks:
            self.upsert_task(task)

    def list_tasks(self) -> List[SwarmTask]:
        return list(self._tasks.values())

    def generate_startup_tasks(self, sectors: Sequence[object]) -> List[SwarmTask]:
        if self._startup_generated:
            return []

        created: List[SwarmTask] = []
        for sector in sectors:
            if sector is None:
                continue
            task_id = f"sector_{getattr(sector, 'drone_id', 'unknown')}"
            center = getattr(sector, "center", Vector3())
            radius = float(getattr(sector, "radius", 50.0))
            task = SwarmTask(
                task_id=task_id,
                task_type=TaskType.SECTOR_COVERAGE,
                position=center,
                radius=radius,
                priority=4.0,
                estimated_duration=120.0,
                estimated_energy=8.0,
            )
            self.upsert_task(task)
            created.append(task)

        self._startup_generated = True
        return created

    def generate_threat_task(self, threat_data: Dict) -> SwarmTask:
        threat_id = str(threat_data.get("threat_id") or "unknown")
        pos = threat_data.get("position", [0.0, 0.0, 0.0])
        task = SwarmTask(
            task_id=f"threat_{threat_id}",
            task_type=TaskType.THREAT_INVESTIGATE,
            position=Vector3(float(pos[0]), float(pos[1]), float(pos[2])),
            radius=float(threat_data.get("radius", 30.0)),
            priority=float(threat_data.get("priority", 9.0)),
            estimated_duration=float(threat_data.get("estimated_duration", 90.0)),
            estimated_energy=float(threat_data.get("estimated_energy", 10.0)),
            deadline=threat_data.get("deadline"),
        )
        self.upsert_task(task)
        return task

    def generate_rtl_task(self, drone_id: int, home_position: Vector3) -> SwarmTask:
        task = SwarmTask(
            task_id=f"rtl_{drone_id}",
            task_type=TaskType.RTL,
            position=home_position,
            radius=8.0,
            priority=10.0,
            estimated_duration=90.0,
            estimated_energy=3.0,
            assigned_to=drone_id,
        )
        self.upsert_task(task)
        return task

    def generate_perimeter_tasks(
        self,
        center: Vector3,
        radius: float,
        segments: int = 4,
    ) -> List[SwarmTask]:
        now = time.time()
        if now - self._last_perimeter_gen < self.config.perimeter_interval_s:
            return []

        self._last_perimeter_gen = now
        created: List[SwarmTask] = []
        if segments <= 0:
            return created

        for i in range(segments):
            angle = (2.0 * 3.141592653589793 * i) / segments
            point = Vector3(
                x=center.x + radius * 0.8 * math.cos(angle),
                y=center.y + radius * 0.8 * math.sin(angle),
                z=center.z,
            )
            task = SwarmTask(
                task_id=f"perimeter_{i}",
                task_type=TaskType.PERIMETER_PATROL,
                position=point,
                radius=max(15.0, radius * 0.2),
                priority=3.0,
                estimated_duration=60.0,
                estimated_energy=4.0,
            )
            self.upsert_task(task)
            created.append(task)

        return created

    def generate_relay_task(self, partition_id: str, position: Vector3) -> SwarmTask:
        task = SwarmTask(
            task_id=f"relay_{partition_id}",
            task_type=TaskType.RELAY_STATION,
            position=position,
            radius=20.0,
            priority=7.0,
            estimated_duration=180.0,
            estimated_energy=6.0,
        )
        self.upsert_task(task)
        return task
