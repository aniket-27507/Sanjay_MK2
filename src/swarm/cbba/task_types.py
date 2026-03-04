"""
Project Sanjay Mk2 - CBBA Task Types
====================================
Task model used by decentralized task allocation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from src.core.types.drone_types import SensorType, Vector3


class TaskType(Enum):
    SECTOR_COVERAGE = auto()
    THREAT_INVESTIGATE = auto()
    PERIMETER_PATROL = auto()
    RELAY_STATION = auto()
    RTL = auto()


@dataclass
class SwarmTask:
    task_id: str
    task_type: TaskType
    position: Vector3
    radius: float
    priority: float = 1.0
    deadline: Optional[float] = None
    required_sensors: List[SensorType] = field(default_factory=list)
    estimated_duration: float = 60.0
    estimated_energy: float = 5.0
    created_at: float = field(default_factory=time.time)
    assigned_to: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.name,
            "position": [self.position.x, self.position.y, self.position.z],
            "radius": self.radius,
            "priority": self.priority,
            "deadline": self.deadline,
            "required_sensors": [s.name for s in self.required_sensors],
            "estimated_duration": self.estimated_duration,
            "estimated_energy": self.estimated_energy,
            "created_at": self.created_at,
            "assigned_to": self.assigned_to,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "SwarmTask":
        pos = payload.get("position", [0.0, 0.0, 0.0])
        sensors = [SensorType[s] for s in payload.get("required_sensors", [])]
        return cls(
            task_id=str(payload.get("task_id", "task_unknown")),
            task_type=TaskType[payload.get("task_type", "SECTOR_COVERAGE")],
            position=Vector3(float(pos[0]), float(pos[1]), float(pos[2])),
            radius=float(payload.get("radius", 20.0)),
            priority=float(payload.get("priority", 1.0)),
            deadline=payload.get("deadline"),
            required_sensors=sensors,
            estimated_duration=float(payload.get("estimated_duration", 60.0)),
            estimated_energy=float(payload.get("estimated_energy", 5.0)),
            created_at=float(payload.get("created_at", time.time())),
            assigned_to=payload.get("assigned_to"),
        )
