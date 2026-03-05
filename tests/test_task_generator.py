"""
Project Sanjay Mk2 - Task Generator Tests
==========================================
"""

import time
import pytest
from dataclasses import dataclass

from src.core.types.drone_types import Vector3
from src.swarm.cbba.task_generator import TaskGenerator, TaskGeneratorConfig
from src.swarm.cbba.task_types import TaskType


@dataclass
class FakeSector:
    drone_id: int
    center: Vector3
    radius: float = 100.0


class TestTaskGenerator:

    def test_startup_tasks_generated_once(self):
        gen = TaskGenerator()
        sectors = [FakeSector(drone_id=i, center=Vector3(x=float(i * 80))) for i in range(3)]
        first = gen.generate_startup_tasks(sectors)
        assert len(first) == 3
        second = gen.generate_startup_tasks(sectors)
        assert len(second) == 0

    def test_threat_task(self):
        gen = TaskGenerator()
        task = gen.generate_threat_task({
            "threat_id": "t1",
            "position": [100.0, 200.0, -65.0],
            "priority": 9.0,
        })
        assert task.task_type == TaskType.THREAT_INVESTIGATE
        assert task.priority == 9.0
        assert task.task_id == "threat_t1"

    def test_rtl_task_assigned_to_drone(self):
        gen = TaskGenerator()
        task = gen.generate_rtl_task(drone_id=3, home_position=Vector3(0, 0, 0))
        assert task.task_type == TaskType.RTL
        assert task.assigned_to == 3
        assert task.priority == 10.0

    def test_perimeter_tasks_throttled(self):
        gen = TaskGenerator(config=TaskGeneratorConfig(perimeter_interval_s=10.0))
        first = gen.generate_perimeter_tasks(Vector3(), 100.0, segments=4)
        assert len(first) == 4
        second = gen.generate_perimeter_tasks(Vector3(), 100.0, segments=4)
        assert len(second) == 0

    def test_relay_task(self):
        gen = TaskGenerator()
        task = gen.generate_relay_task("part_a", Vector3(50, 50, -30))
        assert task.task_type == TaskType.RELAY_STATION
        assert task.task_id == "relay_part_a"

    def test_upsert_preserves_task(self):
        gen = TaskGenerator()
        task = gen.generate_threat_task({"threat_id": "x", "position": [0, 0, 0]})
        tasks = gen.list_tasks()
        assert any(t.task_id == "threat_x" for t in tasks)

    def test_empty_sectors(self):
        gen = TaskGenerator()
        result = gen.generate_startup_tasks([])
        assert result == []
