"""
Project Sanjay Mk2 - Obstacle Avoidance Submodule
==================================================
Three-tier obstacle avoidance stack for Alpha Drones.

Components:
    - APF3DAvoidance: 3D Artificial Potential Field core algorithm
    - HardwareProtectionLayer: Last-resort collision prevention (HPL)
    - TacticalPlanner: A* pathfinding for local-minima escape
    - AvoidanceManager: Central orchestrator for all three tiers

@author: Archishman Paul
"""

from .apf_3d import APF3DAvoidance, APF3DConfig, AvoidanceState, Obstacle3D
from .hardware_protection import HardwareProtectionLayer, HPLConfig, HPLState
from .tactical_planner import TacticalPlanner, PlannerConfig


def __getattr__(name):
    if name in {"AvoidanceManager", "AvoidanceManagerConfig"}:
        from .avoidance_manager import AvoidanceManager, AvoidanceManagerConfig

        return {
            "AvoidanceManager": AvoidanceManager,
            "AvoidanceManagerConfig": AvoidanceManagerConfig,
        }[name]
    raise AttributeError(name)

__all__ = [
    "APF3DAvoidance",
    "APF3DConfig",
    "AvoidanceState",
    "Obstacle3D",
    "HardwareProtectionLayer",
    "HPLConfig",
    "HPLState",
    "TacticalPlanner",
    "PlannerConfig",
    "AvoidanceManager",
    "AvoidanceManagerConfig",
]
