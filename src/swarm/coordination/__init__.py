"""
Project Sanjay Mk2 - Swarm Coordination Submodule
==================================================
Regiment-level coordination for multi-drone operations.

Components:
    - AlphaRegimentCoordinator: 6-drone Alpha regiment management

@author: Archishman Paul
"""

from .regiment_coordinator import AlphaRegimentCoordinator, RegimentConfig

RegimentCoordinator = AlphaRegimentCoordinator

__all__ = [
    "AlphaRegimentCoordinator",
    "RegimentCoordinator",
    "RegimentConfig",
]
