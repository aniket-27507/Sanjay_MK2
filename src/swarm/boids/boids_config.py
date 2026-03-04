"""
Project Sanjay Mk2 - Boids Configuration
========================================
Configuration dataclasses for decentralized boids flocking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoidsConfig:
    """Runtime-tunable parameters for the boids steering engine."""

    # Neighbor detection
    neighbor_radius: float = 120.0
    min_separation: float = 50.0

    # Steering weights
    w_separation: float = 2.5
    w_alignment: float = 1.0
    w_cohesion: float = 0.8
    w_goal_seeking: float = 1.5
    w_obstacle_avoidance: float = 3.0
    w_formation_bias: float = 0.6
    w_energy_saving: float = 0.4

    # Velocity limits
    max_speed: float = 8.0
    cruise_speed: float = 5.0
    max_vertical_speed: float = 3.0

    # Energy optimization
    acceleration_penalty: float = 0.5
    speed_convergence_rate: float = 0.3

    # Obstacle handling
    obstacle_detection_range: float = 30.0
    obstacle_safe_distance: float = 15.0
