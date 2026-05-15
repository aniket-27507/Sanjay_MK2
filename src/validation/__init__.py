"""Validation rigs for the MINCO pivot.

See docs/MINCO_PIVOT.md §5 for the rig design. Each rig is a self-contained
Python module that loads obstacle / drone / mission configuration, runs the
planner under stress, and emits a JSON metrics file.

Rigs:
    rig1_corridor_benchmark  single-drone planner performance vs density
    rig2_swarm_avoidance     N-drone scaling and collision avoidance
    rig3_vio_perimeter       GPS-denied drift + perimeter fencing
    rig4_mission_response    threat detect -> inspect -> regroup
    rig5_endurance           battery / motor degradation / drone loss
    rig6_disturbance         wind, fog, sensor failure

The shared utilities are obstacle_gen (procedural obstacle clouds),
broadcast_channel (simulated WiFi mesh), vio_drift_model (drift injection),
motor_model (thrust degradation), metrics (collection + export), and plots
(matplotlib dashboards). These will be added as the rigs come online.
"""

from src.validation.metrics import MetricsCollector, summarise
from src.validation.obstacle_gen import (
    clear_around,
    random_obstacle_field,
    random_pillars,
)

__all__ = [
    "MetricsCollector",
    "clear_around",
    "random_obstacle_field",
    "random_pillars",
    "summarise",
]
