"""MINCO planning core.

This package contains the clean-room Python port of ZJU-FAST-Lab GCOPTER's
trajectory optimization stack. See docs/MINCO_PIVOT.md for the full
specification.

Module order (data flow):
    voxel_map       -> 3D occupancy grid (depth camera output goes here)
    sfc_gen         -> RRT path search over voxel_map
    corridor_generator
                    -> FIRI convex polytope corridors around the RRT route
    minco           -> MINCO trajectory representation (waypoints + durations)
    gcopter         -> L-BFGS optimizer with smooth-map penalties
    flatness        -> Differential flatness map (p, v, a, j) -> (thrust, q, omega)
    trajectory_tracker
                    -> Trajectory -> flight controller setpoints
"""

from src.single_drone.planning.corridor_generator import (
    Polytope,
    convex_cover,
    inflate_segment_polytope,
    polytope_contains,
    shortcut,
)
from src.single_drone.planning.flatness import (
    evaluate_trajectory_dynamics,
    flat_state,
    is_dynamically_feasible,
    rotate_vector_by_quat,
    rotation_matrix_to_quat,
)
from src.single_drone.planning.gcopter import (
    GCopterConfig,
    gcopter_optimize,
)
from src.single_drone.planning.minco import (
    M_matrix,
    Q_matrix,
    Trajectory,
)
from src.single_drone.planning.sfc_gen import (
    plan_path_rrt,
    plan_path_rrt_connect,
    shortcut_path,
)
from src.single_drone.planning.voxel_map import VoxelMap

__all__ = [
    "GCopterConfig",
    "M_matrix",
    "Polytope",
    "Q_matrix",
    "Trajectory",
    "VoxelMap",
    "convex_cover",
    "evaluate_trajectory_dynamics",
    "flat_state",
    "gcopter_optimize",
    "inflate_segment_polytope",
    "is_dynamically_feasible",
    "plan_path_rrt",
    "plan_path_rrt_connect",
    "polytope_contains",
    "rotate_vector_by_quat",
    "rotation_matrix_to_quat",
    "shortcut",
    "shortcut_path",
]
