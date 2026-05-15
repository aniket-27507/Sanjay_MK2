# MINCO Pivot: Complete Specification

**Date:** 2026-05-15
**Author:** Archishman Paul
**Status:** Approved — ready for implementation
**Scope:** Replace APF/A*/Boids/servo-LiDAR with MINCO trajectory optimization + depth camera

---

## 1. What is changing and why

### 1.1 The old stack (being replaced)

```
Sensors:     RPLiDAR A1 (2D) + servo → scan-then-move → 3D point cloud
Mapping:     Voxel occupancy grid from accumulated LiDAR scans
Tactical:    A* on 2D costmap (fires when APF gets STUCK)
Operational: APF 3D (potential field from voxels) → velocity command
Swarm:       Boids (Reynolds velocity blend)
Safety:      HPL (raw LiDAR range override)
```

**Why it fails:**
- APF has unavoidable local minima (Morse theory: any smooth potential on the punctured configuration space must have critical points besides the global minimum)
- A* fallback is slow, 2D-only, fixed altitude
- Boids + APF blending creates velocity discontinuities
- Servo scan-then-move halts the drone, 12m range, mechanical complexity
- Simulation of servo physics required days of custom Blender code

### 1.2 The new stack (MINCO + depth camera)

```
Sensors:     OAK-D Lite depth camera (stereo depth 8-10m + RGB 4K + IMU, 61g, $150)
Mapping:     Depth images → 3D occupancy grid (30fps, continuous flight)
Path:        RRT on voxel map → coarse waypoint route
Corridors:   FIRI (fast iterative region inflation) → convex polytopes in free space
Trajectory:  MINCO optimizer (L-BFGS, minimizes snap + time, within corridors)
Swarm:       Broadcast MINCO trajectories (~0.5KB each) → penalty-based avoidance
Safety:      HPL (depth image minimum distance override)
GPS-denied:  VIO from same depth camera + IMU (Phase 2 post-demo)
```

**Why it works:**
- No local minima: corridors are topologically correct by construction
- Trajectories are provably smooth, time-optimal, dynamically feasible
- Depth camera is the native input for the entire ZJU-FAST-Lab ecosystem
- RGB stream feeds YOLO surveillance for free (one sensor, two pipelines)
- Continuous flight (no scan-then-move), lighter weight, no mechanical parts
- Simulation is trivial (Z-buffer depth render vs custom servo physics)
- GPS-denied upgrade requires zero new hardware (stereo + IMU → VIO)

### 1.3 Reference implementation

GCOPTER by ZJU-FAST-Lab: https://github.com/ZJU-FAST-Lab/GCOPTER

- Paper: "Geometrically Constrained Trajectory Optimization for Multicopters" (IEEE T-RO 2022)
- Author: Zhepei Wang, Fei Gao
- License: MIT (core optimizer), GPL-3.0 (EGO-Planner-v2 swarm system)
- Our implementation: clean-room Python port of the MINCO algorithm. No GPL code copied.

---

## 2. The MINCO algorithm

### 2.1 Trajectory representation

MINCO (Minimum Control) parameterizes a trajectory by **intermediate waypoints** q_i and **time allocations** T_i, not by basis function coefficients.

Given waypoints q_0, q_1, ..., q_M and durations T_1, ..., T_M, the minimum-control-effort trajectory connecting consecutive waypoints with given durations has a **closed-form solution**. For snap minimization (s=3), each segment is a degree-2s+1 = 7th order polynomial.

Decision variables: (q, T) — where q = [q_1, ..., q_{M-1}] are free intermediate waypoints and T = [T_1, ..., T_M] are segment durations.

Start/end states (position, velocity, acceleration) are fixed boundary conditions.

### 2.2 Optimization formulation

```
min_{q, T}  w_T * sum(T_i)  +  sum_r [ chi_r * penalty_r(q, T) ]

where penalty terms (via smooth maps) include:
  - Corridor containment:  trajectory must stay inside convex polytopes
  - Velocity limit:        ||v(t)|| <= v_max
  - Thrust limits:         f_min <= ||f(t)|| <= f_max  (with drag)
  - Tilt angle:            tilt(t) <= theta_max
  - Body rate:             ||omega(t)|| <= omega_max
```

Each penalty is a smooth function of (q, T), computed via numerical quadrature along the trajectory. The total cost is unconstrained and differentiable → solved by L-BFGS.

### 2.3 Safe flight corridors (FIRI)

FIRI (Fast Iterative Region Inflation) generates convex polytopes in free space:

1. Start with a seed point on the RRT path
2. Inflate an ellipsoid outward from the seed
3. When the ellipsoid hits an obstacle, convert to a halfplane constraint
4. Repeat until the polytope stabilizes
5. Advance to next seed point, overlap with previous polytope

Output: a sequence of convex polytopes H_1, ..., H_K where each H_i = {x : A_i x <= b_i}. Consecutive polytopes overlap. The MINCO trajectory passes through them in order.

### 2.4 Swarm trajectory broadcast

Each drone broadcasts its MINCO trajectory (waypoints q + times T + boundary conditions) over WiFi mesh. Size: ~0.5 KB per trajectory.

When drone k receives drone i's trajectory, it evaluates the trajectory at future times to predict i's position. If predicted distance is below threshold, drone k adds a swarm penalty J_w to its MINCO cost and re-optimizes.

### 2.5 Differential flatness

Quadrotor dynamics are differentially flat: given a position trajectory p(t), the full state (attitude, thrust, body rates) is uniquely determined. GCOPTER's `flatness.hpp` computes:

```
(p, v, a, j) → (thrust, quaternion, body_rate)
```

accounting for nonlinear aerodynamic drag. This lets us validate dynamic feasibility analytically — no physics simulation needed.

---

## 3. Sensor suite pivot

### 3.1 Primary sensor: OAK-D Lite

| Spec | Value |
|------|-------|
| Depth | Stereo, 640x480 @ 30fps |
| Depth range | 0.2m - 10m (8m reliable outdoors) |
| RGB | 4056x3040 (4K) |
| IMU | BNO086 (accel + gyro) |
| Neural compute | Myriad X VPU (4 TOPS, can run YOLO on-camera) |
| Interface | USB-C |
| Weight | 61g |
| Price | ~$150 |
| FOV | 73° HFOV (depth), 81° HFOV (RGB) |

Alternative: Intel RealSense D435i ($300, structured light, better outdoor depth, heavier).

### 3.2 What the depth camera replaces

| Old component | Replaced by | Notes |
|---------------|------------|-------|
| RPLiDAR A1 ($100) | OAK-D Lite depth stream | Continuous 30fps vs scan-then-move |
| Servo ($15) + bracket | Eliminated | No mechanical parts |
| `src/single_drone/sensors/lidar_3d.py` | New `depth_camera.py` | |
| `src/single_drone/sensors/real_lidar.py` | Deprecated | |
| `src/simulation/physics/servo_lidar_model.py` | Eliminated | Depth sim = Z-buffer render |
| `src/simulation/physics/lidar_noise_model.py` | New depth noise model | Stereo matching noise vs ToF noise |

### 3.3 Dual-use: obstacle avoidance + surveillance

The same OAK-D Lite feeds two pipelines:

```
OAK-D Lite
  ├── Depth stream → occupancy grid → FIRI corridors → MINCO (obstacle avoidance)
  └── RGB stream → YOLO inference (threat detection / surveillance)
```

The Myriad X VPU can run YOLO11s on-camera at ~10fps, offloading the RPi entirely for perception. The RPi focuses on MINCO planning.

### 3.4 GPS-denied readiness

The OAK-D Lite's stereo + IMU enables VIO (Visual-Inertial Odometry) with no additional hardware:

| Phase | Localization | Hardware needed |
|-------|-------------|----------------|
| Demo (June 2026) | GPS (Pixhawk module) | None new |
| Post-demo Phase 2 | GPS + VIO fusion | None new — same OAK-D Lite |
| Phase 3 | GPS-denied (VIO only) | Optional: UWB modules ($10-15/drone) |

The MINCO optimizer is SE(3)-equivariant — it works identically regardless of whether position comes from GPS or VIO.

---

## 4. Codebase changes

### 4.1 Modules to delete / deprecate

```
DEPRECATE (move to src/_legacy/, do not delete yet):
  src/single_drone/obstacle_avoidance/apf_3d.py          # Replaced by MINCO
  src/single_drone/obstacle_avoidance/tactical_planner.py # A* replaced by RRT+FIRI
  src/single_drone/sensors/lidar_3d.py                    # Replaced by depth_camera.py
  src/single_drone/sensors/real_lidar.py                  # Replaced by depth_camera.py
  src/single_drone/world_model/lidar_polar_grid.py        # LiDAR-specific
  src/single_drone/world_model/lidar_world_model.py       # LiDAR-specific
  src/single_drone/world_model/lidar_dataset_io.py        # LiDAR-specific
  src/simulation/physics/servo_lidar_model.py              # No servo
  src/simulation/physics/lidar_noise_model.py              # Replaced by depth noise
  src/swarm/boids/                                         # Replaced by trajectory broadcast

KEEP (refactor interface):
  src/single_drone/obstacle_avoidance/avoidance_manager.py # Refactor to orchestrate MINCO
  src/single_drone/obstacle_avoidance/hardware_protection.py # HPL stays, input changes
  src/single_drone/obstacle_avoidance/urban_geofence.py    # Geofence stays
  src/single_drone/flight_control/flight_controller.py     # Trajectory tracking mode added
  src/single_drone/flight_control/mavsdk_interface.py      # Position setpoint interface
  src/swarm/coordination/regiment_coordinator.py           # Strategic layer stays
  src/swarm/cbba/                                          # Task allocation stays
  src/swarm/formation/                                     # Formation as MINCO waypoint constraints

KEEP (unchanged):
  src/response/mission_policy.py
  src/gcs/
  src/surveillance/
  src/communication/
  src/simulation/physics/battery_model.py                  # Reuse in Rig 5
  src/simulation/physics/wind_model.py                     # Reuse in Rig 6
  src/simulation/physics/flight_dynamics.py                # Reuse for flatness validation
  src/simulation/physics/imu_model.py                      # Reuse for VIO testing
  src/simulation/physics/gps_model.py                      # Reuse for GPS-denied testing
  src/simulation/metrics_collector.py                      # Extend for MINCO metrics
  src/simulation/scenario_executor.py                      # Refactor to use MINCO pipeline
```

### 4.2 New modules: MINCO planning core

```
src/single_drone/planning/
├── __init__.py
├── minco.py                  # MINCO trajectory representation
│                              #   - Piece: single polynomial segment (degree 2s+1)
│                              #   - Trajectory: sequence of Pieces with time allocation
│                              #   - evaluate(t, derivative_order) → position/vel/acc/jerk
│                              #   - gradient w.r.t. waypoints and times
│
├── gcopter.py                # L-BFGS trajectory optimizer
│                              #   - setup(weight_T, ini_state, fin_state, corridors, ...)
│                              #   - optimize(traj, rel_cost_tol) → converged Trajectory
│                              #   - smooth-map penalty functionals for each constraint
│                              #   - Uses scipy.optimize.minimize(method='L-BFGS-B')
│
├── corridor_generator.py     # FIRI safe flight corridor generation
│                              #   - firi(seed_point, obstacle_points, bounds) → Polytope
│                              #   - convex_cover(route, surface_pts, ...) → list[Polytope]
│                              #   - shortcut(polytopes) → pruned list
│                              #   - Polytope = namedtuple('Polytope', ['A', 'b'])  (Ax <= b)
│
├── sfc_gen.py                # Path search + corridor utilities
│                              #   - plan_path_rrt(start, goal, voxel_map, timeout) → route
│                              #   - Contains RRT implementation on VoxelMap
│
├── voxel_map.py              # Binary 3D occupancy grid with dilation
│                              #   - set_occupied(point)
│                              #   - query(point) → 0 (free) or 1 (occupied)
│                              #   - dilate(radius_voxels)
│                              #   - get_surface_points() → list of boundary voxel centers
│
├── flatness.py               # Differential flatness: (p,v,a,j) → (thrust, quat, omega)
│                              #   - Accounts for nonlinear drag
│                              #   - Used for dynamic feasibility checking
│
└── trajectory_tracker.py     # Converts MINCO trajectory to flight controller setpoints
                               #   - sample(t) → (position, velocity, acceleration)
                               #   - Interfaces with FlightController / MAVSDK
```

### 4.3 New modules: depth camera

```
src/single_drone/sensors/
├── depth_camera.py           # Depth camera driver (OAK-D Lite / RealSense)
│                              #   - get_depth_image() → np.ndarray (H x W, float32, meters)
│                              #   - get_rgb_image() → np.ndarray (H x W x 3, uint8)
│                              #   - get_pointcloud() → np.ndarray (N x 3)
│                              #   - Simulated mode: loads depth from file/Blender render
│                              #   - Real mode: OAK-D DepthAI SDK
│
└── depth_noise_model.py      # Stereo depth noise model for simulation
                               #   - Noise increases with distance (quadratic for stereo)
                               #   - Range reduction for outdoor/fog conditions
                               #   - Missing pixels at occlusion boundaries
```

### 4.4 New modules: swarm trajectory broadcast

```
src/swarm/
├── trajectory_broadcast.py   # MINCO trajectory serialization + broadcast
│                              #   - serialize(trajectory) → bytes (~0.5KB)
│                              #   - deserialize(bytes) → Trajectory
│                              #   - SwarmBroadcaster: sends own trajectory, receives neighbors'
│                              #   - Simulated mode: Python queue with configurable latency/loss
│                              #   - Real mode: UDP over WiFi mesh
│
└── swarm_penalty.py          # Inter-drone collision penalty for MINCO
                               #   - compute_swarm_cost(own_traj, neighbor_trajs, clearance)
                               #   - Returns cost + gradient w.r.t. own waypoints
                               #   - Ellipsoidal distance (shorter z-axis for downwash)
```

### 4.5 New modules: validation rigs

```
src/validation/
├── __init__.py
├── obstacle_gen.py            # Procedural obstacle generation
│                               #   - perlin_map(seed, size, density) → point cloud
│                               #   - random_pillars(n, size) → point cloud
│                               #   - urban_canyon(width, height, gap) → point cloud
│                               #   - load_blender_scene(path) → point cloud
│
├── broadcast_channel.py       # Simulated WiFi mesh
│                               #   - SimChannel(latency_ms, packet_loss_pct, bandwidth_kbps)
│                               #   - send(drone_id, data) / receive(drone_id) → data | None
│
├── vio_drift_model.py         # VIO drift injection
│                               #   - VIODrift(sigma_walk, bias_rate, jump_prob, jump_mag)
│                               #   - step(dt) → accumulated_drift_vector
│                               #   - correct(observed_error) → apply drift correction
│
├── motor_model.py             # Thrust degradation
│                               #   - MotorWear(efficiency_0, degradation_rate_per_hour)
│                               #   - thrust_max(flight_hours) → effective max thrust
│
├── rig1_corridor_benchmark.py
│   # Single-drone MINCO performance benchmark
│   # Inputs: obstacle density sweep, map size, drone params
│   # Outputs: t_rrt, t_firi, t_minco, t_total, clearance, success rate
│   # Potato test: same benchmark in Docker ARM container
│
├── rig2_swarm_avoidance.py
│   # Multi-drone collision avoidance + scaling to 50
│   # Inputs: drone count, scenario (patrol/crossing/converge), comms params
│   # Outputs: d_min_inter, replan latency, broadcast bandwidth, near misses
│
├── rig3_vio_perimeter.py
│   # GPS-denied drift + perimeter fencing
│   # Inputs: drift model params, correction on/off, drone count
│   # Outputs: drift magnitude, perimeter deviation, time to failure
│
├── rig4_mission_response.py
│   # Threat detect → inspect → regroup
│   # Inputs: threat position, patrol state, CBBA config
│   # Outputs: detect-to-replan time, coverage gap, regroup time
│
├── rig5_endurance.py
│   # Battery relay, motor degradation, drone loss
│   # Uses existing: src/simulation/physics/battery_model.py
│   # Inputs: mission duration, battery config, failure schedule
│   # Outputs: coverage timeline, relay handoff time, degraded thrust feasibility
│
├── rig6_disturbance.py
│   # Wind, fog, sensor failure
│   # Uses existing: src/simulation/physics/wind_model.py
│   # Inputs: wind config, depth range reduction, sensor failure time
│   # Outputs: trajectory tracking error, corridor clearance, safe wind limit
│
├── metrics.py                 # Unified metrics collection
│                               #   - MetricsCollector: records per-tick and per-run metrics
│                               #   - export_json(path) / export_csv(path)
│                               #   - summary() → dict of aggregated stats
│
└── plots.py                   # Visualization dashboards
                                #   - plot_corridor_benchmark(results) → matplotlib figure
                                #   - plot_swarm_scaling(results) → scaling curves
                                #   - plot_vio_drift(results) → drift + correction timeline
                                #   - plot_endurance(results) → coverage over time
```

---

## 5. Validation framework

### 5.1 Design principle

Same as GCOPTER: **point clouds + optimizer + analytical evaluation + metrics.** No physics engine. No rendering in the loop. The optimizer is the system under test.

### 5.2 Rig 1: Corridor escape benchmark

**Question:** Given random obstacles of increasing density, can the pipeline find a path and how fast?

**Pipeline under test:** point cloud → VoxelMap → dilate → RRT → FIRI corridors → MINCO optimize

**Test matrix:**

| Scenario | Fill ratio | Map size | Purpose |
|----------|-----------|----------|---------|
| Open field | 0.05 | 50x50x5m | Baseline timing |
| Sparse forest | 0.15 | 50x50x5m | Easy corridors |
| Dense forest | 0.30 | 50x50x5m | GCOPTER default |
| Urban canyon | 0.45 | 50x50x5m | Tight corridors |
| Worst case | 0.60 | 50x50x5m | Failure rate |
| Ganeshguri | Real scene | Real scale | Demo environment |

**Metrics:** t_rrt (ms), t_firi (ms), t_minco (ms), t_total (ms), clearance_min (m), v_max (m/s), thrust_max (m/s2), tilt_max (rad), cost_J, success (bool), mem_peak_kb

**Potato test:** Same benchmark in `docker run --platform linux/arm64 --cpus=1 --memory=512m` to estimate RPi 5 performance.

### 5.3 Rig 2: Swarm collision avoidance (scaling to 50)

**Question:** With N drones broadcasting MINCO trajectories, how close do they get and how fast do they replan?

**Scenarios:** head-on (2 drones), crossing (3), converge (3), patrol (3/6/12/25/50), comms delay (50-200ms), comms loss (30% drop)

**Metrics:** d_min_inter (m), d_mean_inter (m), near_misses, collisions, t_replan_swarm (ms), broadcast_bandwidth (kbps), network_congestion_pct

**Scaling assertion:** t_replan_per_agent should stay flat (O(k) where k = neighbor count, not O(n) total agents).

### 5.4 Rig 3: VIO drift + perimeter fencing

**Question:** With VIO drift injected, does the swarm maintain hex patrol perimeter? At what drift rate does it break?

**VIO drift model:** random walk (sigma=0.02 m/sqrt(s)) + systematic bias (0.01 m/s) + occasional jump (0.3m, P=0.005/s)

**Inter-agent drift correction:** compare broadcast trajectory prediction with depth-detected position of neighbor. Feed discrepancy into Kalman filter.

**Experiments:** baseline (no drift), drift only (correction OFF), corrected (correction ON), aggressive drift (5x rate), scale (3/6 drones)

**Metrics:** drift_magnitude (m), drift_corrected (m), perimeter_deviation (m), sector_coverage_pct, true_obstacle_clearance (m), true_inter_drone_dist (m), time_to_failure (s)

### 5.5 Rig 4: Mission response

**Question:** When a threat is detected, how fast can one drone break off to inspect while others close the coverage gap?

**Scenario:** 3 drones in hex patrol → threat detected at t=30s → CBBA re-auction → inspect → regroup

**Metrics:** t_detect_to_replan (ms), t_coverage_gap (s), coverage_pct_during (%), t_regroup (s)

### 5.6 Rig 5: Endurance + attrition

**Question:** Over a 30-minute mission with battery drain and component degradation, does coverage persist?

**Models (reuse existing):**
- Battery: `src/simulation/physics/battery_model.py` (Li-Po discharge, temp derating, RTL trigger)
- Motor: new `motor_model.py` (linear thrust degradation, ~2%/flight-hour)

**Scenarios:** normal patrol (30min), battery relay (3+1 drones), drone down (hard fail at t=15min), graceful degradation (80% motor efficiency), cascading failure

**Metrics:** mission_duration (s), coverage_pct_timeline, coverage_gap_max (s), battery_consumed (Wh), relay_handoff_time (s), redistribution_time (s), degraded_thrust_ratio

### 5.7 Rig 6: Environmental disturbance

**Question:** How robust is the system to wind gusts, fog, and sensor failure?

**Models (reuse existing):**
- Wind: `src/simulation/physics/wind_model.py` (base wind + gusts + Perlin turbulence)
- Sensor: new depth range/noise scaling in `depth_noise_model.py`

**Scenarios:** calm (<1 m/s), breezy (3+5 gust), windy (5+8 gust), foggy (3m depth range), rain (5m + 2x noise), sensor failure (depth camera dies → GPS-only RTL)

**Metrics:** trajectory_tracking_error (m), corridor_clearance_min (m), safe_wind_limit (m/s), depth_range_threshold (m)

---

## 6. Hardware BOM (per drone, demo)

| Component | Model | Cost | Weight |
|-----------|-------|------|--------|
| Frame | Off-shelf 450mm | $30 | 300g |
| Flight controller | Pixhawk Mini | $80 | 20g |
| Companion computer | Raspberry Pi 5 (4GB) | $60 | 45g |
| Depth camera | OAK-D Lite | $150 | 61g |
| GPS module | Pixhawk GPS (u-blox) | $30 | 15g |
| Battery | 3S 2200mAh LiPo | $20 | 180g |
| ESC + motors | Cheap brushless kit | $40 | 120g |
| WiFi | Pi onboard WiFi | $0 | 0g |
| Props + misc | Standard 10" | $10 | 20g |
| **Total** | | **$420** | **~761g** |

Upgrade path: Jetson Orin Nano ($200, replaces RPi) if FIRI is too heavy on Pi.

---

## 7. Implementation order

### Phase 0: Core MINCO (Week 1)

Build the trajectory optimization engine. No swarm, no sensors, no validation rigs. Just the math.

**Task 0.1: `src/single_drone/planning/voxel_map.py`**
- Binary 3D occupancy grid with hash-based sparse storage
- `set_occupied(point)`, `query(point)`, `dilate(radius)`, `get_surface_points()`
- Port of GCOPTER's `voxel_map.hpp`
- Test: create grid, set obstacles, query free/occupied, dilate, verify

**Task 0.2: `src/single_drone/planning/sfc_gen.py`**
- RRT path search on VoxelMap
- `plan_path_rrt(start, goal, voxel_map, timeout)` → list of 3D waypoints
- Test: find path in simple obstacle field, verify all waypoints are in free space

**Task 0.3: `src/single_drone/planning/corridor_generator.py`**
- FIRI convex cover: given route + surface points → list of Polytopes
- Shortcut: remove redundant polytopes
- Each Polytope is (A, b) where Ax <= b defines the halfspaces
- Test: generate corridors around a known route, verify they contain the route and don't contain obstacles

**Task 0.4: `src/single_drone/planning/minco.py`**
- MINCO trajectory representation: Piece (single segment) + Trajectory (sequence)
- `evaluate(t, derivative_order)` → np.ndarray
- Gradient computation w.r.t. waypoints q and times T
- Matrix M_{s+1} construction for the polynomial basis
- Test: construct trajectory from waypoints, evaluate at sample points, verify smoothness

**Task 0.5: `src/single_drone/planning/gcopter.py`**
- L-BFGS optimizer wrapping MINCO
- Smooth-map penalty functionals for corridor containment, velocity, thrust, tilt
- `setup(weight_T, ini_state, fin_state, corridors, ...)` + `optimize(traj, tol)`
- Uses `scipy.optimize.minimize(method='L-BFGS-B')`
- Test: optimize trajectory through known corridors, verify constraints satisfied

**Task 0.6: `src/single_drone/planning/flatness.py`**
- Differential flatness map: (pos, vel, acc, jerk) → (thrust, quaternion, body_rate)
- With configurable drag coefficients
- Test: evaluate on known trajectory, verify thrust within [f_min, f_max]

**Exit criteria:** Run the GCOPTER example end-to-end in Python: generate random obstacles → RRT → FIRI → MINCO → evaluate trajectory → print timing + dynamic quantities. Verify against GCOPTER's published results.

### Phase 1: Validation Rig 1 (Week 1-2, overlaps with Phase 0)

**Task 1.1: `src/validation/obstacle_gen.py`**
- Perlin noise map generator (port of mockamap)
- Random pillars, urban canyon presets
- Blender scene loader (point cloud from .ply/.pcd)

**Task 1.2: `src/validation/rig1_corridor_benchmark.py`**
- Benchmark harness: sweep obstacle densities, run N trials, collect metrics
- Output JSON + summary table
- Potato test: Docker ARM runner

**Exit criteria:** Produce the density vs replan-latency plot. Verify t_total < 50ms at 0.30 density on Mac.

### Phase 2: Swarm layer (Week 2)

**Task 2.1: `src/swarm/trajectory_broadcast.py`**
- Serialize/deserialize MINCO trajectory to bytes
- SimBroadcaster: Python queue with configurable latency and packet loss

**Task 2.2: `src/swarm/swarm_penalty.py`**
- Compute J_w penalty: predicted inter-drone distance along trajectory
- Gradient w.r.t. own waypoints
- Ellipsoidal distance (compressed z-axis for downwash)

**Task 2.3: `src/validation/rig2_swarm_avoidance.py`**
- Multi-agent simulation loop: N MINCO instances, broadcast channel, penalty-based avoidance
- Scaling sweep: 3 to 50 agents

**Exit criteria:** 3-drone patrol with zero collisions. Scaling plot shows flat replan time to 50 agents.

### Phase 3: VIO + perimeter fencing (Week 2-3)

**Task 3.1: `src/validation/vio_drift_model.py`**
- Random walk + bias + jump drift model
- Inter-agent correction algorithm (predict-observe-filter)

**Task 3.2: `src/validation/rig3_vio_perimeter.py`**
- 3-drone hex patrol with drift injection
- Correction ON/OFF experiments
- Time-to-failure measurement

**Exit criteria:** With correction ON, perimeter maintained for 30+ minutes at standard drift rate.

### Phase 4: Mission + endurance + disturbance (Week 3)

**Task 4.1: `src/validation/rig4_mission_response.py`**
- Threat detection → CBBA re-auction → inspect → regroup
- Interfaces with existing `src/response/mission_policy.py` and `src/swarm/cbba/`

**Task 4.2: `src/validation/rig5_endurance.py`**
- Battery relay, motor degradation, drone loss
- Reuses existing `src/simulation/physics/battery_model.py`

**Task 4.3: `src/validation/rig6_disturbance.py`**
- Wind + fog + sensor failure
- Reuses existing `src/simulation/physics/wind_model.py`

**Exit criteria:** All 6 rigs produce clean JSON metrics. Dashboard plots generated.

### Phase 5: Integration + refactor (Week 3-4)

**Task 5.1: Refactor `avoidance_manager.py`**
- Replace APF+A* orchestration with MINCO pipeline
- depth_camera → voxel_map → corridor_generator → gcopter → trajectory_tracker

**Task 5.2: Refactor `scenario_executor.py`**
- Replace Boids+APF motion with MINCO trajectory evaluation
- Wire depth camera simulation (load depth from file per tick)

**Task 5.3: Refactor `flight_controller.py`**
- Add trajectory tracking mode: follow MINCO setpoints via MAVSDK
- Existing waypoint mode stays for backwards compatibility

**Task 5.4: Move deprecated modules to `src/_legacy/`**

**Task 5.5: Update `CLAUDE.md`, `STATE.md`, `Roadmap.md`**

**Exit criteria:** `python scripts/run_scenario.py --scenario S01` runs with MINCO pipeline end-to-end. All existing tests that don't depend on APF/Boids still pass.

---

## 8. Dependencies

### Python packages (add to requirements.txt)

```
numpy>=1.24
scipy>=1.11         # L-BFGS-B optimizer, spatial algorithms
```

No new dependencies beyond what's already in the project. The entire MINCO implementation uses NumPy + SciPy only.

### Optional (for validation plots)

```
matplotlib>=3.7     # Already in project
plotly>=5.0         # Interactive dashboards (optional)
```

### Optional (for real OAK-D Lite hardware)

```
depthai>=2.25       # Luxonis DepthAI SDK (only needed for real camera, not simulation)
```

---

## 9. Verification commands

```bash
# Phase 0: Core MINCO works
python -m pytest tests/test_minco.py -v

# Phase 1: Corridor benchmark
python -m src.validation.rig1_corridor_benchmark --densities 0.05,0.15,0.30,0.45 --runs 50

# Phase 2: Swarm avoidance
python -m src.validation.rig2_swarm_avoidance --drones 3,6,12,25,50 --scenario patrol

# Phase 3: VIO perimeter
python -m src.validation.rig3_vio_perimeter --drones 3 --drift-rate 0.02 --correction on,off

# Phase 4: Mission + endurance + disturbance
python -m src.validation.rig4_mission_response
python -m src.validation.rig5_endurance --duration 1800 --failures drone_down@900
python -m src.validation.rig6_disturbance --wind 5.0 --depth-range 3.0

# Phase 5: Integration
python scripts/run_scenario.py --scenario S01
python -m pytest tests/ -q

# Potato test (RPi 5 estimate)
docker run --platform linux/arm64 --cpus=1 --memory=512m \
  -v $(pwd):/app python:3.11-slim \
  bash -c "pip install numpy scipy && python -m src.validation.rig1_corridor_benchmark --runs 10"
```

---

## 10. Immediate next steps for Claude Code CLI

Open a new Claude Code session and give it these instructions in order:

### Step 1: Core MINCO

```
Read docs/MINCO_PIVOT.md sections 2 and 4.2. Implement Phase 0 (Tasks 0.1 through 0.6)
in src/single_drone/planning/. The reference implementation is GCOPTER
(github.com/ZJU-FAST-Lab/GCOPTER) — read the C++ headers in gcopter/include/gcopter/
for the algorithm, but write clean-room Python using NumPy + SciPy only. Start with
voxel_map.py, then sfc_gen.py, then corridor_generator.py, then minco.py, then gcopter.py,
then flatness.py. Write pytest tests in tests/test_minco.py for each module. The exit
criterion is: generate a random obstacle point cloud, run the full pipeline
(RRT → FIRI → MINCO → evaluate), and print timing + trajectory stats.
```

### Step 2: Rig 1

```
Read docs/MINCO_PIVOT.md section 5.2. Implement Rig 1 (Tasks 1.1-1.2). The obstacle
generator should support Perlin noise maps (like GCOPTER's mockamap). The benchmark
harness should sweep obstacle densities [0.05, 0.15, 0.30, 0.45, 0.60], run 50 trials
each, and output a JSON results file + printed summary table with t_rrt, t_firi, t_minco,
t_total, success_rate, clearance_min for each density.
```

### Step 3: Swarm layer

```
Read docs/MINCO_PIVOT.md sections 2.4, 4.4, and 5.3. Implement Phase 2 (Tasks 2.1-2.3).
The swarm penalty J_w uses ellipsoidal distance (compressed z-axis). The broadcast channel
simulator should support configurable latency and packet loss. Rig 2 should run 3-drone
patrol, then scale to 50. Output: min inter-drone distance, replan latency, scaling plot.
```

### Step 4: VIO + perimeter

```
Read docs/MINCO_PIVOT.md sections 3.4, 5.4. Implement Phase 3 (Tasks 3.1-3.2). The VIO
drift model uses random walk (sigma=0.02) + bias (0.01 m/s) + jumps (0.3m, P=0.005/s).
Inter-agent drift correction compares broadcast trajectory predictions with simulated
depth-detected positions. Rig 3 should show time-to-failure with correction OFF vs ON.
```

### Step 5: Remaining rigs + integration

```
Read docs/MINCO_PIVOT.md sections 5.5-5.7 and 4.1. Implement Phase 4 (rigs 4-6) reusing
existing battery_model.py and wind_model.py. Then implement Phase 5: refactor
avoidance_manager.py, scenario_executor.py, and flight_controller.py to use the MINCO
pipeline. Move deprecated APF/Boids/LiDAR modules to src/_legacy/. Update STATE.md.
```

---

## 11. Success criteria

The MINCO pivot is complete when:

1. `python -m pytest tests/test_minco.py` passes — core optimizer works
2. Rig 1 shows t_total < 50ms at density 0.30 on Mac — fast enough for real-time
3. Rig 2 shows zero collisions at 3 drones, flat scaling to 50 — swarm works
4. Rig 3 shows perimeter maintained 30+ minutes with VIO drift correction ON — GPS-denied ready
5. Rig 4 shows coverage gap < 5s during threat inspection — mission response works
6. Rig 5 shows continuous coverage over 30min with battery relay — endurance works
7. Rig 6 shows safe operation up to 5 m/s wind — weather tolerance validated
8. `python scripts/run_scenario.py --scenario S01` runs with MINCO pipeline — integration complete
9. No APF/Boids/LiDAR code in the active import path — clean deprecation
