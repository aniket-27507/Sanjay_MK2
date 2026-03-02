# Project Sanjay MK2
> **Author**: Prathamesh Hiwarkar

Project Sanjay MK2 is a comprehensive drone surveillance swarm system that leverages a two-tier architecture (Alpha and Beta drones) for advanced area coverage, multi-sensor anomaly detection, and autonomous threat confirmation.

## Features

- **Two-Tier Swarm Architecture**: 
  - **Alpha Drones**: High-altitude (65m) surveillance drones equipped with RGB and Thermal cameras, paired with AI depth estimation, designed to scan large areas efficiently.
  - **Beta Drones**: Low-altitude (25m) interceptor drones triggered automatically to confirm threats with high-detail visual cameras.
- **Sensor Fusion Pipeline**: Cross-references observations from simulated RGB, Thermal, and Depth sensors to build boosted confidence bounding boxes for detected objects.
- **Change Detection**: Compares live fused observations against a procedurally generated baseline terrain map to identify anomalies (new objects, missing objects, or thermal anomalies).
- **Physics & Simulation**: Full MAVSDK-compatible interface backed by either MuJoCo physics simulation or NVIDIA Isaac Sim for photorealistic sensors and physically accurate flight dynamics.
- **Fault Injection & Auto-Redistribution**: Built-in runtime fault injection (motor failure, comms loss, battery drain) with automatic CBBA-based task redistribution among surviving swarm members.
- **WebSocket Visualization**: Real-time simulation backend streaming telemetry and detection data to an interactive 3D frontend.

## Installation & Setup

Please refer to the following guides for detailed setup instructions:
- [macOS/Linux Environment Setup](docs/INSTALLATION_SUMMARY.md)
- [NVIDIA Isaac Sim + WSL2 Hybrid Setup](docs/ISAAC_SIM_SETUP.md)

## Architecture

The system comprises over 8 major packages:
- `core`: Configurations, typing (Vector3, FlightMode), and singleton config managers.
- `communication`: Mesh networking and swarm state synchronization.
- `integration`: ROS 2 to Isaac Sim bridging for photorealistic testing.
- `simulation`: Core physics simulation via MuJoCo.
- `single_drone`: Flight controllers, Mavskd interfacing, and sensor models.
- `surveillance`: World model procedural generation, baseline map, change detection, sensor fusion, and threat lifecycle management.
- `swarm`: Flocking (boids), coordination, task allocation (CBBA), formation control, and fault injection.

Read the full architecture overview in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Usage

### Running the Python Simulation

Launch the real-time simulation server which executes the hexagonal coverage pattern and serves the WebSocket visualization:

```bash
python scripts/simulation_server.py
```

Open one of the HTML visualizers (e.g. `drone_visualization_live.html`) in your browser to watch the swarm operate.

### Running with Isaac Sim

Launch the ROS 2 Bridge (requires Isaac Sim to be running on Windows, and ROS 2 inside WSL2 / Docker):

```bash
python scripts/isaac_sim/launch_bridge.py
```

## Team Credits

- **Archishman Paul**: Algorithmic work (sensor fusion, flight control, physics sim, swarm algorithms, change detection, fault injection) & Infrastructure (Docker, WSL2, Isaac Sim, config scripts).
- **Aniket More**: Visualizations (HTML/JS), communication modules, test suites, and examples.
- **Prathamesh Hiwarkar**: Type system, data models, integration layer, surveillance wiring, and project documentation.
