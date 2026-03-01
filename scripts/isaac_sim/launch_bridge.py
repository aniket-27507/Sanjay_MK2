"""
Project Sanjay Mk2 - Isaac Sim Bridge Launcher
===============================================
Convenience launcher for the Isaac Sim ↔ ROS 2 bridge node.

Run inside WSL2 Docker:
    python scripts/isaac_sim/launch_bridge.py
    python scripts/isaac_sim/launch_bridge.py --config config/isaac_sim.yaml --drone alpha_0

Or via docker compose:
    docker compose --profile isaac up
"""

import argparse
import sys
import os

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Isaac Sim ↔ Sanjay MK2 bridge node"
    )
    parser.add_argument(
        "--config",
        default="config/isaac_sim.yaml",
        help="Path to isaac_sim.yaml (default: config/isaac_sim.yaml)",
    )
    parser.add_argument(
        "--drone",
        default=None,
        help="Launch bridge for a specific drone only (e.g. 'alpha_0')",
    )
    args = parser.parse_args()

    # Validate config exists
    config_path = os.path.join(PROJECT_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        print("Make sure you're running from the project root.")
        sys.exit(1)

    # Import and launch
    try:
        from src.integration.isaac_sim_bridge import (
            BridgeConfig,
            is_ros2_available,
        )
    except ImportError as e:
        print(f"ERROR: Cannot import bridge module: {e}")
        print("Ensure you're running from the project root with deps installed.")
        sys.exit(1)

    if not is_ros2_available():
        print(
            "═══════════════════════════════════════════════════════\n"
            "  ERROR: ROS 2 (rclpy) is not available.\n"
            "═══════════════════════════════════════════════════════\n"
            "\n"
            "  The bridge requires ROS 2 Humble. Run this inside\n"
            "  a ROS 2 Docker container on WSL2:\n"
            "\n"
            "    docker compose --profile isaac up\n"
            "\n"
            "  Or run directly in WSL2 with ROS 2 installed:\n"
            "\n"
            "    source /opt/ros/humble/setup.bash\n"
            "    python scripts/isaac_sim/launch_bridge.py\n"
            "═══════════════════════════════════════════════════════"
        )
        sys.exit(1)

    # Load config
    config = BridgeConfig.from_yaml(config_path)

    # Filter to specific drone if requested
    if args.drone:
        original_count = len(config.drones)
        config.drones = [d for d in config.drones if d.name == args.drone]
        if not config.drones:
            print(f"ERROR: Drone '{args.drone}' not found in config.")
            print(f"Available drones: {[d.name for d in BridgeConfig.from_yaml(config_path).drones]}")
            sys.exit(1)
        print(f"Filtered to drone '{args.drone}' (from {original_count})")

    # Launch
    import rclpy
    from src.integration.isaac_sim_bridge import IsaacSimBridgeNode

    print(f"Launching Isaac Sim bridge with {len(config.drones)} drone(s)...")
    print(f"Config: {config_path}")
    print(f"Fusion rate: {config.tick_rate_hz} Hz")
    print()

    rclpy.init()
    node = IsaacSimBridgeNode(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down bridge...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Bridge stopped.")


if __name__ == "__main__":
    main()
