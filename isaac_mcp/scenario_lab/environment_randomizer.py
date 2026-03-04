"""Generate Kit API Python scripts to set up randomized environments."""

from __future__ import annotations

from typing import Any

from isaac_mcp.scenario_lab.scenario_generator import GeneratedScenario


class EnvironmentRandomizer:
    """Generate Kit API Python scripts to set up a randomized environment."""

    def generate_kit_script(self, scenario: GeneratedScenario) -> str:
        """Generate a Kit API Python script to configure the randomized environment.

        The script sets physics properties, spawns obstacles, and adjusts
        environmental parameters based on the scenario's randomized parameters.
        """
        params = scenario.parameters
        lines: list[str] = [
            "# Auto-generated environment setup script",
            f"# Base scenario: {scenario.base_scenario_id}",
            f"# Scenario ID: {scenario.scenario_id}",
            "import omni.kit.commands",
            "import omni.physx",
            "",
        ]

        # Gravity scale
        if "gravity_scale" in params:
            scale = params["gravity_scale"]
            gravity = round(-9.81 * scale, 4)
            lines.append(f"# Set gravity (scale={scale})")
            lines.append(f"omni.physx.get_physx_interface().set_gravity(0.0, 0.0, {gravity})")
            lines.append("")

        # Floor friction
        if "floor_friction" in params:
            friction = params["floor_friction"]
            lines.append(f"# Set floor friction to {friction}")
            lines.append(f"omni.kit.commands.execute('ChangeProperty', prop_path='/World/GroundPlane.physics:dynamicFriction', value={friction})")
            lines.append(f"omni.kit.commands.execute('ChangeProperty', prop_path='/World/GroundPlane.physics:staticFriction', value={friction})")
            lines.append("")

        # Obstacle count
        if "obstacle_count" in params:
            count = int(params["obstacle_count"])
            if count > 0:
                lines.append(f"# Spawn {count} obstacles")
                lines.append("import random")
                lines.append(f"for i in range({count}):")
                lines.append("    x = random.uniform(-10.0, 10.0)")
                lines.append("    y = random.uniform(-10.0, 10.0)")
                lines.append("    omni.kit.commands.execute('CreatePrimWithDefaultXform',")
                lines.append("        prim_type='Cube',")
                lines.append("        prim_path=f'/World/Obstacle_{i}',")
                lines.append("        attributes={'xformOp:translate': (x, y, 0.5), 'xformOp:scale': (0.5, 0.5, 1.0)})")
                lines.append("")

        # Payload mass
        if "payload_mass" in params:
            mass = params["payload_mass"]
            if mass > 0:
                lines.append(f"# Set payload mass to {mass} kg")
                lines.append(f"omni.kit.commands.execute('ChangeProperty', prop_path='/World/Robot/Payload.physics:mass', value={mass})")
                lines.append("")

        # Sensor noise
        if "sensor_noise_scale" in params:
            noise = params["sensor_noise_scale"]
            lines.append(f"# Set sensor noise scale to {noise}")
            lines.append(f"# Note: Sensor noise configuration is application-specific")
            lines.append("")

        # Lighting
        if "lighting" in params:
            lighting = params["lighting"]
            intensity_map = {"bright": 5000.0, "dim": 1000.0, "dark": 200.0}
            intensity = intensity_map.get(lighting, 3000.0)
            lines.append(f"# Set lighting to {lighting} (intensity={intensity})")
            lines.append(f"omni.kit.commands.execute('ChangeProperty', prop_path='/World/Light.intensity', value={intensity})")
            lines.append("")

        lines.append("print('Environment setup complete')")
        return "\n".join(lines)
