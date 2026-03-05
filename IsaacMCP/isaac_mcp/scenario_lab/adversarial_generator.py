"""Adversarial scenario generation for robustness testing.

Extends the standard ScenarioGenerator with extreme parameter ranges,
correlated failures, and targeted stress testing to discover robot
failure modes that mild randomization would never find.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.scenario_lab.scenario_generator import GeneratedScenario


# Adversarial parameter ranges -- deliberately extreme to stress-test
ADVERSARIAL_PARAMETERS = {
    # Environmental extremes
    "floor_friction": {"min": 0.01, "max": 2.0},
    "gravity_scale": {"min": 0.3, "max": 2.5},
    "obstacle_count": {"min": 10, "max": 50},
    "terrain_type": {"choices": ["flat", "inclined", "rough", "steep_slope", "stairs", "uneven"]},
    "terrain_slope_deg": {"min": 15.0, "max": 45.0},
    "payload_mass": {"min": 5.0, "max": 50.0},
    "wind_force": {"min": 5.0, "max": 50.0},
    "wind_direction": {"choices": ["north", "south", "east", "west", "gusting"]},
    "fog_density": {"min": 0.3, "max": 1.0},
    "lighting": {"choices": ["dark", "strobe", "glare", "dusk", "nightvision"]},
    # Sensor degradation
    "camera_occlusion_pct": {"min": 10.0, "max": 80.0},
    "lidar_noise_scale": {"min": 0.3, "max": 2.0},
    "imu_drift_rate": {"min": 0.05, "max": 0.5},
    "sensor_noise_scale": {"min": 0.3, "max": 2.0},
    "gps_dropout_pct": {"min": 10.0, "max": 90.0},
    # Actuator failures
    "motor_torque_reduction_pct": {"min": 20.0, "max": 80.0},
    "joint_lock_probability": {"min": 0.05, "max": 0.5},
    "motor_delay_ms": {"min": 50.0, "max": 500.0},
    # Physics stress
    "collision_speed_mps": {"min": 5.0, "max": 30.0},
    "dynamic_mass_change_pct": {"min": 20.0, "max": 100.0},
}

# Predefined adversarial profiles -- correlated failure patterns
ADVERSARIAL_PROFILES: dict[str, dict[str, Any]] = {
    "sensor_blackout": {
        "description": "All sensors degraded simultaneously",
        "params": {
            "camera_occlusion_pct": 70.0,
            "lidar_noise_scale": 1.5,
            "imu_drift_rate": 0.3,
            "gps_dropout_pct": 80.0,
            "fog_density": 0.8,
        },
    },
    "motor_degradation": {
        "description": "Progressive motor failure under load",
        "params": {
            "motor_torque_reduction_pct": 60.0,
            "joint_lock_probability": 0.2,
            "motor_delay_ms": 200.0,
            "payload_mass": 30.0,
        },
    },
    "extreme_environment": {
        "description": "Harsh terrain with wind and poor visibility",
        "params": {
            "terrain_type": "steep_slope",
            "terrain_slope_deg": 35.0,
            "wind_force": 30.0,
            "fog_density": 0.6,
            "lighting": "dark",
            "floor_friction": 0.1,
        },
    },
    "physics_stress": {
        "description": "High-speed collisions with dynamic mass changes",
        "params": {
            "collision_speed_mps": 20.0,
            "dynamic_mass_change_pct": 80.0,
            "gravity_scale": 2.0,
            "obstacle_count": 30,
        },
    },
    "combined_failure": {
        "description": "Simultaneous sensor, motor, and environmental stress",
        "params": {
            "camera_occlusion_pct": 50.0,
            "lidar_noise_scale": 1.0,
            "motor_torque_reduction_pct": 40.0,
            "wind_force": 20.0,
            "terrain_type": "rough",
            "floor_friction": 0.2,
            "payload_mass": 20.0,
        },
    },
}


@dataclass(slots=True)
class AdversarialScenario:
    """A scenario designed to stress-test a robot."""
    scenario_id: str
    base_scenario_id: str
    profile: str  # Name of the adversarial profile or "random"
    parameters: dict[str, Any] = field(default_factory=dict)
    fault_sequence: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "high"  # low | medium | high | extreme

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "base_scenario_id": self.base_scenario_id,
            "profile": self.profile,
            "parameters": self.parameters,
            "fault_sequence": self.fault_sequence,
            "severity": self.severity,
        }


class AdversarialGenerator:
    """Generate adversarial scenarios with extreme parameters and correlated failures."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate_from_profile(
        self,
        base_scenario_id: str,
        profile_name: str,
        scenario_index: int = 0,
        additional_randomization: bool = True,
    ) -> AdversarialScenario:
        """Generate a scenario from a predefined adversarial profile.

        If additional_randomization is True, adds random perturbation to
        the profile parameters.
        """
        profile = ADVERSARIAL_PROFILES.get(profile_name)
        if profile is None:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(ADVERSARIAL_PROFILES)}")

        params = dict(profile["params"])

        if additional_randomization:
            # Add random perturbation to each numeric parameter
            for key, value in list(params.items()):
                if isinstance(value, (int, float)):
                    perturbation = self._rng.gauss(0, abs(value) * 0.1)
                    params[key] = round(value + perturbation, 4)

        scenario_id = f"{base_scenario_id}_adv_{profile_name}_{scenario_index}"
        return AdversarialScenario(
            scenario_id=scenario_id,
            base_scenario_id=base_scenario_id,
            profile=profile_name,
            parameters=params,
            fault_sequence=self._generate_fault_sequence(params),
            severity="extreme" if profile_name == "combined_failure" else "high",
        )

    def generate_random(
        self,
        base_scenario_id: str,
        scenario_index: int = 0,
        num_params: int = 5,
    ) -> AdversarialScenario:
        """Generate a fully randomized adversarial scenario.

        Selects `num_params` random adversarial parameters and generates
        extreme values for each.
        """
        available_params = list(ADVERSARIAL_PARAMETERS.keys())
        selected = self._rng.sample(available_params, min(num_params, len(available_params)))

        params: dict[str, Any] = {}
        for param in selected:
            spec = ADVERSARIAL_PARAMETERS[param]
            if "choices" in spec:
                params[param] = self._rng.choice(spec["choices"])
            elif "min" in spec and "max" in spec:
                lo, hi = float(spec["min"]), float(spec["max"])
                params[param] = round(self._rng.uniform(lo, hi), 4)

        scenario_id = f"{base_scenario_id}_adv_random_{scenario_index}"
        return AdversarialScenario(
            scenario_id=scenario_id,
            base_scenario_id=base_scenario_id,
            profile="random",
            parameters=params,
            fault_sequence=self._generate_fault_sequence(params),
            severity=self._estimate_severity(params),
        )

    def generate_campaign(
        self,
        base_scenario_id: str,
        count: int = 20,
        include_profiles: bool = True,
    ) -> list[AdversarialScenario]:
        """Generate a campaign of adversarial scenarios.

        If include_profiles is True, generates one scenario per profile
        then fills the rest with random scenarios.
        """
        scenarios: list[AdversarialScenario] = []
        idx = 0

        if include_profiles:
            for profile_name in ADVERSARIAL_PROFILES:
                if idx >= count:
                    break
                scenarios.append(
                    self.generate_from_profile(base_scenario_id, profile_name, idx)
                )
                idx += 1

        while idx < count:
            scenarios.append(self.generate_random(base_scenario_id, idx))
            idx += 1

        return scenarios

    def _generate_fault_sequence(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate a time-sequenced fault injection plan from parameters."""
        faults: list[dict[str, Any]] = []
        time_offset = 0.0

        # Map parameters to fault injection commands
        if "motor_torque_reduction_pct" in params:
            faults.append({
                "time_s": time_offset,
                "fault_type": "motor_degradation",
                "params": {"reduction_pct": params["motor_torque_reduction_pct"]},
            })
            time_offset += 2.0

        if "camera_occlusion_pct" in params:
            faults.append({
                "time_s": time_offset,
                "fault_type": "sensor_noise",
                "params": {"camera_occlusion": params["camera_occlusion_pct"]},
            })
            time_offset += 1.0

        if "lidar_noise_scale" in params:
            faults.append({
                "time_s": time_offset,
                "fault_type": "sensor_noise",
                "params": {"lidar_noise": params["lidar_noise_scale"]},
            })
            time_offset += 1.0

        if "imu_drift_rate" in params:
            faults.append({
                "time_s": time_offset,
                "fault_type": "sensor_noise",
                "params": {"imu_drift": params["imu_drift_rate"]},
            })
            time_offset += 1.0

        if "wind_force" in params:
            faults.append({
                "time_s": time_offset,
                "fault_type": "wind_gust",
                "params": {"force": params["wind_force"], "direction": params.get("wind_direction", "north")},
            })

        if "joint_lock_probability" in params:
            faults.append({
                "time_s": time_offset + 5.0,
                "fault_type": "joint_lock",
                "params": {"probability": params["joint_lock_probability"]},
            })

        return faults

    def _estimate_severity(self, params: dict[str, Any]) -> str:
        """Estimate the severity level of an adversarial scenario."""
        score = 0
        # Count how many extreme parameters are present
        if params.get("gravity_scale", 1.0) > 1.8 or params.get("gravity_scale", 1.0) < 0.5:
            score += 2
        if params.get("motor_torque_reduction_pct", 0) > 50:
            score += 2
        if params.get("camera_occlusion_pct", 0) > 60:
            score += 1
        if params.get("wind_force", 0) > 25:
            score += 1
        if params.get("terrain_slope_deg", 0) > 30:
            score += 1
        if params.get("obstacle_count", 0) > 30:
            score += 1

        if score >= 5:
            return "extreme"
        elif score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        return "low"

    @staticmethod
    def list_profiles() -> list[dict[str, str]]:
        """List available adversarial profiles."""
        return [
            {"name": name, "description": profile["description"]}
            for name, profile in ADVERSARIAL_PROFILES.items()
        ]
