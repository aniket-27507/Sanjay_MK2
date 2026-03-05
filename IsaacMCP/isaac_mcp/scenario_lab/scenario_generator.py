"""Generate randomized simulation scenarios with parameter variations."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# Default randomization ranges
DEFAULT_RANDOMIZATION = {
    "floor_friction": {"min": 0.1, "max": 1.0},
    "gravity_scale": {"min": 0.8, "max": 1.2},
    "obstacle_count": {"min": 0, "max": 15},
    "terrain_type": {"choices": ["flat", "inclined", "rough"]},
    "payload_mass": {"min": 0.0, "max": 10.0},
    "sensor_noise_scale": {"min": 0.0, "max": 0.5},
    "lighting": {"choices": ["bright", "dim", "dark"]},
}


@dataclass(slots=True)
class GeneratedScenario:
    scenario_id: str
    base_scenario_id: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "base_scenario_id": self.base_scenario_id,
            "parameters": self.parameters,
        }


class ScenarioGenerator:
    """Generate randomized scenarios with parameter variations."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(
        self,
        base_scenario_id: str,
        randomization_config: dict[str, Any] | None = None,
        scenario_index: int = 0,
    ) -> GeneratedScenario:
        """Generate a randomized scenario with parameter variations.

        Args:
            base_scenario_id: The base scenario to randomize.
            randomization_config: Override default ranges. Each key maps to
                {"min": float, "max": float} for numeric or {"choices": [...]} for categorical.
            scenario_index: Index for generating unique scenario IDs.
        """
        config = dict(DEFAULT_RANDOMIZATION)
        if randomization_config:
            config.update(randomization_config)

        parameters: dict[str, Any] = {}
        for param, spec in config.items():
            if isinstance(spec, dict):
                if "choices" in spec:
                    parameters[param] = self._rng.choice(spec["choices"])
                elif "min" in spec and "max" in spec:
                    lo, hi = float(spec["min"]), float(spec["max"])
                    # Use integers for count-like params
                    if param in ("obstacle_count",):
                        parameters[param] = self._rng.randint(int(lo), int(hi))
                    else:
                        parameters[param] = round(self._rng.uniform(lo, hi), 4)

        scenario_id = f"{base_scenario_id}_rand_{scenario_index}"
        return GeneratedScenario(
            scenario_id=scenario_id,
            base_scenario_id=base_scenario_id,
            parameters=parameters,
        )

    def generate_batch(
        self,
        base_scenario_id: str,
        count: int,
        randomization_config: dict[str, Any] | None = None,
    ) -> list[GeneratedScenario]:
        """Generate multiple randomized scenarios."""
        return [
            self.generate(base_scenario_id, randomization_config, i)
            for i in range(count)
        ]
