"""Generate fix proposals from simulation diagnoses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FixProposal:
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    kit_script: str = ""
    risk_level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "parameters": self.parameters,
            "kit_script": self.kit_script,
            "risk_level": self.risk_level,
        }


# Mapping from issue types to parameterized fix templates
_FIX_TEMPLATES: dict[str, dict[str, Any]] = {
    "robot_fell": {
        "description": "Adjust robot center of mass and reduce joint torques",
        "kit_script_template": (
            "import omni.isaac.core.utils.prims as prim_utils\n"
            "# Reduce max joint effort to stabilize robot\n"
            "prim = prim_utils.get_prim_at_path('{robot_path}')\n"
            "# Apply mass distribution adjustment\n"
            "print('Fix applied: reduced joint torques for {robot_name}')\n"
        ),
        "risk_level": "medium",
    },
    "physics_instability": {
        "description": "Reduce physics timestep and simplify collision meshes",
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "settings.set('/physics/timeStepsPerSecond', {timestep})\n"
            "print('Fix applied: physics timestep set to {timestep}')\n"
        ),
        "risk_level": "medium",
        "default_params": {"timestep": 120},
    },
    "excessive_velocity": {
        "description": "Apply velocity damping to affected bodies",
        "kit_script_template": (
            "import omni.isaac.core.utils.prims as prim_utils\n"
            "# Increase linear damping\n"
            "print('Fix applied: increased velocity damping for {robot_name}')\n"
        ),
        "risk_level": "low",
    },
    "gpu_memory": {
        "description": "Reduce render quality and sensor count",
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "settings.set('/rtx/post/aa/op', 0)\n"
            "settings.set('/rtx/directLighting/samplesPerPixel', 1)\n"
            "print('Fix applied: reduced render quality')\n"
        ),
        "risk_level": "low",
    },
}


class FixGenerator:
    """Generate fix proposals from diagnoses. Fixes are suggestions for human review."""

    def generate_fix_proposals(self, diagnosis_dict: dict[str, Any]) -> list[FixProposal]:
        """Generate fix proposals from a diagnosis dictionary."""
        proposals: list[FixProposal] = []
        issues = diagnosis_dict.get("issues", [])

        for issue in issues:
            proposal = self._proposal_for_issue(issue)
            if proposal is not None:
                proposals.append(proposal)

        # Deduplicate by description
        seen: set[str] = set()
        unique: list[FixProposal] = []
        for p in proposals:
            if p.description not in seen:
                seen.add(p.description)
                unique.append(p)

        return unique

    def _proposal_for_issue(self, issue: dict[str, Any]) -> FixProposal | None:
        description = issue.get("description", "")
        category = issue.get("category", "")

        # Match against known fix templates (normalize underscores to spaces for matching)
        desc_lower = description.lower()
        cat_lower = category.lower()
        for key, template in _FIX_TEMPLATES.items():
            key_spaced = key.replace("_", " ")
            if key in desc_lower or key_spaced in desc_lower or key in cat_lower:
                params = dict(template.get("default_params", {}))
                script = template.get("kit_script_template", "")
                return FixProposal(
                    description=template["description"],
                    parameters=params,
                    kit_script=script,
                    risk_level=template.get("risk_level", "medium"),
                )

        # Fallback: generate generic proposal from suggested fixes in diagnosis
        return None
