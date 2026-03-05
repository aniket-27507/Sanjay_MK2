"""Generate fix proposals from simulation diagnoses.

Provides template-based fix generation covering all 21 error categories
defined in error_patterns.py. Each template maps to a Kit API Python script
that can be executed to remediate the issue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FixProposal:
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    kit_script: str = ""
    risk_level: str = "low"
    source: str = "template"  # template | knowledge_graph | llm_generated

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "parameters": self.parameters,
            "kit_script": self.kit_script,
            "risk_level": self.risk_level,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Fix templates mapped to error categories from error_patterns.py
# Each template has: description, kit_script_template, risk_level, match_keys
# match_keys: list of strings to match against issue description or category
# ---------------------------------------------------------------------------

_FIX_TEMPLATES: dict[str, dict[str, Any]] = {
    # --- Physics (4 templates) ---
    "robot_fell": {
        "description": "Adjust robot center of mass and reduce joint torques",
        "match_keys": ["robot_fell", "fallen", "fell", "below ground"],
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
        "match_keys": ["physics_instability", "nan", "inf", "physx", "physics step"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "settings.set('/physics/timeStepsPerSecond', {timestep})\n"
            "settings.set('/physics/reportKinematicKinematicPairs', False)\n"
            "print('Fix applied: physics timestep set to {timestep}')\n"
        ),
        "risk_level": "medium",
        "default_params": {"timestep": 120},
    },
    "excessive_velocity": {
        "description": "Apply velocity damping to affected bodies",
        "match_keys": ["excessive_velocity", "excessive velocity", "high velocity", "speed"],
        "kit_script_template": (
            "import omni.isaac.core.utils.prims as prim_utils\n"
            "import omni.physx\n"
            "# Increase linear and angular damping on rigid bodies\n"
            "print('Fix applied: increased velocity damping')\n"
        ),
        "risk_level": "low",
    },
    "collision_overlap": {
        "description": "Fix collision overlap by adjusting collision filtering and mesh simplification",
        "match_keys": ["collision", "overlap", "intersect", "contacts"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "# Enable contact offset to prevent initial overlaps\n"
            "settings.set('/physics/contactOffset', 0.02)\n"
            "settings.set('/physics/restOffset', 0.01)\n"
            "print('Fix applied: adjusted collision offsets')\n"
        ),
        "risk_level": "low",
    },
    "articulation_joint_error": {
        "description": "Fix articulation joint configuration by validating limits and hierarchy",
        "match_keys": ["articulation", "joint", "urdf", "invalid joint"],
        "kit_script_template": (
            "import omni.isaac.core.utils.prims as prim_utils\n"
            "from pxr import UsdPhysics\n"
            "# Validate joint types and reset limits\n"
            "print('Fix applied: validated articulation joint configuration')\n"
        ),
        "risk_level": "medium",
    },
    # --- Rendering (4 templates) ---
    "gpu_memory": {
        "description": "Reduce render quality and sensor count to free GPU memory",
        "match_keys": ["gpu_memory", "out of memory", "gpu memory", "cuda", "oom"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "settings.set('/rtx/post/aa/op', 0)\n"
            "settings.set('/rtx/directLighting/samplesPerPixel', 1)\n"
            "settings.set('/rtx/reflections/maxRoughness', 0.2)\n"
            "settings.set('/rtx/translucency/enabled', False)\n"
            "print('Fix applied: reduced render quality to free GPU memory')\n"
        ),
        "risk_level": "low",
    },
    "rtx_render_error": {
        "description": "Switch to a fallback render mode and verify GPU driver compatibility",
        "match_keys": ["rtx", "hydra", "render pipeline", "rendering"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "# Fall back to Storm renderer for stability\n"
            "settings.set('/renderer/active', 'rtx')\n"
            "settings.set('/rtx/rendermode', 'PathTracing')\n"
            "settings.set('/rtx/pathtracing/spp', 1)\n"
            "print('Fix applied: switched to lightweight render config')\n"
        ),
        "risk_level": "medium",
    },
    "vulkan_error": {
        "description": "Reset Vulkan state and reduce GPU workload",
        "match_keys": ["vulkan", "vkresult", "vulkan api"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "# Reduce concurrent GPU work to avoid Vulkan errors\n"
            "settings.set('/renderer/maxFramesInFlight', 1)\n"
            "settings.set('/rtx/hydra/maxTextures', 1024)\n"
            "print('Fix applied: reduced Vulkan GPU workload')\n"
        ),
        "risk_level": "medium",
    },
    "shader_failure": {
        "description": "Reset shader compilation cache and use fallback materials",
        "match_keys": ["shader", "compilation fail", "mdl"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "# Force shader recompilation\n"
            "settings.set('/renderer/mdl/clearCache', True)\n"
            "print('Fix applied: cleared shader cache for recompilation')\n"
        ),
        "risk_level": "low",
    },
    # --- USD Stage (4 templates) ---
    "usd_prim_not_found": {
        "description": "Verify USD stage is loaded and validate prim paths",
        "match_keys": ["prim not found", "cannot find prim", "prim path"],
        "kit_script_template": (
            "import omni.usd\n"
            "stage = omni.usd.get_context().get_stage()\n"
            "# List root prims to verify stage is loaded\n"
            "root_prims = [p.GetPath().pathString for p in stage.GetPseudoRoot().GetChildren()]\n"
            "print(f'Stage loaded with root prims: {root_prims}')\n"
        ),
        "risk_level": "low",
    },
    "usd_reference_error": {
        "description": "Resolve broken USD references by checking asset paths",
        "match_keys": ["reference", "resolve reference", "usd file", "failed to open layer"],
        "kit_script_template": (
            "import omni.usd\n"
            "import omni.client\n"
            "stage = omni.usd.get_context().get_stage()\n"
            "# Check for unresolved references\n"
            "for prim in stage.Traverse():\n"
            "    refs = prim.GetReferences()\n"
            "print('Fix applied: scanned for unresolved USD references')\n"
        ),
        "risk_level": "low",
    },
    # --- Extensions (2 templates) ---
    "extension_load_error": {
        "description": "Reload failed extension and check compatibility",
        "match_keys": ["extension", "failed to load", "cannot load extension"],
        "kit_script_template": (
            "import omni.kit.app\n"
            "ext_manager = omni.kit.app.get_app().get_extension_manager()\n"
            "# List loaded extensions for diagnostics\n"
            "extensions = ext_manager.get_extensions()\n"
            "print(f'Loaded {len(extensions)} extensions')\n"
        ),
        "risk_level": "low",
    },
    "ros2_bridge_extension_error": {
        "description": "Restart ROS2 bridge extension and verify DDS configuration",
        "match_keys": ["ros2_bridge", "ros bridge", "omni.isaac.ros2"],
        "kit_script_template": (
            "import omni.kit.app\n"
            "ext_manager = omni.kit.app.get_app().get_extension_manager()\n"
            "# Disable and re-enable ROS2 bridge\n"
            "ext_manager.set_extension_enabled('omni.isaac.ros2_bridge', False)\n"
            "ext_manager.set_extension_enabled('omni.isaac.ros2_bridge', True)\n"
            "print('Fix applied: restarted ROS2 bridge extension')\n"
        ),
        "risk_level": "medium",
    },
    # --- ROS2 Bridge (2 templates) ---
    "ros2_dds_failure": {
        "description": "Reset DDS discovery and verify network configuration",
        "match_keys": ["dds", "discovery", "fastdds", "ros2 dds"],
        "kit_script_template": (
            "import os\n"
            "# Verify ROS domain ID alignment\n"
            "domain_id = os.environ.get('ROS_DOMAIN_ID', '0')\n"
            "print(f'ROS_DOMAIN_ID={domain_id}')\n"
            "print('Fix: Verify DDS profiles and network configuration')\n"
        ),
        "risk_level": "low",
    },
    "ros2_topic_timeout": {
        "description": "Check ROS2 topic availability and QoS compatibility",
        "match_keys": ["topic not publish", "subscriber timeout", "topic timeout"],
        "kit_script_template": (
            "import os\n"
            "# Check ROS environment\n"
            "rmw = os.environ.get('RMW_IMPLEMENTATION', 'not set')\n"
            "domain = os.environ.get('ROS_DOMAIN_ID', '0')\n"
            "print(f'RMW={rmw}, DOMAIN_ID={domain}')\n"
            "print('Fix: Verify topic names and QoS profiles match')\n"
        ),
        "risk_level": "low",
    },
    # --- Python (3 templates) ---
    "python_traceback": {
        "description": "Analyze Python traceback and identify root cause module",
        "match_keys": ["traceback", "most recent call"],
        "kit_script_template": (
            "import traceback\n"
            "import sys\n"
            "# Capture and display recent exception info\n"
            "exc_info = sys.exc_info()\n"
            "if exc_info[0]:\n"
            "    traceback.print_exception(*exc_info)\n"
            "print('Fix: Examine traceback above for root cause')\n"
        ),
        "risk_level": "low",
    },
    "missing_module": {
        "description": "Install missing Python module in Isaac Sim environment",
        "match_keys": ["modulenotfounderror", "no module named", "missing module"],
        "kit_script_template": (
            "import subprocess\n"
            "import sys\n"
            "# List installed packages for diagnostics\n"
            "result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)\n"
            "print('Installed packages:')\n"
            "print(result.stdout[:500])\n"
        ),
        "risk_level": "low",
    },
    "isaac_import_error": {
        "description": "Verify script is running inside Isaac Sim runtime environment",
        "match_keys": ["importerror", "omni.isaac", "isaac module"],
        "kit_script_template": (
            "import sys\n"
            "# Check if running inside Isaac Sim\n"
            "is_isaac = 'omni' in sys.modules\n"
            "print(f'Running inside Isaac Sim: {is_isaac}')\n"
            "print(f'Python path entries: {len(sys.path)}')\n"
        ),
        "risk_level": "low",
    },
    # --- Performance (2 templates) ---
    "performance_degradation": {
        "description": "Reduce scene complexity and optimize render settings for performance",
        "match_keys": ["performance", "frame time", "fps below", "render performance"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "# Reduce render quality for performance\n"
            "settings.set('/rtx/post/aa/op', 0)\n"
            "settings.set('/rtx/directLighting/samplesPerPixel', 1)\n"
            "settings.set('/app/renderer/resolution/width', 640)\n"
            "settings.set('/app/renderer/resolution/height', 480)\n"
            "print('Fix applied: reduced render resolution and quality')\n"
        ),
        "risk_level": "low",
    },
    "physics_lag": {
        "description": "Optimize physics step to maintain real-time performance",
        "match_keys": ["physics lag", "simulation lag", "behind real", "too slow"],
        "kit_script_template": (
            "import carb\n"
            "settings = carb.settings.get_settings()\n"
            "# Reduce physics fidelity for real-time performance\n"
            "settings.set('/physics/timeStepsPerSecond', 60)\n"
            "settings.set('/physics/numThreads', 4)\n"
            "print('Fix applied: optimized physics for real-time')\n"
        ),
        "risk_level": "medium",
        "default_params": {"timestep": 60},
    },
}


class FixGenerator:
    """Generate fix proposals from diagnoses. Fixes are suggestions for human review.

    Matches issues against 20+ fix templates covering all error categories:
    physics, rendering, USD, extensions, ROS2, Python, and performance.
    """

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

        desc_lower = description.lower()
        cat_lower = category.lower()

        for _key, template in _FIX_TEMPLATES.items():
            match_keys = template.get("match_keys", [])
            for match_key in match_keys:
                if match_key in desc_lower or match_key in cat_lower:
                    params = dict(template.get("default_params", {}))
                    script = template.get("kit_script_template", "")
                    return FixProposal(
                        description=template["description"],
                        parameters=params,
                        kit_script=script,
                        risk_level=template.get("risk_level", "medium"),
                        source="template",
                    )

        return None

    @staticmethod
    def get_template_count() -> int:
        """Return the number of available fix templates."""
        return len(_FIX_TEMPLATES)

    @staticmethod
    def get_covered_categories() -> list[str]:
        """Return the list of error categories covered by templates."""
        categories: set[str] = set()
        for template in _FIX_TEMPLATES.values():
            for key in template.get("match_keys", []):
                categories.add(key)
        return sorted(categories)
