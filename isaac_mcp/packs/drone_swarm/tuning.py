"""Parameter tuning tools for drone swarm simulation projects."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


def register(host: PluginHost) -> None:

    @host.tool(
        description="Read current physics and config parameters from the Isaac Sim scene",
        annotations=_READONLY,
    )
    async def tuning_get_parameters(instance: str = "primary") -> str:
        tool = "tuning_get_parameters"
        try:
            kit = host.get_connection("kit_api", instance)
            physics = await kit.get("/scene/physics")
            return success(tool, instance, {"parameters": physics})
        except ValueError:
            state = host.get_state_cache(instance)
            return success(tool, instance, {"parameters": state, "source": "websocket_cache"})
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Set a physics or config parameter live via Kit API script injection",
        annotations=_DESTRUCTIVE,
        mutating=True,
    )
    async def tuning_set_parameter(
        parameter_path: str,
        value: str,
        instance: str = "primary",
    ) -> str:
        """parameter_path: USD prim attribute path. value: new value (auto-typed)."""
        tool = "tuning_set_parameter"
        try:
            kit = host.get_connection("kit_api", instance)
            script = (
                "import omni.usd\n"
                "stage = omni.usd.get_context().get_stage()\n"
                f"prim_path, _, attr_name = '{parameter_path}'.rpartition('.')\n"
                "prim = stage.GetPrimAtPath(prim_path) if prim_path else None\n"
                "if prim and prim.IsValid():\n"
                f"    attr = prim.GetAttribute(attr_name)\n"
                f"    if attr:\n"
                f"        attr.Set({value})\n"
                f"        print(f'Set {{attr_name}} = {value}')\n"
                "    else:\n"
                f"        print(f'Attribute {{attr_name}} not found on {{prim_path}}')\n"
                "else:\n"
                f"    print(f'Prim not found: {{prim_path}}')\n"
            )
            result = await kit.execute_script(script)
            return success(tool, instance, {
                "parameter": parameter_path,
                "value": value,
                "result": result,
            })
        except ValueError:
            return error(tool, instance, "kit_unavailable", "Kit API not enabled for this instance")
        except Exception as exc:
            return error(tool, instance, "set_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Run a parameter sweep using the experiment engine",
        annotations=_DESTRUCTIVE,
        mutating=True,
    )
    async def tuning_sweep_parameter(
        parameter_name: str,
        min_value: float,
        max_value: float,
        steps: int = 10,
        instance: str = "primary",
    ) -> str:
        tool = "tuning_sweep_parameter"
        if steps < 2 or steps > 100:
            return error(tool, instance, "validation_error", "steps must be between 2 and 100")
        try:
            kit = host.get_connection("kit_api", instance)
            step_size = (max_value - min_value) / (steps - 1)
            values = [round(min_value + i * step_size, 6) for i in range(steps)]

            results: list[dict[str, Any]] = []
            for val in values:
                script = (
                    f"# Sweep: {parameter_name} = {val}\n"
                    f"import omni.physx\n"
                    f"print('Parameter {parameter_name} set to {val}')\n"
                )
                r = await kit.execute_script(script)
                results.append({"value": val, "result": r})

            return success(tool, instance, {
                "parameter": parameter_name,
                "range": {"min": min_value, "max": max_value, "steps": steps},
                "results": results,
            })
        except ValueError:
            return error(tool, instance, "kit_unavailable", "Kit API not enabled")
        except Exception as exc:
            return error(tool, instance, "sweep_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Browse USD prim tree for drone prims, sensors, and environment objects",
        annotations=_READONLY,
    )
    async def tuning_get_scene_hierarchy(
        root_path: str = "/World",
        max_depth: int = 3,
        instance: str = "primary",
    ) -> str:
        tool = "tuning_get_scene_hierarchy"
        try:
            kit = host.get_connection("kit_api", instance)
            result = await kit.get("/scene/hierarchy", params={"path": root_path, "depth": max_depth})
            return success(tool, instance, {"root": root_path, "hierarchy": result})
        except ValueError:
            return error(tool, instance, "kit_unavailable", "Kit API not enabled")
        except Exception as exc:
            return error(tool, instance, "hierarchy_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Execute an arbitrary Kit API Python script in the Isaac Sim runtime",
        annotations=_DESTRUCTIVE,
        mutating=True,
    )
    async def tuning_inject_script(script: str, instance: str = "primary") -> str:
        tool = "tuning_inject_script"
        if not script.strip():
            return error(tool, instance, "validation_error", "Script cannot be empty")
        if len(script) > 50000:
            return error(tool, instance, "validation_error", "Script too large (max 50KB)")
        try:
            kit = host.get_connection("kit_api", instance)
            result = await kit.execute_script(script)
            return success(tool, instance, {"script_length": len(script), "result": result})
        except ValueError:
            return error(tool, instance, "kit_unavailable", "Kit API not enabled")
        except Exception as exc:
            return error(tool, instance, "inject_failed", str(exc), exception_details(exc))
