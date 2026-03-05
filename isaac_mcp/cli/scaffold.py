"""Plugin scaffolding generator for IsaacMCP."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any


def run_scaffold(name: str, from_class: str | None = None, output_dir: str = ".") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if from_class:
        code = _scaffold_from_class(name, from_class)
    else:
        code = _scaffold_blank(name)

    file_path = out / f"{name}.py"
    file_path.write_text(code)
    print(f"Generated plugin: {file_path}")
    print(f"\nTo use: add to your plugin directory or reference in isaac-mcp.yaml custom_plugins")


def _scaffold_blank(name: str) -> str:
    return f'''"""Custom IsaacMCP plugin: {name}."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


def register(host: PluginHost) -> None:
    """Register tools for {name}."""

    @host.tool(
        description="TODO: describe what this tool does",
        annotations=_READONLY,
    )
    async def {name}_status(instance: str = "primary") -> str:
        tool = "{name}_status"
        try:
            ws = host.get_connection("websocket", instance)
            state = host.get_state_cache(instance)
            # TODO: implement
            return success(tool, instance, {{"status": "ok"}})
        except Exception as exc:
            return error(tool, instance, "failed", str(exc), exception_details(exc))

    @host.tool(
        description="TODO: describe what this tool does",
        annotations=_MUTATING,
        mutating=True,
    )
    async def {name}_execute(command: str, instance: str = "primary") -> str:
        tool = "{name}_execute"
        try:
            kit = host.get_connection("kit_api", instance)
            result = await kit.execute_script(command)
            return success(tool, instance, {{"result": result}})
        except Exception as exc:
            return error(tool, instance, "failed", str(exc), exception_details(exc))
'''


def _scaffold_from_class(name: str, class_path: str) -> str:
    module_path, _, class_name = class_path.rpartition(":")
    if not module_path or not class_name:
        print(f"Error: --from-class must be in 'module.path:ClassName' format")
        sys.exit(1)

    try:
        sys.path.insert(0, ".")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
    except Exception as exc:
        print(f"Error importing {class_path}: {exc}")
        print("Generating stub-based scaffold instead...")
        return _scaffold_blank(name)

    methods = []
    for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if method_name.startswith("_"):
            continue
        sig = inspect.signature(method)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                params.append(f"{pname}: {param.annotation.__name__}")
            else:
                params.append(pname)
        methods.append((method_name, params, method.__doc__ or f"Wraps {class_name}.{method_name}"))

    tool_defs = []
    for method_name, params, doc in methods:
        param_str = ", ".join(params)
        if param_str:
            param_str = f", {param_str}"

        tool_defs.append(f'''
    @host.tool(
        description="{doc.strip().split(chr(10))[0]}",
        annotations=_READONLY,
    )
    async def {name}_{method_name}({param_str}instance: str = "primary") -> str:
        tool = "{name}_{method_name}"
        try:
            # TODO: implement connection to {class_name}.{method_name}
            return success(tool, instance, {{"method": "{method_name}", "status": "not_implemented"}})
        except Exception as exc:
            return error(tool, instance, "failed", str(exc), exception_details(exc))''')

    tools_code = "\n".join(tool_defs)

    return f'''"""Auto-generated IsaacMCP plugin wrapping {class_name}."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


def register(host: PluginHost) -> None:
    """Auto-generated tools from {class_name}."""
{tools_code}
'''


def _scaffold_from_class_stub(name: str, class_path: str) -> str:
    return _scaffold_blank(name)
