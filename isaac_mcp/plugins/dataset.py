"""Dataset generation plugin for AI training data collection.

Provides MCP tools to collect RGB images, depth maps, sensor data,
and COCO-format annotations during simulation runs.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.dataset.dataset_manager import DatasetConfig, DatasetManager
from isaac_mcp.plugin_host import PluginHost

_READONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

# Module-level manager shared across tool calls
_manager = DatasetManager()


def _success(data: Any) -> str:
    return json.dumps({"status": "ok", "data": data}, indent=2, default=str)


def _error(code: str, message: str) -> str:
    return json.dumps({"status": "error", "error": {"code": code, "message": message}})


def register(host: PluginHost) -> None:
    """Register dataset generation tools."""

    @host.tool(
        name="start_data_collection",
        description=(
            "Start collecting training data (images + sensor readings) during "
            "a simulation run. Configure cameras, image types (rgb, depth, "
            "segmentation), sensor types (imu, odometry), and capture interval."
        ),
        annotations=_MUTATING,
        mutating=True,
    )
    async def start_data_collection(
        scenario_id: str,
        camera_paths: str = "/World/Camera",
        image_types: str = "rgb",
        sensor_types: str = "odometry,imu",
        capture_interval_s: float = 0.5,
        resolution: str = "1280x720",
        instance: str = "primary",
    ) -> str:
        config = DatasetConfig(
            scenario_id=scenario_id,
            camera_paths=[p.strip() for p in camera_paths.split(",") if p.strip()],
            image_types=[t.strip() for t in image_types.split(",") if t.strip()],
            sensor_types=[t.strip() for t in sensor_types.split(",") if t.strip()],
            capture_interval_s=capture_interval_s,
            resolution=resolution,
        )
        dataset = _manager.start_collection(config)
        return _success(dataset.to_dict())

    @host.tool(
        name="stop_data_collection",
        description=(
            "Stop an active data collection session. Finalizes the dataset by "
            "exporting sensor CSVs and writing the dataset manifest."
        ),
        annotations=_MUTATING,
        mutating=True,
    )
    async def stop_data_collection(
        dataset_id: str,
        instance: str = "primary",
    ) -> str:
        dataset = _manager.finalize(dataset_id)
        if dataset is None:
            return _error("not_found", f"Dataset {dataset_id} not found or already finalized")
        return _success(dataset.to_dict())

    @host.tool(
        name="list_datasets",
        description="List all collected datasets with summary statistics.",
        annotations=_READONLY,
    )
    async def list_datasets(instance: str = "primary") -> str:
        datasets = _manager.list_datasets()
        return _success({
            "total": len(datasets),
            "datasets": datasets,
        })

    @host.tool(
        name="get_dataset_info",
        description="Get detailed information about a specific dataset.",
        annotations=_READONLY,
    )
    async def get_dataset_info(
        dataset_id: str,
        instance: str = "primary",
    ) -> str:
        dataset = _manager.get_dataset(dataset_id)
        if dataset is None:
            return _error("not_found", f"Dataset {dataset_id} not found")
        return _success(dataset.to_dict())
