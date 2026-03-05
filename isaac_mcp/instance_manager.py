"""Instance registry and connection lifecycle management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from isaac_mcp.config import InstanceConfig, ServerConfig
from isaac_mcp.connections import KitApiClient, Ros2Client, SSHLogReader, WebSocketClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IsaacInstance:
    """Runtime container for one Isaac instance and all related clients."""

    name: str
    label: str
    config: InstanceConfig
    ws_client: WebSocketClient
    kit_client: KitApiClient | None
    ssh_client: SSHLogReader | None
    ros2_client: Ros2Client | None

    @property
    def state_cache(self) -> dict[str, Any]:
        return self.ws_client.get_cached_state()


class InstanceManager:
    """Manages one or more Isaac instances and their connection clients."""

    def __init__(self, config: ServerConfig):
        self._instances: dict[str, IsaacInstance] = {}
        self._started = False

        for name, instance_cfg in config.instances.items():
            ws_client = WebSocketClient(
                url=instance_cfg.simulation.websocket_url,
                reconnect_interval=instance_cfg.simulation.reconnect_interval_s,
                command_timeout=instance_cfg.simulation.command_timeout_s,
            )

            kit_client = (
                KitApiClient(base_url=instance_cfg.kit_api.base_url)
                if instance_cfg.kit_api.enabled
                else None
            )

            ssh_client = None
            if instance_cfg.logs.method == "ssh":
                ssh_client = SSHLogReader(
                    host=instance_cfg.logs.ssh.host,
                    user=instance_cfg.logs.ssh.user,
                    key_path=instance_cfg.logs.ssh.key_path,
                    remote_log_dir=instance_cfg.logs.remote_path,
                )

            ros2_client = None
            if instance_cfg.ros2.enabled:
                ros2_client = Ros2Client(
                    domain_id=instance_cfg.ros2.domain_id,
                    configured_topics=[{"name": t.name, "type": t.type} for t in instance_cfg.ros2.topics],
                    qos_depth=instance_cfg.ros2.qos_depth,
                    reliability=instance_cfg.ros2.reliability,
                    auto_subscribe=[{"name": t.name, "type": t.type} for t in instance_cfg.ros2.auto_subscribe],
                    coordinate_frame=instance_cfg.ros2.coordinate_frame,
                )

            self._instances[name] = IsaacInstance(
                name=name,
                label=instance_cfg.label,
                config=instance_cfg,
                ws_client=ws_client,
                kit_client=kit_client,
                ssh_client=ssh_client,
                ros2_client=ros2_client,
            )

    async def start(self) -> None:
        """Start background/lazy connections for all instances."""
        if self._started:
            return

        for inst in self._instances.values():
            await inst.ws_client.connect()
            if inst.ssh_client is not None:
                await self._safe_call(inst.name, "ssh.connect", inst.ssh_client.connect)
            if inst.ros2_client is not None:
                await self._safe_call(inst.name, "ros2.connect", inst.ros2_client.connect)

        self._started = True

    async def stop(self) -> None:
        """Gracefully stop all clients across instances."""
        tasks: list[asyncio.Future[Any] | asyncio.Task[Any]] = []
        for inst in self._instances.values():
            tasks.append(asyncio.create_task(inst.ws_client.disconnect()))

            if inst.kit_client is not None:
                tasks.append(asyncio.create_task(inst.kit_client.close()))
            if inst.ssh_client is not None:
                tasks.append(asyncio.create_task(inst.ssh_client.disconnect()))
            if inst.ros2_client is not None:
                tasks.append(asyncio.create_task(inst.ros2_client.disconnect()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._started = False

    def list_instances(self) -> list[str]:
        return sorted(self._instances.keys())

    def get_instance(self, name: str = "primary") -> IsaacInstance:
        try:
            return self._instances[name]
        except KeyError as exc:
            available = ", ".join(self.list_instances())
            raise ValueError(f"Unknown instance '{name}'. Available: {available}") from exc

    def health_snapshot(self) -> dict[str, Any]:
        """Return current connectivity and feature status for each instance."""
        status: dict[str, Any] = {}
        for name, inst in self._instances.items():
            status[name] = {
                "label": inst.label,
                "ws_connected": inst.ws_client.is_connected,
                "kit_enabled": inst.kit_client is not None,
                "ssh_enabled": inst.ssh_client is not None,
                "ssh_connected": inst.ssh_client.is_connected if inst.ssh_client is not None else False,
                "ros2_enabled": inst.ros2_client is not None,
                "ros2_connected": inst.ros2_client.is_connected if inst.ros2_client is not None else False,
                "ros2_available": inst.ros2_client.available if inst.ros2_client is not None else False,
            }
        return status

    async def _safe_call(self, instance_name: str, action: str, fn: Any) -> None:
        try:
            await fn()
        except Exception as exc:
            logger.warning("Instance '%s' %s failed: %s", instance_name, action, exc)
