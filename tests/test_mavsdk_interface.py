import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types.drone_types import TelemetryData, Vector3
from src.single_drone.flight_control.mavsdk_interface import MAVSDKInterface


async def _single_item_stream(item):
    yield item


def _make_interface_without_mavsdk() -> MAVSDKInterface:
    iface = MAVSDKInterface.__new__(MAVSDKInterface)
    iface._telemetry = TelemetryData()
    iface._telemetry_lock = asyncio.Lock()
    iface._running = False
    iface._connected = True
    iface._offboard_active = False
    return iface


@pytest.mark.asyncio
async def test_local_ned_position_velocity_updates_cache():
    iface = _make_interface_without_mavsdk()
    sample = SimpleNamespace(
        position=SimpleNamespace(north_m=12.5, east_m=-3.0, down_m=-8.0),
        velocity=SimpleNamespace(north_m_s=1.5, east_m_s=0.25, down_m_s=-0.1),
    )
    iface._drone = SimpleNamespace(
        telemetry=SimpleNamespace(
            position_velocity_ned=MagicMock(return_value=_single_item_stream(sample))
        )
    )

    await iface._subscribe_position_velocity_ned()

    assert iface.get_position() == Vector3(12.5, -3.0, -8.0)
    assert iface.get_velocity() == Vector3(1.5, 0.25, -0.1)
    assert iface.get_altitude() == 8.0


@pytest.mark.asyncio
async def test_global_position_does_not_reset_local_xy():
    iface = _make_interface_without_mavsdk()
    iface._telemetry.position = Vector3(12.5, -3.0, -8.0)
    global_position = SimpleNamespace(
        latitude_deg=47.397,
        longitude_deg=8.545,
        absolute_altitude_m=500.0,
        relative_altitude_m=8.0,
    )
    iface._drone = SimpleNamespace(
        telemetry=SimpleNamespace(
            position=MagicMock(return_value=_single_item_stream(global_position))
        )
    )

    await iface._subscribe_global_position()

    assert iface.get_position() == Vector3(12.5, -3.0, -8.0)
    assert iface.telemetry.latitude == 47.397
    assert iface.telemetry.longitude == 8.545
    assert iface.telemetry.altitude_rel == 8.0


@pytest.mark.asyncio
async def test_mavsdk_land_stops_offboard_before_action_land():
    iface = _make_interface_without_mavsdk()
    iface._offboard_active = True
    calls = []

    async def offboard_stop():
        calls.append("stop_offboard")

    async def action_land():
        calls.append("land")

    iface._drone = SimpleNamespace(
        offboard=SimpleNamespace(stop=AsyncMock(side_effect=offboard_stop)),
        action=SimpleNamespace(land=AsyncMock(side_effect=action_land)),
    )

    assert await iface.land() is True
    assert iface._offboard_active is False
    assert calls == ["stop_offboard", "land"]
