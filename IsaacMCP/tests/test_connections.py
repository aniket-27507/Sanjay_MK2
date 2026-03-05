from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from isaac_mcp.connections.kit_api_client import KitApiClient
from isaac_mcp.connections.ssh_client import SSHLogReader
from isaac_mcp.connections.websocket_client import WebSocketClient


class FakeWS:
    def __init__(self, *, send_pushes_response: bool = True) -> None:
        self.sent: list[dict] = []
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.send_pushes_response = send_pushes_response

    async def send(self, message: str) -> None:
        payload = json.loads(message)
        self.sent.append(payload)
        if self.send_pushes_response:
            await self.queue.put(json.dumps({"command_ack": payload["command"]}))

    async def close(self) -> None:
        await self.queue.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


class FakeDialer:
    def __init__(self, ws: FakeWS):
        self.ws = ws

    def __call__(self, _url: str):
        return self

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False


class FakeResult:
    def __init__(self, exit_status: int, stdout: str = "", stderr: str = ""):
        self.exit_status = exit_status
        self.stdout = stdout
        self.stderr = stderr


class FakeSSHConn:
    def __init__(self):
        self.closed = False

    async def run(self, command: str):
        if command.startswith("ls -t"):
            return FakeResult(0, "/tmp/kit_001.log\n")
        if command.startswith("tail -n"):
            return FakeResult(0, "line1\nline2\n")
        if command.startswith("grep"):
            return FakeResult(0, "match1\nmatch2\n")
        return FakeResult(1, "", "unknown")

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


@pytest.mark.asyncio
async def test_websocket_client_state_and_command_flow() -> None:
    fake_ws = FakeWS(send_pushes_response=True)
    client = WebSocketClient(
        "ws://example:8765",
        reconnect_interval=0.01,
        command_timeout=0.2,
        dialer=FakeDialer(fake_ws),
    )

    await client.connect()
    await client.ensure_connected(timeout=0.2)

    await fake_ws.queue.put(json.dumps({"time": 123, "drones": []}))
    await asyncio.sleep(0.05)
    assert client.get_cached_state()["time"] == 123

    response = await client.send_command("start")
    assert response["command_ack"] == "start"
    assert fake_ws.sent[0]["command"] == "start"

    await client.disconnect()


@pytest.mark.asyncio
async def test_websocket_client_timeout() -> None:
    fake_ws = FakeWS(send_pushes_response=False)
    client = WebSocketClient(
        "ws://example:8765",
        reconnect_interval=0.01,
        command_timeout=0.05,
        dialer=FakeDialer(fake_ws),
    )

    await client.connect()
    await client.ensure_connected(timeout=0.2)

    with pytest.raises(TimeoutError):
        await client.send_command("start")

    await client.disconnect()


@pytest.mark.asyncio
async def test_kit_api_client_get_post() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/scene/prims":
            return httpx.Response(200, json={"ok": True, "path": request.url.path})
        if request.url.path == "/scene/find":
            return httpx.Response(200, json={"found": ["/World/A"]})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://kit")
    client = KitApiClient("http://kit", client=async_client)

    get_result = await client.get("/scene/prims")
    post_result = await client.post("/scene/find", {"pattern": "A"})
    alive = await client.is_alive()

    assert get_result["ok"] is True
    assert post_result["found"] == ["/World/A"]
    assert alive is True

    await client.close()


@pytest.mark.asyncio
async def test_ssh_log_reader_read_and_search() -> None:
    fake_conn = FakeSSHConn()

    async def fake_connector(*_args, **_kwargs):
        return fake_conn

    reader = SSHLogReader(
        host="host",
        user="user",
        key_path="~/.ssh/id_rsa",
        remote_log_dir="/tmp",
        connector=fake_connector,
    )

    await reader.connect()
    lines = await reader.read_lines(2)
    matches = await reader.search("line", 2)

    assert lines == ["line1", "line2"]
    assert matches == ["match1", "match2"]

    await reader.disconnect()
    assert fake_conn.closed is True
