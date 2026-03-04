"""HTTP client for Isaac Kit REST API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class KitApiClient:
    """Simple async client wrapper around Isaac Kit API endpoints."""

    def __init__(self, base_url: str, timeout: float = 30.0, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = await self._client.get(endpoint, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return {"result": payload}
            return payload
        except httpx.HTTPError as exc:
            logger.error("Kit API GET failed endpoint=%s error=%s", endpoint, exc)
            raise

    async def post(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = await self._client.post(endpoint, json=data or {})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return {"result": payload}
            return payload
        except httpx.HTTPError as exc:
            logger.error("Kit API POST failed endpoint=%s error=%s", endpoint, exc)
            raise

    async def execute_script(self, script: str) -> str:
        result = await self.post("/kit/script/execute", {"code": script})
        return str(result.get("output", ""))

    async def is_alive(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
