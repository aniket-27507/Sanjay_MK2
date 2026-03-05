from __future__ import annotations

import base64
import time

import httpx
import jwt
import pytest

from isaac_mcp.auth import OAuthTokenVerifier, build_auth_components
from isaac_mcp.config import AuthConfig


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


@pytest.mark.asyncio
async def test_oauth_token_verifier_accepts_valid_token() -> None:
    secret = b"a" * 32
    jwk = {"kty": "oct", "k": _b64url(secret), "alg": "HS256", "kid": "kid-1"}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/.well-known/jwks.json"
        return httpx.Response(200, json={"keys": [jwk]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://auth.example.com")

    cfg = AuthConfig(
        enabled=True,
        issuer_url="https://auth.example.com",
        resource_server_url="https://mcp.example.com",
        required_scopes=["mcp:read"],
        jwks_url="https://auth.example.com/.well-known/jwks.json",
        algorithms=["HS256"],
        client_id_claim="sub",
        scopes_claim="scope",
    )
    verifier = OAuthTokenVerifier(cfg, http_client=client)

    token = jwt.encode(
        {
            "sub": "client-123",
            "scope": "mcp:read mcp:write",
            "iss": "https://auth.example.com",
            "exp": int(time.time()) + 300,
        },
        secret,
        algorithm="HS256",
        headers={"kid": "kid-1"},
    )

    access = await verifier.verify_token(token)

    assert access is not None
    assert access.client_id == "client-123"
    assert "mcp:read" in access.scopes
    await client.aclose()


@pytest.mark.asyncio
async def test_oauth_token_verifier_rejects_missing_scope() -> None:
    secret = b"b" * 32
    jwk = {"kty": "oct", "k": _b64url(secret), "alg": "HS256", "kid": "kid-2"}

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"keys": [jwk]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://auth.example.com")

    cfg = AuthConfig(
        enabled=True,
        issuer_url="https://auth.example.com",
        resource_server_url="https://mcp.example.com",
        required_scopes=["mcp:write"],
        jwks_url="https://auth.example.com/.well-known/jwks.json",
        algorithms=["HS256"],
        scopes_claim="scope",
    )
    verifier = OAuthTokenVerifier(cfg, http_client=client)

    token = jwt.encode(
        {
            "sub": "client-123",
            "scope": "mcp:read",
            "iss": "https://auth.example.com",
            "exp": int(time.time()) + 300,
        },
        secret,
        algorithm="HS256",
        headers={"kid": "kid-2"},
    )

    access = await verifier.verify_token(token)

    assert access is None
    await client.aclose()


def test_build_auth_components_requires_resource_server_url() -> None:
    cfg = AuthConfig(
        enabled=True,
        issuer_url="https://auth.example.com",
        resource_server_url="",
        required_scopes=["mcp:read"],
    )

    with pytest.raises(ValueError):
        build_auth_components(cfg, public_base_url="")


def test_build_auth_components_uses_public_base_url_fallback() -> None:
    cfg = AuthConfig(
        enabled=True,
        issuer_url="https://auth.example.com",
        resource_server_url="",
        required_scopes=["mcp:read"],
        jwks_url="https://auth.example.com/.well-known/jwks.json",
    )

    components = build_auth_components(cfg, public_base_url="https://mcp.example.com")

    assert str(components.settings.resource_server_url) == "https://mcp.example.com/"
