"""OAuth and bearer-token verification helpers for remote MCP mode."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from isaac_mcp.config import AuthConfig

logger = logging.getLogger(__name__)

try:
    import jwt
except Exception:  # pragma: no cover - runtime dependency failure
    jwt = None  # type: ignore


@dataclass(slots=True)
class AuthComponents:
    settings: AuthSettings
    token_verifier: TokenVerifier


class OAuthTokenVerifier(TokenVerifier):
    """JWT bearer token verifier backed by OIDC discovery + JWKS."""

    def __init__(
        self,
        auth_config: AuthConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        jwks_cache_ttl_s: float = 300.0,
    ) -> None:
        if jwt is None:
            raise RuntimeError("PyJWT is required for OAuth token verification")

        self._auth = auth_config
        self._client = http_client or httpx.AsyncClient(timeout=10.0)
        self._owns_client = http_client is None
        self._jwks_cache_ttl_s = jwks_cache_ttl_s

        self._jwks_url: str = auth_config.jwks_url.strip()
        self._jwks_cache_expiry: float = 0.0
        self._keys_by_kid: dict[str, Any] = {}

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None

        header = jwt.get_unverified_header(token)
        kid = str(header.get("kid", "")).strip()
        if not kid:
            logger.warning("OAuth token missing 'kid' header")
            return None

        key = await self._get_key(kid)
        if key is None:
            logger.warning("No JWKS key found for kid=%s", kid)
            return None

        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=self._auth.algorithms,
                issuer=self._auth.issuer_url,
                audience=self._auth.audience or None,
                options={"verify_aud": bool(self._auth.audience)},
                leeway=5,
            )
        except jwt.ExpiredSignatureError:
            logger.info("OAuth token expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.info("OAuth token rejected: %s", exc)
            return None

        scopes = _parse_scopes(claims.get(self._auth.scopes_claim))
        for required_scope in self._auth.required_scopes:
            if required_scope not in scopes:
                logger.info("OAuth token missing required scope=%s", required_scope)
                return None

        client_id = str(claims.get(self._auth.client_id_claim) or claims.get("sub") or "oauth-client")
        exp_value = claims.get("exp")
        expires_at = int(exp_value) if isinstance(exp_value, (int, float)) else None

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=self._auth.resource_server_url or None,
        )

    async def _get_key(self, kid: str) -> Any | None:
        now = time.time()
        if now >= self._jwks_cache_expiry:
            await self._refresh_jwks()

        key = self._keys_by_kid.get(kid)
        if key is not None:
            return key

        # key rotation fallback: force refresh once if kid is unknown.
        await self._refresh_jwks(force=True)
        return self._keys_by_kid.get(kid)

    async def _refresh_jwks(self, force: bool = False) -> None:
        if not force and time.time() < self._jwks_cache_expiry:
            return

        jwks_url = await self._resolve_jwks_url()
        response = await self._client.get(jwks_url)
        response.raise_for_status()

        payload = response.json()
        keys = payload.get("keys", []) if isinstance(payload, dict) else []
        new_keys: dict[str, Any] = {}

        for jwk_dict in keys:
            if not isinstance(jwk_dict, dict):
                continue
            kid = str(jwk_dict.get("kid", "")).strip()
            if not kid:
                continue
            try:
                new_keys[kid] = jwt.PyJWK.from_dict(jwk_dict).key
            except Exception as exc:
                logger.warning("Ignoring invalid JWK for kid=%s: %s", kid, exc)

        self._keys_by_kid = new_keys
        self._jwks_cache_expiry = time.time() + self._jwks_cache_ttl_s

    async def _resolve_jwks_url(self) -> str:
        if self._jwks_url:
            return self._jwks_url

        discovery_url = f"{self._auth.issuer_url.rstrip('/')}/.well-known/openid-configuration"
        response = await self._client.get(discovery_url)
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("OIDC discovery response is not an object")

        jwks_uri = payload.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri.strip():
            raise ValueError("OIDC discovery response missing jwks_uri")

        self._jwks_url = jwks_uri.strip()
        return self._jwks_url


def build_auth_components(auth_config: AuthConfig, *, public_base_url: str = "") -> AuthComponents:
    """Build FastMCP auth settings + token verifier from typed config."""
    if not auth_config.enabled:
        raise ValueError("Auth config is disabled")

    issuer_url = auth_config.issuer_url.strip()
    if not issuer_url:
        raise ValueError("Auth is enabled but issuer_url is missing")

    resource_server_url = auth_config.resource_server_url.strip() or public_base_url.strip()
    if not resource_server_url:
        raise ValueError("Auth is enabled but resource_server_url/public_base_url is missing")

    required_scopes = auth_config.required_scopes or ["mcp:read"]

    settings_kwargs: dict[str, Any] = {
        "issuer_url": issuer_url,
        "resource_server_url": resource_server_url,
        "required_scopes": required_scopes,
    }
    if auth_config.service_documentation_url.strip():
        settings_kwargs["service_documentation_url"] = auth_config.service_documentation_url.strip()

    settings = AuthSettings(**settings_kwargs)
    verifier = OAuthTokenVerifier(auth_config)

    return AuthComponents(settings=settings, token_verifier=verifier)


def _parse_scopes(raw: Any) -> list[str]:
    if isinstance(raw, str):
        # Most providers use space-delimited scope claim.
        scopes = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
        return scopes
    if isinstance(raw, list):
        scopes = [str(item).strip() for item in raw if str(item).strip()]
        return scopes
    return []
