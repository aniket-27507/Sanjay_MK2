"""Standardized tool response contract helpers."""

from __future__ import annotations

import json
from typing import Any


JSONDict = dict[str, Any]


def _default_json_serializer(value: Any) -> Any:
    """Best-effort serializer for objects that are not natively JSON serializable."""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def success(tool: str, instance: str, data: JSONDict) -> str:
    """Return standardized success payload as a JSON string."""
    return json.dumps(
        {
            "status": "ok",
            "tool": tool,
            "instance": instance,
            "data": data,
            "error": None,
        },
        default=_default_json_serializer,
        ensure_ascii=True,
    )


def error(
    tool: str,
    instance: str,
    code: str,
    message: str,
    details: JSONDict | None = None,
) -> str:
    """Return standardized error payload as a JSON string."""
    return json.dumps(
        {
            "status": "error",
            "tool": tool,
            "instance": instance,
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            },
        },
        default=_default_json_serializer,
        ensure_ascii=True,
    )


def exception_details(exc: Exception) -> JSONDict:
    """Convert an exception into a structured details dictionary."""
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
