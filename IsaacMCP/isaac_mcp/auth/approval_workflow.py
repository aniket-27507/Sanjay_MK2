"""Async approval gate for destructive operations.

When a tool is marked as requiring approval, execution is paused and a
pending approval request is created. The request can be approved or
denied via a dedicated MCP tool. Only after approval does the original
operation proceed.

This is an in-memory implementation suitable for single-server deployments.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass(slots=True)
class ApprovalRequest:
    """A pending approval request for a destructive operation."""

    request_id: str
    tool_name: str
    instance: str
    requestor: str
    description: str
    parameters: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = 0.0
    resolved_at: float = 0.0
    resolved_by: str = ""
    ttl_s: float = 300.0

    def is_expired(self) -> bool:
        if self.status != ApprovalStatus.PENDING:
            return False
        return time.time() > (self.created_at + self.ttl_s)

    def to_dict(self) -> dict[str, Any]:
        status = self.status
        if self.is_expired():
            status = ApprovalStatus.EXPIRED
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "instance": self.instance,
            "requestor": self.requestor,
            "description": self.description,
            "parameters": self.parameters,
            "status": status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "ttl_s": self.ttl_s,
        }


class ApprovalManager:
    """Manage approval requests for destructive operations.

    Tools that require approval create a request here. The request must
    be approved (or denied) before the operation executes.
    """

    def __init__(self, default_ttl_s: float = 300.0) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._default_ttl_s = default_ttl_s
        # Set of tool names that always require approval
        self._requires_approval: set[str] = set()

    def register_tool(self, tool_name: str) -> None:
        """Mark a tool as requiring approval before execution."""
        self._requires_approval.add(tool_name)

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires approval."""
        return tool_name in self._requires_approval

    def create_request(
        self,
        tool_name: str,
        instance: str = "primary",
        requestor: str = "",
        description: str = "",
        parameters: dict[str, Any] | None = None,
        ttl_s: float | None = None,
    ) -> ApprovalRequest:
        """Create a new pending approval request."""
        request = ApprovalRequest(
            request_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            instance=instance,
            requestor=requestor,
            description=description,
            parameters=parameters or {},
            created_at=time.time(),
            ttl_s=ttl_s or self._default_ttl_s,
        )
        self._requests[request.request_id] = request
        return request

    def approve(self, request_id: str, approver: str = "") -> ApprovalRequest | None:
        """Approve a pending request. Returns None if not found or not pending."""
        request = self._requests.get(request_id)
        if request is None:
            return None
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            return None
        if request.status != ApprovalStatus.PENDING:
            return None

        request.status = ApprovalStatus.APPROVED
        request.resolved_at = time.time()
        request.resolved_by = approver
        return request

    def deny(self, request_id: str, denier: str = "") -> ApprovalRequest | None:
        """Deny a pending request."""
        request = self._requests.get(request_id)
        if request is None:
            return None
        if request.status != ApprovalStatus.PENDING:
            return None

        request.status = ApprovalStatus.DENIED
        request.resolved_at = time.time()
        request.resolved_by = denier
        return request

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Retrieve a request by ID."""
        return self._requests.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        """Return all pending (non-expired) requests."""
        self._expire_stale()
        return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def list_all(self, limit: int = 50) -> list[ApprovalRequest]:
        """Return recent requests regardless of status."""
        self._expire_stale()
        requests = sorted(self._requests.values(), key=lambda r: r.created_at, reverse=True)
        return requests[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about approval requests."""
        self._expire_stale()
        by_status: dict[str, int] = {}
        for req in self._requests.values():
            status = req.status.value
            if req.is_expired():
                status = "expired"
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_requests": len(self._requests),
            "by_status": by_status,
            "tools_requiring_approval": sorted(self._requires_approval),
        }

    def _expire_stale(self) -> None:
        """Mark expired pending requests."""
        for req in self._requests.values():
            if req.status == ApprovalStatus.PENDING and req.is_expired():
                req.status = ApprovalStatus.EXPIRED
