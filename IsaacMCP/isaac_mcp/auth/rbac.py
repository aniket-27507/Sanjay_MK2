"""Role-Based Access Control (RBAC) for MCP tools.

Defines roles (viewer, operator, admin) and maps them to tool categories.
Each tool can declare a minimum required role; access is denied if the
caller's role is insufficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class Role(IntEnum):
    """Ordered role hierarchy. Higher numeric value = more privileged."""

    VIEWER = 0
    OPERATOR = 1
    ADMIN = 2


# Mapping from string to Role enum
ROLE_NAMES: dict[str, Role] = {
    "viewer": Role.VIEWER,
    "operator": Role.OPERATOR,
    "admin": Role.ADMIN,
}


# Default tool-category -> minimum required role mapping
_DEFAULT_CATEGORY_ROLES: dict[str, Role] = {
    # Read-only inspection tools
    "scene_inspect": Role.VIEWER,
    "camera_render": Role.VIEWER,
    "log_monitor": Role.VIEWER,
    "diagnostics": Role.VIEWER,
    # Operational tools (mutating simulation state)
    "sim_control": Role.OPERATOR,
    "ros2_bridge": Role.OPERATOR,
    "experiments": Role.OPERATOR,
    "scenario_lab": Role.OPERATOR,
    "dataset": Role.OPERATOR,
    # Administrative / destructive tools
    "autonomous_loop": Role.OPERATOR,
    "rl_training": Role.OPERATOR,
    "adversarial": Role.OPERATOR,
    "cicd": Role.OPERATOR,
    # Admin-only
    "admin": Role.ADMIN,
}


@dataclass(slots=True)
class RBACPolicy:
    """RBAC policy configuration.

    Attributes
    ----------
    enabled:
        Whether RBAC enforcement is active.
    default_role:
        Role assigned to unauthenticated or unmapped users.
    category_roles:
        Mapping of tool category -> minimum required role.
    tool_roles:
        Per-tool overrides (tool_name -> minimum required role).
    user_roles:
        Mapping of user/client_id -> role name.
    """

    enabled: bool = False
    default_role: str = "viewer"
    category_roles: dict[str, str] = field(default_factory=dict)
    tool_roles: dict[str, str] = field(default_factory=dict)
    user_roles: dict[str, str] = field(default_factory=dict)


class RBACEnforcer:
    """Evaluate tool access based on caller role and tool requirements."""

    def __init__(self, policy: RBACPolicy | None = None) -> None:
        self._policy = policy or RBACPolicy()
        self._category_roles = dict(_DEFAULT_CATEGORY_ROLES)
        self._tool_roles: dict[str, Role] = {}
        self._tool_categories: dict[str, str] = {}

        # Apply config overrides
        if self._policy.category_roles:
            for cat, role_name in self._policy.category_roles.items():
                role = ROLE_NAMES.get(role_name.lower())
                if role is not None:
                    self._category_roles[cat] = role

        if self._policy.tool_roles:
            for tool, role_name in self._policy.tool_roles.items():
                role = ROLE_NAMES.get(role_name.lower())
                if role is not None:
                    self._tool_roles[tool] = role

    @property
    def enabled(self) -> bool:
        return self._policy.enabled

    def register_tool_category(self, tool_name: str, category: str) -> None:
        """Map a tool name to its plugin category for RBAC lookups."""
        self._tool_categories[tool_name] = category

    def resolve_role(self, user_id: str = "") -> Role:
        """Resolve a user's role from the policy."""
        if user_id and self._policy.user_roles:
            role_name = self._policy.user_roles.get(user_id, "")
            if role_name:
                return ROLE_NAMES.get(role_name.lower(), self._default_role)
        return self._default_role

    @property
    def _default_role(self) -> Role:
        return ROLE_NAMES.get(self._policy.default_role.lower(), Role.VIEWER)

    def get_required_role(self, tool_name: str) -> Role:
        """Return the minimum required role for a tool."""
        # Per-tool override takes priority
        if tool_name in self._tool_roles:
            return self._tool_roles[tool_name]

        # Category-based lookup
        category = self._tool_categories.get(tool_name, "")
        if category and category in self._category_roles:
            return self._category_roles[category]

        # Default: viewer can access unknown tools
        return Role.VIEWER

    def check_access(self, tool_name: str, user_role: Role) -> AccessDecision:
        """Check if a role has access to a tool.

        Returns an AccessDecision with allowed=True/False and reason.
        """
        if not self._policy.enabled:
            return AccessDecision(allowed=True, reason="RBAC disabled")

        required = self.get_required_role(tool_name)
        if user_role >= required:
            return AccessDecision(
                allowed=True,
                reason=f"Role {user_role.name} >= required {required.name}",
            )
        return AccessDecision(
            allowed=False,
            required_role=required.name.lower(),
            user_role=user_role.name.lower(),
            reason=f"Insufficient role: {user_role.name} < {required.name} for tool '{tool_name}'",
        )

    def list_accessible_tools(self, user_role: Role, all_tools: list[str]) -> list[str]:
        """Return tools accessible to the given role."""
        if not self._policy.enabled:
            return list(all_tools)
        return [t for t in all_tools if self.check_access(t, user_role).allowed]

    def get_policy_summary(self) -> dict[str, Any]:
        """Return a summary of the RBAC policy for diagnostics."""
        return {
            "enabled": self._policy.enabled,
            "default_role": self._policy.default_role,
            "category_roles": {
                cat: role.name.lower() for cat, role in self._category_roles.items()
            },
            "tool_overrides": {
                tool: role.name.lower() for tool, role in self._tool_roles.items()
            },
            "user_mappings": len(self._policy.user_roles),
        }


@dataclass(slots=True)
class AccessDecision:
    """Result of an RBAC access check."""

    allowed: bool
    reason: str = ""
    required_role: str = ""
    user_role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "required_role": self.required_role,
            "user_role": self.user_role,
        }
