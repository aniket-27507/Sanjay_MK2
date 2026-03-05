"""Tests for RBAC and approval workflows."""

import time

import pytest

from isaac_mcp.auth.rbac import (
    AccessDecision,
    RBACEnforcer,
    RBACPolicy,
    Role,
    ROLE_NAMES,
)
from isaac_mcp.auth.approval_workflow import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalStatus,
)


# --- RBAC tests ---


class TestRole:
    def test_role_ordering(self):
        assert Role.VIEWER < Role.OPERATOR < Role.ADMIN

    def test_role_names(self):
        assert ROLE_NAMES["viewer"] == Role.VIEWER
        assert ROLE_NAMES["operator"] == Role.OPERATOR
        assert ROLE_NAMES["admin"] == Role.ADMIN


class TestRBACEnforcer:
    def test_disabled_allows_everything(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=False))
        decision = enforcer.check_access("sim_start", Role.VIEWER)
        assert decision.allowed

    def test_viewer_can_read(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True))
        enforcer.register_tool_category("scene_list_prims", "scene_inspect")
        decision = enforcer.check_access("scene_list_prims", Role.VIEWER)
        assert decision.allowed

    def test_viewer_blocked_from_operator(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True))
        enforcer.register_tool_category("sim_start", "sim_control")
        decision = enforcer.check_access("sim_start", Role.VIEWER)
        assert not decision.allowed
        assert decision.required_role == "operator"

    def test_operator_can_access_operator_tools(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True))
        enforcer.register_tool_category("sim_start", "sim_control")
        decision = enforcer.check_access("sim_start", Role.OPERATOR)
        assert decision.allowed

    def test_admin_can_access_everything(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True))
        enforcer.register_tool_category("admin_tool", "admin")
        decision = enforcer.check_access("admin_tool", Role.ADMIN)
        assert decision.allowed

    def test_per_tool_override(self):
        enforcer = RBACEnforcer(RBACPolicy(
            enabled=True,
            tool_roles={"dangerous_tool": "admin"},
        ))
        decision = enforcer.check_access("dangerous_tool", Role.OPERATOR)
        assert not decision.allowed
        decision = enforcer.check_access("dangerous_tool", Role.ADMIN)
        assert decision.allowed

    def test_category_role_override(self):
        enforcer = RBACEnforcer(RBACPolicy(
            enabled=True,
            category_roles={"scene_inspect": "operator"},
        ))
        enforcer.register_tool_category("scene_list_prims", "scene_inspect")
        # Normally viewer, but overridden to operator
        decision = enforcer.check_access("scene_list_prims", Role.VIEWER)
        assert not decision.allowed

    def test_resolve_role_from_user_mapping(self):
        enforcer = RBACEnforcer(RBACPolicy(
            enabled=True,
            user_roles={"user1": "admin", "user2": "viewer"},
        ))
        assert enforcer.resolve_role("user1") == Role.ADMIN
        assert enforcer.resolve_role("user2") == Role.VIEWER
        assert enforcer.resolve_role("unknown") == Role.VIEWER  # default

    def test_list_accessible_tools(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True))
        enforcer.register_tool_category("read_tool", "scene_inspect")
        enforcer.register_tool_category("write_tool", "sim_control")
        tools = enforcer.list_accessible_tools(Role.VIEWER, ["read_tool", "write_tool"])
        assert "read_tool" in tools
        assert "write_tool" not in tools

    def test_list_accessible_tools_disabled(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=False))
        tools = enforcer.list_accessible_tools(Role.VIEWER, ["a", "b", "c"])
        assert tools == ["a", "b", "c"]

    def test_get_policy_summary(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True, default_role="operator"))
        summary = enforcer.get_policy_summary()
        assert summary["enabled"] is True
        assert summary["default_role"] == "operator"

    def test_unknown_tool_defaults_to_viewer(self):
        enforcer = RBACEnforcer(RBACPolicy(enabled=True))
        decision = enforcer.check_access("unknown_tool", Role.VIEWER)
        assert decision.allowed

    def test_access_decision_to_dict(self):
        decision = AccessDecision(allowed=False, reason="test", required_role="admin", user_role="viewer")
        d = decision.to_dict()
        assert d["allowed"] is False
        assert d["required_role"] == "admin"


# --- ApprovalWorkflow tests ---


class TestApprovalManager:
    def test_create_request(self):
        mgr = ApprovalManager()
        req = mgr.create_request("apply_fix", description="Fix physics")
        assert req.status == ApprovalStatus.PENDING
        assert req.tool_name == "apply_fix"

    def test_approve_request(self):
        mgr = ApprovalManager()
        req = mgr.create_request("apply_fix")
        approved = mgr.approve(req.request_id, approver="admin")
        assert approved is not None
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.resolved_by == "admin"

    def test_deny_request(self):
        mgr = ApprovalManager()
        req = mgr.create_request("apply_fix")
        denied = mgr.deny(req.request_id, denier="admin")
        assert denied is not None
        assert denied.status == ApprovalStatus.DENIED

    def test_cannot_approve_twice(self):
        mgr = ApprovalManager()
        req = mgr.create_request("apply_fix")
        mgr.approve(req.request_id)
        second = mgr.approve(req.request_id)
        assert second is None

    def test_expired_request(self):
        mgr = ApprovalManager(default_ttl_s=0.01)
        req = mgr.create_request("apply_fix")
        time.sleep(0.02)
        assert req.is_expired()
        result = mgr.approve(req.request_id)
        assert result is None

    def test_list_pending(self):
        mgr = ApprovalManager()
        mgr.create_request("tool_1")
        mgr.create_request("tool_2")
        req3 = mgr.create_request("tool_3")
        mgr.approve(req3.request_id)
        pending = mgr.list_pending()
        assert len(pending) == 2

    def test_list_all(self):
        mgr = ApprovalManager()
        mgr.create_request("tool_1")
        req2 = mgr.create_request("tool_2")
        mgr.deny(req2.request_id)
        all_reqs = mgr.list_all()
        assert len(all_reqs) == 2

    def test_register_tool(self):
        mgr = ApprovalManager()
        mgr.register_tool("apply_fix_script")
        assert mgr.requires_approval("apply_fix_script")
        assert not mgr.requires_approval("sim_start")

    def test_get_stats(self):
        mgr = ApprovalManager()
        mgr.register_tool("fix_tool")
        mgr.create_request("fix_tool")
        req2 = mgr.create_request("fix_tool")
        mgr.approve(req2.request_id)
        stats = mgr.get_stats()
        assert stats["total_requests"] == 2
        assert "pending" in stats["by_status"]

    def test_get_request(self):
        mgr = ApprovalManager()
        req = mgr.create_request("tool_1")
        found = mgr.get_request(req.request_id)
        assert found is not None
        assert found.tool_name == "tool_1"
        assert mgr.get_request("nonexistent") is None

    def test_request_to_dict(self):
        mgr = ApprovalManager()
        req = mgr.create_request("tool_1", instance="primary", description="test")
        d = req.to_dict()
        assert d["tool_name"] == "tool_1"
        assert d["status"] == "pending"
