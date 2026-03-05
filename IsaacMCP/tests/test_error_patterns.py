from __future__ import annotations

import re

from isaac_mcp.error_patterns import ERROR_PATTERNS


def test_error_patterns_have_required_fields() -> None:
    required = {"category", "pattern", "severity", "description", "fix"}
    for pattern in ERROR_PATTERNS:
        assert required.issubset(pattern.keys())


def test_error_pattern_matches_physx() -> None:
    text = "[Error] [omni.physics] PhysX Error: body invalid"
    assert any(re.search(p["pattern"], text, re.IGNORECASE) for p in ERROR_PATTERNS if p["category"] == "physics")
