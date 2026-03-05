"""
Project Sanjay Mk2 - Test Configuration
========================================
Handles optional dependency detection so tests produce clear skip
messages instead of opaque collection errors.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that require unavailable optional dependencies."""

    _optional_deps = {
        "torch": ["change_detection", "world_and_sensors"],
        "mujoco": ["mujoco"],
        "ultralytics": ["change_detection"],
    }

    missing = {}
    for pkg, patterns in _optional_deps.items():
        try:
            __import__(pkg)
        except ImportError:
            missing[pkg] = patterns

    if not missing:
        return

    for item in items:
        for pkg, patterns in missing.items():
            if any(p in item.nodeid for p in patterns):
                item.add_marker(
                    pytest.mark.skip(reason=f"{pkg} not installed")
                )
                break
