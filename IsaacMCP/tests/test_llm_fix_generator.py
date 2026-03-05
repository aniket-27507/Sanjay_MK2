"""Tests for the LLM fix generator."""

import pytest

from isaac_mcp.autonomous_loop.llm_fix_generator import LlmFixGenerator


@pytest.fixture
def gen():
    return LlmFixGenerator()


def test_build_fix_prompt(gen):
    diagnosis = {
        "root_cause": "Unknown sensor drift",
        "issues": [
            {"description": "Sensor reading anomaly", "severity": "warning", "category": "sensor"},
        ],
        "telemetry_snapshot": {"robots": [{"name": "bot"}]},
        "log_evidence": ["WARNING: sensor drift detected"],
    }
    result = gen.build_fix_prompt(diagnosis)
    assert "prompt" in result
    assert "context" in result
    assert "metadata" in result
    assert "Unknown sensor drift" in result["prompt"]


def test_build_fix_prompt_with_knowledge_context(gen):
    diagnosis = {"root_cause": "Error X", "issues": []}
    knowledge = [
        {"fix_label": "Fix A", "success_rate": 0.75},
        {"fix_label": "Fix B", "success_rate": 0.5},
    ]
    result = gen.build_fix_prompt(diagnosis, knowledge_context=knowledge)
    assert "Fix A" in result["prompt"]
    assert result["metadata"]["has_knowledge_context"] is True


def test_validate_and_create_proposal_valid(gen):
    script = (
        "import carb\n"
        "settings = carb.settings.get_settings()\n"
        "settings.set('/physics/timeStepsPerSecond', 120)\n"
        "print('Fix applied')\n"
    )
    proposal = gen.validate_and_create_proposal(script, {"root_cause": "Physics error"})
    assert proposal is not None
    assert proposal.source == "llm_generated"
    assert proposal.risk_level == "high"
    assert "Physics error" in proposal.description


def test_validate_strips_markdown_fences(gen):
    script = "```python\nimport carb\nprint('ok')\n```"
    proposal = gen.validate_and_create_proposal(script, {"root_cause": "test"})
    assert proposal is not None
    assert "```" not in proposal.kit_script


def test_validate_rejects_empty(gen):
    assert gen.validate_and_create_proposal("", {"root_cause": "test"}) is None
    assert gen.validate_and_create_proposal("   ", {"root_cause": "test"}) is None


def test_validate_rejects_no_import(gen):
    script = "print('no import')"
    assert gen.validate_and_create_proposal(script, {"root_cause": "test"}) is None


def test_validate_rejects_dangerous_patterns(gen):
    scripts = [
        "import os\nos.system('rm -rf /')",
        "import os\nos.remove('/etc/passwd')",
        "import subprocess\nsubprocess.run(['ls'])",
        "import sys\nsys.exit(1)",
        "import os\nexec('bad code')",
    ]
    for script in scripts:
        proposal = gen.validate_and_create_proposal(script, {"root_cause": "test"})
        assert proposal is None, f"Should reject: {script[:40]}"


def test_validate_rejects_oversized_script(gen):
    script = "import carb\n" + "x = 1\n" * 10000
    assert gen.validate_and_create_proposal(script, {"root_cause": "test"}) is None


def test_expanded_templates():
    """Verify fix_generator now has 20+ templates."""
    from isaac_mcp.autonomous_loop.fix_generator import FixGenerator
    gen = FixGenerator()
    count = gen.get_template_count()
    assert count >= 20, f"Expected 20+ templates, got {count}"
