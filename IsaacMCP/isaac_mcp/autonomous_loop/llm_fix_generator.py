"""LLM-assisted fix generation for issues not covered by templates.

When no template matches a diagnosis, this module constructs a structured
prompt with diagnosis context for the LLM to generate a Kit API fix script.

The LLM fix is returned as a FixProposal with source="llm_generated" and
risk_level="high" to ensure human review before execution.
"""

from __future__ import annotations

import json
from typing import Any

from isaac_mcp.autonomous_loop.fix_generator import FixProposal

# Maximum script size the LLM can generate (safety cap)
_MAX_SCRIPT_SIZE = 10_000  # 10KB

# Prompt template for LLM fix generation
_LLM_FIX_PROMPT = """You are an NVIDIA Isaac Sim expert. A simulation has encountered an issue
that does not match any known fix template.

## Diagnosis
{diagnosis_summary}

## Root Cause
{root_cause}

## Issues Found
{issues_list}

## Telemetry Snapshot
{telemetry}

## Log Evidence
{log_evidence}

## Knowledge Graph Context
{knowledge_context}

## Instructions
Generate a Python script that uses the NVIDIA Kit API to fix this issue.
The script will be executed inside Isaac Sim's Python runtime.

Requirements:
- Use only `import` statements for Kit/Isaac modules (omni.*, carb, pxr)
- Keep the script under 50 lines
- Include a `print()` at the end confirming what was changed
- Do NOT call `sys.exit()` or shutdown functions
- Do NOT modify files on disk
- Focus on the most likely fix based on the diagnosis

Return ONLY the Python script, no markdown fences or explanations."""


class LlmFixGenerator:
    """Generate fix proposals using LLM when templates don't match.

    This class does NOT call an LLM directly -- it constructs the prompt
    and context that the MCP tool layer passes to the LLM for generation.
    The LLM response is then validated and packaged as a FixProposal.
    """

    def build_fix_prompt(
        self,
        diagnosis_dict: dict[str, Any],
        knowledge_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a structured prompt for LLM fix generation.

        Returns a dict with:
        - prompt: The full prompt text
        - context: Structured context data
        - metadata: Info about the request
        """
        issues = diagnosis_dict.get("issues", [])
        issues_text = "\n".join(
            f"- [{issue.get('severity', 'unknown')}] {issue.get('category', '')}: {issue.get('description', '')}"
            for issue in issues
        )

        telemetry = diagnosis_dict.get("telemetry_snapshot", {})
        telemetry_text = json.dumps(telemetry, indent=2, default=str)[:2000]

        log_evidence = diagnosis_dict.get("log_evidence", [])
        log_text = "\n".join(log_evidence[:10])

        knowledge_text = "No prior knowledge available."
        if knowledge_context:
            knowledge_entries = []
            for entry in knowledge_context[:5]:
                fix_label = entry.get("fix_label", entry.get("fix_applied", ""))
                success_rate = entry.get("success_rate", entry.get("confidence", 0))
                knowledge_entries.append(
                    f"- {fix_label} (success rate: {success_rate:.0%})"
                )
            knowledge_text = "\n".join(knowledge_entries)

        prompt = _LLM_FIX_PROMPT.format(
            diagnosis_summary=diagnosis_dict.get("root_cause", "Unknown"),
            root_cause=diagnosis_dict.get("root_cause", "Unknown"),
            issues_list=issues_text or "No specific issues listed",
            telemetry=telemetry_text,
            log_evidence=log_text or "No log evidence",
            knowledge_context=knowledge_text,
        )

        return {
            "prompt": prompt,
            "context": {
                "diagnosis": diagnosis_dict,
                "knowledge": knowledge_context or [],
            },
            "metadata": {
                "issue_count": len(issues),
                "has_telemetry": bool(telemetry),
                "has_log_evidence": bool(log_evidence),
                "has_knowledge_context": bool(knowledge_context),
            },
        }

    def validate_and_create_proposal(
        self,
        llm_script: str,
        diagnosis_dict: dict[str, Any],
    ) -> FixProposal | None:
        """Validate an LLM-generated script and create a FixProposal.

        Returns None if the script fails validation.
        """
        if not llm_script or not llm_script.strip():
            return None

        script = llm_script.strip()

        # Strip markdown code fences if present
        if script.startswith("```"):
            lines = script.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            script = "\n".join(lines).strip()

        # Safety validation
        issues = self._validate_script(script)
        if issues:
            return None

        root_cause = diagnosis_dict.get("root_cause", "Unknown issue")
        return FixProposal(
            description=f"[LLM-generated] Fix for: {root_cause[:100]}",
            parameters={"generated": True},
            kit_script=script,
            risk_level="high",
            source="llm_generated",
        )

    def _validate_script(self, script: str) -> list[str]:
        """Validate a script for safety. Returns list of issues (empty = valid)."""
        issues: list[str] = []

        if len(script) > _MAX_SCRIPT_SIZE:
            issues.append(f"Script exceeds {_MAX_SCRIPT_SIZE} byte limit")

        # Must contain at least one import
        if "import " not in script:
            issues.append("Script must contain at least one import statement")

        # Blocked patterns
        _BLOCKED_PATTERNS = [
            "sys.exit",
            "os.remove",
            "os.unlink",
            "shutil.rmtree",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "exec(",
            "eval(",
            "__import__",
            "open(",
            "os.system",
            "shutdown",
            "os.kill",
        ]
        for pattern in _BLOCKED_PATTERNS:
            if pattern in script:
                issues.append(f"Blocked pattern found: {pattern}")

        return issues
