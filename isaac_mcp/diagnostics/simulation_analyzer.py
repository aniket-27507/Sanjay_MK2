"""Cross-correlate telemetry, logs, and scene data into structured diagnoses."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Issue:
    category: str
    description: str
    severity: str
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SuggestedFix:
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high


@dataclass(slots=True)
class Diagnosis:
    issues: list[Issue] = field(default_factory=list)
    root_cause: str = ""
    confidence: float = 0.0
    category: str = "unknown"
    suggested_fixes: list[SuggestedFix] = field(default_factory=list)
    telemetry_snapshot: dict[str, Any] = field(default_factory=dict)
    log_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [
                {
                    "category": i.category,
                    "description": i.description,
                    "severity": i.severity,
                    "evidence": i.evidence,
                }
                for i in self.issues
            ],
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "category": self.category,
            "suggested_fixes": [
                {
                    "description": f.description,
                    "parameters": f.parameters,
                    "risk_level": f.risk_level,
                }
                for f in self.suggested_fixes
            ],
            "telemetry_snapshot": self.telemetry_snapshot,
            "log_evidence": self.log_evidence,
        }


@dataclass(slots=True)
class ClassifiedError:
    category: str
    severity: str
    description: str
    fix: str
    log_line: str


@dataclass(slots=True)
class PhysicsIssue:
    issue_type: str
    description: str
    severity: str
    details: dict[str, Any] = field(default_factory=dict)


# Priority ordering for root cause selection
_SEVERITY_PRIORITY = {"critical": 0, "error": 1, "warning": 2, "info": 3, "unknown": 4}


class SimulationAnalyzer:
    """Cross-correlates telemetry, logs, and scene data to produce structured diagnoses."""

    def __init__(
        self,
        error_patterns: list[dict[str, str]],
        knowledge_base: Any | None = None,
    ):
        self._error_patterns = error_patterns
        self._knowledge_base = knowledge_base

    def analyze(
        self,
        telemetry: dict[str, Any],
        log_entries: list[dict[str, Any]],
        scene_data: dict[str, Any],
    ) -> Diagnosis:
        """Cross-correlate inputs into a structured diagnosis."""
        classified_errors = self._classify_errors(log_entries)
        physics_issues = self._detect_physics_issues(telemetry)

        diagnosis = self._correlate_findings(classified_errors, physics_issues, scene_data)
        diagnosis.telemetry_snapshot = _safe_snapshot(telemetry)
        diagnosis.suggested_fixes = self._suggest_fixes(diagnosis)

        if self._knowledge_base is not None:
            self._enrich_from_knowledge_base(diagnosis)

        return diagnosis

    def _classify_errors(self, log_entries: list[dict[str, Any]]) -> list[ClassifiedError]:
        classified: list[ClassifiedError] = []
        for entry in log_entries:
            raw = str(entry.get("raw_line", entry.get("message", "")))
            for pattern in self._error_patterns:
                if re.search(pattern["pattern"], raw, re.IGNORECASE):
                    classified.append(ClassifiedError(
                        category=pattern["category"],
                        severity=pattern["severity"],
                        description=pattern["description"],
                        fix=pattern["fix"],
                        log_line=raw,
                    ))
                    break
        return classified

    def _detect_physics_issues(self, telemetry: dict[str, Any]) -> list[PhysicsIssue]:
        issues: list[PhysicsIssue] = []

        robots = telemetry.get("robots", [])
        if isinstance(robots, list):
            for robot in robots:
                if not isinstance(robot, dict):
                    continue
                issues.extend(self._check_robot_state(robot))

        physics = telemetry.get("physics", {})
        if isinstance(physics, dict):
            issues.extend(self._check_physics_state(physics))

        return issues

    def _check_robot_state(self, robot: dict[str, Any]) -> list[PhysicsIssue]:
        issues: list[PhysicsIssue] = []
        name = robot.get("name", "unknown")

        # Check for NaN/Inf in position
        position = robot.get("position")
        if isinstance(position, (list, tuple)):
            if any(_is_nan_or_inf(v) for v in position):
                issues.append(PhysicsIssue(
                    issue_type="nan_position",
                    description=f"Robot '{name}' has NaN/Inf position values",
                    severity="critical",
                    details={"robot": name, "position": position},
                ))

        # Check for fallen robot (z position below threshold)
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            z = position[2] if isinstance(position[2], (int, float)) else None
            if z is not None and not _is_nan_or_inf(z) and z < -1.0:
                issues.append(PhysicsIssue(
                    issue_type="robot_fell",
                    description=f"Robot '{name}' appears to have fallen (z={z:.2f})",
                    severity="error",
                    details={"robot": name, "z_position": z},
                ))

        # Check for excessive velocity
        velocity = robot.get("velocity")
        if isinstance(velocity, (list, tuple)):
            speed = sum(v ** 2 for v in velocity if isinstance(v, (int, float))) ** 0.5
            if speed > 100.0:
                issues.append(PhysicsIssue(
                    issue_type="excessive_velocity",
                    description=f"Robot '{name}' has excessive velocity ({speed:.1f})",
                    severity="warning",
                    details={"robot": name, "speed": speed},
                ))

        return issues

    def _check_physics_state(self, physics: dict[str, Any]) -> list[PhysicsIssue]:
        issues: list[PhysicsIssue] = []

        contacts = physics.get("contacts")
        if isinstance(contacts, list) and len(contacts) > 50:
            issues.append(PhysicsIssue(
                issue_type="excessive_contacts",
                description=f"High number of physics contacts ({len(contacts)})",
                severity="warning",
                details={"contact_count": len(contacts)},
            ))

        collisions = physics.get("collisions")
        if isinstance(collisions, list) and len(collisions) > 0:
            issues.append(PhysicsIssue(
                issue_type="active_collisions",
                description=f"{len(collisions)} active collision(s) detected",
                severity="warning",
                details={"collision_count": len(collisions)},
            ))

        return issues

    def _correlate_findings(
        self,
        errors: list[ClassifiedError],
        physics_issues: list[PhysicsIssue],
        scene_data: dict[str, Any],
    ) -> Diagnosis:
        all_issues: list[Issue] = []

        for err in errors:
            all_issues.append(Issue(
                category=err.category,
                description=err.description,
                severity=err.severity,
                evidence=[err.log_line],
            ))

        for pi in physics_issues:
            all_issues.append(Issue(
                category="physics",
                description=pi.description,
                severity=pi.severity,
                evidence=[str(pi.details)],
            ))

        if not all_issues:
            return Diagnosis(
                issues=[],
                root_cause="no_issues_detected",
                confidence=1.0,
                category="healthy",
                log_evidence=[],
            )

        # Sort by severity and pick the most critical as root cause
        all_issues.sort(key=lambda i: _SEVERITY_PRIORITY.get(i.severity, 4))
        root = all_issues[0]

        # Compute confidence based on evidence count and severity
        confidence = self._compute_confidence(all_issues, errors, physics_issues)

        # Determine primary category
        category_counts: dict[str, int] = {}
        for issue in all_issues:
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        primary_category = max(category_counts, key=lambda k: category_counts[k])

        log_evidence = [err.log_line for err in errors]

        return Diagnosis(
            issues=all_issues,
            root_cause=root.description,
            confidence=confidence,
            category=primary_category,
            log_evidence=log_evidence,
        )

    def _compute_confidence(
        self,
        all_issues: list[Issue],
        errors: list[ClassifiedError],
        physics_issues: list[PhysicsIssue],
    ) -> float:
        if not all_issues:
            return 1.0

        score = 0.0

        # More evidence sources = higher confidence
        has_log_evidence = len(errors) > 0
        has_physics_evidence = len(physics_issues) > 0

        if has_log_evidence and has_physics_evidence:
            score = 0.9  # Corroborated across sources
        elif has_log_evidence:
            score = 0.7
        elif has_physics_evidence:
            score = 0.6
        else:
            score = 0.3

        # Adjust by severity
        worst = all_issues[0].severity
        if worst == "critical":
            score = min(score + 0.1, 1.0)

        return round(score, 2)

    def _suggest_fixes(self, diagnosis: Diagnosis) -> list[SuggestedFix]:
        fixes: list[SuggestedFix] = []
        seen: set[str] = set()

        for issue in diagnosis.issues:
            fix_desc = self._fix_for_issue(issue)
            if fix_desc and fix_desc not in seen:
                seen.add(fix_desc)
                fixes.append(SuggestedFix(
                    description=fix_desc,
                    risk_level=_risk_for_severity(issue.severity),
                ))

        return fixes

    def _fix_for_issue(self, issue: Issue) -> str:
        # Try to find a matching error pattern with fix guidance
        for pattern in self._error_patterns:
            if pattern.get("category") == issue.category:
                if pattern.get("description") == issue.description:
                    return pattern.get("fix", "")

        # Fallback fixes based on physics issue types
        if "NaN/Inf position" in issue.description:
            return "Reset simulation and check object masses, colliders, and initial overlaps."
        if "fallen" in issue.description:
            return "Check robot center of mass, joint torques, and ground collision setup."
        if "excessive velocity" in issue.description:
            return "Reduce applied forces or check for physics instability."
        if "excessive contacts" in issue.description:
            return "Simplify collision meshes or review contact filtering."

        return ""

    def _enrich_from_knowledge_base(self, diagnosis: Diagnosis) -> None:
        if self._knowledge_base is None:
            return
        for issue in diagnosis.issues:
            try:
                entries = self._knowledge_base.query(issue.description, issue.category)
                for entry in entries:
                    fix = entry.get("fix_applied", "")
                    if fix and fix not in {f.description for f in diagnosis.suggested_fixes}:
                        diagnosis.suggested_fixes.append(SuggestedFix(
                            description=f"[From knowledge base] {fix}",
                            risk_level="low",
                        ))
            except Exception:
                pass


def _is_nan_or_inf(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return math.isnan(value) or math.isinf(value)


def _risk_for_severity(severity: str) -> str:
    if severity == "critical":
        return "high"
    if severity == "error":
        return "medium"
    return "low"


def _safe_snapshot(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Create a JSON-safe snapshot of telemetry data."""
    snapshot: dict[str, Any] = {}
    for key in ("robots", "physics", "performance", "active_faults", "scenario"):
        if key in telemetry:
            snapshot[key] = telemetry[key]
    return snapshot
