"""Parser and matcher utilities for Isaac Sim Kit logs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from isaac_mcp.error_patterns import ERROR_PATTERNS

_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+\[(\w+)\]\s+\[([^\]]+)\]\s+(.+)$"
)


@dataclass(slots=True)
class LogEntry:
    timestamp: str
    severity: str
    source: str
    message: str
    raw_line: str
    matched_pattern: dict[str, str] | None = None


def parse_log_line(line: str) -> LogEntry:
    text = line.rstrip("\n")
    match = _LOG_LINE_RE.match(text.strip())
    if not match:
        return LogEntry(
            timestamp="",
            severity="unknown",
            source="",
            message=text.strip(),
            raw_line=text.strip(),
        )

    return LogEntry(
        timestamp=match.group(1),
        severity=match.group(2),
        source=match.group(3),
        message=match.group(4),
        raw_line=text.strip(),
    )


def parse_log_lines(lines: list[str]) -> list[LogEntry]:
    return [parse_log_line(line) for line in lines if line and line.strip()]


def match_error_patterns(entries: list[LogEntry]) -> list[LogEntry]:
    matched: list[LogEntry] = []
    for entry in entries:
        for pattern in ERROR_PATTERNS:
            if re.search(pattern["pattern"], entry.raw_line, re.IGNORECASE):
                entry.matched_pattern = pattern
                matched.append(entry)
                break
    return matched


def summarize_errors(entries: list[LogEntry]) -> str:
    matched = match_error_patterns(entries)
    if not matched:
        return "No known error patterns detected in the logs."

    categories: dict[str, list[LogEntry]] = {}
    for entry in matched:
        category = entry.matched_pattern["category"] if entry.matched_pattern else "unknown"
        categories.setdefault(category, []).append(entry)

    lines: list[str] = [f"Found {len(matched)} matched error(s) across {len(categories)} categories."]

    for category in sorted(categories.keys()):
        category_entries = categories[category]
        lines.append(f"\n## {category.upper().replace('_', ' ')} ({len(category_entries)})")
        seen_descriptions: set[str] = set()

        for entry in category_entries:
            pattern = entry.matched_pattern or {}
            description = pattern.get("description", "Unknown issue")
            if description in seen_descriptions:
                continue
            seen_descriptions.add(description)
            lines.append(f"- {description}")
            lines.append(f"  Severity: {pattern.get('severity', 'unknown')}")
            lines.append(f"  Last seen: {entry.timestamp or 'unknown'}")
            lines.append(f"  Fix: {pattern.get('fix', 'No fix guidance available')}")

    return "\n".join(lines)
