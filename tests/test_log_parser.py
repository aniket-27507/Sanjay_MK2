from __future__ import annotations

from isaac_mcp.log_parser import match_error_patterns, parse_log_line, parse_log_lines, summarize_errors


def test_parse_log_line_valid() -> None:
    entry = parse_log_line("2026-01-01 10:10:10.123 [Error] [omni.physics] PhysX Error happened")

    assert entry.timestamp == "2026-01-01 10:10:10.123"
    assert entry.severity == "Error"
    assert entry.source == "omni.physics"


def test_parse_log_line_invalid() -> None:
    entry = parse_log_line("not a valid log line")

    assert entry.severity == "unknown"
    assert entry.message == "not a valid log line"


def test_match_and_summary() -> None:
    entries = parse_log_lines(
        [
            "2026-01-01 10:10:10.123 [Error] [omni.physics] PhysX Error happened",
            "2026-01-01 10:10:11.123 [Warning] [omni.usd] Warning: attribute has no authored value",
        ]
    )

    matched = match_error_patterns(entries)
    summary = summarize_errors(entries)

    assert len(matched) >= 1
    assert "Found" in summary
