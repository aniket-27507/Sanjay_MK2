#!/usr/bin/env python3
"""Generate Add-to-Cursor deeplink and install URL for Isaac MCP."""

from __future__ import annotations

import argparse
import json

from isaac_mcp.onboarding import (
    build_cursor_deeplink,
    build_cursor_install_url,
    build_local_cursor_stdio_config,
    build_remote_cursor_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Cursor one-click install links")
    parser.add_argument("--name", default="isaac-sim", help="Cursor MCP server name")
    parser.add_argument("--remote-url", help="Remote MCP URL, e.g. https://mcp.example.com/mcp")
    parser.add_argument("--stdio-command", help="Local stdio command executable")
    parser.add_argument("--stdio-args", nargs="*", default=[], help="Local stdio args")
    parser.add_argument("--stdio-env", nargs="*", default=[], help="Env in KEY=VALUE format")
    parser.add_argument("--json-only", action="store_true", help="Print only generated config JSON")
    return parser.parse_args()


def parse_env(entries: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid env entry '{item}'. Use KEY=VALUE")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def main() -> None:
    args = parse_args()

    if args.remote_url:
        config = build_remote_cursor_config(args.name, args.remote_url)
    elif args.stdio_command:
        config = build_local_cursor_stdio_config(
            args.name,
            args.stdio_command,
            args.stdio_args,
            parse_env(args.stdio_env),
        )
    else:
        raise SystemExit("Provide either --remote-url or --stdio-command")

    if args.json_only:
        print(json.dumps(config, indent=2, ensure_ascii=True))
        return

    deeplink = build_cursor_deeplink(args.name, config)
    install_url = build_cursor_install_url(args.name, config)

    print("Config JSON:")
    print(json.dumps(config, indent=2, ensure_ascii=True))
    print()
    print(f"Cursor deeplink: {deeplink}")
    print(f"Cursor HTTPS install URL: {install_url}")


if __name__ == "__main__":
    main()
