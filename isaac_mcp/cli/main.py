"""isaac-mcp CLI entrypoint."""

from __future__ import annotations

import argparse
import sys


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="isaac-mcp",
        description="IsaacMCP — The AI Copilot for Robotics Simulation",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Auto-detect project and generate config")
    init_p.add_argument("--project-dir", default=".", help="Project root directory")
    init_p.add_argument("--docker", action="store_true", help="Also generate Docker files")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing config")

    # start
    start_p = sub.add_parser("start", help="Start the MCP server")
    start_p.add_argument("--config", default=None, help="Config path (auto-detected if not given)")
    start_p.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default=None)

    # register
    reg_p = sub.add_parser("register", help="Register with an IDE")
    reg_p.add_argument("--cursor", action="store_true", help="Register with Cursor")
    reg_p.add_argument("--claude", action="store_true", help="Register with Claude Code")
    reg_p.add_argument("--claude-desktop", action="store_true", help="Register with Claude Desktop")
    reg_p.add_argument("--name", default="isaac-sim", help="Server name")
    reg_p.add_argument("--url", default="http://localhost:8000/mcp", help="Remote MCP URL")

    # scaffold
    scaffold_p = sub.add_parser("scaffold", help="Generate a custom plugin template")
    scaffold_p.add_argument("--name", required=True, help="Plugin name")
    scaffold_p.add_argument("--from-class", default=None, help="Python class to introspect (module:Class)")
    scaffold_p.add_argument("--output-dir", default=".", help="Output directory")

    # doctor
    sub.add_parser("doctor", help="Diagnose connectivity and dependencies")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "init":
        from isaac_mcp.cli.detect import run_init
        run_init(args.project_dir, docker=args.docker, force=args.force)

    elif args.command == "start":
        from isaac_mcp.server import main as server_main
        if args.config:
            sys.argv = ["isaac-mcp", "--config", args.config]
            if args.transport:
                sys.argv += ["--transport", args.transport]
        elif args.transport:
            sys.argv = ["isaac-mcp", "--transport", args.transport]
        else:
            sys.argv = ["isaac-mcp"]
        server_main()

    elif args.command == "register":
        from isaac_mcp.cli.register import run_register
        run_register(
            cursor=args.cursor,
            claude=args.claude,
            claude_desktop=args.claude_desktop,
            name=args.name,
            url=args.url,
        )

    elif args.command == "scaffold":
        from isaac_mcp.cli.scaffold import run_scaffold
        run_scaffold(name=args.name, from_class=args.from_class, output_dir=args.output_dir)

    elif args.command == "doctor":
        from isaac_mcp.cli.doctor import run_doctor
        run_doctor()


if __name__ == "__main__":
    cli()
