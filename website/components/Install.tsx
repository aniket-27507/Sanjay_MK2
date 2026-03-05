"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const tabs = [
  {
    id: "local",
    label: "Local (stdio)",
    content: `# Clone & install
git clone https://github.com/your-org/isaac-mcp.git
cd isaac-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -U pip
pip install -e '.[dev]'

# Register with Claude Code
claude mcp add --transport stdio \\
  --scope project isaac-sim -- \\
  .venv/bin/python -m isaac_mcp.server

# Or generate a Cursor deeplink
.venv/bin/python scripts/generate_cursor_deeplink.py \\
  --name isaac-sim \\
  --remote-url 'http://localhost:8000/mcp'`,
  },
  {
    id: "remote",
    label: "Remote (HTTPS)",
    content: `# Start the server in streamable-http mode
ISAAC_MCP_TRANSPORT=streamable-http \\
ISAAC_MCP_HOST=0.0.0.1 \\
ISAAC_MCP_PORT=8000 \\
ISAAC_MCP_PUBLIC_BASE_URL='https://mcp.your-domain.com' \\
.venv/bin/python -m isaac_mcp.server

# Health check
curl -fsS http://127.0.0.1:8000/healthz

# Enable mutations (optional)
export ISAAC_MCP_ENABLE_MUTATIONS=true`,
  },
];

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className="absolute right-3 top-3 rounded-md border border-border bg-panel px-2.5 py-1 text-xs text-muted transition-colors hover:text-foreground"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

export default function Install() {
  const [active, setActive] = useState("local");
  const tab = tabs.find((t) => t.id === active)!;

  return (
    <section id="install" className="relative py-28">
      <div className="mx-auto max-w-4xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Get <span className="text-accent">Started</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            Up and running in minutes. Requires Python 3.10+ and NVIDIA Isaac
            Sim.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mt-12"
        >
          {/* Tabs */}
          <div className="flex gap-1 rounded-xl border border-border bg-surface p-1">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setActive(t.id)}
                className={`flex-1 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors ${
                  active === t.id
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:text-foreground"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Code block */}
          <div className="relative mt-4 overflow-hidden rounded-xl border border-border bg-surface">
            <CopyButton text={tab.content} />
            <pre className="overflow-x-auto p-6 font-mono text-sm leading-relaxed text-muted">
              <code>{tab.content}</code>
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
