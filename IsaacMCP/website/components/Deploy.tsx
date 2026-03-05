"use client";

import { useState } from "react";
import { motion } from "framer-motion";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className="btn-bevel absolute right-3 top-3"
      style={{ fontSize: "0.6rem", padding: "0.35rem 0.75rem" }}
    >
      {copied ? "COPIED" : "COPY"}
    </button>
  );
}

const modes = [
  {
    id: "local",
    label: "LOCAL",
    description:
      "Run the server locally over stdio. Easiest way to get started — just register and go.",
    code: `# Claude Code CLI
claude mcp add --transport stdio \\
  --scope project isaac-sim -- \\
  .venv/bin/python -m isaac_mcp.server

# Or for Cursor, generate a deeplink
.venv/bin/python scripts/generate_cursor_deeplink.py \\
  --name isaac-sim \\
  --remote-url 'http://localhost:8000/mcp'`,
  },
  {
    id: "remote",
    label: "REMOTE",
    description:
      "Host IsaacMCP near your GPU boxes and query it securely from anywhere. Supports OAuth bearer-token verification.",
    code: `ISAAC_MCP_TRANSPORT=streamable-http \\
ISAAC_MCP_HOST=0.0.0.1 \\
ISAAC_MCP_PORT=8000 \\
ISAAC_MCP_PUBLIC_BASE_URL='https://mcp.your-domain.com' \\
ISAAC_MCP_AUTH_ENABLED=true \\
.venv/bin/python -m isaac_mcp.server`,
  },
  {
    id: "cloudflare",
    label: "CF TUNNEL",
    description:
      "Zero-trust deployment via Cloudflare Tunnel. Keep the runtime close to Isaac services, expose only HTTPS.",
    code: `# 1. Create tunnel
cloudflared tunnel login
cloudflared tunnel create isaac-mcp

# 2. Route DNS
cloudflared tunnel route dns isaac-mcp mcp.your-domain.com

# 3. Run tunnel
cloudflared tunnel run --config /etc/cloudflared/config.yml

# 4. Enable as systemd services
sudo systemctl enable --now isaac-mcp
sudo systemctl enable --now cloudflared`,
  },
];

export default function Deploy() {
  const [active, setActive] = useState("local");
  const mode = modes.find((m) => m.id === active)!;

  return (
    <section id="deploy" className="section-technical relative py-28">
      <div className="divider-check mb-20" />

      <div className="mx-auto max-w-4xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.5 }}
          className="text-center"
        >
          <span
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              fontSize: "0.65rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "var(--color-accent)",
            }}
          >
            CHAPTER 08 — DEPLOYMENT
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            Deployment <span className="text-accent">Modes</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            Run locally for quick iteration or deploy remotely for enterprise scale.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45, delay: 0.1 }}
          className="mt-12"
        >
          {/* Tabs */}
          <div className="flex border border-border">
            {modes.map((m) => (
              <button
                key={m.id}
                onClick={() => setActive(m.id)}
                className={`flex-1 px-3 py-3 transition-colors ${active === m.id
                    ? "bg-cobalt-light text-accent border-b-2 border-accent"
                    : "bg-surface text-muted hover:text-foreground"
                  }`}
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.65rem",
                  letterSpacing: "0.1em",
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          <p className="mt-6 text-sm text-muted">{mode.description}</p>

          <div className="code-block relative mt-4 p-6">
            <CopyButton text={mode.code} />
            <pre className="overflow-x-auto leading-relaxed">
              <code>{mode.code}</code>
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
