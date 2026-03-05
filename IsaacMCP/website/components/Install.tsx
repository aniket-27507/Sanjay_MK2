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

const tabs = [
  {
    id: "quick",
    label: "QUICK START",
    content: `# Install IsaacMCP
pip install isaac-mcp

# Navigate to your project
cd /path/to/your-robotics-project

# Auto-detect, configure, and register
isaac-mcp init
isaac-mcp register --cursor  # or --claude

# Start the server
isaac-mcp start`,
  },
  {
    id: "local",
    label: "FROM SOURCE",
    content: `# Clone & install
git clone https://github.com/yanitedhacker/IsaacMCP.git
cd IsaacMCP

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all optional dependencies
pip install -e '.[all,dev]'

# Run the doctor to check connectivity
isaac-mcp doctor

# Register with your IDE
isaac-mcp register --cursor  # or --claude`,
  },
  {
    id: "docker",
    label: "DOCKER",
    content: `# Initialize with Docker support
cd /path/to/your-project
isaac-mcp init --docker

# Start alongside your project's Docker stack
docker compose \\
  -f docker-compose.yml \\
  -f docker-compose.isaac-mcp.yml \\
  up

# Connect your IDE to http://localhost:8000/mcp`,
  },
];

export default function Install() {
  const [active, setActive] = useState("local");
  const tab = tabs.find((t) => t.id === active)!;

  return (
    <section id="install" className="section-technical relative py-28">
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
            CHAPTER 06 — INSTALLATION
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            Get <span className="text-accent">Started</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            Up and running in minutes. Requires Python 3.10+ and NVIDIA Isaac Sim.
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
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setActive(t.id)}
                className={`flex-1 px-4 py-3 transition-colors ${active === t.id
                    ? "bg-cobalt-light text-accent border-b-2 border-accent"
                    : "bg-surface text-muted hover:text-foreground"
                  }`}
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.65rem",
                  letterSpacing: "0.1em",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Code block */}
          <div className="code-block relative mt-0 p-6">
            <CopyButton text={tab.content} />
            <pre className="overflow-x-auto leading-relaxed">
              <code>{tab.content}</code>
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
