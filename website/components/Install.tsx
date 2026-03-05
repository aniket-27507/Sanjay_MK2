"use client";

import { useState } from "react";
import { motion } from "framer-motion";

const tabs = [
  {
    id: "quick",
    label: "Quick Start",
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
    label: "From Source",
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
    label: "Docker",
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
