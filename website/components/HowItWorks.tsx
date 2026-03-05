"use client";

import { motion } from "framer-motion";

const steps = [
  {
    number: "01",
    title: "Connect",
    description:
      "Link your LLM (Claude, Cursor, Claude Code) to the IsaacMCP server via stdio, WebSocket, or HTTPS.",
    detail: "Supports local and remote deployment with OAuth security.",
  },
  {
    number: "02",
    title: "Command",
    description:
      'Use natural language to instruct your AI assistant. Say "Analyze why the robot keeps falling" or "Run a parameter sweep."',
    detail: "54 semantic tools are automatically discovered by your LLM.",
  },
  {
    number: "03",
    title: "Control",
    description:
      "IsaacMCP translates your intent into Kit API calls, physics adjustments, experiment campaigns, and live script injection.",
    detail:
      "Results flow back through MCP resources for real-time monitoring.",
  },
];

const archNodes = [
  { label: "Claude / Cursor", sub: "LLM" },
  { label: "MCP Protocol", sub: "JSON-RPC" },
  { label: "IsaacMCP Server", sub: "54 Tools" },
  { label: "NVIDIA Isaac Sim", sub: "PhysX · Sensors · Kit API" },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="relative py-28">
      <div className="mx-auto max-w-7xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            How It <span className="text-accent">Works</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            Three steps from natural language to full simulation control.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="mt-16 grid gap-8 md:grid-cols-3">
          {steps.map((s, i) => (
            <motion.div
              key={s.number}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15, duration: 0.5 }}
              className="relative rounded-2xl border border-border bg-panel p-8"
            >
              <span className="text-5xl font-black text-accent/20">
                {s.number}
              </span>
              <h3 className="mt-2 text-xl font-bold">{s.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                {s.description}
              </p>
              <p className="mt-2 text-xs text-accent/70">{s.detail}</p>
              {i < steps.length - 1 && (
                <div className="absolute -right-4 top-1/2 hidden text-2xl text-accent/40 md:block">
                  →
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Architecture diagram */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-20"
        >
          <h3 className="mb-8 text-center text-lg font-semibold text-muted">
            Architecture
          </h3>
          <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center sm:gap-0">
            {archNodes.map((node, i) => (
              <div key={node.label} className="flex items-center gap-3">
                <div className="rounded-xl border border-border bg-surface px-6 py-4 text-center">
                  <div className="text-sm font-semibold">{node.label}</div>
                  <div className="mt-1 text-xs text-muted">{node.sub}</div>
                </div>
                {i < archNodes.length - 1 && (
                  <span className="hidden text-accent/50 sm:inline">
                    <svg
                      width="32"
                      height="12"
                      viewBox="0 0 32 12"
                      fill="none"
                      className="text-accent/50"
                    >
                      <path
                        d="M0 6h28m0 0l-5-5m5 5l-5 5"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
