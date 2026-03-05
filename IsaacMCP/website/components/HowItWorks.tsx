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
  { label: "IsaacMCP Server", sub: "80+ Tools" },
  { label: "NVIDIA Isaac Sim", sub: "PhysX · Sensors · Kit API" },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="section-technical relative py-28">
      <div className="divider-check mb-20" />

      <div className="mx-auto max-w-6xl px-6">
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
            CHAPTER 04 — WORKFLOW
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            How It <span className="text-accent">Works</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            Three steps from natural language to full simulation control.
          </p>
        </motion.div>

        {/* Steps — horizontal three-column */}
        <div className="mt-16 grid gap-0 md:grid-cols-3" style={{ border: "1px solid var(--color-border)" }}>
          {steps.map((s, i) => (
            <motion.div
              key={s.number}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12, duration: 0.45 }}
              className={`relative bg-background p-8 ${i < steps.length - 1 ? "md:border-r border-border" : ""
                }`}
            >
              <span
                className="text-accent/15"
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "3.5rem",
                  fontWeight: 900,
                  lineHeight: 1,
                }}
              >
                {s.number}
              </span>
              <h3
                className="mt-3 text-xl font-semibold tracking-tight"
                style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
              >
                {s.title}
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                {s.description}
              </p>
              <p
                className="mt-3 text-accent/60"
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.65rem",
                  letterSpacing: "0.04em",
                }}
              >
                → {s.detail}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Architecture Diagram */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mt-20"
        >
          <h3
            className="mb-8 text-center text-muted"
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              fontSize: "0.65rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}
          >
            ARCHITECTURE OVERVIEW
          </h3>
          <div className="flex flex-col items-center gap-0 sm:flex-row sm:justify-center sm:gap-0">
            {archNodes.map((node, i) => (
              <div key={node.label} className="flex items-center gap-0">
                <div className="border border-border bg-surface px-5 py-4 text-center transition-colors hover:bg-cobalt-light hover:border-accent/30">
                  <div
                    className="text-sm font-semibold"
                    style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
                  >
                    {node.label}
                  </div>
                  <div
                    className="mt-1 text-muted"
                    style={{
                      fontFamily: "var(--font-jetbrains), monospace",
                      fontSize: "0.6rem",
                      letterSpacing: "0.06em",
                    }}
                  >
                    {node.sub}
                  </div>
                </div>
                {i < archNodes.length - 1 && (
                  <div className="hidden sm:flex items-center px-1">
                    <svg width="32" height="12" viewBox="0 0 32 12" fill="none">
                      <path
                        d="M0 6h28m0 0l-5-5m5 5l-5 5"
                        stroke="rgba(0,0,255,0.3)"
                        strokeWidth="1"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
