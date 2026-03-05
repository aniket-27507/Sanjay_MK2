"use client";

import { motion } from "framer-motion";

const steps = [
  {
    number: "01",
    title: "Install",
    command: "pip install isaac-mcp",
    description: "One pip install. No system dependencies required.",
  },
  {
    number: "02",
    title: "Initialize",
    command: "isaac-mcp init",
    description:
      "Auto-detects your project type, ROS 2 topics, drones, and generates optimized config.",
  },
  {
    number: "03",
    title: "Register",
    command: "isaac-mcp register --cursor",
    description:
      "Opens your IDE and configures the MCP connection automatically.",
  },
  {
    number: "04",
    title: "Use",
    command: '"Show me the fleet status"',
    description:
      "Ask your AI assistant anything — it now has full access to your simulation.",
  },
];

export default function QuickStart() {
  return (
    <section id="quickstart" className="section-technical relative py-28">
      <div className="divider-check mb-20" />

      <div className="mx-auto max-w-5xl px-6">
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
            CHAPTER 03 — QUICK START
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            Up and Running in{" "}
            <span className="text-accent">60 Seconds</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            No config files to write. No manual topic mapping. Just point
            IsaacMCP at your project and go.
          </p>
        </motion.div>

        {/* Steps as a table-of-contents style layout */}
        <div className="mt-16 space-y-0 border border-border">
          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, x: -16 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className={`group flex items-start gap-6 p-6 transition-colors hover:bg-cobalt-light ${i < steps.length - 1 ? "border-b border-border" : ""
                }`}
            >
              {/* Step number */}
              <span
                className="shrink-0 text-accent/20 group-hover:text-accent/40 transition-colors"
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "2.5rem",
                  fontWeight: 900,
                  lineHeight: 1,
                }}
              >
                {step.number}
              </span>

              <div className="flex-1">
                <h3 className="mb-1 text-lg font-semibold tracking-tight group-hover:text-accent transition-colors"
                  style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
                >
                  {step.title}
                </h3>
                <code
                  className="mb-2 inline-block border border-border bg-surface px-3 py-1.5 text-accent"
                  style={{
                    fontFamily: "var(--font-jetbrains), monospace",
                    fontSize: "0.75rem",
                  }}
                >
                  $ {step.command}
                </code>
                <p className="mt-2 text-sm leading-relaxed text-muted">
                  {step.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
