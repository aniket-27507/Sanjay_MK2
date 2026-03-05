"use client";

import { motion } from "framer-motion";

const steps = [
  {
    number: "1",
    title: "Install",
    command: "pip install isaac-mcp",
    description: "One pip install. No system dependencies required.",
  },
  {
    number: "2",
    title: "Initialize",
    command: "isaac-mcp init",
    description:
      "Auto-detects your project type, ROS 2 topics, drones, and generates optimized config.",
  },
  {
    number: "3",
    title: "Register",
    command: "isaac-mcp register --cursor",
    description:
      "Opens your IDE and configures the MCP connection automatically.",
  },
  {
    number: "4",
    title: "Use",
    command: '"Show me the fleet status"',
    description:
      "Ask your AI assistant anything — it now has full access to your simulation.",
  },
];

export default function QuickStart() {
  return (
    <section id="quickstart" className="relative py-28">
      <div className="mx-auto max-w-5xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Up and Running in{" "}
            <span className="text-accent">60 Seconds</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            No config files to write. No manual topic mapping. Just point
            IsaacMCP at your project and go.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="rounded-2xl border border-border bg-panel p-6"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-accent/15 text-lg font-bold text-accent">
                {step.number}
              </div>
              <h3 className="mb-2 text-lg font-semibold">{step.title}</h3>
              <code className="mb-3 block rounded-lg bg-surface px-3 py-2 font-mono text-xs text-accent">
                {step.command}
              </code>
              <p className="text-sm leading-relaxed text-muted">
                {step.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
