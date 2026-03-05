"use client";

import { motion } from "framer-motion";

const stats = [
  { value: "54", label: "MCP Tools" },
  { value: "10", label: "Plugins" },
  { value: "100%", label: "Test Coverage" },
];

export default function Hero() {
  return (
    <section className="relative flex min-h-screen items-center justify-center overflow-hidden pt-16">
      {/* Gradient orbs */}
      <div className="pointer-events-none absolute -top-32 left-1/4 h-[600px] w-[600px] rounded-full bg-accent/8 blur-[120px]" />
      <div className="pointer-events-none absolute -bottom-32 right-1/4 h-[500px] w-[500px] rounded-full bg-blue-600/6 blur-[100px]" />

      <div className="relative z-10 mx-auto max-w-5xl px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <span className="mb-4 inline-block rounded-full border border-accent/30 bg-accent/10 px-4 py-1.5 text-sm font-medium text-accent">
            AI-Driven Robotics Simulation
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="mx-auto mt-6 max-w-4xl text-5xl font-bold leading-tight tracking-tight sm:text-6xl lg:text-7xl"
        >
          The AI Copilot for{" "}
          <span className="bg-gradient-to-r from-accent to-blue-400 bg-clip-text text-transparent">
            Robotics Simulation
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="mx-auto mt-6 max-w-2xl text-lg text-muted sm:text-xl"
        >
          Seamlessly bridge LLMs like Claude and Cursor with NVIDIA Isaac Sim
          using the Model Context Protocol. Diagnose failures, auto-fix
          simulations, and run massive experiment campaigns — all through natural
          language.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row"
        >
          <a
            href="#install"
            className="rounded-xl bg-gradient-to-r from-accent to-blue-500 px-8 py-3.5 text-base font-semibold text-background shadow-lg shadow-accent/20 transition-all hover:shadow-accent/40 hover:brightness-110"
          >
            Get Started
          </a>
          <a
            href="https://github.com/your-org/isaac-mcp"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-xl border border-border px-8 py-3.5 text-base font-semibold text-foreground transition-colors hover:border-accent/50 hover:bg-panel"
          >
            View on GitHub
          </a>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.45 }}
          className="mx-auto mt-16 flex max-w-md justify-center gap-8 sm:gap-16"
        >
          {stats.map((s) => (
            <div key={s.label} className="text-center">
              <div className="text-3xl font-bold text-accent sm:text-4xl">
                {s.value}
              </div>
              <div className="mt-1 text-sm text-muted">{s.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
