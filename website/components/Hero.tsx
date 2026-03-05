"use client";

import { motion } from "framer-motion";

const stats = [
  { value: "80+", label: "MCP TOOLS" },
  { value: "12", label: "PLUGINS" },
  { value: "1 CMD", label: "SETUP" },
];

export default function Hero() {
  return (
    <section className="section-technical relative flex min-h-screen items-center justify-center overflow-hidden pt-16">
      {/* Corner accents */}
      <span className="corner-accent top-24 left-6 hidden lg:block">SYS_STATUS: ONLINE</span>
      <span className="corner-accent bottom-8 right-6 hidden lg:block">© 2025</span>

      <div className="relative z-10 mx-auto max-w-4xl px-6 text-center">
        {/* Label */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        >
          <span
            className="mb-6 inline-block border border-accent/20 px-4 py-1.5 text-accent"
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              fontSize: "0.65rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}
          >
            FIG_001 — AI-DRIVEN ROBOTICS SIMULATION
          </span>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15 }}
          className="mx-auto mt-8 max-w-4xl font-display leading-[1.1]"
          style={{ fontSize: "clamp(2.8rem, 7vw, 5.5rem)" }}
        >
          The AI Copilot for{" "}
          <span className="text-accent">
            Robotics Simulation
          </span>
        </motion.h1>

        {/* Sub */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3 }}
          className="mx-auto mt-8 max-w-2xl text-lg leading-relaxed text-muted"
          style={{ fontFamily: "var(--font-garamond), Georgia, serif" }}
        >
          Seamlessly bridge LLMs like Claude and Cursor with NVIDIA Isaac Sim
          using the Model Context Protocol. Diagnose failures, auto-fix
          simulations, and run massive experiment campaigns — all through natural
          language.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.45 }}
          className="mt-12 flex flex-col items-center justify-center gap-4 sm:flex-row"
        >
          <a href="#install" className="btn-bevel btn-bevel-primary">
            Get Started →
          </a>
          <a
            href="https://github.com/yanitedhacker/IsaacMCP"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-bevel"
          >
            View Source ↗
          </a>
        </motion.div>

        {/* Stats Row */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mx-auto mt-20 flex max-w-md justify-center gap-12 sm:gap-20"
        >
          {stats.map((s) => (
            <div key={s.label} className="text-center">
              <div
                className="text-3xl font-bold text-accent sm:text-4xl"
                style={{ fontFamily: "var(--font-jetbrains), monospace" }}
              >
                {s.value}
              </div>
              <div
                className="mt-2"
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.6rem",
                  letterSpacing: "0.12em",
                  color: "var(--color-muted)",
                }}
              >
                {s.label}
              </div>
            </div>
          ))}
        </motion.div>

        {/* Blueprint illustration placeholder — isometric line art deco */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.8 }}
          className="mx-auto mt-16 max-w-lg"
        >
          <svg viewBox="0 0 500 160" fill="none" className="w-full">
            {/* Grid Lines */}
            {[0, 40, 80, 120, 160].map((y) => (
              <line key={`h-${y}`} x1="0" y1={y} x2="500" y2={y} stroke="rgba(0,0,255,0.06)" strokeWidth="0.5" />
            ))}
            {[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500].map((x) => (
              <line key={`v-${x}`} x1={x} y1="0" x2={x} y2="160" stroke="rgba(0,0,255,0.06)" strokeWidth="0.5" />
            ))}
            {/* Isometric Robot Arm */}
            <path d="M100,140 L100,80 L160,50 L230,80 L230,120" stroke="rgba(0,0,255,0.3)" strokeWidth="1.5" fill="none" />
            <circle cx="160" cy="50" r="4" stroke="rgba(0,0,255,0.4)" strokeWidth="1" fill="none" />
            <circle cx="100" cy="80" r="3" stroke="rgba(0,0,255,0.3)" strokeWidth="1" fill="none" />
            <circle cx="230" cy="80" r="3" stroke="rgba(0,0,255,0.3)" strokeWidth="1" fill="none" />
            {/* Data Flow Lines */}
            <path d="M250,80 L320,60 L400,80" stroke="rgba(0,0,255,0.2)" strokeWidth="1" strokeDasharray="4 4" fill="none" />
            <path d="M250,100 L320,120 L400,100" stroke="rgba(0,0,255,0.2)" strokeWidth="1" strokeDasharray="4 4" fill="none" />
            {/* Connection nodes */}
            <rect x="395" y="70" width="20" height="20" rx="2" stroke="rgba(0,0,255,0.3)" strokeWidth="1" fill="none" />
            <rect x="245" y="75" width="10" height="10" rx="1" stroke="rgba(0,0,255,0.25)" strokeWidth="1" fill="rgba(0,0,255,0.05)" />
            {/* Signal broadcast */}
            <circle cx="405" cy="40" r="8" stroke="rgba(0,0,255,0.15)" strokeWidth="0.5" fill="none" />
            <circle cx="405" cy="40" r="16" stroke="rgba(0,0,255,0.1)" strokeWidth="0.5" fill="none" />
            <circle cx="405" cy="40" r="24" stroke="rgba(0,0,255,0.05)" strokeWidth="0.5" fill="none" />
          </svg>
        </motion.div>
      </div>
    </section>
  );
}
