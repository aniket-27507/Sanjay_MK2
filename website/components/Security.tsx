"use client";

import { motion } from "framer-motion";

const cards = [
  {
    fig: "FIG_017",
    icon: (
      <svg className="h-7 w-7 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
      </svg>
    ),
    title: "Read-Only by Default",
    description:
      "The server boots in read-only mode. No simulation state can be altered unless mutations are explicitly enabled via environment variable.",
  },
  {
    fig: "FIG_018",
    icon: (
      <svg className="h-7 w-7 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
      </svg>
    ),
    title: "Gated Mutations",
    description:
      "Destructive operations (changing USD stage, applying scripts) check a strict mutation gate (ISAAC_MCP_ENABLE_MUTATIONS) before executing.",
  },
  {
    fig: "FIG_019",
    icon: (
      <svg className="h-7 w-7 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
      </svg>
    ),
    title: "Tool Annotations",
    description:
      "All 80+ tools carry readOnlyHint, destructiveHint, and idempotentHint annotations so your LLM inherently understands the weight of each action.",
  },
];

export default function Security() {
  return (
    <section id="security" className="section-technical relative py-28">
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
            CHAPTER 07 — SECURITY
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            Security &amp; <span className="text-accent">Safety</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            Safety is a first-class citizen. Your simulations are protected by default.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-0 md:grid-cols-3" style={{ border: "1px solid var(--color-border)" }}>
          {cards.map((c, i) => (
            <motion.div
              key={c.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.45 }}
              className={`relative bg-background p-8 transition-colors hover:bg-cobalt-light ${i < cards.length - 1 ? "md:border-r border-border" : ""
                }`}
            >
              <span
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.55rem",
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "rgba(0,0,255,0.3)",
                }}
              >
                {c.fig}
              </span>
              <div className="mt-3 mb-4">{c.icon}</div>
              <h3
                className="mb-3 text-lg font-semibold tracking-tight"
                style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
              >
                {c.title}
              </h3>
              <p className="text-sm leading-relaxed text-muted">
                {c.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
