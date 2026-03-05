"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const faqs = [
  {
    q: "What is the Model Context Protocol (MCP)?",
    a: "MCP is an open standard that lets AI assistants (like Claude, Cursor, and Claude Code) discover and use external tools, resources, and data sources through a unified protocol. IsaacMCP implements MCP to expose NVIDIA Isaac Sim capabilities as semantic tools your LLM can call.",
  },
  {
    q: "Which LLMs are supported?",
    a: "Any LLM or AI assistant that supports the Model Context Protocol can use IsaacMCP. This includes Claude (via Claude Desktop, Claude Code CLI), Cursor, and any MCP-compatible client. The server communicates over standard JSON-RPC.",
  },
  {
    q: "Is it safe to let an AI control my simulation?",
    a: "Yes. IsaacMCP boots in read-only mode by default. All destructive operations are gated behind the ISAAC_MCP_ENABLE_MUTATIONS environment variable. Every tool carries safety annotations (readOnlyHint, destructiveHint, idempotentHint) so the LLM understands the consequences before acting.",
  },
  {
    q: "How do I deploy for remote access?",
    a: "IsaacMCP supports streamable-http transport with optional OAuth bearer-token security. You can run it directly on your GPU server and expose it via HTTPS, or use a Cloudflare Tunnel for zero-trust enterprise deployment. See the Deploy section above for step-by-step instructions.",
  },
  {
    q: "Does it support ROS2?",
    a: "Yes. Install the optional ROS2 dependency with `pip install -e '.[ros2]'`. The ROS2 Bridge plugin provides bidirectional topic, service, and action integration between Isaac Sim and your robot stack via rclpy.",
  },
  {
    q: "What are the system requirements?",
    a: "Python 3.10 or higher and a running instance of NVIDIA Isaac Sim (local or remote). For the full development setup, install with `pip install -e '.[dev]'` to get testing dependencies as well.",
  },
];

function Accordion({
  q,
  a,
  open,
  toggle,
  index,
}: {
  q: string;
  a: string;
  open: boolean;
  toggle: () => void;
  index: number;
}) {
  return (
    <div className={`border-b border-border ${open ? "bg-cobalt-light" : ""} transition-colors`}>
      <button
        onClick={toggle}
        aria-expanded={open}
        aria-controls={`faq-answer-${index}`}
        className="flex w-full items-center justify-between py-5 px-6 text-left group"
      >
        <div className="flex items-start gap-4">
          <span
            className="shrink-0 mt-0.5 text-accent/25 group-hover:text-accent/50 transition-colors"
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              fontSize: "0.65rem",
              letterSpacing: "0.06em",
            }}
          >
            Q_{String(index + 1).padStart(2, "0")}
          </span>
          <span className="pr-4 text-base font-medium group-hover:text-accent transition-colors"
            style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
          >
            {q}
          </span>
        </div>
        <span
          className={`shrink-0 text-muted transition-transform duration-200 ${open ? "rotate-45" : ""
            }`}
          style={{
            fontFamily: "var(--font-jetbrains), monospace",
            fontSize: "1.1rem",
          }}
        >
          +
        </span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            id={`faq-answer-${index}`}
            role="region"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <p className="pb-5 pl-16 pr-6 text-sm leading-relaxed text-muted">
              {a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function FAQ() {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  return (
    <section id="faq" className="section-technical relative py-28">
      <div className="divider-check mb-20" />

      <div className="mx-auto max-w-3xl px-6">
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
            CHAPTER 09 — FAQ
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            Frequently Asked <span className="text-accent">Questions</span>
          </h2>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45, delay: 0.1 }}
          className="mt-12 border-t border-border"
          style={{ border: "1px solid var(--color-border)" }}
        >
          {faqs.map((f, i) => (
            <Accordion
              key={i}
              index={i}
              q={f.q}
              a={f.a}
              open={openIdx === i}
              toggle={() => setOpenIdx(openIdx === i ? null : i)}
            />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
