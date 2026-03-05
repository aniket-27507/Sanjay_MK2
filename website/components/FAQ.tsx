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
  id,
}: {
  q: string;
  a: string;
  open: boolean;
  toggle: () => void;
  id: string;
}) {
  return (
    <div className="border-b border-border">
      <button
        onClick={toggle}
        aria-expanded={open}
        aria-controls={`faq-answer-${id}`}
        className="flex w-full items-center justify-between py-5 text-left"
      >
        <span className="pr-4 text-base font-medium">{q}</span>
        <span
          className={`shrink-0 text-xl text-muted transition-transform ${open ? "rotate-45" : ""}`}
        >
          +
        </span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            id={`faq-answer-${id}`}
            role="region"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <p className="pb-5 text-sm leading-relaxed text-muted">{a}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function FAQ() {
  const [openIdx, setOpenIdx] = useState<number | null>(null);

  return (
    <section id="faq" className="relative py-28">
      <div className="mx-auto max-w-3xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Frequently Asked <span className="text-accent">Questions</span>
          </h2>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mt-12"
        >
          {faqs.map((f, i) => (
            <Accordion
              key={i}
              id={String(i)}
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
