"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";

const prompts = [
  {
    input: "Analyze why the robot keeps falling over.",
    tools: "analyze_simulation → get_diagnosis_history",
    output:
      "Root cause: floor friction coefficient too low (0.12). Recommended range: 0.4-0.8 for bipedal locomotion.",
  },
  {
    input: "Run a parameter sweep on floor friction from 0.1 to 1.0.",
    tools: "run_parameter_sweep → record_experiment",
    output:
      "Sweep complete: 10 trials recorded. Optimal friction: 0.65 (98% stability rate).",
  },
  {
    input: "Generate 50 randomized scenarios and test robustness.",
    tools: "generate_scenario → run_robustness_test",
    output:
      "50/50 scenarios complete. Pass rate: 94%. 3 failures in low-gravity + high-payload conditions.",
  },
  {
    input: "Has this physics error happened before?",
    tools: "query_knowledge_base",
    output:
      'Found 3 similar incidents. Fix "increase_joint_damping" has 87% success rate.',
  },
  {
    input: "Fix it.",
    tools: "generate_fix → apply_fix_script",
    output:
      "Fix applied: joint_damping increased from 0.5 to 1.2. Simulation re-running... PASS.",
  },
];

function TypewriterText({ text, delay = 0 }: { text: string; delay?: number }) {
  const [displayed, setDisplayed] = useState("");
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const timeout = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(timeout);
  }, [delay]);

  useEffect(() => {
    if (!started) return;
    let i = 0;
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
      }
    }, 20);
    return () => clearInterval(interval);
  }, [started, text]);

  if (!started) return null;
  return <>{displayed}<span className="animate-blink text-accent">█</span></>;
}

export default function Demo() {
  return (
    <section id="demo" className="section-technical relative py-28">
      <div className="divider-check mb-20" />

      <div className="mx-auto max-w-4xl px-6">
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
            CHAPTER 05 — DEMONSTRATION
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            See It in <span className="text-accent">Action</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            Natural language in, simulation intelligence out.
          </p>
        </motion.div>

        {/* Terminal */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mt-12 overflow-hidden border border-border"
          style={{ boxShadow: "4px 4px 0 rgba(0,0,255,0.06)" }}
        >
          {/* Terminal header — OS style */}
          <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-2.5">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-red-400/70 border border-red-500/50" />
              <span className="h-2.5 w-2.5 rounded-full bg-yellow-400/70 border border-yellow-500/50" />
              <span className="h-2.5 w-2.5 rounded-full bg-green-400/70 border border-green-500/50" />
            </div>
            <span
              className="text-muted"
              style={{
                fontFamily: "var(--font-jetbrains), monospace",
                fontSize: "0.6rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              ISAAC-MCP SESSION_001
            </span>
          </div>

          {/* Terminal body */}
          <div className="space-y-5 bg-background p-6"
            style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: "0.8rem" }}
          >
            {prompts.map((p, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08, duration: 0.35 }}
              >
                <div className="flex gap-2">
                  <span className="shrink-0 text-accent">❯</span>
                  <span className="text-foreground">{p.input}</span>
                </div>
                <div className="mt-1 pl-5 text-accent/40" style={{ fontSize: "0.7rem" }}>
                  ↳ {p.tools}
                </div>
                <div className="mt-1 pl-5 text-muted">{p.output}</div>
              </motion.div>
            ))}
            <div className="flex gap-2 pt-2 border-t border-border/50">
              <span className="shrink-0 text-accent">❯</span>
              <span className="animate-blink text-accent">█</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
