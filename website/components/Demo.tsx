"use client";

import { motion } from "framer-motion";

const prompts = [
  {
    input: "Analyze why the robot keeps falling over.",
    tools: "analyze_simulation → get_diagnosis_history",
    output: "Root cause: floor friction coefficient too low (0.12). Recommended range: 0.4-0.8 for bipedal locomotion.",
  },
  {
    input: "Run a parameter sweep on floor friction from 0.1 to 1.0.",
    tools: "run_parameter_sweep → record_experiment",
    output: "Sweep complete: 10 trials recorded. Optimal friction: 0.65 (98% stability rate).",
  },
  {
    input: "Generate 50 randomized scenarios and test robustness.",
    tools: "generate_scenario → run_robustness_test",
    output: "50/50 scenarios complete. Pass rate: 94%. 3 failures in low-gravity + high-payload conditions.",
  },
  {
    input: "Has this physics error happened before?",
    tools: "query_knowledge_base",
    output: 'Found 3 similar incidents. Fix "increase_joint_damping" has 87% success rate.',
  },
  {
    input: "Fix it.",
    tools: "generate_fix → apply_fix_script",
    output: "Fix applied: joint_damping increased from 0.5 to 1.2. Simulation re-running... PASS.",
  },
];

export default function Demo() {
  return (
    <section id="demo" className="relative py-28">
      <div className="mx-auto max-w-4xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            See It in <span className="text-accent">Action</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            Natural language in, simulation intelligence out.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="mt-12 overflow-hidden rounded-2xl border border-border bg-surface shadow-2xl shadow-black/40"
        >
          {/* Terminal header */}
          <div className="flex items-center gap-2 border-b border-border px-4 py-3">
            <span className="h-3 w-3 rounded-full bg-red-500/80" />
            <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
            <span className="h-3 w-3 rounded-full bg-green-500/80" />
            <span className="ml-3 text-xs text-muted">
              isaac-mcp session
            </span>
          </div>

          {/* Terminal body */}
          <div className="space-y-5 p-6 font-mono text-sm">
            {prompts.map((p, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.4 }}
              >
                <div className="flex gap-2">
                  <span className="shrink-0 text-accent">❯</span>
                  <span className="text-foreground">{p.input}</span>
                </div>
                <div className="mt-1 pl-5 text-xs text-accent/60">
                  ↳ {p.tools}
                </div>
                <div className="mt-1 pl-5 text-muted">{p.output}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
