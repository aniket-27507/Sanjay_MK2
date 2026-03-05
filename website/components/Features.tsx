"use client";

import { motion } from "framer-motion";

const features = [
  {
    fig: "FIG_002",
    title: "Intelligent Diagnostics",
    description:
      "Cross-correlates telemetry, logs, and scene hierarchy for automated root-cause analysis of simulation failures.",
  },
  {
    fig: "FIG_003",
    title: "Autonomous Fix Loop",
    description:
      "Self-healing simulations that synthesize fixes, generate Kit API scripts, and inject them live to remediate issues.",
  },
  {
    fig: "FIG_004",
    title: "Experiment Engine",
    description:
      "Run massive batch simulations and parameter sweeps with async SQLite tracking of success rates and metrics.",
  },
  {
    fig: "FIG_005",
    title: "Scenario Lab",
    description:
      "Procedurally generate randomized scenarios covering friction, gravity, obstacles, payloads, and lighting.",
  },
  {
    fig: "FIG_006",
    title: "Knowledge Memory",
    description:
      "Self-learning memory base that records error patterns and tracks statistical success rates of applied fixes.",
  },
  {
    fig: "FIG_007",
    title: "Camera & Render",
    description:
      "Capture viewport renders, manage camera positions, and stream visual data from Isaac Sim's rendering pipeline.",
  },
  {
    fig: "FIG_008",
    title: "Log Monitor",
    description:
      "Real-time log parsing and monitoring with intelligent filtering, error detection, and pattern recognition.",
  },
  {
    fig: "FIG_009",
    title: "Scene Inspector",
    description:
      "Navigate and inspect USD scene hierarchies, prim properties, physics materials, and sensor configurations.",
  },
  {
    fig: "FIG_010",
    title: "Sim Control",
    description:
      "Start, stop, reset, and step simulations. Control physics parameters, time scales, and world state.",
  },
  {
    fig: "FIG_011",
    title: "ROS 2 Bridge",
    description:
      "Full rclpy integration with real-time topic subscriptions, publishing, discovery, and ENU/NED coordinate conversion.",
  },
  {
    fig: "FIG_012",
    title: "Drone Swarm Pack",
    description:
      "25+ tools for fleet management, mission control, threat tracking, telemetry, and parameter tuning for multi-drone projects.",
  },
  {
    fig: "FIG_013",
    title: "One-Command Setup",
    description:
      "Auto-detect your project, generate configs, and register with your IDE in seconds via the isaac-mcp CLI.",
  },
];

const staggerVariant = {
  hidden: { opacity: 0, y: 16 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.05, duration: 0.45, ease: "easeOut" },
  }),
};

export default function Features() {
  return (
    <section id="features" className="section-technical relative py-28">
      {/* Checkered divider */}
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
            className="mb-4 inline-block text-accent"
            style={{
              fontFamily: "var(--font-jetbrains), monospace",
              fontSize: "0.65rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}
          >
            CHAPTER 01 — CAPABILITIES
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            80+ Tools. Plugin Packs.{" "}
            <span className="text-accent">One Protocol.</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            A comprehensive toolkit that turns your LLM into a full-stack
            robotics simulation engineer — for drones, manipulators, and beyond.
          </p>
        </motion.div>

        {/* Feature Grid */}
        <div className="mt-16 grid gap-[1px] bg-border sm:grid-cols-2 lg:grid-cols-4"
          style={{ border: "1px solid var(--color-border)" }}
        >
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              custom={i}
              variants={staggerVariant}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.15 }}
              className="group relative bg-background p-6 transition-colors duration-200 hover:bg-cobalt-light cursor-default"
            >
              {/* Figure label */}
              <span
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.55rem",
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "rgba(0,0,255,0.3)",
                }}
              >
                {f.fig}
              </span>
              <h3 className="mt-2 mb-2 text-base font-semibold tracking-tight group-hover:text-accent transition-colors"
                style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
              >
                {f.title}
              </h3>
              <p className="text-sm leading-relaxed text-muted">
                {f.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
