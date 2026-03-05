"use client";

import { motion } from "framer-motion";

const features = [
  {
    icon: "🧠",
    title: "Intelligent Diagnostics",
    description:
      "Cross-correlates telemetry, logs, and scene hierarchy for automated root-cause analysis of simulation failures.",
  },
  {
    icon: "💊",
    title: "Autonomous Fix Loop",
    description:
      "Self-healing simulations that synthesize fixes, generate Kit API scripts, and inject them live to remediate issues.",
  },
  {
    icon: "🔬",
    title: "Experiment Engine",
    description:
      "Run massive batch simulations and parameter sweeps with async SQLite tracking of success rates and metrics.",
  },
  {
    icon: "🌪️",
    title: "Scenario Lab",
    description:
      "Procedurally generate randomized scenarios covering friction, gravity, obstacles, payloads, and lighting.",
  },
  {
    icon: "📚",
    title: "Knowledge Memory",
    description:
      "Self-learning memory base that records error patterns and tracks statistical success rates of applied fixes.",
  },
  {
    icon: "📷",
    title: "Camera & Render",
    description:
      "Capture viewport renders, manage camera positions, and stream visual data from Isaac Sim's rendering pipeline.",
  },
  {
    icon: "📋",
    title: "Log Monitor",
    description:
      "Real-time log parsing and monitoring with intelligent filtering, error detection, and pattern recognition.",
  },
  {
    icon: "🔍",
    title: "Scene Inspector",
    description:
      "Navigate and inspect USD scene hierarchies, prim properties, physics materials, and sensor configurations.",
  },
  {
    icon: "🎮",
    title: "Sim Control",
    description:
      "Start, stop, reset, and step simulations. Control physics parameters, time scales, and world state.",
  },
  {
    icon: "🤖",
    title: "ROS 2 Bridge",
    description:
      "Full rclpy integration with real-time topic subscriptions, publishing, discovery, and ENU/NED coordinate conversion.",
  },
  {
    icon: "🚁",
    title: "Drone Swarm Pack",
    description:
      "25+ tools for fleet management, mission control, threat tracking, telemetry, and parameter tuning for multi-drone projects.",
  },
  {
    icon: "⚡",
    title: "One-Command Setup",
    description:
      "Auto-detect your project, generate configs, and register with your IDE in seconds via the isaac-mcp CLI.",
  },
];

const cardVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.06, duration: 0.5 },
  }),
};

export default function Features() {
  return (
    <section id="features" className="relative py-28">
      <div className="mx-auto max-w-7xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            80+ Tools. Plugin Packs. <span className="text-accent">One Protocol.</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            A comprehensive toolkit that turns your LLM into a full-stack
            robotics simulation engineer — for drones, manipulators, and beyond.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              custom={i}
              variants={cardVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true, amount: 0.2 }}
              className="group rounded-2xl border border-border bg-panel p-6 transition-all hover:border-accent/40 hover:bg-panel-hover hover:shadow-lg hover:shadow-accent/5"
            >
              <div className="mb-3 text-3xl">{f.icon}</div>
              <h3 className="mb-2 text-base font-semibold">{f.title}</h3>
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
