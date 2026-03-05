"use client";

import { motion } from "framer-motion";

const packs = [
  {
    name: "drone-swarm",
    fig: "FIG_014",
    tools: 25,
    status: "████████████████ ACTIVE",
    description:
      "Fleet management, mission control, threat tracking, telemetry, and parameter tuning for multi-drone surveillance and swarm coordination projects.",
    highlights: [
      "Auto-discover drones from ROS 2 topics",
      "Send velocity commands and waypoints",
      "Real-time formation geometry analysis",
      "ENU/NED coordinate conversion",
    ],
  },
  {
    name: "manipulator",
    fig: "FIG_015",
    tools: 0,
    status: "████▓▓▓░░░░░░░░░ DEV",
    description:
      "Joint control, trajectory planning, URDF inspection, and MoveIt integration for robotic arm projects.",
    highlights: [
      "Joint state monitoring",
      "Trajectory planning tools",
      "URDF/XACRO introspection",
      "Gripper and end-effector control",
    ],
    comingSoon: true,
  },
  {
    name: "mobile-robot",
    fig: "FIG_016",
    tools: 0,
    status: "██▓▓░░░░░░░░░░░░ PLAN",
    description:
      "Navigation, SLAM, path planning, and sensor fusion for mobile robot projects using Nav2.",
    highlights: [
      "Nav2 integration",
      "Costmap and path visualization",
      "Localization monitoring",
      "Obstacle detection telemetry",
    ],
    comingSoon: true,
  },
];

export default function PluginPacks() {
  return (
    <section id="packs" className="section-technical relative py-28">
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
            CHAPTER 02 — DOMAIN PACKS
          </span>
          <h2 className="mt-3 font-display text-4xl tracking-tight sm:text-5xl">
            Domain <span className="text-accent">Plugin Packs</span>
          </h2>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-muted">
            Pre-built tool collections for common robotics domains. Auto-detected
            by{" "}
            <code
              className="border border-border bg-surface px-2 py-0.5 text-sm text-accent"
              style={{ fontFamily: "var(--font-jetbrains), monospace" }}
            >
              isaac-mcp init
            </code>{" "}
            or enabled manually.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-8 lg:grid-cols-3">
          {packs.map((pack, i) => (
            <motion.div
              key={pack.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className={`relative border bg-background p-8 transition-colors ${pack.comingSoon
                  ? "border-border opacity-60"
                  : "border-accent/30 hover:bg-cobalt-light"
                }`}
            >
              {/* Fig label */}
              <span
                style={{
                  fontFamily: "var(--font-jetbrains), monospace",
                  fontSize: "0.55rem",
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "rgba(0,0,255,0.3)",
                }}
              >
                {pack.fig}
              </span>

              <h3
                className="mt-3 mb-1 text-xl font-semibold tracking-tight"
                style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
              >
                {pack.name}
              </h3>

              {/* ASCII Progress Bar */}
              <div className="progress-ascii mb-4">{pack.status}</div>

              {pack.tools > 0 && (
                <span
                  className="mb-3 inline-block text-accent"
                  style={{
                    fontFamily: "var(--font-jetbrains), monospace",
                    fontSize: "0.65rem",
                    letterSpacing: "0.08em",
                  }}
                >
                  {pack.tools} TOOLS
                </span>
              )}

              <p className="mb-6 text-sm leading-relaxed text-muted">
                {pack.description}
              </p>

              <ul className="space-y-2">
                {pack.highlights.map((h) => (
                  <li
                    key={h}
                    className="flex items-start gap-2 text-sm text-muted"
                  >
                    <span className="mt-0.5 text-accent" style={{ fontFamily: "var(--font-jetbrains), monospace", fontSize: "0.7rem" }}>→</span>
                    {h}
                  </li>
                ))}
              </ul>

              {pack.comingSoon && (
                <span
                  className="absolute right-4 top-4 border border-border px-3 py-1 text-muted"
                  style={{
                    fontFamily: "var(--font-jetbrains), monospace",
                    fontSize: "0.6rem",
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                  }}
                >
                  Coming Soon
                </span>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
