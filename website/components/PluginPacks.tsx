"use client";

import { motion } from "framer-motion";

const packs = [
  {
    name: "drone-swarm",
    icon: "🚁",
    tools: 25,
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
    icon: "🦾",
    tools: 0,
    description:
      "Joint control, trajectory planning, URDF inspection, and MoveIt integration for robotic arm projects. Coming soon.",
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
    icon: "🤖",
    tools: 0,
    description:
      "Navigation, SLAM, path planning, and sensor fusion for mobile robot projects using Nav2. Coming soon.",
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
    <section id="packs" className="relative py-28">
      <div className="mx-auto max-w-6xl px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <h2 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Domain <span className="text-accent">Plugin Packs</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            Pre-built tool collections for common robotics domains. Auto-detected
            by <code className="rounded bg-panel px-1.5 py-0.5 text-sm text-accent">isaac-mcp init</code> or enabled manually.
          </p>
        </motion.div>

        <div className="mt-16 grid gap-8 lg:grid-cols-3">
          {packs.map((pack, i) => (
            <motion.div
              key={pack.name}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className={`relative overflow-hidden rounded-2xl border bg-panel p-8 ${
                pack.comingSoon
                  ? "border-border/50 opacity-70"
                  : "border-accent/30 shadow-lg shadow-accent/5"
              }`}
            >
              {pack.comingSoon && (
                <span className="absolute right-4 top-4 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-muted">
                  Coming Soon
                </span>
              )}
              <div className="mb-4 text-4xl">{pack.icon}</div>
              <h3 className="mb-1 text-xl font-bold">{pack.name}</h3>
              {pack.tools > 0 && (
                <span className="mb-3 inline-block text-sm text-accent">
                  {pack.tools} tools
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
                    <span className="mt-0.5 text-accent">&#10003;</span>
                    {h}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
