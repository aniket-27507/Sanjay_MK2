/**
 * Color utilities for the GCS Police Dashboard.
 * Heatmap scales, risk-level palettes, and drone-state colors.
 */

/* ── Crowd-density heatmap ── */

/**
 * Map a 0..1 density value to a heatmap RGBA string.
 * green -> yellow -> orange -> red
 */
export function densityToColor(value, alpha = 0.6) {
  const v = Math.max(0, Math.min(1, value));
  let r, g, b;

  if (v < 0.25) {
    // green -> yellow-green
    const t = v / 0.25;
    r = Math.round(30 + 170 * t);
    g = Math.round(180 + 55 * t);
    b = Math.round(60 * (1 - t));
  } else if (v < 0.5) {
    // yellow-green -> yellow
    const t = (v - 0.25) / 0.25;
    r = Math.round(200 + 55 * t);
    g = Math.round(235 - 35 * t);
    b = 0;
  } else if (v < 0.75) {
    // yellow -> orange
    const t = (v - 0.5) / 0.25;
    r = 255;
    g = Math.round(200 - 120 * t);
    b = 0;
  } else {
    // orange -> red
    const t = (v - 0.75) / 0.25;
    r = 255;
    g = Math.round(80 - 80 * t);
    b = Math.round(20 * t);
  }

  return `rgba(${r},${g},${b},${alpha})`;
}

/* ── Risk levels ── */

export const RISK_COLORS = {
  low: '#22c55e',
  moderate: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
};

export function riskLevelColor(level) {
  if (typeof level === 'number') {
    if (level < 0.25) return RISK_COLORS.low;
    if (level < 0.5) return RISK_COLORS.moderate;
    if (level < 0.75) return RISK_COLORS.high;
    return RISK_COLORS.critical;
  }
  return RISK_COLORS[level] || RISK_COLORS.low;
}

export function riskLevelLabel(value) {
  if (value < 0.25) return 'LOW';
  if (value < 0.5) return 'MODERATE';
  if (value < 0.75) return 'HIGH';
  return 'CRITICAL';
}

/* ── Drone state ── */

export const DRONE_STATE_COLORS = {
  active: '#22c55e',
  idle: '#64748b',
  returning: '#3b82f6',
  charging: '#a855f7',
  fault: '#ef4444',
  offline: '#334155',
};

/* ── Threat severity ── */

export const THREAT_COLORS = {
  info: '#38bdf8',
  warning: '#eab308',
  danger: '#f97316',
  critical: '#ef4444',
};

/* ── General palette ── */

export const PALETTE = {
  bg: '#0a0e1a',
  surface: '#111827',
  surfaceLight: '#1e293b',
  border: '#1e3a5f',
  borderLight: '#334155',
  text: '#c8d0e0',
  textMuted: '#64748b',
  textBright: '#f1f5f9',
  accent: '#3b82f6',
  accentDim: '#1e40af',
  success: '#22c55e',
  warning: '#eab308',
  error: '#ef4444',
};
