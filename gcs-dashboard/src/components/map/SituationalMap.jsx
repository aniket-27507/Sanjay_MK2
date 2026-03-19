import React, { useRef, useEffect, useCallback, useState } from 'react';
import useGCSState from '../../hooks/useGCSState';
import CrowdHeatmapLayer from './CrowdHeatmapLayer';
import { DRONE_STATE_COLORS, THREAT_COLORS, PALETTE } from '../../utils/colors';

const WORLD_SIZE = 1000; // 1000 x 1000 m

export default function SituationalMap() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const mapPositions = useGCSState((s) => s.mapPositions);
  const drones = useGCSState((s) => s.drones);
  const mapThreats = useGCSState((s) => s.mapThreats);
  const zones = useGCSState((s) => s.zones);
  const crowdGrid = useGCSState((s) => s.crowdGrid);

  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 600 });
  const [hoveredDrone, setHoveredDrone] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const [showThreats, setShowThreats] = useState(true);

  /* Resize observer */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setCanvasSize({ w: Math.floor(width), h: Math.floor(height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  /* World -> canvas coordinate.
   * World coords are -500 to +500. Map display is 0 to 1000.
   * Shift by WORLD_SIZE/2 to center the origin. */
  const scale = Math.min(canvasSize.w, canvasSize.h) / WORLD_SIZE;
  const toCanvas = useCallback(
    (wx, wy) => ({
      cx: (wx + WORLD_SIZE / 2) * scale,
      cy: (wy + WORLD_SIZE / 2) * scale,
    }),
    [scale],
  );

  /* ── Draw ── */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { w, h } = canvasSize;
    canvas.width = w;
    canvas.height = h;

    // Background
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(30,58,95,0.35)';
    ctx.lineWidth = 0.5;
    const gridStep = 100 * scale;
    for (let x = 0; x <= w; x += gridStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y <= h; y += gridStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Grid labels (every 200m)
    ctx.fillStyle = 'rgba(100,116,139,0.5)';
    ctx.font = '10px monospace';
    for (let i = 0; i <= WORLD_SIZE; i += 200) {
      const { cx } = toCanvas(i, 0);
      ctx.fillText(`${i}m`, cx + 2, 12);
      const { cy } = toCanvas(0, i);
      ctx.fillText(`${i}m`, 2, cy + 12);
    }

    // Zones
    if (showZones && zones.length > 0) {
      zones.forEach((zone) => {
        if (!zone.polygon || zone.polygon.length < 3) return;
        ctx.beginPath();
        zone.polygon.forEach((pt, i) => {
          const { cx, cy } = toCanvas(pt.x, pt.y);
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        });
        ctx.closePath();
        const zoneColor = zone.color || '#3b82f6';
        ctx.fillStyle = `${zoneColor}15`;
        ctx.fill();
        ctx.strokeStyle = `${zoneColor}60`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        if (zone.label) {
          const ctr = zone.polygon.reduce(
            (acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }),
            { x: 0, y: 0 },
          );
          ctr.x /= zone.polygon.length;
          ctr.y /= zone.polygon.length;
          const { cx, cy } = toCanvas(ctr.x, ctr.y);
          ctx.fillStyle = 'rgba(200,208,224,0.7)';
          ctx.font = 'bold 11px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(zone.label, cx, cy);
          ctx.textAlign = 'start';
        }
      });
    }

    // Threats
    if (showThreats) {
      (mapThreats || []).forEach((t) => {
        if (!t.position) return;
        const { cx, cy } = toCanvas(t.position.x, t.position.y);
        const col = THREAT_COLORS[t.severity] || THREAT_COLORS.warning;

        // Pulse ring
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.fillStyle = `${col}20`;
        ctx.fill();

        // Inner
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, Math.PI * 2);
        ctx.fillStyle = `${col}90`;
        ctx.fill();
        ctx.strokeStyle = col;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        if (t.label) {
          ctx.fillStyle = col;
          ctx.font = 'bold 10px sans-serif';
          ctx.fillText(t.label, cx + 14, cy + 4);
        }
      });
    }

    // Drones
    const droneIds = Object.keys(mapPositions);
    droneIds.forEach((id) => {
      const pos = mapPositions[id];
      if (!pos) return;
      const { cx, cy } = toCanvas(pos.x, pos.y);
      const drone = drones[id] || {};
      const state = drone.state || drone.status || 'active';
      const col = DRONE_STATE_COLORS[state] || DRONE_STATE_COLORS.active;
      const heading = pos.heading || 0;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate((heading * Math.PI) / 180);

      // Triangle body
      ctx.beginPath();
      ctx.moveTo(0, -10);
      ctx.lineTo(-7, 7);
      ctx.lineTo(7, 7);
      ctx.closePath();
      ctx.fillStyle = col;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();

      // Label
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '10px sans-serif';
      ctx.fillText(drone.name || id, cx + 12, cy + 4);

      // Highlight hovered
      if (hoveredDrone === id) {
        ctx.beginPath();
        ctx.arc(cx, cy, 16, 0, Math.PI * 2);
        ctx.strokeStyle = '#93bbfd';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  }, [canvasSize, mapPositions, drones, mapThreats, zones, hoveredDrone, showZones, showThreats, toCanvas, scale]);

  /* Mouse hover detection */
  const handleMouseMove = useCallback(
    (e) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let found = null;
      for (const [id, pos] of Object.entries(mapPositions)) {
        const { cx, cy } = toCanvas(pos.x, pos.y);
        if (Math.hypot(mx - cx, my - cy) < 16) {
          found = id;
          break;
        }
      }
      setHoveredDrone(found);
    },
    [mapPositions, toCanvas],
  );

  const hasDrones = Object.keys(mapPositions).length > 0;

  return (
    <div style={styles.container}>
      {/* Toolbar */}
      <div style={styles.toolbar}>
        <span style={styles.toolbarTitle}>SITUATIONAL MAP</span>
        <label style={styles.toggle}>
          <input
            type="checkbox"
            checked={showHeatmap}
            onChange={() => setShowHeatmap(!showHeatmap)}
          />
          <span>Heatmap</span>
        </label>
        <label style={styles.toggle}>
          <input
            type="checkbox"
            checked={showZones}
            onChange={() => setShowZones(!showZones)}
          />
          <span>Zones</span>
        </label>
        <label style={styles.toggle}>
          <input
            type="checkbox"
            checked={showThreats}
            onChange={() => setShowThreats(!showThreats)}
          />
          <span>Threats</span>
        </label>
        <span style={styles.coordLabel}>
          {WORLD_SIZE}m x {WORLD_SIZE}m
        </span>
      </div>

      {/* Canvas stack */}
      <div ref={containerRef} style={styles.canvasWrap}>
        {/* Heatmap underlay */}
        {showHeatmap && crowdGrid && (
          <CrowdHeatmapLayer
            grid={crowdGrid}
            width={canvasSize.w}
            height={canvasSize.h}
            worldSize={WORLD_SIZE}
          />
        )}
        <canvas
          ref={canvasRef}
          style={styles.canvas}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredDrone(null)}
        />
        {!hasDrones && (
          <div style={styles.emptyOverlay}>
            <div style={styles.emptyText}>Waiting for drone positions...</div>
            <div style={styles.emptyHint}>
              Connect to ws://localhost:8765 to receive map_update messages
            </div>
          </div>
        )}
      </div>

      {/* Hovered drone tooltip */}
      {hoveredDrone && mapPositions[hoveredDrone] && (
        <div style={styles.tooltip}>
          <strong>{drones[hoveredDrone]?.name || hoveredDrone}</strong>
          <br />
          X: {mapPositions[hoveredDrone].x?.toFixed(1)}m &nbsp;
          Y: {mapPositions[hoveredDrone].y?.toFixed(1)}m &nbsp;
          Z: {mapPositions[hoveredDrone].z?.toFixed(1)}m
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    overflow: 'hidden',
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    padding: '8px 14px',
    background: PALETTE.surface,
    borderBottom: `1px solid ${PALETTE.border}`,
    flexShrink: 0,
  },
  toolbarTitle: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: PALETTE.textMuted,
    marginRight: 'auto',
  },
  toggle: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 11,
    color: PALETTE.text,
    cursor: 'pointer',
  },
  coordLabel: { fontSize: 10, color: PALETTE.textMuted },
  canvasWrap: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
  },
  emptyOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    pointerEvents: 'none',
  },
  emptyText: { fontSize: 16, color: PALETTE.textMuted, marginBottom: 8 },
  emptyHint: { fontSize: 12, color: PALETTE.borderLight },
  tooltip: {
    position: 'absolute',
    bottom: 14,
    left: 14,
    background: 'rgba(17,24,39,0.92)',
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 6,
    padding: '8px 12px',
    fontSize: 12,
    color: PALETTE.text,
    pointerEvents: 'none',
  },
};
