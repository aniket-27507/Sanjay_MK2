import React, { useRef, useState, useEffect, useCallback } from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE } from '../../utils/colors';

const WORLD_SIZE = 1000;
const ZONE_COLORS = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ec4899', '#14b8a6', '#eab308'];

export default function ZoneEditor({ wsSend }) {
  const zones = useGCSState((s) => s.zones);
  const updateZones = useGCSState((s) => s.updateZones);
  const addAuditEntry = useGCSState((s) => s.addAuditEntry);

  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [canvasSize, setCanvasSize] = useState({ w: 800, h: 600 });
  const [currentPoly, setCurrentPoly] = useState([]); // points being drawn
  const [drawing, setDrawing] = useState(false);
  const [zoneName, setZoneName] = useState('');
  const [selectedZone, setSelectedZone] = useState(null);

  /* Resize */
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

  const scale = Math.min(canvasSize.w, canvasSize.h) / WORLD_SIZE;

  const toCanvas = useCallback((wx, wy) => ({ cx: wx * scale, cy: wy * scale }), [scale]);
  const toWorld = useCallback(
    (cx, cy) => ({ x: cx / scale, y: cy / scale }),
    [scale],
  );

  /* Draw */
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
    ctx.strokeStyle = 'rgba(30,58,95,0.3)';
    ctx.lineWidth = 0.5;
    const step = 100 * scale;
    for (let x = 0; x <= w; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y <= h; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Existing zones
    zones.forEach((zone, idx) => {
      if (!zone.polygon || zone.polygon.length < 3) return;
      const col = zone.color || ZONE_COLORS[idx % ZONE_COLORS.length];
      const isSelected = selectedZone === idx;

      ctx.beginPath();
      zone.polygon.forEach((pt, i) => {
        const { cx, cy } = toCanvas(pt.x, pt.y);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.closePath();

      ctx.fillStyle = isSelected ? `${col}30` : `${col}18`;
      ctx.fill();
      ctx.strokeStyle = isSelected ? col : `${col}80`;
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.stroke();

      // Vertices
      zone.polygon.forEach((pt) => {
        const { cx, cy } = toCanvas(pt.x, pt.y);
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
      });

      // Label
      if (zone.label) {
        const ctr = zone.polygon.reduce(
          (a, p) => ({ x: a.x + p.x, y: a.y + p.y }),
          { x: 0, y: 0 },
        );
        ctr.x /= zone.polygon.length;
        ctr.y /= zone.polygon.length;
        const { cx, cy } = toCanvas(ctr.x, ctr.y);
        ctx.fillStyle = 'rgba(200,208,224,0.85)';
        ctx.font = 'bold 12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(zone.label, cx, cy);
        ctx.textAlign = 'start';
      }
    });

    // Current polygon being drawn
    if (currentPoly.length > 0) {
      ctx.beginPath();
      currentPoly.forEach((pt, i) => {
        const { cx, cy } = toCanvas(pt.x, pt.y);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.strokeStyle = '#93bbfd';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Points
      currentPoly.forEach((pt) => {
        const { cx, cy } = toCanvas(pt.x, pt.y);
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#3b82f6';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }
  }, [canvasSize, zones, currentPoly, selectedZone, scale, toCanvas]);

  /* Click to add point */
  const handleCanvasClick = useCallback(
    (e) => {
      if (!drawing) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const world = toWorld(mx, my);
      setCurrentPoly((prev) => [...prev, { x: Math.round(world.x), y: Math.round(world.y) }]);
    },
    [drawing, toWorld],
  );

  /* Start drawing */
  const startDrawing = () => {
    setDrawing(true);
    setCurrentPoly([]);
    setSelectedZone(null);
  };

  /* Finish polygon */
  const finishPolygon = () => {
    if (currentPoly.length < 3) return;
    const name = zoneName.trim() || `Zone ${zones.length + 1}`;
    const color = ZONE_COLORS[zones.length % ZONE_COLORS.length];
    const newZone = {
      id: `zone-${Date.now()}`,
      label: name,
      polygon: currentPoly,
      color,
    };
    const updated = [...zones, newZone];
    updateZones(updated);
    addAuditEntry({
      action: 'ZONE_CREATED',
      detail: `Created zone "${name}" with ${currentPoly.length} vertices`,
      source: 'operator',
    });
    if (wsSend) {
      wsSend({ type: 'zone_update', zones: updated });
    }
    setCurrentPoly([]);
    setDrawing(false);
    setZoneName('');
  };

  /* Cancel drawing */
  const cancelDrawing = () => {
    setCurrentPoly([]);
    setDrawing(false);
  };

  /* Undo last point */
  const undoPoint = () => {
    setCurrentPoly((prev) => prev.slice(0, -1));
  };

  /* Delete selected zone */
  const deleteZone = () => {
    if (selectedZone == null) return;
    const removed = zones[selectedZone];
    const updated = zones.filter((_, i) => i !== selectedZone);
    updateZones(updated);
    addAuditEntry({
      action: 'ZONE_DELETED',
      detail: `Deleted zone "${removed?.label || selectedZone}"`,
      source: 'operator',
    });
    if (wsSend) {
      wsSend({ type: 'zone_update', zones: updated });
    }
    setSelectedZone(null);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>ZONE EDITOR</span>
        <span style={styles.subtitle}>{zones.length} zone{zones.length !== 1 ? 's' : ''} defined</span>
      </div>

      {/* Toolbar */}
      <div style={styles.toolbar}>
        {!drawing ? (
          <>
            <button style={styles.primaryBtn} onClick={startDrawing}>
              + NEW ZONE
            </button>
            {selectedZone != null && (
              <button style={styles.dangerBtn} onClick={deleteZone}>
                DELETE SELECTED
              </button>
            )}
          </>
        ) : (
          <>
            <input
              style={styles.nameInput}
              placeholder="Zone name..."
              value={zoneName}
              onChange={(e) => setZoneName(e.target.value)}
            />
            <span style={styles.pointCount}>{currentPoly.length} pts</span>
            <button
              style={styles.secondaryBtn}
              onClick={undoPoint}
              disabled={currentPoly.length === 0}
            >
              UNDO
            </button>
            <button
              style={{
                ...styles.primaryBtn,
                opacity: currentPoly.length < 3 ? 0.4 : 1,
              }}
              onClick={finishPolygon}
              disabled={currentPoly.length < 3}
            >
              FINISH
            </button>
            <button style={styles.cancelBtn} onClick={cancelDrawing}>
              CANCEL
            </button>
          </>
        )}
      </div>

      {/* Instructions */}
      {drawing && (
        <div style={styles.instructions}>
          Click on the canvas to place polygon vertices. At least 3 points required. Press FINISH when done.
        </div>
      )}

      {/* Canvas + zone list */}
      <div style={styles.body}>
        <div ref={containerRef} style={styles.canvasWrap}>
          <canvas
            ref={canvasRef}
            style={styles.canvas}
            onClick={handleCanvasClick}
          />
        </div>

        {/* Zone list panel */}
        <div style={styles.zoneList}>
          <div style={styles.zoneListTitle}>ZONES</div>
          {zones.length === 0 && (
            <div style={styles.noZones}>No zones defined. Click "+ NEW ZONE" to begin.</div>
          )}
          {zones.map((zone, idx) => (
            <div
              key={zone.id || idx}
              style={{
                ...styles.zoneItem,
                borderLeftColor: zone.color || ZONE_COLORS[idx % ZONE_COLORS.length],
                background: selectedZone === idx ? PALETTE.surfaceLight : PALETTE.surface,
              }}
              onClick={() => setSelectedZone(selectedZone === idx ? null : idx)}
            >
              <div style={styles.zoneItemName}>{zone.label || `Zone ${idx + 1}`}</div>
              <div style={styles.zoneItemMeta}>
                {zone.polygon?.length || 0} vertices
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    padding: 16,
    gap: 10,
  },
  header: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 12,
    flexShrink: 0,
  },
  title: { fontSize: 13, fontWeight: 700, letterSpacing: 1.2, color: PALETTE.textBright },
  subtitle: { fontSize: 11, color: PALETTE.textMuted },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    flexShrink: 0,
    flexWrap: 'wrap',
  },
  primaryBtn: {
    fontSize: 11,
    fontWeight: 700,
    padding: '6px 14px',
    borderRadius: 5,
    border: 'none',
    background: PALETTE.accent,
    color: '#fff',
    cursor: 'pointer',
  },
  secondaryBtn: {
    fontSize: 11,
    fontWeight: 600,
    padding: '5px 12px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.borderLight}`,
    background: 'transparent',
    color: PALETTE.text,
    cursor: 'pointer',
  },
  dangerBtn: {
    fontSize: 11,
    fontWeight: 700,
    padding: '6px 14px',
    borderRadius: 5,
    border: `1px solid rgba(239,68,68,0.4)`,
    background: 'rgba(239,68,68,0.1)',
    color: '#ef4444',
    cursor: 'pointer',
  },
  cancelBtn: {
    fontSize: 11,
    fontWeight: 600,
    padding: '5px 12px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.borderLight}`,
    background: 'transparent',
    color: PALETTE.textMuted,
    cursor: 'pointer',
  },
  nameInput: {
    fontSize: 12,
    padding: '5px 10px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.border}`,
    background: PALETTE.surfaceLight,
    color: PALETTE.textBright,
    outline: 'none',
    width: 150,
  },
  pointCount: { fontSize: 11, color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums' },
  instructions: {
    fontSize: 11,
    color: PALETTE.accent,
    background: 'rgba(59,130,246,0.08)',
    padding: '6px 12px',
    borderRadius: 5,
    border: `1px solid rgba(59,130,246,0.2)`,
    flexShrink: 0,
  },
  body: {
    flex: 1,
    display: 'flex',
    gap: 12,
    overflow: 'hidden',
  },
  canvasWrap: {
    flex: 1,
    position: 'relative',
    borderRadius: 8,
    overflow: 'hidden',
    border: `1px solid ${PALETTE.border}`,
  },
  canvas: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    cursor: 'crosshair',
  },
  zoneList: {
    width: 200,
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    overflowY: 'auto',
  },
  zoneListTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.5,
    color: PALETTE.textMuted,
    marginBottom: 4,
  },
  noZones: { fontSize: 11, color: PALETTE.borderLight },
  zoneItem: {
    padding: '8px 10px',
    borderRadius: 6,
    border: `1px solid ${PALETTE.border}`,
    borderLeft: '3px solid',
    cursor: 'pointer',
    transition: 'background .15s',
  },
  zoneItemName: { fontSize: 12, fontWeight: 600, color: PALETTE.textBright },
  zoneItemMeta: { fontSize: 10, color: PALETTE.textMuted, marginTop: 2 },
};
