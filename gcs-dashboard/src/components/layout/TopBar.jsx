import React, { useState, useEffect } from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE, THREAT_COLORS } from '../../utils/colors';

export default function TopBar({ connected, latencyMs }) {
  const missionName = useGCSState((s) => s.missionName);
  const activeAlerts = useGCSState((s) => s.activeAlerts);
  const acknowledgeAlert = useGCSState((s) => s.acknowledgeAlert);
  const clearAlerts = useGCSState((s) => s.clearAlerts);

  const [clock, setClock] = useState(formatClock());

  useEffect(() => {
    const id = setInterval(() => setClock(formatClock()), 1000);
    return () => clearInterval(id);
  }, []);

  const topAlert = activeAlerts[0] || null;
  const alertCount = activeAlerts.length;

  return (
    <header style={styles.bar}>
      {/* Left — mission */}
      <div style={styles.left}>
        <span style={styles.missionLabel}>MISSION</span>
        <span style={styles.missionName}>{missionName}</span>
      </div>

      {/* Center — alert banner */}
      {topAlert && (
        <div
          style={{
            ...styles.alertBanner,
            borderColor: THREAT_COLORS[topAlert.severity] || THREAT_COLORS.warning,
            background: `${THREAT_COLORS[topAlert.severity] || THREAT_COLORS.warning}18`,
          }}
        >
          <span
            style={{
              ...styles.alertSeverity,
              color: THREAT_COLORS[topAlert.severity] || THREAT_COLORS.warning,
            }}
          >
            {(topAlert.severity || 'ALERT').toUpperCase()}
          </span>
          <span style={styles.alertText}>
            {topAlert.label}
            {topAlert.description ? ` — ${topAlert.description}` : ''}
          </span>
          {alertCount > 1 && (
            <span style={styles.alertBadge}>+{alertCount - 1}</span>
          )}
          <button
            style={styles.alertAck}
            onClick={() => acknowledgeAlert(topAlert.id)}
            title="Acknowledge"
          >
            ACK
          </button>
          {alertCount > 1 && (
            <button
              style={styles.alertClear}
              onClick={clearAlerts}
              title="Clear all"
            >
              CLEAR ALL
            </button>
          )}
        </div>
      )}

      {/* Right — clock + connection */}
      <div style={styles.right}>
        <div style={styles.connBox}>
          <span
            style={{
              ...styles.connDot,
              backgroundColor: connected ? PALETTE.success : PALETTE.error,
            }}
          />
          <span style={styles.connLabel}>
            {connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
          {connected && latencyMs != null && (
            <span style={styles.latency}>{latencyMs}ms</span>
          )}
        </div>
        <div style={styles.clock}>{clock}</div>
      </div>
    </header>
  );
}

function formatClock() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

const styles = {
  bar: {
    height: 48,
    minHeight: 48,
    background: '#0d1224',
    borderBottom: `1px solid ${PALETTE.border}`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 20px',
    gap: 16,
  },
  left: { display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 },
  missionLabel: {
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: 1.5,
    color: PALETTE.textMuted,
  },
  missionName: {
    fontSize: 14,
    fontWeight: 600,
    color: PALETTE.textBright,
  },
  alertBanner: {
    flex: 1,
    maxWidth: 700,
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '5px 14px',
    borderRadius: 6,
    border: '1px solid',
  },
  alertSeverity: { fontSize: 11, fontWeight: 800, letterSpacing: 1 },
  alertText: { flex: 1, fontSize: 12, color: PALETTE.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  alertBadge: {
    fontSize: 10,
    fontWeight: 700,
    color: PALETTE.textMuted,
    background: PALETTE.surfaceLight,
    padding: '2px 6px',
    borderRadius: 10,
  },
  alertAck: {
    fontSize: 10,
    fontWeight: 700,
    color: PALETTE.accent,
    background: 'transparent',
    border: `1px solid ${PALETTE.accent}`,
    borderRadius: 4,
    padding: '2px 8px',
    cursor: 'pointer',
  },
  alertClear: {
    fontSize: 10,
    fontWeight: 700,
    color: PALETTE.textMuted,
    background: 'transparent',
    border: `1px solid ${PALETTE.borderLight}`,
    borderRadius: 4,
    padding: '2px 8px',
    cursor: 'pointer',
  },
  right: { display: 'flex', alignItems: 'center', gap: 18, flexShrink: 0 },
  connBox: { display: 'flex', alignItems: 'center', gap: 6 },
  connDot: { width: 8, height: 8, borderRadius: '50%' },
  connLabel: { fontSize: 10, fontWeight: 700, letterSpacing: 0.8, color: PALETTE.textMuted },
  latency: { fontSize: 10, color: PALETTE.textMuted },
  clock: {
    fontSize: 18,
    fontWeight: 700,
    fontVariantNumeric: 'tabular-nums',
    color: PALETTE.textBright,
    letterSpacing: 1,
  },
};
