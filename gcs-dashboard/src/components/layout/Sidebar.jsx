import React from 'react';
import useGCSState from '../../hooks/useGCSState';
import { DRONE_STATE_COLORS, PALETTE } from '../../utils/colors';

const TABS = [
  { id: 'map', icon: '\u25C9', label: 'Situational Map' },
  { id: 'cameras', icon: '\u25A3', label: 'Camera Feeds' },
  { id: 'crowd', icon: '\u2630', label: 'Crowd Analysis' },
  { id: 'command', icon: '\u26A0', label: 'Incident Command' },
  { id: 'zones', icon: '\u2B21', label: 'Zone Editor' },
  { id: 'evidence', icon: '\u25CF', label: 'Evidence' },
  { id: 'audit', icon: '\u2263', label: 'Audit Log' },
];

export default function Sidebar({ activeTab, onTabChange }) {
  const drones = useGCSState((s) => s.drones);
  const telemetry = useGCSState((s) => s.telemetry);
  const droneOrder = useGCSState((s) => s.droneOrder);

  const droneList = droneOrder.length > 0 ? droneOrder : Object.keys(drones);
  const totalDrones = droneList.length;
  const activeDrones = droneList.filter(
    (id) => drones[id]?.state === 'active' || drones[id]?.status === 'active',
  ).length;

  return (
    <aside style={styles.sidebar}>
      {/* Logo area */}
      <div style={styles.logo}>
        <div style={styles.logoIcon}>S</div>
        <div style={styles.logoText}>
          <span style={styles.logoTitle}>SANJAY MK2</span>
          <span style={styles.logoSub}>GCS COMMAND</span>
        </div>
      </div>

      {/* Navigation */}
      <nav style={styles.nav}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            style={{
              ...styles.navItem,
              ...(activeTab === tab.id ? styles.navItemActive : {}),
            }}
            title={tab.label}
          >
            <span style={styles.navIcon}>{tab.icon}</span>
            <span style={styles.navLabel}>{tab.label}</span>
          </button>
        ))}
      </nav>

      {/* Fleet summary */}
      <div style={styles.fleetBox}>
        <div style={styles.fleetTitle}>FLEET STATUS</div>
        <div style={styles.fleetStats}>
          <div style={styles.statRow}>
            <span style={styles.statLabel}>Total</span>
            <span style={styles.statValue}>{totalDrones || '--'}</span>
          </div>
          <div style={styles.statRow}>
            <span style={styles.statLabel}>Active</span>
            <span style={{ ...styles.statValue, color: PALETTE.success }}>
              {totalDrones > 0 ? activeDrones : '--'}
            </span>
          </div>
        </div>

        {/* Mini drone list */}
        <div style={styles.droneList}>
          {droneList.length === 0 && (
            <div style={styles.noDrones}>No drones connected</div>
          )}
          {droneList.slice(0, 10).map((id) => {
            const d = drones[id] || {};
            const t = telemetry[id] || {};
            const state = d.state || d.status || 'offline';
            const color = DRONE_STATE_COLORS[state] || DRONE_STATE_COLORS.offline;
            const battery = t.battery != null ? `${Math.round(t.battery)}%` : '--';
            return (
              <div key={id} style={styles.droneRow}>
                <span style={{ ...styles.droneIndicator, backgroundColor: color }} />
                <span style={styles.droneId}>{d.name || id}</span>
                <span style={styles.droneBat}>{battery}</span>
              </div>
            );
          })}
        </div>
      </div>
    </aside>
  );
}

const styles = {
  sidebar: {
    width: 220,
    minWidth: 220,
    height: '100vh',
    background: '#0d1224',
    borderRight: `1px solid ${PALETTE.border}`,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '18px 16px 14px',
    borderBottom: `1px solid ${PALETTE.border}`,
  },
  logoIcon: {
    width: 36,
    height: 36,
    borderRadius: 8,
    background: 'linear-gradient(135deg, #3b82f6, #1e40af)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 800,
    fontSize: 18,
    color: '#fff',
  },
  logoText: { display: 'flex', flexDirection: 'column' },
  logoTitle: { fontSize: 13, fontWeight: 700, color: '#f1f5f9', letterSpacing: 1.2 },
  logoSub: { fontSize: 9, color: PALETTE.textMuted, letterSpacing: 1.5, marginTop: 1 },
  nav: {
    flex: '0 0 auto',
    display: 'flex',
    flexDirection: 'column',
    padding: '10px 8px',
    gap: 2,
  },
  navItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '9px 12px',
    border: 'none',
    borderRadius: 6,
    background: 'transparent',
    color: PALETTE.textMuted,
    fontSize: 13,
    cursor: 'pointer',
    transition: 'all .15s',
    textAlign: 'left',
  },
  navItemActive: {
    background: 'rgba(59,130,246,0.15)',
    color: '#93bbfd',
  },
  navIcon: { fontSize: 16, width: 20, textAlign: 'center' },
  navLabel: { whiteSpace: 'nowrap' },
  fleetBox: {
    flex: 1,
    margin: '0 8px 12px',
    padding: '10px 10px',
    borderRadius: 8,
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  fleetTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.5,
    color: PALETTE.textMuted,
    marginBottom: 8,
  },
  fleetStats: { display: 'flex', gap: 16, marginBottom: 10 },
  statRow: { display: 'flex', flexDirection: 'column' },
  statLabel: { fontSize: 10, color: PALETTE.textMuted },
  statValue: { fontSize: 20, fontWeight: 700, color: PALETTE.textBright },
  droneList: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 },
  noDrones: { fontSize: 11, color: PALETTE.textMuted, padding: '6px 0' },
  droneRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '4px 0',
    fontSize: 12,
  },
  droneIndicator: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    flexShrink: 0,
  },
  droneId: { flex: 1, color: PALETTE.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  droneBat: { color: PALETTE.textMuted, fontSize: 11, flexShrink: 0 },
};
