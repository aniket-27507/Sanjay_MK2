import React, { useState, useCallback } from 'react';
import useGCSState from '../../hooks/useGCSState';
import { THREAT_COLORS, PALETTE, riskLevelLabel } from '../../utils/colors';

export default function IncidentCommand({ wsSend }) {
  const threats = useGCSState((s) => s.threats);
  const threatHistory = useGCSState((s) => s.threatHistory);
  const acknowledgeAlert = useGCSState((s) => s.acknowledgeAlert);
  const addAuditEntry = useGCSState((s) => s.addAuditEntry);
  const drones = useGCSState((s) => s.drones);

  const [filter, setFilter] = useState('all'); // all | active | acknowledged
  const [dispatchTarget, setDispatchTarget] = useState(null);

  const displayed =
    filter === 'active'
      ? threats.filter((t) => !t.acknowledged)
      : filter === 'acknowledged'
        ? threats.filter((t) => t.acknowledged)
        : threats;

  const handleAcknowledge = useCallback(
    (threat) => {
      acknowledgeAlert(threat.id);
      addAuditEntry({
        action: 'THREAT_ACKNOWLEDGED',
        detail: `Acknowledged: ${threat.label} (${threat.id})`,
        source: 'operator',
      });
      if (wsSend) {
        wsSend({ type: 'ack_threat', threat_id: threat.id });
      }
    },
    [acknowledgeAlert, addAuditEntry, wsSend],
  );

  const handleEscalate = useCallback(
    (threat) => {
      addAuditEntry({
        level: 'warning',
        action: 'THREAT_ESCALATED',
        detail: `Escalated: ${threat.label} (${threat.id}) to command authority`,
        source: 'operator',
      });
      if (wsSend) {
        wsSend({ type: 'escalate_threat', threat_id: threat.id });
      }
    },
    [addAuditEntry, wsSend],
  );

  const handleDispatch = useCallback(
    (threat, droneId) => {
      addAuditEntry({
        action: 'DRONE_DISPATCHED',
        detail: `Dispatched ${droneId} to threat ${threat.label} (${threat.id})`,
        source: 'operator',
      });
      if (wsSend) {
        wsSend({
          type: 'dispatch_drone',
          drone_id: droneId,
          threat_id: threat.id,
          position: threat.position,
        });
      }
      setDispatchTarget(null);
    },
    [addAuditEntry, wsSend],
  );

  const droneIds = Object.keys(drones);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>INCIDENT COMMAND</span>
        <div style={styles.filterGroup}>
          {['all', 'active', 'acknowledged'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                ...styles.filterBtn,
                ...(filter === f ? styles.filterBtnActive : {}),
              }}
            >
              {f.toUpperCase()}
            </button>
          ))}
        </div>
        <span style={styles.count}>
          {displayed.length} incident{displayed.length !== 1 ? 's' : ''}
        </span>
      </div>

      {displayed.length === 0 ? (
        <div style={styles.empty}>
          <div style={styles.emptyIcon}>{'\u26A0'}</div>
          <div style={styles.emptyText}>
            {threats.length === 0
              ? 'No incidents reported'
              : 'No matching incidents'}
          </div>
          <div style={styles.emptyHint}>
            Threat events from the backend will appear here as actionable cards
          </div>
        </div>
      ) : (
        <div style={styles.list}>
          {displayed.map((threat) => {
            const sevColor =
              THREAT_COLORS[threat.severity] || THREAT_COLORS.warning;
            const ts = threat.timestamp
              ? new Date(threat.timestamp).toLocaleTimeString()
              : '--:--:--';

            return (
              <div
                key={threat.id}
                style={{
                  ...styles.card,
                  borderLeftColor: sevColor,
                  opacity: threat.acknowledged ? 0.6 : 1,
                }}
              >
                <div style={styles.cardHeader}>
                  <span style={{ ...styles.sevBadge, backgroundColor: `${sevColor}25`, color: sevColor }}>
                    {(threat.severity || 'ALERT').toUpperCase()}
                  </span>
                  <span style={styles.cardLabel}>{threat.label}</span>
                  <span style={styles.cardTime}>{ts}</span>
                </div>

                {threat.description && (
                  <div style={styles.cardDesc}>{threat.description}</div>
                )}

                {threat.position && (
                  <div style={styles.cardPos}>
                    Position: ({threat.position.x?.toFixed(1)}, {threat.position.y?.toFixed(1)})
                  </div>
                )}

                <div style={styles.cardActions}>
                  {!threat.acknowledged && (
                    <button
                      style={{ ...styles.actionBtn, ...styles.ackBtn }}
                      onClick={() => handleAcknowledge(threat)}
                    >
                      ACKNOWLEDGE
                    </button>
                  )}
                  <button
                    style={{ ...styles.actionBtn, ...styles.escBtn }}
                    onClick={() => handleEscalate(threat)}
                  >
                    ESCALATE
                  </button>
                  <button
                    style={{ ...styles.actionBtn, ...styles.dispatchBtn }}
                    onClick={() =>
                      setDispatchTarget(
                        dispatchTarget === threat.id ? null : threat.id,
                      )
                    }
                  >
                    DISPATCH
                  </button>
                </div>

                {/* Drone dispatch selector */}
                {dispatchTarget === threat.id && (
                  <div style={styles.dispatchPanel}>
                    <div style={styles.dispatchTitle}>Select drone to dispatch:</div>
                    {droneIds.length === 0 ? (
                      <div style={styles.noDrones}>No drones available</div>
                    ) : (
                      <div style={styles.droneSelector}>
                        {droneIds.map((did) => (
                          <button
                            key={did}
                            style={styles.droneOption}
                            onClick={() => handleDispatch(threat, did)}
                          >
                            {drones[did]?.name || did}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Recent history */}
      {threatHistory.length > 0 && (
        <div style={styles.historySection}>
          <div style={styles.historyTitle}>RECENT EVENTS ({threatHistory.length})</div>
          <div style={styles.historyList}>
            {threatHistory.slice(0, 20).map((t, i) => {
              const sevColor = THREAT_COLORS[t.severity] || THREAT_COLORS.warning;
              const ts = t.timestamp
                ? new Date(t.timestamp).toLocaleTimeString()
                : '';
              return (
                <div key={t.id || i} style={styles.historyRow}>
                  <span style={{ ...styles.historyDot, backgroundColor: sevColor }} />
                  <span style={styles.historyTime}>{ts}</span>
                  <span style={styles.historyLabel}>{t.label}</span>
                </div>
              );
            })}
          </div>
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
    overflow: 'hidden',
    padding: 16,
    gap: 12,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    flexShrink: 0,
    flexWrap: 'wrap',
  },
  title: {
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: PALETTE.textBright,
    marginRight: 'auto',
  },
  filterGroup: { display: 'flex', gap: 4 },
  filterBtn: {
    fontSize: 10,
    fontWeight: 600,
    padding: '4px 10px',
    borderRadius: 4,
    border: `1px solid ${PALETTE.borderLight}`,
    background: 'transparent',
    color: PALETTE.textMuted,
    cursor: 'pointer',
  },
  filterBtnActive: {
    background: 'rgba(59,130,246,0.15)',
    borderColor: PALETTE.accent,
    color: '#93bbfd',
  },
  count: { fontSize: 11, color: PALETTE.textMuted },
  empty: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  emptyIcon: { fontSize: 40, color: PALETTE.borderLight },
  emptyText: { fontSize: 14, color: PALETTE.textMuted },
  emptyHint: { fontSize: 11, color: PALETTE.borderLight, textAlign: 'center', maxWidth: 320 },
  list: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  card: {
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderLeft: '4px solid',
    borderRadius: 8,
    padding: '12px 14px',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    marginBottom: 6,
  },
  sevBadge: {
    fontSize: 9,
    fontWeight: 800,
    letterSpacing: 1,
    padding: '2px 8px',
    borderRadius: 4,
  },
  cardLabel: { flex: 1, fontSize: 13, fontWeight: 600, color: PALETTE.textBright },
  cardTime: { fontSize: 11, color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums' },
  cardDesc: { fontSize: 12, color: PALETTE.text, marginBottom: 6 },
  cardPos: { fontSize: 11, color: PALETTE.textMuted, marginBottom: 8 },
  cardActions: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  actionBtn: {
    fontSize: 10,
    fontWeight: 700,
    padding: '5px 12px',
    borderRadius: 4,
    border: 'none',
    cursor: 'pointer',
    letterSpacing: 0.5,
  },
  ackBtn: {
    background: 'rgba(34,197,94,0.15)',
    color: '#22c55e',
    border: '1px solid rgba(34,197,94,0.3)',
  },
  escBtn: {
    background: 'rgba(249,115,22,0.15)',
    color: '#f97316',
    border: '1px solid rgba(249,115,22,0.3)',
  },
  dispatchBtn: {
    background: 'rgba(59,130,246,0.15)',
    color: '#3b82f6',
    border: '1px solid rgba(59,130,246,0.3)',
  },
  dispatchPanel: {
    marginTop: 8,
    padding: '8px 10px',
    background: PALETTE.surfaceLight,
    borderRadius: 6,
  },
  dispatchTitle: { fontSize: 11, color: PALETTE.textMuted, marginBottom: 6 },
  noDrones: { fontSize: 11, color: PALETTE.borderLight },
  droneSelector: { display: 'flex', gap: 6, flexWrap: 'wrap' },
  droneOption: {
    fontSize: 11,
    fontWeight: 600,
    padding: '4px 10px',
    borderRadius: 4,
    border: `1px solid ${PALETTE.accent}`,
    background: 'rgba(59,130,246,0.1)',
    color: '#93bbfd',
    cursor: 'pointer',
  },
  historySection: {
    flexShrink: 0,
    maxHeight: 200,
    borderTop: `1px solid ${PALETTE.border}`,
    paddingTop: 10,
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  historyTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: PALETTE.textMuted,
  },
  historyList: { overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 3 },
  historyRow: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 },
  historyDot: { width: 6, height: 6, borderRadius: '50%', flexShrink: 0 },
  historyTime: { color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums', width: 70, flexShrink: 0 },
  historyLabel: { color: PALETTE.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
};
