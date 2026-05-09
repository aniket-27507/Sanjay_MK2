import React, { useMemo } from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE } from '../../utils/colors';

/**
 * AI Incident Review Panel
 * ------------------------
 * Operator-facing surface for the AI workflow (`scripts/demo_operator_workflow.py`).
 *
 * Pending incidents arrive as `ai_incident` WebSocket messages and render as
 * cards with the cropped detection thumbnail, class, confidence, and three
 * classification buttons (SAFE / THREAT / DISMISS).  Clicking a button sends
 * `incident_decision` back over the WebSocket; the AI workflow finalises the
 * incident and broadcasts `ai_incident_resolved` so the card disappears.
 *
 * Recent decisions are kept in `aiIncidentHistory` and listed below the
 * pending cards as the run-time audit trail.
 */
export default function AIIncidentReview({ wsSend }) {
  const aiIncidents = useGCSState((s) => s.aiIncidents);
  const aiIncidentHistory = useGCSState((s) => s.aiIncidentHistory);
  const classifyIncident = useGCSState((s) => s.classifyIncident);
  const addAuditEntry = useGCSState((s) => s.addAuditEntry);

  const pending = useMemo(
    () => [...aiIncidents].sort((a, b) => (b.triggered_at || 0) - (a.triggered_at || 0)),
    [aiIncidents],
  );

  const handleClassify = (incident, decision) => {
    classifyIncident(incident.incident_id, decision, wsSend);
    addAuditEntry({
      level: decision === 'THREAT' ? 'warning' : 'info',
      action: `INCIDENT_${decision}`,
      detail: `Operator classified ${incident.class} (${(incident.confidence * 100).toFixed(0)}%) as ${decision} -- ${incident.incident_id}`,
      source: 'operator',
    });
  };

  return (
    <div style={styles.root}>
      <div style={styles.header}>
        <div style={styles.title}>AI INCIDENT REVIEW</div>
        <div style={styles.count}>
          {pending.length} pending &nbsp;&middot;&nbsp; {aiIncidentHistory.length} resolved
        </div>
      </div>

      {pending.length === 0 ? (
        <div style={styles.empty}>
          <div style={styles.emptyIcon}>—</div>
          <div style={styles.emptyTitle}>No pending incidents</div>
          <div style={styles.emptyHint}>
            The AI workflow will push detections here when a threat-class object is
            recognised. Start it with{' '}
            <code style={styles.code}>--gcs-port 8765</code> to enable the bridge.
          </div>
        </div>
      ) : (
        <div style={styles.cardGrid}>
          {pending.map((inc) => (
            <IncidentCard
              key={inc.incident_id}
              incident={inc}
              onClassify={handleClassify}
            />
          ))}
        </div>
      )}

      <div style={styles.historyHeader}>RECENT DECISIONS</div>
      {aiIncidentHistory.length === 0 ? (
        <div style={styles.historyEmpty}>No decisions logged yet this session.</div>
      ) : (
        <div style={styles.historyList}>
          {aiIncidentHistory.slice(0, 12).map((h) => (
            <HistoryRow key={h.incident_id + h.decided_at} entry={h} />
          ))}
        </div>
      )}
    </div>
  );
}

function IncidentCard({ incident, onClassify }) {
  const ageSec = Math.max(0, (Date.now() - (incident.triggered_at || Date.now())) / 1000);
  const classUpper = (incident.class || 'unknown').toUpperCase();
  const isThreatClass = ['WEAPON_PERSON', 'EXPLOSIVE_DEVICE', 'FIRE'].includes(classUpper);
  const confPct = Math.round((incident.confidence || 0) * 100);

  return (
    <div style={{ ...styles.card, borderColor: isThreatClass ? PALETTE.error : PALETTE.warning }}>
      <div style={styles.cardHeader}>
        <span
          style={{
            ...styles.classBadge,
            backgroundColor: isThreatClass ? PALETTE.error : PALETTE.warning,
          }}
        >
          {classUpper}
        </span>
        <span style={styles.cardAge}>{ageSec.toFixed(0)}s ago</span>
      </div>
      {incident.thumbnail_b64 ? (
        <img
          src={`data:image/jpeg;base64,${incident.thumbnail_b64}`}
          alt={`${classUpper} detection thumbnail`}
          style={styles.thumb}
        />
      ) : (
        <div style={styles.thumbMissing}>(no thumbnail)</div>
      )}
      <div style={styles.confRow}>
        <span style={styles.confLabel}>CONFIDENCE</span>
        <span style={styles.confValue}>{confPct}%</span>
      </div>
      <div style={styles.confBarTrack}>
        <div
          style={{
            ...styles.confBarFill,
            width: `${confPct}%`,
            backgroundColor: isThreatClass ? PALETTE.error : PALETTE.warning,
          }}
        />
      </div>
      <div style={styles.metaRow}>
        <span style={styles.metaKey}>SESSION</span>
        <span style={styles.metaVal}>{incident.session_id || '—'}</span>
      </div>
      <div style={styles.btnRow}>
        <button
          type="button"
          style={{ ...styles.btn, ...styles.btnSafe }}
          onClick={() => onClassify(incident, 'SAFE')}
        >
          SAFE
        </button>
        <button
          type="button"
          style={{ ...styles.btn, ...styles.btnThreat }}
          onClick={() => onClassify(incident, 'THREAT')}
        >
          THREAT
        </button>
        <button
          type="button"
          style={{ ...styles.btn, ...styles.btnDismiss }}
          onClick={() => onClassify(incident, 'DISMISSED')}
        >
          DISMISS
        </button>
      </div>
    </div>
  );
}

function HistoryRow({ entry }) {
  const dec = entry.decision || 'UNKNOWN';
  const color = dec === 'SAFE'
    ? PALETTE.success
    : dec.includes('THREAT')
      ? PALETTE.error
      : PALETTE.textMuted;
  const when = entry.decided_at
    ? new Date(entry.decided_at).toLocaleTimeString()
    : '--';
  const cls = (entry.class || 'unknown').toUpperCase();
  const lat = entry.latency_sec != null ? `${entry.latency_sec.toFixed(1)}s` : '—';
  return (
    <div style={styles.historyRow}>
      <span style={{ ...styles.historyDot, backgroundColor: color }} />
      <span style={styles.historyTime}>{when}</span>
      <span style={{ ...styles.historyDecision, color }}>{dec}</span>
      <span style={styles.historyClass}>{cls}</span>
      <span style={styles.historyLatency}>{lat}</span>
      <span style={styles.historyBy}>by {entry.decided_by || 'op'}</span>
    </div>
  );
}

const styles = {
  root: {
    padding: 24,
    color: PALETTE.text,
    fontFamily: 'Inter, system-ui, sans-serif',
    height: '100%',
    overflowY: 'auto',
    boxSizing: 'border-box',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: 16,
  },
  title: { fontSize: 13, fontWeight: 700, letterSpacing: 1.2, color: PALETTE.textBright },
  count: { fontSize: 11, color: PALETTE.textMuted },
  empty: {
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 6,
    padding: '40px 20px',
    textAlign: 'center',
  },
  emptyIcon: { fontSize: 28, color: PALETTE.textMuted, marginBottom: 12 },
  emptyTitle: { fontSize: 14, color: PALETTE.textBright, fontWeight: 600 },
  emptyHint: { fontSize: 11, color: PALETTE.textMuted, marginTop: 8, lineHeight: 1.5 },
  code: {
    background: PALETTE.surfaceLight,
    padding: '2px 6px',
    borderRadius: 3,
    fontFamily: 'Menlo, monospace',
    fontSize: 10,
    color: PALETTE.accent,
  },
  cardGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: 16,
    marginBottom: 24,
  },
  card: {
    background: PALETTE.surface,
    border: `2px solid ${PALETTE.error}`,
    borderRadius: 6,
    padding: 14,
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  classBadge: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 0.8,
    color: '#fff',
    padding: '3px 10px',
    borderRadius: 3,
  },
  cardAge: { fontSize: 11, color: PALETTE.textMuted },
  thumb: {
    width: '100%',
    height: 160,
    objectFit: 'contain',
    background: '#000',
    borderRadius: 4,
  },
  thumbMissing: {
    height: 160,
    background: PALETTE.surfaceLight,
    borderRadius: 4,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: PALETTE.textMuted,
    fontSize: 11,
  },
  confRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confLabel: { fontSize: 10, color: PALETTE.textMuted, letterSpacing: 0.8 },
  confValue: { fontSize: 18, fontWeight: 700, color: PALETTE.textBright },
  confBarTrack: {
    height: 6,
    background: PALETTE.surfaceLight,
    borderRadius: 3,
    overflow: 'hidden',
  },
  confBarFill: {
    height: '100%',
    transition: 'width 0.3s ease',
  },
  metaRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 10,
    color: PALETTE.textMuted,
  },
  metaKey: { letterSpacing: 0.6 },
  metaVal: { fontFamily: 'Menlo, monospace' },
  btnRow: {
    display: 'flex',
    gap: 8,
    marginTop: 4,
  },
  btn: {
    flex: 1,
    padding: '10px 0',
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: 0.8,
    border: 'none',
    borderRadius: 4,
    cursor: 'pointer',
    color: '#fff',
  },
  btnSafe: { background: PALETTE.success },
  btnThreat: { background: PALETTE.error },
  btnDismiss: { background: PALETTE.textMuted, color: PALETTE.textBright },
  historyHeader: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: PALETTE.textMuted,
    marginBottom: 10,
    paddingBottom: 6,
    borderBottom: `1px solid ${PALETTE.border}`,
  },
  historyEmpty: {
    fontSize: 11,
    color: PALETTE.textMuted,
    fontStyle: 'italic',
    padding: '6px 0',
  },
  historyList: { display: 'flex', flexDirection: 'column', gap: 6 },
  historyRow: {
    display: 'grid',
    gridTemplateColumns: '14px 80px 110px 1fr 60px 90px',
    gap: 10,
    alignItems: 'center',
    fontSize: 11,
    padding: '6px 8px',
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 3,
  },
  historyDot: { width: 8, height: 8, borderRadius: '50%' },
  historyTime: { color: PALETTE.textMuted, fontFamily: 'Menlo, monospace' },
  historyDecision: { fontWeight: 700, letterSpacing: 0.6 },
  historyClass: { color: PALETTE.text, letterSpacing: 0.4 },
  historyLatency: { color: PALETTE.textMuted, fontFamily: 'Menlo, monospace' },
  historyBy: { color: PALETTE.textMuted, fontStyle: 'italic' },
};
