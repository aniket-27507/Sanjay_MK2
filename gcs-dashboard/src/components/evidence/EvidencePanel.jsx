import React, { useState, useCallback } from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE } from '../../utils/colors';

export default function EvidencePanel({ wsSend }) {
  const drones = useGCSState((s) => s.drones);
  const droneOrder = useGCSState((s) => s.droneOrder);
  const addAuditEntry = useGCSState((s) => s.addAuditEntry);

  const [recordings, setRecordings] = useState([]);
  const [activeRecordings, setActiveRecordings] = useState(new Set());
  const [selectedDrone, setSelectedDrone] = useState('all');
  const [notes, setNotes] = useState('');

  const droneIds = droneOrder.length > 0 ? droneOrder : Object.keys(drones);

  const startRecording = useCallback(
    (droneId) => {
      const id = `rec-${Date.now()}-${droneId || 'all'}`;
      const entry = {
        id,
        droneId: droneId === 'all' ? null : droneId,
        startTime: Date.now(),
        stopTime: null,
        status: 'recording',
        notes: '',
      };

      setRecordings((prev) => [entry, ...prev]);
      setActiveRecordings((prev) => new Set(prev).add(id));

      addAuditEntry({
        action: 'RECORDING_STARTED',
        detail: `Started recording: ${droneId === 'all' ? 'All drones' : drones[droneId]?.name || droneId}`,
        source: 'operator',
      });

      if (wsSend) {
        wsSend({ type: 'start_recording', recording_id: id, drone_id: droneId === 'all' ? null : droneId });
      }
    },
    [drones, addAuditEntry, wsSend],
  );

  const stopRecording = useCallback(
    (recId) => {
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recId
            ? { ...r, stopTime: Date.now(), status: 'stopped' }
            : r,
        ),
      );
      setActiveRecordings((prev) => {
        const next = new Set(prev);
        next.delete(recId);
        return next;
      });

      addAuditEntry({
        action: 'RECORDING_STOPPED',
        detail: `Stopped recording: ${recId}`,
        source: 'operator',
      });

      if (wsSend) {
        wsSend({ type: 'stop_recording', recording_id: recId });
      }
    },
    [addAuditEntry, wsSend],
  );

  const addNote = useCallback(
    (recId) => {
      if (!notes.trim()) return;
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recId ? { ...r, notes: r.notes ? `${r.notes}\n${notes}` : notes } : r,
        ),
      );
      addAuditEntry({
        action: 'EVIDENCE_NOTE',
        detail: `Note added to ${recId}: ${notes.trim()}`,
        source: 'operator',
      });
      setNotes('');
    },
    [notes, addAuditEntry],
  );

  const markEvidence = useCallback(
    (recId) => {
      setRecordings((prev) =>
        prev.map((r) =>
          r.id === recId ? { ...r, status: 'evidence' } : r,
        ),
      );
      addAuditEntry({
        level: 'warning',
        action: 'EVIDENCE_FLAGGED',
        detail: `Recording ${recId} flagged as evidence`,
        source: 'operator',
      });
      if (wsSend) {
        wsSend({ type: 'flag_evidence', recording_id: recId });
      }
    },
    [addAuditEntry, wsSend],
  );

  const isRecording = activeRecordings.size > 0;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>EVIDENCE MANAGEMENT</span>
        {isRecording && (
          <span style={styles.recIndicator}>
            <span style={styles.recDot} />
            {activeRecordings.size} ACTIVE
          </span>
        )}
      </div>

      {/* Controls */}
      <div style={styles.controls}>
        <div style={styles.controlRow}>
          <label style={styles.controlLabel}>Target:</label>
          <select
            style={styles.select}
            value={selectedDrone}
            onChange={(e) => setSelectedDrone(e.target.value)}
          >
            <option value="all">All Drones</option>
            {droneIds.map((id) => (
              <option key={id} value={id}>
                {drones[id]?.name || id}
              </option>
            ))}
          </select>
          <button
            style={styles.recordBtn}
            onClick={() => startRecording(selectedDrone)}
          >
            START RECORDING
          </button>
        </div>
      </div>

      {/* Active recordings */}
      {activeRecordings.size > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>ACTIVE RECORDINGS</div>
          {recordings
            .filter((r) => activeRecordings.has(r.id))
            .map((rec) => (
              <div key={rec.id} style={styles.activeCard}>
                <div style={styles.activeHeader}>
                  <span style={styles.recBadge}>
                    <span style={styles.recBadgeDot} />
                    RECORDING
                  </span>
                  <span style={styles.recTarget}>
                    {rec.droneId ? drones[rec.droneId]?.name || rec.droneId : 'All Drones'}
                  </span>
                  <span style={styles.recDuration}>
                    {formatDuration(Date.now() - rec.startTime)}
                  </span>
                </div>
                <div style={styles.activeActions}>
                  <button
                    style={styles.stopBtn}
                    onClick={() => stopRecording(rec.id)}
                  >
                    STOP
                  </button>
                  <input
                    style={styles.noteInput}
                    placeholder="Add note..."
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addNote(rec.id)}
                  />
                  <button
                    style={styles.noteBtn}
                    onClick={() => addNote(rec.id)}
                  >
                    + NOTE
                  </button>
                </div>
              </div>
            ))}
        </div>
      )}

      {/* Recording history */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          RECORDING HISTORY ({recordings.filter((r) => !activeRecordings.has(r.id)).length})
        </div>
        <div style={styles.historyList}>
          {recordings.filter((r) => !activeRecordings.has(r.id)).length === 0 && (
            <div style={styles.emptyHistory}>
              No completed recordings yet. Start a recording to begin capturing evidence.
            </div>
          )}
          {recordings
            .filter((r) => !activeRecordings.has(r.id))
            .map((rec) => (
              <div key={rec.id} style={styles.historyCard}>
                <div style={styles.historyHeader}>
                  <span
                    style={{
                      ...styles.statusBadge,
                      background:
                        rec.status === 'evidence'
                          ? 'rgba(234,179,8,0.15)'
                          : 'rgba(100,116,139,0.15)',
                      color: rec.status === 'evidence' ? '#eab308' : PALETTE.textMuted,
                      borderColor:
                        rec.status === 'evidence'
                          ? 'rgba(234,179,8,0.3)'
                          : PALETTE.borderLight,
                    }}
                  >
                    {rec.status === 'evidence' ? 'EVIDENCE' : 'ARCHIVED'}
                  </span>
                  <span style={styles.historyTarget}>
                    {rec.droneId ? drones[rec.droneId]?.name || rec.droneId : 'All'}
                  </span>
                  <span style={styles.historyTime}>
                    {new Date(rec.startTime).toLocaleTimeString()}
                    {rec.stopTime && ` — ${new Date(rec.stopTime).toLocaleTimeString()}`}
                  </span>
                </div>
                {rec.stopTime && (
                  <div style={styles.historyDuration}>
                    Duration: {formatDuration(rec.stopTime - rec.startTime)}
                  </div>
                )}
                {rec.notes && <div style={styles.historyNotes}>{rec.notes}</div>}
                <div style={styles.historyActions}>
                  {rec.status !== 'evidence' && (
                    <button
                      style={styles.evidenceBtn}
                      onClick={() => markEvidence(rec.id)}
                    >
                      FLAG AS EVIDENCE
                    </button>
                  )}
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

function formatDuration(ms) {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
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
  },
  title: { fontSize: 13, fontWeight: 700, letterSpacing: 1.2, color: PALETTE.textBright },
  recIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 11,
    fontWeight: 700,
    color: '#ef4444',
  },
  recDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    backgroundColor: '#ef4444',
  },
  controls: {
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 8,
    padding: 12,
    flexShrink: 0,
  },
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    flexWrap: 'wrap',
  },
  controlLabel: { fontSize: 11, color: PALETTE.textMuted, fontWeight: 600 },
  select: {
    fontSize: 12,
    padding: '5px 10px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.border}`,
    background: PALETTE.surfaceLight,
    color: PALETTE.textBright,
    outline: 'none',
  },
  recordBtn: {
    fontSize: 11,
    fontWeight: 700,
    padding: '6px 16px',
    borderRadius: 5,
    border: 'none',
    background: '#dc2626',
    color: '#fff',
    cursor: 'pointer',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.5,
    color: PALETTE.textMuted,
  },
  activeCard: {
    background: PALETTE.surface,
    border: `1px solid rgba(239,68,68,0.3)`,
    borderRadius: 8,
    padding: 12,
  },
  activeHeader: { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 },
  recBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 9,
    fontWeight: 800,
    color: '#ef4444',
    letterSpacing: 1,
  },
  recBadgeDot: { width: 6, height: 6, borderRadius: '50%', backgroundColor: '#ef4444' },
  recTarget: { flex: 1, fontSize: 12, fontWeight: 600, color: PALETTE.textBright },
  recDuration: { fontSize: 12, color: PALETTE.textMuted, fontVariantNumeric: 'tabular-nums' },
  activeActions: { display: 'flex', gap: 8, alignItems: 'center' },
  stopBtn: {
    fontSize: 10,
    fontWeight: 700,
    padding: '5px 12px',
    borderRadius: 4,
    border: `1px solid rgba(239,68,68,0.4)`,
    background: 'rgba(239,68,68,0.15)',
    color: '#ef4444',
    cursor: 'pointer',
  },
  noteInput: {
    flex: 1,
    fontSize: 12,
    padding: '5px 10px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.border}`,
    background: PALETTE.surfaceLight,
    color: PALETTE.textBright,
    outline: 'none',
  },
  noteBtn: {
    fontSize: 10,
    fontWeight: 600,
    padding: '5px 10px',
    borderRadius: 4,
    border: `1px solid ${PALETTE.borderLight}`,
    background: 'transparent',
    color: PALETTE.text,
    cursor: 'pointer',
  },
  historyList: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  emptyHistory: { fontSize: 12, color: PALETTE.textMuted, padding: '12px 0' },
  historyCard: {
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 8,
    padding: 10,
  },
  historyHeader: { display: 'flex', alignItems: 'center', gap: 8 },
  statusBadge: {
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: 0.8,
    padding: '2px 8px',
    borderRadius: 4,
    border: '1px solid',
  },
  historyTarget: { flex: 1, fontSize: 12, fontWeight: 600, color: PALETTE.textBright },
  historyTime: { fontSize: 10, color: PALETTE.textMuted },
  historyDuration: { fontSize: 11, color: PALETTE.textMuted, marginTop: 4 },
  historyNotes: {
    fontSize: 11,
    color: PALETTE.text,
    marginTop: 6,
    padding: '6px 8px',
    background: PALETTE.surfaceLight,
    borderRadius: 4,
    whiteSpace: 'pre-wrap',
  },
  historyActions: { marginTop: 6 },
  evidenceBtn: {
    fontSize: 10,
    fontWeight: 700,
    padding: '4px 12px',
    borderRadius: 4,
    border: '1px solid rgba(234,179,8,0.4)',
    background: 'rgba(234,179,8,0.1)',
    color: '#eab308',
    cursor: 'pointer',
  },
};
