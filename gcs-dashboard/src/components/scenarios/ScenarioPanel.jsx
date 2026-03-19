import React from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE } from '../../utils/colors';

const CATEGORY_COLORS = {
  high_rise: '#3b82f6',
  crowd: '#f59e0b',
  stampede: '#ef4444',
  armed: '#dc2626',
  vehicle: '#6366f1',
  false_alarm: '#6b7280',
  degraded: '#f97316',
  multi: '#8b5cf6',
  edge: '#14b8a6',
  stress: '#e11d48',
  baseline: '#22c55e',
};

const STATUS_LABELS = {
  idle: { text: 'IDLE', color: PALETTE.textMuted },
  running: { text: 'RUNNING', color: '#22c55e' },
  completed: { text: 'COMPLETE', color: '#3b82f6' },
  failed: { text: 'FAILED', color: '#ef4444' },
};

export default function ScenarioPanel({ wsSend }) {
  const scenario = useGCSState((s) => s.scenario);
  const threats = useGCSState((s) => s.threats);
  const auditLog = useGCSState((s) => s.auditLog);

  const status = STATUS_LABELS[scenario?.status] || STATUS_LABELS.idle;
  const progress = scenario?.duration_sec
    ? Math.min(100, ((scenario.elapsed_sec || 0) / scenario.duration_sec) * 100)
    : 0;

  const threatCount = threats?.length || 0;
  const recentEvents = (auditLog || []).slice(-8);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Scenario Control</h2>
        <div style={{ ...styles.statusBadge, background: status.color + '22', color: status.color }}>
          {status.text}
        </div>
      </div>

      {/* Active Scenario */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>ACTIVE SCENARIO</div>
        {scenario?.scenario_id ? (
          <>
            <div style={styles.scenarioName}>
              <span style={{
                ...styles.categoryTag,
                background: (CATEGORY_COLORS[scenario.category] || '#6b7280') + '22',
                color: CATEGORY_COLORS[scenario.category] || '#6b7280',
              }}>
                {scenario.category}
              </span>
              {scenario.scenario_id}: {scenario.scenario_name}
            </div>
            <div style={styles.progressContainer}>
              <div style={styles.progressBar}>
                <div style={{ ...styles.progressFill, width: `${progress}%` }} />
              </div>
              <span style={styles.progressText}>
                {Math.round(scenario.elapsed_sec || 0)}s / {scenario.duration_sec}s
              </span>
            </div>
          </>
        ) : (
          <div style={styles.noScenario}>
            No scenario running. Start one with:<br />
            <code style={styles.code}>python scripts/run_scenario.py --scenario S01 --realtime</code>
          </div>
        )}
      </div>

      {/* Metrics */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>LIVE METRICS</div>
        <div style={styles.metricsGrid}>
          <MetricBox label="Active Threats" value={threatCount} color="#ef4444" />
          <MetricBox label="Drones Active" value={scenario?.drones_active || '--'} color="#22c55e" />
          <MetricBox label="Coverage" value={scenario?.coverage_pct ? `${scenario.coverage_pct.toFixed(1)}%` : '--'} color="#3b82f6" />
          <MetricBox label="FP Rate" value={scenario?.fp_rate != null ? `${(scenario.fp_rate * 100).toFixed(1)}%` : '--'} color="#f59e0b" />
        </div>
      </div>

      {/* Recent Events */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>RECENT EVENTS</div>
        <div style={styles.eventList}>
          {recentEvents.length === 0 && (
            <div style={styles.noEvents}>No events yet</div>
          )}
          {recentEvents.map((ev, i) => (
            <div key={i} style={styles.eventRow}>
              <span style={styles.eventTime}>
                {typeof ev.ts === 'number' ? `${ev.ts.toFixed(1)}s` : '--'}
              </span>
              <span style={styles.eventType}>{ev.event || ev.type || '--'}</span>
              <span style={styles.eventDetail}>{ev.detail || ''}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Category Legend */}
      <div style={styles.card}>
        <div style={styles.cardTitle}>CATEGORIES</div>
        <div style={styles.legendGrid}>
          {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
            <div key={cat} style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: color }} />
              <span style={styles.legendLabel}>{cat}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function MetricBox({ label, value, color }) {
  return (
    <div style={styles.metricBox}>
      <div style={styles.metricLabel}>{label}</div>
      <div style={{ ...styles.metricValue, color }}>{value}</div>
    </div>
  );
}

const styles = {
  container: { padding: 20, display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 900, overflowY: 'auto', height: '100%' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
  title: { fontSize: 20, fontWeight: 700, color: PALETTE.textBright, margin: 0 },
  statusBadge: { padding: '4px 12px', borderRadius: 12, fontSize: 11, fontWeight: 700, letterSpacing: 1 },
  card: { background: PALETTE.surface, border: `1px solid ${PALETTE.border}`, borderRadius: 10, padding: 16 },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.5, color: PALETTE.textMuted, marginBottom: 12 },
  scenarioName: { fontSize: 15, fontWeight: 600, color: PALETTE.textBright, display: 'flex', alignItems: 'center', gap: 10 },
  categoryTag: { padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase' },
  progressContainer: { display: 'flex', alignItems: 'center', gap: 12, marginTop: 12 },
  progressBar: { flex: 1, height: 6, background: PALETTE.border, borderRadius: 3, overflow: 'hidden' },
  progressFill: { height: '100%', background: 'linear-gradient(90deg, #3b82f6, #22c55e)', borderRadius: 3, transition: 'width 0.3s' },
  progressText: { fontSize: 12, color: PALETTE.textMuted, flexShrink: 0 },
  noScenario: { color: PALETTE.textMuted, fontSize: 13, lineHeight: 1.6 },
  code: { background: PALETTE.border, padding: '2px 6px', borderRadius: 4, fontSize: 12, fontFamily: 'monospace' },
  metricsGrid: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 },
  metricBox: { textAlign: 'center' },
  metricLabel: { fontSize: 10, color: PALETTE.textMuted, marginBottom: 4 },
  metricValue: { fontSize: 22, fontWeight: 700 },
  eventList: { display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 200, overflowY: 'auto' },
  noEvents: { color: PALETTE.textMuted, fontSize: 12 },
  eventRow: { display: 'flex', gap: 8, fontSize: 12, padding: '3px 0', borderBottom: `1px solid ${PALETTE.border}` },
  eventTime: { color: PALETTE.textMuted, width: 50, flexShrink: 0, fontFamily: 'monospace' },
  eventType: { color: '#93bbfd', width: 120, flexShrink: 0, fontWeight: 600 },
  eventDetail: { color: PALETTE.text, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  legendGrid: { display: 'flex', flexWrap: 'wrap', gap: 8 },
  legendItem: { display: 'flex', alignItems: 'center', gap: 4 },
  legendDot: { width: 8, height: 8, borderRadius: '50%' },
  legendLabel: { fontSize: 11, color: PALETTE.text, textTransform: 'capitalize' },
};
