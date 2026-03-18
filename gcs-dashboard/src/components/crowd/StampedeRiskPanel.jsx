import React from 'react';
import {
  RadialBarChart,
  RadialBar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import useGCSState from '../../hooks/useGCSState';
import { riskLevelColor, riskLevelLabel, PALETTE, RISK_COLORS } from '../../utils/colors';

export default function StampedeRiskPanel() {
  const stampedeRisks = useGCSState((s) => s.stampedeRisks);
  const crowdZones = useGCSState((s) => s.crowdZones);

  const zoneIds = Object.keys(stampedeRisks);
  const hasData = zoneIds.length > 0;

  // Build bar-chart data
  const barData = zoneIds.map((zid) => {
    const r = stampedeRisks[zid];
    return {
      zone: r.label || zid,
      risk: Math.round((r.risk || 0) * 100),
      fill: riskLevelColor(r.risk || 0),
    };
  });

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>STAMPEDE RISK ANALYSIS</span>
        <span style={styles.subtitle}>
          {hasData ? `${zoneIds.length} zone${zoneIds.length !== 1 ? 's' : ''} monitored` : 'No zone data'}
        </span>
      </div>

      {!hasData ? (
        <div style={styles.empty}>
          <div style={styles.emptyIcon}>{'\u2630'}</div>
          <div style={styles.emptyText}>Awaiting stampede_risk data...</div>
          <div style={styles.emptyHint}>
            Zone risk levels will appear here once the backend sends stampede_risk messages
          </div>
        </div>
      ) : (
        <div style={styles.body}>
          {/* Gauge cards row */}
          <div style={styles.gaugeRow}>
            {zoneIds.map((zid) => {
              const r = stampedeRisks[zid];
              const risk = r.risk || 0;
              const pct = Math.round(risk * 100);
              const col = riskLevelColor(risk);
              const label = riskLevelLabel(risk);

              const gaugeData = [{ value: pct, fill: col }];

              return (
                <div key={zid} style={styles.gaugeCard}>
                  <div style={styles.gaugeTitle}>{r.label || zid}</div>
                  <div style={styles.gaugeWrap}>
                    <ResponsiveContainer width="100%" height={120}>
                      <RadialBarChart
                        innerRadius="65%"
                        outerRadius="100%"
                        startAngle={180}
                        endAngle={0}
                        data={gaugeData}
                        cx="50%"
                        cy="85%"
                      >
                        <RadialBar
                          dataKey="value"
                          background={{ fill: PALETTE.surfaceLight }}
                          cornerRadius={6}
                        />
                      </RadialBarChart>
                    </ResponsiveContainer>
                    <div style={styles.gaugeCenterText}>
                      <span style={{ ...styles.gaugePct, color: col }}>{pct}%</span>
                      <span style={{ ...styles.gaugeLabel, color: col }}>{label}</span>
                    </div>
                  </div>

                  {/* Indicators */}
                  {r.indicators && Object.keys(r.indicators).length > 0 && (
                    <div style={styles.indicators}>
                      {Object.entries(r.indicators).map(([key, val]) => (
                        <div key={key} style={styles.indicatorRow}>
                          <span style={styles.indicatorKey}>{formatIndicator(key)}</span>
                          <span style={styles.indicatorVal}>
                            {typeof val === 'number' ? val.toFixed(2) : String(val)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Bar chart overview */}
          {barData.length > 1 && (
            <div style={styles.chartSection}>
              <div style={styles.chartTitle}>RISK COMPARISON</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={barData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={PALETTE.border} />
                  <XAxis dataKey="zone" tick={{ fill: PALETTE.textMuted, fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: PALETTE.textMuted, fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: PALETTE.surface,
                      border: `1px solid ${PALETTE.border}`,
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                    labelStyle={{ color: PALETTE.textBright }}
                    itemStyle={{ color: PALETTE.text }}
                  />
                  <Bar dataKey="risk" radius={[4, 4, 0, 0]}>
                    {barData.map((d, i) => (
                      <rect key={i} fill={d.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Legend */}
          <div style={styles.legend}>
            {Object.entries(RISK_COLORS).map(([level, color]) => (
              <div key={level} style={styles.legendItem}>
                <span style={{ ...styles.legendDot, backgroundColor: color }} />
                <span style={styles.legendText}>{level.toUpperCase()}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatIndicator(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

const styles = {
  container: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'auto',
    padding: 16,
    gap: 12,
  },
  header: {
    display: 'flex',
    alignItems: 'baseline',
    gap: 12,
    flexShrink: 0,
  },
  title: {
    fontSize: 13,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: PALETTE.textBright,
  },
  subtitle: { fontSize: 11, color: PALETTE.textMuted },
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
  body: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
    overflow: 'auto',
  },
  gaugeRow: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
  },
  gaugeCard: {
    flex: '1 1 200px',
    maxWidth: 280,
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 10,
    padding: 14,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  gaugeTitle: {
    fontSize: 12,
    fontWeight: 700,
    color: PALETTE.textBright,
    marginBottom: 4,
  },
  gaugeWrap: {
    position: 'relative',
    width: '100%',
    minHeight: 120,
  },
  gaugeCenterText: {
    position: 'absolute',
    bottom: 8,
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  gaugePct: { fontSize: 22, fontWeight: 800 },
  gaugeLabel: { fontSize: 9, fontWeight: 700, letterSpacing: 1.2 },
  indicators: {
    width: '100%',
    marginTop: 8,
    borderTop: `1px solid ${PALETTE.border}`,
    paddingTop: 8,
    display: 'flex',
    flexDirection: 'column',
    gap: 3,
  },
  indicatorRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 11,
  },
  indicatorKey: { color: PALETTE.textMuted },
  indicatorVal: { color: PALETTE.text, fontWeight: 600, fontVariantNumeric: 'tabular-nums' },
  chartSection: {
    background: PALETTE.surface,
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 10,
    padding: 14,
  },
  chartTitle: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: PALETTE.textMuted,
    marginBottom: 8,
  },
  legend: {
    display: 'flex',
    gap: 16,
    justifyContent: 'center',
    flexShrink: 0,
  },
  legendItem: { display: 'flex', alignItems: 'center', gap: 5 },
  legendDot: { width: 10, height: 10, borderRadius: 3 },
  legendText: { fontSize: 10, fontWeight: 600, color: PALETTE.textMuted, letterSpacing: 0.8 },
};
