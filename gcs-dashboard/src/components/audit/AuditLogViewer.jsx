import React, { useState, useMemo } from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE } from '../../utils/colors';

const LEVEL_COLORS = {
  info: '#3b82f6',
  warning: '#eab308',
  error: '#ef4444',
  critical: '#ef4444',
  debug: '#64748b',
};

const PAGE_SIZE = 50;

export default function AuditLogViewer() {
  const auditLog = useGCSState((s) => s.auditLog);

  const [filterLevel, setFilterLevel] = useState('all');
  const [filterSource, setFilterSource] = useState('all');
  const [searchText, setSearchText] = useState('');
  const [page, setPage] = useState(0);

  /* Unique sources */
  const sources = useMemo(() => {
    const set = new Set(auditLog.map((e) => e.source));
    return ['all', ...Array.from(set).sort()];
  }, [auditLog]);

  /* Filtered log */
  const filtered = useMemo(() => {
    let result = auditLog;
    if (filterLevel !== 'all') {
      result = result.filter((e) => e.level === filterLevel);
    }
    if (filterSource !== 'all') {
      result = result.filter((e) => e.source === filterSource);
    }
    if (searchText.trim()) {
      const q = searchText.toLowerCase();
      result = result.filter(
        (e) =>
          (e.action || '').toLowerCase().includes(q) ||
          (e.detail || '').toLowerCase().includes(q) ||
          (e.user || '').toLowerCase().includes(q),
      );
    }
    return result;
  }, [auditLog, filterLevel, filterSource, searchText]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageEntries = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>AUDIT LOG</span>
        <span style={styles.count}>{filtered.length} entries</span>
      </div>

      {/* Filters */}
      <div style={styles.filters}>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Level:</label>
          <select
            style={styles.select}
            value={filterLevel}
            onChange={(e) => { setFilterLevel(e.target.value); setPage(0); }}
          >
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
            <option value="debug">Debug</option>
          </select>
        </div>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Source:</label>
          <select
            style={styles.select}
            value={filterSource}
            onChange={(e) => { setFilterSource(e.target.value); setPage(0); }}
          >
            {sources.map((s) => (
              <option key={s} value={s}>
                {s === 'all' ? 'All' : s}
              </option>
            ))}
          </select>
        </div>
        <div style={{ ...styles.filterGroup, flex: 1 }}>
          <label style={styles.filterLabel}>Search:</label>
          <input
            style={styles.searchInput}
            placeholder="Filter by action or detail..."
            value={searchText}
            onChange={(e) => { setSearchText(e.target.value); setPage(0); }}
          />
        </div>
      </div>

      {/* Table */}
      <div style={styles.tableWrap}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={{ ...styles.th, width: 90 }}>TIME</th>
              <th style={{ ...styles.th, width: 70 }}>LEVEL</th>
              <th style={{ ...styles.th, width: 90 }}>SOURCE</th>
              <th style={{ ...styles.th, width: 80 }}>USER</th>
              <th style={{ ...styles.th, width: 160 }}>ACTION</th>
              <th style={styles.th}>DETAIL</th>
            </tr>
          </thead>
          <tbody>
            {pageEntries.length === 0 && (
              <tr>
                <td colSpan={6} style={styles.emptyTd}>
                  {auditLog.length === 0
                    ? 'No audit entries yet. Actions and events will be logged here.'
                    : 'No entries match your filters.'}
                </td>
              </tr>
            )}
            {pageEntries.map((entry) => {
              const ts =
                typeof entry.timestamp === 'number'
                  ? new Date(entry.timestamp).toLocaleTimeString()
                  : entry.timestamp || '--';
              const levelColor = LEVEL_COLORS[entry.level] || LEVEL_COLORS.info;

              return (
                <tr key={entry.id} style={styles.tr}>
                  <td style={styles.td}>
                    <span style={styles.timeCell}>{ts}</span>
                  </td>
                  <td style={styles.td}>
                    <span
                      style={{
                        ...styles.levelBadge,
                        backgroundColor: `${levelColor}20`,
                        color: levelColor,
                        borderColor: `${levelColor}40`,
                      }}
                    >
                      {(entry.level || 'info').toUpperCase()}
                    </span>
                  </td>
                  <td style={styles.td}>
                    <span style={styles.sourceCell}>{entry.source}</span>
                  </td>
                  <td style={styles.td}>
                    <span style={styles.userCell}>{entry.user || '--'}</span>
                  </td>
                  <td style={styles.td}>
                    <span style={styles.actionCell}>{entry.action}</span>
                  </td>
                  <td style={styles.td}>
                    <span style={styles.detailCell}>{entry.detail}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={styles.pagination}>
          <button
            style={styles.pageBtn}
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            PREV
          </button>
          <span style={styles.pageInfo}>
            Page {page + 1} of {totalPages}
          </span>
          <button
            style={styles.pageBtn}
            disabled={page >= totalPages - 1}
            onClick={() => setPage(page + 1)}
          >
            NEXT
          </button>
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
    alignItems: 'baseline',
    gap: 12,
    flexShrink: 0,
  },
  title: { fontSize: 13, fontWeight: 700, letterSpacing: 1.2, color: PALETTE.textBright },
  count: { fontSize: 11, color: PALETTE.textMuted },
  filters: {
    display: 'flex',
    gap: 12,
    flexWrap: 'wrap',
    flexShrink: 0,
  },
  filterGroup: { display: 'flex', alignItems: 'center', gap: 6 },
  filterLabel: { fontSize: 11, fontWeight: 600, color: PALETTE.textMuted },
  select: {
    fontSize: 12,
    padding: '4px 8px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.border}`,
    background: PALETTE.surfaceLight,
    color: PALETTE.textBright,
    outline: 'none',
  },
  searchInput: {
    flex: 1,
    minWidth: 150,
    fontSize: 12,
    padding: '5px 10px',
    borderRadius: 5,
    border: `1px solid ${PALETTE.border}`,
    background: PALETTE.surfaceLight,
    color: PALETTE.textBright,
    outline: 'none',
  },
  tableWrap: {
    flex: 1,
    overflow: 'auto',
    borderRadius: 8,
    border: `1px solid ${PALETTE.border}`,
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 12,
  },
  th: {
    position: 'sticky',
    top: 0,
    background: PALETTE.surface,
    padding: '8px 10px',
    textAlign: 'left',
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1,
    color: PALETTE.textMuted,
    borderBottom: `1px solid ${PALETTE.border}`,
    whiteSpace: 'nowrap',
  },
  tr: {
    borderBottom: `1px solid ${PALETTE.border}08`,
  },
  td: {
    padding: '6px 10px',
    verticalAlign: 'top',
  },
  emptyTd: {
    padding: '30px 10px',
    textAlign: 'center',
    color: PALETTE.textMuted,
    fontSize: 13,
  },
  timeCell: {
    fontVariantNumeric: 'tabular-nums',
    color: PALETTE.textMuted,
    fontSize: 11,
  },
  levelBadge: {
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: 0.8,
    padding: '2px 6px',
    borderRadius: 3,
    border: '1px solid',
    whiteSpace: 'nowrap',
  },
  sourceCell: { color: PALETTE.text, fontSize: 11 },
  userCell: { color: PALETTE.textMuted, fontSize: 11 },
  actionCell: { color: PALETTE.textBright, fontWeight: 600, fontSize: 11 },
  detailCell: {
    color: PALETTE.text,
    fontSize: 11,
    maxWidth: 400,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    display: 'inline-block',
  },
  pagination: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    flexShrink: 0,
  },
  pageBtn: {
    fontSize: 10,
    fontWeight: 700,
    padding: '5px 12px',
    borderRadius: 4,
    border: `1px solid ${PALETTE.borderLight}`,
    background: 'transparent',
    color: PALETTE.text,
    cursor: 'pointer',
  },
  pageInfo: { fontSize: 11, color: PALETTE.textMuted },
};
