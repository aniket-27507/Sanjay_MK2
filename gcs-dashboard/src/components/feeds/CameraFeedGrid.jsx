import React from 'react';
import useGCSState from '../../hooks/useGCSState';
import { PALETTE, DRONE_STATE_COLORS } from '../../utils/colors';

const GRID_SLOTS = 8; // 2 rows x 4 columns

export default function CameraFeedGrid() {
  const drones = useGCSState((s) => s.drones);
  const droneOrder = useGCSState((s) => s.droneOrder);
  const cameraFrames = useGCSState((s) => s.cameraFrames);
  const telemetry = useGCSState((s) => s.telemetry);

  const ids = droneOrder.length > 0 ? droneOrder : Object.keys(drones);
  const slots = [];
  for (let i = 0; i < GRID_SLOTS; i++) {
    const id = ids[i] || null;
    slots.push(id);
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>CAMERA FEEDS</span>
        <span style={styles.subtitle}>
          {ids.length} drone{ids.length !== 1 ? 's' : ''} available
        </span>
      </div>
      <div style={styles.grid}>
        {slots.map((droneId, idx) => (
          <CameraSlot
            key={idx}
            slotIndex={idx}
            droneId={droneId}
            drone={droneId ? drones[droneId] : null}
            frame={droneId ? cameraFrames[droneId] : null}
            telem={droneId ? telemetry[droneId] : null}
          />
        ))}
      </div>
    </div>
  );
}

function CameraSlot({ slotIndex, droneId, drone, frame, telem }) {
  const name = drone?.name || droneId || `Slot ${slotIndex + 1}`;
  const state = drone?.state || drone?.status || 'offline';
  const stateColor = DRONE_STATE_COLORS[state] || DRONE_STATE_COLORS.offline;

  return (
    <div style={styles.slot}>
      {/* HUD overlay */}
      <div style={styles.hud}>
        <div style={styles.hudTop}>
          <span style={{ ...styles.hudDot, backgroundColor: stateColor }} />
          <span style={styles.hudName}>{name}</span>
          {telem?.battery != null && (
            <span style={styles.hudBat}>
              {Math.round(telem.battery)}%
            </span>
          )}
        </div>
        {telem && (
          <div style={styles.hudBottom}>
            {telem.altitude != null && (
              <span style={styles.hudStat}>ALT {telem.altitude.toFixed(1)}m</span>
            )}
            {telem.speed != null && (
              <span style={styles.hudStat}>SPD {telem.speed.toFixed(1)}m/s</span>
            )}
          </div>
        )}
      </div>

      {/* Feed content */}
      {frame ? (
        <img
          src={frame}
          alt={`Feed ${name}`}
          style={styles.feedImg}
          onError={(e) => {
            e.target.style.display = 'none';
          }}
        />
      ) : (
        <div style={styles.noFeed}>
          {droneId ? (
            <>
              <div style={styles.noFeedIcon}>{'\u25A3'}</div>
              <div style={styles.noFeedText}>Awaiting feed</div>
              <div style={styles.noFeedHint}>{name}</div>
            </>
          ) : (
            <>
              <div style={styles.noFeedIcon}>{'\u2014'}</div>
              <div style={styles.noFeedText}>Empty slot</div>
            </>
          )}
        </div>
      )}

      {/* REC indicator */}
      {droneId && frame && (
        <div style={styles.recBadge}>
          <span style={styles.recDot} />
          REC
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
    padding: 12,
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
  subtitle: {
    fontSize: 11,
    color: PALETTE.textMuted,
  },
  grid: {
    flex: 1,
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gridTemplateRows: 'repeat(2, 1fr)',
    gap: 8,
    minHeight: 0,
  },
  slot: {
    position: 'relative',
    background: '#0b1120',
    border: `1px solid ${PALETTE.border}`,
    borderRadius: 8,
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  hud: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    padding: '6px 10px',
    background: 'linear-gradient(180deg, rgba(0,0,0,0.7) 0%, transparent 100%)',
    zIndex: 2,
  },
  hudTop: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  hudDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
  },
  hudName: {
    fontSize: 11,
    fontWeight: 600,
    color: '#e2e8f0',
    flex: 1,
  },
  hudBat: {
    fontSize: 10,
    fontWeight: 600,
    color: PALETTE.textMuted,
  },
  hudBottom: {
    display: 'flex',
    gap: 10,
    marginTop: 2,
  },
  hudStat: {
    fontSize: 9,
    color: PALETTE.textMuted,
    letterSpacing: 0.5,
  },
  feedImg: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  noFeed: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
  },
  noFeedIcon: { fontSize: 28, color: PALETTE.borderLight },
  noFeedText: { fontSize: 11, color: PALETTE.textMuted },
  noFeedHint: { fontSize: 10, color: PALETTE.borderLight },
  recBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    display: 'flex',
    alignItems: 'center',
    gap: 4,
    fontSize: 9,
    fontWeight: 700,
    color: '#ef4444',
    background: 'rgba(0,0,0,0.6)',
    padding: '2px 6px',
    borderRadius: 4,
    zIndex: 3,
  },
  recDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    backgroundColor: '#ef4444',
    animation: 'none', /* CSS animation handled in App.css */
  },
};
