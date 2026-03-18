import { create } from 'zustand';

/**
 * Centralised Zustand store for all GCS state.
 *
 * Every WebSocket message is routed through `dispatch(msg)`.
 * Components select the slices they need, e.g.
 *   const drones = useGCSState(s => s.drones);
 */
const useGCSState = create((set, get) => ({
  /* ── Connection ── */
  wsConnected: false,
  lastMessageAt: null,

  /* ── Fleet ── */
  drones: {},          // keyed by drone_id
  droneOrder: [],      // sorted list of drone ids

  /* ── Threats ── */
  threats: [],         // array of active threat objects
  threatHistory: [],   // last 200 events

  /* ── Telemetry ── */
  telemetry: {},       // keyed by drone_id → { battery, altitude, speed, ... }

  /* ── Map ── */
  mapPositions: {},    // keyed by drone_id → { x, y, z, heading }
  mapThreats: [],      // threat markers on map

  /* ── Crowd / Stampede ── */
  crowdGrid: null,     // { cols, rows, cell_size, data: Float32Array-like }
  crowdZones: [],      // named zone definitions with density
  stampedeRisks: {},   // keyed by zone_id → { risk, indicators }

  /* ── Camera ── */
  cameraFrames: {},    // keyed by drone_id → frame URL / data-URI

  /* ── Zones ── */
  zones: [],           // operational zone polygons

  /* ── Audit log ── */
  auditLog: [],        // last 500 entries

  /* ── Config / Mission ── */
  missionName: 'Sanjay MK2 Operations',
  config: {},

  /* ── Alerts banner ── */
  activeAlerts: [],    // unacknowledged alerts

  /* ────────────────────────────────────────────────
   *  dispatch — single entry-point for WS messages
   * ──────────────────────────────────────────────── */
  dispatch: (msg) => {
    const now = Date.now();

    switch (msg.type) {
      /* ── ws lifecycle ── */
      case 'ws_connected':
        set({ wsConnected: true });
        break;

      case 'ws_disconnected':
        set({ wsConnected: false });
        break;

      /* ── full state snapshot ── */
      case 'state': {
        const drones = {};
        const droneOrder = [];
        const telemetry = {};
        const mapPositions = {};

        (msg.drones || []).forEach((d) => {
          drones[d.id] = d;
          droneOrder.push(d.id);
          if (d.telemetry) telemetry[d.id] = d.telemetry;
          if (d.position) mapPositions[d.id] = d.position;
        });

        set({
          drones,
          droneOrder,
          telemetry: { ...get().telemetry, ...telemetry },
          mapPositions: { ...get().mapPositions, ...mapPositions },
          threats: msg.threats || get().threats,
          mapThreats: msg.threats || get().mapThreats,
          config: msg.config || get().config,
          missionName: msg.mission_name || get().missionName,
          zones: msg.zones || get().zones,
          lastMessageAt: now,
        });
        break;
      }

      /* ── map update (positions + threats) ── */
      case 'map_update': {
        const mapPositions = { ...get().mapPositions };
        (msg.drones || []).forEach((d) => {
          mapPositions[d.id] = { x: d.x, y: d.y, z: d.z, heading: d.heading };
        });
        set({
          mapPositions,
          mapThreats: msg.threats || get().mapThreats,
          lastMessageAt: now,
        });
        break;
      }

      /* ── per-drone telemetry ── */
      case 'telemetry': {
        const id = msg.drone_id;
        if (!id) break;
        set({
          telemetry: {
            ...get().telemetry,
            [id]: {
              battery: msg.battery,
              altitude: msg.altitude,
              speed: msg.speed,
              heading: msg.heading,
              gps_fix: msg.gps_fix,
              signal: msg.signal,
              timestamp: msg.timestamp || now,
              ...(msg.extras || {}),
            },
          },
          lastMessageAt: now,
        });
        break;
      }

      /* ── real-time threat notification ── */
      case 'threat_event': {
        const threat = {
          id: msg.threat_id || `t-${now}`,
          severity: msg.severity || 'warning',
          label: msg.label || 'Unknown Threat',
          description: msg.description || '',
          position: msg.position,
          timestamp: msg.timestamp || now,
          acknowledged: false,
        };
        const hist = [threat, ...get().threatHistory].slice(0, 200);
        set({
          threats: [...get().threats, threat],
          threatHistory: hist,
          mapThreats: [...get().mapThreats, threat],
          activeAlerts: [...get().activeAlerts, threat],
          lastMessageAt: now,
        });
        break;
      }

      /* ── crowd density grid ── */
      case 'crowd_density': {
        set({
          crowdGrid: {
            cols: msg.cols,
            rows: msg.rows,
            cell_size: msg.cell_size || 10,
            data: msg.data, // flat array [rows*cols] 0..1
          },
          crowdZones: msg.zones || get().crowdZones,
          lastMessageAt: now,
        });
        break;
      }

      /* ── stampede risk per zone ── */
      case 'stampede_risk': {
        const risks = { ...get().stampedeRisks };
        (msg.zones || []).forEach((z) => {
          risks[z.zone_id] = {
            risk: z.risk,
            indicators: z.indicators || {},
            label: z.label || z.zone_id,
            timestamp: msg.timestamp || now,
          };
        });
        set({ stampedeRisks: risks, lastMessageAt: now });
        break;
      }

      /* ── camera frame ── */
      case 'camera_frame': {
        if (!msg.drone_id) break;
        set({
          cameraFrames: {
            ...get().cameraFrames,
            [msg.drone_id]: msg.url || msg.data_uri || null,
          },
          lastMessageAt: now,
        });
        break;
      }

      /* ── zone definitions ── */
      case 'zone_update': {
        set({
          zones: msg.zones || [],
          lastMessageAt: now,
        });
        break;
      }

      /* ── audit log entry ── */
      case 'audit': {
        const entry = {
          id: msg.entry_id || `a-${now}-${Math.random().toString(36).slice(2, 6)}`,
          timestamp: msg.timestamp || now,
          level: msg.level || 'info',
          source: msg.source || 'system',
          action: msg.action || '',
          detail: msg.detail || '',
          user: msg.user || 'system',
        };
        set({
          auditLog: [entry, ...get().auditLog].slice(0, 500),
          lastMessageAt: now,
        });
        break;
      }

      default:
        /* Unknown message type — silently ignore */
        break;
    }
  },

  /* ── Actions callable from UI ── */

  acknowledgeAlert: (id) => {
    set({
      activeAlerts: get().activeAlerts.filter((a) => a.id !== id),
      threats: get().threats.map((t) =>
        t.id === id ? { ...t, acknowledged: true } : t,
      ),
    });
  },

  clearAlerts: () => set({ activeAlerts: [] }),

  addAuditEntry: (entry) => {
    const full = {
      id: `local-${Date.now()}`,
      timestamp: Date.now(),
      level: 'info',
      source: 'operator',
      ...entry,
    };
    set({ auditLog: [full, ...get().auditLog].slice(0, 500) });
  },

  updateZones: (zones) => set({ zones }),
}));

export default useGCSState;
