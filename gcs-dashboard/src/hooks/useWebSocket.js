import { useEffect, useRef, useCallback, useState } from 'react';
import useGCSState from './useGCSState';

const WS_URL = 'ws://localhost:8765';
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

/**
 * Custom hook that manages a single WebSocket connection to the GCS backend.
 * All incoming messages are routed into the Zustand store via `dispatch`.
 *
 * Returns { connected, latencyMs, send, reconnect }.
 */
export default function useWebSocket() {
  const dispatch = useGCSState((s) => s.dispatch);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const reconnectDelay = useRef(RECONNECT_BASE_MS);
  const pingTs = useRef(null);

  const [connected, setConnected] = useState(false);
  const [latencyMs, setLatencyMs] = useState(null);

  const cleanup = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    cleanup();

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      reconnectDelay.current = RECONNECT_BASE_MS;
      dispatch({ type: 'ws_connected' });
    };

    ws.onclose = () => {
      setConnected(false);
      dispatch({ type: 'ws_disconnected' });
      scheduleReconnect();
    };

    ws.onerror = () => {
      /* onclose fires after onerror, reconnect handled there */
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        /* measure round-trip if backend echoes pong */
        if (msg.type === 'pong' && pingTs.current) {
          setLatencyMs(Date.now() - pingTs.current);
          pingTs.current = null;
          return;
        }

        dispatch(msg);
      } catch {
        /* ignore non-JSON frames */
      }
    };
  }, [cleanup, dispatch]);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current) return;
    reconnectTimer.current = setTimeout(() => {
      reconnectTimer.current = null;
      reconnectDelay.current = Math.min(
        reconnectDelay.current * 2,
        RECONNECT_MAX_MS,
      );
      connect();
    }, reconnectDelay.current);
  }, [connect]);

  /** Send a JSON-serialisable object to the backend. */
  const send = useCallback((obj) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  }, []);

  /** Manually trigger a reconnect. */
  const reconnect = useCallback(() => {
    reconnectDelay.current = RECONNECT_BASE_MS;
    connect();
  }, [connect]);

  /* Ping every 10 s for latency measurement */
  useEffect(() => {
    const id = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        pingTs.current = Date.now();
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 10000);
    return () => clearInterval(id);
  }, []);

  /* Connect on mount, clean up on unmount */
  useEffect(() => {
    connect();
    return cleanup;
  }, [connect, cleanup]);

  return { connected, latencyMs, send, reconnect };
}
