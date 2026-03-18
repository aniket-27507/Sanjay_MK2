import React, { useRef, useEffect } from 'react';
import { densityToColor } from '../../utils/colors';

/**
 * Canvas overlay that renders a density grid as a green-to-red heatmap.
 *
 * Props:
 *   grid   — { cols, rows, cell_size, data: number[] }  (values 0..1)
 *   width  — canvas pixel width
 *   height — canvas pixel height
 *   worldSize — world coordinate size (e.g. 1000)
 */
export default function CrowdHeatmapLayer({ grid, width, height, worldSize }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid || !grid.data) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    const { cols, rows, cell_size, data } = grid;
    if (!cols || !rows || data.length === 0) return;

    const scale = Math.min(width, height) / worldSize;
    const cellW = cell_size * scale;
    const cellH = cell_size * scale;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const value = data[idx] || 0;
        if (value < 0.02) continue; // skip near-zero for perf

        const x = c * cellW;
        const y = r * cellH;

        ctx.fillStyle = densityToColor(value, 0.45);
        ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // Smooth pass — light gaussian-like blur via globalCompositeOperation
    ctx.globalAlpha = 0.3;
    ctx.filter = 'blur(6px)';
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = 'none';
    ctx.globalAlpha = 1.0;
  }, [grid, width, height, worldSize]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 1,
      }}
    />
  );
}
