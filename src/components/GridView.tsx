/** Renders a numeric grid as a heat-mapped bitmap. Used for inputs,
 *  filters and feature maps. */

import { useEffect, useRef } from 'react';
import type { Grid } from '../lib/conv.ts';

interface Props {
  grid: Grid;
  cell?: number;
  /** 'mono' for non-negative data, 'signed' for weights */
  mode?: 'mono' | 'signed';
  /** override the normalisation range; otherwise taken from the grid */
  max?: number;
  label?: string;
  gap?: boolean;
}

export function GridView({ grid, cell = 10, mode = 'mono', max, label, gap = false }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const h = grid.length;
  const w = grid[0]?.length ?? 0;

  useEffect(() => {
    const c = ref.current;
    if (!c || !w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * cell * dpr; c.height = h * cell * dpr;
    c.style.width = w * cell + 'px'; c.style.height = h * cell + 'px';
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    let hi = max ?? 0;
    if (hi === 0) {
      for (const row of grid) for (const v of row) hi = Math.max(hi, Math.abs(v));
    }
    if (hi === 0) hi = 1;

    for (let r = 0; r < h; r++) {
      for (let cx = 0; cx < w; cx++) {
        const v = grid[r][cx];
        const t = Math.max(-1, Math.min(1, v / hi));
        if (mode === 'signed') {
          g.fillStyle = t >= 0
            ? `rgba(217,122,69,${0.06 + Math.abs(t) * 0.94})`
            : `rgba(91,141,190,${0.06 + Math.abs(t) * 0.94})`;
        } else {
          const a = Math.max(0, t);
          g.fillStyle = `rgba(236,235,228,${0.04 + a * 0.96})`;
        }
        const inset = gap ? 0.5 : 0;
        g.fillRect(cx * cell + inset, r * cell + inset, cell - inset * 2, cell - inset * 2);
      }
    }

    g.strokeStyle = '#32322c'; g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, w * cell - 1, h * cell - 1);
  }, [grid, cell, mode, max, w, h, gap]);

  return (
    <div style={{ display: 'inline-block', textAlign: 'center' }}>
      <canvas ref={ref} />
      {label && (
        <div style={{ fontSize: 10, color: '#74716a', marginTop: 3, fontFamily: 'ui-monospace, monospace' }}>
          {label}
        </div>
      )}
    </div>
  );
}
