/** Small canvas-based charts: the loss curve and the decision boundary. */

import { useEffect, useRef } from 'react';
import type { MLP } from '../lib/nn.ts';
import type { Point, FeatureId } from '../lib/datasets.ts';
import { featurise } from '../lib/datasets.ts';

const POS = '#d97a45';
const NEG = '#5b8dbe';

/* ---------------- loss curve ---------------- */

export function LossChart({ train, test, width = 312, height = 132 }: {
  train: number[]; test: number[]; width?: number; height?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr; cv.height = height * dpr;
    cv.style.width = width + 'px'; cv.style.height = height + 'px';
    const g = cv.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);

    const pad = { l: 30, r: 6, t: 8, b: 16 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;

    const all = [...train, ...test].filter(Number.isFinite);
    const hi = all.length ? Math.max(...all, 0.01) : 1;
    const n = Math.max(train.length, test.length, 2);

    // grid
    g.strokeStyle = '#2b2b25'; g.lineWidth = 1;
    g.fillStyle = '#74716a'; g.font = '9px ui-monospace, monospace';
    g.textAlign = 'right';
    for (let i = 0; i <= 2; i++) {
      const y = pad.t + (ih * i) / 2;
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(pad.l + iw, y); g.stroke();
      g.fillText((hi * (1 - i / 2)).toFixed(2), pad.l - 5, y + 3);
    }

    const draw = (series: number[], colour: string, dash: number[]) => {
      if (series.length < 2) return;
      g.strokeStyle = colour; g.lineWidth = 1.6; g.setLineDash(dash);
      g.beginPath();
      series.forEach((v, i) => {
        const x = pad.l + (iw * i) / (n - 1);
        const y = pad.t + ih * (1 - Math.min(v / hi, 1));
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke(); g.setLineDash([]);
    };

    draw(train, POS, []);
    draw(test, NEG, [3, 3]);
  }, [train, test, width, height]);

  return <canvas ref={ref} />;
}

/* ---------------- decision boundary ---------------- */

export function BoundaryChart({ net, points, features, size = 312, res = 44, tick }: {
  net: MLP; points: Point[]; features: FeatureId[]; size?: number; res?: number; tick: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr; cv.height = size * dpr;
    cv.style.width = size + 'px'; cv.style.height = size + 'px';
    const g = cv.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);

    // sample the network over a grid and paint the confidence field
    const cell = size / res;
    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const x = (i + 0.5) / res * 2 - 1;
        const y = 1 - (j + 0.5) / res * 2;
        const out = net.forward(featurise(x, y, features))[0];
        const t = Math.max(0, Math.min(1, out));
        // 0 -> blue, 1 -> orange, 0.5 -> flat
        const strength = Math.abs(t - 0.5) * 2;
        g.fillStyle = t >= 0.5
          ? `rgba(217,122,69,${0.10 + strength * 0.46})`
          : `rgba(91,141,190,${0.10 + strength * 0.46})`;
        g.fillRect(i * cell, j * cell, cell + 1, cell + 1);
      }
    }

    // the data on top
    for (const p of points) {
      const cx = ((p.x + 1) / 2) * size;
      const cy = ((1 - p.y) / 2) * size;
      g.beginPath();
      g.arc(cx, cy, 3.4, 0, Math.PI * 2);
      g.fillStyle = p.label === 1 ? POS : NEG;
      g.fill();
      g.lineWidth = 1;
      g.strokeStyle = 'rgba(20,20,15,.85)';
      g.stroke();
    }

    g.strokeStyle = '#32322c'; g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, size - 1, size - 1);
  }, [net, points, features, size, res, tick]);

  return <canvas ref={ref} />;
}
