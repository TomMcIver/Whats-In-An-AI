/**
 * Two-dimensional toy datasets.
 *
 * All points live in [-1, 1]² with a label of 0 or 1, which makes the
 * decision boundary directly drawable and keeps inputs pre-normalised.
 */

import { RNG } from './rng.ts';

export interface Point {
  x: number;
  y: number;
  label: 0 | 1;
}

export type DatasetKind = 'circle' | 'xor' | 'gauss' | 'spiral' | 'moons';

export const DATASETS: { id: DatasetKind; name: string; note: string }[] = [
  { id: 'circle', name: 'Circle',   note: 'one class ringed by the other — needs a curved boundary' },
  { id: 'xor',    name: 'XOR',      note: 'the classic case a single neuron provably cannot solve' },
  { id: 'gauss',  name: 'Gaussian', note: 'two blobs — linearly separable, solvable without hidden layers' },
  { id: 'spiral', name: 'Spiral',   note: 'two interleaved arms — the hardest of these by a wide margin' },
  { id: 'moons',  name: 'Moons',    note: 'two crescents — mildly non-linear' },
];

export function generate(kind: DatasetKind, n: number, noise: number, seed: number): Point[] {
  const rng = new RNG(seed);
  const pts: Point[] = [];
  const jitter = () => rng.normal(0, noise);

  switch (kind) {
    case 'circle': {
      for (let i = 0; i < n; i++) {
        const inner = i % 2 === 0;
        const r = inner ? rng.range(0, 0.45) : rng.range(0.65, 1.0);
        const t = rng.range(0, Math.PI * 2);
        pts.push({
          x: r * Math.cos(t) + jitter(),
          y: r * Math.sin(t) + jitter(),
          label: inner ? 1 : 0,
        });
      }
      break;
    }

    case 'xor': {
      for (let i = 0; i < n; i++) {
        let x = rng.range(-1, 1);
        let y = rng.range(-1, 1);
        // push away from the axes so the quadrants stay legible
        x += x > 0 ? 0.06 : -0.06;
        y += y > 0 ? 0.06 : -0.06;
        pts.push({
          x: x + jitter(),
          y: y + jitter(),
          label: x * y > 0 ? 1 : 0,
        });
      }
      break;
    }

    case 'gauss': {
      for (let i = 0; i < n; i++) {
        const a = i % 2 === 0;
        const cx = a ? 0.45 : -0.45;
        const cy = a ? 0.45 : -0.45;
        pts.push({
          x: cx + rng.normal(0, 0.22 + noise),
          y: cy + rng.normal(0, 0.22 + noise),
          label: a ? 1 : 0,
        });
      }
      break;
    }

    case 'spiral': {
      const per = Math.floor(n / 2);
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < per; i++) {
          const r = (i / per) * 0.95;
          const t = 1.75 * i / per * Math.PI * 2 + c * Math.PI;
          pts.push({
            x: r * Math.sin(t) + jitter(),
            y: r * Math.cos(t) + jitter(),
            label: c as 0 | 1,
          });
        }
      }
      break;
    }

    case 'moons': {
      const per = Math.floor(n / 2);
      for (let i = 0; i < per; i++) {
        const t = (i / per) * Math.PI;
        pts.push({ x: Math.cos(t) * 0.75 - 0.25 + jitter(), y: Math.sin(t) * 0.6 - 0.2 + jitter(), label: 0 });
        pts.push({ x: Math.cos(t + Math.PI) * 0.75 + 0.25 + jitter(), y: Math.sin(t + Math.PI) * 0.6 + 0.2 + jitter(), label: 1 });
      }
      break;
    }
  }

  return pts.map(p => ({
    x: Math.max(-1, Math.min(1, p.x)),
    y: Math.max(-1, Math.min(1, p.y)),
    label: p.label,
  }));
}

/* ------------------------------------------------------------------
   Input features
   Letting the user add x², xy and sin terms shows very directly that
   "depth" and "clever features" are substitutes for one another.
   ------------------------------------------------------------------ */

export type FeatureId = 'x' | 'y' | 'x2' | 'y2' | 'xy' | 'sinx' | 'siny';

export const FEATURES: { id: FeatureId; label: string; fn: (x: number, y: number) => number }[] = [
  { id: 'x',    label: 'X₁',        fn: (x) => x },
  { id: 'y',    label: 'X₂',        fn: (_, y) => y },
  { id: 'x2',   label: 'X₁²',       fn: (x) => x * x },
  { id: 'y2',   label: 'X₂²',       fn: (_, y) => y * y },
  { id: 'xy',   label: 'X₁X₂',      fn: (x, y) => x * y },
  { id: 'sinx', label: 'sin(X₁)',   fn: (x) => Math.sin(3 * x) },
  { id: 'siny', label: 'sin(X₂)',   fn: (_, y) => Math.sin(3 * y) },
];

export function featurise(x: number, y: number, active: FeatureId[]): number[] {
  return active.map(id => FEATURES.find(f => f.id === id)!.fn(x, y));
}

export function splitTrainTest(pts: Point[], trainRatio: number, seed: number) {
  const rng = new RNG(seed);
  const shuffled = rng.shuffle(pts.slice());
  const cut = Math.floor(shuffled.length * trainRatio);
  return { train: shuffled.slice(0, cut), test: shuffled.slice(cut) };
}
