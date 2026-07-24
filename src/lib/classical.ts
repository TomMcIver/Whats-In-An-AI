/**
 * Classical ML algorithms, each written so it can be *stepped*.
 *
 * Every model exposes a step() that advances the fit by one visible unit —
 * one gradient step, one k-means reassignment, one tree level — so the UI
 * can show the model converging rather than jumping to the answer.
 */

import { RNG } from './rng.ts';
import type { Point } from './datasets.ts';

/* ================================================================
   1-D data for line fitting
   ================================================================ */

export interface XY { x: number; y: number; }

export type RegressionShape = 'linear' | 'curved' | 'noisy';

export function makeRegression(shape: RegressionShape, n: number, noise: number, seed: number): XY[] {
  const rng = new RNG(seed);
  const out: XY[] = [];
  for (let i = 0; i < n; i++) {
    const x = rng.range(-1, 1);
    let y: number;
    if (shape === 'linear') y = 0.75 * x + 0.1;
    else if (shape === 'curved') y = 0.9 * x * x - 0.35;
    else y = 0.6 * x + 0.15;
    out.push({ x, y: y + rng.normal(0, noise) });
  }
  return out;
}

/* ================================================================
   Linear regression by gradient descent
   ================================================================ */

export class LinearRegressionGD {
  w = 0;
  b = 0;
  steps = 0;
  history: number[] = [];

  reset() { this.w = 0; this.b = 0; this.steps = 0; this.history = []; }

  predict(x: number) { return this.w * x + this.b; }

  loss(data: XY[]) {
    let s = 0;
    for (const p of data) { const d = this.predict(p.x) - p.y; s += d * d; }
    return s / (2 * Math.max(data.length, 1));
  }

  /** one full-batch gradient step */
  step(data: XY[], lr: number) {
    let dw = 0, db = 0;
    for (const p of data) {
      const e = this.predict(p.x) - p.y;
      dw += e * p.x;
      db += e;
    }
    const n = Math.max(data.length, 1);
    this.w -= lr * (dw / n);
    this.b -= lr * (db / n);
    this.steps++;
    this.history.push(this.loss(data));
    if (this.history.length > 300) this.history.shift();
  }

  /** the exact least-squares answer, for comparison against the iterative fit */
  static closedForm(data: XY[]) {
    const n = data.length;
    if (!n) return { w: 0, b: 0 };
    let sx = 0, sy = 0, sxy = 0, sxx = 0;
    for (const p of data) { sx += p.x; sy += p.y; sxy += p.x * p.y; sxx += p.x * p.x; }
    const denom = n * sxx - sx * sx;
    if (Math.abs(denom) < 1e-12) return { w: 0, b: sy / n };
    const w = (n * sxy - sx * sy) / denom;
    return { w, b: (sy - w * sx) / n };
  }
}

/* ================================================================
   Logistic regression by gradient descent (2-D -> class)
   ================================================================ */

export class LogisticRegressionGD {
  w = [0, 0];
  b = 0;
  steps = 0;
  history: number[] = [];

  reset() { this.w = [0, 0]; this.b = 0; this.steps = 0; this.history = []; }

  prob(x: number, y: number) {
    const z = this.w[0] * x + this.w[1] * y + this.b;
    return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, z))));
  }

  loss(pts: Point[]) {
    let s = 0;
    for (const p of pts) {
      const q = Math.min(Math.max(this.prob(p.x, p.y), 1e-9), 1 - 1e-9);
      s += -(p.label * Math.log(q) + (1 - p.label) * Math.log(1 - q));
    }
    return s / Math.max(pts.length, 1);
  }

  accuracy(pts: Point[]) {
    let ok = 0;
    for (const p of pts) if ((this.prob(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) ok++;
    return ok / Math.max(pts.length, 1);
  }

  step(pts: Point[], lr: number) {
    let dw0 = 0, dw1 = 0, db = 0;
    for (const p of pts) {
      const e = this.prob(p.x, p.y) - p.label;
      dw0 += e * p.x; dw1 += e * p.y; db += e;
    }
    const n = Math.max(pts.length, 1);
    this.w[0] -= lr * (dw0 / n);
    this.w[1] -= lr * (dw1 / n);
    this.b -= lr * (db / n);
    this.steps++;
    this.history.push(this.loss(pts));
    if (this.history.length > 300) this.history.shift();
  }
}

/* ================================================================
   k-means
   ================================================================ */

export interface Centroid { x: number; y: number; }

export class KMeans {
  centroids: Centroid[] = [];
  assignment: number[] = [];
  iterations = 0;
  converged = false;
  /** which half-step comes next — the two are shown separately on purpose */
  phase: 'assign' | 'update' = 'assign';
  k: number;

  constructor(k: number, pts: Point[], seed: number) {
    this.k = k;
    this.init(pts, seed);
  }

  init(pts: Point[], seed: number) {
    const rng = new RNG(seed);
    this.centroids = [];
    for (let i = 0; i < this.k; i++) {
      const p = pts[Math.floor(rng.next() * pts.length)] ?? { x: 0, y: 0 };
      this.centroids.push({ x: p.x + rng.normal(0, 0.08), y: p.y + rng.normal(0, 0.08) });
    }
    this.assignment = new Array(pts.length).fill(-1);
    this.iterations = 0;
    this.converged = false;
    this.phase = 'assign';
  }

  private nearest(x: number, y: number) {
    let best = 0, bd = Infinity;
    for (let c = 0; c < this.centroids.length; c++) {
      const dx = x - this.centroids[c].x, dy = y - this.centroids[c].y;
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = c; }
    }
    return best;
  }

  /** advance one half-step: assign, then update, then assign… */
  step(pts: Point[]) {
    if (this.phase === 'assign') {
      let changed = false;
      for (let i = 0; i < pts.length; i++) {
        const a = this.nearest(pts[i].x, pts[i].y);
        if (a !== this.assignment[i]) changed = true;
        this.assignment[i] = a;
      }
      if (!changed && this.iterations > 0) this.converged = true;
      this.phase = 'update';
    } else {
      for (let c = 0; c < this.centroids.length; c++) {
        let sx = 0, sy = 0, n = 0;
        for (let i = 0; i < pts.length; i++) {
          if (this.assignment[i] === c) { sx += pts[i].x; sy += pts[i].y; n++; }
        }
        if (n > 0) { this.centroids[c] = { x: sx / n, y: sy / n }; }
      }
      this.iterations++;
      this.phase = 'assign';
    }
  }

  /** mean squared distance of each point to its centroid */
  inertia(pts: Point[]) {
    let s = 0;
    for (let i = 0; i < pts.length; i++) {
      const c = this.centroids[this.assignment[i]];
      if (!c) continue;
      s += (pts[i].x - c.x) ** 2 + (pts[i].y - c.y) ** 2;
    }
    return s / Math.max(pts.length, 1);
  }
}

/* ================================================================
   Decision tree, grown one level at a time
   ================================================================ */

export interface TreeNode {
  /** null on a leaf */
  axis: 'x' | 'y' | null;
  threshold: number;
  left?: TreeNode;
  right?: TreeNode;
  /** proportion of class 1 in this region */
  value: number;
  count: number;
  depth: number;
  /** the region this node owns, for drawing */
  box: { x0: number; x1: number; y0: number; y1: number };
}

function gini(pts: Point[]) {
  if (!pts.length) return 0;
  let ones = 0;
  for (const p of pts) ones += p.label;
  const p1 = ones / pts.length;
  return 1 - (p1 * p1 + (1 - p1) * (1 - p1));
}

function meanLabel(pts: Point[]) {
  if (!pts.length) return 0.5;
  let s = 0;
  for (const p of pts) s += p.label;
  return s / pts.length;
}

/** best single split over both axes, by weighted Gini */
function bestSplit(pts: Point[]) {
  let best: { axis: 'x' | 'y'; threshold: number; gain: number } | null = null;
  const parent = gini(pts);

  for (const axis of ['x', 'y'] as const) {
    const vals = pts.map(p => p[axis]).sort((a, b) => a - b);
    for (let i = 1; i < vals.length; i++) {
      if (vals[i] === vals[i - 1]) continue;
      const t = (vals[i] + vals[i - 1]) / 2;
      const L = pts.filter(p => p[axis] <= t);
      const R = pts.filter(p => p[axis] > t);
      if (!L.length || !R.length) continue;
      const g = parent - (L.length * gini(L) + R.length * gini(R)) / pts.length;
      if (!best || g > best.gain) best = { axis, threshold: t, gain: g };
    }
  }
  return best;
}

export class DecisionTree {
  root: TreeNode;
  depth = 0;
  minLeaf: number;
  private pointsAt = new Map<TreeNode, Point[]>();

  constructor(pts: Point[], minLeaf = 4) {
    this.minLeaf = minLeaf;
    this.root = {
      axis: null, threshold: 0, value: meanLabel(pts), count: pts.length, depth: 0,
      box: { x0: -1, x1: 1, y0: -1, y1: 1 },
    };
    this.pointsAt.set(this.root, pts);
  }

  /** split every current leaf that is still worth splitting */
  growOneLevel(): boolean {
    const leaves: TreeNode[] = [];
    const walk = (n: TreeNode) => {
      if (n.axis === null) leaves.push(n);
      else { walk(n.left!); walk(n.right!); }
    };
    walk(this.root);

    let grew = false;
    for (const leaf of leaves) {
      const pts = this.pointsAt.get(leaf) ?? [];
      if (pts.length < this.minLeaf * 2) continue;
      if (gini(pts) < 1e-9) continue;              // already pure
      const s = bestSplit(pts);
      if (!s || s.gain <= 1e-9) continue;

      const L = pts.filter(p => p[s.axis] <= s.threshold);
      const R = pts.filter(p => p[s.axis] > s.threshold);
      if (L.length < this.minLeaf || R.length < this.minLeaf) continue;

      leaf.axis = s.axis;
      leaf.threshold = s.threshold;

      const b = leaf.box;
      const lBox = s.axis === 'x'
        ? { ...b, x1: s.threshold }
        : { ...b, y1: s.threshold };
      const rBox = s.axis === 'x'
        ? { ...b, x0: s.threshold }
        : { ...b, y0: s.threshold };

      leaf.left = { axis: null, threshold: 0, value: meanLabel(L), count: L.length, depth: leaf.depth + 1, box: lBox };
      leaf.right = { axis: null, threshold: 0, value: meanLabel(R), count: R.length, depth: leaf.depth + 1, box: rBox };
      this.pointsAt.set(leaf.left, L);
      this.pointsAt.set(leaf.right, R);
      grew = true;
    }
    if (grew) this.depth++;
    return grew;
  }

  predict(x: number, y: number): number {
    let n = this.root;
    while (n.axis !== null) {
      n = (n.axis === 'x' ? x : y) <= n.threshold ? n.left! : n.right!;
    }
    return n.value;
  }

  accuracy(pts: Point[]) {
    let ok = 0;
    for (const p of pts) if ((this.predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) ok++;
    return ok / Math.max(pts.length, 1);
  }

  leaves(): TreeNode[] {
    const out: TreeNode[] = [];
    const walk = (n: TreeNode) => { if (n.axis === null) out.push(n); else { walk(n.left!); walk(n.right!); } };
    walk(this.root);
    return out;
  }

  nodeCount(): number {
    let c = 0;
    const walk = (n: TreeNode) => { c++; if (n.axis !== null) { walk(n.left!); walk(n.right!); } };
    walk(this.root);
    return c;
  }
}

/* ================================================================
   k-nearest neighbours — no training at all
   ================================================================ */

export class KNN {
  k: number;
  pts: Point[];

  constructor(k: number, pts: Point[]) {
    this.k = k;
    this.pts = pts;
  }

  /** returns the fraction of the k neighbours that are class 1 */
  predict(x: number, y: number): number {
    if (!this.pts.length) return 0.5;
    const d = this.pts.map(p => ({ d: (p.x - x) ** 2 + (p.y - y) ** 2, label: p.label }));
    d.sort((a, b) => a.d - b.d);
    const k = Math.min(this.k, d.length);
    let s = 0;
    for (let i = 0; i < k; i++) s += d[i].label;
    return s / k;
  }

  /** the k nearest neighbours of a probe point, for drawing the links */
  neighbours(x: number, y: number) {
    const d = this.pts.map((p, i) => ({ i, p, d: Math.hypot(p.x - x, p.y - y) }));
    d.sort((a, b) => a.d - b.d);
    return d.slice(0, Math.min(this.k, d.length));
  }

  accuracy(test: Point[]) {
    let ok = 0;
    for (const p of test) if ((this.predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) ok++;
    return ok / Math.max(test.length, 1);
  }
}
