/**
 * A small convolutional network, written from scratch with backprop
 * running all the way through the filters.
 *
 * Architecture:  12x12 input
 *                -> conv 3x3 x F filters (valid)  -> 10x10 x F
 *                -> ReLU
 *                -> 2x2 max pool                  -> 5x5 x F
 *                -> flatten -> dense -> softmax over C classes
 *
 * Everything is kept as plain arrays and every intermediate is retained,
 * so the UI can draw the filters, the feature maps and the pooled maps
 * as they change during training.
 */

import { RNG } from './rng.ts';

export const IMG = 12;
export const K = 3;
export const CONV_OUT = IMG - K + 1;   // 10
export const POOL = 2;
export const POOL_OUT = CONV_OUT / POOL; // 5

export type Grid = number[][];

export function zeros(h: number, w: number): Grid {
  return Array.from({ length: h }, () => new Array(w).fill(0));
}

/* ================================================================
   Shape dataset — simple glyphs the net has to tell apart
   ================================================================ */

export const SHAPES = ['vertical', 'horizontal', 'cross', 'diagonal'] as const;
export type Shape = typeof SHAPES[number];

export function drawShape(shape: Shape, rng: RNG, jitter = true): Grid {
  const g = zeros(IMG, IMG);
  const off = jitter ? Math.floor(rng.range(-2, 2.99)) : 0;
  const off2 = jitter ? Math.floor(rng.range(-2, 2.99)) : 0;
  const mid = Math.floor(IMG / 2);
  const put = (r: number, c: number, v = 1) => {
    if (r >= 0 && r < IMG && c >= 0 && c < IMG) g[r][c] = Math.max(g[r][c], v);
  };

  const thick = 1;
  switch (shape) {
    case 'vertical':
      for (let r = 2; r < IMG - 2; r++)
        for (let t = 0; t <= thick; t++) put(r, mid + off + t);
      break;
    case 'horizontal':
      for (let c = 2; c < IMG - 2; c++)
        for (let t = 0; t <= thick; t++) put(mid + off2 + t, c);
      break;
    case 'cross':
      for (let r = 2; r < IMG - 2; r++) put(r, mid + off);
      for (let c = 2; c < IMG - 2; c++) put(mid + off2, c);
      break;
    case 'diagonal':
      for (let i = 2; i < IMG - 2; i++) {
        put(i + off2, i + off);
        put(i + off2, i + off + 1);
      }
      break;
  }

  // light speckle so the task is not trivially clean
  if (jitter) {
    for (let i = 0; i < 6; i++) {
      const r = Math.floor(rng.next() * IMG);
      const c = Math.floor(rng.next() * IMG);
      g[r][c] = Math.max(g[r][c], rng.range(0.15, 0.45));
    }
  }
  return g;
}

export interface Sample { img: Grid; label: number; }

export function makeShapeSet(n: number, seed: number): Sample[] {
  const rng = new RNG(seed);
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % SHAPES.length;
    out.push({ img: drawShape(SHAPES[label], rng), label });
  }
  return rng.shuffle(out);
}

/* ================================================================
   The network
   ================================================================ */

export class ConvNet {
  nFilters: number;
  nClasses: number;
  rng: RNG;

  /** filters[f][ky][kx] */
  filters: Grid[] = [];
  filterBias: number[] = [];

  /** dense weights[c][flatIndex] */
  W: number[][] = [];
  bOut: number[] = [];

  /* retained intermediates from the last forward pass, for drawing */
  lastInput: Grid = zeros(IMG, IMG);
  convMaps: Grid[] = [];      // pre-ReLU
  reluMaps: Grid[] = [];
  poolMaps: Grid[] = [];
  poolArgmax: [number, number][][] = [];   // per filter, per pooled cell
  logits: number[] = [];
  probs: number[] = [];

  epoch = 0;
  lossHistory: number[] = [];
  accHistory: number[] = [];

  constructor(nFilters = 4, nClasses = 4, seed = 3) {
    this.nFilters = nFilters;
    this.nClasses = nClasses;
    this.rng = new RNG(seed);
    this.init();
  }

  init() {
    const rng = this.rng;
    this.filters = Array.from({ length: this.nFilters }, () =>
      Array.from({ length: K }, () =>
        Array.from({ length: K }, () => rng.normal(0, Math.sqrt(2 / (K * K))))
      )
    );
    this.filterBias = new Array(this.nFilters).fill(0);

    const flat = this.nFilters * POOL_OUT * POOL_OUT;
    this.W = Array.from({ length: this.nClasses }, () =>
      Array.from({ length: flat }, () => rng.normal(0, Math.sqrt(1 / flat)))
    );
    this.bOut = new Array(this.nClasses).fill(0);
    this.epoch = 0;
    this.lossHistory = [];
    this.accHistory = [];
  }

  reset(seed?: number) {
    if (seed !== undefined) this.rng.reseed(seed);
    this.init();
  }

  forward(img: Grid): number[] {
    this.lastInput = img;
    this.convMaps = [];
    this.reluMaps = [];
    this.poolMaps = [];
    this.poolArgmax = [];

    for (let f = 0; f < this.nFilters; f++) {
      const conv = zeros(CONV_OUT, CONV_OUT);
      const relu = zeros(CONV_OUT, CONV_OUT);
      for (let r = 0; r < CONV_OUT; r++) {
        for (let c = 0; c < CONV_OUT; c++) {
          let s = this.filterBias[f];
          for (let ky = 0; ky < K; ky++)
            for (let kx = 0; kx < K; kx++)
              s += img[r + ky][c + kx] * this.filters[f][ky][kx];
          conv[r][c] = s;
          relu[r][c] = s > 0 ? s : 0;
        }
      }
      this.convMaps.push(conv);
      this.reluMaps.push(relu);

      const pool = zeros(POOL_OUT, POOL_OUT);
      const arg: [number, number][] = [];
      for (let r = 0; r < POOL_OUT; r++) {
        for (let c = 0; c < POOL_OUT; c++) {
          let best = -Infinity, br = 0, bc = 0;
          for (let dr = 0; dr < POOL; dr++)
            for (let dc = 0; dc < POOL; dc++) {
              const v = relu[r * POOL + dr][c * POOL + dc];
              if (v > best) { best = v; br = r * POOL + dr; bc = c * POOL + dc; }
            }
          pool[r][c] = best;
          arg.push([br, bc]);
        }
      }
      this.poolMaps.push(pool);
      this.poolArgmax.push(arg);
    }

    // flatten -> dense -> softmax
    const flat = this.flatten();
    this.logits = [];
    for (let c = 0; c < this.nClasses; c++) {
      let s = this.bOut[c];
      for (let i = 0; i < flat.length; i++) s += this.W[c][i] * flat[i];
      this.logits.push(s);
    }
    const mx = Math.max(...this.logits);
    const ex = this.logits.map(v => Math.exp(v - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    this.probs = ex.map(v => v / sum);
    return this.probs;
  }

  private flatten(): number[] {
    const out: number[] = [];
    for (let f = 0; f < this.nFilters; f++)
      for (let r = 0; r < POOL_OUT; r++)
        for (let c = 0; c < POOL_OUT; c++) out.push(this.poolMaps[f][r][c]);
    return out;
  }

  /**
   * Backprop one example and apply the update immediately.
   * Returns the cross-entropy loss for this example.
   */
  trainOne(img: Grid, label: number, lr: number): number {
    const probs = this.forward(img);
    const loss = -Math.log(Math.max(probs[label], 1e-9));

    const flat = this.flatten();
    const flatLen = flat.length;

    // dL/dlogits for softmax + cross-entropy
    const dLogits = probs.slice();
    dLogits[label] -= 1;

    // gradient into the flattened pooled maps
    const dFlat = new Array(flatLen).fill(0);
    for (let c = 0; c < this.nClasses; c++) {
      const d = dLogits[c];
      for (let i = 0; i < flatLen; i++) dFlat[i] += this.W[c][i] * d;
    }

    // dense update
    for (let c = 0; c < this.nClasses; c++) {
      const d = dLogits[c];
      for (let i = 0; i < flatLen; i++) this.W[c][i] -= lr * d * flat[i];
      this.bOut[c] -= lr * d;
    }

    // route gradient back through max pool, then ReLU, then the convolution
    let idx = 0;
    for (let f = 0; f < this.nFilters; f++) {
      const dRelu = zeros(CONV_OUT, CONV_OUT);
      for (let r = 0; r < POOL_OUT; r++) {
        for (let c = 0; c < POOL_OUT; c++) {
          const [br, bc] = this.poolArgmax[f][r * POOL_OUT + c];
          dRelu[br][bc] += dFlat[idx++];      // only the max position receives gradient
        }
      }

      const dFilter = zeros(K, K);
      let dBias = 0;
      for (let r = 0; r < CONV_OUT; r++) {
        for (let c = 0; c < CONV_OUT; c++) {
          if (this.convMaps[f][r][c] <= 0) continue;   // ReLU gate
          const d = dRelu[r][c];
          if (d === 0) continue;
          dBias += d;
          for (let ky = 0; ky < K; ky++)
            for (let kx = 0; kx < K; kx++)
              dFilter[ky][kx] += d * this.lastInput[r + ky][c + kx];
        }
      }

      for (let ky = 0; ky < K; ky++)
        for (let kx = 0; kx < K; kx++)
          this.filters[f][ky][kx] -= lr * dFilter[ky][kx];
      this.filterBias[f] -= lr * dBias;
    }

    return loss;
  }

  trainEpoch(set: Sample[], lr: number): number {
    const order = this.rng.shuffle(set.map((_, i) => i));
    let total = 0;
    for (const i of order) total += this.trainOne(set[i].img, set[i].label, lr);
    this.epoch++;
    const loss = total / Math.max(set.length, 1);
    this.lossHistory.push(loss);
    if (this.lossHistory.length > 300) this.lossHistory.shift();
    return loss;
  }

  accuracy(set: Sample[]): number {
    let ok = 0;
    for (const s of set) {
      const p = this.forward(s.img);
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      if (best === s.label) ok++;
    }
    return ok / Math.max(set.length, 1);
  }

  paramCount(): number {
    return this.nFilters * (K * K + 1) + this.nClasses * (this.nFilters * POOL_OUT * POOL_OUT + 1);
  }
}
