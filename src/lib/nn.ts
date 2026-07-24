/**
 * A multilayer perceptron written from scratch.
 *
 * No ML library. Every weight, bias, pre-activation, activation and
 * gradient is a plain number held on the layer object, so the UI can read
 * the network's internal state directly at any point during training.
 *
 * Forward:   z[l] = W[l] · a[l-1] + b[l]      a[l] = f(z[l])
 * Backward:  delta[L] = a[L] - y              (sigmoid+BCE and linear+MSE
 *                                              both reduce to this)
 *            delta[l] = (W[l+1]ᵀ · delta[l+1]) ⊙ f'(z[l])
 *            dW[l]    = delta[l] · a[l-1]ᵀ
 *            db[l]    = delta[l]
 */

import { RNG } from './rng.ts';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear';
export type Task = 'classification' | 'regression';

export interface Layer {
  nIn: number;
  nOut: number;
  /** weights[j][i] — from input i to unit j */
  w: number[][];
  b: number[];
  /** pre-activations from the most recent forward pass */
  z: number[];
  /** activations from the most recent forward pass */
  a: number[];
  /** accumulated gradients for the current batch */
  dw: number[][];
  db: number[];
  /** dL/dz for the most recent backward pass */
  delta: number[];
  act: Activation;
}

export function applyAct(v: number, act: Activation): number {
  switch (act) {
    case 'tanh': return Math.tanh(v);
    case 'relu': return v > 0 ? v : 0;
    case 'sigmoid': return 1 / (1 + Math.exp(-clamp(v, -60, 60)));
    case 'linear': return v;
  }
}

/** derivative expressed in terms of the activation output where possible */
function actPrime(z: number, a: number, act: Activation): number {
  switch (act) {
    case 'tanh': return 1 - a * a;
    case 'relu': return z > 0 ? 1 : 0;
    case 'sigmoid': return a * (1 - a);
    case 'linear': return 1;
  }
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

export interface MLPConfig {
  /** e.g. [2, 4, 4, 1] — input size, hidden sizes, output size */
  sizes: number[];
  hiddenAct?: Activation;
  task?: Task;
  seed?: number;
  /** L2 penalty applied on each step */
  weightDecay?: number;
}

export class MLP {
  layers: Layer[] = [];
  sizes: number[];
  hiddenAct: Activation;
  task: Task;
  weightDecay: number;
  rng: RNG;

  /** how many examples have been accumulated into the current gradient */
  private batchCount = 0;

  epoch = 0;
  stepCount = 0;

  constructor(cfg: MLPConfig) {
    this.sizes = cfg.sizes.slice();
    this.hiddenAct = cfg.hiddenAct ?? 'tanh';
    this.task = cfg.task ?? 'classification';
    this.weightDecay = cfg.weightDecay ?? 0;
    this.rng = new RNG(cfg.seed ?? 42);
    this.build();
  }

  private build() {
    this.layers = [];
    for (let l = 1; l < this.sizes.length; l++) {
      const nIn = this.sizes[l - 1];
      const nOut = this.sizes[l];
      const last = l === this.sizes.length - 1;
      const act: Activation = last
        ? (this.task === 'classification' ? 'sigmoid' : 'linear')
        : this.hiddenAct;

      // He scaling for ReLU, Xavier otherwise — keeps early signal alive
      const scale = act === 'relu' ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);

      const w: number[][] = [];
      const dw: number[][] = [];
      for (let j = 0; j < nOut; j++) {
        const row: number[] = [];
        const drow: number[] = [];
        for (let i = 0; i < nIn; i++) {
          row.push(this.rng.normal(0, scale));
          drow.push(0);
        }
        w.push(row);
        dw.push(drow);
      }

      this.layers.push({
        nIn, nOut, w, dw,
        b: new Array(nOut).fill(0),
        db: new Array(nOut).fill(0),
        z: new Array(nOut).fill(0),
        a: new Array(nOut).fill(0),
        delta: new Array(nOut).fill(0),
        act,
      });
    }
    this.epoch = 0;
    this.stepCount = 0;
    this.batchCount = 0;
  }

  reset(seed?: number) {
    if (seed !== undefined) this.rng.reseed(seed);
    this.build();
  }

  /** activations of the input layer for the most recent forward pass */
  lastInput: number[] = [];

  forward(x: number[]): number[] {
    this.lastInput = x;
    let a = x;
    for (const L of this.layers) {
      for (let j = 0; j < L.nOut; j++) {
        let sum = L.b[j];
        const row = L.w[j];
        for (let i = 0; i < L.nIn; i++) sum += row[i] * a[i];
        L.z[j] = sum;
        L.a[j] = applyAct(sum, L.act);
      }
      a = L.a;
    }
    return a;
  }

  /**
   * Backpropagate one example. Assumes forward(x) was just called.
   * Gradients accumulate — call step() to apply and clear them.
   */
  backward(target: number[]) {
    const last = this.layers[this.layers.length - 1];

    // sigmoid+BCE and linear+MSE both give this exact simplification
    for (let j = 0; j < last.nOut; j++) last.delta[j] = last.a[j] - target[j];

    for (let l = this.layers.length - 2; l >= 0; l--) {
      const L = this.layers[l];
      const next = this.layers[l + 1];
      for (let j = 0; j < L.nOut; j++) {
        let sum = 0;
        for (let k = 0; k < next.nOut; k++) sum += next.w[k][j] * next.delta[k];
        L.delta[j] = sum * actPrime(L.z[j], L.a[j], L.act);
      }
    }

    for (let l = 0; l < this.layers.length; l++) {
      const L = this.layers[l];
      const prev = l === 0 ? this.lastInput : this.layers[l - 1].a;
      for (let j = 0; j < L.nOut; j++) {
        const d = L.delta[j];
        const drow = L.dw[j];
        for (let i = 0; i < L.nIn; i++) drow[i] += d * prev[i];
        L.db[j] += d;
      }
    }
    this.batchCount++;
  }

  /** apply accumulated gradients, averaged over the batch, then clear */
  step(lr: number) {
    const n = Math.max(this.batchCount, 1);
    for (const L of this.layers) {
      for (let j = 0; j < L.nOut; j++) {
        const drow = L.dw[j];
        const row = L.w[j];
        for (let i = 0; i < L.nIn; i++) {
          row[i] -= lr * (drow[i] / n + this.weightDecay * row[i]);
          drow[i] = 0;
        }
        L.b[j] -= lr * (L.db[j] / n);
        L.db[j] = 0;
      }
    }
    this.batchCount = 0;
    this.stepCount++;
  }

  /** loss for a single already-forwarded example */
  private lossOne(out: number[], target: number[]): number {
    let s = 0;
    if (this.task === 'classification') {
      for (let i = 0; i < out.length; i++) {
        const p = clamp(out[i], 1e-9, 1 - 1e-9);
        s += -(target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p));
      }
    } else {
      for (let i = 0; i < out.length; i++) {
        const d = out[i] - target[i];
        s += 0.5 * d * d;
      }
    }
    return s;
  }

  /** one pass over the data in mini-batches; returns mean training loss */
  trainEpoch(xs: number[][], ys: number[][], lr: number, batchSize = 10): number {
    const idx = this.rng.shuffle(xs.map((_, i) => i));
    let total = 0;
    let sinceStep = 0;

    for (const i of idx) {
      const out = this.forward(xs[i]);
      total += this.lossOne(out, ys[i]);
      this.backward(ys[i]);
      if (++sinceStep >= batchSize) {
        this.step(lr);
        sinceStep = 0;
      }
    }
    if (sinceStep > 0) this.step(lr);

    this.epoch++;
    return total / xs.length;
  }

  /** mean loss over a set, without touching any weights */
  evaluate(xs: number[][], ys: number[][]): number {
    let total = 0;
    for (let i = 0; i < xs.length; i++) {
      total += this.lossOne(this.forward(xs[i]), ys[i]);
    }
    return total / Math.max(xs.length, 1);
  }

  /** fraction correct, classification only (threshold 0.5) */
  accuracy(xs: number[][], ys: number[][]): number {
    let ok = 0;
    for (let i = 0; i < xs.length; i++) {
      const out = this.forward(xs[i]);
      if ((out[0] >= 0.5 ? 1 : 0) === (ys[i][0] >= 0.5 ? 1 : 0)) ok++;
    }
    return ok / Math.max(xs.length, 1);
  }

  /** largest |w| in the network — used to scale edge thickness in the UI */
  maxAbsWeight(): number {
    let m = 1e-6;
    for (const L of this.layers)
      for (const row of L.w)
        for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }

  /** total learnable parameters */
  paramCount(): number {
    return this.layers.reduce((n, L) => n + L.nOut * L.nIn + L.nOut, 0);
  }
}
