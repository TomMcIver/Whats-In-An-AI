/**
 * A single-layer, multi-head self-attention model, with backprop running
 * all the way through the attention softmax.
 *
 *   tokens -> embedding + positional embedding
 *          -> per head:  Q,K,V projections
 *                        scores = QKᵀ / sqrt(dHead), causal-masked
 *                        attn   = softmax(scores)
 *                        head   = attn · V
 *          -> concat heads -> output projection -> logits over vocab
 *
 * Trained on a copy task: a random prefix, a separator, then the same
 * prefix again. To predict the second half the model has to look back at
 * the matching position in the first half — which is exactly the pattern
 * that shows up in the attention matrix once it has learned.
 */

import { RNG } from './rng.ts';

export const LETTERS = ['a', 'b', 'c', 'd'] as const;
export const SEP = '|';
export const VOCAB = [...LETTERS, SEP];
export const V = VOCAB.length;

/** prefix length; full sequence is PREFIX + 1 separator + PREFIX */
export const PREFIX = 4;
export const SEQ = PREFIX * 2 + 1;

export function makeSequence(rng: RNG): number[] {
  const pre: number[] = [];
  for (let i = 0; i < PREFIX; i++) pre.push(Math.floor(rng.next() * LETTERS.length));
  return [...pre, LETTERS.length, ...pre];   // LETTERS.length is the SEP id
}

export function tokensToString(t: number[]): string {
  return t.map(i => VOCAB[i]).join(' ');
}

/* ---------- small matrix helpers ---------- */

function mat(r: number, c: number, fill = 0): number[][] {
  return Array.from({ length: r }, () => new Array(c).fill(fill));
}
function randMat(r: number, c: number, rng: RNG, sd: number): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => rng.normal(0, sd)));
}

export interface HeadState {
  /** [SEQ][SEQ] attention weights from the last forward pass */
  attn: number[][];
  scores: number[][];
  Q: number[][];
  Kk: number[][];
  Vv: number[][];
  out: number[][];
}

export class AttentionModel {
  d: number;          // model dim
  H: number;          // heads
  dh: number;         // per-head dim
  rng: RNG;

  emb: number[][] = [];      // [V][d]
  pos: number[][] = [];      // [SEQ][d]
  Wq: number[][][] = [];     // [H][d][dh]
  Wk: number[][][] = [];
  Wv: number[][][] = [];
  Wo: number[][] = [];       // [H*dh][d]
  Wout: number[][] = [];     // [d][V]
  bout: number[] = [];

  /* retained from the last forward */
  lastTokens: number[] = [];
  x: number[][] = [];        // [SEQ][d] embeddings
  heads: HeadState[] = [];
  concat: number[][] = [];   // [SEQ][H*dh]
  hidden: number[][] = [];   // [SEQ][d] after output projection
  logits: number[][] = [];   // [SEQ][V]
  probs: number[][] = [];

  epoch = 0;
  lossHistory: number[] = [];

  constructor(d = 24, H = 2, seed = 4) {
    this.d = d;
    this.H = H;
    this.dh = Math.floor(d / H);
    this.rng = new RNG(seed);
    this.init();
  }

  init() {
    const { d, H, dh, rng } = this;
    const s = 1 / Math.sqrt(d);
    this.emb = randMat(V, d, rng, s);
    this.pos = randMat(SEQ, d, rng, s);
    this.Wq = Array.from({ length: H }, () => randMat(d, dh, rng, s));
    this.Wk = Array.from({ length: H }, () => randMat(d, dh, rng, s));
    this.Wv = Array.from({ length: H }, () => randMat(d, dh, rng, s));
    this.Wo = randMat(H * dh, d, rng, s);
    this.Wout = randMat(d, V, rng, s);
    this.bout = new Array(V).fill(0);
    this.epoch = 0;
    this.lossHistory = [];
  }

  reset(seed?: number) {
    if (seed !== undefined) this.rng.reseed(seed);
    this.init();
  }

  forward(tokens: number[]): number[][] {
    const { d, H, dh } = this;
    const T = tokens.length;
    this.lastTokens = tokens;

    // embeddings + positions
    this.x = mat(T, d);
    for (let t = 0; t < T; t++)
      for (let i = 0; i < d; i++)
        this.x[t][i] = this.emb[tokens[t]][i] + this.pos[t][i];

    this.heads = [];
    this.concat = mat(T, H * dh);

    for (let h = 0; h < H; h++) {
      const Q = mat(T, dh), Kk = mat(T, dh), Vv = mat(T, dh);
      for (let t = 0; t < T; t++)
        for (let j = 0; j < dh; j++) {
          let q = 0, k = 0, v = 0;
          for (let i = 0; i < d; i++) {
            q += this.x[t][i] * this.Wq[h][i][j];
            k += this.x[t][i] * this.Wk[h][i][j];
            v += this.x[t][i] * this.Wv[h][i][j];
          }
          Q[t][j] = q; Kk[t][j] = k; Vv[t][j] = v;
        }

      const scale = 1 / Math.sqrt(dh);
      const scores = mat(T, T);
      const attn = mat(T, T);
      for (let t = 0; t < T; t++) {
        let mx = -Infinity;
        for (let u = 0; u <= t; u++) {          // causal: only look backwards
          let s = 0;
          for (let j = 0; j < dh; j++) s += Q[t][j] * Kk[u][j];
          scores[t][u] = s * scale;
          if (scores[t][u] > mx) mx = scores[t][u];
        }
        let sum = 0;
        for (let u = 0; u <= t; u++) { attn[t][u] = Math.exp(scores[t][u] - mx); sum += attn[t][u]; }
        for (let u = 0; u <= t; u++) attn[t][u] /= sum;
      }

      const out = mat(T, dh);
      for (let t = 0; t < T; t++)
        for (let j = 0; j < dh; j++) {
          let s = 0;
          for (let u = 0; u <= t; u++) s += attn[t][u] * Vv[u][j];
          out[t][j] = s;
          this.concat[t][h * dh + j] = s;
        }

      this.heads.push({ attn, scores, Q, Kk, Vv, out });
    }

    // output projection then vocab logits
    this.hidden = mat(T, d);
    for (let t = 0; t < T; t++)
      for (let i = 0; i < d; i++) {
        let s = 0;
        for (let c = 0; c < H * dh; c++) s += this.concat[t][c] * this.Wo[c][i];
        this.hidden[t][i] = s + this.x[t][i];      // residual connection
      }

    this.logits = mat(T, V);
    this.probs = mat(T, V);
    for (let t = 0; t < T; t++) {
      let mx = -Infinity;
      for (let vv = 0; vv < V; vv++) {
        let s = this.bout[vv];
        for (let i = 0; i < this.d; i++) s += this.hidden[t][i] * this.Wout[i][vv];
        this.logits[t][vv] = s;
        if (s > mx) mx = s;
      }
      let sum = 0;
      for (let vv = 0; vv < V; vv++) { this.probs[t][vv] = Math.exp(this.logits[t][vv] - mx); sum += this.probs[t][vv]; }
      for (let vv = 0; vv < V; vv++) this.probs[t][vv] /= sum;
    }
    return this.probs;
  }

  /**
   * Train on one sequence. Loss is next-token prediction, scored only on
   * the copied half — the first half is unpredictable by construction.
   */
  trainOne(tokens: number[], lr: number): number {
    const { d, H, dh } = this;
    const T = tokens.length;
    this.forward(tokens);

    // gradient accumulators
    const dEmb = mat(V, d), dPos = mat(SEQ, d);
    const dWq = Array.from({ length: H }, () => mat(d, dh));
    const dWk = Array.from({ length: H }, () => mat(d, dh));
    const dWv = Array.from({ length: H }, () => mat(d, dh));
    const dWo = mat(H * dh, d);
    const dWout = mat(d, V);
    const dbout = new Array(V).fill(0);
    const dx = mat(T, d);
    const dHidden = mat(T, d);

    let loss = 0;
    let scored = 0;

    // ---- output layer ----
    for (let t = 0; t < T - 1; t++) {
      if (t < PREFIX) continue;                 // only score the copied half
      const target = tokens[t + 1];
      loss += -Math.log(Math.max(this.probs[t][target], 1e-12));
      scored++;

      const dLogit = this.probs[t].slice();
      dLogit[target] -= 1;

      for (let vv = 0; vv < V; vv++) {
        dbout[vv] += dLogit[vv];
        for (let i = 0; i < d; i++) {
          dWout[i][vv] += this.hidden[t][i] * dLogit[vv];
          dHidden[t][i] += this.Wout[i][vv] * dLogit[vv];
        }
      }
    }

    // ---- output projection + residual ----
    const dConcat = mat(T, H * dh);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < d; i++) {
        const g = dHidden[t][i];
        if (g === 0) continue;
        for (let c = 0; c < H * dh; c++) {
          dWo[c][i] += this.concat[t][c] * g;
          dConcat[t][c] += this.Wo[c][i] * g;
        }
        dx[t][i] += g;                          // residual path
      }
    }

    // ---- back through each head ----
    for (let h = 0; h < H; h++) {
      const st = this.heads[h];
      const scale = 1 / Math.sqrt(dh);
      const dQ = mat(T, dh), dK = mat(T, dh), dV = mat(T, dh);

      for (let t = 0; t < T; t++) {
        // dL/d(head out) for this position
        const dOut: number[] = [];
        for (let j = 0; j < dh; j++) dOut.push(dConcat[t][h * dh + j]);

        // through the weighted sum: context = sum_u attn[t][u] * V[u]
        const dAttn = new Array(T).fill(0);
        for (let u = 0; u <= t; u++) {
          let s = 0;
          for (let j = 0; j < dh; j++) {
            s += dOut[j] * st.Vv[u][j];
            dV[u][j] += st.attn[t][u] * dOut[j];
          }
          dAttn[u] = s;
        }

        // through softmax:  ds_j = a_j (da_j - sum_k a_k da_k)
        let dot = 0;
        for (let u = 0; u <= t; u++) dot += st.attn[t][u] * dAttn[u];
        for (let u = 0; u <= t; u++) {
          const dScore = st.attn[t][u] * (dAttn[u] - dot) * scale;
          for (let j = 0; j < dh; j++) {
            dQ[t][j] += dScore * st.Kk[u][j];
            dK[u][j] += dScore * st.Q[t][j];
          }
        }
      }

      // through the Q/K/V projections back to x
      for (let t = 0; t < T; t++)
        for (let j = 0; j < dh; j++) {
          const gq = dQ[t][j], gk = dK[t][j], gv = dV[t][j];
          for (let i = 0; i < d; i++) {
            dWq[h][i][j] += this.x[t][i] * gq;
            dWk[h][i][j] += this.x[t][i] * gk;
            dWv[h][i][j] += this.x[t][i] * gv;
            dx[t][i] += this.Wq[h][i][j] * gq + this.Wk[h][i][j] * gk + this.Wv[h][i][j] * gv;
          }
        }
    }

    // ---- into the embedding tables ----
    for (let t = 0; t < T; t++)
      for (let i = 0; i < d; i++) {
        dEmb[this.lastTokens[t]][i] += dx[t][i];
        dPos[t][i] += dx[t][i];
      }

    // ---- apply ----
    // Scale by the number of scored positions so the gradient corresponds to
    // the *mean* loss we report, not the sum. Without this the effective
    // learning rate would scale with sequence length.
    const step = lr / Math.max(scored, 1);
    const apply2 = (W: number[][], G: number[][]) => {
      for (let a = 0; a < W.length; a++)
        for (let b = 0; b < W[a].length; b++) W[a][b] -= step * G[a][b];
    };
    apply2(this.emb, dEmb);
    apply2(this.pos, dPos);
    for (let h = 0; h < H; h++) {
      apply2(this.Wq[h], dWq[h]);
      apply2(this.Wk[h], dWk[h]);
      apply2(this.Wv[h], dWv[h]);
    }
    apply2(this.Wo, dWo);
    apply2(this.Wout, dWout);
    for (let vv = 0; vv < V; vv++) this.bout[vv] -= step * dbout[vv];

    return loss / Math.max(scored, 1);
  }

  trainEpoch(batch: number[][], lr: number): number {
    let total = 0;
    for (const seq of batch) total += this.trainOne(seq, lr);
    this.epoch++;
    const l = total / Math.max(batch.length, 1);
    this.lossHistory.push(l);
    if (this.lossHistory.length > 300) this.lossHistory.shift();
    return l;
  }

  /** fraction of copied-half tokens predicted correctly */
  accuracy(batch: number[][]): number {
    let ok = 0, n = 0;
    for (const seq of batch) {
      this.forward(seq);
      for (let t = PREFIX; t < seq.length - 1; t++) {
        let best = 0;
        for (let vv = 1; vv < V; vv++) if (this.probs[t][vv] > this.probs[t][best]) best = vv;
        if (best === seq[t + 1]) ok++;
        n++;
      }
    }
    return ok / Math.max(n, 1);
  }

  paramCount(): number {
    const { d, H, dh } = this;
    return V * d + SEQ * d + H * 3 * d * dh + H * dh * d + d * V + V;
  }
}

export function makeBatch(n: number, seed: number): number[][] {
  const rng = new RNG(seed);
  return Array.from({ length: n }, () => makeSequence(rng));
}
