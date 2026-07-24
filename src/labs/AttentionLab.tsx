/**
 * Attention lab — a one-layer multi-head transformer learning a copy task,
 * with the attention matrix for every head drawn live as it trains.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AttentionModel, makeBatch, makeSequence, tokensToString,
  VOCAB, V, PREFIX, SEQ,
} from '../lib/attention.ts';
import { RNG } from '../lib/rng.ts';
import { LossChart } from '../components/Charts.tsx';

const CELL = 30;

function AttnMatrix({ attn, tokens, title }: { attn: number[][]; tokens: number[]; title: string }) {
  const T = tokens.length;
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const size = T * CELL;
    c.width = size * dpr; c.height = size * dpr;
    c.style.width = size + 'px'; c.style.height = size + 'px';
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);

    for (let t = 0; t < T; t++) {
      for (let u = 0; u < T; u++) {
        const v = u <= t ? (attn[t]?.[u] ?? 0) : -1;
        if (v < 0) {
          g.fillStyle = 'rgba(30,30,26,.85)';    // masked out by causality
        } else {
          g.fillStyle = `rgba(217,122,69,${0.04 + v * 0.96})`;
        }
        g.fillRect(u * CELL, t * CELL, CELL - 1, CELL - 1);
        if (v > 0.35) {
          g.fillStyle = v > 0.6 ? '#1a1108' : '#e8e4dc';
          g.font = '9px ui-monospace, monospace';
          g.textAlign = 'center';
          g.fillText(v.toFixed(2).slice(1), u * CELL + CELL / 2 - 0.5, t * CELL + CELL / 2 + 3);
        }
      }
    }
    g.strokeStyle = '#32322c'; g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, size - 1, size - 1);
  }, [attn, tokens, T]);

  return (
    <div>
      <div style={{ fontSize: 11.5, color: '#a9a69c', marginBottom: 6 }}>{title}</div>
      <div style={{ display: 'flex' }}>
        {/* row labels = the querying position */}
        <div style={{ display: 'flex', flexDirection: 'column', marginRight: 4, marginTop: 0 }}>
          {tokens.map((tok, i) => (
            <div key={i} style={{
              height: CELL, lineHeight: CELL + 'px', fontSize: 10,
              fontFamily: 'ui-monospace, monospace',
              color: i >= PREFIX ? '#d97a45' : '#74716a', textAlign: 'right', width: 26,
            }}>{VOCAB[tok]}{i}</div>
          ))}
        </div>
        <div>
          <canvas ref={ref} />
          {/* column labels = the position being attended to */}
          <div style={{ display: 'flex' }}>
            {tokens.map((tok, i) => (
              <div key={i} style={{
                width: CELL, fontSize: 10, fontFamily: 'ui-monospace, monospace',
                color: '#74716a', textAlign: 'center', paddingTop: 2,
              }}>{VOCAB[tok]}{i}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function AttentionLab() {
  const [heads, setHeads] = useState(2);
  const [dim, setDim] = useState(24);
  const [lr, setLr] = useState(0.05);
  const [seed, setSeed] = useState(4);
  const [batchN, setBatchN] = useState(64);
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const [probeSeed, setProbeSeed] = useState(31);

  const train = useMemo(() => makeBatch(batchN, seed), [batchN, seed]);
  const test = useMemo(() => makeBatch(48, seed + 800), [seed]);

  const mRef = useRef<AttentionModel | null>(null);
  if (!mRef.current) mRef.current = new AttentionModel(dim, heads, seed);

  const rebuild = useCallback(() => {
    mRef.current = new AttentionModel(dim, heads, seed);
    setTick(t => t + 1);
  }, [dim, heads, seed]);

  useEffect(() => { setRunning(false); rebuild(); }, [rebuild]);

  const m = mRef.current!;

  const step = useCallback(() => {
    mRef.current!.trainEpoch(train, lr);
    setTick(t => t + 1);
  }, [train, lr]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 30);
    return () => clearInterval(id);
  }, [running, step]);

  const probe = useMemo(() => makeSequence(new RNG(probeSeed)), [probeSeed]);

  const trainAcc = useMemo(() => m.accuracy(train), [m, train, tick]);
  const testAcc = useMemo(() => m.accuracy(test), [m, test, tick]);

  // forward the probe last so the attention matrices shown belong to it
  m.forward(probe);
  const probs = m.probs.map(r => r.slice());

  return (
    <div className="lab">
      <aside className="side">
        <div className="grp">
          <h4>The task</h4>
          <p className="hint">
            A random prefix, a separator, then the same prefix again:
            <br />
            <code style={{ color: '#d97a45', fontFamily: 'ui-monospace, monospace' }}>
              {tokensToString(probe)}
            </code>
            <br /><br />
            The model predicts each token of the second half. It cannot guess — it has to look back
            at the matching position in the first half. That is what attention is for.
          </p>
        </div>

        <div className="grp">
          <h4>Model</h4>
          <div className="row"><label>Heads</label><span className="val">{heads}</span></div>
          <input type="range" min={1} max={4} step={1} value={heads}
                 onChange={e => setHeads(+e.target.value)} />
          <div className="row"><label>Model dim</label><span className="val">{dim}</span></div>
          <input type="range" min={8} max={40} step={4} value={dim}
                 onChange={e => setDim(+e.target.value)} />
        </div>

        <div className="grp">
          <h4>Training</h4>
          <div className="row"><label>Learning rate</label><span className="val">{lr.toFixed(3)}</span></div>
          <input type="range" min={0.005} max={0.2} step={0.005} value={lr}
                 onChange={e => setLr(+e.target.value)} />
          <div className="row"><label>Sequences</label><span className="val">{batchN}</span></div>
          <input type="range" min={16} max={128} step={8} value={batchN}
                 onChange={e => setBatchN(+e.target.value)} />
          <div className="row"><label>Seed</label><span className="val">{seed}</span></div>
          <input type="range" min={1} max={40} step={1} value={seed}
                 onChange={e => setSeed(+e.target.value)} />
        </div>

        <div className="grp">
          <h4>Probe sequence</h4>
          <div className="row"><label>Which one</label><span className="val">{probeSeed}</span></div>
          <input type="range" min={1} max={99} step={1} value={probeSeed}
                 onChange={e => setProbeSeed(+e.target.value)} />
          <p className="hint">Any sequence, including ones never trained on.</p>
        </div>

        <div className="grp">
          <div className="btnrow">
            <button className="btn primary" onClick={() => setRunning(r => !r)}>
              {running ? '❚❚ Pause' : '▶ Train'}
            </button>
            <button className="btn" onClick={step} disabled={running}>Epoch</button>
          </div>
          <div className="btnrow" style={{ marginTop: 6 }}>
            <button className="btn" onClick={() => { setRunning(false); rebuild(); }}>Reset weights</button>
          </div>
        </div>
      </aside>

      <main className="stage">
        <div className="stats">
          <div className="stat"><span>Epoch</span><b>{m.epoch}</b></div>
          <div className="stat"><span>Loss</span><b>{(m.lossHistory.at(-1) ?? 0).toFixed(4)}</b></div>
          <div className="stat"><span>Train acc</span><b>{(trainAcc * 100).toFixed(1)}%</b></div>
          <div className="stat"><span>Test acc</span>
            <b style={{ color: testAcc > 0.85 ? 'var(--ok)' : undefined }}>{(testAcc * 100).toFixed(1)}%</b></div>
          <div className="stat"><span>Chance</span><b>{(100 / V).toFixed(0)}%</b></div>
          <div className="stat"><span>Parameters</span><b>{m.paramCount()}</b></div>
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <h3>Attention — who looks at whom</h3>
          <p className="sub">
            Each row is a position asking a question; each column is a position it can read.
            The upper-right triangle is greyed out because a token may never see the future.
            Before training this is a smear. After training, the orange cells line up on a diagonal
            offset by {PREFIX} — the model has found the matching position.
          </p>
          <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap' }}>
            {m.heads.map((h, i) => (
              <AttnMatrix key={i} attn={h.attn} tokens={probe} title={`head ${i + 1}`} />
            ))}
          </div>
        </div>

        <div className="panels">
          <div className="card">
            <h3>Predictions</h3>
            <p className="sub">
              What the model expects next at each position of the copied half. Green means it
              matched the true next token.
            </p>
            {Array.from({ length: SEQ - 1 }, (_, t) => t)
              .filter(t => t >= PREFIX)
              .map(t => {
                const target = probe[t + 1];
                let best = 0;
                for (let vv = 1; vv < V; vv++) if (probs[t][vv] > probs[t][best]) best = vv;
                const ok = best === target;
                return (
                  <div key={t} style={{ marginBottom: 9 }}>
                    <div style={{ fontSize: 11.5, color: '#a9a69c', marginBottom: 3 }}>
                      after position {t} (<code style={{ color: '#d97a45' }}>{VOCAB[probe[t]]}</code>)
                      {' → true next is '}
                      <code style={{ color: ok ? '#8fae62' : '#d97a45' }}>{VOCAB[target]}</code>
                      {ok ? ' ✓' : ' ✗'}
                    </div>
                    <div style={{ display: 'flex', gap: 4 }}>
                      {VOCAB.map((tokName, vv) => (
                        <div key={vv} style={{ flex: 1, textAlign: 'center' }}>
                          <div style={{
                            height: 34, background: '#232320', borderRadius: 3,
                            display: 'flex', alignItems: 'flex-end', overflow: 'hidden',
                          }}>
                            <div style={{
                              width: '100%', height: `${(probs[t][vv] ?? 0) * 100}%`,
                              background: vv === target ? '#8fae62' : '#5b8dbe',
                            }} />
                          </div>
                          <div style={{ fontSize: 10, color: '#74716a', fontFamily: 'ui-monospace, monospace' }}>
                            {tokName}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
          </div>

          <div style={{ display: 'grid', gap: 16 }}>
            <div className="card">
              <h3>Loss</h3>
              <p className="sub">Cross-entropy on the copied half only — the first half is random by construction and cannot be predicted.</p>
              <LossChart train={m.lossHistory} test={[]} />
            </div>

            <div className="card">
              <h3>What to notice</h3>
              <p className="sub" style={{ marginBottom: 0 }}>
                Reset the weights and look at the attention matrix: every row is roughly uniform over
                the past, because the model has no idea what matters. Train it and watch a clean
                diagonal stripe appear, offset by exactly {PREFIX} positions. Nobody told it to do
                that — it is the only strategy that reduces the loss. Then drag the probe slider to a
                sequence it has never seen; the same stripe holds, which is what generalisation
                looks like.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
