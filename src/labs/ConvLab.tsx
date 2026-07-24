/**
 * Convolution lab — a real CNN training on simple shapes, with the learned
 * filters and every intermediate feature map drawn live.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ConvNet, makeShapeSet, drawShape, SHAPES, IMG, zeros,
  type Shape, type Grid,
} from '../lib/conv.ts';
import { RNG } from '../lib/rng.ts';
import { GridView } from '../components/GridView.tsx';
import { LossChart } from '../components/Charts.tsx';

export function ConvLab() {
  const [nFilters, setNFilters] = useState(4);
  const [lr, setLr] = useState(0.01);
  const [seed, setSeed] = useState(3);
  const [nTrain, setNTrain] = useState(120);
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const [probeShape, setProbeShape] = useState<Shape>('cross');
  const [drawn, setDrawn] = useState<Grid | null>(null);

  const train = useMemo(() => makeShapeSet(nTrain, seed), [nTrain, seed]);
  const test = useMemo(() => makeShapeSet(60, seed + 500), [seed]);

  const netRef = useRef<ConvNet | null>(null);
  if (!netRef.current) netRef.current = new ConvNet(nFilters, SHAPES.length, seed);

  const rebuild = useCallback(() => {
    netRef.current = new ConvNet(nFilters, SHAPES.length, seed);
    setTick(t => t + 1);
  }, [nFilters, seed]);

  useEffect(() => { setRunning(false); rebuild(); }, [rebuild]);

  const net = netRef.current!;

  const step = useCallback(() => {
    netRef.current!.trainEpoch(train, lr);
    setTick(t => t + 1);
  }, [train, lr]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 60);
    return () => clearInterval(id);
  }, [running, step]);

  /* the image currently pushed through the network for display */
  const probeImg = useMemo(() => {
    if (drawn) return drawn;
    return drawShape(probeShape, new RNG(99), false);
  }, [drawn, probeShape, seed]);

  // run it so the maps below reflect this image
  net.forward(probeImg);
  const probs = net.probs.slice();
  let predicted = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[predicted]) predicted = i;

  const trainAcc = useMemo(() => net.accuracy(train), [net, train, tick]);
  const testAcc = useMemo(() => net.accuracy(test), [net, test, tick]);
  // forward again — accuracy() left the network on the last test image
  net.forward(probeImg);

  /* click-to-draw canvas */
  const paint = (r: number, c: number, erase: boolean) => {
    setDrawn(prev => {
      const g = prev ? prev.map(row => row.slice()) : zeros(IMG, IMG);
      if (r >= 0 && r < IMG && c >= 0 && c < IMG) g[r][c] = erase ? 0 : 1;
      return g;
    });
  };

  const drawRef = useRef<HTMLCanvasElement>(null);
  const CELL = 18;
  useEffect(() => {
    const cv = drawRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = IMG * CELL * dpr; cv.height = IMG * CELL * dpr;
    cv.style.width = IMG * CELL + 'px'; cv.style.height = IMG * CELL + 'px';
    const g = cv.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (let r = 0; r < IMG; r++) for (let c = 0; c < IMG; c++) {
      const v = probeImg[r][c];
      g.fillStyle = `rgba(236,235,228,${0.04 + v * 0.96})`;
      g.fillRect(c * CELL, r * CELL, CELL, CELL);
      g.strokeStyle = 'rgba(50,50,44,.7)'; g.lineWidth = 1;
      g.strokeRect(c * CELL + 0.5, r * CELL + 0.5, CELL - 1, CELL - 1);
    }
  }, [probeImg]);

  const onCanvas = (e: React.MouseEvent<HTMLCanvasElement>, force?: boolean) => {
    if (!force && e.buttons !== 1 && e.buttons !== 2) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / CELL);
    const r = Math.floor((e.clientY - rect.top) / CELL);
    paint(r, c, e.buttons === 2 || e.shiftKey);
  };

  return (
    <div className="lab">
      <aside className="side">
        <div className="grp">
          <h4>Network</h4>
          <div className="row"><label>Filters</label><span className="val">{nFilters}</span></div>
          <input type="range" min={2} max={8} step={1} value={nFilters}
                 onChange={e => setNFilters(+e.target.value)} />
          <p className="hint">
            Each filter is a 3×3 patch of weights. Every one starts as random noise and has to
            become an edge detector on its own.
          </p>
        </div>

        <div className="grp">
          <h4>Training</h4>
          <div className="row"><label>Learning rate</label><span className="val">{lr.toFixed(4)}</span></div>
          <input type="range" min={0.001} max={0.06} step={0.001} value={lr}
                 onChange={e => setLr(+e.target.value)} />
          <div className="row"><label>Train images</label><span className="val">{nTrain}</span></div>
          <input type="range" min={40} max={280} step={20} value={nTrain}
                 onChange={e => setNTrain(+e.target.value)} />
          <div className="row"><label>Seed</label><span className="val">{seed}</span></div>
          <input type="range" min={1} max={40} step={1} value={seed}
                 onChange={e => setSeed(+e.target.value)} />
        </div>

        <div className="grp">
          <h4>Probe image</h4>
          <select value={probeShape}
                  onChange={e => { setProbeShape(e.target.value as Shape); setDrawn(null); }}>
            {SHAPES.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <div className="btnrow" style={{ marginTop: 6 }}>
            <button className="btn" onClick={() => setDrawn(zeros(IMG, IMG))}>Blank</button>
            <button className="btn" onClick={() => setDrawn(null)}>Reset</button>
          </div>
          <p className="hint" style={{ marginTop: 7 }}>
            Draw on the grid to feed your own image in. Left click paints, shift-click erases.
          </p>
        </div>

        <div className="grp">
          <div className="btnrow">
            <button className="btn primary" onClick={() => setRunning(r => !r)}>
              {running ? '❚❚ Pause' : '▶ Train'}
            </button>
            <button className="btn" onClick={step} disabled={running}>Epoch</button>
          </div>
          <div className="btnrow" style={{ marginTop: 6 }}>
            <button className="btn" onClick={() => { setRunning(false); rebuild(); }}>Reset filters</button>
          </div>
        </div>
      </aside>

      <main className="stage">
        <div className="stats">
          <div className="stat"><span>Epoch</span><b>{net.epoch}</b></div>
          <div className="stat"><span>Loss</span><b>{(net.lossHistory.at(-1) ?? 0).toFixed(4)}</b></div>
          <div className="stat"><span>Train acc</span><b>{(trainAcc * 100).toFixed(1)}%</b></div>
          <div className="stat"><span>Test acc</span>
            <b style={{ color: testAcc > 0.85 ? 'var(--ok)' : undefined }}>{(testAcc * 100).toFixed(1)}%</b></div>
          <div className="stat"><span>Parameters</span><b>{net.paramCount()}</b></div>
          <div className="stat"><span>Prediction</span>
            <b style={{ color: 'var(--accent)' }}>{SHAPES[predicted]}</b></div>
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <h3>Learned filters — 3×3 each</h3>
          <p className="sub">
            These start as random noise. As training runs they organise into edge and orientation
            detectors, because that is what separates the shapes. Orange is a positive weight,
            blue negative.
          </p>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            {net.filters.map((f, i) => (
              <GridView key={i} grid={f} cell={22} mode="signed" gap
                        label={`filter ${i + 1}  b=${net.filterBias[i].toFixed(2)}`} />
            ))}
          </div>
        </div>

        <div className="panels">
          <div className="card">
            <h3>Input</h3>
            <p className="sub">12×12. Draw on it to test the network on something it has never seen.</p>
            <canvas
              ref={drawRef}
              onMouseDown={e => onCanvas(e, true)}
              onMouseMove={e => onCanvas(e)}
              onContextMenu={e => e.preventDefault()}
              style={{ cursor: 'crosshair' }}
            />
            <div style={{ marginTop: 12 }}>
              <h3 style={{ fontSize: 12.5, margin: '0 0 6px' }}>Output probabilities</h3>
              {SHAPES.map((s, i) => (
                <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                  <span style={{ width: 74, fontSize: 11.5, color: i === predicted ? '#d97a45' : '#a9a69c' }}>{s}</span>
                  <div style={{ flex: 1, height: 10, background: '#232320', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{
                      width: `${(probs[i] ?? 0) * 100}%`, height: '100%',
                      background: i === predicted ? '#d97a45' : '#5b8dbe',
                    }} />
                  </div>
                  <span style={{ width: 42, textAlign: 'right', fontSize: 11, fontFamily: 'ui-monospace, monospace' }}>
                    {((probs[i] ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gap: 16 }}>
            <div className="card">
              <h3>Loss</h3>
              <p className="sub">Cross-entropy, averaged over the training set each epoch.</p>
              <LossChart train={net.lossHistory} test={[]} />
            </div>
          </div>
        </div>

        <div className="card" style={{ marginTop: 16 }}>
          <h3>Feature maps — what each filter found in this image</h3>
          <p className="sub">
            Left is the convolution output after ReLU (10×10): bright where that filter matched.
            Right is after 2×2 max pooling (5×5) — the same information, coarser, which is what the
            classifier actually reads.
          </p>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {net.reluMaps.map((m, i) => (
              <div key={i} style={{ display: 'flex', gap: 7, alignItems: 'flex-start' }}>
                <GridView grid={m} cell={11} mode="mono" label={`relu ${i + 1}`} />
                <GridView grid={net.poolMaps[i]} cell={11} mode="mono" label={`pool ${i + 1}`} />
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
