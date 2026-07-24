/**
 * Neural network trainer.
 *
 * A real MLP training in the browser, one epoch per animation frame, with
 * the network's internal state drawn live: weights as edges, biases as
 * node rings, activations as node fill, plus the decision boundary and
 * loss curve redrawn every epoch.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MLP, type Activation } from '../lib/nn.ts';
import {
  generate, featurise, splitTrainTest, DATASETS, FEATURES,
  type DatasetKind, type FeatureId, type Point,
} from '../lib/datasets.ts';
import { NetworkGraph } from '../components/NetworkGraph.tsx';
import { LossChart, BoundaryChart } from '../components/Charts.tsx';

export function NeuralNetLab() {
  /* ---------- configuration ---------- */
  const [dataset, setDataset] = useState<DatasetKind>('circle');
  const [noise, setNoise] = useState(0.06);
  const [nSamples, setNSamples] = useState(240);
  const [hidden, setHidden] = useState<number[]>([4, 4]);
  const [features, setFeatures] = useState<FeatureId[]>(['x', 'y']);
  const [lr, setLr] = useState(0.3);
  const [act, setAct] = useState<Activation>('tanh');
  const [batchSize, setBatchSize] = useState(10);
  const [seed, setSeed] = useState(11);

  /* ---------- runtime ---------- */
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);            // forces redraws
  const [trainLoss, setTrainLoss] = useState<number[]>([]);
  const [testLoss, setTestLoss] = useState<number[]>([]);
  const [selected, setSelected] = useState<{ layer: number; unit: number } | null>(null);
  const [probe, setProbe] = useState({ x: 0.45, y: 0.45 });

  const data = useMemo(
    () => generate(dataset, nSamples, noise, seed),
    [dataset, nSamples, noise, seed]
  );
  const split = useMemo(() => splitTrainTest(data, 0.7, seed + 1), [data, seed]);

  const sizes = useMemo(() => [features.length, ...hidden, 1], [features, hidden]);

  const netRef = useRef<MLP | null>(null);
  if (!netRef.current) netRef.current = new MLP({ sizes, hiddenAct: act, seed });
  const net = netRef.current;

  /** rebuild the network whenever its shape or activation changes */
  const rebuild = useCallback(() => {
    netRef.current = new MLP({ sizes, hiddenAct: act, task: 'classification', seed });
    setTrainLoss([]); setTestLoss([]); setSelected(null); setTick(t => t + 1);
  }, [sizes, act, seed]);

  useEffect(() => { rebuild(); }, [rebuild]);
  useEffect(() => { setRunning(false); }, [dataset, nSamples, noise]);

  const xy = useMemo(() => {
    const enc = (pts: Point[]) => ({
      xs: pts.map(p => featurise(p.x, p.y, features)),
      ys: pts.map(p => [p.label] as number[]),
    });
    return { train: enc(split.train), test: enc(split.test) };
  }, [split, features]);

  /* ---------- training loop ---------- */

  const stepOnce = useCallback(() => {
    const n = netRef.current!;
    const tl = n.trainEpoch(xy.train.xs, xy.train.ys, lr, batchSize);
    const vl = n.evaluate(xy.test.xs, xy.test.ys);
    setTrainLoss(a => [...a.slice(-299), tl]);
    setTestLoss(a => [...a.slice(-299), vl]);
    setTick(t => t + 1);
  }, [xy, lr, batchSize]);

  // A timer rather than requestAnimationFrame: rAF is throttled to zero in
  // backgrounded tabs and never fires in headless browsers, which makes the
  // training loop both unverifiable and liable to silently stall.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(stepOnce, 16);
    return () => clearInterval(id);
  }, [running, stepOnce]);

  /* ---------- derived readouts ---------- */

  const trainAcc = net.accuracy(xy.train.xs, xy.train.ys);
  const testAcc = net.accuracy(xy.test.xs, xy.test.ys);
  const probeVec = featurise(probe.x, probe.y, features);
  net.forward(probeVec);   // leaves activations set for the graph
  const probeOut = net.layers[net.layers.length - 1].a[0];

  const toggleFeature = (id: FeatureId) => {
    setFeatures(f => {
      if (f.includes(id)) return f.length > 1 ? f.filter(v => v !== id) : f;
      return [...FEATURES.map(x => x.id).filter(x => f.includes(x) || x === id)];
    });
  };

  const setLayerCount = (n: number) => {
    setHidden(h => {
      const out = h.slice(0, n);
      while (out.length < n) out.push(4);
      return out;
    });
  };
  const setLayerSize = (i: number, n: number) =>
    setHidden(h => h.map((v, j) => (j === i ? n : v)));

  const dsNote = DATASETS.find(d => d.id === dataset)?.note ?? '';

  return (
    <div className="lab">
      <aside className="side">
        <div className="grp">
          <h4>Data</h4>
          <select value={dataset} onChange={e => setDataset(e.target.value as DatasetKind)}>
            {DATASETS.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <p className="hint" style={{ marginTop: 7 }}>{dsNote}</p>
          <div className="row" style={{ marginTop: 10 }}>
            <label>Samples</label><span className="val">{nSamples}</span>
          </div>
          <input type="range" min={60} max={500} step={20} value={nSamples}
                 onChange={e => setNSamples(+e.target.value)} />
          <div className="row"><label>Noise</label><span className="val">{noise.toFixed(2)}</span></div>
          <input type="range" min={0} max={0.35} step={0.01} value={noise}
                 onChange={e => setNoise(+e.target.value)} />
        </div>

        <div className="grp">
          <h4>Input features</h4>
          <div className="chips">
            {FEATURES.map(f => (
              <button key={f.id}
                      className={'chip' + (features.includes(f.id) ? ' on' : '')}
                      onClick={() => toggleFeature(f.id)}>
                {f.label}
              </button>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 8 }}>
            Adding X₁² or X₁X₂ lets a shallower net solve harder shapes — features and depth
            substitute for each other.
          </p>
        </div>

        <div className="grp">
          <h4>Architecture</h4>
          <div className="row"><label>Hidden layers</label><span className="val">{hidden.length}</span></div>
          <input type="range" min={0} max={4} step={1} value={hidden.length}
                 onChange={e => setLayerCount(+e.target.value)} />
          {hidden.map((n, i) => (
            <div key={i}>
              <div className="row"><label>Layer {i + 1} units</label><span className="val">{n}</span></div>
              <input type="range" min={1} max={8} step={1} value={n}
                     onChange={e => setLayerSize(i, +e.target.value)} />
            </div>
          ))}
          <div className="row" style={{ marginTop: 8 }}><label>Activation</label></div>
          <select value={act} onChange={e => setAct(e.target.value as Activation)}>
            <option value="tanh">tanh</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">sigmoid</option>
            <option value="linear">linear (no non-linearity)</option>
          </select>
        </div>

        <div className="grp">
          <h4>Training</h4>
          <div className="row"><label>Learning rate</label><span className="val">{lr.toFixed(3)}</span></div>
          <input type="range" min={0.001} max={1} step={0.001} value={lr}
                 onChange={e => setLr(+e.target.value)} />
          <div className="row"><label>Batch size</label><span className="val">{batchSize}</span></div>
          <input type="range" min={1} max={40} step={1} value={batchSize}
                 onChange={e => setBatchSize(+e.target.value)} />
          <div className="row"><label>Seed</label><span className="val">{seed}</span></div>
          <input type="range" min={1} max={99} step={1} value={seed}
                 onChange={e => setSeed(+e.target.value)} />
        </div>

        <div className="grp">
          <div className="btnrow">
            <button className="btn primary" onClick={() => setRunning(r => !r)}>
              {running ? '❚❚ Pause' : '▶ Train'}
            </button>
            <button className="btn" onClick={stepOnce} disabled={running}>Step</button>
          </div>
          <div className="btnrow" style={{ marginTop: 6 }}>
            <button className="btn" onClick={() => { setRunning(false); rebuild(); }}>Reset weights</button>
          </div>
        </div>
      </aside>

      <main className="stage">
        <div className="stats">
          <div className="stat"><span>Epoch</span><b>{net.epoch}</b></div>
          <div className="stat"><span>Train loss</span><b>{(trainLoss.at(-1) ?? 0).toFixed(4)}</b></div>
          <div className="stat"><span>Test loss</span><b>{(testLoss.at(-1) ?? 0).toFixed(4)}</b></div>
          <div className="stat"><span>Train acc</span><b>{(trainAcc * 100).toFixed(1)}%</b></div>
          <div className="stat"><span>Test acc</span><b style={{ color: testAcc > 0.9 ? 'var(--ok)' : undefined }}>
            {(testAcc * 100).toFixed(1)}%</b></div>
          <div className="stat"><span>Parameters</span><b>{net.paramCount()}</b></div>
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <h3>The network, live</h3>
          <p className="sub">
            Edge colour is the sign of the weight and thickness is its magnitude. Node fill is the
            activation for the probe point; the ring around each node is its bias. Click a node to
            isolate the weights feeding it.
          </p>
          <NetworkGraph
            net={net}
            inputLabels={features.map(f => FEATURES.find(x => x.id === f)!.label)}
            inputValues={probeVec}
            selected={selected}
            onSelect={setSelected}
          />
          <div className="legend">
            <span><i style={{ background: '#d97a45' }} />positive weight / class A</span>
            <span><i style={{ background: '#5b8dbe' }} />negative weight / class B</span>
            <span>probe ({probe.x.toFixed(2)}, {probe.y.toFixed(2)}) → output {probeOut.toFixed(3)}</span>
          </div>
        </div>

        <div className="panels">
          <div className="card">
            <h3>Decision boundary</h3>
            <p className="sub">
              The network's prediction over the whole plane, resampled every epoch. Click to move the
              probe point.
            </p>
            <BoundaryChart
              net={net} points={data} features={features} tick={tick}
              size={312}
            />
            <div style={{ marginTop: 8 }}>
              <button className="btn" onClick={() => setProbe({ x: 0.45, y: 0.45 })}>
                Probe ({probe.x.toFixed(2)}, {probe.y.toFixed(2)})
              </button>
              <input type="range" min={-1} max={1} step={0.05} value={probe.x}
                     onChange={e => setProbe(p => ({ ...p, x: +e.target.value }))}
                     style={{ marginTop: 8 }} />
              <input type="range" min={-1} max={1} step={0.05} value={probe.y}
                     onChange={e => setProbe(p => ({ ...p, y: +e.target.value }))} />
            </div>
          </div>

          <div style={{ display: 'grid', gap: 16 }}>
            <div className="card">
              <h3>Loss</h3>
              <p className="sub">Solid = training, dashed = held-out test. A gap opening up is overfitting.</p>
              <LossChart train={trainLoss} test={testLoss} />
            </div>

            <div className="card">
              <h3>Weights &amp; biases</h3>
              <p className="sub">
                {selected
                  ? `Layer ${selected.layer + 1}, unit ${selected.unit + 1} — its incoming weights.`
                  : 'Every parameter in the network, updating as it trains.'}
              </p>
              <div className="tablewrap" style={{ maxHeight: 210, overflowY: 'auto' }}>
                <table>
                  <thead>
                    <tr><th>param</th><th>value</th><th>grad·lr</th></tr>
                  </thead>
                  <tbody>
                    {net.layers.flatMap((L, l) =>
                      L.w.flatMap((row, j) =>
                        (selected && (selected.layer !== l || selected.unit !== j))
                          ? []
                          : [
                            ...row.map((w, i) => (
                              <tr key={`w${l}-${j}-${i}`}>
                                <td style={{ textAlign: 'left', color: '#74716a' }}>w{l + 1}[{j + 1}←{i + 1}]</td>
                                <td style={{ color: w >= 0 ? '#d97a45' : '#5b8dbe' }}>{w.toFixed(4)}</td>
                                <td style={{ color: '#74716a' }}>{(L.dw[j][i] * lr).toExponential(1)}</td>
                              </tr>
                            )),
                            <tr key={`b${l}-${j}`}>
                              <td style={{ textAlign: 'left', color: '#74716a' }}>b{l + 1}[{j + 1}]</td>
                              <td style={{ color: L.b[j] >= 0 ? '#d97a45' : '#5b8dbe' }}>{L.b[j].toFixed(4)}</td>
                              <td style={{ color: '#74716a' }}>{(L.db[j] * lr).toExponential(1)}</td>
                            </tr>,
                          ]
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
