/**
 * Classical ML lab — five algorithms, each fitting step by step so you can
 * watch the parameters move rather than seeing only the final answer.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  LinearRegressionGD, LogisticRegressionGD, KMeans, DecisionTree, KNN,
  makeRegression, type RegressionShape,
} from '../lib/classical.ts';
import { generate, splitTrainTest, DATASETS, type DatasetKind, type Point } from '../lib/datasets.ts';
import { LossChart } from '../components/Charts.tsx';

type Algo = 'linreg' | 'logreg' | 'kmeans' | 'tree' | 'knn';

const ALGOS: { id: Algo; name: string; blurb: string }[] = [
  { id: 'linreg', name: 'Linear Regression',   blurb: 'Gradient descent walking a line onto the data. The exact least-squares answer is drawn behind it for comparison.' },
  { id: 'logreg', name: 'Logistic Regression', blurb: 'The same gradient descent, but squashed through a sigmoid so the output is a probability.' },
  { id: 'kmeans', name: 'k-Means',             blurb: 'Assign every point to the nearest centre, then move each centre to the mean of its points. Repeat.' },
  { id: 'tree',   name: 'Decision Tree',       blurb: 'Each level picks the single split that most reduces Gini impurity, carving the plane into rectangles.' },
  { id: 'knn',    name: 'k-Nearest Neighbours',blurb: 'No training at all. The boundary is implied by the stored points and only computed when asked.' },
];

const POS = '#d97a45';
const NEG = '#5b8dbe';
const SIZE = 360;

export function ClassicalLab() {
  const [algo, setAlgo] = useState<Algo>('linreg');
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);

  /* shared data controls */
  const [dataset, setDataset] = useState<DatasetKind>('gauss');
  const [shape, setShape] = useState<RegressionShape>('linear');
  const [n, setN] = useState(120);
  const [noise, setNoise] = useState(0.12);
  const [seed, setSeed] = useState(7);
  const [lr, setLr] = useState(0.35);
  const [k, setK] = useState(3);
  const [knnK, setKnnK] = useState(5);
  const [probe, setProbe] = useState({ x: 0.3, y: 0.3 });

  const reg = useMemo(() => makeRegression(shape, n, noise, seed), [shape, n, noise, seed]);
  const pts = useMemo(() => generate(dataset, n, noise * 0.5, seed), [dataset, n, noise, seed]);
  const split = useMemo(() => splitTrainTest(pts, 0.7, seed + 1), [pts, seed]);

  /* model instances, rebuilt when their inputs change */
  const linRef = useRef(new LinearRegressionGD());
  const logRef = useRef(new LogisticRegressionGD());
  const kmRef = useRef<KMeans | null>(null);
  const treeRef = useRef<DecisionTree | null>(null);

  const rebuild = useCallback(() => {
    linRef.current = new LinearRegressionGD();
    logRef.current = new LogisticRegressionGD();
    kmRef.current = new KMeans(k, pts, seed);
    treeRef.current = new DecisionTree(split.train);
    setTick(t => t + 1);
  }, [pts, split, k, seed]);

  useEffect(() => { setRunning(false); rebuild(); }, [rebuild, algo]);

  const knn = useMemo(() => new KNN(knnK, split.train), [knnK, split]);

  /* ---------- stepping ---------- */

  const step = useCallback(() => {
    if (algo === 'linreg') linRef.current.step(reg, lr);
    else if (algo === 'logreg') logRef.current.step(split.train, lr);
    else if (algo === 'kmeans') kmRef.current?.step(pts);
    else if (algo === 'tree') {
      const grew = treeRef.current?.growOneLevel();
      if (!grew) setRunning(false);
    } else setRunning(false);      // k-NN has nothing to step
    setTick(t => t + 1);
  }, [algo, reg, lr, split, pts]);

  useEffect(() => {
    if (!running) return;
    const delay = algo === 'tree' ? 700 : algo === 'kmeans' ? 420 : 24;
    const id = setInterval(step, delay);
    return () => clearInterval(id);
  }, [running, step, algo]);

  /* ---------- canvas ---------- */

  const cv = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = cv.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = SIZE * dpr; c.height = SIZE * dpr;
    c.style.width = SIZE + 'px'; c.style.height = SIZE + 'px';
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, SIZE, SIZE);

    const px = (x: number) => ((x + 1) / 2) * SIZE;
    const py = (y: number) => ((1 - y) / 2) * SIZE;

    const axes = () => {
      g.strokeStyle = '#2b2b25'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, py(0)); g.lineTo(SIZE, py(0));
      g.moveTo(px(0), 0); g.lineTo(px(0), SIZE); g.stroke();
    };

    const field = (fn: (x: number, y: number) => number, res = 44) => {
      const cell = SIZE / res;
      for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
        const x = (i + 0.5) / res * 2 - 1;
        const y = 1 - (j + 0.5) / res * 2;
        const t = Math.max(0, Math.min(1, fn(x, y)));
        const s = Math.abs(t - 0.5) * 2;
        g.fillStyle = t >= 0.5
          ? `rgba(217,122,69,${0.10 + s * 0.44})`
          : `rgba(91,141,190,${0.10 + s * 0.44})`;
        g.fillRect(i * cell, j * cell, cell + 1, cell + 1);
      }
    };

    const dots = (list: Point[]) => {
      for (const p of list) {
        g.beginPath(); g.arc(px(p.x), py(p.y), 3.4, 0, Math.PI * 2);
        g.fillStyle = p.label === 1 ? POS : NEG; g.fill();
        g.strokeStyle = 'rgba(20,20,15,.85)'; g.lineWidth = 1; g.stroke();
      }
    };

    if (algo === 'linreg') {
      axes();
      const m = linRef.current;
      const exact = LinearRegressionGD.closedForm(reg);

      // exact least-squares answer, drawn behind as the target
      g.strokeStyle = '#5f5f58'; g.lineWidth = 1.4; g.setLineDash([5, 4]);
      g.beginPath();
      g.moveTo(px(-1), py(exact.w * -1 + exact.b));
      g.lineTo(px(1), py(exact.w * 1 + exact.b));
      g.stroke(); g.setLineDash([]);

      // residuals
      g.strokeStyle = 'rgba(217,122,69,.32)'; g.lineWidth = 1;
      for (const p of reg) {
        g.beginPath(); g.moveTo(px(p.x), py(p.y)); g.lineTo(px(p.x), py(m.predict(p.x))); g.stroke();
      }

      for (const p of reg) {
        g.beginPath(); g.arc(px(p.x), py(p.y), 3.2, 0, Math.PI * 2);
        g.fillStyle = NEG; g.fill();
      }

      g.strokeStyle = POS; g.lineWidth = 2.4;
      g.beginPath();
      g.moveTo(px(-1), py(m.predict(-1)));
      g.lineTo(px(1), py(m.predict(1)));
      g.stroke();
    }

    if (algo === 'logreg') {
      const m = logRef.current;
      field((x, y) => m.prob(x, y));
      dots(split.train);
      // the boundary itself: w·x + b = 0
      const [w0, w1] = m.w;
      if (Math.abs(w1) > 1e-6) {
        g.strokeStyle = '#f0e6dc'; g.lineWidth = 2;
        g.beginPath();
        g.moveTo(px(-1), py((-m.b - w0 * -1) / w1));
        g.lineTo(px(1), py((-m.b - w0 * 1) / w1));
        g.stroke();
      }
    }

    if (algo === 'kmeans') {
      const m = kmRef.current;
      if (m) {
        const palette = ['#d97a45', '#5b8dbe', '#8fae62', '#b98cc4', '#c9a227'];
        pts.forEach((p, i) => {
          const c = m.assignment[i];
          g.beginPath(); g.arc(px(p.x), py(p.y), 3.6, 0, Math.PI * 2);
          g.fillStyle = c >= 0 ? palette[c % palette.length] : '#55554e';
          g.fill();
          if (c >= 0) {
            g.strokeStyle = palette[c % palette.length] + '55'; g.lineWidth = 1;
            g.beginPath(); g.moveTo(px(p.x), py(p.y));
            g.lineTo(px(m.centroids[c].x), py(m.centroids[c].y)); g.stroke();
          }
        });
        m.centroids.forEach((c, i) => {
          const col = palette[i % palette.length];
          g.strokeStyle = '#14140f'; g.lineWidth = 5;
          g.beginPath(); g.moveTo(px(c.x) - 9, py(c.y)); g.lineTo(px(c.x) + 9, py(c.y));
          g.moveTo(px(c.x), py(c.y) - 9); g.lineTo(px(c.x), py(c.y) + 9); g.stroke();
          g.strokeStyle = col; g.lineWidth = 2.6;
          g.beginPath(); g.moveTo(px(c.x) - 9, py(c.y)); g.lineTo(px(c.x) + 9, py(c.y));
          g.moveTo(px(c.x), py(c.y) - 9); g.lineTo(px(c.x), py(c.y) + 9); g.stroke();
        });
      }
    }

    if (algo === 'tree') {
      const t = treeRef.current;
      if (t) {
        for (const leaf of t.leaves()) {
          const b = leaf.box;
          const s = Math.abs(leaf.value - 0.5) * 2;
          g.fillStyle = leaf.value >= 0.5
            ? `rgba(217,122,69,${0.12 + s * 0.44})`
            : `rgba(91,141,190,${0.12 + s * 0.44})`;
          g.fillRect(px(b.x0), py(b.y1), px(b.x1) - px(b.x0), py(b.y0) - py(b.y1));
          g.strokeStyle = 'rgba(240,230,220,.28)'; g.lineWidth = 1;
          g.strokeRect(px(b.x0), py(b.y1), px(b.x1) - px(b.x0), py(b.y0) - py(b.y1));
        }
        dots(split.train);
      }
    }

    if (algo === 'knn') {
      field((x, y) => knn.predict(x, y), 40);
      dots(split.train);
      // links from the probe to its k neighbours
      const nb = knn.neighbours(probe.x, probe.y);
      g.strokeStyle = '#f0e6dc'; g.lineWidth = 1.2;
      for (const q of nb) {
        g.beginPath(); g.moveTo(px(probe.x), py(probe.y)); g.lineTo(px(q.p.x), py(q.p.y)); g.stroke();
      }
      if (nb.length) {
        g.strokeStyle = '#f0e6dc'; g.lineWidth = 1.4; g.setLineDash([3, 3]);
        g.beginPath(); g.arc(px(probe.x), py(probe.y),
          Math.abs(px(probe.x + nb[nb.length - 1].d) - px(probe.x)), 0, Math.PI * 2);
        g.stroke(); g.setLineDash([]);
      }
      g.beginPath(); g.arc(px(probe.x), py(probe.y), 6, 0, Math.PI * 2);
      g.fillStyle = '#f0e6dc'; g.fill();
    }

    g.strokeStyle = '#32322c'; g.lineWidth = 1;
    g.strokeRect(0.5, 0.5, SIZE - 1, SIZE - 1);
  }, [algo, tick, reg, pts, split, knn, probe]);

  /* ---------- readouts ---------- */

  const meta = ALGOS.find(a => a.id === algo)!;
  const lin = linRef.current, lg = logRef.current, km = kmRef.current, tr = treeRef.current;
  const exact = useMemo(() => LinearRegressionGD.closedForm(reg), [reg]);

  const stats: [string, string][] =
    algo === 'linreg' ? [
      ['Step', String(lin.steps)],
      ['Loss (MSE)', lin.loss(reg).toFixed(5)],
      ['w', lin.w.toFixed(4)],
      ['b', lin.b.toFixed(4)],
      ['exact w', exact.w.toFixed(4)],
      ['exact b', exact.b.toFixed(4)],
    ] : algo === 'logreg' ? [
      ['Step', String(lg.steps)],
      ['Loss', lg.loss(split.train).toFixed(5)],
      ['Train acc', (lg.accuracy(split.train) * 100).toFixed(1) + '%'],
      ['Test acc', (lg.accuracy(split.test) * 100).toFixed(1) + '%'],
      ['w₁', lg.w[0].toFixed(3)],
      ['w₂', lg.w[1].toFixed(3)],
      ['b', lg.b.toFixed(3)],
    ] : algo === 'kmeans' ? [
      ['Iteration', String(km?.iterations ?? 0)],
      ['Next step', km?.phase === 'assign' ? 'assign' : 'move centres'],
      ['Inertia', (km?.inertia(pts) ?? 0).toFixed(5)],
      ['Converged', km?.converged ? 'yes' : 'no'],
    ] : algo === 'tree' ? [
      ['Depth', String(tr?.depth ?? 0)],
      ['Nodes', String(tr?.nodeCount() ?? 0)],
      ['Leaves', String(tr?.leaves().length ?? 0)],
      ['Train acc', ((tr?.accuracy(split.train) ?? 0) * 100).toFixed(1) + '%'],
      ['Test acc', ((tr?.accuracy(split.test) ?? 0) * 100).toFixed(1) + '%'],
    ] : [
      ['k', String(knnK)],
      ['Stored points', String(split.train.length)],
      ['Train acc', (knn.accuracy(split.train) * 100).toFixed(1) + '%'],
      ['Test acc', (knn.accuracy(split.test) * 100).toFixed(1) + '%'],
      ['Training cost', 'zero'],
    ];

  return (
    <div className="lab">
      <aside className="side">
        <div className="grp">
          <h4>Algorithm</h4>
          <select value={algo} onChange={e => setAlgo(e.target.value as Algo)}>
            {ALGOS.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
          <p className="hint" style={{ marginTop: 8 }}>{meta.blurb}</p>
        </div>

        <div className="grp">
          <h4>Data</h4>
          {algo === 'linreg' ? (
            <select value={shape} onChange={e => setShape(e.target.value as RegressionShape)}>
              <option value="linear">Linear trend</option>
              <option value="curved">Curved (a line cannot fit this)</option>
              <option value="noisy">Noisy linear</option>
            </select>
          ) : (
            <select value={dataset} onChange={e => setDataset(e.target.value as DatasetKind)}>
              {DATASETS.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          )}
          <div className="row" style={{ marginTop: 10 }}><label>Points</label><span className="val">{n}</span></div>
          <input type="range" min={40} max={400} step={10} value={n} onChange={e => setN(+e.target.value)} />
          <div className="row"><label>Noise</label><span className="val">{noise.toFixed(2)}</span></div>
          <input type="range" min={0} max={0.4} step={0.01} value={noise} onChange={e => setNoise(+e.target.value)} />
          <div className="row"><label>Seed</label><span className="val">{seed}</span></div>
          <input type="range" min={1} max={60} step={1} value={seed} onChange={e => setSeed(+e.target.value)} />
        </div>

        {(algo === 'linreg' || algo === 'logreg') && (
          <div className="grp">
            <h4>Optimiser</h4>
            <div className="row"><label>Learning rate</label><span className="val">{lr.toFixed(3)}</span></div>
            <input type="range" min={0.005} max={2} step={0.005} value={lr} onChange={e => setLr(+e.target.value)} />
            <p className="hint">Push this past ~1.5 and the fit oscillates instead of settling.</p>
          </div>
        )}

        {algo === 'kmeans' && (
          <div className="grp">
            <h4>Clusters</h4>
            <div className="row"><label>k</label><span className="val">{k}</span></div>
            <input type="range" min={2} max={5} step={1} value={k} onChange={e => setK(+e.target.value)} />
            <p className="hint">k is chosen by you, not learned — that is the awkward part of k-means.</p>
          </div>
        )}

        {algo === 'knn' && (
          <div className="grp">
            <h4>Neighbours</h4>
            <div className="row"><label>k</label><span className="val">{knnK}</span></div>
            <input type="range" min={1} max={31} step={2} value={knnK} onChange={e => setKnnK(+e.target.value)} />
            <p className="hint">k = 1 gives a jagged boundary that memorises noise. Large k smooths it flat.</p>
            <div className="row" style={{ marginTop: 10 }}><label>Probe X</label><span className="val">{probe.x.toFixed(2)}</span></div>
            <input type="range" min={-1} max={1} step={0.02} value={probe.x} onChange={e => setProbe(p => ({ ...p, x: +e.target.value }))} />
            <div className="row"><label>Probe Y</label><span className="val">{probe.y.toFixed(2)}</span></div>
            <input type="range" min={-1} max={1} step={0.02} value={probe.y} onChange={e => setProbe(p => ({ ...p, y: +e.target.value }))} />
          </div>
        )}

        <div className="grp">
          <div className="btnrow">
            <button className="btn primary" onClick={() => setRunning(r => !r)} disabled={algo === 'knn'}>
              {running ? '❚❚ Pause' : algo === 'knn' ? 'Nothing to fit' : '▶ Fit'}
            </button>
            <button className="btn" onClick={step} disabled={running || algo === 'knn'}>Step</button>
          </div>
          <div className="btnrow" style={{ marginTop: 6 }}>
            <button className="btn" onClick={() => { setRunning(false); rebuild(); }}>Reset</button>
          </div>
        </div>
      </aside>

      <main className="stage">
        <div className="stats">
          {stats.map(([label, value]) => (
            <div className="stat" key={label}><span>{label}</span><b>{value}</b></div>
          ))}
        </div>

        <div className="panels">
          <div className="card">
            <h3>{meta.name}</h3>
            <p className="sub">
              {algo === 'linreg' && 'Orange is the current fit, grey dashed is the exact least-squares answer, thin lines are the residuals being minimised.'}
              {algo === 'logreg' && 'Shading is predicted probability; the white line is where it crosses 0.5.'}
              {algo === 'kmeans' && 'Each point is linked to its assigned centre. Crosses are the centres.'}
              {algo === 'tree' && 'Every rectangle is a leaf. Each level adds one split per impure region.'}
              {algo === 'knn' && 'White lines link the probe to its k nearest neighbours; the dashed circle is their radius.'}
            </p>
            <canvas ref={cv} />
          </div>

          <div style={{ display: 'grid', gap: 16 }}>
            {(algo === 'linreg' || algo === 'logreg') && (
              <div className="card">
                <h3>Loss</h3>
                <p className="sub">Falling as the parameters descend the gradient.</p>
                <LossChart train={algo === 'linreg' ? lin.history : lg.history} test={[]} />
              </div>
            )}

            <div className="card">
              <h3>What to notice</h3>
              <p className="sub" style={{ marginBottom: 0 }}>
                {algo === 'linreg' && 'Switch the data to "Curved". Gradient descent still converges perfectly — to the best possible straight line, which is a bad model. Converging is not the same as being right.'}
                {algo === 'logreg' && 'Try the Circle or XOR dataset. The boundary is always a straight line, so it cannot get far above chance no matter how long it fits.'}
                {algo === 'kmeans' && 'Step it one half-step at a time and watch assign and move alternate. Change the seed a few times on Spiral — where it lands depends entirely on where it started.'}
                {algo === 'tree' && 'Keep growing. Training accuracy climbs toward 100% while test accuracy stalls or falls — the tree is memorising individual points.'}
                {algo === 'knn' && 'There is no fit button because there is no training. All the cost moved to prediction time: every query scans all stored points.'}
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
