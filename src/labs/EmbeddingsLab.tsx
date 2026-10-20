import { useCallback, useEffect, useRef, useState } from 'react';
import { LossChart } from '../components/Charts.tsx';
import { CO_OCCURRENCES, EmbeddingModel, FOCUS_WORDS, VOCABULARY } from '../lib/embeddings.ts';
import './embeddings.css';

const SEED = 17;
const RATE = 0.08;

export function EmbeddingsLab({ guideAction = null }: { guideAction?: { step: number; id: number } | null }) {
  const modelRef = useRef(new EmbeddingModel(SEED));
  const consumedGuideId = useRef<number | null>(null);
  const [, setTick] = useState(0);
  const [selected, setSelected] = useState('cat');
  const [running, setRunning] = useState(false);
  const [stopAt, setStopAt] = useState<number | null>(null);
  const model = modelRef.current;

  const reset = useCallback(() => {
    setRunning(false);
    setStopAt(null);
    modelRef.current = new EmbeddingModel(SEED);
    setTick(t => t + 1);
  }, []);

  const step = useCallback(() => {
    modelRef.current.trainEpoch(RATE);
    setTick(t => t + 1);
  }, []);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      const m = modelRef.current;
      for (let i = 0; i < 2 && (stopAt === null || m.epoch < stopAt); i++) m.trainEpoch(RATE);
      setTick(t => t + 1);
      if (stopAt !== null && m.epoch >= stopAt) {
        setRunning(false);
        setStopAt(null);
      }
    }, 32);
    return () => window.clearInterval(timer);
  }, [running, stopAt]);

  useEffect(() => {
    if (!guideAction || consumedGuideId.current === guideAction.id) return;
    consumedGuideId.current = guideAction.id;
    if (guideAction.step === 0) {
      reset();
      setSelected('cat');
    } else if (guideAction.step === 1) {
      setStopAt(modelRef.current.epoch + 120);
      setRunning(true);
    } else if (guideAction.step === 2) {
      setSelected('kitten');
    }
  }, [guideAction, reset]);

  const bounds = Math.max(0.75, ...model.input.flatMap(vector => vector.map(Math.abs))) * 1.22;
  const pos = (vector: readonly number[]) => ({
    x: 310 + (vector[0] / bounds) * 245,
    y: 210 - (vector[1] / bounds) * 155,
  });
  const selectedIndex = FOCUS_WORDS.indexOf(selected);
  const neighbors = model.neighbors(selected);
  const predictions = model.probabilities(selected)
    .map((probability, i) => ({ word: VOCABULARY[i], probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 5);
  const observations = CO_OCCURRENCES.find(([word]) => word === selected)?.[1] ?? [];

  return (
    <div className="lab embeddings-lab">
      <aside className="side">
        <div className="grp">
          <h4>The task</h4>
          <p className="hint">Given a focus word, predict one of the words observed beside it. Each focus/context combination is a training pair in this deliberately small toy corpus.</p>
          <p className="hint">Words with similar observed contexts receive similar updates. The model learns two numbers per focus word; no categories or meanings are supplied to it.</p>
        </div>
        <div className="grp">
          <h4>Choose a focus word</h4>
          <select aria-label="Focus word" value={selected} onChange={event => setSelected(event.target.value)}>
            {FOCUS_WORDS.map(word => <option key={word} value={word}>{word}</option>)}
          </select>
          <p className="hint" style={{ marginTop: 9 }}>Observed nearby: {observations.join(', ')}.</p>
        </div>
        <div className="grp">
          <h4>Training</h4>
          <div className="btnrow">
            <button className="btn primary" onClick={() => { setStopAt(null); setRunning(value => !value); }}>
              {running ? 'Pause' : 'Train'}
            </button>
            <button className="btn" onClick={step} disabled={running}>Epoch</button>
          </div>
          <div className="btnrow" style={{ marginTop: 6 }}>
            <button className="btn" onClick={reset}>Reset weights</button>
          </div>
          <p className="hint" style={{ marginTop: 10 }}>Reset uses the same random seed so changes are reproducible.</p>
        </div>
        <div className="grp">
          <h4>Training pairs</h4>
          <p className="hint">{selected} → {observations.map(word => `${word}`).join(' · ')}</p>
          <p className="hint">The objective is average cross-entropy across all {CO_OCCURRENCES.reduce((n, [, words]) => n + words.length, 0)} pairs. It rewards probability on the observed context word.</p>
        </div>
      </aside>

      <main className="stage">
        <div className="stats">
          <div className="stat"><span>Epoch</span><b>{model.epoch}</b></div>
          <div className="stat"><span>Pair loss</span><b>{(model.history.at(-1) ?? model.loss()).toFixed(3)}</b></div>
          <div className="stat"><span>Focus words</span><b>{FOCUS_WORDS.length}</b></div>
          <div className="stat"><span>Vector size</span><b>2</b></div>
          <div className="stat"><span>Parameters</span><b>{model.input.length * 2 + model.output.length * 2 + model.bias.length}</b></div>
        </div>

        <div className="card embeddings-map-card">
          <h3>Learned embedding space</h3>
          <p className="sub">Each dot is a focus word’s actual two-dimensional input vector. Click a word to inspect it. Axes rescale as training expands the vectors; relative positions and cosine scores matter more than absolute coordinates.</p>
          <svg className="embeddings-map" viewBox="0 0 620 420" role="img" aria-label="Two-dimensional positions of learned word embeddings">
            <line x1="310" x2="310" y1="18" y2="402" className="embeddings-axis" />
            <line x1="18" x2="602" y1="210" y2="210" className="embeddings-axis" />
            <text x="591" y="225" className="embeddings-axis-label">x</text>
            <text x="316" y="28" className="embeddings-axis-label">y</text>
            {model.input.map((vector, i) => {
              const point = pos(vector);
              const active = i === selectedIndex;
              return (
                <g key={FOCUS_WORDS[i]} className="embeddings-point" onClick={() => setSelected(FOCUS_WORDS[i])} tabIndex={0}
                   role="button" aria-label={`Inspect ${FOCUS_WORDS[i]}`} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelected(FOCUS_WORDS[i]); } }}>
                  <circle cx={point.x} cy={point.y} r={active ? 13 : 9} className={active ? 'selected' : ''} />
                  <text x={point.x + 13} y={point.y - 11} className={active ? 'selected' : ''}>{FOCUS_WORDS[i]}</text>
                </g>
              );
            })}
          </svg>
          <div className="embeddings-map-foot">The plot uses the learned coordinates directly. There is no PCA or t-SNE projection.</div>
        </div>

        <div className="panels embeddings-panels">
          <div className="card">
            <h3>Nearest to “{selected}”</h3>
            <p className="sub">Cosine similarity compares vector direction. A value near 1 means the words point the same way.</p>
            <div className="embeddings-rank-list">
              {neighbors.slice(0, 5).map(({ word, score }, i) => (
                <button type="button" key={word} onClick={() => setSelected(word)} className="embeddings-rank">
                  <span className="embeddings-rank-index">{String(i + 1).padStart(2, '0')}</span>
                  <span className="embeddings-rank-word">{word}</span>
                  <span className="embeddings-rank-bar"><i style={{ width: `${Math.max(0, (score + 1) * 50)}%` }} /></span>
                  <strong>{score.toFixed(2)}</strong>
                </button>
              ))}
            </div>
          </div>
          <div className="card">
            <h3>Predicted nearby words</h3>
            <p className="sub">The softmax output for “{selected}”. The model can assign probability to every vocabulary word; the top five are shown.</p>
            <div className="embeddings-rank-list">
              {predictions.map(({ word, probability }, i) => (
                <div key={word} className="embeddings-rank">
                  <span className="embeddings-rank-index">{String(i + 1).padStart(2, '0')}</span>
                  <span className="embeddings-rank-word">{word}</span>
                  <span className="embeddings-rank-bar"><i style={{ width: `${probability * 100}%` }} /></span>
                  <strong>{(probability * 100).toFixed(1)}%</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <h3>Training loss</h3>
            <p className="sub">Average negative log probability of each observed pair.</p>
            <LossChart train={model.history} test={[]} />
          </div>
          <div className="card">
            <h3>What this shows</h3>
            <p className="sub">After training, compare cat with kitten, then car or apple. The corpus gives similar contexts to each related group, so nearby vectors are a useful way to compress those observations.</p>
            <p className="sub">This toy corpus is curated and tiny. The distances describe its co-occurrence pattern; they are not a general measure of meaning, and the two axes have no fixed interpretation.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
