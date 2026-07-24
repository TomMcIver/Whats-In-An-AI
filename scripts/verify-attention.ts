/**
 * Correctness harness for the attention model.
 *   npm run verify:attn
 *
 * Gradient-checks a weight inside the Q projection — which means the check
 * runs through the attention softmax — then confirms the model learns the
 * copy task and that its attention actually points at the matching position.
 */

import { AttentionModel, makeBatch, makeSequence, PREFIX, V } from '../src/lib/attention.ts';
import { RNG } from '../src/lib/rng.ts';

const checks: [string, boolean][] = [];

/* ---------- gradient check through the attention softmax ---------- */

console.log('\n--- gradient check (Wq, through the softmax) ---');
{
  const seed = 9;
  const seq = makeSequence(new RNG(2));

  const lossOf = (m: AttentionModel) => {
    m.forward(seq);
    let l = 0, n = 0;
    for (let t = PREFIX; t < seq.length - 1; t++) {
      l += -Math.log(Math.max(m.probs[t][seq[t + 1]], 1e-12));
      n++;
    }
    return l / n;
  };

  // analytic, read back from an update with a tiny learning rate
  const m1 = new AttentionModel(12, 2, seed);
  const before = m1.Wq[0][3][1];
  const lr = 1e-5;
  m1.trainOne(seq, lr);
  const analytic = (before - m1.Wq[0][3][1]) / lr;

  // numerical
  const m2 = new AttentionModel(12, 2, seed);
  const w0 = m2.Wq[0][3][1];
  const eps = 1e-5;
  m2.Wq[0][3][1] = w0 + eps; const lp = lossOf(m2);
  m2.Wq[0][3][1] = w0 - eps; const lm = lossOf(m2);
  m2.Wq[0][3][1] = w0;
  const numeric = (lp - lm) / (2 * eps);

  const rel = Math.abs(analytic - numeric) / Math.max(1e-9, Math.abs(analytic) + Math.abs(numeric));
  console.log('  analytic  ' + analytic.toExponential(8));
  console.log('  numerical ' + numeric.toExponential(8));
  console.log('  rel error ' + rel.toExponential(3));
  checks.push(['attention backprop matches numerical gradient (rel err < 1e-3)', rel < 1e-3]);
}

/* ---------- does it learn to copy? ---------- */

console.log('\n--- does it learn the copy task? ---');
{
  const train = makeBatch(64, 1);
  const test = makeBatch(48, 999);
  const m = new AttentionModel(24, 2, 4);

  const accBefore = m.accuracy(test);
  let loss = 0;
  for (let e = 0; e < 300; e++) loss = m.trainEpoch(train, 0.05);
  const accTrain = m.accuracy(train);
  const accTest = m.accuracy(test);

  console.log('  chance                 : ' + (100 / V).toFixed(1) + '%');
  console.log('  test acc before        : ' + (accBefore * 100).toFixed(1) + '%');
  console.log('  final loss             : ' + loss.toFixed(4));
  console.log('  train acc              : ' + (accTrain * 100).toFixed(1) + '%');
  console.log('  test acc (unseen seqs) : ' + (accTest * 100).toFixed(1) + '%');
  console.log('  parameters             : ' + m.paramCount());

  checks.push(['learns the copy task on training data (>85%)', accTrain > 0.85]);
  checks.push(['generalises to unseen sequences (>80%)', accTest > 0.80]);
  checks.push(['loss finite', Number.isFinite(loss)]);

  /* the interesting one: does attention point at the matching position?
     predicting position t (t >= PREFIX) requires token t-PREFIX          */
  console.log('\n--- where does it attend? ---');
  const seq = makeSequence(new RNG(31));
  m.forward(seq);
  let hits = 0, total = 0;
  for (let t = PREFIX; t < seq.length - 1; t++) {
    const want = t - PREFIX + 1;          // the position holding the answer
    let bestHead = 0, bestPos = 0, bestVal = -1;
    for (let h = 0; h < m.H; h++)
      for (let u = 0; u <= t; u++)
        if (m.heads[h].attn[t][u] > bestVal) { bestVal = m.heads[h].attn[t][u]; bestPos = u; bestHead = h; }
    const ok = Math.abs(bestPos - want) <= 1;
    if (ok) hits++;
    total++;
    console.log('  predicting pos ' + (t + 1) + ': strongest attention -> pos ' + bestPos +
                ' (head ' + bestHead + ', ' + (bestVal * 100).toFixed(0) + '%)' +
                '  wanted ~' + want + (ok ? '  ok' : '  miss'));
  }
  checks.push(['attention concentrates near the matching position (majority)', hits > total / 2]);
}

console.log('\n--- results ---');
let bad = 0;
for (const [name, ok] of checks) {
  console.log('  ' + (ok ? 'PASS' : 'FAIL') + '  ' + name);
  if (!ok) bad++;
}
console.log('\n' + (bad === 0 ? 'All checks passed.' : bad + ' check(s) FAILED.') + '\n');
process.exit(bad === 0 ? 0 : 1);
