/**
 * Correctness harness for the convolutional network.
 *   npm run verify:conv
 *
 * Gradient-checks a filter weight through the whole conv -> ReLU -> pool ->
 * dense -> softmax stack, then confirms the net actually learns the shapes.
 */

import { ConvNet, makeShapeSet, drawShape, SHAPES } from '../src/lib/conv.ts';
import { RNG } from '../src/lib/rng.ts';

const checks: [string, boolean][] = [];

/* ---------- gradient check through the convolution ---------- */

console.log('\n--- gradient check (filter weight) ---');
{
  const net = new ConvNet(3, SHAPES.length, 5);
  const rng = new RNG(1);
  const img = drawShape('cross', rng, false);
  const label = 2;

  const lossOf = () => {
    const p = net.forward(img);
    return -Math.log(Math.max(p[label], 1e-12));
  };

  // analytic: run trainOne with lr = 0 and read the gradient we would apply
  const before = net.filters[1][2][0];
  const lr = 1e-4;
  net.trainOne(img, label, lr);
  const after = net.filters[1][2][0];
  const analytic = (before - after) / lr;

  // restore, then estimate numerically
  net.reset(5);
  const w0 = net.filters[1][2][0];
  const eps = 1e-5;
  net.filters[1][2][0] = w0 + eps; const lPlus = lossOf();
  net.filters[1][2][0] = w0 - eps; const lMinus = lossOf();
  net.filters[1][2][0] = w0;
  const numeric = (lPlus - lMinus) / (2 * eps);

  const rel = Math.abs(analytic - numeric) / Math.max(1e-9, Math.abs(analytic) + Math.abs(numeric));
  console.log('  analytic  ' + analytic.toExponential(8));
  console.log('  numerical ' + numeric.toExponential(8));
  console.log('  rel error ' + rel.toExponential(3));
  checks.push(['conv backprop matches numerical gradient (rel err < 1e-4)', rel < 1e-4]);
}

/* ---------- does it learn the shapes? ---------- */

console.log('\n--- does the CNN learn? ---');
{
  const train = makeShapeSet(160, 3);
  const test = makeShapeSet(80, 777);
  const net = new ConvNet(4, SHAPES.length, 3);

  const accBefore = net.accuracy(test);
  let loss = 0;
  for (let e = 0; e < 60; e++) loss = net.trainEpoch(train, 0.01);
  const accTrain = net.accuracy(train);
  const accTest = net.accuracy(test);

  console.log('  test acc before training : ' + (accBefore * 100).toFixed(1) + '%  (chance = 25%)');
  console.log('  final loss               : ' + loss.toFixed(4));
  console.log('  train acc                : ' + (accTrain * 100).toFixed(1) + '%');
  console.log('  test acc                 : ' + (accTest * 100).toFixed(1) + '%');
  console.log('  parameters               : ' + net.paramCount());

  checks.push(['CNN beats chance on held-out shapes (>70%)', accTest > 0.70]);
  checks.push(['CNN fits its training set (>80%)', accTrain > 0.80]);
  checks.push(['loss is finite', Number.isFinite(loss)]);

  // filters must have actually moved away from their initialisation
  const fresh = new ConvNet(4, SHAPES.length, 3);
  let drift = 0;
  for (let f = 0; f < 4; f++)
    for (let a = 0; a < 3; a++)
      for (let b = 0; b < 3; b++)
        drift += Math.abs(net.filters[f][a][b] - fresh.filters[f][a][b]);
  console.log('  total filter drift       : ' + drift.toFixed(4));
  checks.push(['filters changed during training (drift > 0.1)', drift > 0.1]);
}

/* ---------- shapes are actually distinguishable ---------- */

console.log('\n--- sanity: are the classes distinct? ---');
{
  const rng = new RNG(2);
  const sums = SHAPES.map(s => {
    const g = drawShape(s, rng, false);
    let n = 0;
    for (const row of g) for (const v of row) n += v;
    return { s, n };
  });
  sums.forEach(x => console.log('  ' + x.s.padEnd(11) + ' lit pixels ' + x.n.toFixed(0)));
  checks.push(['all four shapes contain ink', sums.every(x => x.n > 4)]);
}

console.log('\n--- results ---');
let bad = 0;
for (const [name, ok] of checks) {
  console.log('  ' + (ok ? 'PASS' : 'FAIL') + '  ' + name);
  if (!ok) bad++;
}
console.log('\n' + (bad === 0 ? 'All checks passed.' : bad + ' check(s) FAILED.') + '\n');
process.exit(bad === 0 ? 0 : 1);
