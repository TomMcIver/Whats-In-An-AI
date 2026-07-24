/**
 * Correctness harness for the from-scratch network.
 *
 *   npm run verify
 *
 * Checks two separate things:
 *   1. the network actually learns (and fails where theory says it must)
 *   2. backprop's analytic gradient matches a numerical estimate
 */

import { MLP } from '../src/lib/nn.ts';
import { generate, featurise, splitTrainTest, type DatasetKind } from '../src/lib/datasets.ts';

function run(kind: DatasetKind, sizes: number[], lr: number, epochs: number) {
  const pts = generate(kind, 300, 0.05, 7);
  const { train, test } = splitTrainTest(pts, 0.7, 3);

  const xs = train.map(p => featurise(p.x, p.y, ['x', 'y']));
  const ys = train.map(p => [p.label]);
  const txs = test.map(p => featurise(p.x, p.y, ['x', 'y']));
  const tys = test.map(p => [p.label]);

  const net = new MLP({ sizes, hiddenAct: 'tanh', task: 'classification', seed: 11 });
  const first = net.evaluate(xs, ys);

  let loss = first;
  for (let e = 0; e < epochs; e++) loss = net.trainEpoch(xs, ys, lr, 10);

  const trainAcc = net.accuracy(xs, ys);
  const testAcc = net.accuracy(txs, tys);

  console.log(
    '  ' + kind.padEnd(7) +
    ('[' + sizes.join(',') + ']').padEnd(11) +
    ' loss ' + first.toFixed(3) + ' -> ' + loss.toFixed(3) +
    '   train ' + (trainAcc * 100).toFixed(1).padStart(5) + '%' +
    '   test ' + (testAcc * 100).toFixed(1).padStart(5) + '%' +
    '   params ' + net.paramCount()
  );
  return { first, loss, trainAcc, testAcc };
}

console.log('\n--- does it learn? ---');
const gauss  = run('gauss',  [2, 1],       0.5, 120);  // linearly separable
const xorNo  = run('xor',    [2, 1],       0.5, 200);  // must fail: no hidden layer
const xorYes = run('xor',    [2, 4, 1],    0.5, 400);  // must succeed
const circle = run('circle', [2, 6, 1],    0.5, 400);
const spiral = run('spiral', [2, 8, 8, 1], 0.4, 900);

const checks: [string, boolean][] = [
  ['linear model separates two blobs (>90%)',        gauss.trainAcc > 0.90],
  ['loss decreased',                                  gauss.loss < gauss.first],
  ['XOR without a hidden layer FAILS (<70%)',         xorNo.trainAcc < 0.70],
  ['XOR with a hidden layer succeeds (>90%)',         xorYes.trainAcc > 0.90],
  ['circle solved (>90%)',                            circle.trainAcc > 0.90],
  ['spiral well above chance (>75%)',                 spiral.trainAcc > 0.75],
  ['no NaN anywhere',  [gauss, xorYes, circle, spiral].every(r => Number.isFinite(r.loss))],
];

console.log('\n--- gradient check ---');
{
  const net = new MLP({ sizes: [2, 3, 1], hiddenAct: 'tanh', seed: 5 });
  const x = [0.3, -0.7];
  const y = [1];
  net.forward(x);
  net.backward(y);
  const analytic = net.layers[0].dw[1][0];

  const eps = 1e-5;
  const w0 = net.layers[0].w[1][0];
  const lossAt = (w: number) => {
    net.layers[0].w[1][0] = w;
    const o = net.forward(x);
    const p = Math.min(Math.max(o[0], 1e-9), 1 - 1e-9);
    return -(y[0] * Math.log(p) + (1 - y[0]) * Math.log(1 - p));
  };
  const numeric = (lossAt(w0 + eps) - lossAt(w0 - eps)) / (2 * eps);
  net.layers[0].w[1][0] = w0;

  const relErr = Math.abs(analytic - numeric) /
                 Math.max(1e-9, Math.abs(analytic) + Math.abs(numeric));
  console.log('  analytic  ' + analytic.toExponential(8));
  console.log('  numerical ' + numeric.toExponential(8));
  console.log('  rel error ' + relErr.toExponential(3));
  checks.push(['backprop matches numerical gradient (rel err < 1e-6)', relErr < 1e-6]);
}

console.log('\n--- results ---');
let bad = 0;
for (const [name, ok] of checks) {
  console.log('  ' + (ok ? 'PASS' : 'FAIL') + '  ' + name);
  if (!ok) bad++;
}
console.log('\n' + (bad === 0 ? 'All checks passed.' : bad + ' check(s) FAILED.') + '\n');
process.exit(bad === 0 ? 0 : 1);
