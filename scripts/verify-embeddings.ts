import { strict as assert } from 'node:assert';
import { EmbeddingModel, FOCUS_WORDS } from '../src/lib/embeddings.ts';

const first = new EmbeddingModel(17);
const duplicate = new EmbeddingModel(17);
assert.deepEqual(first.input, duplicate.input, 'seeded initialization should be reproducible');

const initial = first.loss();
for (let epoch = 0; epoch < 120; epoch++) first.trainEpoch(0.08);
const final = first.loss();
assert.ok(final < initial * 0.7, `training should reduce pair loss substantially (${initial} → ${final})`);
assert.ok(first.neighbors('cat').slice(0, 2).some(item => item.word === 'kitten'), 'shared contexts should make cat and kitten neighbors');
assert.ok(first.neighbors('apple').slice(0, 3).some(item => item.word === 'pear'), 'shared contexts should make apple and pear neighbors');

for (const word of FOCUS_WORDS) {
  const probabilities = first.probabilities(word);
  assert.ok(probabilities.every(value => Number.isFinite(value) && value > 0));
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-10);
}

for (let epoch = 0; epoch < 120; epoch++) duplicate.trainEpoch(0.08);
assert.deepEqual(first.input, duplicate.input, 'training should be deterministic for a given seed');
console.log(`Embeddings verified: loss ${initial.toFixed(3)} → ${final.toFixed(3)} in 120 epochs.`);
