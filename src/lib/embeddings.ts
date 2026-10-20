import { RNG } from './rng.ts';

/** A deliberately small corpus of observed (focus word, nearby word) pairs. */
export const CO_OCCURRENCES = [
  ['cat', ['pet', 'fur', 'paws', 'home', 'meows']],
  ['kitten', ['pet', 'fur', 'paws', 'home', 'meows']],
  ['dog', ['pet', 'fur', 'paws', 'home', 'barks']],
  ['puppy', ['pet', 'fur', 'paws', 'home', 'barks']],
  ['car', ['vehicle', 'wheels', 'road', 'engine', 'drives']],
  ['truck', ['vehicle', 'wheels', 'road', 'engine', 'carries']],
  ['bus', ['vehicle', 'wheels', 'road', 'engine', 'carries']],
  ['apple', ['fruit', 'sweet', 'tree', 'food', 'red']],
  ['pear', ['fruit', 'sweet', 'tree', 'food', 'green']],
  ['banana', ['fruit', 'sweet', 'tree', 'food', 'yellow']],
] as const;

export const FOCUS_WORDS: string[] = CO_OCCURRENCES.map(([word]) => word);
export const CONTEXT_WORDS: string[] = [...new Set(CO_OCCURRENCES.flatMap(([, contexts]) => contexts))];
export const VOCABULARY: string[] = [...FOCUS_WORDS, ...CONTEXT_WORDS];
const PAIRS = CO_OCCURRENCES.flatMap(([focus, contexts]) => contexts.map(context => [FOCUS_WORDS.indexOf(focus), VOCABULARY.indexOf(context)] as const));

function softmax(logits: number[]): number[] {
  const largest = Math.max(...logits);
  const exp = logits.map(value => Math.exp(value - largest));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map(value => value / total);
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  const dot = a[0] * b[0] + a[1] * b[1];
  const magnitude = Math.hypot(...a) * Math.hypot(...b);
  return magnitude > 0 ? dot / magnitude : 0;
}

/** Two-dimensional skip-gram with a full softmax output layer. */
export class EmbeddingModel {
  readonly input: number[][];
  readonly output: number[][];
  readonly bias: number[];
  readonly history: number[] = [];
  epoch = 0;
  private readonly rng: RNG;

  constructor(seed = 17) {
    this.rng = new RNG(seed);
    this.input = FOCUS_WORDS.map(() => [this.rng.normal(0, 0.12), this.rng.normal(0, 0.12)]);
    this.output = VOCABULARY.map(() => [this.rng.normal(0, 0.12), this.rng.normal(0, 0.12)]);
    this.bias = VOCABULARY.map(() => 0);
  }

  probabilities(focus: string): number[] {
    const row = FOCUS_WORDS.indexOf(focus);
    if (row < 0) throw new Error(`Unknown focus word: ${focus}`);
    const vector = this.input[row];
    return softmax(this.output.map(([x, y], i) => vector[0] * x + vector[1] * y + this.bias[i]));
  }

  /** Mean cross-entropy over every observed pair, before the next update. */
  loss(): number {
    let total = 0;
    for (const [focus, context] of PAIRS) {
      const probs = this.probabilities(FOCUS_WORDS[focus]);
      total -= Math.log(Math.max(probs[context], 1e-12));
    }
    return total / PAIRS.length;
  }

  trainEpoch(learningRate = 0.08): number {
    const order = this.rng.shuffle([...PAIRS]);
    for (const [focus, context] of order) {
      const vector = this.input[focus];
      const probs = this.probabilities(FOCUS_WORDS[focus]);
      let gradX = 0;
      let gradY = 0;
      for (let i = 0; i < VOCABULARY.length; i++) {
        const gradient = probs[i] - (i === context ? 1 : 0);
        gradX += gradient * this.output[i][0];
        gradY += gradient * this.output[i][1];
        this.output[i][0] -= learningRate * gradient * vector[0];
        this.output[i][1] -= learningRate * gradient * vector[1];
        this.bias[i] -= learningRate * gradient;
      }
      vector[0] -= learningRate * gradX;
      vector[1] -= learningRate * gradY;
    }
    this.epoch++;
    const loss = this.loss();
    this.history.push(loss);
    return loss;
  }

  neighbors(focus: string): { word: string; score: number }[] {
    const row = FOCUS_WORDS.indexOf(focus);
    if (row < 0) throw new Error(`Unknown focus word: ${focus}`);
    return FOCUS_WORDS.map((word, i) => ({ word, score: cosine(this.input[row], this.input[i]) }))
      .filter(item => item.word !== focus)
      .sort((a, b) => b.score - a.score);
  }
}
