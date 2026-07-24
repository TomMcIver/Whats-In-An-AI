# What's In An AI

Machine learning models you can actually **train in the browser** and watch learn —
weights, biases and activations updating live, not diagrams of them.

No ML library. The networks are written from scratch in TypeScript so every
weight, gradient and activation is a plain number the UI can read and draw.

```bash
npm install
npm run dev       # http://localhost:5173
npm run verify    # correctness harness for the network engine
npm run build     # production build
```

---

## Status

| Lab | State | What it does |
|---|---|---|
| **Neural Network** | ✅ working | Trains a real MLP on 2-D data. Live weights, biases, activations, decision boundary, loss curve. |
| Classical | ⬜ not built | Linear/logistic regression, k-means, decision tree, k-NN fitting step by step. |
| Convolution | ⬜ not built | Filters sliding over an input, feature maps, filters changing during training. |
| Attention | ⬜ not built | Live attention matrix and per-head weights over a typed sentence. |

Unbuilt tabs are visible but disabled — they are placeholders, not broken features.

---

## The Neural Network lab

A genuine multilayer perceptron trained with mini-batch SGD, one epoch per tick.

**What you can see**
- **Edges** — one per weight. Colour is the sign, thickness and opacity are the magnitude.
  Watch them thicken and flip sign as the network learns.
- **Node rings** — each unit's bias, warm for positive and cool for negative.
- **Node fill** — that unit's activation for the probe point, so you can trace information
  moving forward through the layers.
- **Decision boundary** — the network's prediction resampled over the whole plane every epoch.
- **Loss curve** — training solid, held-out test dashed. When the gap opens, that's overfitting,
  and on the default settings it does.
- **Weights & biases table** — every parameter and its last gradient step, live. Click any node
  to isolate the weights feeding it.

**What you can change**

Dataset (circle, XOR, gaussian, spiral, moons), sample count, noise, input features
(X₁, X₂, X₁², X₂², X₁X₂, sin X₁, sin X₂), hidden layer count and width, activation
(tanh / ReLU / sigmoid / linear), learning rate, batch size and seed.

**Things worth trying**
- Set hidden layers to **0** on XOR. It cannot get above chance — a single layer draws a single
  line. Add one hidden layer and it solves it immediately.
- Set activation to **linear**. Depth stops helping entirely, because stacked linear layers
  collapse into one linear layer.
- Turn on **X₁²** and **X₂²** for the circle dataset. A network with no hidden layer can now
  solve it — good features and depth substitute for one another.
- Push the **learning rate** to 1.0 and watch training diverge.

---

## Is the maths right?

`npm run verify` checks the engine two ways:

1. **Gradient check** — the analytic gradient from backprop against a numerical estimate.
   Current agreement: **1.5 × 10⁻¹¹** relative error.
2. **Learning behaviour**, including a case that must *fail*:

```
  gauss  [2,1]       loss 0.268 -> 0.038   train  99.0%   test  97.8%
  xor    [2,1]       loss 0.911 -> 0.701   train  52.4%   test  44.4%   <- must fail
  xor    [2,4,1]     loss 0.692 -> 0.026   train 100.0%   test  98.9%
  circle [2,6,1]     loss 0.695 -> 0.012   train 100.0%   test  98.9%
  spiral [2,8,8,1]   loss 0.716 -> 0.215   train  91.0%   test  85.6%
```

A model that passed the XOR-without-a-hidden-layer case would be wrong, not impressive.

---

## Layout

```
src/
  lib/
    rng.ts          seeded RNG — training runs are reproducible
    nn.ts           the MLP: forward, backprop, mini-batch SGD
    datasets.ts     2-D toy datasets and input feature transforms
  components/
    NetworkGraph.tsx  the live network diagram
    Charts.tsx        loss curve + decision boundary (canvas)
  labs/
    NeuralNetLab.tsx  training loop, controls, readouts
  App.tsx           shell and tab routing
  styles.css
scripts/
  verify-nn.ts      correctness harness (npm run verify)
```

Internal imports use explicit `.ts` / `.tsx` extensions so the same source runs
under both Vite and bare Node — which is what lets `npm run verify` execute the
real engine rather than a copy of it.

---

## Branch workflow

Nothing lands on `main` directly.

```
feat/*  ->  staging  ->  main
fix/*   ->  staging  ->  main
docs/*  ->  staging  ->  main
```

`staging` is the integration branch; merges use `--no-ff` so each piece of work
stays a visible, revertable unit. `main` is only fast-forwarded from `staging`
once staging is verified.

---

## No AI files

`.gitignore` blocks assistant artefacts (`CLAUDE.md`, `.claude/`, `.cursor/`,
`.aider*`, Copilot instruction files) and model binaries and data (`*.pt`, `*.h5`,
`*.onnx`, `*.safetensors`, `*.ckpt`, `*.gguf`, `checkpoints/`, `weights/`,
`datasets/`, `wandb/`, `mlruns/`), along with `node_modules/` and `dist/`.

The repo stays source-only.

---

## History

This started as a static reference guide to 97 model architectures. That version is
still in git history on the `feat/*` branches from before the React rewrite, if the
write-ups are ever wanted again.
