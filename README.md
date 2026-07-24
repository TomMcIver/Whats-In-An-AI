# What's In An AI

Machine learning models you can actually **train in the browser** and watch learn —
weights, biases, filters and attention updating live, not diagrams of them.

**No ML library.** Every model is written from scratch in TypeScript, including
backpropagation, so each weight and gradient is a plain number the UI reads directly.

```bash
npm install
npm run dev       # http://localhost:5173
npm run verify    # gradient-checks every engine
npm run build     # production build
```

---

## The four labs

### 1. Neural Network
A multilayer perceptron trained with mini-batch SGD on 2-D data.

- **Edges** are weights — colour is the sign, thickness the magnitude
- **Node rings** are biases; **node fill** is that unit's activation for the probe point
- **Decision boundary** resampled over the whole plane every epoch
- **Loss curve** — train solid, held-out test dashed
- **Weights & biases table**, live, with the last gradient step. Click a node to isolate its inputs.

Datasets: circle, XOR, gaussian, spiral, moons. Adjustable features (X₁, X₂, X₁², X₂², X₁X₂,
sin X₁, sin X₂), layer count and width, activation, learning rate, batch size, seed.

> **Try:** set hidden layers to **0** on XOR. It cannot beat chance — one layer draws one line.
> Add a layer and it solves instantly. Or set activation to **linear** and watch depth stop
> helping entirely.

### 2. Classical
Five algorithms, each **stepped** so the fit is watchable rather than instant.

| | What you see |
|---|---|
| **Linear regression** | Gradient descent walking a line onto the data, with the exact least-squares answer drawn behind it and the residuals it is minimising |
| **Logistic regression** | The same descent through a sigmoid; shading is probability, the white line is the 0.5 boundary |
| **k-means** | Assign and move as *separate* half-steps, each point linked to its centre |
| **Decision tree** | One level per step, each split chosen by Gini gain, carving the plane into rectangles |
| **k-NN** | No fit button, because there is no training — the boundary is implied by the stored points |

> **Try:** switch linear regression to **Curved**. It converges perfectly — to the best possible
> straight line, which is a bad model. Converging is not the same as being right.

### 3. Convolution
A real CNN: `conv 3×3 → ReLU → 2×2 max pool → dense → softmax`, with backprop routed
through the pool argmax and the ReLU gate into the filters themselves.

- **Learned filters** drawn as 3×3 heatmaps. They start as random noise and organise into
  edge and orientation detectors on their own
- **Feature maps** after ReLU and after pooling, per filter, for the current image
- **Draw your own input** on the 12×12 grid and watch the prediction change

### 4. Attention
A one-layer, multi-head self-attention model trained on a copy task:
a random prefix, a separator, then the same prefix again. To predict the second half it
must look back at the matching position — which is exactly what attention is for.

- **Attention matrix per head** — rows query, columns are attended to, upper triangle
  masked out by causality
- **Per-position predictions** over the vocabulary
- Before training the matrix is a smear. After training a clean diagonal stripe appears,
  offset by the prefix length. Nobody specified that; it is just the only strategy that
  reduces the loss.

---

## Is the maths right?

`npm run verify` gradient-checks all three engines against numerical estimates and asserts
behaviour that must hold — including cases that must **fail**.

```
NEURAL NET
  rel error 1.5e-11    backprop vs numerical gradient
  gauss  [2,1]       train  99.0%   test  97.8%
  xor    [2,1]       train  52.4%   test  44.4%   <- must fail, no hidden layer
  xor    [2,4,1]     train 100.0%   test  98.9%
  spiral [2,8,8,1]   train  91.0%   test  85.6%

CONVOLUTION
  rel error 2.8e-11    gradient through conv -> ReLU -> pool -> dense -> softmax
  test acc 100%        from 33.8% (chance 25%)
  filter drift 12.67   filters genuinely moved from initialisation

ATTENTION
  rel error 0.0        gradient through the attention softmax
  test acc 100%        on unseen sequences, from 19.3% (chance 20%)
  attention concentrates 84-94% on the matching position
```

19 checks, all passing. A model that passed the XOR-without-a-hidden-layer case would be
wrong, not impressive — so that case asserts failure.

The classical lab has an independent check built into the UI: gradient descent converges to
`w=0.7525, b=0.0953`, matching the closed-form least-squares solution shown beside it.

---

## Layout

```
src/
  lib/
    rng.ts          seeded RNG — runs are reproducible
    nn.ts           MLP: forward, backprop, mini-batch SGD
    datasets.ts     2-D toy datasets and feature transforms
    classical.ts    regression, k-means, decision tree, k-NN
    conv.ts         CNN with backprop through the filters
    attention.ts    multi-head self-attention with backprop through the softmax
  components/
    NetworkGraph.tsx  live network diagram
    Charts.tsx        loss curve + decision boundary
    GridView.tsx      grid heatmaps for filters and feature maps
  labs/
    NeuralNetLab.tsx  ClassicalLab.tsx  ConvLab.tsx  AttentionLab.tsx
  App.tsx           shell and tab routing
scripts/
  verify-nn.ts  verify-conv.ts  verify-attention.ts
```

Internal imports use explicit `.ts` / `.tsx` extensions so the same source runs under both
Vite and bare Node — which is what lets the verify scripts execute the real engines rather
than a copy of them.

Training loops use `setInterval`, not `requestAnimationFrame`: rAF is throttled to zero in
background tabs and never fires in headless browsers, which made the loop both unverifiable
and liable to silently stall.

---

## Branch workflow

Nothing lands on `main` directly.

```
feat/*  ->  staging  ->  main
fix/*   ->  staging  ->  main
docs/*  ->  staging  ->  main
```

`staging` is the integration branch; merges use `--no-ff` so each piece of work stays a
visible, revertable unit. `main` is only fast-forwarded from `staging` once verified.

---

## No AI files

`.gitignore` blocks assistant artefacts (`CLAUDE.md`, `.claude/`, `.cursor/`, `.aider*`,
Copilot instruction files) and model binaries and data (`*.pt`, `*.h5`, `*.onnx`,
`*.safetensors`, `*.ckpt`, `*.gguf`, `checkpoints/`, `weights/`, `datasets/`, `wandb/`,
`mlruns/`), plus `node_modules/` and `dist/`. The repo stays source-only.

---

## History

This began as a static reference guide to 97 model architectures. That version is still in
git history on the `feat/*` branches from before the React rewrite, if those write-ups are
ever wanted again.
