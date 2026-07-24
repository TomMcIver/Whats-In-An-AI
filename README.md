# What's In An AI

A visual field guide to machine learning. **97 models** across 9 categories — every
major family from linear regression to Mamba — each one drawn as a diagram and
explained in four steps.

Open `index.html` in a browser. That's it. No build step, no dependencies, no
server, no tracking.

---

## What's in it

| Category | Models | Covers |
|---|---:|---|
| Classical ML | 15 | regression, trees, forests, boosting, SVM, k-NN, Bayes, clustering, PCA, t-SNE |
| Neural Foundations | 8 | perceptron, MLP, activations, backprop, optimisers, regularisation, embeddings, losses |
| Convolutional Nets | 12 | LeNet, AlexNet, VGG, ResNet, Inception, DenseNet, EfficientNet, MobileNet, U-Net, R-CNN, YOLO |
| Recurrent & Sequence | 8 | RNN, LSTM, GRU, BiRNN, seq2seq, attention, CTC, TCN |
| Transformers | 13 | Transformer, self-attention, positional encoding, BERT, GPT, T5, ViT, Swin, CLIP, Whisper, MoE, Mamba |
| Generative Models | 11 | autoencoder, VAE, GAN, DCGAN, StyleGAN, CycleGAN, diffusion, latent diffusion, flows, EBMs |
| Graph Networks | 7 | GNN, GCN, GAT, GraphSAGE, GIN, graph transformers, KG embeddings |
| Reinforcement Learning | 10 | Q-learning, DQN, policy gradient, actor-critic, PPO, SAC, AlphaZero, model-based, RLHF, MARL |
| Specialised & Historical | 13 | Hopfield, RBM, DBN, SOM, capsules, siamese, NTM, spiking, RBF, reservoir, NeRF, neural ODE, ELM |

Every entry has a diagram, a four-step walkthrough, the key idea, the trade-off it
makes, and what it's typically used for.

---

## Layout

```
index.html              page shell
css/style.css           all styling, light + dark themes
js/diagram.js           SVG engine — 14 reusable diagram primitives
js/app.js               grid, detail panel, search, filtering
js/data/*.js            one file per category, one file per branch
```

### Adding a model

Append an object to the `models` array in the relevant `js/data/*.js` file:

```js
{
  id:        'unique-slug',
  name:      'Model Name',
  aka:       'Other names it goes by',      // optional
  year:      '2017',
  learns:    'Supervised',
  dataFor:   'What shape of data it takes',
  tagline:   'One sentence, plain English.',
  diagram:   { type: 'layers', cols: [...] },
  how:       ['step one', 'step two', ...],  // *asterisks* render bold
  keyIdea:   'The thing worth remembering.',
  strength:  'What it is good at.',
  limitation:'Where it falls down.',
  usedFor:   ['tag', 'tag']
}
```

### Diagram types

`layers` · `blocks` · `stack` · `conv` · `recurrent` · `encdec` · `tree` · `graph`
· `scatter` · `matrix` · `cycle` · `adversarial` · `hourglass` · `sequence`

All are declarative and theme-aware — they draw from CSS custom properties, so
diagrams follow light/dark automatically. See `js/diagram.js` for each spec.

---

## Branch workflow

Work never lands on `main` directly. Three tiers:

```
feat/*  ->  staging  ->  main
fix/*   ->  staging  ->  main
docs/*  ->  staging  ->  main
```

- **`feat/*`, `fix/*`, `docs/*`** — one branch per unit of work. Each UI concern and
  each content category got its own.
- **`staging`** — integration branch. Everything merges here first, with
  `--no-ff` so each feature stays a visible, revertable unit in the history.
- **`main`** — only ever fast-forwarded from `staging` once staging is verified.

```bash
git checkout -b feat/my-thing staging
# ... work ...
git commit -m "Describe the change"
git checkout staging
git merge --no-ff feat/my-thing
# verify, then:
git checkout main
git merge staging
```

---

## No AI files

`.gitignore` blocks two classes of file from ever being committed:

- **Assistant artefacts** — `CLAUDE.md`, `.claude/`, `.cursor/`, `.aider*`,
  Copilot instruction files, `.continue/`, `.windsurfrules`, and similar.
- **Model binaries and data** — `*.pt`, `*.h5`, `*.onnx`, `*.safetensors`,
  `*.ckpt`, `*.gguf`, `*.pkl`, plus `checkpoints/`, `weights/`, `models/`,
  `datasets/`, `wandb/`, `mlruns/`.

The repo stays source-only: HTML, CSS, JS and this README.

---

## Verifying

The data files are plain JavaScript, so they can be checked outside a browser:

```bash
node --check js/data/*.js        # syntax
```

The site itself is static — open `index.html` directly, or serve the folder with
any static file server.
