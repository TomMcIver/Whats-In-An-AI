/* Neural network foundations — the pieces every deep architecture
   is assembled from. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'neural-foundations',
    name: 'Neural Foundations',
    order: 2,
    models: [

    {
      id: 'perceptron',
      name: 'Perceptron',
      year: '1958',
      learns: 'Supervised',
      dataFor: 'Numeric vectors',
      tagline: 'One artificial neuron: weigh the inputs, add them up, fire or don\'t.',
      diagram: {
        type: 'layers',
        cols: [{ n: 4, kind: 'in', label: 'inputs', sub: 'x₁…x₄' }, { n: 1, kind: 'out', label: 'output', sub: 'fire / not' }]
      },
      how: [
        'Each input arrives with a *weight* saying how much it matters.',
        'Multiply every input by its weight, sum them, and add a bias term.',
        'If the total clears a threshold, output 1; otherwise output 0.',
        'When it gets an answer wrong, nudge the weights toward the correct response and try again.'
      ],
      keyIdea: 'The ancestor of everything that follows. Its famous limitation — it cannot learn XOR, because XOR is not linearly separable — is exactly what forced the invention of hidden layers.',
      strength: 'Provably converges if the data is linearly separable.',
      limitation: 'Cannot solve anything that is not linearly separable. A single neuron draws a single line.',
      usedFor: ['teaching', 'historical interest', 'linear classification']
    },

    {
      id: 'mlp',
      name: 'Multilayer Perceptron',
      aka: 'Feedforward Network, Dense Network, ANN',
      year: '1986',
      learns: 'Supervised',
      dataFor: 'Numeric vectors',
      tagline: 'Stack layers of neurons and it can approximate essentially any function.',
      diagram: {
        type: 'layers',
        cols: [
          { n: 4, kind: 'in', label: 'input' },
          { n: 6, kind: 'hidden', label: 'hidden' },
          { n: 5, kind: 'hidden', label: 'hidden' },
          { n: 2, kind: 'out', label: 'output' }
        ]
      },
      how: [
        'Arrange neurons in layers. Every neuron in one layer connects to every neuron in the next — hence "fully connected".',
        'Data flows forward: each layer computes a weighted sum, then applies a non-linear activation function.',
        'That non-linearity is essential. Without it, stacking layers would collapse into a single linear layer, and depth would buy nothing.',
        'Compare the output to the truth, then use backpropagation to assign blame and adjust every weight.'
      ],
      keyIdea: 'The *universal approximation theorem* says one sufficiently wide hidden layer can approximate any continuous function. In practice depth is far more efficient than width — which is why we build deep networks, not fat ones.',
      strength: 'General-purpose. Given enough data it learns features you never specified.',
      limitation: 'Ignores structure. It has no idea that neighbouring pixels are related, so it needs far more data than a CNN for images.',
      usedFor: ['tabular prediction', 'the final layers of larger networks', 'function approximation']
    },

    {
      id: 'activation-functions',
      name: 'Activation Functions',
      aka: 'ReLU, sigmoid, tanh, GELU',
      year: '1943 →',
      learns: 'Component',
      dataFor: 'Any layer output',
      tagline: 'The bend in each neuron — without it, depth would be pointless.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Σ w·x + b', kind: 'in' },
          { label: 'ReLU\nmax(0, x)', kind: 'attn' },
          { label: 'output', kind: 'out' }
        ],
        caption: 'the non-linearity is what lets stacked layers do more than one layer'
      },
      how: [
        '*Sigmoid* squashes to 0–1. Smooth, but saturates: for large inputs the gradient vanishes and learning stalls.',
        '*Tanh* squashes to −1 to 1. Zero-centred, which helps, but saturates the same way.',
        '*ReLU* outputs max(0, x) — negative in, zero out. Trivially cheap, does not saturate on the positive side, and is the reason deep nets became trainable.',
        '*GELU* and *SiLU* are smooth variants of ReLU that perform slightly better in transformers, and are what modern language models actually use.'
      ],
      keyIdea: 'Stack two linear layers and you get another linear layer. The activation function is the *only* reason a deep network is more expressive than a shallow one.',
      strength: 'ReLU made deep learning practical — it fixed the vanishing gradient that killed sigmoid networks.',
      limitation: 'ReLU units can "die": if a neuron always outputs zero its gradient is zero forever. Leaky ReLU exists to patch this.',
      usedFor: ['every neural network ever built']
    },

    {
      id: 'backpropagation',
      name: 'Backpropagation',
      year: '1986',
      learns: 'Training algorithm',
      dataFor: 'Any differentiable network',
      tagline: 'Work out how much each weight contributed to the error, layer by layer, backwards.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Forward\npass', kind: 'in' },
          { label: 'Loss', kind: 'attn' },
          { label: 'Gradients\nbackward', kind: 'hidden' },
          { label: 'Update\nweights', kind: 'out' }
        ],
        caption: 'the chain rule, applied efficiently across millions of parameters'
      },
      how: [
        'Run the input forward through the network and compute the loss — how wrong the output was.',
        'At the output layer, work out how the loss changes with respect to each weight there.',
        'Propagate that signal backwards using the *chain rule*, reusing each layer\'s result to compute the layer before it.',
        'Nudge every weight a small step in the direction that reduces loss. Repeat for millions of batches.'
      ],
      keyIdea: 'Not an architecture but the algorithm that makes all of them trainable. The insight is efficiency: computing gradients naively would cost one full pass per weight, whereas backprop gets all of them in a single backward pass.',
      strength: 'Exact gradients, and computationally the same order of cost as the forward pass.',
      limitation: 'Requires everything to be differentiable, and gradients can vanish or explode across many layers.',
      usedFor: ['training every neural network in this guide']
    },

    {
      id: 'gradient-descent',
      name: 'Gradient Descent & Optimisers',
      aka: 'SGD, Adam, AdamW',
      year: '1847 →',
      learns: 'Training algorithm',
      dataFor: 'Any differentiable loss',
      tagline: 'Roll downhill on the loss surface, one small step at a time.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Batch', kind: 'in' },
          { label: 'Gradient', kind: 'hidden' },
          { label: 'Momentum\n+ scaling', kind: 'attn' },
          { label: 'Step', kind: 'out' }
        ],
        caption: 'Adam adapts the step size per parameter'
      },
      how: [
        'The gradient points uphill on the loss surface, so step in the opposite direction.',
        '*Stochastic* gradient descent estimates that gradient from a small batch rather than the whole dataset — noisier, but vastly faster.',
        '*Momentum* accumulates a running average of past gradients, so the optimiser keeps rolling through small bumps instead of getting stuck.',
        '*Adam* additionally keeps a per-parameter estimate of gradient magnitude, giving rarely-updated parameters larger steps.'
      ],
      keyIdea: 'The learning rate is the single most important hyperparameter in deep learning. Too large and training diverges; too small and it never arrives. Almost everything else is secondary.',
      strength: 'Adam works acceptably on almost anything with almost no tuning.',
      limitation: 'Only finds a local minimum. In practice, high-dimensional loss surfaces have many equally good ones, so this matters less than it sounds.',
      usedFor: ['training every model in this guide']
    },

    {
      id: 'regularisation',
      name: 'Regularisation',
      aka: 'Dropout, weight decay, batch norm, early stopping',
      year: '1990s →',
      learns: 'Component',
      dataFor: 'Any network',
      tagline: 'Deliberately handicap the network so it generalises instead of memorising.',
      diagram: {
        type: 'layers',
        cols: [
          { n: 4, kind: 'in', label: 'input' },
          { n: 6, kind: 'hidden', label: 'dropout', sub: '50% off' },
          { n: 4, kind: 'out', label: 'output' }
        ]
      },
      how: [
        '*Dropout* randomly switches off half the neurons on each training step, so no single neuron can become indispensable.',
        '*Weight decay* adds a penalty for large weights, pushing the model toward simpler explanations.',
        '*Batch normalisation* rescales each layer\'s outputs to a stable distribution, which speeds training and incidentally regularises.',
        '*Early stopping* simply halts training when validation loss stops improving, before the model starts memorising.'
      ],
      keyIdea: 'A model with millions of parameters can memorise the training set outright. Regularisation is the family of tricks that force it to find patterns that actually generalise.',
      strength: 'The difference between a model that works in a notebook and one that works in production.',
      limitation: 'Too much and the model underfits — it never learns the real signal either.',
      usedFor: ['every deep network trained on finite data']
    },

    {
      id: 'embeddings',
      name: 'Embeddings',
      aka: 'Word2Vec, GloVe, vector representations',
      year: '2013',
      learns: 'Self-supervised',
      dataFor: 'Discrete tokens',
      tagline: 'Turn words or items into coordinates, where nearby means similar.',
      diagram: {
        type: 'scatter',
        a: [[.16,.30],[.22,.24],[.28,.34],[.20,.40]],
        b: [[.70,.62],[.78,.56],[.72,.72],[.82,.66]],
        c: [[.48,.20],[.54,.28]],
        caption: 'king − man + woman lands near queen'
      },
      how: [
        'Assign every word a vector of a few hundred numbers, initialised randomly.',
        'Train by prediction: given a word, guess its neighbours in real sentences (or the reverse).',
        'Words appearing in similar contexts get pushed toward similar vectors, because that makes the prediction easier.',
        'The geometry ends up meaningful — directions in the space encode relationships like gender, tense and plurality.'
      ],
      keyIdea: 'Discrete symbols have no notion of similarity — "cat" and "dog" are just two different IDs. Embeddings give them a continuous space where similarity is distance, which is what lets neural networks work with language at all.',
      strength: 'Transfers well. Embeddings trained on a large corpus improve small downstream tasks.',
      limitation: 'Classic embeddings give each word one fixed vector, so "bank" gets a single blurred meaning. Transformers fixed this with contextual embeddings.',
      usedFor: ['NLP', 'recommendation', 'search', 'any categorical feature']
    },

    {
      id: 'loss-functions',
      name: 'Loss Functions',
      aka: 'Cross-entropy, MSE, contrastive loss',
      year: 'various',
      learns: 'Component',
      dataFor: 'Any supervised task',
      tagline: 'The definition of "wrong" — and therefore of what the model will become.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Prediction', kind: 'in' },
          { label: 'Truth', kind: 'in' },
          { label: 'Loss', kind: 'attn' },
          { label: 'Gradient', kind: 'out' }
        ],
        caption: 'whatever you measure is what the model optimises'
      },
      how: [
        '*Mean squared error* penalises the square of the gap — standard for regression, and heavily influenced by outliers.',
        '*Cross-entropy* penalises confident wrong answers far more than hesitant ones — the default for classification.',
        '*Contrastive* and *triplet* losses do not score a single prediction; they pull similar pairs together and push dissimilar ones apart.',
        'Custom losses encode domain priorities — weighting false negatives more heavily in medical screening, for instance.'
      ],
      keyIdea: 'The loss function is the only thing the model actually cares about. If it is misaligned with what you want, the model will optimise the metric and defeat your intent — a failure mode that looks like a bug but is really a specification error.',
      strength: 'The cleanest place to encode what actually matters about the problem.',
      limitation: 'Easy to get subtly wrong, and a wrong loss produces a confidently wrong model.',
      usedFor: ['every trained model']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
