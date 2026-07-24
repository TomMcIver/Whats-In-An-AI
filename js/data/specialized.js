/* Specialised and historical architectures — the branches of the family
   tree that took a different route, plus the ideas the mainstream forgot
   and then rediscovered. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'specialized',
    name: 'Specialised & Historical',
    order: 9,
    models: [

    {
      id: 'hopfield',
      name: 'Hopfield Network',
      year: '1982',
      learns: 'Unsupervised (associative memory)',
      dataFor: 'Binary patterns',
      tagline: 'A network that settles into the stored pattern nearest what you showed it.',
      diagram: {
        type: 'graph',
        nodes: [[.30,.22],[.62,.20],[.80,.48],[.66,.76],[.34,.78],[.18,.50]],
        edges: [[0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]],
        caption: 'every neuron connects to every other, symmetrically'
      },
      how: [
        'Connect every neuron to every other with symmetric weights.',
        'Store patterns by setting weights so each stored pattern becomes a stable state — an energy minimum.',
        'Present a corrupted or partial pattern and let the neurons update repeatedly.',
        'The network rolls downhill in energy and settles into the nearest stored pattern, reconstructing the whole from a fragment.'
      ],
      keyIdea: '*Content-addressable memory* — you retrieve by giving part of the content, not an address. Largely a curiosity for decades, then modern Hopfield networks were shown to be mathematically equivalent to transformer attention, which made it suddenly relevant again.',
      strength: 'Genuine pattern completion, and a clean theoretical account via energy minimisation.',
      limitation: 'Capacity is only about 0.14 patterns per neuron; beyond that stored memories interfere and corrupt.',
      usedFor: ['associative memory', 'optimisation', 'theoretical links to attention']
    },

    {
      id: 'boltzmann-machine',
      name: 'Boltzmann Machine & RBM',
      aka: 'Restricted Boltzmann Machine',
      year: '1985 / 2006',
      learns: 'Unsupervised (generative)',
      dataFor: 'Binary features',
      tagline: 'A stochastic network that learns the probability distribution of its inputs.',
      diagram: {
        type: 'layers',
        cols: [
          { n: 5, kind: 'in', label: 'visible', sub: 'data' },
          { n: 3, kind: 'hidden', label: 'hidden', sub: 'features' }
        ]
      },
      how: [
        'Split the units into a *visible* layer holding data and a *hidden* layer learning features.',
        '"Restricted" means no connections within a layer — only between them. This is what makes training tractable.',
        'Units switch on and off probabilistically, so the network defines a distribution rather than a single output.',
        'Train with *contrastive divergence*: raise the probability of observed data and lower it for the network\'s own fantasies.'
      ],
      keyIdea: 'Stacking RBMs and training them one layer at a time was the *unsupervised pretraining* that first made deep networks trainable in 2006 — the trigger for the deep learning revival. Better initialisation and ReLU later made it unnecessary.',
      strength: 'Historically pivotal, and learns a genuine generative model of the data.',
      limitation: 'Slow, awkward to train, and comprehensively superseded.',
      usedFor: ['historical significance', 'collaborative filtering', 'feature learning']
    },

    {
      id: 'dbn',
      name: 'Deep Belief Network',
      aka: 'DBN',
      year: '2006',
      learns: 'Unsupervised → supervised',
      dataFor: 'Images, speech',
      tagline: 'A stack of RBMs, trained one layer at a time then fine-tuned.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'RBM layer 3', kind: 'out' },
          { label: 'RBM layer 2', kind: 'hidden' },
          { label: 'RBM layer 1', kind: 'hidden' },
          { label: 'input data', kind: 'in' }
        ],
        caption: 'greedy layer-wise pretraining, then backprop over the whole stack'
      },
      how: [
        'Train the first RBM on the raw data.',
        'Freeze it, use its hidden activations as input, and train a second RBM on those.',
        'Repeat for several layers, each learning progressively more abstract features without any labels.',
        'Finally add an output layer and fine-tune the whole stack with ordinary backpropagation.'
      ],
      keyIdea: 'It solved the problem that made deep networks untrainable in 2006 — random initialisation left gradients too weak to reach early layers. Layer-wise pretraining put the weights somewhere sensible first. The problem was later solved better, but this is what proved depth was worth pursuing.',
      strength: 'Made deep architectures work when nothing else did, and exploits unlabelled data.',
      limitation: 'Obsolete. ReLU, better initialisation and batch norm removed the need entirely.',
      usedFor: ['historical reference', 'the architecture that restarted deep learning']
    },

    {
      id: 'som',
      name: 'Self-Organising Map',
      aka: 'SOM, Kohonen Map',
      year: '1982',
      learns: 'Unsupervised',
      dataFor: 'High-dimensional vectors',
      tagline: 'A grid of neurons that arranges itself to mirror the shape of the data.',
      diagram: {
        type: 'graph',
        nodes: [[.20,.20],[.50,.16],[.80,.22],[.18,.50],[.50,.50],[.82,.52],[.22,.80],[.50,.84],[.78,.80]],
        edges: [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],[0,3],[3,6],[1,4],[4,7],[2,5],[5,8]],
        highlight: 4,
        caption: 'the winning neuron and its grid neighbours move together'
      },
      how: [
        'Lay out neurons on a fixed 2-D grid, each holding a vector the same size as the input.',
        'For each data point, find the neuron whose vector is closest — the *best matching unit*.',
        'Move that neuron\'s vector toward the data point, and move its *grid neighbours* too, by a smaller amount.',
        'Shrink the neighbourhood radius over time, so the map goes from coarse global arrangement to fine local tuning.'
      ],
      keyIdea: 'Dragging neighbours along is the whole mechanism. It forces nearby grid positions to represent similar data, so the 2-D grid ends up as a topology-preserving map of a high-dimensional space — readable directly as a picture.',
      strength: 'Produces an interpretable 2-D visualisation while also clustering.',
      limitation: 'Grid size and shape must be fixed in advance, and it scales poorly.',
      usedFor: ['data visualisation', 'clustering', 'process monitoring']
    },

    {
      id: 'capsule-network',
      name: 'Capsule Network',
      aka: 'CapsNet',
      year: '2017',
      learns: 'Supervised',
      dataFor: 'Images',
      tagline: 'Replace scalar neurons with vectors that encode pose as well as presence.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'conv\nfeatures', kind: 'in' },
          { label: 'primary\ncapsules', kind: 'hidden' },
          { label: 'routing by\nagreement', kind: 'attn' },
          { label: 'class\ncapsules', kind: 'out' }
        ],
        caption: 'vector length = probability, orientation = pose'
      },
      how: [
        'Each *capsule* outputs a vector rather than a single number. Its length encodes how confident the feature is present; its direction encodes pose — rotation, scale, position.',
        'Lower capsules predict what higher capsules should be, given their own pose.',
        '*Routing by agreement*: when several low-level capsules make consistent predictions about a higher one, their connection is strengthened.',
        'This means a nose, eyes and mouth only activate "face" if their poses are mutually consistent.'
      ],
      keyIdea: 'A direct attack on a real CNN weakness: max pooling discards spatial relationships, so a CNN can be fooled by a face with the eyes and mouth rearranged. Capsules keep the relationships explicit.',
      strength: 'Better generalisation to unseen viewpoints, and handles overlapping objects well.',
      limitation: 'Slow, and never scaled beyond small datasets. An influential idea more than a practical tool.',
      usedFor: ['research', 'viewpoint-invariant recognition', 'small-image benchmarks']
    },

    {
      id: 'siamese-network',
      name: 'Siamese & Triplet Networks',
      year: '1993',
      learns: 'Metric learning',
      dataFor: 'Pairs, few-shot problems',
      tagline: 'Two copies of one network, trained to measure whether inputs match.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Network A', rightTitle: 'Network B',
        left: ['image 1', 'embedding'], right: ['image 2', 'embedding'],
        bridge: 'distance', leftIn: 'shared weights', rightOut: 'same or different?'
      },
      how: [
        'Run two inputs through the *same* network with the same weights, producing an embedding for each.',
        'Compute the distance between the embeddings.',
        'Train so matching pairs are close and non-matching pairs are far apart.',
        'The *triplet* variant uses three inputs at once — an anchor, a positive and a negative — and requires the positive to be closer than the negative by a margin.'
      ],
      keyIdea: 'It learns a *similarity function*, not a set of classes. That is what makes face recognition work: you cannot retrain a classifier every time an employee joins, but you can compare a new photo to a stored embedding.',
      strength: 'Handles classes never seen in training, and needs only one example per identity.',
      limitation: 'Training needs careful mining of hard pairs, or it converges to a trivial solution.',
      usedFor: ['face verification', 'signature verification', 'one-shot learning', 'duplicate detection']
    },

    {
      id: 'ntm-dnc',
      name: 'Neural Turing Machine',
      aka: 'NTM, Differentiable Neural Computer',
      year: '2014',
      learns: 'Supervised',
      dataFor: 'Algorithmic tasks',
      tagline: 'A network with an external memory it learns to read from and write to.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'controller\nnetwork', kind: 'in' },
          { label: 'read/write\nheads', kind: 'attn' },
          { label: 'memory\nmatrix', kind: 'hidden' },
          { label: 'output', kind: 'out' }
        ],
        caption: 'addressing is soft, so the whole thing is differentiable'
      },
      how: [
        'Pair a neural controller with an external memory matrix, like a CPU with RAM.',
        'The controller emits read and write *heads* that address the memory.',
        'Addressing is *soft*: rather than picking one location, it produces weights over all of them, which keeps the operation differentiable.',
        'The whole system trains end to end with backpropagation, learning algorithms like copying and sorting from examples alone.'
      ],
      keyIdea: 'It separates computation from storage — the architectural insight behind actual computers, applied to neural networks. Attention over an external memory is a direct ancestor of retrieval-augmented generation.',
      strength: 'Learns genuine algorithms that generalise to longer inputs than trained on.',
      limitation: 'Slow, unstable, and hard to scale. Superseded by transformers and explicit retrieval.',
      usedFor: ['algorithmic reasoning research', 'conceptual ancestor of RAG']
    },

    {
      id: 'spiking-nn',
      name: 'Spiking Neural Network',
      aka: 'SNN, neuromorphic computing',
      year: '1990s',
      learns: 'Various',
      dataFor: 'Event streams, sensors',
      tagline: 'Neurons that fire discrete spikes in time, as biological ones do.',
      diagram: {
        type: 'sequence',
        tokens: ['│', ' ', '│', '│', ' ', '│'],
        arrows: false, predictLast: false,
        below: 'information is in the timing, not the magnitude',
        caption: 'neurons stay silent until their membrane potential crosses threshold'
      },
      how: [
        'Each neuron accumulates incoming charge in a *membrane potential* over time.',
        'When that potential crosses a threshold, the neuron emits a spike and resets.',
        'Information is carried by *when* spikes occur, not by a continuous activation value.',
        'Because the spike is a step function it is not differentiable, so training uses surrogate gradients or biologically-inspired local rules.'
      ],
      keyIdea: 'Neurons stay silent unless there is something to say, so power is consumed only on events. On neuromorphic hardware this can be orders of magnitude more efficient than a conventional network — the appeal is energy, not accuracy.',
      strength: 'Extremely low power on suitable hardware, and naturally suited to event-based sensors.',
      limitation: 'Non-differentiable spikes make training awkward, and accuracy trails conventional nets.',
      usedFor: ['neuromorphic chips', 'event cameras', 'always-on sensing', 'computational neuroscience']
    },

    {
      id: 'rbf-network',
      name: 'Radial Basis Function Network',
      aka: 'RBF Network',
      year: '1988',
      learns: 'Supervised',
      dataFor: 'Numeric vectors',
      tagline: 'Hidden units that respond to how close the input is to a stored centre.',
      diagram: {
        type: 'layers',
        cols: [
          { n: 3, kind: 'in', label: 'input' },
          { n: 5, kind: 'hidden', label: 'RBF units', sub: 'distance to centre' },
          { n: 1, kind: 'out', label: 'linear output' }
        ]
      },
      how: [
        'Each hidden unit stores a centre point and responds according to the *distance* from the input to that centre.',
        'The response is typically a Gaussian: maximal at the centre, decaying smoothly with distance.',
        'The output layer is a simple linear combination of those responses.',
        'Centres are usually placed by clustering first, leaving only the output weights to solve — which has a closed-form solution.'
      ],
      keyIdea: 'Its activations are *local*. A standard neuron responds to a whole half-space; an RBF unit responds only near its centre. That makes training fast and behaviour interpretable, but means coverage requires many units in high dimensions.',
      strength: 'Very fast training, and it can signal when an input is far from anything it has seen.',
      limitation: 'Needs exponentially many centres as dimensionality grows.',
      usedFor: ['function approximation', 'control systems', 'interpolation', 'time series']
    },

    {
      id: 'reservoir-computing',
      name: 'Echo State & Reservoir Computing',
      aka: 'ESN, Liquid State Machine',
      year: '2001',
      learns: 'Supervised (output only)',
      dataFor: 'Time series',
      tagline: 'A large random recurrent network, left untrained, with only the readout learned.',
      diagram: {
        type: 'graph',
        nodes: [[.14,.50],[.36,.24],[.42,.54],[.34,.80],[.60,.36],[.62,.70],[.88,.50]],
        edges: [[0,1],[0,2],[0,3],[1,2],[1,4],[2,4],[2,5],[3,5],[4,5],[4,6],[5,6],[4,1],[5,2]],
        caption: 'the reservoir is random and fixed; only the output weights train'
      },
      how: [
        'Create a large recurrent network with random, fixed weights — the *reservoir*.',
        'Feed the input signal in and let it echo around, producing a rich, high-dimensional, time-varying response.',
        'Train *only* a linear readout mapping reservoir state to output. Nothing inside is ever adjusted.',
        'Because the readout is linear, training is a least-squares solve — no backpropagation, no iteration.'
      ],
      keyIdea: 'A random non-linear dynamical system already projects input into a space where the answer is often linearly separable. If that is true, there is no need to train the hard part at all.',
      strength: 'Training takes seconds, avoids all recurrent-gradient problems, and works with tiny datasets.',
      limitation: 'Performance depends on getting the random reservoir right, and it cannot learn task-specific features.',
      usedFor: ['chaotic time series', 'signal processing', 'photonic and analogue computing']
    },

    {
      id: 'nerf',
      name: 'Neural Radiance Fields',
      aka: 'NeRF',
      year: '2020',
      learns: 'Supervised (per scene)',
      dataFor: 'Multi-view photographs',
      tagline: 'A small network that stores an entire 3-D scene in its weights.',
      diagram: {
        type: 'blocks',
        items: [
          { label: '3D point\n+ view angle', kind: 'in' },
          { label: 'MLP', kind: 'hidden' },
          { label: 'colour +\ndensity', kind: 'attn' },
          { label: 'rendered\nimage', kind: 'out' }
        ],
        caption: 'the network is the scene; rendering is ray marching through it'
      },
      how: [
        'Train a small MLP that maps a 3-D coordinate plus a viewing direction to a colour and a density.',
        'Render a pixel by marching a ray through the scene, querying the network at many points along it and compositing the results.',
        'The rendering step is fully differentiable, so the error between rendered and real photographs backpropagates into the weights.',
        'Train on a few dozen photos of one scene. The finished network *is* the 3-D representation.'
      ],
      keyIdea: 'The scene is stored as a *continuous function* rather than a mesh or voxel grid — so resolution is unlimited and view-dependent effects like reflections come out naturally. Gaussian splatting has since largely replaced it on speed, but the idea of a neural field persists.',
      strength: 'Photorealistic novel views, including transparency and specular highlights.',
      limitation: 'One network per scene, hours to train, and originally very slow to render.',
      usedFor: ['3-D reconstruction', 'virtual production', 'AR/VR', 'cultural heritage scanning']
    },

    {
      id: 'neural-ode',
      name: 'Neural ODE',
      year: '2018',
      learns: 'Supervised',
      dataFor: 'Continuous-time data',
      tagline: 'Treat depth as continuous, and let a differential equation solver decide the layers.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'input\nstate', kind: 'in' },
          { label: 'dz/dt =\nf(z, t)', kind: 'attn' },
          { label: 'ODE\nsolver', kind: 'hidden' },
          { label: 'output\nstate', kind: 'out' }
        ],
        caption: 'the number of evaluations is chosen at runtime, not designed in'
      },
      how: [
        'Notice that a ResNet layer computes z + f(z) — which is one step of Euler integration.',
        'Take the limit: instead of discrete layers, define a derivative dz/dt = f(z, t) parameterised by a network.',
        'Compute the output by handing that derivative to an off-the-shelf ODE solver.',
        'Train using the *adjoint method*, which computes gradients with constant memory regardless of how many steps the solver takes.'
      ],
      keyIdea: 'Depth stops being an architectural choice and becomes a solver tolerance you can adjust at inference time. Constant memory cost during training is the practical payoff.',
      strength: 'Memory-efficient, and handles irregularly-sampled time series naturally.',
      limitation: 'Slower than a fixed network, and the solver can take many steps on stiff problems.',
      usedFor: ['irregular time series', 'physics modelling', 'continuous normalising flows']
    },

    {
      id: 'extreme-learning-machine',
      name: 'Extreme Learning Machine',
      aka: 'ELM',
      year: '2006',
      learns: 'Supervised',
      dataFor: 'Tabular, numeric',
      tagline: 'Random hidden layer, solve the output layer exactly, done in one shot.',
      diagram: {
        type: 'layers',
        cols: [
          { n: 4, kind: 'in', label: 'input' },
          { n: 7, kind: 'hidden', label: 'random', sub: 'never trained' },
          { n: 2, kind: 'out', label: 'solved', sub: 'least squares' }
        ]
      },
      how: [
        'Initialise the hidden layer weights randomly and then freeze them permanently.',
        'Push the training data through, producing a fixed matrix of hidden activations.',
        'Solve for the output weights in closed form with a least-squares pseudo-inverse.',
        'There is no iteration and no backpropagation — training is a single matrix operation.'
      ],
      keyIdea: 'The same bet as reservoir computing, applied to feedforward networks: a random non-linear projection into a high-enough dimension usually makes the problem linearly separable, so only the last layer needs solving.',
      strength: 'Training is essentially instantaneous, with no learning rate or epochs to tune.',
      limitation: 'Needs many hidden units to compensate for random features, and the claim of novelty was contested.',
      usedFor: ['fast prototyping', 'embedded systems', 'online learning']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
