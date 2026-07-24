/* Graph neural networks — for data whose structure is a network of
   relationships rather than a grid or a sequence. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'graph',
    name: 'Graph Networks',
    order: 7,
    models: [

    {
      id: 'gnn',
      name: 'Graph Neural Network',
      aka: 'GNN, Message Passing Network',
      year: '2005',
      learns: 'Supervised / self-supervised',
      dataFor: 'Graphs, networks, molecules',
      tagline: 'Every node updates itself by listening to its neighbours, repeatedly.',
      diagram: { type: 'graph', highlight: 4, caption: 'one round of message passing from the highlighted node' },
      how: [
        'Give every node a feature vector describing what it is.',
        'Each node collects messages from its immediate neighbours and aggregates them — usually by sum, mean or max.',
        'The node combines that aggregate with its own current state to produce an updated state.',
        'Repeat. After *k* rounds every node knows about everything within *k* hops of itself.'
      ],
      keyIdea: 'The aggregation must be *permutation invariant* — a node\'s neighbours have no inherent order, so the result cannot depend on how you list them. That single constraint is what distinguishes graph networks from every other architecture here.',
      strength: 'Handles irregular structure that grids and sequences cannot represent.',
      limitation: '*Over-smoothing*: stack too many rounds and every node converges to the same vector.',
      usedFor: ['molecules', 'social networks', 'recommendation', 'traffic', 'fraud rings']
    },

    {
      id: 'gcn',
      name: 'Graph Convolutional Network',
      aka: 'GCN',
      year: '2016',
      learns: 'Semi-supervised',
      dataFor: 'Graphs with node features',
      tagline: 'Average your neighbours\' features, transform, repeat.',
      diagram: {
        type: 'graph', highlight: 1,
        caption: 'neighbours are averaged with degree-based weights'
      },
      how: [
        'For each node, take the mean of its own features and those of its neighbours.',
        'Normalise by node degree, so a node with 500 connections does not dominate one with 3.',
        'Apply a shared learned weight matrix and a non-linearity.',
        'Two or three layers is usually optimal — deeper tends to over-smooth.'
      ],
      keyIdea: 'The graph analogue of a convolution. A CNN averages over a fixed spatial neighbourhood; a GCN averages over graph neighbours, which vary in number and have no ordering.',
      strength: 'Simple, fast, and remarkably effective with very few labelled nodes.',
      limitation: 'Treats all neighbours as equally important, and needs the whole graph in memory.',
      usedFor: ['citation networks', 'node classification', 'semi-supervised learning']
    },

    {
      id: 'gat',
      name: 'Graph Attention Network',
      aka: 'GAT',
      year: '2017',
      learns: 'Supervised',
      dataFor: 'Graphs with node features',
      tagline: 'Learn how much each neighbour should count, instead of averaging equally.',
      diagram: {
        type: 'graph', highlight: 4,
        caption: 'edge weights are learned, not fixed by degree'
      },
      how: [
        'For each edge, compute an attention score between the two nodes\' features.',
        'Normalise the scores across each node\'s neighbours so they sum to one.',
        'Aggregate neighbour features weighted by those scores rather than uniformly.',
        'Use several attention heads in parallel and concatenate, as a transformer does.'
      ],
      keyIdea: 'Self-attention applied to graph edges. In a citation network one reference may be far more relevant than another — GCN cannot express that, GAT learns it.',
      strength: 'Handles varying neighbour importance, and the attention weights are interpretable.',
      limitation: 'More expensive than GCN, and can overfit on small graphs.',
      usedFor: ['molecular property prediction', 'knowledge graphs', 'recommendation']
    },

    {
      id: 'graphsage',
      name: 'GraphSAGE',
      year: '2017',
      learns: 'Supervised / unsupervised',
      dataFor: 'Very large graphs',
      tagline: 'Sample a fixed number of neighbours so it scales to billions of nodes.',
      diagram: {
        type: 'graph', highlight: 2,
        nodes: [[.18,.28],[.40,.16],[.60,.30],[.28,.62],[.52,.68],[.76,.54],[.88,.26]],
        edges: [[2,1],[2,5],[2,6],[1,0],[5,4],[0,3],[3,4]],
        caption: 'only a sampled subset of neighbours is used per step'
      },
      how: [
        'Rather than aggregating over *all* neighbours, sample a fixed number — say 25 — at each hop.',
        'Aggregate the sample with a learned function: mean, max-pool, or an LSTM over the sampled set.',
        'Train the aggregator function itself, not one embedding per node.',
        'Because the function is what is learned, it can be applied to nodes that did not exist during training.'
      ],
      keyIdea: 'It is *inductive*. Earlier methods learned a fixed embedding table and were useless for new nodes; GraphSAGE learns a rule for computing an embedding, so a user who joins tomorrow can be embedded immediately.',
      strength: 'Scales to graphs with billions of edges, and generalises to unseen nodes.',
      limitation: 'Sampling introduces variance, so results vary slightly run to run.',
      usedFor: ['production recommendation', 'social network feeds', 'large-scale search']
    },

    {
      id: 'gin',
      name: 'Graph Isomorphism Network',
      aka: 'GIN',
      year: '2018',
      learns: 'Supervised',
      dataFor: 'Whole-graph classification',
      tagline: 'The most expressive message-passing network that standard theory allows.',
      diagram: {
        type: 'graph', highlight: 0,
        caption: 'sum aggregation preserves neighbourhood multiplicity'
      },
      how: [
        'Observe that mean and max aggregation lose information: they cannot distinguish a node with two identical neighbours from one with three.',
        'Use *sum* aggregation instead, which preserves how many neighbours of each type there are.',
        'Follow it with a multilayer perceptron, which can represent any function of that sum.',
        'This makes the network provably as powerful as the Weisfeiler-Lehman graph isomorphism test.'
      ],
      keyIdea: 'The theoretical ceiling for this family. Message-passing networks cannot exceed the 1-WL test, and GIN reaches it — which also means no such network can distinguish certain structurally different graphs, a genuine and permanent limitation.',
      strength: 'Maximum discriminative power within the message-passing framework.',
      limitation: 'Bounded by 1-WL, so some non-isomorphic graphs are indistinguishable no matter how it is trained.',
      usedFor: ['molecular classification', 'graph-level prediction', 'theoretical benchmarks']
    },

    {
      id: 'graph-transformer',
      name: 'Graph Transformer',
      aka: 'Graphormer, SAN',
      year: '2020',
      learns: 'Supervised',
      dataFor: 'Molecules, small graphs',
      tagline: 'Let every node attend to every other, with structure added as a bias.',
      diagram: {
        type: 'matrix', n: 6, pattern: 'random',
        top: 'all nodes →', side: 'node',
        caption: 'full attention, with graph distance encoded as a bias term'
      },
      how: [
        'Treat the nodes as a set and apply full self-attention, ignoring the edges initially.',
        'Reintroduce structure by adding a bias to the attention scores based on shortest-path distance between nodes.',
        'Encode node degree as an additional feature so the model knows local connectivity.',
        'Encode edge features along the path between each pair of nodes.'
      ],
      keyIdea: 'It escapes the two structural problems of message passing at once — over-smoothing and the 1-WL ceiling — because information no longer has to travel hop by hop. The graph becomes a bias on attention rather than a constraint on communication.',
      strength: 'State of the art on molecular property prediction; no over-smoothing.',
      limitation: 'Quadratic in node count, so it is limited to graphs of a few hundred nodes.',
      usedFor: ['drug discovery', 'quantum chemistry', 'molecular benchmarks']
    },

    {
      id: 'knowledge-graph-embedding',
      name: 'Knowledge Graph Embeddings',
      aka: 'TransE, RotatE, ComplEx',
      year: '2013',
      learns: 'Self-supervised',
      dataFor: 'Triples (subject, relation, object)',
      tagline: 'Embed entities and relations so that valid facts become simple geometry.',
      diagram: {
        type: 'scatter',
        a: [[.18,.30],[.24,.24],[.30,.36]],
        b: [[.62,.62],[.68,.56],[.74,.68]],
        c: [[.44,.46]],
        caption: 'Paris + capital-of ≈ France'
      },
      how: [
        'Represent every entity and every relation type as a vector.',
        'Impose a geometric rule for validity — *TransE* requires subject + relation ≈ object.',
        'Train on known true triples, generating false ones by corrupting them, and push true triples to satisfy the rule.',
        'Predict missing facts by checking which entity best completes the geometry.'
      ],
      keyIdea: 'Turning symbolic facts into geometry makes *inference* into arithmetic. Whether an unstated fact holds becomes a question of vector distance rather than logical deduction.',
      strength: 'Scales to millions of entities and predicts plausible missing links.',
      limitation: 'Simple models like TransE cannot represent one-to-many or symmetric relations properly.',
      usedFor: ['knowledge base completion', 'search', 'question answering', 'recommendation']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
