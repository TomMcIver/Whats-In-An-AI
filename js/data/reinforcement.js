/* Reinforcement learning — agents that learn by acting and observing
   the consequences, rather than from a labelled dataset. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'reinforcement',
    name: 'Reinforcement Learning',
    order: 8,
    models: [

    {
      id: 'q-learning',
      name: 'Q-Learning',
      year: '1989',
      learns: 'Reinforcement',
      dataFor: 'Small discrete environments',
      tagline: 'Build a table of how good each action is in each state.',
      diagram: {
        type: 'cycle',
        left: 'Q-table', right: 'Environment',
        forward: 'best action', back: 'reward + new state'
      },
      how: [
        'Keep a table Q(state, action) estimating the total future reward of taking that action in that state.',
        'Act, observe the reward and the new state, then update the entry toward *reward + discounted best value of the next state*.',
        'That update rule — the Bellman equation — propagates value backwards from rewards to the actions that led to them.',
        'Balance *exploration* against *exploitation*: mostly take the best known action, but sometimes act randomly to discover better ones.'
      ],
      keyIdea: 'It is *off-policy*: it learns the value of the optimal policy while behaving suboptimally, so it can learn from random exploration or even from someone else\'s actions.',
      strength: 'Provably converges to optimal in finite environments, with no model of the world required.',
      limitation: 'The table has one entry per state-action pair. Chess has more states than atoms in the observable universe.',
      usedFor: ['grid worlds', 'simple control', 'teaching RL fundamentals']
    },

    {
      id: 'dqn',
      name: 'Deep Q-Network',
      aka: 'DQN',
      year: '2013',
      learns: 'Reinforcement',
      dataFor: 'High-dimensional states',
      tagline: 'Replace the Q-table with a neural network, and add two tricks to stabilise it.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'screen\npixels', kind: 'in' },
          { label: 'CNN', kind: 'hidden' },
          { label: 'Q per\naction', kind: 'attn' },
          { label: 'best\naction', kind: 'out' }
        ],
        caption: 'a network approximates Q for states never seen before'
      },
      how: [
        'Use a neural network to approximate Q(state, action), so unseen states can be evaluated by generalisation.',
        'Add an *experience replay* buffer: store past transitions and train on random samples, breaking the correlation between consecutive frames.',
        'Add a *target network* — a frozen copy updated occasionally — so the value being chased does not move every step.',
        'Feed raw pixels through a CNN, learning directly from the screen with no hand-crafted features.'
      ],
      keyIdea: 'Naively combining Q-learning with neural networks diverges, because the network is chasing a target that it changes as it learns. Replay and the target network are what make the combination stable — and they turned RL from a toy into something that could play Atari from pixels.',
      strength: 'Learned dozens of Atari games to superhuman level from raw pixels and score alone.',
      limitation: 'Only handles discrete actions, and is extremely sample-hungry — millions of frames per game.',
      usedFor: ['game playing', 'discrete control', 'RL research baselines']
    },

    {
      id: 'policy-gradient',
      name: 'Policy Gradient',
      aka: 'REINFORCE',
      year: '1992',
      learns: 'Reinforcement',
      dataFor: 'Discrete or continuous actions',
      tagline: 'Learn the policy directly — make good actions more likely, bad ones less.',
      diagram: {
        type: 'cycle',
        left: 'Policy π(a|s)', right: 'Environment',
        forward: 'sampled action', back: 'reward signal'
      },
      how: [
        'Represent the policy as a network outputting a probability distribution over actions.',
        'Run a full episode, sampling actions from that distribution.',
        'If the episode went well, increase the probability of every action taken; if badly, decrease them.',
        'Weight each adjustment by how much better or worse than average the outcome was.'
      ],
      keyIdea: 'Value methods learn *how good things are* and derive behaviour from that; policy methods learn *what to do* directly. This handles continuous action spaces naturally, where taking the max over actions is not even well-defined.',
      strength: 'Works with continuous actions, and can learn deliberately stochastic policies.',
      limitation: 'Very high variance — one lucky episode can reinforce genuinely bad actions.',
      usedFor: ['robotics', 'continuous control', 'the basis of most modern RL']
    },

    {
      id: 'actor-critic',
      name: 'Actor-Critic',
      aka: 'A2C, A3C',
      year: '2000 / 2016',
      learns: 'Reinforcement',
      dataFor: 'Continuous and discrete control',
      tagline: 'One network chooses actions, another judges them, and they train each other.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Actor', rightTitle: 'Critic',
        left: ['policy π(a|s)', 'picks action'], right: ['value V(s)', 'scores state'],
        bridge: 'advantage', leftIn: 'acts in the world', rightOut: 'reduces variance'
      },
      how: [
        'The *actor* is a policy network that chooses actions.',
        'The *critic* is a value network estimating how good the current state is.',
        'Instead of waiting for the episode to end, use the critic to judge each action immediately via the *advantage* — how much better the action was than the critic expected.',
        'The critic learns from observed rewards; the actor learns from the critic\'s advantage estimates.'
      ],
      keyIdea: 'It combines both families to fix the weakness of each. Pure policy gradient has crippling variance because it waits for the final outcome; the critic supplies an immediate, lower-variance judgement.',
      strength: 'Far more sample-efficient than plain policy gradient, and parallelises well.',
      limitation: 'Two networks means two sets of hyperparameters and more ways to destabilise.',
      usedFor: ['robotics', 'game AI', 'the foundation of PPO and SAC']
    },

    {
      id: 'ppo',
      name: 'Proximal Policy Optimization',
      aka: 'PPO',
      year: '2017',
      learns: 'Reinforcement',
      dataFor: 'General control, RLHF',
      tagline: 'Actor-critic with a hard limit on how far the policy may move each update.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'old policy', kind: 'in' },
          { label: 'collect\nexperience', kind: 'hidden' },
          { label: 'clipped\nupdate', kind: 'attn' },
          { label: 'new policy', kind: 'out' }
        ],
        caption: 'the clip prevents any single update from destroying the policy'
      },
      how: [
        'Collect a batch of experience using the current policy.',
        'Compute how much the proposed new policy would change the probability of each action taken.',
        '*Clip* that ratio — typically to within ±20% — so no update can move the policy too far in one step.',
        'Reuse the batch for several epochs, which the clipping makes safe.'
      ],
      keyIdea: 'RL\'s central instability is that a single bad update can collapse a policy irrecoverably, since the policy also determines what data you collect next. PPO\'s clip is a crude but extremely effective guardrail — and its reliability is why it became the default.',
      strength: 'Robust across wildly different problems with little tuning. The workhorse of applied RL.',
      limitation: 'Still on-policy, so it cannot reuse old experience and remains sample-hungry.',
      usedFor: ['robotics', 'game AI', 'RLHF for language models', 'industrial control']
    },

    {
      id: 'sac',
      name: 'Soft Actor-Critic',
      aka: 'SAC',
      year: '2018',
      learns: 'Reinforcement',
      dataFor: 'Continuous control',
      tagline: 'Maximise reward *and* randomness, so the agent keeps exploring.',
      diagram: {
        type: 'cycle',
        left: 'Actor + entropy bonus', right: 'Environment',
        forward: 'stochastic action', back: 'reward + replay buffer'
      },
      how: [
        'Add an *entropy* term to the objective, rewarding the policy for staying unpredictable.',
        'Automatically tune how much that term matters, so exploration decays as the agent improves.',
        'Learn off-policy from a replay buffer, so old experience can be reused many times.',
        'Train two critics and use the lower estimate, which counteracts the tendency to overestimate values.'
      ],
      keyIdea: 'Explicitly rewarding randomness prevents the classic failure where an agent finds a mediocre strategy, stops exploring, and never discovers the better one. Exploration becomes part of the objective instead of a bolted-on heuristic.',
      strength: 'Very sample-efficient for continuous control, and robust to hyperparameters.',
      limitation: 'More complex than PPO, and designed for continuous rather than discrete actions.',
      usedFor: ['robot locomotion', 'manipulation', 'real-world control where samples are expensive']
    },

    {
      id: 'alphazero',
      name: 'AlphaZero & MCTS',
      aka: 'Monte Carlo Tree Search',
      year: '2017',
      learns: 'Reinforcement (self-play)',
      dataFor: 'Perfect-information games',
      tagline: 'A network guides a tree search; the search results retrain the network.',
      diagram: { type: 'tree', depth: 4, caption: 'the network prunes the search; the search improves the network' },
      how: [
        'A single network takes a board position and outputs both a move distribution and an estimate of who is winning.',
        '*Monte Carlo Tree Search* explores promising lines, using the network to decide which branches deserve attention.',
        'The search produces a better move choice than the raw network did — search improves on intuition.',
        'Train the network to imitate the search output, then play more games with the improved network. Repeat from random initialisation, with no human games at all.'
      ],
      keyIdea: 'A bootstrapping loop between intuition and deliberation. The network makes search tractable by pruning; search makes the network better by finding moves it would not have chosen. Neither alone would be enough.',
      strength: 'Superhuman at chess, shogi and Go from self-play alone, in hours.',
      limitation: 'Requires a perfect simulator and full information. Most real problems offer neither.',
      usedFor: ['board games', 'combinatorial optimisation', 'chip layout', 'theorem proving']
    },

    {
      id: 'model-based-rl',
      name: 'Model-Based RL',
      aka: 'Dreamer, World Models, MuZero',
      year: '2018 →',
      learns: 'Reinforcement',
      dataFor: 'Sample-limited environments',
      tagline: 'Learn a model of the world, then practise inside your own imagination.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'real\nexperience', kind: 'in' },
          { label: 'learn world\nmodel', kind: 'hidden' },
          { label: 'imagine\nrollouts', kind: 'attn' },
          { label: 'improve\npolicy', kind: 'out' }
        ],
        caption: 'most training happens inside the learned model'
      },
      how: [
        'Collect some real experience and use it to train a model predicting the next state and reward.',
        'Train the policy largely inside that learned model, generating imagined trajectories cheaply.',
        'Periodically act in the real environment to gather fresh data and correct the model where it is wrong.',
        '*MuZero* goes further: it learns a model only of what matters for planning, never predicting raw observations at all.'
      ],
      keyIdea: 'Model-free RL needs millions of real interactions, which is fine in a simulator and ruinous on a physical robot. A learned model turns expensive real experience into unlimited cheap imagined experience.',
      strength: 'Orders of magnitude more sample-efficient, which matters when data costs real time or hardware.',
      limitation: 'Errors compound over imagined rollouts — the policy can learn to exploit flaws in its own model.',
      usedFor: ['robotics', 'expensive simulations', 'real-world control']
    },

    {
      id: 'rlhf',
      name: 'RLHF & Preference Learning',
      aka: 'Reinforcement Learning from Human Feedback, DPO',
      year: '2017 / 2022',
      learns: 'Reinforcement',
      dataFor: 'Language models',
      tagline: 'Learn a reward model from human comparisons, then optimise against it.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'two\nresponses', kind: 'in' },
          { label: 'human picks\nbetter one', kind: 'hidden' },
          { label: 'reward\nmodel', kind: 'attn' },
          { label: 'tune LLM\nvia PPO', kind: 'out' }
        ],
        caption: 'preferences are easier to collect than absolute scores'
      },
      how: [
        'Generate two responses to the same prompt and ask a human which is better. Comparisons are far easier and more consistent than numerical ratings.',
        'Train a *reward model* to predict those preferences.',
        'Fine-tune the language model with PPO to maximise that predicted reward.',
        'Penalise drifting too far from the original model, or it degenerates into reward-hacking gibberish.'
      ],
      keyIdea: 'Some goals — "be helpful", "do not be rude" — cannot be written as a loss function, but people can reliably judge which of two answers is better. RLHF converts that judgement into a differentiable objective, and it is the main reason chat models feel usable.',
      strength: 'Encodes qualities that resist formal specification.',
      limitation: 'The reward model is a proxy, and optimising hard against a proxy reliably finds its flaws.',
      usedFor: ['chat assistants', 'model alignment', 'summarisation quality', 'safety tuning']
    },

    {
      id: 'multi-agent-rl',
      name: 'Multi-Agent RL',
      aka: 'MARL, self-play',
      year: '1990s →',
      learns: 'Reinforcement',
      dataFor: 'Multi-player environments',
      tagline: 'Several agents learning at once, each one part of the others\' environment.',
      diagram: {
        type: 'graph',
        nodes: [[.22,.28],[.72,.24],[.48,.68],[.86,.62]],
        edges: [[0,1],[0,2],[1,2],[1,3],[2,3]],
        caption: 'each agent is part of every other agent\'s environment'
      },
      how: [
        'Place several learning agents in a shared environment, cooperating, competing, or both.',
        'Each agent observes, acts and receives reward — but the environment now includes other agents that are themselves changing.',
        'Under *self-play*, an agent trains against copies of itself, so the difficulty scales automatically with skill.',
        'Maintain a pool of past opponents to prevent cycles where strategies endlessly counter each other.'
      ],
      keyIdea: 'The environment is *non-stationary* by construction — the optimal policy shifts as opponents learn, so there is no fixed target to converge to. This breaks the assumptions underpinning single-agent RL and is what makes MARL genuinely harder.',
      strength: 'Self-play generates an automatic curriculum, producing superhuman play in Go, poker and StarCraft.',
      limitation: 'Convergence is not guaranteed, and training can cycle indefinitely.',
      usedFor: ['competitive games', 'market simulation', 'traffic control', 'swarm robotics']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
