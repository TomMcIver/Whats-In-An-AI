/* Recurrent and sequence models — architectures with a memory that
   carries information from one step to the next. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'rnn',
    name: 'Recurrent & Sequence',
    order: 4,
    models: [

    {
      id: 'rnn',
      name: 'Recurrent Neural Network',
      aka: 'RNN, Vanilla RNN',
      year: '1986',
      learns: 'Supervised',
      dataFor: 'Sequences of any length',
      tagline: 'One network applied over and over, passing a memory forward each step.',
      diagram: {
        type: 'recurrent', steps: 4, cell: 'RNN',
        inputs: ['The', 'cat', 'sat', 'on'],
        caption: 'the same weights are reused at every time step'
      },
      how: [
        'Process the sequence one element at a time.',
        'At each step, combine the current input with the *hidden state* carried over from the previous step.',
        'Produce a new hidden state, and pass it forward to the next step.',
        'The same weights are used at every step, so the network handles sequences of any length.'
      ],
      keyIdea: 'The hidden state is a running summary of everything seen so far. That is what gives the network memory — but it is a single fixed-size vector, so it must constantly overwrite old information.',
      strength: 'Handles variable-length input naturally, with a constant parameter count.',
      limitation: 'Vanishing gradients. Over long sequences the gradient shrinks toward zero, so it effectively cannot learn dependencies more than a few dozen steps back.',
      usedFor: ['time series', 'sequence labelling', 'historical NLP']
    },

    {
      id: 'lstm',
      name: 'Long Short-Term Memory',
      aka: 'LSTM',
      year: '1997',
      learns: 'Supervised',
      dataFor: 'Long sequences',
      tagline: 'An RNN with gates that decide what to keep, what to forget and what to output.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'forget gate — what to discard', kind: 'in' },
          { label: 'input gate — what to store', kind: 'hidden' },
          { label: 'cell state — the long memory', kind: 'attn' },
          { label: 'output gate — what to expose', kind: 'out' }
        ],
        caption: 'three gates guarding one protected memory channel'
      },
      how: [
        'Maintain a separate *cell state* — a memory highway running straight through the sequence with only minor linear interactions.',
        'The *forget gate* looks at the current input and decides which parts of that memory to erase.',
        'The *input gate* decides which parts of the new information are worth writing in.',
        'The *output gate* decides how much of the memory to expose as this step\'s output. Gates are learned, not hand-designed.'
      ],
      keyIdea: 'The cell state is modified by addition rather than repeated multiplication. That is precisely what stops the gradient from vanishing, and it is why LSTMs can hold information across hundreds of steps.',
      strength: 'Genuinely learns long-range dependencies. Dominated sequence modelling for two decades.',
      limitation: 'Inherently sequential, so it cannot be parallelised across time — the reason transformers displaced it.',
      usedFor: ['speech recognition', 'machine translation (pre-2017)', 'time series', 'handwriting']
    },

    {
      id: 'gru',
      name: 'Gated Recurrent Unit',
      aka: 'GRU',
      year: '2014',
      learns: 'Supervised',
      dataFor: 'Long sequences',
      tagline: 'An LSTM simplified to two gates and no separate cell state.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'reset gate — how much past to ignore', kind: 'in' },
          { label: 'update gate — blend old and new', kind: 'attn' },
          { label: 'hidden state', kind: 'out' }
        ],
        caption: 'one state, two gates, ~25% fewer parameters than an LSTM'
      },
      how: [
        'Merge the LSTM\'s forget and input gates into one *update gate*, which decides how much of the old state to keep versus overwrite.',
        'Add a *reset gate* controlling how much of the previous state feeds into the new candidate value.',
        'Drop the separate cell state entirely — the hidden state does both jobs.',
        'The result has about three-quarters of the LSTM\'s parameters and trains slightly faster.'
      ],
      keyIdea: 'A demonstration that the LSTM was more complicated than it needed to be. On most tasks GRU and LSTM perform within noise of each other, so the simpler one is often the better default.',
      strength: 'Faster to train than an LSTM, and better on smaller datasets.',
      limitation: 'Still sequential, and slightly less expressive on very long dependencies.',
      usedFor: ['time series', 'speech', 'lightweight sequence models']
    },

    {
      id: 'bidirectional-rnn',
      name: 'Bidirectional RNN',
      aka: 'BiLSTM, BiGRU',
      year: '1997',
      learns: 'Supervised',
      dataFor: 'Complete sequences',
      tagline: 'Read the sequence forwards and backwards, then combine both readings.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Forward pass', rightTitle: 'Backward pass',
        left: ['x₁ → x₂ → x₃', 'hidden →'], right: ['x₃ → x₂ → x₁', '← hidden'],
        bridge: 'concat', leftIn: 'reads left to right', rightOut: 'reads right to left'
      },
      how: [
        'Run one RNN over the sequence from start to end.',
        'Run a second, entirely separate RNN from end to start.',
        'At each position, concatenate the two hidden states.',
        'Every position now has access to both its past and its future context.'
      ],
      keyIdea: 'Meaning often depends on what comes *after*. In "he went to the bank to fish", the disambiguating word arrives late — a forward-only model has already committed. This requires the whole sequence up front, so it cannot be used for generation.',
      strength: 'Substantially better on tagging and classification where full context is available.',
      limitation: 'Useless for real-time or generative tasks, since the future is not yet known.',
      usedFor: ['named entity recognition', 'part-of-speech tagging', 'speech transcription']
    },

    {
      id: 'seq2seq',
      name: 'Sequence-to-Sequence',
      aka: 'Encoder-Decoder',
      year: '2014',
      learns: 'Supervised',
      dataFor: 'Sequence in → sequence out',
      tagline: 'Compress the input into one vector, then unroll it into the output.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Encoder', rightTitle: 'Decoder',
        left: ['le', 'chat', 'noir'], right: ['the', 'black', 'cat'],
        bridge: 'context vector', leftIn: 'source sentence', rightOut: 'target sentence'
      },
      how: [
        'An *encoder* RNN reads the entire input sequence and compresses it into a single fixed-size context vector.',
        'A *decoder* RNN is initialised with that vector and generates the output one token at a time.',
        'Each generated token is fed back in as the next step\'s input.',
        'Generation continues until the decoder emits an end-of-sequence marker.'
      ],
      keyIdea: 'It decoupled input length from output length, which made neural translation possible. But the entire source sentence has to squeeze through one vector — the *information bottleneck* that motivated attention.',
      strength: 'Handles mismatched input and output lengths cleanly.',
      limitation: 'The fixed context vector degrades badly on long inputs. Performance falls off sharply past ~30 words.',
      usedFor: ['machine translation', 'summarisation', 'dialogue (historical)']
    },

    {
      id: 'attention',
      name: 'Attention Mechanism',
      aka: 'Bahdanau attention, additive attention',
      year: '2014',
      learns: 'Component',
      dataFor: 'Sequences',
      tagline: 'Instead of one summary vector, let the decoder look back at every input word.',
      diagram: {
        type: 'matrix', n: 6, pattern: 'diagonal',
        top: 'source words →', side: 'output',
        caption: 'each output word draws from a weighted blend of all inputs'
      },
      how: [
        'Keep the encoder\'s hidden state at *every* input position, rather than only the last one.',
        'At each decoding step, score how relevant each input position is to what is being generated now.',
        'Normalise those scores into weights summing to one, and take a weighted average of the encoder states.',
        'Feed that custom-built context into the decoder. Every output step gets its own view of the input.'
      ],
      keyIdea: 'This removed the bottleneck by letting the model *choose what to look at* dynamically. Three years later the field discovered you could throw away the RNN and keep only this — which is the transformer.',
      strength: 'Fixed long-sequence translation, and the weights are interpretable as alignments.',
      limitation: 'Cost grows with the product of input and output length.',
      usedFor: ['translation', 'summarisation', 'the foundation of every transformer']
    },

    {
      id: 'ctc',
      name: 'Connectionist Temporal Classification',
      aka: 'CTC',
      year: '2006',
      learns: 'Supervised',
      dataFor: 'Unaligned sequences',
      tagline: 'Train on sequences where you know the answer but not the timing.',
      diagram: {
        type: 'sequence',
        tokens: ['h', 'h', '–', 'e', 'l', 'l', '–', 'o'],
        arrows: false, predictLast: false,
        below: 'collapses to "hello"',
        caption: 'a blank token lets the model avoid committing to alignment'
      },
      how: [
        'The problem: audio has thousands of frames but the transcript has a few dozen characters, and nobody labelled which frame is which letter.',
        'Let the network emit a character or a special *blank* symbol at every frame.',
        'Collapse the output by removing blanks and merging repeats: "h h – e l l – o" becomes "hello".',
        'Train by summing the probability of *all* frame alignments that collapse to the correct transcript.'
      ],
      keyIdea: 'It removes the need for frame-level labels, which are prohibitively expensive to produce. The model learns the alignment implicitly as a by-product of learning the transcription.',
      strength: 'Made end-to-end speech recognition practical without hand-aligned data.',
      limitation: 'Assumes outputs are conditionally independent given the input, so it needs a separate language model to sound fluent.',
      usedFor: ['speech recognition', 'handwriting recognition', 'OCR']
    },

    {
      id: 'tcn',
      name: 'Temporal Convolutional Network',
      aka: 'TCN, WaveNet-style',
      year: '2016',
      learns: 'Supervised',
      dataFor: 'Time series, audio',
      tagline: 'Convolutions with growing gaps, so a few layers cover a long history.',
      diagram: {
        type: 'graph',
        nodes: [[.05,.9],[.19,.9],[.33,.9],[.47,.9],[.61,.9],[.75,.9],[.89,.9],
                [.26,.5],[.54,.5],[.82,.5],[.54,.1]],
        edges: [[0,7],[2,7],[1,7],[3,8],[4,8],[5,8],[6,9],[5,9],[7,10],[8,10],[9,10]],
        caption: 'dilation doubles each layer, so the receptive field grows exponentially'
      },
      how: [
        'Use 1-D convolutions along the time axis instead of recurrence.',
        'Make them *causal*: each output may only see the present and past, never the future.',
        'Apply *dilation* — skip gaps between the inputs a filter reads — doubling the gap at each successive layer.',
        'Ten layers of doubling dilation reach back over a thousand steps, and every step computes in parallel.'
      ],
      keyIdea: 'It shows recurrence was never strictly necessary for sequences. Dilated convolutions get a long receptive field while remaining fully parallelisable — the same motivation that drove transformers.',
      strength: 'Trains far faster than an RNN, with stable gradients and a fixed memory footprint.',
      limitation: 'The receptive field is fixed at design time, unlike attention which is unbounded.',
      usedFor: ['raw audio generation', 'time series forecasting', 'real-time signal processing']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
