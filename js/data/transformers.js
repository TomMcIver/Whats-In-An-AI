/* Transformers — the architecture that replaced recurrence with
   attention, and now underpins most of modern AI. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'transformers',
    name: 'Transformers',
    order: 5,
    models: [

    {
      id: 'transformer',
      name: 'Transformer',
      year: '2017',
      learns: 'Supervised / self-supervised',
      dataFor: 'Sequences, sets, patches',
      tagline: 'Drop recurrence entirely. Every position looks at every other position at once.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'input + positional encoding', kind: 'in' },
          { label: 'multi-head self-attention', kind: 'attn' },
          { label: 'add & normalise', kind: 'hidden' },
          { label: 'feed-forward network', kind: 'hidden' },
          { label: 'add & normalise', kind: 'out' }
        ],
        repeat: '×N',
        caption: 'the block that everything else in this category is built from'
      },
      how: [
        'Every token produces three vectors: a *query* (what am I looking for), a *key* (what do I offer) and a *value* (what I will contribute).',
        'Compare every query against every key to score how much each token should attend to each other token, then normalise those scores.',
        'Take a weighted sum of the values. Each token is now a blend of the whole sequence, weighted by relevance.',
        'Do this several times in parallel with different learned projections — *multi-head* attention — so different heads can track syntax, coreference and so on.'
      ],
      keyIdea: 'Because every position is computed independently, the whole sequence processes in parallel — unlike an RNN, which must wait for step *t* before starting *t+1*. That parallelism is what made training on internet-scale data feasible, and it is the real reason transformers won.',
      strength: 'Massively parallel, and models arbitrarily long-range relationships in one hop.',
      limitation: 'Attention cost grows with the *square* of sequence length, which is why context windows are expensive.',
      usedFor: ['language', 'vision', 'audio', 'protein folding', 'almost everything']
    },

    {
      id: 'self-attention',
      name: 'Self-Attention',
      aka: 'Scaled dot-product attention, QKV',
      year: '2017',
      learns: 'Component',
      dataFor: 'Any set of tokens',
      tagline: 'Each token asks every other token how relevant it is, and blends accordingly.',
      diagram: {
        type: 'matrix', n: 7, pattern: 'random',
        top: 'keys →', side: 'queries',
        caption: 'one row = how much that token attends to every other'
      },
      how: [
        'Project each token into a query, a key and a value vector using three learned weight matrices.',
        'Score each query against each key with a dot product — a large value means "this is relevant to me".',
        'Divide by the square root of the dimension to stop the scores growing large enough to saturate the softmax, then normalise.',
        'Multiply the resulting weights by the value vectors and sum.'
      ],
      keyIdea: 'It is a *learned, content-based lookup*. A convolution always reads its fixed neighbours; attention reads whatever is relevant, wherever it sits. In "the animal didn\'t cross the street because it was too tired", attention is what links "it" to "animal".',
      strength: 'Any two positions are one hop apart, regardless of distance.',
      limitation: 'Quadratic in sequence length. Doubling the context quadruples the cost.',
      usedFor: ['every transformer variant']
    },

    {
      id: 'positional-encoding',
      name: 'Positional Encoding',
      aka: 'Sinusoidal, learned, RoPE, ALiBi',
      year: '2017',
      learns: 'Component',
      dataFor: 'Transformer inputs',
      tagline: 'Attention has no sense of order, so position has to be added explicitly.',
      diagram: {
        type: 'sequence',
        tokens: ['tok₁', 'tok₂', 'tok₃', 'tok₄', 'tok₅'],
        arrows: false, predictLast: false,
        below: '+ position 1, 2, 3, 4, 5',
        caption: 'without this, "dog bites man" equals "man bites dog"'
      },
      how: [
        'Self-attention treats its input as an unordered *set* — permute the tokens and the output permutes identically. Word order would be invisible.',
        'The original fix adds sine and cosine waves of different frequencies to each embedding, giving every position a unique signature.',
        'Learned encodings simply train a vector per position, which works but cannot extrapolate past the trained length.',
        '*RoPE* rotates the query and key vectors by an angle proportional to position, so attention scores depend naturally on *relative* distance. Most modern LLMs use it.'
      ],
      keyIdea: 'A small component with outsized consequences. How position is encoded largely determines whether a model can generalise to sequences longer than it was trained on — which is why context-window extension work focuses here.',
      strength: 'RoPE extrapolates far better than the alternatives and has become the default.',
      limitation: 'All schemes degrade well beyond the trained context length.',
      usedFor: ['every transformer']
    },

    {
      id: 'bert',
      name: 'BERT',
      aka: 'Bidirectional Encoder Representations from Transformers',
      year: '2018',
      learns: 'Self-supervised → fine-tuned',
      dataFor: 'Text understanding',
      tagline: 'Hide random words and train the model to fill them in, using both sides.',
      diagram: {
        type: 'sequence',
        tokens: ['the', 'cat', '[MASK]', 'on', 'the', 'mat'],
        arrows: false, predictLast: false,
        below: 'predict the masked word from both directions',
        caption: 'encoder-only — sees the whole sentence at once'
      },
      how: [
        'Take an enormous text corpus and randomly mask about 15% of the tokens.',
        'Train the model to predict each masked token using *both* the left and right context.',
        'Also train it to predict whether two sentences were adjacent in the original text.',
        'Then fine-tune the pre-trained model on a small labelled dataset for a specific task.'
      ],
      keyIdea: 'It established the *pre-train then fine-tune* recipe. One expensive general model, then cheap task-specific adaptation — a shift that made strong NLP available to anyone without a supercomputer.',
      strength: 'Bidirectional context makes it excellent at understanding tasks.',
      limitation: 'Cannot generate text. Masked prediction is not next-token prediction.',
      usedFor: ['search ranking', 'classification', 'named entity recognition', 'sentence embeddings']
    },

    {
      id: 'gpt',
      name: 'GPT',
      aka: 'Generative Pre-trained Transformer, decoder-only LLM',
      year: '2018',
      learns: 'Self-supervised',
      dataFor: 'Text generation',
      tagline: 'Predict the next token. Scale it up enough and everything else emerges.',
      diagram: {
        type: 'matrix', n: 7, pattern: 'causal',
        top: 'can attend to →', side: 'position',
        caption: 'the causal mask: each token sees only what came before it'
      },
      how: [
        'Train on one objective only: given all previous tokens, predict the next one.',
        'Apply a *causal mask* so each position can attend to earlier positions but never later ones — otherwise the answer would be visible.',
        'Generate by sampling a token, appending it, and feeding the whole sequence back in.',
        'Scale parameters, data and compute together. Capabilities not present in smaller models appear as size increases.'
      ],
      keyIdea: 'Next-token prediction sounds trivially narrow, but doing it *well* over the whole internet requires grammar, facts, reasoning and style. The task is a proxy; the competence is a side effect of pursuing it at scale.',
      strength: 'One model, many tasks, specified in plain language rather than retrained.',
      limitation: 'No notion of truth — it models what text is *likely*, which is why it can be fluently wrong.',
      usedFor: ['chat', 'code generation', 'writing', 'reasoning', 'agents']
    },

    {
      id: 't5',
      name: 'T5',
      aka: 'Text-to-Text Transfer Transformer',
      year: '2019',
      learns: 'Self-supervised → fine-tuned',
      dataFor: 'Any text task',
      tagline: 'Cast every NLP problem as text in, text out.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Encoder', rightTitle: 'Decoder',
        left: ['translate En→De:', 'the cat sat'], right: ['die Katze', 'saß'],
        bridge: 'cross-attention', leftIn: 'task prefix + input', rightOut: 'generated text'
      },
      how: [
        'Reframe everything as text-to-text. Translation, summarisation, classification and regression all become "read this string, write that string".',
        'Prefix the input with the task: "summarize:" or "translate English to German:".',
        'Use a full encoder-decoder transformer — the encoder reads bidirectionally, the decoder generates while cross-attending to it.',
        'Pre-train by corrupting spans of text and having the model reconstruct them.'
      ],
      keyIdea: 'Unifying every task under one format means one model, one loss and one decoding procedure. Even "predict a number between 1 and 5" becomes generating the character "4".',
      strength: 'Extremely flexible, and multi-task training transfers between tasks.',
      limitation: 'Encoder-decoder is heavier than decoder-only, which is why most modern LLMs dropped the encoder.',
      usedFor: ['translation', 'summarisation', 'question answering', 'multi-task NLP']
    },

    {
      id: 'vit',
      name: 'Vision Transformer',
      aka: 'ViT',
      year: '2020',
      learns: 'Supervised / self-supervised',
      dataFor: 'Images',
      tagline: 'Chop the image into patches, treat each as a word, run a transformer.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'image', kind: 'in' },
          { label: '16×16\npatches', kind: 'hidden' },
          { label: 'linear\nembed', kind: 'hidden' },
          { label: 'transformer', kind: 'attn' },
          { label: 'class', kind: 'out' }
        ],
        caption: 'a patch is a token; the rest is a standard transformer'
      },
      how: [
        'Split the image into a grid of fixed patches — typically 16×16 pixels.',
        'Flatten each patch and project it linearly into an embedding, exactly as a word would be.',
        'Add positional encodings so the model knows where each patch sat, then feed the sequence into a standard transformer.',
        'A special classification token collects global information and drives the final prediction.'
      ],
      keyIdea: 'It removes the convolutional prior entirely. CNNs *assume* locality and translation invariance; ViT assumes nothing and learns them from data — which is why it needs far more data, but eventually surpasses CNNs.',
      strength: 'Scales better than CNNs with very large datasets, and models global context from layer one.',
      limitation: 'Data-hungry. Below roughly 10 million images, a CNN with its built-in priors wins.',
      usedFor: ['image classification', 'multimodal models', 'segmentation backbones']
    },

    {
      id: 'swin',
      name: 'Swin Transformer',
      year: '2021',
      learns: 'Supervised',
      dataFor: 'Images, dense prediction',
      tagline: 'Attention inside local windows that shift each layer, restoring hierarchy.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 50, depth: 2, label: 'windows', sub: '56²', op: 'shift' },
          { size: 38, depth: 3, label: 'merge', sub: '28²', op: 'shift' },
          { size: 26, depth: 4, label: 'merge', sub: '14²', op: 'shift' },
          { size: 16, depth: 1, label: 'output', kind: 'out' }
        ]
      },
      how: [
        'Compute attention only within small local windows rather than globally, making cost linear in image size instead of quadratic.',
        '*Shift* the window boundaries by half a window on alternate layers, so information crosses between neighbouring windows.',
        'Merge adjacent patches periodically to build a hierarchy, exactly as CNN pooling does.',
        'The result is a multi-scale feature pyramid, which dense prediction tasks need.'
      ],
      keyIdea: 'A reconciliation. ViT threw away the CNN\'s hierarchy and paid for it on detection and segmentation; Swin puts the hierarchy back while keeping attention, and became a general-purpose vision backbone.',
      strength: 'Linear cost in image size, and produces the multi-scale features detectors require.',
      limitation: 'More complex than plain ViT, and the shifted-window scheme is fiddly to implement.',
      usedFor: ['object detection', 'semantic segmentation', 'high-resolution vision']
    },

    {
      id: 'clip',
      name: 'CLIP',
      aka: 'Contrastive Language-Image Pre-training',
      year: '2021',
      learns: 'Contrastive self-supervised',
      dataFor: 'Image + text pairs',
      tagline: 'Train images and captions into the same space so they can be compared directly.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Image encoder', rightTitle: 'Text encoder',
        left: ['ViT', 'image vector'], right: ['transformer', 'text vector'],
        bridge: 'cosine similarity', leftIn: 'photo', rightOut: '"a dog on a beach"'
      },
      how: [
        'Collect 400 million image-caption pairs scraped from the web — no manual labelling.',
        'Encode images with a vision model and captions with a text model, into vectors of the same size.',
        'Train contrastively: for a batch of N pairs, push each image toward its own caption and away from the other N−1.',
        'At inference, classify by writing candidate labels as sentences and picking the nearest one in the shared space.'
      ],
      keyIdea: 'It enables *zero-shot* classification. Because images and text share a space, you can classify into categories the model was never trained on simply by describing them — the label set becomes a runtime argument rather than a training decision.',
      strength: 'Zero-shot transfer, and robust to distribution shift that breaks supervised classifiers.',
      limitation: 'Inherits the biases of uncurated web data, and is weak at counting and fine spatial relations.',
      usedFor: ['zero-shot classification', 'image search', 'guiding diffusion models', 'multimodal LLMs']
    },

    {
      id: 'whisper',
      name: 'Whisper',
      year: '2022',
      learns: 'Weakly supervised',
      dataFor: 'Audio → text',
      tagline: 'Speech recognition as plain sequence-to-sequence, trained on 680k hours.',
      diagram: {
        type: 'encdec',
        leftTitle: 'Audio encoder', rightTitle: 'Text decoder',
        left: ['mel spectrogram', 'transformer'], right: ['transformer', 'tokens'],
        bridge: 'cross-attention', leftIn: '30s of audio', rightOut: 'transcript'
      },
      how: [
        'Convert audio into a mel spectrogram — a picture of frequency over time — in 30-second chunks.',
        'Encode it with a transformer, and decode text with a second transformer that cross-attends to the audio.',
        'Train on 680,000 hours of diverse, noisy, weakly-labelled web audio in 96 languages.',
        'Use special tokens to switch task: transcribe, translate, detect language, or predict timestamps.'
      ],
      keyIdea: 'It abandons the specialised speech pipeline — acoustic model, pronunciation dictionary, language model — for one plain transformer and a great deal of messy data. Scale and diversity substituted for architecture.',
      strength: 'Remarkably robust to accents, background noise and technical vocabulary.',
      limitation: 'Will occasionally hallucinate fluent text during silence, a direct consequence of its generative objective.',
      usedFor: ['transcription', 'subtitling', 'voice interfaces', 'speech translation']
    },

    {
      id: 'moe',
      name: 'Mixture of Experts',
      aka: 'MoE, Sparse MoE',
      year: '2017 / 2021',
      learns: 'Self-supervised',
      dataFor: 'Very large models',
      tagline: 'Many expert sub-networks; a router sends each token to just a couple.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'token', kind: 'in' },
          { label: 'router', kind: 'attn' },
          { label: 'expert 3', kind: 'hidden' },
          { label: 'expert 7', kind: 'hidden' },
          { label: 'combine', kind: 'out' }
        ],
        caption: '64 experts exist; 2 run per token'
      },
      how: [
        'Replace the feed-forward block in each transformer layer with many parallel copies, called experts.',
        'Add a small *router* network that inspects each token and picks the top two experts for it.',
        'Only those experts run. The other sixty-two sit idle for that token.',
        'Add a load-balancing penalty during training, or the router collapses onto a few favourites and the rest never learn.'
      ],
      keyIdea: 'It decouples parameter count from compute cost. A model can hold a trillion parameters while activating only a few billion per token — more knowledge stored, same cost to run. Most frontier models now use this.',
      strength: 'Far more capacity per unit of inference compute.',
      limitation: 'All experts must be held in memory, and training is unstable without careful balancing.',
      usedFor: ['frontier language models', 'large-scale multilingual models']
    },

    {
      id: 'mamba-ssm',
      name: 'Mamba & State Space Models',
      aka: 'SSM, S4, selective state space',
      year: '2023',
      learns: 'Self-supervised',
      dataFor: 'Very long sequences',
      tagline: 'A learned recurrent state that scales linearly instead of quadratically.',
      diagram: {
        type: 'recurrent', steps: 5, cell: 'SSM',
        inputs: ['x₁', 'x₂', 'x₃', 'x₄', 'x₅'],
        caption: 'linear in sequence length, but parallelisable during training'
      },
      how: [
        'Model the sequence with a continuous state-space equation — a compressed state updated at each step, like an RNN.',
        'Make the update parameters *depend on the input*, so the model can choose to remember or ignore based on content. This selectivity is Mamba\'s contribution.',
        'During training, the recurrence can be rewritten as a convolution and computed in parallel, avoiding the RNN\'s sequential bottleneck.',
        'At inference it runs as a true recurrence with constant memory, regardless of how long the sequence gets.'
      ],
      keyIdea: 'The most credible challenger to attention. Attention is quadratic and stores everything; SSMs are linear and store a compressed summary. The open question is whether that compression loses too much for tasks needing exact recall.',
      strength: 'Linear scaling makes million-token contexts tractable; inference memory is constant.',
      limitation: 'Weaker at precise retrieval from long context, where attention still leads. Hybrids are common.',
      usedFor: ['long documents', 'genomics', 'audio', 'efficient long-context models']
    },

    {
      id: 'efficient-attention',
      name: 'Efficient Attention',
      aka: 'FlashAttention, sparse attention, linear attention',
      year: '2019 →',
      learns: 'Component',
      dataFor: 'Long-context transformers',
      tagline: 'Techniques to survive attention\'s quadratic cost.',
      diagram: {
        type: 'matrix', n: 7, pattern: 'causal',
        top: 'only some pairs computed →', side: 'query',
        caption: 'sparse patterns skip most of the matrix'
      },
      how: [
        '*Sparse attention* restricts each token to a subset — local neighbours plus a few global tokens — instead of everything.',
        '*Linear attention* reorders the matrix multiplications so cost grows linearly, at some cost in expressiveness.',
        '*FlashAttention* changes nothing mathematically but reorganises the computation to avoid writing the huge attention matrix to slow GPU memory.',
        '*Grouped-query attention* shares key and value projections across heads, shrinking the cache that dominates inference memory.'
      ],
      keyIdea: 'FlashAttention is the instructive case: an exact, unchanged result made several times faster purely by respecting the memory hierarchy. Sometimes the bottleneck is not arithmetic but data movement.',
      strength: 'FlashAttention is a free win — same output, less memory, more speed.',
      limitation: 'Approximate methods trade quality for speed, and often underperform on tasks needing exact recall.',
      usedFor: ['long-context LLMs', 'efficient training', 'inference optimisation']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
