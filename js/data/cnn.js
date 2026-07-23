/* Convolutional networks — architectures built around the idea that
   nearby pixels are related and a useful feature is useful anywhere. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'cnn',
    name: 'Convolutional Nets',
    order: 3,
    models: [

    {
      id: 'cnn',
      name: 'Convolutional Neural Network',
      aka: 'CNN, ConvNet',
      year: '1989',
      learns: 'Supervised',
      dataFor: 'Images, grids, spectrograms',
      tagline: 'Slide small filters across the image; stack them to build up from edges to objects.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 52, depth: 1, label: 'image', sub: '224²', op: 'conv' },
          { size: 42, depth: 4, label: 'edges', sub: '112²', op: 'pool' },
          { size: 30, depth: 5, label: 'shapes', sub: '56²', op: 'conv' },
          { size: 20, depth: 5, label: 'parts', sub: '28²', op: 'fc' },
          { size: 12, depth: 1, label: 'class', kind: 'out' }
        ]
      },
      how: [
        'Slide a small filter — say 3×3 — across the image, computing a dot product at every position. The result is a *feature map* showing where that pattern occurs.',
        'Use many filters per layer, each learning to detect something different: a horizontal edge, a colour blob, a texture.',
        'Downsample with *pooling*, so later layers see a larger area of the original image through the same small window.',
        'Stack these blocks. Early layers detect edges, middle layers combine edges into shapes, deep layers respond to whole objects.'
      ],
      keyIdea: 'Two structural assumptions do all the work. *Locality*: pixels near each other are related. *Weight sharing*: a cat detector is useful in the top-left and the bottom-right, so use the same filter everywhere. This cuts parameters by orders of magnitude versus a dense layer.',
      strength: 'Translation invariance for free, and dramatically fewer parameters than a fully connected net.',
      limitation: 'Fixed-size receptive fields make long-range relationships awkward — one reason transformers took over.',
      usedFor: ['image classification', 'medical imaging', 'audio spectrograms', 'video']
    },

    {
      id: 'lenet',
      name: 'LeNet-5',
      year: '1998',
      learns: 'Supervised',
      dataFor: 'Small greyscale images',
      tagline: 'The first CNN that worked commercially — it read cheques.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 46, depth: 1, label: '32×32', op: 'conv' },
          { size: 38, depth: 3, label: 'C1', sub: '6 maps', op: 'pool' },
          { size: 26, depth: 4, label: 'C3', sub: '16 maps', op: 'fc' },
          { size: 16, depth: 1, label: '10 digits', kind: 'out' }
        ]
      },
      how: [
        'Two convolution-and-pooling blocks extract features from a 32×32 greyscale digit.',
        'Two fully connected layers then classify the extracted features.',
        'Total size: about 60,000 parameters — trainable on 1998 hardware.',
        'Deployed by banks in the United States to read handwritten cheque amounts at scale.'
      ],
      keyIdea: 'Everything the modern CNN does was already here — convolution, pooling, hierarchy. What was missing was data and compute, and it took another fourteen years for both to arrive.',
      strength: 'Proved the architecture worked on a real commercial problem.',
      limitation: 'Far too small for natural images with colour, clutter and scale variation.',
      usedFor: ['digit recognition', 'historical reference', 'teaching']
    },

    {
      id: 'alexnet',
      name: 'AlexNet',
      year: '2012',
      learns: 'Supervised',
      dataFor: 'Natural images',
      tagline: 'The network that started the deep learning boom.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 50, depth: 1, label: '227²', op: '11×11' },
          { size: 40, depth: 5, label: '96 maps', op: '5×5' },
          { size: 28, depth: 5, label: '256 maps', op: '3×3' },
          { size: 18, depth: 5, label: '384 maps', op: 'fc' },
          { size: 12, depth: 1, label: '1000', kind: 'out' }
        ]
      },
      how: [
        'Eight learned layers — five convolutional, three fully connected — with 60 million parameters.',
        'Used *ReLU* instead of tanh, which trained several times faster and made the depth practical.',
        'Applied *dropout* in the dense layers to control overfitting on 1.2 million images.',
        'Split across two GPUs because a single card of the era had insufficient memory.'
      ],
      keyIdea: 'It cut the ImageNet error rate from 26% to 15% in one step — an unprecedented margin. That result is what convinced the field that deep learning was not a dead end, and effectively started the modern era.',
      strength: 'Demonstrated that scale plus GPUs plus ReLU changed what was possible.',
      limitation: 'Crude by modern standards: huge filters, enormous dense layers, inefficient.',
      usedFor: ['historical reference', 'transfer learning (legacy)']
    },

    {
      id: 'vgg',
      name: 'VGGNet',
      year: '2014',
      learns: 'Supervised',
      dataFor: 'Natural images',
      tagline: 'Just 3×3 filters, stacked very deep, and nothing clever.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'conv 3×3 ×2', kind: 'in' },
          { label: 'conv 3×3 ×2', kind: 'hidden' },
          { label: 'conv 3×3 ×3', kind: 'hidden' },
          { label: 'conv 3×3 ×3', kind: 'hidden' },
          { label: 'FC → 1000', kind: 'out' }
        ],
        caption: '16 or 19 layers of identical blocks'
      },
      how: [
        'Use only 3×3 convolutions throughout — the smallest filter that still has a notion of direction.',
        'Two stacked 3×3 layers see the same area as one 5×5, but with fewer parameters and an extra non-linearity.',
        'Double the channel count every time the spatial resolution halves.',
        'Repeat to 16 or 19 layers. That is the entire design.'
      ],
      keyIdea: 'Its contribution was showing that *uniform depth beats architectural cleverness*. The design is so regular it became the default feature extractor for years.',
      strength: 'Simple, regular, and its features transfer extremely well to other tasks.',
      limitation: 'Enormously wasteful — 138 million parameters, most of them in one dense layer.',
      usedFor: ['transfer learning', 'perceptual loss', 'style transfer']
    },

    {
      id: 'resnet',
      name: 'ResNet',
      aka: 'Residual Network',
      year: '2015',
      learns: 'Supervised',
      dataFor: 'Natural images',
      tagline: 'Add shortcuts around each block, and suddenly 150 layers trains fine.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'input', kind: 'in' },
          { label: 'conv → BN → ReLU', kind: 'hidden' },
          { label: 'conv → BN', kind: 'hidden' },
          { label: '+ skip connection', kind: 'attn' },
          { label: 'output', kind: 'out' }
        ],
        repeat: '×50',
        caption: 'the skip lets gradient reach early layers unchanged'
      },
      how: [
        'Observe the problem: beyond ~20 layers, adding depth made networks *worse*, even on training data. This was not overfitting but an optimisation failure.',
        'Add a *skip connection* that carries the input around each pair of conv layers and adds it to the output.',
        'The block now only has to learn the *residual* — the difference from the input — rather than the whole transformation.',
        'If a layer is not useful, it can learn to output zero, and the skip passes the input through untouched.'
      ],
      keyIdea: 'The skip connection gives gradients an unobstructed path back to early layers. It is arguably the single most important architectural idea of the last decade — transformers use it too, on every single block.',
      strength: 'Made networks of 50, 101, even 1000 layers trainable. Still a top-tier vision backbone.',
      limitation: 'Depth costs inference latency; convolutions still limit the receptive field.',
      usedFor: ['image classification', 'detection backbones', 'transfer learning', 'feature extraction']
    },

    {
      id: 'inception',
      name: 'Inception / GoogLeNet',
      year: '2014',
      learns: 'Supervised',
      dataFor: 'Natural images',
      tagline: 'Run several filter sizes in parallel and let the network choose.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'input', kind: 'in' },
          { label: '1×1', kind: 'hidden' },
          { label: '3×3', kind: 'hidden' },
          { label: '5×5', kind: 'hidden' },
          { label: 'concat', kind: 'out' }
        ],
        caption: 'all branches run on the same input, outputs are stacked'
      },
      how: [
        'Rather than choosing a filter size, apply 1×1, 3×3 and 5×5 convolutions plus max pooling to the same input, in parallel.',
        'Concatenate all the outputs along the channel dimension.',
        'Insert 1×1 convolutions first as *bottlenecks* to cut channel count, which keeps the cost affordable.',
        'The network learns how much to weight each branch, effectively selecting the scale it needs.'
      ],
      keyIdea: 'Objects appear at wildly different scales, so a fixed filter size is always a compromise. Inception refuses to compromise and processes multiple scales simultaneously — at 12× fewer parameters than AlexNet.',
      strength: 'Very high accuracy for its parameter count.',
      limitation: 'Hand-designed and irregular, which makes it awkward to modify or scale.',
      usedFor: ['image classification', 'efficient inference']
    },

    {
      id: 'densenet',
      name: 'DenseNet',
      year: '2016',
      learns: 'Supervised',
      dataFor: 'Natural images',
      tagline: 'Every layer receives the output of every earlier layer.',
      diagram: {
        type: 'graph',
        nodes: [[.10,.5],[.32,.5],[.54,.5],[.76,.5],[.95,.5]],
        edges: [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]],
        caption: 'each layer concatenates all previous feature maps'
      },
      how: [
        'Within a block, layer 4 receives the concatenated outputs of layers 1, 2 and 3 — not just layer 3.',
        'Features are *concatenated*, not summed as in ResNet, so nothing is overwritten.',
        'Because every layer can reuse all earlier features, each one only needs to add a handful of new channels.',
        'Transition layers between blocks downsample and trim the channel count back down.'
      ],
      keyIdea: 'Extreme feature reuse. If layer 2 already computed a useful edge map, layer 10 can just read it rather than recomputing it — so DenseNet matches ResNet accuracy with roughly a third of the parameters.',
      strength: 'Very parameter-efficient, with strong gradient flow.',
      limitation: 'Memory-hungry during training, since all intermediate maps must be retained.',
      usedFor: ['medical imaging', 'small-data vision', 'segmentation backbones']
    },

    {
      id: 'efficientnet',
      name: 'EfficientNet',
      year: '2019',
      learns: 'Supervised',
      dataFor: 'Natural images',
      tagline: 'Scale depth, width and resolution together in a fixed ratio.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 30, depth: 2, label: 'B0', sub: '5M', op: '×φ' },
          { size: 38, depth: 3, label: 'B3', sub: '12M', op: '×φ' },
          { size: 46, depth: 4, label: 'B5', sub: '30M', op: '×φ' },
          { size: 54, depth: 5, label: 'B7', sub: '66M', kind: 'out' }
        ]
      },
      how: [
        'Note that people scale networks arbitrarily — sometimes deeper, sometimes wider, sometimes higher resolution.',
        'Search for the best small base network, then find the optimal *ratio* in which to scale all three dimensions.',
        'Scale by a single compound coefficient φ, so depth, width and input resolution grow together in that fixed proportion.',
        'This produces a family, B0 to B7, trading accuracy against cost along one dial.'
      ],
      keyIdea: 'Balanced scaling beats scaling any one dimension. A deeper network needs proportionally more width to carry the information, and higher resolution to have detail worth carrying — EfficientNet-B7 hit state of the art with 8× fewer parameters than the previous best.',
      strength: 'Excellent accuracy-per-FLOP. Strong choice when inference cost matters.',
      limitation: 'The architecture came from an expensive neural architecture search, hard to replicate.',
      usedFor: ['mobile vision', 'edge deployment', 'cost-sensitive inference']
    },

    {
      id: 'mobilenet',
      name: 'MobileNet',
      year: '2017',
      learns: 'Supervised',
      dataFor: 'Natural images, on-device',
      tagline: 'Split each convolution into two cheap ones so it runs on a phone.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'input', kind: 'in' },
          { label: 'depthwise\n3×3', kind: 'hidden' },
          { label: 'pointwise\n1×1', kind: 'hidden' },
          { label: 'output', kind: 'out' }
        ],
        caption: 'roughly 9× cheaper than a standard convolution'
      },
      how: [
        'A standard convolution filters *and* combines channels in one expensive operation.',
        'Split it: a *depthwise* convolution applies one filter per input channel, doing no cross-channel mixing.',
        'Then a *pointwise* 1×1 convolution combines the channels, doing no spatial work.',
        'The two steps together do the same job for roughly one-ninth of the computation.'
      ],
      keyIdea: 'Factorising an operation into cheaper pieces that compose to the same effect. The same trick — separating spatial mixing from channel mixing — shows up repeatedly in efficient architecture design.',
      strength: 'Runs in real time on phones and embedded hardware.',
      limitation: 'Gives up some accuracy relative to full convolutions.',
      usedFor: ['mobile apps', 'embedded vision', 'real-time inference']
    },

    {
      id: 'unet',
      name: 'U-Net',
      year: '2015',
      learns: 'Supervised',
      dataFor: 'Images → pixel masks',
      tagline: 'Compress the image down, build it back up, and wire the two halves together.',
      diagram: {
        type: 'hourglass',
        labels: ['image', 'encode', 'bottleneck', 'decode', 'mask'],
        heights: [122, 78, 32, 78, 122],
        caption: 'skip connections carry fine detail across the U'
      },
      how: [
        'The *contracting* path is a normal CNN: convolutions and pooling reduce resolution while increasing semantic depth.',
        'The *expanding* path upsamples back to full resolution, so the output is a per-pixel map rather than a single label.',
        'Critically, *skip connections* copy feature maps from each contracting level directly across to the matching expanding level.',
        'Those skips restore the fine spatial detail that pooling destroyed — without them boundaries come out blurred.'
      ],
      keyIdea: 'Segmentation needs two things in tension: broad context to know *what* something is, and fine detail to know exactly *where* it ends. The U-shape gets context from the bottleneck and detail from the skips.',
      strength: 'Trains on remarkably few images — it was designed for biomedical data where labels are scarce.',
      limitation: 'Fixed input size, and heavy on memory at high resolution.',
      usedFor: ['medical segmentation', 'satellite imagery', 'the denoiser inside diffusion models']
    },

    {
      id: 'rcnn-family',
      name: 'R-CNN Family',
      aka: 'Fast R-CNN, Faster R-CNN, Mask R-CNN',
      year: '2014–2017',
      learns: 'Supervised',
      dataFor: 'Images → boxes',
      tagline: 'Propose regions that might contain objects, then classify each one.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'image', kind: 'in' },
          { label: 'backbone\nCNN', kind: 'hidden' },
          { label: 'region\nproposals', kind: 'attn' },
          { label: 'classify\n+ refine', kind: 'out' }
        ],
        caption: 'two stages: where might objects be, then what are they'
      },
      how: [
        'Original R-CNN used an external algorithm to propose ~2000 candidate regions, then ran a CNN on each — accurate but painfully slow.',
        '*Fast R-CNN* ran the CNN once over the whole image and cropped features per region instead, a large speedup.',
        '*Faster R-CNN* replaced the external proposer with a learned *Region Proposal Network*, making the whole pipeline end-to-end.',
        '*Mask R-CNN* added a parallel branch predicting a pixel mask per box, giving instance segmentation.'
      ],
      keyIdea: 'Two-stage detection: separate "where might something be" from "what is it". More accurate than single-stage detectors, at the cost of speed.',
      strength: 'High accuracy, especially on small or overlapping objects.',
      limitation: 'Too slow for real-time video without significant optimisation.',
      usedFor: ['object detection', 'instance segmentation', 'medical imaging']
    },

    {
      id: 'yolo',
      name: 'YOLO',
      aka: 'You Only Look Once, SSD',
      year: '2015',
      learns: 'Supervised',
      dataFor: 'Images → boxes, real-time',
      tagline: 'Predict every box in one pass over the image.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'image', kind: 'in' },
          { label: 'single CNN\npass', kind: 'hidden' },
          { label: 'grid of\nbox predictions', kind: 'out' }
        ],
        caption: 'no proposal stage — detection is one forward pass'
      },
      how: [
        'Divide the image into a grid — say 13×13 cells.',
        'Each cell simultaneously predicts a fixed number of bounding boxes, a confidence score, and class probabilities.',
        'The whole thing is one forward pass through one network, with no separate proposal stage.',
        'Overlapping duplicate boxes are removed afterwards by *non-maximum suppression*.'
      ],
      keyIdea: 'Reframing detection as a single regression problem instead of a pipeline. That change is what took detection from a few frames per second to real-time video.',
      strength: 'Extremely fast — real-time on modest hardware. Sees global context, so fewer background false positives.',
      limitation: 'Historically weaker on small and clustered objects, though later versions closed much of the gap.',
      usedFor: ['real-time video', 'autonomous vehicles', 'surveillance', 'robotics']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
