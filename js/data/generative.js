/* Generative models — architectures that learn a data distribution
   well enough to produce new samples from it. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'generative',
    name: 'Generative Models',
    order: 6,
    models: [

    {
      id: 'autoencoder',
      name: 'Autoencoder',
      year: '1987',
      learns: 'Self-supervised',
      dataFor: 'Any data',
      tagline: 'Squeeze the input through a narrow layer and rebuild it on the far side.',
      diagram: {
        type: 'hourglass',
        labels: ['input', 'encode', 'code', 'decode', 'output'],
        caption: 'the bottleneck forces it to keep only what matters'
      },
      how: [
        'An *encoder* compresses the input down to a small vector — the bottleneck or latent code.',
        'A *decoder* attempts to reconstruct the original input from that code alone.',
        'Train by minimising reconstruction error. The label is the input itself, so no annotation is needed.',
        'Because the bottleneck is far narrower than the input, the network cannot simply copy; it must learn what is worth keeping.'
      ],
      keyIdea: 'Compression forces understanding. To rebuild a face from 32 numbers you must have learned what faces have in common — the constraint is the entire teaching mechanism.',
      strength: 'Learns useful representations from completely unlabelled data.',
      limitation: 'The latent space has gaps. Sample a random code and you usually get noise, which is what the VAE fixes.',
      usedFor: ['dimensionality reduction', 'denoising', 'anomaly detection', 'pretraining']
    },

    {
      id: 'vae',
      name: 'Variational Autoencoder',
      aka: 'VAE',
      year: '2013',
      learns: 'Self-supervised (generative)',
      dataFor: 'Images, molecules, audio',
      tagline: 'An autoencoder whose latent space is smooth enough to sample from.',
      diagram: {
        type: 'hourglass',
        labels: ['input', 'encode', 'μ, σ → sample', 'decode', 'output'],
        heights: [116, 76, 40, 76, 116],
        caption: 'the code is a distribution, not a point'
      },
      how: [
        'The encoder outputs not a single code but a *distribution* — a mean and a standard deviation per dimension.',
        'Sample a point from that distribution and pass it to the decoder, so nearby codes must decode to similar outputs.',
        'Add a penalty pulling every encoded distribution toward a standard normal, which stops the space fragmenting into isolated islands.',
        'Use the *reparameterisation trick* to keep the sampling step differentiable, so gradients can flow through it.'
      ],
      keyIdea: 'The regularisation makes the latent space *continuous and complete*. Every point decodes to something plausible, so you can generate by drawing random noise, and interpolate smoothly between two inputs.',
      strength: 'Principled, stable to train, and gives a genuinely navigable latent space.',
      limitation: 'Outputs are noticeably blurry — averaging over the distribution smooths away fine detail.',
      usedFor: ['generation', 'molecule design', 'interpolation', 'the compression stage of Stable Diffusion']
    },

    {
      id: 'gan',
      name: 'Generative Adversarial Network',
      aka: 'GAN',
      year: '2014',
      learns: 'Adversarial',
      dataFor: 'Images, audio, video',
      tagline: 'A forger and a detective, each getting better by trying to beat the other.',
      diagram: { type: 'adversarial' },
      how: [
        'A *generator* turns random noise into a fake sample. Initially it produces pure garbage.',
        'A *discriminator* is shown a mixture of real and generated samples and must judge which is which.',
        'The discriminator trains to catch fakes; the generator trains specifically to fool the discriminator.',
        'Each improvement forces the other to improve. At equilibrium the fakes are indistinguishable from real data.'
      ],
      keyIdea: 'There is no hand-written loss function for "looks like a real photograph" — so GANs *learn* one. The discriminator is a trainable loss function, which is why GAN outputs are sharp where VAE outputs are blurry.',
      strength: 'Produces exceptionally sharp, realistic samples.',
      limitation: 'Notoriously unstable. *Mode collapse* — where the generator finds one convincing output and produces only that — is a constant hazard.',
      usedFor: ['photorealistic faces', 'super-resolution', 'data augmentation', 'image translation']
    },

    {
      id: 'dcgan',
      name: 'DCGAN',
      aka: 'Deep Convolutional GAN',
      year: '2015',
      learns: 'Adversarial',
      dataFor: 'Images',
      tagline: 'The set of architectural rules that made GANs actually train.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 14, depth: 1, label: 'noise', sub: '100-d', op: 'deconv' },
          { size: 26, depth: 4, label: '8²', op: 'deconv' },
          { size: 38, depth: 3, label: '16²', op: 'deconv' },
          { size: 50, depth: 1, label: '64² image', kind: 'out' }
        ]
      },
      how: [
        'Replace all pooling with strided convolutions, so up- and down-sampling is learned rather than fixed.',
        'Use batch normalisation in both networks to keep activations well-scaled.',
        'Remove fully connected hidden layers entirely.',
        'Use ReLU in the generator, LeakyReLU in the discriminator — asymmetric on purpose, to keep gradient flowing to the generator.'
      ],
      keyIdea: 'No new theory, just a recipe that made an unstable idea reproducible. It also showed the noise vector was meaningfully structured — arithmetic on it produced sensible edits, hinting that the generator had learned real concepts.',
      strength: 'Turned GANs from a fragile curiosity into something practitioners could actually use.',
      limitation: 'Limited to fairly low resolution — around 64×64.',
      usedFor: ['image generation', 'the template most later GANs built on']
    },

    {
      id: 'stylegan',
      name: 'StyleGAN',
      year: '2018',
      learns: 'Adversarial',
      dataFor: 'High-resolution images',
      tagline: 'Inject style at every resolution, so coarse and fine details are controlled separately.',
      diagram: {
        type: 'stack',
        items: [
          { label: 'latent z → mapping network → w', kind: 'in' },
          { label: '4×4  — pose, face shape', kind: 'hidden' },
          { label: '32×32 — features, hair', kind: 'hidden' },
          { label: '1024² — skin, freckles', kind: 'out' }
        ],
        caption: 'style is applied separately at each scale'
      },
      how: [
        'Pass the noise vector through a mapping network first, producing an intermediate space that is less tangled than raw noise.',
        'Instead of feeding it in at the bottom, inject it at *every* resolution via adaptive instance normalisation.',
        'Add fresh random noise at each layer for stochastic detail — the exact placement of individual hairs and pores.',
        'Grow the resolution progressively during training, from 4×4 up to 1024×1024.'
      ],
      keyIdea: 'Injecting style per scale *disentangles* the factors of variation. Styles from early layers control pose and face shape; later layers control colour and texture — so you can mix two faces\' attributes independently.',
      strength: 'Produced the first genuinely photorealistic synthetic faces, with meaningful control.',
      limitation: 'Expensive to train, and specialised to aligned domains like faces.',
      usedFor: ['face generation', 'art', 'data augmentation', 'image editing']
    },

    {
      id: 'cyclegan',
      name: 'CycleGAN',
      year: '2017',
      learns: 'Adversarial (unpaired)',
      dataFor: 'Image domains',
      tagline: 'Translate between two image styles without a single matched pair.',
      diagram: {
        type: 'cycle',
        left: 'Horse → Zebra', right: 'Zebra → Horse',
        forward: 'translate', back: 'translate back'
      },
      how: [
        'Train two generators: one converting domain A to B, and one converting B back to A.',
        'Train two discriminators, one judging realism in each domain.',
        'Add the *cycle consistency* loss: translating a horse to a zebra and back must return the original horse.',
        'That round-trip constraint is what removes the need for paired training images.'
      ],
      keyIdea: 'Paired data — the same scene photographed in summer and winter — barely exists. Cycle consistency replaces it: the requirement to be reversible forces the translation to preserve content while changing only style.',
      strength: 'Needs only two unrelated piles of images, which is a far easier ask.',
      limitation: 'Handles texture and colour well, but cannot make large geometric changes.',
      usedFor: ['style transfer', 'photo enhancement', 'domain adaptation', 'medical imaging']
    },

    {
      id: 'diffusion',
      name: 'Diffusion Model',
      aka: 'DDPM, Stable Diffusion, Imagen',
      year: '2020',
      learns: 'Self-supervised (generative)',
      dataFor: 'Images, audio, video, molecules',
      tagline: 'Learn to remove noise, then start from pure noise and remove it all.',
      diagram: {
        type: 'conv',
        maps: [
          { size: 46, depth: 1, label: 'pure noise', sub: 't=1000', op: 'denoise' },
          { size: 46, depth: 1, label: 'vague shape', sub: 't=600', op: 'denoise' },
          { size: 46, depth: 1, label: 'structure', sub: 't=200', op: 'denoise' },
          { size: 46, depth: 1, label: 'image', sub: 't=0', kind: 'out' }
        ]
      },
      how: [
        'Take a real image and add a little Gaussian noise, repeatedly, until after a thousand steps nothing remains but static. This *forward process* needs no learning.',
        'Train a network — usually a U-Net — to look at a noisy image and predict the noise that was added.',
        'To generate, start from pure random noise and apply the trained denoiser repeatedly, removing a little each step.',
        'Guide the process with a text embedding, typically from CLIP, so the denoising is steered toward the prompt.'
      ],
      keyIdea: 'Generating an image in one shot is extremely hard; removing a small amount of noise is easy. Diffusion decomposes an impossible problem into a thousand easy ones — and unlike GANs, training is stable because it is plain supervised regression.',
      strength: 'Best-in-class sample quality and diversity, with stable training and strong text control.',
      limitation: 'Slow — many forward passes per image, though distillation has cut this to a handful.',
      usedFor: ['text-to-image', 'video generation', 'audio', 'molecule design', 'inpainting']
    },

    {
      id: 'latent-diffusion',
      name: 'Latent Diffusion',
      aka: 'Stable Diffusion',
      year: '2021',
      learns: 'Self-supervised (generative)',
      dataFor: 'High-resolution images',
      tagline: 'Run diffusion in a compressed space instead of on raw pixels.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'image\n512²', kind: 'in' },
          { label: 'VAE\nencode', kind: 'hidden' },
          { label: 'diffuse\n64² latent', kind: 'attn' },
          { label: 'VAE\ndecode', kind: 'out' }
        ],
        caption: 'diffusion happens in a space ~48× smaller'
      },
      how: [
        'Train a VAE to compress images roughly eightfold in each dimension, into a compact latent space.',
        'Run the entire diffusion process in that latent space rather than on pixels.',
        'Decode the final latent back into a full-resolution image with the VAE decoder.',
        'Inject the text prompt through cross-attention layers inside the denoising U-Net.'
      ],
      keyIdea: 'A 512×512 image is 786,432 numbers; its latent is about 16,000. Diffusing in the smaller space cuts cost by more than an order of magnitude — which is exactly why Stable Diffusion could run on consumer hardware while its predecessors needed a datacentre.',
      strength: 'Made high-quality open image generation feasible on a single GPU.',
      limitation: 'The VAE round-trip loses some fine detail, occasionally visible in text and faces.',
      usedFor: ['text-to-image', 'inpainting', 'consumer creative tools']
    },

    {
      id: 'normalizing-flows',
      name: 'Normalizing Flows',
      aka: 'RealNVP, Glow',
      year: '2015',
      learns: 'Self-supervised (generative)',
      dataFor: 'Continuous data',
      tagline: 'A stack of exactly reversible transformations from noise to data.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'noise', kind: 'in' },
          { label: 'invertible\nlayer', kind: 'hidden' },
          { label: 'invertible\nlayer', kind: 'hidden' },
          { label: 'data', kind: 'out' }
        ],
        caption: 'every arrow runs equally well in reverse'
      },
      how: [
        'Build the network from layers that are *invertible by construction*, so every step can be undone exactly.',
        'Design them so the determinant of the Jacobian is cheap to compute — the usual trick is to transform half the dimensions conditioned on the other half.',
        'The change-of-variables formula then gives the *exact* likelihood of any data point, not a bound.',
        'Generate by sampling noise and running the network forward; compute likelihood by running it backward.'
      ],
      keyIdea: 'Alone among generative models, flows give exact likelihoods. VAEs give a lower bound, GANs give nothing at all — which makes flows the right tool when you need calibrated density estimates rather than pretty samples.',
      strength: 'Exact likelihood, exact inversion, and fast sampling in one pass.',
      limitation: 'Invertibility is a severe architectural constraint, so sample quality lags diffusion badly.',
      usedFor: ['density estimation', 'anomaly detection', 'physics simulation', 'lossless compression']
    },

    {
      id: 'autoregressive-generative',
      name: 'Autoregressive Generation',
      aka: 'PixelCNN, WaveNet, PixelRNN',
      year: '2016',
      learns: 'Self-supervised (generative)',
      dataFor: 'Images, audio',
      tagline: 'Generate one pixel or sample at a time, each conditioned on all the previous ones.',
      diagram: {
        type: 'sequence',
        tokens: ['p₁', 'p₂', 'p₃', 'p₄', 'p₅', '?'],
        below: 'each conditioned on everything before it',
        caption: 'exact likelihood, but strictly sequential'
      },
      how: [
        'Impose an arbitrary order on the data — raster-scan for images, time order for audio.',
        'Model the probability of each element given all preceding elements.',
        'Multiply those conditional probabilities together to get the exact likelihood of the whole sample.',
        'Generate by sampling elements one at a time and feeding each back in.'
      ],
      keyIdea: 'The same principle as GPT, applied to pixels and waveforms. Exact and principled, but generating a 1024×1024 image means a million sequential steps — which is why images moved to diffusion while text stayed autoregressive.',
      strength: 'Exact likelihoods and very high fidelity, especially for audio.',
      limitation: 'Painfully slow generation, and the imposed ordering is artificial for 2-D data.',
      usedFor: ['raw audio synthesis', 'lossless compression', 'density modelling']
    },

    {
      id: 'energy-based',
      name: 'Energy-Based Models',
      aka: 'EBM',
      year: '1980s →',
      learns: 'Unsupervised',
      dataFor: 'Any data',
      tagline: 'Learn a landscape where real data sits in the valleys.',
      diagram: {
        type: 'scatter', boundary: 'curve',
        caption: 'low energy where data is, high energy everywhere else'
      },
      how: [
        'Define a network that maps any input to a single scalar *energy* value.',
        'Train so real data receives low energy and everything else receives high energy.',
        'Generate by starting anywhere and descending the energy landscape, usually with Langevin dynamics.',
        'The hard part is the normalising constant — summing over all possible inputs is intractable, so training relies on approximations.'
      ],
      keyIdea: 'The most flexible generative framework: any function producing a scalar will do, with no architectural constraints at all. That freedom is exactly why they are so difficult to train, and diffusion models are best understood as a well-behaved special case.',
      strength: 'Maximum architectural flexibility; composes naturally with other models.',
      limitation: 'Training is unstable and sampling is slow. Largely superseded in practice by diffusion.',
      usedFor: ['theoretical grounding', 'compositional generation', 'structured prediction']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
