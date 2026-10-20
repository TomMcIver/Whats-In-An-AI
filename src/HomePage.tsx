import type { PageId } from './App.tsx';

const chapters: {
  id: PageId;
  number: string;
  eyebrow: string;
  title: string;
  description: string;
  takeaway: string;
  art: string;
}[] = [
  {
    id: 'classical', number: '01', eyebrow: 'The foundations', title: 'Classical machine learning',
    description: 'Five ways to learn from data, from fitting a line to carving the world into regions.',
    takeaway: 'Compare how different algorithms learn', art: 'art-classical',
  },
  {
    id: 'nn', number: '02', eyebrow: 'Learning representations', title: 'Neural networks',
    description: 'Follow signals through a multilayer network and watch its decision boundary take shape.',
    takeaway: 'See weights, activations and gradients', art: 'art-neural',
  },
  {
    id: 'conv', number: '03', eyebrow: 'Learning to see', title: 'Convolution',
    description: 'Watch small filters discover useful patterns in simple images, one training step at a time.',
    takeaway: 'Draw an image and inspect its feature maps', art: 'art-conv',
  },
  {
    id: 'embeddings', number: '04', eyebrow: 'Learning relationships', title: 'Embeddings',
    description: 'Train tiny word vectors on a curated context task and watch neighbours take shape in 2D.',
    takeaway: 'Compare learned positions and context predictions', art: 'art-embeddings',
  },
  {
    id: 'attention', number: '05', eyebrow: 'Learning what matters', title: 'Attention',
    description: 'Train a tiny sequence model and see which earlier tokens each attention head reads.',
    takeaway: 'Reveal the pattern behind a copy task', art: 'art-attention',
  },
];

function NetworkArtwork() {
  const layers = [
    [{ x: 42, y: 62 }, { x: 42, y: 138 }, { x: 42, y: 214 }],
    [{ x: 154, y: 42 }, { x: 154, y: 100 }, { x: 154, y: 158 }, { x: 154, y: 216 }, { x: 154, y: 274 }],
    [{ x: 266, y: 74 }, { x: 266, y: 138 }, { x: 266, y: 202 }],
    [{ x: 376, y: 138 }],
  ];
  return (
    <svg className="hero-network" viewBox="0 0 420 310" role="img" aria-label="A neural network diagram, with connections between four layers of neurons">
      <defs>
        <linearGradient id="pathGlow" x1="0" x2="1">
          <stop offset="0" stopColor="#d5a15b" stopOpacity=".24" />
          <stop offset="1" stopColor="#6c8a70" stopOpacity=".7" />
        </linearGradient>
      </defs>
      {layers.slice(0, -1).flatMap((layer, li) => layer.flatMap((from, fi) =>
        layers[li + 1].map((to, ti) => (
          <line key={`${li}-${fi}-${ti}`} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                className={`wire wire-${(li + fi + ti) % 4}`} />
        ))
      ))}
      {layers.flatMap((layer, li) => layer.map((node, ni) => (
        <g key={`${li}-${ni}`} className="neuron" style={{ animationDelay: `${(li * 3 + ni) * 90}ms` }}>
          <circle cx={node.x} cy={node.y} r={li === 3 ? 16 : 11} className={li === 3 ? 'neuron-output' : `neuron-${li}`} />
          <circle cx={node.x} cy={node.y} r="3" className="neuron-core" />
        </g>
      )))}
      <text x="42" y="300" textAnchor="middle">INPUT</text>
      <text x="154" y="300" textAnchor="middle">HIDDEN</text>
      <text x="266" y="300" textAnchor="middle">HIDDEN</text>
      <text x="376" y="300" textAnchor="middle">OUTPUT</text>
      <path d="M48 28 C120 -4 278 0 349 28" className="orbit" />
      <circle cx="349" cy="28" r="3" className="orbit-dot" />
    </svg>
  );
}

export function HomePage({ onNavigate, onStartGuide }: {
  onNavigate: (page: PageId) => void;
  onStartGuide: () => void;
}) {
  return (
    <div className="home-page">
      <header className="home-nav">
        <a href="#/" className="home-brand" aria-label="What's In An AI home">
          <span className="logo-mark" aria-hidden />
          <span>WHAT'S IN AN AI<span className="brand-period">.</span></span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#journey">Explore the models</a>
          <a href="#approach">How it works</a>
        </nav>
        <a className="nav-cta" href="#/nn">Open the lab <span aria-hidden>↗</span></a>
      </header>

      <main>
        <section className="hero-shell">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-dot" /> AN INTERACTIVE FIELD GUIDE TO MACHINE LEARNING</div>
            <h1>AI makes more sense<br />when you can <em>see it learn.</em></h1>
            <p className="hero-description">From the first decision boundary to the attention inside a language model, explore what happens beneath the surface.</p>
            <div className="hero-actions">
              <button className="button-dark" onClick={onStartGuide}>Start the XOR walkthrough <span aria-hidden>→</span></button>
              <a href="#journey" className="text-link">Explore the labs <span aria-hidden>↓</span></a>
            </div>
            <div className="hero-note"><span className="note-line" /> BUILT FROM SCRATCH · RUNS IN YOUR BROWSER</div>
          </div>
          <div className="hero-art-wrap">
            <div className="art-label art-label-top"><span>THE HIDDEN LAYERS</span><i>01 — 05</i></div>
            <NetworkArtwork />
            <div className="art-caption"><span className="caption-mark">↗</span><span>Signals move forward.<br /><b>Learning moves the weights.</b></span></div>
            <div className="hero-index">A SMALL WINDOW INTO<br />HOW MODELS WORK</div>
          </div>
          <div className="hero-bottomline"><span>01 / 05</span><span>START WITH A QUESTION. LEAVE WITH A MENTAL MODEL.</span><span>SCROLL TO EXPLORE ↓</span></div>
        </section>

        <section className="intro-band" id="approach">
          <p className="section-kicker">THE IDEA</p>
          <p className="intro-statement">A model is more than an answer. <span>It’s a process — made of choices, signals and small updates.</span> This is a place to watch that process happen.</p>
          <div className="intro-aside"><span className="aside-number">19</span><span>gradient checks<br />keep the maths honest</span><span className="aside-separator" /><span className="aside-number">0</span><span>machine learning<br />libraries used</span></div>
        </section>

        <section className="journey-section" id="journey">
          <div className="section-heading">
            <div><p className="section-kicker">A PATH THROUGH THE IDEAS</p><h2>Start simple.<br /><em>Follow the signal.</em></h2></div>
            <p className="section-heading-note">Each lab is a small, working model. Change what goes in, step through training, and inspect what changes inside.</p>
          </div>
          <div className="chapter-list">
            {chapters.map(chapter => (
              <article className="chapter" key={chapter.id}>
                <button className="chapter-hit" onClick={() => onNavigate(chapter.id)} aria-label={`Explore ${chapter.title}`}>
                  <div className="chapter-meta"><span>{chapter.number}</span><span>{chapter.eyebrow}</span></div>
                  <div className="chapter-main"><h3>{chapter.title}</h3><p>{chapter.description}</p><span className="chapter-takeaway">↳ &nbsp;{chapter.takeaway}</span></div>
                  <div className={`chapter-art ${chapter.art}`} aria-hidden>
                    {chapter.id === 'classical' && <><span className="scatter-dot d1"/><span className="scatter-dot d2"/><span className="scatter-dot d3"/><span className="scatter-dot d4"/><span className="scatter-dot d5"/><span className="scatter-dot d6"/><span className="fit-line"/><span className="scatter-axis x"/><span className="scatter-axis y"/></>}
                    {chapter.id === 'nn' && <NetworkArtwork />}
                    {chapter.id === 'conv' && <div className="conv-tiles"><span>░▒</span><span>▒█</span><span>░█</span><span>██</span><span>▒░</span><span>██</span><span>█▒</span><span>░▒</span><span>▒█</span></div>}
                    {chapter.id === 'embeddings' && <div className="embedding-art"><span className="embedding-axis embedding-axis-x"/><span className="embedding-axis embedding-axis-y"/><i className="embedding-point ep-one"/><i className="embedding-point ep-two"/><i className="embedding-point ep-three"/><i className="embedding-point ep-four"/><i className="embedding-point ep-five"/><i className="embedding-point ep-six"/><span className="embedding-art-label label-one">cat</span><span className="embedding-art-label label-two">dog</span><span className="embedding-art-label label-three">car</span></div>}
                    {chapter.id === 'attention' && <div className="attention-grid">{Array.from({ length: 64 }, (_, i) => <i key={i} style={{ opacity: i % 9 === 0 || i % 11 === 0 ? .9 : i % 7 === 0 ? .5 : .13 }} />)}</div>}
                  </div>
                  <span className="chapter-arrow" aria-hidden>↗</span>
                </button>
              </article>
            ))}
          </div>
        </section>

        <section className="closing-note">
          <span className="closing-star" aria-hidden>✳</span>
          <p>Not a diagram of a model.<br /><em>The model, learning in front of you.</em></p>
          <button onClick={() => onNavigate('nn')}>Take a closer look <span aria-hidden>↗</span></button>
        </section>
      </main>

      <footer className="home-footer"><a href="#/" className="footer-brand">WHAT'S IN AN AI<span>.</span></a><span>Small models. Visible learning. A better feel for what AI is doing.</span><a href="https://github.com/TomMcIver/Whats-In-An-AI" target="_blank" rel="noreferrer">SOURCE CODE <span aria-hidden>↗</span></a></footer>
    </div>
  );
}
