import { useEffect, useState } from 'react';
import { NeuralNetLab } from './labs/NeuralNetLab.tsx';
import { ClassicalLab } from './labs/ClassicalLab.tsx';
import { ConvLab } from './labs/ConvLab.tsx';
import { AttentionLab } from './labs/AttentionLab.tsx';
import { EmbeddingsLab } from './labs/EmbeddingsLab.tsx';
import { HomePage } from './HomePage.tsx';
import { InterviewGuide } from './InterviewGuide.tsx';

export type PageId = 'home' | 'classical' | 'nn' | 'conv' | 'attention' | 'embeddings';

const PAGES: { id: PageId; label: string }[] = [
  { id: 'classical', label: 'Classical ML' },
  { id: 'nn', label: 'Neural nets' },
  { id: 'conv', label: 'Convolution' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'attention', label: 'Attention' },
];

function readPage(): PageId {
  const route = window.location.hash.replace(/^#\/?/, '');
  if (route === 'classical' || route === 'nn' || route === 'conv' || route === 'embeddings' || route === 'attention') return route;
  return 'home';
}

export default function App() {
  const [page, setPage] = useState<PageId>(readPage);
  const [guideOpen, setGuideOpen] = useState(false);
  const [guideAction, setGuideAction] = useState<{ page: PageId; step: number; id: number } | null>(null);

  const runGuideStep = (step: number, actionPage: PageId = page) => {
    setGuideAction(previous => ({ page: actionPage, step, id: (previous?.id ?? 0) + 1 }));
  };

  const actionForPage = guideAction?.page === page ? guideAction : null;

  useEffect(() => {
    const syncPage = () => setPage(readPage());
    window.addEventListener('hashchange', syncPage);
    return () => window.removeEventListener('hashchange', syncPage);
  }, []);

  const navigate = (next: PageId) => {
    const hash = next === 'home' ? '#/' : `#/${next}`;
    if (window.location.hash === hash) setPage(next);
    else window.location.hash = hash;
  };

  return (
    <div className="app">
      {page === 'home' ? (
        <HomePage onNavigate={navigate} onStartGuide={() => {
          navigate('nn');
          setGuideOpen(true);
          runGuideStep(0, 'nn');
        }} />
      ) : (
        <>
          <header className="topbar lab-topbar">
            <button className="brand-button" onClick={() => navigate('home')} aria-label="Back to What's In An AI home">
              <span className="logo-mark" aria-hidden />
              <span className="brand-name">What's In An AI</span>
            </button>
            <nav className="tabs" aria-label="Model labs">
              {PAGES.map(item => (
                <button key={item.id} className={'tab' + (page === item.id ? ' on' : '')}
                        aria-current={page === item.id ? 'page' : undefined}
                        onClick={() => navigate(item.id)}>{item.label}</button>
              ))}
            </nav>
            <button className={'guide-launcher' + (guideOpen ? ' active' : '')}
                    aria-expanded={guideOpen}
                    onClick={() => setGuideOpen(open => !open)}>
              <span aria-hidden>✳</span> Interview guide
            </button>
            <button className="back-home" onClick={() => navigate('home')}>← Overview</button>
          </header>
          {page === 'classical' && <ClassicalLab guideAction={actionForPage} />}
          {page === 'nn' && <NeuralNetLab guideAction={actionForPage} />}
          {page === 'conv' && <ConvLab guideAction={actionForPage} />}
          {page === 'embeddings' && <EmbeddingsLab guideAction={actionForPage} />}
          {page === 'attention' && <AttentionLab guideAction={actionForPage} />}
          <InterviewGuide key={page} page={page} open={guideOpen} onClose={() => setGuideOpen(false)} onRunStep={step => runGuideStep(step)} />
        </>
      )}
    </div>
  );
}
