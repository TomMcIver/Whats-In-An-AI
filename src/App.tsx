import { useEffect, useState } from 'react';
import { NeuralNetLab } from './labs/NeuralNetLab.tsx';
import { ClassicalLab } from './labs/ClassicalLab.tsx';
import { ConvLab } from './labs/ConvLab.tsx';
import { AttentionLab } from './labs/AttentionLab.tsx';
import { HomePage } from './HomePage.tsx';

export type PageId = 'home' | 'classical' | 'nn' | 'conv' | 'attention';

const PAGES: { id: PageId; label: string }[] = [
  { id: 'classical', label: 'Classical ML' },
  { id: 'nn', label: 'Neural nets' },
  { id: 'conv', label: 'Convolution' },
  { id: 'attention', label: 'Attention' },
];

function readPage(): PageId {
  const route = window.location.hash.replace(/^#\/?/, '');
  if (route === 'classical' || route === 'nn' || route === 'conv' || route === 'attention') return route;
  return 'home';
}

export default function App() {
  const [page, setPage] = useState<PageId>(readPage);

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
        <HomePage onNavigate={navigate} />
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
            <button className="back-home" onClick={() => navigate('home')}>← Overview</button>
          </header>
          {page === 'classical' && <ClassicalLab />}
          {page === 'nn' && <NeuralNetLab />}
          {page === 'conv' && <ConvLab />}
          {page === 'attention' && <AttentionLab />}
        </>
      )}
    </div>
  );
}
