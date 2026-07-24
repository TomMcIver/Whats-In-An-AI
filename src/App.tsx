import { useState } from 'react';
import { NeuralNetLab } from './labs/NeuralNetLab.tsx';
import { ClassicalLab } from './labs/ClassicalLab.tsx';
import { ConvLab } from './labs/ConvLab.tsx';

type TabId = 'nn' | 'classical' | 'conv' | 'attention';

const TABS: { id: TabId; label: string; ready: boolean }[] = [
  { id: 'nn',        label: 'Neural Network', ready: true },
  { id: 'classical', label: 'Classical',      ready: true },
  { id: 'conv',      label: 'Convolution',    ready: true },
  { id: 'attention', label: 'Attention',      ready: false },
];

export default function App() {
  const [tab, setTab] = useState<TabId>('nn');

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo">
          <span className="logo-mark" aria-hidden />
          <div>
            <h1>What's In An AI</h1>
            <p>Models you can actually train, running in your browser</p>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map(t => (
            <button key={t.id}
                    className={'tab' + (tab === t.id ? ' on' : '')}
                    onClick={() => setTab(t.id)}
                    disabled={!t.ready}
                    title={t.ready ? undefined : 'Not built yet'}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === 'nn' && <NeuralNetLab />}
      {tab === 'classical' && <ClassicalLab />}
      {tab === 'conv' && <ConvLab />}
    </div>
  );
}
