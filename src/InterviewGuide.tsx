import { useEffect, useState } from 'react';
import type { PageId } from './App.tsx';

type LabPage = Exclude<PageId, 'home'>;
type GuideStep = { title: string; action: string; notice: string; question: string };
type Guide = { title: string; duration: string; steps: GuideStep[] };

const GUIDES: Record<LabPage, Guide> = {
  nn: { title: 'Why XOR needs a hidden layer', duration: 'About 4 minutes', steps: [
    { title: 'Start with a straight line', action: 'Set up XOR with two input features, no hidden layer and fresh weights.', notice: 'The model has two inputs and one output. Its decision boundary can only be a line.', question: 'What shapes can a model with no hidden layer represent?' },
    { title: 'Watch it reach its limit', action: 'Train the straight-line model for 120 epochs, then inspect its boundary and accuracy.', notice: 'XOR alternates labels across the four quadrants. One line cannot separate those groups.', question: 'Why does more training fail to fix a model that cannot represent the pattern?' },
    { title: 'Give it one hidden layer', action: 'Add four hidden units, train for 240 epochs, then compare the new boundary.', notice: 'Hidden units combine simple boundaries into a more useful shape.', question: 'What changed: the data, the optimiser, or the model’s capacity?' },
  ] },
  classical: { title: 'When a good fit is still wrong', duration: 'About 3 minutes', steps: [
    { title: 'Give a line curved data', action: 'Set up a fresh linear regression model on curved data.', notice: 'The residuals have a pattern because a straight line is too simple for the curve.', question: 'What does the shape of the residuals tell you about the model?' },
    { title: 'Compare the two answers', action: 'Fit for 90 steps, then compare the learned solid line with the exact dashed line.', notice: 'Gradient descent can find the best straight line here. The result still misses the curve.', question: 'How would you distinguish optimisation error from model error?' },
    { title: 'Compare with a lazy learner', action: 'Switch to k-Nearest Neighbours on XOR with k = 1. Change k manually to see the boundary smooth out.', notice: 'k-NN stores examples and predicts from nearby points. Its work happens at prediction time.', question: 'How does k change the bias and variance of the boundary?' },
  ] },
  conv: { title: 'How a convolution sees patterns', duration: 'About 3 minutes', steps: [
    { title: 'Inspect the starting filters', action: 'Reset four filters and show a cross probe image before training.', notice: 'The filters start as random values. Each one is shared across every location in the image.', question: 'Why reuse the same small filter across the whole image?' },
    { title: 'Train one step at a time', action: 'Train for 12 epochs and inspect how the filters, feature maps and loss change.', notice: 'Convolution finds local patterns; ReLU gates negative responses; max pooling keeps strong nearby matches.', question: 'What information does max pooling keep, and what does it discard?' },
    { title: 'Test a new drawing', action: 'Load a new cross drawing, then edit it on the 12×12 grid and watch the probabilities respond.', notice: 'The classifier combines local filter responses into a whole-image prediction.', question: 'How do local features become a global class prediction?' },
  ] },
  attention: { title: 'What attention is looking for', duration: 'About 3 minutes', steps: [
    { title: 'Read the sequence task', action: 'Reset the attention model and inspect the prefix, separator and copy in the probe sequence.', notice: 'Rows are positions asking for information; columns are positions they can read. The grey upper triangle masks future context.', question: 'Why must a causal model mask tokens that come later?' },
    { title: 'Train the heads', action: 'Train for 100 epochs, then inspect the attention maps and loss.', notice: 'A diagonal stripe shifted by the prefix length reveals the learned lookup.', question: 'How does the attention map help explain a token prediction?' },
    { title: 'Try an unseen sequence', action: 'Switch to a new probe sequence and inspect its predictions and attention maps.', notice: 'The token identities change, but the copy relationship stays the same.', question: 'What evidence suggests the model learned the task rather than memorised examples?' },
  ] },
  embeddings: { title: 'How tokens find their neighbours', duration: 'About 3 minutes', steps: [
    { title: 'Start with untrained vectors', action: 'Reset the vectors and select cat as the probe token.', notice: 'The starting positions are seeded random values. Nearby words have no learned relationship yet.', question: 'What does an embedding represent before training?' },
    { title: 'Let context shape the space', action: 'Train for 120 epochs, then inspect the vectors and nearest neighbours.', notice: 'Related words in this curated toy corpus appear in paired contexts. Training moves their vectors into more useful positions.', question: 'Why can shared context pull two different words closer together?' },
    { title: 'Test another word', action: 'Select kitten and compare its neighbours and position with cat.', notice: 'Similarity follows learned vector geometry from the toy corpus; the model never received a hand-written similarity score.', question: 'What can you infer from distance in this space, and what can you not infer?' },
  ] },
};

export function InterviewGuide({ page, open, onClose, onRunStep }: {
  page: LabPage; open: boolean; onClose: () => void; onRunStep: (step: number) => void;
}) {
  const [stepIndex, setStepIndex] = useState(0);
  const guide = GUIDES[page];
  const step = guide.steps[stepIndex];
  useEffect(() => setStepIndex(0), [page]);
  if (!open) return null;
  const previous = () => setStepIndex(i => Math.max(0, i - 1));
  const next = () => setStepIndex(i => Math.min(guide.steps.length - 1, i + 1));
  return (
    <aside className="interview-guide" aria-labelledby="guide-title" aria-live="polite">
      <div className="guide-topline">
        <span className="guide-kicker"><i aria-hidden /> INTERVIEW WALKTHROUGH</span>
        <button className="guide-close" onClick={onClose} aria-label="Close interview guide">×</button>
      </div>
      <h2 id="guide-title">{guide.title}</h2>
      <div className="guide-progress-row"><span>STEP {String(stepIndex + 1).padStart(2, '0')} / {String(guide.steps.length).padStart(2, '0')}</span><span>{guide.duration}</span></div>
      <div className="guide-progress" aria-label={`Step ${stepIndex + 1} of ${guide.steps.length}`}>
        {guide.steps.map((_, i) => <i key={i} className={i <= stepIndex ? 'done' : undefined} />)}
      </div>
      <div className="guide-step-content">
        <h3>{step.title}</h3>
        <p className="guide-label">DO THIS</p>
        <p className="guide-action">{step.action}</p>
        <button className="guide-run btn primary" type="button" onClick={() => onRunStep(stepIndex)}>{stepIndex === 0 ? 'Set up this step' : 'Run this step'}</button>
        <p className="guide-label">WHAT TO NOTICE</p>
        <p className="guide-notice">{step.notice}</p>
        <div className="guide-question"><span>ASK YOURSELF</span><p>{step.question}</p></div>
      </div>
      <div className="guide-controls">
        <button className="guide-prev" onClick={previous} disabled={stepIndex === 0}>← Back</button>
        {stepIndex < guide.steps.length - 1 ? <button className="guide-next" onClick={next}>Next step <span aria-hidden>→</span></button> : <button className="guide-next" onClick={() => setStepIndex(0)}>Start again <span aria-hidden>↺</span></button>}
      </div>
    </aside>
  );
}
