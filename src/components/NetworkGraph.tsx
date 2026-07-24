/**
 * The live network diagram.
 *
 * Edges  — one per weight. Colour = sign, thickness + opacity = |value|.
 * Nodes  — fill shows the activation for the probe input; the ring shows
 *          the bias. Both update on every render, so training is visible
 *          as the picture physically changing.
 */

import type { MLP } from '../lib/nn.ts';

interface Props {
  net: MLP;
  inputLabels: string[];
  /** activations of the input layer for the probe point */
  inputValues: number[];
  /** highlight the weights feeding this [layer, unit], or null */
  selected?: { layer: number; unit: number } | null;
  onSelect?: (sel: { layer: number; unit: number } | null) => void;
}

const POS = '#d97a45';
const NEG = '#5b8dbe';

/** map a signed value to a colour + alpha, scaled against the largest magnitude */
function edgeStyle(w: number, max: number) {
  const t = Math.min(Math.abs(w) / max, 1);
  return {
    stroke: w >= 0 ? POS : NEG,
    width: 0.4 + t * 3.4,
    opacity: 0.12 + t * 0.78,
  };
}

/** activation -> fill. tanh-ish range assumed, clamped */
function nodeFill(a: number) {
  const t = Math.max(-1, Math.min(1, a));
  if (t >= 0) return `color-mix(in srgb, ${POS} ${Math.round(t * 100)}%, #2a2a24)`;
  return `color-mix(in srgb, ${NEG} ${Math.round(-t * 100)}%, #2a2a24)`;
}

export function NetworkGraph({ net, inputLabels, inputValues, selected, onSelect }: Props) {
  const cols: { n: number; kind: 'in' | 'hidden' | 'out' }[] = [
    { n: net.sizes[0], kind: 'in' },
    ...net.layers.map((L, i) => ({
      n: L.nOut,
      kind: (i === net.layers.length - 1 ? 'out' : 'hidden') as 'hidden' | 'out',
    })),
  ];

  const R = 13;
  const colGap = 116;
  const rowGap = 40;
  const padX = 74;
  const padY = 42;

  const maxRows = Math.max(...cols.map(c => c.n));
  const W = padX * 2 + (cols.length - 1) * colGap;
  const H = padY * 2 + Math.max(maxRows - 1, 1) * rowGap;

  const pos = (ci: number, ui: number) => ({
    x: padX + ci * colGap,
    y: H / 2 + (ui - (cols[ci].n - 1) / 2) * rowGap,
  });

  // activation per column: column 0 is the raw input, the rest come from layers
  const actOf = (ci: number, ui: number) =>
    ci === 0 ? (inputValues[ui] ?? 0) : (net.layers[ci - 1].a[ui] ?? 0);

  const maxW = net.maxAbsWeight();

  return (
    <div className="netwrap">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img"
           aria-label="Neural network with live weights and activations">

        {/* edges: layer index l connects column l to column l+1 */}
        {net.layers.map((L, l) =>
          L.w.map((row, j) =>
            row.map((w, i) => {
              const a = pos(l, i);
              const b = pos(l + 1, j);
              const st = edgeStyle(w, maxW);
              const dim = selected && !(selected.layer === l && selected.unit === j);
              return (
                <line
                  key={`e${l}-${j}-${i}`}
                  x1={a.x + R} y1={a.y} x2={b.x - R} y2={b.y}
                  stroke={st.stroke}
                  strokeWidth={st.width}
                  opacity={dim ? st.opacity * 0.13 : st.opacity}
                >
                  <title>{`w[${l}][${j}][${i}] = ${w.toFixed(4)}`}</title>
                </line>
              );
            })
          )
        )}

        {/* nodes */}
        {cols.map((c, ci) =>
          Array.from({ length: c.n }, (_, ui) => {
            const p = pos(ci, ui);
            const a = actOf(ci, ui);
            const bias = ci === 0 ? null : net.layers[ci - 1].b[ui];
            const isSel = selected?.layer === ci - 1 && selected?.unit === ui;

            // bias drawn as a ring: warm = positive, cool = negative
            const biasRing = bias === null ? '#454540'
              : bias >= 0 ? POS : NEG;
            const biasWidth = bias === null ? 1
              : 1 + Math.min(Math.abs(bias) / 1.2, 1) * 3;

            return (
              <g key={`n${ci}-${ui}`}
                 style={{ cursor: ci === 0 ? 'default' : 'pointer' }}
                 onClick={() => {
                   if (ci === 0 || !onSelect) return;
                   onSelect(isSel ? null : { layer: ci - 1, unit: ui });
                 }}>
                <circle
                  cx={p.x} cy={p.y} r={R}
                  fill={nodeFill(a)}
                  stroke={biasRing}
                  strokeWidth={biasWidth}
                />
                <text x={p.x} y={p.y + 3.2} textAnchor="middle"
                      fontSize="8.5" fontFamily="ui-monospace, monospace"
                      fill={Math.abs(a) > 0.55 ? '#14140f' : '#b8b5ac'}
                      pointerEvents="none">
                  {a.toFixed(2).replace('0.', '.').replace('-.', '-.')}
                </text>
                <title>
                  {ci === 0
                    ? `${inputLabels[ui] ?? 'input'} = ${a.toFixed(4)}`
                    : `activation = ${a.toFixed(4)}\nbias = ${bias!.toFixed(4)}`}
                </title>
              </g>
            );
          })
        )}

        {/* column captions */}
        {cols.map((c, ci) => {
          const p = pos(ci, 0);
          const label = ci === 0 ? 'input'
            : ci === cols.length - 1 ? 'output'
            : `hidden ${ci}`;
          return (
            <text key={`c${ci}`} x={p.x} y={18} textAnchor="middle"
                  fontSize="10" fill="#74716a">
              {label} · {c.n}
            </text>
          );
        })}

        {/* input feature names down the left */}
        {inputLabels.map((lab, i) => {
          const p = pos(0, i);
          return (
            <text key={`il${i}`} x={p.x - R - 8} y={p.y + 3.5} textAnchor="end"
                  fontSize="10.5" fontFamily="ui-monospace, monospace" fill="#a9a69c">
              {lab}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
