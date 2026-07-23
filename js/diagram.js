/* ------------------------------------------------------------------
   What's In An AI — diagram engine

   Turns a small declarative spec into an inline SVG. Everything is
   drawn from CSS custom properties so diagrams follow the page theme.

   Usage:  WIAA.diagram({ type: 'layers', cols: [...] })  -> svg string
   ------------------------------------------------------------------ */

window.WIAA = window.WIAA || {};

(function (NS) {
  'use strict';

  var W = 440, H = 210;          // shared viewBox for every diagram
  var C = {
    node:  'var(--d-node)',
    line:  'var(--d-node-line)',
    in:    'var(--d-in)',
    hid:   'var(--d-hidden)',
    out:   'var(--d-out)',
    attn:  'var(--d-attn)',
    edge:  'var(--d-edge)',
    text:  'var(--d-text)'
  };

  function kindColor(k) {
    return k === 'in' ? C.in : k === 'out' ? C.out
         : k === 'attn' ? C.attn : k === 'hidden' ? C.hid : C.node;
  }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* ---------- primitives ---------- */

  function rect(x, y, w, h, o) {
    o = o || {};
    return '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h +
      '" rx="' + (o.r == null ? 4 : o.r) + '" fill="' + (o.fill || C.node) +
      '" stroke="' + (o.stroke || C.line) + '" stroke-width="' + (o.sw || 1) +
      '" opacity="' + (o.op == null ? 1 : o.op) + '"/>';
  }
  function circ(cx, cy, r, o) {
    o = o || {};
    return '<circle cx="' + cx + '" cy="' + cy + '" r="' + r +
      '" fill="' + (o.fill || C.node) + '" stroke="' + (o.stroke || C.line) +
      '" stroke-width="' + (o.sw || 1) + '" opacity="' + (o.op == null ? 1 : o.op) + '"/>';
  }
  function line(x1, y1, x2, y2, o) {
    o = o || {};
    return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 +
      '" stroke="' + (o.stroke || C.edge) + '" stroke-width="' + (o.sw || 1) +
      '" opacity="' + (o.op == null ? 1 : o.op) + '"' +
      (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') +
      (o.arrow ? ' marker-end="url(#wa)"' : '') + '/>';
  }
  function path(d, o) {
    o = o || {};
    return '<path d="' + d + '" fill="' + (o.fill || 'none') +
      '" stroke="' + (o.stroke || C.edge) + '" stroke-width="' + (o.sw || 1) +
      '" opacity="' + (o.op == null ? 1 : o.op) + '"' +
      (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') +
      (o.arrow ? ' marker-end="url(#wa)"' : '') + '/>';
  }
  function text(x, y, s, o) {
    o = o || {};
    return '<text x="' + x + '" y="' + y + '" fill="' + (o.fill || C.text) +
      '" font-size="' + (o.size || 9) + '" font-family="ui-sans-serif,system-ui,sans-serif"' +
      ' text-anchor="' + (o.anchor || 'middle') + '"' +
      (o.weight ? ' font-weight="' + o.weight + '"' : '') +
      (o.style ? ' font-style="' + o.style + '"' : '') +
      '>' + esc(s) + '</text>';
  }
  /* a labelled box, vertically centred text (wraps on \n) */
  function box(x, y, w, h, label, o) {
    o = o || {};
    var s = rect(x, y, w, h, o);
    if (label) {
      var lines = String(label).split('\n');
      var fs = o.size || 9;
      var start = y + h / 2 - (lines.length - 1) * (fs + 1.5) / 2 + fs / 3;
      for (var i = 0; i < lines.length; i++) {
        s += text(x + w / 2, start + i * (fs + 1.5), lines[i],
                  { size: fs, fill: o.tfill || C.text, weight: o.weight });
      }
    }
    return s;
  }

  /* ---------- renderers ---------- */

  var R = {};

  /* columns of neurons, optionally fully connected */
  R.layers = function (s) {
    var cols = s.cols, n = cols.length, out = '';
    var padX = 44, span = W - padX * 2;
    var xs = cols.map(function (_, i) { return n === 1 ? W / 2 : padX + span * i / (n - 1); });
    var cy = H / 2 - 6, r = s.r || 8, gap = s.gap || 24;

    var pts = cols.map(function (c, i) {
      var k = c.n, arr = [];
      for (var j = 0; j < k; j++) arr.push({ x: xs[i], y: cy - (k - 1) * gap / 2 + j * gap });
      return arr;
    });

    if (s.connect !== false) {
      for (var i = 0; i < n - 1; i++)
        for (var a = 0; a < pts[i].length; a++)
          for (var b = 0; b < pts[i + 1].length; b++)
            out += line(pts[i][a].x + r, pts[i][a].y, pts[i + 1][b].x - r, pts[i + 1][b].y,
                        { op: .3, sw: .7 });
    }
    cols.forEach(function (c, i) {
      pts[i].forEach(function (p) {
        out += circ(p.x, p.y, r, { fill: kindColor(c.kind), stroke: C.line });
      });
      if (c.label) out += text(xs[i], H - 22, c.label, { size: 9.5, fill: C.text });
      if (c.sub)   out += text(xs[i], H - 11, c.sub, { size: 8, fill: C.edge });
    });
    return out;
  };

  /* horizontal pipeline of labelled boxes */
  R.blocks = function (s) {
    var it = s.items, n = it.length, out = '';
    var gap = s.gap || 16, padX = 22;
    var bw = (W - padX * 2 - gap * (n - 1)) / n;
    var bh = s.bh || 54, y = H / 2 - bh / 2 - 6;

    it.forEach(function (b, i) {
      var x = padX + i * (bw + gap);
      out += box(x, y, bw, bh, b.label, {
        fill: kindColor(b.kind), op: b.kind ? .8 : .55, r: 5,
        size: bw < 60 ? 8 : 9, weight: 600
      });
      if (b.sub) out += text(x + bw / 2, y + bh + 13, b.sub, { size: 7.8, fill: C.edge });
      if (i < n - 1) out += line(x + bw + 2, y + bh / 2, x + bw + gap - 3, y + bh / 2, { arrow: 1, sw: 1.2 });
    });
    if (s.caption) out += text(W / 2, H - 12, s.caption, { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* vertical stack, for repeated layer blocks */
  R.stack = function (s) {
    var it = s.items, n = it.length, out = '';
    var bh = s.bh || 22, gap = 7, bw = s.bw || 190;
    var x = W / 2 - bw / 2;
    var total = n * bh + (n - 1) * gap;
    var y0 = (H - total) / 2 - 4;

    it.forEach(function (b, i) {
      var y = y0 + i * (bh + gap);
      out += box(x, y, bw, bh, b.label, {
        fill: kindColor(b.kind), op: b.kind ? .8 : .55, r: 4, size: 8.5, weight: 600
      });
      if (i < n - 1) out += line(W / 2, y + bh + 1, W / 2, y + bh + gap - 2, { arrow: 1, sw: 1 });
    });
    if (s.repeat) {
      var bx = x + bw + 12;
      out += path('M' + bx + ' ' + y0 + ' q7 0 7 7 v' + (total - 14) + ' q0 7 -7 7', { sw: 1, stroke: C.attn });
      out += text(bx + 22, y0 + total / 2 + 3, s.repeat, { size: 9, fill: C.attn, weight: 600, anchor: 'start' });
    }
    if (s.caption) out += text(W / 2, H - 8, s.caption, { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* shrinking feature maps — convolutional pipelines */
  R.conv = function (s) {
    var m = s.maps, n = m.length, out = '';
    var padX = 30, span = W - padX * 2;
    var step = span / n;

    m.forEach(function (f, i) {
      var x = padX + i * step, sz = f.size || 46, depth = Math.min(f.depth || 3, 5);
      var cy = H / 2 - 10, off = 3.5;
      for (var d = depth - 1; d >= 0; d--) {
        out += rect(x + d * off, cy - sz / 2 + d * off, sz, sz, {
          fill: kindColor(f.kind || (i === 0 ? 'in' : i === n - 1 ? 'out' : 'hidden')),
          op: .45 + .12 * (depth - d), r: 2, stroke: C.line
        });
      }
      if (f.label) out += text(x + sz / 2, H - 26, f.label, { size: 8.5 });
      if (f.sub)   out += text(x + sz / 2, H - 15, f.sub, { size: 7.5, fill: C.edge });
      if (i < n - 1) {
        var ax = x + sz + depth * off + 1;
        out += line(ax, cy, padX + (i + 1) * step - 3, cy, { arrow: 1, sw: 1.1 });
        if (f.op) out += text((ax + padX + (i + 1) * step) / 2, cy - 10, f.op, { size: 7.5, fill: C.attn });
      }
    });
    return out;
  };

  /* unrolled recurrent cells with a feedback loop */
  R.recurrent = function (s) {
    var n = s.steps || 4, out = '';
    var bw = 54, bh = 42, gap = (W - 60 - n * bw) / (n - 1);
    var y = H / 2 - bh / 2 - 4, x0 = 30;

    for (var i = 0; i < n; i++) {
      var x = x0 + i * (bw + gap);
      out += box(x, y, bw, bh, s.cell || 'RNN', { fill: C.hid, op: .75, r: 5, weight: 600, size: 9 });
      /* input below, output above */
      out += line(x + bw / 2, y + bh + 22, x + bw / 2, y + bh + 3, { arrow: 1, sw: 1 });
      out += text(x + bw / 2, y + bh + 33, (s.inputs && s.inputs[i]) || ('x' + (i + 1)), { size: 8.5, fill: C.in });
      out += line(x + bw / 2, y - 3, x + bw / 2, y - 20, { arrow: 1, sw: 1 });
      out += text(x + bw / 2, y - 24, (s.outputs && s.outputs[i]) || ('h' + (i + 1)), { size: 8.5, fill: C.out });
      /* hidden state passed sideways */
      if (i < n - 1) out += line(x + bw + 1, y + bh / 2, x + bw + gap - 3, y + bh / 2,
                                 { arrow: 1, sw: 1.3, stroke: C.attn });
    }
    if (s.caption) out += text(W / 2, H - 4, s.caption, { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* two-tower encoder / decoder */
  R.encdec = function (s) {
    var out = '', bw = 116, bh = 96;
    var lx = 46, rx = W - 46 - bw, y = 42;

    out += box(lx, y, bw, bh, '', { fill: C.in, op: .3, r: 7 });
    out += text(lx + bw / 2, y - 8, s.leftTitle || 'Encoder', { size: 10, weight: 600 });
    out += box(rx, y, bw, bh, '', { fill: C.out, op: .3, r: 7 });
    out += text(rx + bw / 2, y - 8, s.rightTitle || 'Decoder', { size: 10, weight: 600 });

    (s.left || []).forEach(function (l, i) {
      out += box(lx + 11, y + 12 + i * 26, bw - 22, 20, l, { fill: C.in, op: .65, r: 3, size: 8 });
    });
    (s.right || []).forEach(function (l, i) {
      out += box(rx + 11, y + 12 + i * 26, bw - 22, 20, l, { fill: C.out, op: .65, r: 3, size: 8 });
    });

    var my = y + bh / 2;
    out += line(lx + bw + 3, my, rx - 4, my, { arrow: 1, sw: 1.4, stroke: C.attn });
    out += text((lx + bw + rx) / 2, my - 8, s.bridge || 'context', { size: 8.5, fill: C.attn, weight: 600 });

    out += text(lx + bw / 2, y + bh + 18, s.leftIn || 'input', { size: 8.5, fill: C.edge });
    out += text(rx + bw / 2, y + bh + 18, s.rightOut || 'output', { size: 8.5, fill: C.edge });
    return out;
  };

  /* binary decision tree */
  R.tree = function (s) {
    var depth = s.depth || 3, out = '';
    var top = 28, levelH = (H - top - 34) / (depth - 1);

    for (var d = 0; d < depth; d++) {
      var k = Math.pow(2, d), y = top + d * levelH;
      for (var i = 0; i < k; i++) {
        var x = W * (i + .5) / k;
        var leaf = d === depth - 1;
        if (d > 0) {
          var px = W * (Math.floor(i / 2) + .5) / Math.pow(2, d - 1);
          out += line(px, y - levelH + 9, x, y - 9, { sw: 1, op: .75 });
          if (d === 1) out += text((px + x) / 2 + (i ? 11 : -11), y - levelH / 2, i ? 'yes' : 'no',
                                   { size: 7.5, fill: C.edge });
        }
        if (leaf) out += rect(x - 13, y - 8, 26, 16, { fill: C.out, op: .75, r: 3 });
        else      out += circ(x, y, 9, { fill: C.hid, op: .8 });
        if (!leaf && s.tests && s.tests[d]) out += text(x, y - 14, s.tests[d], { size: 7.5, fill: C.edge });
      }
    }
    out += text(W / 2, H - 6, s.caption || 'each split asks one question about one feature',
                { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* message-passing graph */
  R.graph = function (s) {
    var nodes = s.nodes || [
      [.20, .30], [.42, .18], [.63, .32], [.30, .60], [.52, .66], [.76, .58], [.86, .28]
    ];
    var edges = s.edges || [[0,1],[1,2],[0,3],[1,4],[3,4],[2,5],[4,5],[2,6],[5,6]];
    var out = '', px = 46, py = 26, iw = W - px * 2, ih = H - py - 48;
    var P = nodes.map(function (p) { return { x: px + p[0] * iw, y: py + p[1] * ih }; });

    edges.forEach(function (e) {
      out += line(P[e[0]].x, P[e[0]].y, P[e[1]].x, P[e[1]].y, { sw: 1.1, op: .55 });
    });
    P.forEach(function (p, i) {
      var hl = s.highlight != null && (i === s.highlight || edges.some(function (e) {
        return (e[0] === s.highlight && e[1] === i) || (e[1] === s.highlight && e[0] === i);
      }));
      out += circ(p.x, p.y, i === s.highlight ? 12 : 10, {
        fill: i === s.highlight ? C.attn : hl ? C.hid : C.node, op: hl || i === s.highlight ? .9 : .6
      });
    });
    out += text(W / 2, H - 12, s.caption || 'each node updates itself from its neighbours',
                { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* 2-D points, optionally with a separating boundary */
  R.scatter = function (s) {
    var out = '', px = 44, py = 22, iw = W - px * 2, ih = H - py - 42;
    out += rect(px, py, iw, ih, { fill: 'none', stroke: C.line, r: 4, op: .8 });

    var A = s.a || [[.14,.28],[.22,.15],[.30,.34],[.18,.48],[.34,.20],[.26,.55],[.40,.42],[.12,.66]];
    var B = s.b || [[.72,.70],[.80,.55],[.66,.84],[.86,.78],[.74,.46],[.60,.70],[.90,.62],[.82,.90]];

    if (s.boundary === 'line')
      out += line(px + iw * .06, py + ih * .96, px + iw * .96, py + ih * .06,
                  { stroke: C.attn, sw: 1.8 });
    if (s.boundary === 'margin') {
      out += line(px + iw * .06, py + ih * .96, px + iw * .96, py + ih * .06, { stroke: C.attn, sw: 1.8 });
      out += line(px + iw * .06, py + ih * .74, px + iw * .74, py + ih * .06, { stroke: C.attn, sw: .9, dash: '3 3' });
      out += line(px + iw * .26, py + ih * .96, px + iw * .96, py + ih * .28, { stroke: C.attn, sw: .9, dash: '3 3' });
    }
    if (s.boundary === 'curve')
      out += path('M' + (px + iw * .05) + ' ' + (py + ih * .92) +
                  ' Q' + (px + iw * .5) + ' ' + (py - ih * .18) + ' ' +
                  (px + iw * .95) + ' ' + (py + ih * .55), { stroke: C.attn, sw: 1.8 });
    if (s.boundary === 'step') {
      out += path('M' + (px + iw * .5) + ' ' + py + ' V' + (py + ih * .55) +
                  ' H' + (px + iw) , { stroke: C.attn, sw: 1.6 });
    }
    if (s.centroids) s.centroids.forEach(function (c) {
      out += '<path d="M' + (px + c[0] * iw - 6) + ' ' + (py + c[1] * ih) + ' h12 M' +
             (px + c[0] * iw) + ' ' + (py + c[1] * ih - 6) + ' v12" stroke="' + C.attn +
             '" stroke-width="2.2"/>';
    });

    A.forEach(function (p) { out += circ(px + p[0] * iw, py + p[1] * ih, 4.2, { fill: C.in, op: .85, sw: 0 }); });
    B.forEach(function (p) { out += circ(px + p[0] * iw, py + p[1] * ih, 4.2, { fill: C.out, op: .85, sw: 0 }); });
    if (s.c) s.c.forEach(function (p) { out += circ(px + p[0] * iw, py + p[1] * ih, 4.2, { fill: C.hid, op: .85, sw: 0 }); });

    out += text(W / 2, H - 12, s.caption || '', { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* attention-style heatmap */
  R.matrix = function (s) {
    var n = s.n || 7, out = '', cell = 17, gap = 1.5;
    var total = n * (cell + gap), x0 = W / 2 - total / 2, y0 = 30;
    var seed = 7;
    function rnd() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }

    for (var r = 0; r < n; r++) for (var c = 0; c < n; c++) {
      var v;
      if (s.pattern === 'causal') v = c > r ? 0 : .18 + rnd() * .8;
      else if (s.pattern === 'diagonal') v = Math.max(.06, 1 - Math.abs(r - c) / 2.2) * (.55 + rnd() * .45);
      else v = .12 + rnd() * .85;
      out += rect(x0 + c * (cell + gap), y0 + r * (cell + gap), cell, cell,
                  { fill: C.attn, op: v * .9, r: 1.5, sw: 0 });
    }
    out += text(W / 2, y0 - 10, s.top || 'attends to →', { size: 8.5, fill: C.edge });
    out += text(x0 - 10, y0 + total / 2, s.side || 'query', { size: 8.5, fill: C.edge, anchor: 'end' });
    out += text(W / 2, H - 8, s.caption || '', { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* agent ↔ environment loop */
  R.cycle = function (s) {
    var out = '', bw = 122, bh = 50;
    var lx = 44, rx = W - 44 - bw, y = H / 2 - bh / 2 - 6;
    out += box(lx, y, bw, bh, s.left || 'Agent', { fill: C.hid, op: .8, r: 6, size: 11, weight: 600 });
    out += box(rx, y, bw, bh, s.right || 'Environment', { fill: C.in, op: .55, r: 6, size: 11, weight: 600 });

    out += path('M' + (lx + bw) + ' ' + (y + 14) + ' C' + (W / 2) + ' ' + (y - 22) + ', ' +
                (W / 2) + ' ' + (y - 22) + ', ' + rx + ' ' + (y + 14), { arrow: 1, sw: 1.3, stroke: C.attn });
    out += text(W / 2, y - 14, s.forward || 'action', { size: 9, fill: C.attn, weight: 600 });

    out += path('M' + rx + ' ' + (y + bh - 14) + ' C' + (W / 2) + ' ' + (y + bh + 24) + ', ' +
                (W / 2) + ' ' + (y + bh + 24) + ', ' + (lx + bw) + ' ' + (y + bh - 14),
                { arrow: 1, sw: 1.3, stroke: C.out });
    out += text(W / 2, y + bh + 30, s.back || 'state + reward', { size: 9, fill: C.out, weight: 600 });
    return out;
  };

  /* generator / discriminator */
  R.adversarial = function (s) {
    var out = '', bw = 92, bh = 40;
    out += box(20, 26, bw, bh, s.gen || 'Generator', { fill: C.hid, op: .8, r: 5, size: 9, weight: 600 });
    out += text(66, 20, 'noise z', { size: 8, fill: C.edge });
    out += box(158, 26, 74, bh, 'fake\nsample', { fill: C.node, op: .7, r: 5, size: 8.5 });
    out += box(158, 116, 74, bh, 'real\nsample', { fill: C.in, op: .6, r: 5, size: 8.5 });
    out += box(300, 71, bw, bh, s.disc || 'Discriminator', { fill: C.out, op: .8, r: 5, size: 8.5, weight: 600 });

    out += line(112, 46, 155, 46, { arrow: 1, sw: 1.1 });
    out += line(234, 46, 297, 78, { arrow: 1, sw: 1.1 });
    out += line(234, 136, 297, 104, { arrow: 1, sw: 1.1 });
    out += line(392, 91, 416, 91, { arrow: 1, sw: 1.1 });
    out += text(416, 84, 'real?', { size: 8.5, fill: C.attn, anchor: 'end' });

    out += path('M346 111 C346 178, 66 186, 66 70', { arrow: 1, sw: 1.1, stroke: C.attn, dash: '4 3' });
    out += text(W / 2, H - 6, 'the loss the discriminator produces is what trains the generator',
                { size: 8.2, fill: C.attn, style: 'italic' });
    return out;
  };

  /* wide → narrow → wide bottleneck */
  R.hourglass = function (s) {
    var labels = s.labels || ['input', 'encode', 'code', 'decode', 'output'];
    var heights = s.heights || [110, 70, 30, 70, 110];
    var out = '', n = heights.length, bw = 44;
    var gap = (W - 56 - n * bw) / (n - 1), x0 = 28;

    for (var i = 0; i < n; i++) {
      var x = x0 + i * (bw + gap), h = heights[i], y = H / 2 - h / 2 - 10;
      var mid = i === Math.floor(n / 2);
      out += rect(x, y, bw, h, {
        fill: mid ? C.attn : i === 0 ? C.in : i === n - 1 ? C.out : C.hid,
        op: mid ? .85 : .6, r: 4
      });
      out += text(x + bw / 2, H - 22, labels[i] || '', { size: 8.2, fill: mid ? C.attn : C.text });
      if (i < n - 1) out += line(x + bw + 2, H / 2 - 10, x + bw + gap - 3, H / 2 - 10, { arrow: 1, sw: 1 });
    }
    if (s.caption) out += text(W / 2, H - 7, s.caption, { size: 8.2, fill: C.edge, style: 'italic' });
    return out;
  };

  /* strip of tokens */
  R.sequence = function (s) {
    var toks = s.tokens || ['The', 'cat', 'sat', 'on', 'the', '?'];
    var out = '', n = toks.length, gap = 8;
    var bw = Math.min(58, (W - 56 - gap * (n - 1)) / n), bh = 30;
    var total = n * bw + (n - 1) * gap, x0 = W / 2 - total / 2, y = 58;

    toks.forEach(function (t, i) {
      var x = x0 + i * (bw + gap);
      var last = i === n - 1 && s.predictLast !== false;
      out += box(x, y, bw, bh, t, {
        fill: last ? C.attn : C.in, op: last ? .85 : .5, r: 4, size: 9,
        stroke: last ? C.attn : C.line
      });
      if (s.arrows !== false && i < n - 1)
        out += path('M' + (x + bw / 2) + ' ' + (y - 5) + ' Q' + (x + bw / 2 + (bw + gap) / 2) + ' ' +
                    (y - 30) + ' ' + (x + bw + gap + bw / 2) + ' ' + (y - 5),
                    { arrow: 1, sw: .9, op: .6 });
    });
    if (s.below) out += text(W / 2, y + bh + 24, s.below, { size: 9, fill: C.edge });
    if (s.caption) out += text(W / 2, H - 10, s.caption, { size: 8.5, fill: C.edge, style: 'italic' });
    return out;
  };

  /* ---------- public entry ---------- */

  NS.diagram = function (spec) {
    if (!spec || !R[spec.type]) {
      return svgWrap(text(W / 2, H / 2, 'diagram coming soon', { size: 10, fill: C.edge, style: 'italic' }));
    }
    return svgWrap(R[spec.type](spec));
  };

  function svgWrap(inner) {
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" role="img">' +
      '<defs><marker id="wa" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto">' +
      '<path d="M0 1 L7 4 L0 7 z" fill="' + C.edge + '"/></marker></defs>' + inner + '</svg>';
  }

  NS.diagramTypes = Object.keys(R);

})(window.WIAA);
