/* ------------------------------------------------------------------
   What's In An AI — application layer

   Reads WIAA.categories (populated by the files in js/data/), renders
   the card grid, the category nav and the detail panel, and wires up
   search, filtering and the theme toggle.
   ------------------------------------------------------------------ */

(function (NS) {
  'use strict';

  var cats = (NS.categories || []).slice().sort(function (a, b) {
    return (a.order || 0) - (b.order || 0);
  });

  /* flatten, keeping a back-reference to the owning category */
  var all = [];
  cats.forEach(function (c) {
    (c.models || []).forEach(function (m) {
      m._cat = c;
      all.push(m);
    });
  });

  var state = { cat: 'all', q: '' };

  var el = {
    grid:      document.getElementById('grid'),
    nav:       document.getElementById('category-nav'),
    search:    document.getElementById('search'),
    empty:     document.getElementById('empty'),
    intro:     document.getElementById('intro'),
    shown:     document.getElementById('count-shown'),
    total:     document.getElementById('count-total'),
    overlay:   document.getElementById('detail'),
    body:      document.getElementById('detail-body'),
    close:     document.getElementById('detail-close'),
    themeBtn:  document.getElementById('theme-toggle')
  };

  /* ---------- theme ---------- */

  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem('wiaa-theme', t); } catch (e) {}
    var icon = el.themeBtn.querySelector('[data-theme-icon]');
    if (icon) icon.textContent = t === 'dark' ? '☀' : '☾';
  }
  (function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem('wiaa-theme'); } catch (e) {}
    var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(saved || (sysDark ? 'dark' : 'light'));
  })();
  el.themeBtn.addEventListener('click', function () {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    render();   /* re-render so diagrams pick up new CSS vars cleanly */
  });

  /* ---------- filtering ---------- */

  function matches(m) {
    if (state.cat !== 'all' && m._cat.id !== state.cat) return false;
    if (!state.q) return true;
    var hay = [m.name, m.aka || '', m.tagline || '', (m.usedFor || []).join(' '),
               m._cat.name, String(m.year || '')].join(' ').toLowerCase();
    return state.q.split(/\s+/).every(function (t) { return hay.indexOf(t) !== -1; });
  }

  /* ---------- nav ---------- */

  function renderNav() {
    var html = btn('all', 'All models', all.length);
    cats.forEach(function (c) { html += btn(c.id, c.name, (c.models || []).length); });
    el.nav.innerHTML = html;

    Array.prototype.forEach.call(el.nav.querySelectorAll('.cat-btn'), function (b) {
      b.addEventListener('click', function () {
        state.cat = b.dataset.cat;
        render();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    });
  }
  function btn(id, label, n) {
    return '<button class="cat-btn' + (state.cat === id ? ' active' : '') +
      '" data-cat="' + id + '" type="button"><span>' + label + '</span>' +
      '<span class="n">' + n + '</span></button>';
  }

  /* ---------- grid ---------- */

  function render() {
    Array.prototype.forEach.call(el.nav.querySelectorAll('.cat-btn'), function (b) {
      b.classList.toggle('active', b.dataset.cat === state.cat);
    });

    var visible = all.filter(matches);
    var html = '', lastCat = null;

    visible.forEach(function (m) {
      if (m._cat.id !== lastCat) {
        html += '<h3 class="cat-heading">' + escape(m._cat.name) + '</h3>';
        lastCat = m._cat.id;
      }
      html += '<button class="card" type="button" data-id="' + m.id + '">' +
        '<span class="card-thumb">' + NS.diagram(m.diagram) + '</span>' +
        '<span class="card-title"><h3>' + escape(m.name) + '</h3>' +
          (m.year ? '<span class="card-year">' + escape(m.year) + '</span>' : '') + '</span>' +
        '<p>' + inlineBold(m.tagline || '') + '</p></button>';
    });

    el.grid.innerHTML = html;
    el.empty.hidden = visible.length !== 0;
    el.intro.hidden = state.cat !== 'all' || !!state.q;
    el.shown.textContent = visible.length;
    el.total.textContent = all.length;

    Array.prototype.forEach.call(el.grid.querySelectorAll('.card'), function (c) {
      c.addEventListener('click', function () { openDetail(c.dataset.id); });
    });
  }

  /* ---------- detail ---------- */

  function openDetail(id) {
    var m = all.filter(function (x) { return x.id === id; })[0];
    if (!m) return;

    var h = '<p class="d-kicker">' + escape(m._cat.name) + '</p>' +
      '<h2 id="detail-title">' + escape(m.name) + '</h2>' +
      (m.aka ? '<p class="d-aka">also called ' + escape(m.aka) + '</p>' : '') +
      (m.tagline ? '<p class="d-tagline">' + inlineBold(m.tagline) + '</p>' : '') +
      '<div class="d-figure">' + NS.diagram(m.diagram) + '</div>';

    if (m.how && m.how.length) {
      h += '<h4>How it works</h4><ol class="d-steps">';
      m.how.forEach(function (s) { h += '<li>' + inlineBold(s) + '</li>'; });
      h += '</ol>';
    }
    if (m.keyIdea) h += '<h4>The key idea</h4><p class="d-key">' + inlineBold(m.keyIdea) + '</p>';
    if (m.strength || m.limitation) {
      h += '<h4>Trade-off</h4><ul class="d-steps">';
      if (m.strength)   h += '<li><strong>Good at:</strong> ' + inlineBold(m.strength) + '</li>';
      if (m.limitation) h += '<li><strong>Struggles with:</strong> ' + inlineBold(m.limitation) + '</li>';
      h += '</ul>';
    }
    if (m.usedFor && m.usedFor.length) {
      h += '<h4>Typically used for</h4><div class="d-tags">';
      m.usedFor.forEach(function (t) { h += '<span class="d-tag">' + escape(t) + '</span>'; });
      h += '</div>';
    }

    h += '<div class="d-meta">';
    if (m.year)    h += '<div><span>Introduced</span>' + escape(m.year) + '</div>';
    if (m.learns)  h += '<div><span>Learning type</span>' + escape(m.learns) + '</div>';
    if (m.dataFor) h += '<div><span>Data shape</span>' + escape(m.dataFor) + '</div>';
    h += '</div>';

    el.body.innerHTML = h;
    el.overlay.hidden = false;
    document.body.style.overflow = 'hidden';
    el.close.focus();
  }

  function closeDetail() {
    el.overlay.hidden = true;
    el.body.innerHTML = '';
    document.body.style.overflow = '';
  }

  el.close.addEventListener('click', closeDetail);
  el.overlay.addEventListener('click', function (e) {
    if (e.target === el.overlay) closeDetail();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !el.overlay.hidden) closeDetail();
  });

  /* ---------- search ---------- */

  el.search.addEventListener('input', function () {
    state.q = el.search.value.trim().toLowerCase();
    render();
  });

  /* ---------- helpers ---------- */

  function escape(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  /* allows *emphasis* in authored copy without opening an HTML hole */
  function inlineBold(s) {
    return escape(s).replace(/\*([^*]+)\*/g, '<strong>$1</strong>');
  }

  /* ---------- go ---------- */

  renderNav();
  render();

})(window.WIAA = window.WIAA || {});
