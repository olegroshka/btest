import { fetchJson, escapeHtml, renderJsonError } from './api.js';
import { patchUiState, readQuery, replaceQuery, getUiState } from './state.js';

function flattenCatalogToRows(data) {
  const rows = [];
  if (!data) return rows;
  if (Array.isArray(data.libraries)) {
    for (const lib of data.libraries) {
      const library = String((lib && (lib.library || lib.lib)) || '').trim();
      const symbols = Array.isArray(lib && lib.symbols) ? lib.symbols : [];
      for (const s of symbols) {
        let symbol = '';
        let meta = {};
        if (typeof s === 'string') symbol = String(s).trim();
        else if (s && typeof s === 'object') {
          symbol = String(s.symbol || s.sym || '').trim();
          meta = (s.meta && typeof s.meta === 'object') ? s.meta : {};
        }
        if (!symbol) continue;
        rows.push({
          library,
          symbol,
          provider: String(meta.provider || ''),
          frequency: String(meta.frequency || meta.freq || ''),
          kind: String(meta.kind || ''),
          dataset: String(meta.dataset || meta.dataset_id || ''),
          entity: String(meta.entity || ''),
        });
      }
    }
    return rows;
  }
  if (Array.isArray(data.rows)) return data.rows;
  if (Array.isArray(data)) return data;
  return rows;
}

function ensureLegacyInputsExist() {
  // Some tests / future inspector code expects these.
  const ids = ['pLib', 'pSym'];
  for (const id of ids) {
    if (document.getElementById(id)) continue;
    const el = document.createElement('input');
    el.id = id;
    el.style.display = 'none';
    document.body.appendChild(el);
  }
}

function setSelection(lib, sym, tab = 'catalog') {
  ensureLegacyInputsExist();
  const pLib = document.getElementById('pLib');
  const pSym = document.getElementById('pSym');
  if (pLib) pLib.value = lib;
  if (pSym) pSym.value = sym;

  patchUiState({ pLib: lib, pSym: sym, lastSelectedLibrary: lib, lastSelectedSymbol: sym });
  replaceQuery({ tab, lib, sym });

  // Allow other tabs (Meta/Inspector/etc.) to react immediately.
  try {
    window.dispatchEvent(new CustomEvent('quantdsl:selection', { detail: { lib, sym } }));
  } catch (e) {}
}

function renderRows(host, rows, q) {
  const query = (q || '').trim().toLowerCase();
  const filtered = rows.filter((r) => {
    if (!query) return true;
    const hay = [r.library, r.symbol, r.provider, r.frequency, r.kind, r.dataset, r.entity].join(' ').toLowerCase();
    return hay.includes(query);
  });

  if (!filtered.length) {
    host.innerHTML = '<div style="color:#666">(no matches)</div>';
    return;
  }

  const view = filtered.slice(0, 500);
  let html = '';
  html += '<div class="card" style="overflow:auto">';
  html += '<table class="table">';
  html += '<thead><tr>';
  for (const h of ['symbol','library','provider','freq','kind','dataset','entity']) {
    html += `<th>${escapeHtml(h)}</th>`;
  }
  html += '</tr></thead><tbody>';
  for (const r of view) {
    html += '<tr>';
    html += `<td><a href="#" data-act="preview" data-lib="${escapeHtml(r.library)}" data-sym="${escapeHtml(r.symbol)}">${escapeHtml(r.symbol)}</a></td>`;
    html += `<td>${escapeHtml(r.library)}</td>`;
    html += `<td>${escapeHtml(r.provider||'')}</td>`;
    html += `<td>${escapeHtml(r.frequency||'')}</td>`;
    html += `<td>${escapeHtml(r.kind||'')}</td>`;
    html += `<td>${escapeHtml(r.dataset||'')}</td>`;
    html += `<td>${escapeHtml(r.entity||'')}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  if (filtered.length > 500) {
    html += `<div style="margin-top:8px;color:#666;font-size:12px">(showing first 500 of ${filtered.length} rows; refine search)</div>`;
  }
  host.innerHTML = html;
}

export function mountCatalog(containerId = 'pageCatalog') {
  const app = document.getElementById(containerId);
  if (!app) return;

  app.innerHTML = `
    <div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <button id="btnCatalog" class="btn">Refresh</button>
        <button id="btnCatalogClear" class="btn">Clear</button>
        <input id="catalogSearch" class="input" placeholder="AAPL / SP500 / FRED / ..." style="min-width:320px;flex:1 1 320px" />
        <a href="#" id="linkMeta" style="color:var(--muted);font-size:12px">Meta →</a>
      </div>

      <div id="catalog" style="margin-top:12px">(not loaded)</div>
    </div>
  `;

  const host = document.getElementById('catalog');
  const search = document.getElementById('catalogSearch');
  const linkMeta = document.getElementById('linkMeta');

  let allRows = [];

  function updateMetaLink() {
    try {
      const q = readQuery();
      if (!linkMeta) return;
      const lib = q && q.lib ? String(q.lib) : '';
      const sym = q && q.sym ? String(q.sym) : '';
      linkMeta.href = lib || sym ? `/?tab=meta&library=${encodeURIComponent(lib)}&symbol=${encodeURIComponent(sym)}` : '/?tab=meta';
    } catch (e) {}
  }

  async function refreshCatalog() {
    host.innerHTML = '<div style="color:var(--muted)">(loading...)</div>';
    try {
      const data = await fetchJson('/api/catalog');
      allRows = flattenCatalogToRows(data);
      renderRows(host, allRows, search && search.value);

      const q = readQuery();
      if (q && q.lib && q.sym) setSelection(String(q.lib), String(q.sym));
      updateMetaLink();
    } catch (e) {
      host.innerHTML = renderJsonError(e);
    }
  }

  try { if (linkMeta) linkMeta.onclick = (ev) => { ev.preventDefault(); const tab = document.getElementById('tabMeta'); if (tab) tab.click(); updateMetaLink(); window.location.href = linkMeta.href; }; } catch (e) {}

  // wire
  try { document.getElementById('btnCatalog').onclick = () => refreshCatalog(); } catch (e) {}
  try { document.getElementById('btnCatalogClear').onclick = () => { allRows = []; host.innerHTML='(not loaded)'; if (search) search.value=''; }; } catch (e) {}
  try {
    if (search) {
      search.addEventListener('input', () => {
        renderRows(host, allRows, search.value);
        try {
          const term = String(search.value || '').trim();
          // Contract: Catalog search should not leave a stale selection that can drive Meta queries.
          // Clear selection and remove lib/sym from the URL.

          const prev = getUiState();
          const prevSuggested = String(prev.__catalogEntitySuggest || '').trim();
          const currentMEntity = String(prev.mEntity || '').trim();
          const shouldSuggestEntity = (currentMEntity === '' || currentMEntity === prevSuggested);

          patchUiState({
            catalogSearch: term,
            pLib: '',
            pSym: '',
            lastSelectedLibrary: '',
            lastSelectedSymbol: '',
            __catalogEntitySuggest: term,
            ...(shouldSuggestEntity ? { mEntity: term } : {}),
          });
          try { replaceQuery({ lib: '', sym: '' }); } catch (e3) {}
          try {
            const pl = document.getElementById('pLib');
            const ps = document.getElementById('pSym');
            if (pl) pl.value = '';
            if (ps) ps.value = '';
          } catch (e2) {}
        } catch (e) {}
        updateMetaLink();
      });
      // Some browsers may not fire input for autofill; this keeps it consistent.
      search.addEventListener('change', () => {
        renderRows(host, allRows, search.value);
        try {
          const term = String(search.value || '').trim();
          const prev = getUiState();
          const prevSuggested = String(prev.__catalogEntitySuggest || '').trim();
          const currentMEntity = String(prev.mEntity || '').trim();
          const shouldSuggestEntity = (currentMEntity === '' || currentMEntity === prevSuggested);

          patchUiState({
            catalogSearch: term,
            pLib: '',
            pSym: '',
            lastSelectedLibrary: '',
            lastSelectedSymbol: '',
            __catalogEntitySuggest: term,
            ...(shouldSuggestEntity ? { mEntity: term } : {}),
          });
          try { replaceQuery({ lib: '', sym: '' }); } catch (e3) {}
          try {
            const pl = document.getElementById('pLib');
            const ps = document.getElementById('pSym');
            if (pl) pl.value = '';
            if (ps) ps.value = '';
          } catch (e2) {}
        } catch (e) {}
        updateMetaLink();
      });
    }
  } catch (e) {}

  // delegated click
  try {
    host.addEventListener('click', (ev) => {
      const a = ev.target && ev.target.closest ? ev.target.closest("a[data-act='preview']") : null;
      if (!a) return;
      ev.preventDefault();
      const lib = a.getAttribute('data-lib') || '';
      const sym = a.getAttribute('data-sym') || '';

      // Contract: clicking a catalog symbol is the primary workflow entrypoint.
      // It should populate Inspector inputs and switch to the Inspector tab.
      setSelection(lib, sym, 'inspector');

      // Prefer the explicit navigation API (avoids disabled buttons / overlays)
      try {
        if (window.workspaceApi && typeof window.workspaceApi.setTab === 'function') {
          window.workspaceApi.setTab('inspector');
          return;
        }
      } catch (e) {}

      // Fallback: click the tab button
      try {
        const tabBtn = document.getElementById('tabInspector') || document.querySelector("#mainTabs [data-tab='inspector']");
        if (tabBtn && tabBtn.click) tabBtn.click();
      } catch (e) {}
    });
  } catch (e) {}

  refreshCatalog();
}
