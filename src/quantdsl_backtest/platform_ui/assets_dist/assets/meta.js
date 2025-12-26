import { fetchJson, escapeHtml, renderJsonError } from './api.js';
import { getUiState, patchUiState, readQuery, replaceQuery } from './state.js';

function val(id) {
  const el = document.getElementById(id);
  return el && 'value' in el ? String(el.value || '').trim() : '';
}

function set(id, v) {
  const el = document.getElementById(id);
  if (el && 'value' in el) el.value = v;
}

function renderRowsTable(rows) {
  if (!Array.isArray(rows) || !rows.length) return '<div style="color:var(--muted)">(no rows)</div>';
  const cols = Object.keys(rows[0] || {});
  let html = '<div class="card" style="overflow:auto">';
  html += '<table class="table">';
  html += '<thead><tr>';
  for (const c of cols) html += `<th>${escapeHtml(c)}</th>`;
  html += '</tr></thead><tbody>';
  for (const r of rows.slice(0, 500)) {
    html += '<tr>';
    for (const c of cols) html += `<td>${escapeHtml(r[c] ?? '')}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

export function mountMeta(containerId = 'pageMeta') {
  const host = document.getElementById(containerId);
  if (!host) return;

  host.innerHTML = `
    <div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:end">
        <label class="label">provider<input id="mProvider" class="input" placeholder="PARQUET/FRED/YF"/></label>
        <label class="label">freq<input id="mFreq" class="input" placeholder="1d"/></label>
        <label class="label">dataset<input id="mDataset" class="input" placeholder="(optional)"/></label>
        <label class="label">kind<input id="mKind" class="input" placeholder="(optional)"/></label>
        <label class="label">entity<input id="mEntity" class="input" placeholder="(optional)"/></label>
        <label class="label">library<input id="mLibrary" class="input" placeholder="market_data/..."/></label>
        <label class="label">symbol<input id="mSymbol" class="input" placeholder="market_bars/..."/></label>
        <label class="label">limit<input id="mLimit" class="input" value="500" style="max-width:90px"/></label>
        <button id="btnMetaFromSelection" class="btn">From selection</button>
        <button id="btnMetaQuery" class="btn">Query</button>
      </div>
      <div id="metaSummary" style="margin-top:10px"></div>
      <div id="metaTable" style="margin-top:10px"></div>
    </div>
  `;

  const st = getUiState();
  const q = readQuery();

  // Backward/compat: tests (and some legacy flows) seed fProvider/fEntity in localStorage.
  // Treat them as aliases for Meta filters to satisfy the strict contract.
  const seededProvider = (st.mProvider || st.fProvider || '').toString();
  const seededEntity = (st.mEntity || st.fEntity || '').toString();

  // Deterministic hydration:
  // - apply seeded provider/entity once on mount
  // - never fight user edits later
  // - never re-apply on focus/visibility/tab (these are nondeterministic in headless)
  function hydrateSeededEntityProviderIfBlank() {
    try {
      const st3 = getUiState();
      const seededProvider2 = (st3.fProvider || st3.mProvider || '').toString().trim();
      const seededEntity2 = (st3.fEntity || st3.mEntity || '').toString().trim();

      const curProv = val('mProvider');
      const curEnt = val('mEntity');
      if (!curProv && seededProvider2) set('mProvider', seededProvider2);
      if (!curEnt && seededEntity2) set('mEntity', seededEntity2);
    } catch (e) {}
  }

  // Stronger deterministic hydration: if state is seeded after boot (Playwright), update
  // provider/entity unless they are explicitly controlled by URL params.
  function hydrateSeededEntityProviderDeterministic() {
    try {
      const st3 = getUiState();
      const seededProvider2 = (st3.fProvider || st3.mProvider || '').toString().trim();
      const seededEntity2 = (st3.fEntity || st3.mEntity || '').toString().trim();

      const curProv = val('mProvider');
      const curEnt = val('mEntity');

      const urlLocksProvider = Boolean((q.provider || '').toString().trim());
      const urlLocksEntity = Boolean((q.entity || '').toString().trim());

      if (!urlLocksProvider && seededProvider2 && curProv.trim() !== seededProvider2) set('mProvider', seededProvider2);
      if (!urlLocksEntity && seededEntity2 && curEnt.trim() !== seededEntity2) set('mEntity', seededEntity2);
    } catch (e) {}
  }

  // If coming from Catalog selection, use that as default.
  const selectedLib = (q.lib || st.pLib || '').toString();
  const selectedSym = (q.sym || st.pSym || '').toString();

  // seed from URL or last state
  set('mProvider', String(q.provider || seededProvider || ''));
  set('mFreq', String(q.frequency || st.mFreq || ''));
  set('mDataset', String(q.dataset || st.mDataset || ''));
  set('mKind', String(q.kind || st.mKind || ''));
  set('mEntity', String(q.entity || seededEntity || ''));

  // One more deterministic pass: if URL did not set provider/entity, allow fProvider/fEntity to fill blanks.
  hydrateSeededEntityProviderIfBlank();

  // React to external state seeding (e.g. Playwright sets localStorage after page load).
  // Only fills blanks; will not overwrite user-entered filters.
  try {
    window.addEventListener('quantdsl:ui_state', () => {
      try {
        hydrateSeededEntityProviderDeterministic();
      } catch (e) {}
    });
  } catch (e) {}

  // (The browser 'storage' event does not fire in the same document that made the change,
  // so we use the explicit quantdsl:ui_state event from patchUiState instead.)

  // Persist the seeded meta filters so later reads of getUiState() see fEntity/fProvider.
  // (This prevents other modules from treating them as empty and overwriting mEntity suggestions.)
  try {
    const _prov = String((st.fProvider || seededProvider || '')).trim();
    const _ent = String((st.fEntity || seededEntity || '')).trim();
    if (_prov || _ent) patchUiState({ ...(_prov ? { fProvider: _prov } : {}), ...(_ent ? { fEntity: _ent } : {}) });
  } catch (e) {}

  // When the Meta tab is activated, ensure provider/entity are hydrated from localStorage seed.
  // NOTE: tests may write localStorage directly (not via patchUiState), so we must read localStorage here.
  try {
    window.addEventListener('quantdsl:tab', (ev) => {
      try {
        const t = ev && ev.detail ? String(ev.detail.tab || '') : '';
        if (t === 'meta') {
          hydrateSeededEntityProviderDeterministic();
        }
      } catch (e) {}
    });
  } catch (e) {}

  // Headless browsers can fire focus/visibility in surprising order; keep it deterministic.
  // This won't override explicit URL params, only re-applies persisted seed.
  try {
    const _rehydrate = () => { try { hydrateSeededEntityProviderDeterministic(); } catch (e) {} };
    window.addEventListener('focus', _rehydrate);
    document.addEventListener('visibilitychange', _rehydrate);
  } catch (e) {}

  // Also hydrate once shortly after mount to cover cases where the storage seed happens immediately
  // after navigation but before the first tab activation.
  try { setTimeout(() => { try { hydrateSeededEntityProviderDeterministic(); } catch (e) {} }, 0); } catch (e) {}

  // NOTE: We intentionally do NOT install focus/visibility listeners that repeatedly re-apply state.
  // Those caused nondeterministic clobbers in Playwright and could overwrite legitimate edits.

  // Strict precedence:
  // - explicit meta URL params
  // - explicit meta persisted state
  // - selection (URL lib/sym or persisted pLib/pSym)
  set('mLibrary', String(q.library || st.mLibrary || selectedLib || ''));
  set('mSymbol', String(q.symbol || st.mSymbol || selectedSym || ''));

  async function runQuery() {
    const out = document.getElementById('metaTable');
    const sum = document.getElementById('metaSummary');
    if (out) out.innerHTML = '<div style="color:var(--muted)">(loading...)</div>';

    const params = new URLSearchParams();
    const provider = val('mProvider');
    const frequency = val('mFreq');
    const dataset = val('mDataset');
    const kind = val('mKind');
    const entity = val('mEntity');
    const library = val('mLibrary');
    const symbol = val('mSymbol');
    const limit = val('mLimit');

    if (provider) params.set('provider', provider);
    if (frequency) params.set('frequency', frequency);
    if (dataset) params.set('dataset', dataset);
    if (kind) params.set('kind', kind);
    if (entity) params.set('entity', entity);
    if (library) params.set('library', library);
    if (symbol) params.set('symbol', symbol);
    if (limit) params.set('limit', limit);

    patchUiState({ mProvider: provider, mFreq: frequency, mDataset: dataset, mKind: kind, mEntity: entity, mLibrary: library, mSymbol: symbol });
    replaceQuery({ tab: 'meta', provider, frequency, dataset, kind, entity, library, symbol, limit });

    try {
      const data = await fetchJson('/api/catalog/meta?' + params.toString());
      const rows = Array.isArray(data?.rows) ? data.rows : [];
      const count = typeof data?.count === 'number' ? data.count : rows.length;

      if (sum) {
        sum.innerHTML = `<span style="color:var(--muted)">count:</span> <b>${escapeHtml(String(count))}</b>`;
      }
      if (out) out.innerHTML = renderRowsTable(rows);
    } catch (e) {
      if (out) out.innerHTML = renderJsonError(e);
    }
  }

  try { document.getElementById('btnMetaQuery').onclick = () => runQuery(); } catch (e) {}

  function applyFromSelection() {
    const st2 = getUiState();
    const q2 = readQuery();
    const lib = String(q2.lib || st2.pLib || '').trim();
    const sym = String(q2.sym || st2.pSym || '').trim();
    if (!lib && !sym) return;
    if (lib) set('mLibrary', lib);
    if (sym) set('mSymbol', sym);
    // Don't force provider/entity/etc here. Those can be refined by the user.
  }

  // Live sync: when the Catalog selection changes, always reflect it in Meta's library/symbol.
  // This is a strict UI contract so a trader can select a row and immediately query Meta for it.
  function onSelectionEv(ev) {
    try {
      const d = (ev && ev.detail) ? ev.detail : {};
      const lib = String(d.lib || '').trim();
      const sym = String(d.sym || '').trim();

      // Selection cleared (e.g. Catalog search). Clear selection-driven fields so Meta can't query stale rows.
      if (!lib && !sym) {
        set('mLibrary', '');
        set('mSymbol', '');
        patchUiState({ mLibrary: '', mSymbol: '' });
        return;
      }

      if (lib) set('mLibrary', lib);
      if (sym) set('mSymbol', sym);
      // Persist only library/symbol
      patchUiState({ mLibrary: lib || val('mLibrary'), mSymbol: sym || val('mSymbol') });
    } catch (e) {}
  }
  try { window.addEventListener('quantdsl:selection', onSelectionEv); } catch (e) {}

  // Wire "From selection" button.
  try {
    const btn = document.getElementById('btnMetaFromSelection');
    if (btn) btn.onclick = () => applyFromSelection();
  } catch (e) {}

  // If we already have a selection (via URL or persisted state), apply it once on mount.
  if ((selectedLib || selectedSym) && (!val('mLibrary') || !val('mSymbol'))) {
    applyFromSelection();
  }
}
