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
        <button id="btnMetaQuery" class="btn">Query</button>
      </div>
      <div id="metaSummary" style="margin-top:10px"></div>
      <div id="metaTable" style="margin-top:10px"></div>
    </div>
  `;

  const st = getUiState();
  const q = readQuery();

  // If coming from Catalog selection, use that as default.
  const selectedLib = (q.lib || st.pLib || '').toString();
  const selectedSym = (q.sym || st.pSym || '').toString();

  // seed from URL or last state
  set('mProvider', String(q.provider || st.mProvider || ''));
  set('mFreq', String(q.frequency || st.mFreq || ''));
  set('mDataset', String(q.dataset || st.mDataset || ''));
  set('mKind', String(q.kind || st.mKind || ''));
  set('mEntity', String(q.entity || st.mEntity || ''));
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

  // auto-run when arriving with any params OR a current selection
  if (
    (q && Object.keys(q).some((k) => ['provider','frequency','dataset','kind','entity','library','symbol','lib','sym'].includes(k))) ||
    selectedLib || selectedSym
  ) {
    runQuery();
  }
}

