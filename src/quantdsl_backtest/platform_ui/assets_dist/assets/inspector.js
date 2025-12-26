import { fetchJson, escapeHtml, renderJsonError } from './api.js';
import { getUiState, patchUiState, readQuery, replaceQuery } from './state.js';
import { mountWorkspace, renderPreviewTables, renderPlotlyFromChartPayload } from './workspace.js';
import { mountDownload } from './download.js';
import { mountQuality } from './quality.js';

function ensureInput(id) {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('input');
    el.id = id;
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  return el;
}

function getVal(id) {
  const el = document.getElementById(id);
  return el && 'value' in el ? String(el.value || '').trim() : '';
}

function setVal(id, v) {
  const el = document.getElementById(id);
  if (el && 'value' in el) el.value = v;
}

export function mountInspector(containerId = 'pageInspector') {
  const host = document.getElementById(containerId);
  if (!host) return;

  // Ensure legacy inputs exist (used by E2E right now)
  ensureInput('pLib');
  ensureInput('pSym');

  host.innerHTML = `
    <div style="display:grid;grid-template-columns:360px 1fr;gap:12px;align-items:start">
      <div class="card">
        <div style="font-weight:650;margin-bottom:8px">Inspector</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          <label class="label">library
            <input id="pLib" class="input" style="width:100%" />
          </label>
          <label class="label">symbol
            <input id="pSym" class="input" style="width:100%" />
          </label>
          <label class="label">start
            <input id="dStart" class="input" placeholder="YYYY-MM-DD" style="width:100%" />
          </label>
          <label class="label">end
            <input id="dEnd" class="input" placeholder="YYYY-MM-DD" style="width:100%" />
          </label>
        </div>

        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
          <button id="btnPreview" class="btn">Preview</button>
        </div>

        <div id="inspectorMetaSummary" style="margin-top:10px"></div>

        <div id="downloadPanel"></div>
        <div id="qualityPanel"></div>
      </div>

      <div id="workspace" class="card"></div>
    </div>
  `;

  const workspaceApi = mountWorkspace('workspace');
  mountDownload('downloadPanel');
  mountQuality('qualityPanel');

  // Restore state from URL/localStorage
  const qs = readQuery();
  const st = getUiState();

  const lib = (qs.lib || st.pLib || '').toString();
  const sym = (qs.sym || st.pSym || '').toString();
  if (lib) setVal('pLib', lib);
  if (sym) setVal('pSym', sym);

  function persistSelection() {
    const lib2 = getVal('pLib');
    const sym2 = getVal('pSym');
    patchUiState({ pLib: lib2, pSym: sym2 });
    replaceQuery({ tab: 'inspector', lib: lib2, sym: sym2 });
  }

  try {
    document.getElementById('pLib')?.addEventListener('input', persistSelection);
    document.getElementById('pSym')?.addEventListener('input', persistSelection);
  } catch (e) {}

  async function updateMetaSummary(lib3, sym3) {
    try {
      const m = await fetchJson(`/api/catalog/meta?library=${encodeURIComponent(lib3)}&symbol=${encodeURIComponent(sym3)}`);
      const results = (m && (m.rows || m.results)) || [];
      const first = results && results.length ? results[0] : null;
      const ms = document.getElementById('inspectorMetaSummary');
      if (ms && first) {
        const provider = first.provider || '';
        const freq = first.frequency || first.freq || '';
        const kind = first.kind || '';
        ms.innerHTML = [
          provider && `provider: <code>${escapeHtml(provider)}</code>`,
          freq && `freq: <code>${escapeHtml(freq)}</code>`,
          kind && `kind: <code>${escapeHtml(kind)}</code>`,
        ].filter(Boolean).join(' &nbsp; ');
      }
    } catch (e) {
      // non-fatal
    }
  }

  async function runPreview() {
    const lib3 = getVal('pLib');
    const sym3 = getVal('pSym');
    if (!lib3 || !sym3) {
      const tables = document.getElementById('previewTables');
      if (tables) tables.innerHTML = '<div style="color:var(--danger)">Missing library/symbol</div>';
      return;
    }

    persistSelection();

    // Always show plot tab when previewing
    try { if (workspaceApi && workspaceApi.setTab) workspaceApi.setTab('plot'); } catch (e) {}

    // reset UI
    try {
      const stEl = document.getElementById('plotStatus');
      if (stEl) stEl.textContent = '(loading...)';
      const plot = document.getElementById('plot');
      if (plot) {
        plot.textContent = '';
        // Drop any old plotly state that could interfere with re-renders
        try {
          const Plotly = window.Plotly;
          if (Plotly && Plotly.purge) Plotly.purge(plot);
        } catch (e2) {}
      }
      const tables = document.getElementById('previewTables');
      const raw = document.getElementById('previewRaw');
      const summary = document.getElementById('previewSummary');
      if (tables) tables.innerHTML = '<div style="color:var(--muted)">(loading preview...)</div>';
      if (raw) raw.textContent = '';
      if (summary) summary.textContent = '';
    } catch (e) {}

    try {
      const urlPrev = `/api/catalog/preview/${encodeURIComponent(lib3)}?symbol=${encodeURIComponent(sym3)}&head=12&tail=12`;
      const data = await fetchJson(urlPrev);
      renderPreviewTables(data);

      updateMetaSummary(lib3, sym3);

      try {
        const start = getVal('dStart');
        const end = getVal('dEnd');
        const qs = new URLSearchParams({ symbol: sym3, limit: '1500' });
        if (start) qs.set('start', start);
        if (end) qs.set('end', end);
        const urlChart = `/api/catalog/chart/${encodeURIComponent(lib3)}?` + qs.toString();
        const chart = await fetchJson(urlChart);
        renderPlotlyFromChartPayload(chart);
      } catch (e) {
        try {
          const stEl = document.getElementById('plotStatus');
          if (stEl) stEl.textContent = '(no chart)';
        } catch (e2) {}
      }

    } catch (e) {
      const tables = document.getElementById('previewTables');
      const raw = document.getElementById('previewRaw');
      if (tables) tables.innerHTML = renderJsonError(e);
      if (raw) raw.textContent = JSON.stringify(e, null, 2);
      try {
        const stEl = document.getElementById('plotStatus');
        if (stEl) stEl.textContent = '(error)';
        const plot = document.getElementById('plot');
        if (plot) plot.textContent = '(no data)';
      } catch (e2) {}
    }
  }

  try { document.getElementById('btnPreview').onclick = () => runPreview(); } catch (e) {}
}
