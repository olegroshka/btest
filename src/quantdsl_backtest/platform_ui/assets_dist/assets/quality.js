import { fetchJson, escapeHtml, renderJsonError } from './api.js';

function getVal(id) {
  const el = document.getElementById(id);
  return el && 'value' in el ? String(el.value || '').trim() : '';
}

export function mountQuality(hostId = 'qualityPanel') {
  const host = document.getElementById(hostId);
  if (!host) return;

  host.innerHTML = `
    <div style="margin-top:14px;border-top:1px solid var(--border);padding-top:12px">
      <div style="font-weight:650;margin-bottom:10px">Quality</div>

      <div style="display:grid;grid-template-columns:repeat(2, minmax(0, 1fr));gap:8px">
        <label class="label">provider <input id="qProvider" class="input" placeholder="PARQUET/FRED/YF" style="width:100%" /></label>
        <label class="label">frequency <input id="qFreq" class="input" placeholder="1d" style="width:100%" /></label>
        <label class="label">dataset <input id="qDataset" class="input" placeholder="(optional)" style="width:100%" /></label>
        <label class="label">kind <input id="qKind" class="input" placeholder="(optional)" style="width:100%" /></label>
        <label class="label">entity <input id="qEntity" class="input" placeholder="(optional)" style="width:100%" /></label>
        <label class="label">limit <input id="qLimit" class="input" value="200" style="max-width:120px" /></label>
      </div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
        <button id="btnQualityScan" class="btn">Scan</button>
        <button id="btnQualityIssues" class="btn">Issues</button>
      </div>

      <div id="quality" style="margin-top:10px"></div>
    </div>
  `;

  function renderGrid(data) {
    const out = document.getElementById('quality');

    // Normalize various response shapes:
    // - {rows: [...]}
    // - {issues: [...]}
    // - {result: {...}} / arbitrary dict -> render as key/value rows
    let rows = [];
    if (Array.isArray(data?.rows)) rows = data.rows;
    else if (Array.isArray(data?.issues)) rows = data.issues;
    else if (Array.isArray(data)) rows = data;

    if (!rows.length && data && typeof data === 'object') {
      rows = Object.keys(data).map((k) => ({ key: k, value: data[k] }));
    }

    if (!rows.length) {
      out.innerHTML = '<div style="color:var(--muted)">(no rows)</div>';
      return;
    }

    const cols = Object.keys(rows[0] || {});
    let html = '<div class="card" style="overflow:auto">';
    html += '<table class="table">';
    html += '<thead><tr>';
    for (const c of cols) html += `<th>${escapeHtml(c)}</th>`;
    html += '</tr></thead><tbody>';
    for (const r of rows.slice(0, 300)) {
      html += '<tr>';
      for (const c of cols) {
        const v = r[c];
        html += `<td>${escapeHtml(typeof v === 'string' ? v : JSON.stringify(v))}</td>`;
      }
      html += '</tr>';
    }
    html += '</tbody></table></div>';

    // keep raw available for debugging (collapsed)
    html += '<details style="margin-top:8px"><summary style="color:var(--muted)">raw</summary>';
    html += '<pre style="white-space:pre-wrap">' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre></details>';

    out.innerHTML = html;
  }

  function buildParams() {
    const params = new URLSearchParams();
    const provider = getVal('qProvider');
    const freq = getVal('qFreq');
    const dataset = getVal('qDataset');
    const kind = getVal('qKind');
    const entity = getVal('qEntity');
    const limit = getVal('qLimit');

    if (provider) params.set('provider', provider);
    if (freq) params.set('frequency', freq);
    if (dataset) params.set('dataset', dataset);
    if (kind) params.set('kind', kind);
    if (entity) params.set('entity', entity);
    if (limit) params.set('limit', limit);

    return params;
  }

  async function runScan() {
    const out = document.getElementById('quality');
    out.innerHTML = '<div style="color:var(--muted)">(loading...)</div>';

    try {
      const params = buildParams();
      const data = await fetchJson('/api/quality/scan?' + params.toString(), { method: 'POST' });
      renderGrid(data);
    } catch (e) {
      out.innerHTML = renderJsonError(e);
    }
  }

  async function loadIssues() {
    const out = document.getElementById('quality');
    out.innerHTML = '<div style="color:var(--muted)">(loading...)</div>';

    try {
      const params = buildParams();
      const data = await fetchJson('/api/quality/issues?' + params.toString());
      const rows = Array.isArray(data?.rows) ? data.rows : [];
      if (!rows.length) {
        out.innerHTML = '<div style="color:var(--muted)">(no issues)</div>';
        return;
      }
      renderGrid({ rows });
    } catch (e) {
      out.innerHTML = renderJsonError(e);
    }
  }

  try { document.getElementById('btnQualityScan').onclick = () => runScan(); } catch (e) {}
  try { document.getElementById('btnQualityIssues').onclick = () => loadIssues(); } catch (e) {}
}
