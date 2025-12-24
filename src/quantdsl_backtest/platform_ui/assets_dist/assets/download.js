import { fetchJson, escapeHtml, renderJsonError } from './api.js';
import { patchUiState } from './state.js';

function getVal(id) {
  const el = document.getElementById(id);
  return el && 'value' in el ? String(el.value || '').trim() : '';
}

function setVal(id, v) {
  const el = document.getElementById(id);
  if (el && 'value' in el) el.value = v;
}

export function mountDownload(hostId = 'downloadPanel') {
  const host = document.getElementById(hostId);
  if (!host) return;

  host.innerHTML = `
    <div style="margin-top:14px;border-top:1px solid var(--border);padding-top:12px">
      <div style="font-weight:650;margin-bottom:10px">Download</div>

      <div style="display:grid;grid-template-columns: 1fr 120px; gap:8px; align-items:end">
        <label class="label">source
          <input id="dlSource" class="input" placeholder="parquet://... or fred://..." style="width:100%" />
        </label>
        <button id="btnGuessSource" class="btn">Guess</button>
      </div>

      <div style="display:grid;grid-template-columns: 1fr 1fr; gap:8px; margin-top:8px; align-items:end">
        <label class="label">range
          <select id="dlRangeMode" class="input" style="width:100%">
            <option value="meta">meta</option>
            <option value="custom">custom</option>
          </select>
        </label>
        <div style="display:flex;gap:8px;align-items:end;justify-content:flex-end">
          <button id="btnCopyPayload" class="btn" data-testid="copy-source">Copy payload</button>
        </div>
      </div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
        <button id="btnDryRun" class="btn">Dry-run</button>
        <button id="btnDownload" class="btn">Download</button>
      </div>

      <div id="downloadSummary" style="margin-top:10px;color:var(--muted)"></div>
    </div>
  `;

  // Seed from state if available
  try {
    const src = getVal('dlSource');
    if (src) patchUiState({ dlSource: src });
  } catch (e) {}

  function buildPayload(dryRun) {
    const lib = getVal('pLib');
    const sym = getVal('pSym');
    const start = getVal('dStart');
    const end = getVal('dEnd');

    const source = getVal('dlSource');
    const rangeMode = getVal('dlRangeMode') || 'meta';

    // Minimal stable payload (matches backend model)
    const payload = {
      source: source,
      kind: 'market_bars',
      frequency: '1d',
      start: rangeMode === 'custom' ? start : '',
      end: rangeMode === 'custom' ? end : '',
      dataset_id: null,
      calendar: null,
      tz: null,
      entities: sym ? [String(sym)] : [],
      dry_run: !!dryRun,
      // NOTE: `lib` is NOT part of the download request schema; it is used for inspector.
      _ui: { lib },
    };

    return payload;
  }

  async function copyText(text) {
    try {
      if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (e) {}
    return false;
  }

  async function guessSource() {
    const lib = getVal('pLib');
    const sym = getVal('pSym');
    if (!lib || !sym) return;

    try {
      const meta = await fetchJson(`/api/catalog/meta?library=${encodeURIComponent(lib)}&symbol=${encodeURIComponent(sym)}`);
      const results = (meta && (meta.rows || meta.results)) || [];
      const first = results && results.length ? results[0] : null;
      if (!first) return;

      const provider = String(first.provider || '').trim();
      const dataset = String(first.dataset || first.dataset_id || '').trim();
      let src = provider ? provider.toLowerCase() : '';
      if (src === 'parquet') src = 'parquet';
      if (src === 'fred') src = 'fred';
      if (!src) src = 'parquet';

      let hint = src;
      if (dataset) hint += `://${dataset}`;

      setVal('dlSource', hint);
      patchUiState({ dlSource: hint });
    } catch (e) {
      // non-fatal
    }
  }

  async function runDownload(dryRun) {
    const out = document.getElementById('downloadSummary');
    out.innerHTML = '<div style="color:#666">(loading...)</div>';

    try {
      const payload = buildPayload(dryRun);
      const resp = await fetchJson('/api/catalog/download', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      let html = '';
      if (resp && resp.rows !== undefined) html += `<span style="margin-right:10px">rows: <b>${escapeHtml(String(resp.rows))}</b></span>`;
      if (resp && resp.size_mb !== undefined) html += `<span style="margin-right:10px">size_mb: <b>${escapeHtml(String(resp.size_mb))}</b></span>`;
      if (resp && resp.estimated_time_s !== undefined) html += `<span style="margin-right:10px">est_s: <b>${escapeHtml(String(resp.estimated_time_s))}</b></span>`;

      out.innerHTML = html || '<div style="color:#666">(done)</div>';
    } catch (e) {
      out.innerHTML = renderJsonError(e);
    }
  }

  // wire
  try { document.getElementById('btnGuessSource').onclick = () => guessSource(); } catch (e) {}
  try { document.getElementById('btnDryRun').onclick = () => runDownload(true); } catch (e) {}
  try { document.getElementById('btnDownload').onclick = () => runDownload(false); } catch (e) {}

  try {
    document.getElementById('btnCopyPayload').onclick = async () => {
      const payload = buildPayload(false);
      const ok = await copyText(JSON.stringify(payload, null, 2));
      if (ok) {
        const btn = document.getElementById('btnCopyPayload');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 900);
      }
    };
  } catch (e) {}
}

