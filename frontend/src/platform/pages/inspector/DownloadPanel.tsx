import React from 'react';

async function fetchJson<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...(opts || {}),
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...((opts && opts.headers) || {}),
    },
  });
  const data = (await res.json().catch(() => ({ error: { message: 'non-json response' } }))) as any;
  if (!res.ok) {
    if (data && data.detail && !data.error) throw { error: { code: `HTTP_${res.status}`, message: String(data.detail) } };
    throw data;
  }
  return data as T;
}

function getVal(id: string): string {
  const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement | null;
  return el && 'value' in el ? String((el as any).value || '').trim() : '';
}

function patchUiState(patch: Record<string, unknown>) {
  const LS_KEY = 'quantdsl.platform_ui.state.v1';
  try {
    const raw = localStorage.getItem(LS_KEY);
    const st = raw ? (JSON.parse(raw) as any) : {};
    const next = { ...(st || {}), ...(patch || {}) };
    localStorage.setItem(LS_KEY, JSON.stringify(next));
    try {
      window.dispatchEvent(new CustomEvent('quantdsl:ui_state', { detail: { patch: patch || {}, state: next } }));
    } catch {}
  } catch {}
}

async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator && (navigator as any).clipboard && (navigator as any).clipboard.writeText) {
      await (navigator as any).clipboard.writeText(text);
      return true;
    }
  } catch {}
  return false;
}

export function DownloadPanel({ lib, sym, start, end }: { lib: string; sym: string; start: string; end: string }) {
  const [dlSource, setDlSource] = React.useState('');
  const [rangeMode, setRangeMode] = React.useState<'meta' | 'custom'>('meta');
  const [summary, setSummary] = React.useState<string>('');
  const [busy, setBusy] = React.useState(false);

  // Keep DOM IDs expected by contracts/tests.
  React.useEffect(() => {
    const seed = () => {
      try {
        const raw = localStorage.getItem('quantdsl.platform_ui.state.v1');
        const st = raw ? (JSON.parse(raw) as any) : {};
        const src = String(st?.dlSource || '').trim();
        if (src) setDlSource(src);
      } catch {}
    };
    seed();
  }, []);

  React.useEffect(() => {
    const el = document.getElementById('dlSource') as HTMLInputElement | null;
    if (el) el.value = dlSource;
  }, [dlSource]);

  React.useEffect(() => {
    const el = document.getElementById('dlRangeMode') as HTMLSelectElement | null;
    if (el) el.value = rangeMode;
  }, [rangeMode]);

  const buildPayload = React.useCallback(
    (dryRun: boolean) => {
      const source = (getVal('dlSource') || dlSource).trim();
      const rm = (getVal('dlRangeMode') || rangeMode).trim() as any;

      // NOTE: matches legacy download.js payload schema.
      return {
        source: source,
        kind: 'market_bars',
        frequency: '1d',
        start: rm === 'custom' ? (start || '') : '',
        end: rm === 'custom' ? (end || '') : '',
        dataset_id: null,
        calendar: null,
        tz: null,
        entities: sym ? [String(sym)] : [],
        dry_run: !!dryRun,
        _ui: { lib },
      };
    },
    [dlSource, rangeMode, start, end, sym, lib]
  );

  const guessSource = React.useCallback(async () => {
    const lib2 = lib.trim();
    const sym2 = sym.trim();
    if (!lib2 || !sym2) return;

    setBusy(true);
    try {
      const meta: any = await fetchJson(`/api/catalog/meta?library=${encodeURIComponent(lib2)}&symbol=${encodeURIComponent(sym2)}`);
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

      setDlSource(hint);
      patchUiState({ dlSource: hint });
    } catch {
      // non-fatal
    } finally {
      setBusy(false);
    }
  }, [lib, sym]);

  const runDownload = React.useCallback(
    async (dryRun: boolean) => {
      setBusy(true);
      setSummary('(loading...)');
      try {
        const payload = buildPayload(dryRun);
        // Persist source
        try {
          patchUiState({ dlSource: payload.source });
        } catch {}

        const resp: any = await fetchJson('/api/catalog/download', {
          method: 'POST',
          body: JSON.stringify(payload),
        });

        const bits: string[] = [];
        if (resp && resp.rows !== undefined) bits.push(`rows: ${resp.rows}`);
        if (resp && resp.size_mb !== undefined) bits.push(`size_mb: ${resp.size_mb}`);
        if (resp && resp.estimated_time_s !== undefined) bits.push(`est_s: ${resp.estimated_time_s}`);
        setSummary(bits.join('  '));
      } catch (e: any) {
        setSummary(JSON.stringify(e, null, 2));
      } finally {
        setBusy(false);
      }
    },
    [buildPayload]
  );

  const onCopyPayload = React.useCallback(async () => {
    const payload = buildPayload(false);
    const ok = await copyText(JSON.stringify(payload, null, 2));
    if (ok) {
      try {
        const btn = document.getElementById('btnCopyPayload');
        if (btn) {
          const orig = btn.textContent || '';
          btn.textContent = 'Copied!';
          setTimeout(() => {
            try {
              btn.textContent = orig;
            } catch {}
          }, 900);
        }
      } catch {}
    }
  }, [buildPayload]);

  return (
    <div id="downloadPanel" style={{ marginTop: 14, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
      <div style={{ fontWeight: 650, marginBottom: 10 }}>Download</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 120px', gap: 8, alignItems: 'end' }}>
        <label className="label">
          source
          <input
            id="dlSource"
            className="input"
            placeholder="parquet://... or fred://..."
            style={{ width: '100%' }}
            value={dlSource}
            onChange={(e) => setDlSource(e.target.value)}
            onBlur={() => patchUiState({ dlSource: dlSource.trim() })}
          />
        </label>
        <button id="btnGuessSource" className="btn" onClick={() => void guessSource()} disabled={busy}>
          Guess
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, alignItems: 'end' }}>
        <label className="label">
          range
          <select
            id="dlRangeMode"
            className="input"
            style={{ width: '100%' }}
            value={rangeMode}
            onChange={(e) => setRangeMode(e.target.value as any)}
          >
            <option value="meta">meta</option>
            <option value="custom">custom</option>
          </select>
        </label>
        <div style={{ display: 'flex', gap: 8, alignItems: 'end', justifyContent: 'flex-end' }}>
          <button id="btnCopyPayload" className="btn" data-testid="copy-source" onClick={() => void onCopyPayload()} disabled={busy}>
            Copy payload
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
        <button id="btnDryRun" className="btn" onClick={() => void runDownload(true)} disabled={busy}>
          Dry-run
        </button>
        <button id="btnDownload" className="btn" onClick={() => void runDownload(false)} disabled={busy}>
          Download
        </button>
      </div>

      <div id="downloadSummary" style={{ marginTop: 10, color: 'var(--muted)', whiteSpace: 'pre-wrap' }}>
        {summary}
      </div>
    </div>
  );
}

