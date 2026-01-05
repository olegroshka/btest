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

type SourceSpec = {
  id: string;
  scheme: string;
  label: string;
  file_based?: boolean;
  supports_frequency?: boolean;
  examples?: string[];
};

type SourcesResponse = {
  sources?: SourceSpec[];
};

function basenameOfPath(p: string): string {
  const s = String(p || '').trim().replace(/\\/g, '/');
  const parts = s.split('/').filter(Boolean);
  return parts.length ? parts[parts.length - 1] : s;
}

function stripScheme(source: string, scheme: string): string {
  const s = String(source || '');
  const sc = String(scheme || '');
  if (!sc) return s;
  if (s.toLowerCase().startsWith(sc.toLowerCase())) return s.slice(sc.length);
  return s;
}

function detectInitialSourceId(source: string): string {
  const s = String(source || '').toLowerCase();
  if (s.startsWith('parquet://')) return 'parquet';
  if (s.startsWith('yf://')) return 'yf';
  if (s.startsWith('fred://')) return 'fred';
  if (s.includes('://')) return s.split('://', 1)[0];
  return 'parquet';
}

function normalize_entity_input(s: string): string {
  // Keep this permissive: allow tickers/series ids like BRK-B, ^GSPC, CPIAUCSL.
  return String(s || '').trim();
}

export function DownloadPanel({ lib, sym, start, end }: { lib: string; sym: string; start: string; end: string }) {
  const [rangeMode, setRangeMode] = React.useState<'meta' | 'custom'>('meta');
  const [summary, setSummary] = React.useState<string>('');
  const [busy, setBusy] = React.useState(false);

  const [sources, setSources] = React.useState<SourceSpec[]>([]);
  const [sourceId, setSourceId] = React.useState<string>('parquet');
  const [filePath, setFilePath] = React.useState<string>('');
  const [freq, setFreq] = React.useState<string>('1d');

  // Keep a separate "entity" input for non-file sources. We compose scheme://<entity> behind the scenes.
  const [entity, setEntity] = React.useState<string>('');

  // Custom date range inputs (used for network sources; optional for parquet).
  const [dlStart, setDlStart] = React.useState<string>('');
  const [dlEnd, setDlEnd] = React.useState<string>('');

  // Keep DOM IDs expected by contracts/tests.
  React.useEffect(() => {
    const seed = () => {
      try {
        const raw = localStorage.getItem('quantdsl.platform_ui.state.v1');
        const st = raw ? (JSON.parse(raw) as any) : {};
        const src = String(st?.dlSource || '').trim();
        if (src) {
          setSourceId(detectInitialSourceId(src));

          if (src.toLowerCase().startsWith('parquet://')) {
            setFilePath(stripScheme(src, 'parquet://'));
          } else {
            if (src.toLowerCase().includes('://')) {
              const ent = src.split('://', 2)[1] || '';
              setEntity(ent);
            }
          }
        }
        const f = String(st?.dlFrequency || '').trim();
        if (f) setFreq(f);

        const s = String(st?.dlStart || '').trim();
        const e = String(st?.dlEnd || '').trim();
        if (s) setDlStart(s);
        if (e) setDlEnd(e);
      } catch {}
    };
    seed();
  }, []);

  // Load supported source types
  React.useEffect(() => {
    const load = async () => {
      try {
        const resp = await fetchJson<SourcesResponse>('/api/catalog/sources');
        const ss = Array.isArray(resp?.sources) ? (resp.sources as SourceSpec[]) : [];
        setSources(ss);

        // Ensure chosen sourceId is valid
        if (ss.length) {
          const ok = ss.some((x) => x.id === sourceId);
          if (!ok) setSourceId(ss[0].id);
        }
      } catch {
        // Keep working with defaults
        setSources([
          { id: 'parquet', scheme: 'parquet://', label: 'Parquet (local file)', file_based: true, supports_frequency: true },
          { id: 'yf', scheme: 'yf://', label: 'Yahoo Finance (YF)', file_based: false, supports_frequency: true },
          { id: 'fred', scheme: 'fred://', label: 'FRED', file_based: false, supports_frequency: true },
        ]);
      }
    };
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedSpec = React.useMemo(() => {
    const found = sources.find((s) => s.id === sourceId);
    if (found) return found;
    // fallback
    if (sourceId === 'yf') return { id: 'yf', scheme: 'yf://', label: 'Yahoo Finance (YF)', file_based: false, supports_frequency: true } as SourceSpec;
    if (sourceId === 'fred') return { id: 'fred', scheme: 'fred://', label: 'FRED', file_based: false, supports_frequency: true } as SourceSpec;
    return { id: 'parquet', scheme: 'parquet://', label: 'Parquet (local file)', file_based: true, supports_frequency: true } as SourceSpec;
  }, [sources, sourceId]);

  const composedSource = React.useMemo(() => {
    const scheme = String(selectedSpec?.scheme || '').trim() || '';
    if (selectedSpec?.file_based) {
      const p = filePath.trim();
      return p ? scheme + p : scheme;
    }

    // Non-file sources: entity-only input, we compose scheme://<entity>.
    const e = normalize_entity_input(entity);
    if (!e) return scheme;
    return scheme + e;
  }, [selectedSpec, filePath, entity]);

  React.useEffect(() => {
    const el = document.getElementById('dlRangeMode') as HTMLSelectElement | null;
    if (el) el.value = rangeMode;
  }, [rangeMode]);

  React.useEffect(() => {
    const el = document.getElementById('dlFrequency') as HTMLSelectElement | null;
    if (el) el.value = freq;
  }, [freq]);

  const buildPayload = React.useCallback(
    (dryRun: boolean) => {
      const source = (getVal('dlSource') || composedSource).trim();
      const rm = (getVal('dlRangeMode') || rangeMode).trim() as any;
      const f = (getVal('dlFrequency') || freq).trim() || '1d';

      // Prefer internal date inputs; props are kept only for backwards compatibility.
      const s = (getVal('dlStart') || dlStart || start || '').trim();
      const e = (getVal('dlEnd') || dlEnd || end || '').trim();

      // Parquet uses "selected" entity from catalog (sym). YF/FRED use the typed entity.
      const ent = normalize_entity_input(entity);
      const entity_list = selectedSpec?.file_based ? (sym ? [String(sym)] : []) : (ent ? [ent] : []);

      return {
        source: source,
        kind: 'market_bars',
        frequency: f,
        start: rm === 'custom' ? s : s, // keep explicit if provided; meta mode means provider decides coverage
        end: rm === 'custom' ? e : e,
        dataset_id: null,
        calendar: null,
        tz: null,
        entities: entity_list,
        dry_run: dryRun,
        _ui: { lib },
      };
    },
    [composedSource, rangeMode, dlStart, dlEnd, start, end, sym, lib, freq, entity, selectedSpec]
  );

  const runDownload = React.useCallback(
    async (dryRun: boolean) => {
      setBusy(true);
      setSummary('(loading...)');
      try {
        const payload = buildPayload(dryRun);
        if (!String(payload.source || '').trim()) {
          setSummary('Missing source.');
          return;
        }

        const providerIsNetwork = !selectedSpec?.file_based;
        if (providerIsNetwork) {
          if (!normalize_entity_input(entity)) {
            setSummary('Missing entity (e.g. AAPL or CPIAUCSL).');
            return;
          }
          if (!String(payload.start || '').trim() || !String(payload.end || '').trim()) {
            setSummary('Missing start/end dates for this source (pick a small range).');
            return;
          }
        } else {
          if (!sym) {
            setSummary('Pick a row in Catalog first (no entity selected).');
            return;
          }
        }

        if (!Array.isArray(payload.entities) || payload.entities.length === 0) {
          setSummary('Missing entities.');
          return;
        }

        // Persist UI state
        try {
          patchUiState({ dlSource: payload.source, dlFrequency: payload.frequency, dlStart: payload.start, dlEnd: payload.end });
        } catch {}

        const resp: any = await fetchJson('/api/catalog/download', {
          method: 'POST',
          body: JSON.stringify(payload),
        });

        // Show something useful for both dry-run and execute.
        if (resp && resp.dry_run) {
          const items = Array.isArray(resp.plan) ? resp.plan : [];
          const lines: string[] = [];
          lines.push(`dry_run: true`);
          if (resp.request) lines.push(`source: ${String(resp.request?.source || '')}`);
          if (Array.isArray(resp.entities)) lines.push(`entities: ${resp.entities.join(', ')}`);
          lines.push(`plan_items: ${items.length}`);
          if (items.length) lines.push(JSON.stringify(items.slice(0, 6), null, 2));
          setSummary(lines.join('\n'));
          return;
        }

        const lines: string[] = [];
        lines.push(`dry_run: false`);
        if (resp && resp.kind !== undefined) lines.push(`kind: ${String(resp.kind ?? '')}`);
        if (resp && resp.source !== undefined) lines.push(`source: ${String(resp.source ?? '')}`);
        if (resp && resp.frequency !== undefined) lines.push(`frequency: ${String(resp.frequency ?? '')}`);
        if (resp && resp.start) lines.push(`start: ${String(resp.start)}`);
        if (resp && resp.end) lines.push(`end: ${String(resp.end)}`);
        if (Array.isArray(resp.entities) && resp.entities.length) lines.push(`entities: ${resp.entities.join(', ')}`);
        if (resp && resp.cache_stats) lines.push(`cache_stats: ${JSON.stringify(resp.cache_stats)}`);
        if (resp && resp.actions_by_entity) lines.push(`actions_by_entity: ${JSON.stringify(resp.actions_by_entity)}`);
        setSummary(lines.join('\n'));
      } catch (e: any) {
        setSummary(JSON.stringify(e, null, 2));
      } finally {
        setBusy(false);
      }
    },
    [buildPayload, selectedSpec, sym, entity]
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

  // Default range mode: parquet can use meta, network sources require explicit dates.
  React.useEffect(() => {
    try {
      const isNetwork = !selectedSpec?.file_based;
      if (isNetwork && rangeMode !== 'custom') setRangeMode('custom');
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSpec?.file_based]);

  const providerIsNetwork = !selectedSpec?.file_based;

  const canRun = React.useMemo(() => {
    if (busy) return false;
    if (!selectedSpec) return false;

    // Source required
    if (selectedSpec.file_based) {
      if (!filePath.trim()) return false;
    } else {
      if (!normalize_entity_input(entity)) return false;
      if (!dlStart.trim() || !dlEnd.trim()) return false;
    }

    // Also ensure frequency is set
    if (!String(freq || '').trim()) return false;

    // Date sanity (only when both present)
    if (providerIsNetwork && dlStart && dlEnd && dlStart > dlEnd) return false;

    return true;
  }, [busy, selectedSpec, filePath, entity, dlStart, dlEnd, freq, providerIsNetwork]);

  return (
    <div id="downloadPanel" style={{ marginTop: 0 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(420px, 520px) 1fr', gap: 12, alignItems: 'start' }}>
        {/* Left: inputs */}
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 8, alignItems: 'end' }}>
            <label className="label">
              source
              <select
                id="dlSourceType"
                className="input"
                value={sourceId}
                onChange={(e) => {
                  const next = String(e.target.value || '');
                  setSourceId(next);
                  if (next !== 'parquet') setFilePath('');
                  if (next === 'parquet') setEntity('');
                }}
              >
                {(sources.length
                  ? sources
                  : [
                      { id: 'parquet', scheme: 'parquet://', label: 'Parquet (local file)', file_based: true, supports_frequency: true },
                      { id: 'yf', scheme: 'yf://', label: 'Yahoo Finance (YF)', file_based: false, supports_frequency: true },
                      { id: 'fred', scheme: 'fred://', label: 'FRED', file_based: false, supports_frequency: true },
                    ]
                ).map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>

            {selectedSpec?.file_based ? (
              <label className="label">
                file
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    id="dlFile"
                    className="input"
                    style={{ width: '100%' }}
                    placeholder="equities/sp500_daily"
                    value={filePath}
                    onChange={(e) => {
                      const v = e.target.value;
                      setFilePath(v);
                      patchUiState({ dlFile: v });
                    }}
                  />
                  <input
                    id="dlFilePicker"
                    type="file"
                    style={{ width: 220 }}
                    onChange={(e) => {
                      const f = (e.target as HTMLInputElement).files?.[0];
                      if (!f) return;
                      const name = basenameOfPath((f as any).name || '');
                      setFilePath(name);
                      patchUiState({ dlFile: name });
                    }}
                  />
                </div>
              </label>
            ) : (
              <label className="label">
                entity
                <input
                  id="dlSourceText"
                  className="input"
                  placeholder={(() => {
                    const ex = selectedSpec?.examples?.[0] || '';
                    if (ex.includes('://')) return ex.split('://', 2)[1] || '';
                    return 'AAPL';
                  })()}
                  style={{ width: '100%' }}
                  value={entity}
                  onChange={(e) => setEntity(e.target.value)}
                  onBlur={() => patchUiState({ dlSource: composedSource })}
                />
              </label>
            )}
          </div>

          {/* Hidden legacy input used by Playwright and back-compat code paths */}
          <input id="dlSource" className="input" style={{ display: 'none' }} readOnly value={composedSource} />

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, alignItems: 'end' }}>
            <label className="label">
              range
              <select
                id="dlRangeMode"
                className="input"
                style={{ width: '100%' }}
                value={rangeMode}
                onChange={(e) => setRangeMode(e.target.value as any)}
                disabled={!selectedSpec?.file_based}
                title={!selectedSpec?.file_based ? 'Network sources always require explicit start/end.' : ''}
              >
                <option value="meta">meta</option>
                <option value="custom">custom</option>
              </select>
            </label>

            <label className="label">
              frequency
              <select
                id="dlFrequency"
                className="input"
                style={{ width: '100%' }}
                value={freq}
                onChange={(e) => {
                  const next = String(e.target.value || '1d');
                  setFreq(next);
                  patchUiState({ dlFrequency: next });
                }}
              >
                <option value="1d">1d</option>
                <option value="1h">1h</option>
                <option value="1m">1m</option>
              </select>
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8, alignItems: 'end' }}>
            <label className="label">
              start
              <input
                id="dlStart"
                className="input"
                type="date"
                value={dlStart}
                onChange={(e) => {
                  const v = e.target.value;
                  setDlStart(v);
                  patchUiState({ dlStart: v });
                }}
                style={{ fontSize: 14, paddingRight: 10 }}
              />
            </label>

            <label className="label">
              end
              <input
                id="dlEnd"
                className="input"
                type="date"
                value={dlEnd}
                onChange={(e) => {
                  const v = e.target.value;
                  setDlEnd(v);
                  patchUiState({ dlEnd: v });
                }}
                style={{ fontSize: 14, paddingRight: 10 }}
              />
            </label>
          </div>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
            <button id="btnDryRun" className="btn" onClick={() => void runDownload(true)} disabled={!canRun}>
              Dry-run
            </button>
            <button id="btnDownload" className="btn" onClick={() => void runDownload(false)} disabled={!canRun}>
              Download
            </button>
            <button id="btnCopyPayload" className="btn" data-testid="copy-source" onClick={() => void onCopyPayload()} disabled={busy}>
              Copy payload
            </button>
            {!canRun && !busy && (
              <div style={{ color: 'var(--muted)', fontSize: 12, alignSelf: 'center' }}>
                {providerIsNetwork ? 'Enter entity + start/end dates.' : 'Enter file path.'}
              </div>
            )}
          </div>
        </div>

        {/* Right: output */}
        <div className="card" style={{ padding: 10, minHeight: 220 }}>
          <div style={{ fontWeight: 650, marginBottom: 6 }}>Output</div>
          <div id="downloadSummary" style={{ color: 'var(--muted)', whiteSpace: 'pre-wrap' }}>
            {summary}
          </div>
        </div>
      </div>
    </div>
  );
}
