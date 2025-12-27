import React from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef } from 'ag-grid-community';

type MetaRow = Record<string, unknown>;

type MetaResponse = {
  rows?: MetaRow[];
  count?: number;
};

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  const data = (await res.json().catch(() => ({ error: { message: 'non-json response' } }))) as any;
  if (!res.ok) {
    if (data && data.detail && !data.error) throw { error: { code: `HTTP_${res.status}`, message: String(data.detail) } };
    throw data;
  }
  return data as T;
}

function useSelection() {
  const [sel, setSel] = React.useState<{ lib: string; sym: string }>({ lib: '', sym: '' });

  React.useEffect(() => {
    const read = () => {
      const pLib = document.getElementById('pLib') as HTMLInputElement | null;
      const pSym = document.getElementById('pSym') as HTMLInputElement | null;
      setSel({ lib: (pLib?.value || '').trim(), sym: (pSym?.value || '').trim() });
    };
    read();

    const onSel = (ev: any) => {
      try {
        const d = ev?.detail || {};
        setSel({ lib: String(d.lib || '').trim(), sym: String(d.sym || '').trim() });
      } catch {
        read();
      }
    };

    window.addEventListener('quantdsl:selection', onSel as any);
    return () => window.removeEventListener('quantdsl:selection', onSel as any);
  }, []);

  return sel;
}

function MetaGrid({ rows }: { rows: MetaRow[] }) {
  const columnDefs = React.useMemo<Array<ColDef<MetaRow>>>(() => {
    const first = rows && rows.length ? rows[0] : {};
    const cols = Object.keys(first || {});
    return cols.map((k) => ({ field: k, filter: 'agTextColumnFilter', sortable: true, resizable: true } as ColDef<MetaRow>));
  }, [rows]);

  return (
    <div className="ag-theme-quant" style={{ height: 520, width: '100%' }}>
      <AgGridReact<MetaRow>
        rowData={rows}
        columnDefs={columnDefs}
        defaultColDef={{ filter: true, sortable: true, resizable: true, minWidth: 120 }}
        pagination
        paginationPageSize={50}
        paginationPageSizeSelector={[20, 50, 100, 200]}
      />
    </div>
  );
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

export function MetaPage() {
  const sel = useSelection();

  // seed from localStorage (v1 contract)
  const seed = React.useMemo(() => {
    try {
      const raw = localStorage.getItem('quantdsl.platform_ui.state.v1');
      return raw ? (JSON.parse(raw) as any) : {};
    } catch {
      return {} as any;
    }
  }, []);

  const [provider, setProvider] = React.useState(String(seed?.fProvider || seed?.mProvider || ''));
  const [freq, setFreq] = React.useState(String(seed?.mFreq || ''));
  const [dataset, setDataset] = React.useState(String(seed?.mDataset || ''));
  const [kind, setKind] = React.useState(String(seed?.mKind || ''));
  const [entity, setEntity] = React.useState(String(seed?.fEntity || seed?.mEntity || ''));

  const [library, setLibrary] = React.useState('');
  const [symbol, setSymbol] = React.useState('');
  const [limit, setLimit] = React.useState('500');

  const [rows, setRows] = React.useState<MetaRow[]>([]);
  const [count, setCount] = React.useState<number>(0);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);

  // Sync library/symbol with current selection
  React.useEffect(() => {
    if (sel.lib) setLibrary(sel.lib);
    if (sel.sym) setSymbol(sel.sym);
  }, [sel.lib, sel.sym]);

  // Deterministic hydration behavior compatible with legacy meta.js:
  // on tab activation, re-read localStorage state and apply fProvider/fEntity if URL params don't lock them.
  React.useEffect(() => {
    const hydrate = () => {
      try {
        const u = new URL(window.location.href);
        const urlLocksProvider = Boolean((u.searchParams.get('provider') || '').trim());
        const urlLocksEntity = Boolean((u.searchParams.get('entity') || '').trim());

        const raw = localStorage.getItem('quantdsl.platform_ui.state.v1');
        const st = raw ? (JSON.parse(raw) as any) : {};
        const seededProvider = String(st?.fProvider || st?.mProvider || '').trim();
        const seededEntity = String(st?.fEntity || st?.mEntity || '').trim();

        if (!urlLocksProvider && seededProvider && provider.trim() !== seededProvider) setProvider(seededProvider);
        if (!urlLocksEntity && seededEntity && entity.trim() !== seededEntity) setEntity(seededEntity);

        // reflect into DOM for Playwright strict DOM reads
        const mProvider = document.getElementById('mProvider') as HTMLInputElement | null;
        const mEntity = document.getElementById('mEntity') as HTMLInputElement | null;
        if (mProvider && !urlLocksProvider && seededProvider) mProvider.value = seededProvider;
        if (mEntity && !urlLocksEntity && seededEntity) mEntity.value = seededEntity;
      } catch {}
    };

    const onTab = (ev: any) => {
      const t = String(ev?.detail?.tab || '').toLowerCase();
      if (t === 'meta') hydrate();
    };

    window.addEventListener('quantdsl:tab', onTab as any);
    window.addEventListener('focus', hydrate);
    document.addEventListener('visibilitychange', hydrate);

    // external seeding event (legacy patchUiState emits this)
    window.addEventListener('quantdsl:ui_state', hydrate as any);

    // run once shortly after mount
    setTimeout(hydrate, 0);

    return () => {
      window.removeEventListener('quantdsl:tab', onTab as any);
      window.removeEventListener('focus', hydrate);
      document.removeEventListener('visibilitychange', hydrate);
      window.removeEventListener('quantdsl:ui_state', hydrate as any);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const summary = React.useMemo(() => {
    const n = rows.length;
    if (!n) return { n: 0, cols: [] as Array<{ key: string; present: number; missing: number; pct: number }> };
    const keys = ['provider', 'frequency', 'kind', 'dataset', 'entity', 'library', 'symbol'];
    const cols = keys.map((k) => {
      let present = 0;
      let missing = 0;
      for (const r of rows) {
        const v = (r as any)[k];
        const s = v === null || v === undefined ? '' : String(v).trim();
        if (!s || s.toLowerCase() === 'nan') missing += 1;
        else present += 1;
      }
      const pct = n ? (present / n) * 100 : 0;
      return { key: k, present, missing, pct };
    });
    return { n, cols };
  }, [rows]);

  const runQuery = React.useCallback(async () => {
    setLoading(true);
    setErr(null);

    // Persist current filters like legacy meta.js so refresh/hydrate behave predictably.
    try {
      patchUiState({
        mProvider: provider.trim(),
        fProvider: provider.trim(),
        mEntity: entity.trim(),
        fEntity: entity.trim(),
        mFreq: freq.trim(),
        mDataset: dataset.trim(),
        mKind: kind.trim(),
      });
    } catch {}

    const params = new URLSearchParams();
    if (provider.trim()) params.set('provider', provider.trim());
    if (freq.trim()) params.set('frequency', freq.trim());
    if (dataset.trim()) params.set('dataset', dataset.trim());
    if (kind.trim()) params.set('kind', kind.trim());
    if (entity.trim()) params.set('entity', entity.trim());
    if (library.trim()) params.set('library', library.trim());
    if (symbol.trim()) params.set('symbol', symbol.trim());
    if (limit.trim()) params.set('limit', limit.trim());

    try {
      const data = await fetchJson<MetaResponse>('/api/catalog/meta?' + params.toString());
      const r = Array.isArray(data?.rows) ? data.rows : [];
      setRows(r);
      setCount(typeof data?.count === 'number' ? data.count : r.length);
    } catch (e: any) {
      const msg = e?.error?.message || e?.detail || (typeof e === 'string' ? e : JSON.stringify(e));
      setErr(String(msg));
      setRows([]);
      setCount(0);
    } finally {
      setLoading(false);
    }
  }, [provider, freq, dataset, kind, entity, library, symbol, limit]);

  return (
    <div id="pageMeta" className="page" style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'end' }}>
        <label className="label">provider
          <input
            id="mProvider"
            className="input"
            value={provider}
            onChange={(e) => setProvider((e.target as HTMLInputElement).value)}
            placeholder="PARQUET/FRED/YF"
          />
        </label>
        <label className="label">freq<input id="mFreq" className="input" value={freq} onChange={(e) => setFreq(e.target.value)} placeholder="1d" /></label>
        <label className="label">dataset<input id="mDataset" className="input" value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="(optional)" /></label>
        <label className="label">kind<input id="mKind" className="input" value={kind} onChange={(e) => setKind(e.target.value)} placeholder="(optional)" /></label>
        <label className="label">entity
          <input
            id="mEntity"
            className="input"
            value={entity}
            onChange={(e) => setEntity((e.target as HTMLInputElement).value)}
            placeholder="(optional)"
          />
        </label>
        <label className="label">library<input id="mLibrary" className="input" value={library} onChange={(e) => setLibrary(e.target.value)} placeholder="market_data/..." /></label>
        <label className="label">symbol<input id="mSymbol" className="input" value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="market_bars/..." /></label>
        <label className="label">limit<input id="mLimit" className="input" value={limit} onChange={(e) => setLimit(e.target.value)} style={{ maxWidth: 90 }} /></label>
        <button id="btnMetaQuery" className="btn" onClick={() => void runQuery()} disabled={loading}>Query</button>
      </div>

      <div id="metaSummary" style={{ marginTop: 10 }}>
        <span style={{ color: 'var(--muted)' }}>count:</span> <b>{count}</b>
      </div>

      {/* Results */}
      <div style={{ marginTop: 10 }}>
        {err && <pre style={{ whiteSpace: 'pre-wrap' }}>{err}</pre>}
        {loading && <div style={{ color: 'var(--muted)' }}>(loading...)</div>}
        {!loading && !err && rows.length > 0 && <MetaGrid rows={rows} />}
        {!loading && !err && rows.length === 0 && <div style={{ color: 'var(--muted)' }}>(no results)</div>}
      </div>

      {/* Diagnostics */}
      <div id="metaTable" style={{ marginTop: 10 }}>
        {!loading && !err && (
          <div className="card">
            <div style={{ fontWeight: 650, marginBottom: 6 }}>Meta diagnostics</div>
            <table className="table" style={{ width: '100%' }}>
              <thead>
              <tr>
                <th>field</th>
                <th>present</th>
                <th>missing</th>
                <th>coverage</th>
              </tr>
              </thead>
              <tbody>
              {summary.cols.map((c) => (
                <tr key={c.key}>
                  <td>{c.key}</td>
                  <td>{c.present}</td>
                  <td>{c.missing}</td>
                  <td>{c.pct.toFixed(1)}%</td>
                </tr>
              ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

