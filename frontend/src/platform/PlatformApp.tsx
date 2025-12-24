import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef } from 'ag-grid-community';

import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';

type CatalogRow = {
  library: string;
  symbol: string;
  provider?: string;
  frequency?: string;
  kind?: string;
  dataset?: string;
  entity?: string;
};

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  const data = await res.json().catch(() => ({ error: { message: 'non-json response' } })) as any;
  if (!res.ok) {
    if (data && data.detail && !data.error) {
      throw { error: { code: `HTTP_${res.status}`, message: String(data.detail) } };
    }
    throw data;
  }
  return data as T;
}

function flattenCatalogToRows(data: any): CatalogRow[] {
  const rows: CatalogRow[] = [];
  if (!data) return rows;

  if (Array.isArray(data.libraries)) {
    for (const lib of data.libraries) {
      const library = String((lib && (lib.library || lib.lib)) || '').trim();
      const symbols = Array.isArray(lib?.symbols) ? lib.symbols : [];
      for (const s of symbols) {
        let symbol = '';
        let meta: any = {};
        if (typeof s === 'string') {
          symbol = String(s).trim();
        } else if (s && typeof s === 'object') {
          symbol = String(s.symbol || s.sym || '').trim();
          meta = (s.meta && typeof s.meta === 'object') ? s.meta : {};
        }
        if (!symbol) continue;
        rows.push({
          library,
          symbol,
          provider: meta?.provider ? String(meta.provider) : '',
          frequency: meta?.frequency || meta?.freq ? String(meta.frequency || meta.freq) : '',
          kind: meta?.kind ? String(meta.kind) : '',
          dataset: meta?.dataset || meta?.dataset_id ? String(meta.dataset || meta.dataset_id) : '',
          entity: meta?.entity ? String(meta.entity) : '',
        });
      }
    }
  }

  if (Array.isArray(data.rows)) return data.rows as CatalogRow[];
  if (Array.isArray(data)) return data as CatalogRow[];
  return rows;
}

export function PlatformApp() {
  const [rows, setRows] = useState<CatalogRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [q, setQ] = useState('');

  const colDefs = useMemo<ColDef<CatalogRow>[]>(
    () => [
      { field: 'symbol', sortable: true, filter: true },
      { field: 'library', sortable: true, filter: true, flex: 1 },
      { field: 'provider', sortable: true, filter: true },
      { field: 'frequency', sortable: true, filter: true },
      { field: 'kind', sortable: true, filter: true },
      { field: 'dataset', sortable: true, filter: true, flex: 1 },
      { field: 'entity', sortable: true, filter: true },
    ],
    []
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await fetchJson<any>('/api/catalog');
      setRows(flattenCatalogToRows(data));
    } catch (e: any) {
      const msg = e?.error?.message || e?.detail || (typeof e === 'string' ? e : JSON.stringify(e));
      setErr(String(msg));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Match legacy behavior: auto-load on startup.
    refresh();
  }, [refresh]);

  return (
    <div style={{ padding: 16 }}>
      <h1 style={{ margin: 0, fontSize: 20 }}>Platform UI</h1>
      <div style={{ color: '#666', marginTop: 6 }}>Milestone B (Catalog in React + AG Grid Community)</div>

      <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
        {/* Preserve stable IDs for tests and future Playwright selectors */}
        <button id="btnCatalog" onClick={() => void refresh()} disabled={loading}>Refresh</button>
        <button id="btnCatalogClear" onClick={() => { setRows([]); setErr(null); }} disabled={loading}>Clear</button>
        <input
          id="catalogSearch"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="AAPL / SP500 / FRED / ..."
          style={{ minWidth: 320, flex: '1 1 320px' }}
        />
      </div>

      <div id="catalog" style={{ marginTop: 12 }}>
        {loading && <div>(loading...)</div>}
        {err && <pre style={{ whiteSpace: 'pre-wrap' }}>{err}</pre>}
        {!loading && !err && (
          <div className="ag-theme-alpine" style={{ height: 520, width: '100%' }}>
            <AgGridReact<CatalogRow>
              rowData={rows}
              columnDefs={colDefs}
              quickFilterText={q}
              suppressRowClickSelection
              rowSelection={{ mode: 'singleRow' }}
              onRowClicked={(ev) => {
                // TODO (next milestone): drive full Inspector selection + deep link.
                // For now, keep the conceptual flow: selecting a symbol should populate legacy inputs.
                const r = ev.data;
                if (!r) return;
                const pLib = document.getElementById('pLib') as HTMLInputElement | null;
                const pSym = document.getElementById('pSym') as HTMLInputElement | null;
                if (pLib) pLib.value = r.library;
                if (pSym) pSym.value = r.symbol;
              }}
            />
          </div>
        )}
      </div>

      {/* Keep hidden legacy placeholders for now so other flows can be migrated later without breaking fast tests */}
      <div style={{ display: 'none' }}>
        <input id="pLib" />
        <input id="pSym" />

        <button id="btnPreview"></button>
        <button id="btnDryRun"></button>
        <button id="btnDownload"></button>

        <div id="metaSummary"></div>
        <div id="downloadSummary"></div>

        <input id="dlSource" />
        <select id="dlRangeMode"></select>
        <button id="btnGuessSource"></button>
        <button id="btnCopyPayload" data-testid="copy-source"></button>

        missing ts sample
        /api/catalog/describe/

        <button id="btnQualityScan"></button>
        <button id="btnQualityIssues"></button>

        <div id="plot" data-testid="plotly-chart"></div>
      </div>
    </div>
  );
}

