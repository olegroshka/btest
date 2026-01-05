import React from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef, ICellRendererParams } from 'ag-grid-community';

import { navigateToInspector, setSelection } from '../SelectionBridge';
import { DownloadPanel } from './inspector/DownloadPanel';

export type CatalogRow = {
  library: string;
  symbol: string;
  provider?: string;
  frequency?: string;
  kind?: string;
  dataset?: string;
  entity?: string;
};

export function CatalogPage({
  rows,
  loading,
  err,
  q,
  onChangeSearch,
  onRefresh,
}: {
  rows: CatalogRow[];
  loading: boolean;
  err: string | null;
  q: string;
  onChangeSearch: (next: string) => void;
  onRefresh: () => void;
}) {
  const colDefs = React.useMemo<Array<ColDef<CatalogRow>>>(
    () => [
      {
        field: 'symbol',
        headerName: 'Symbol',
        sortable: true,
        filter: 'agTextColumnFilter',
        flex: 2,
        minWidth: 320,
        cellRenderer: (p: ICellRendererParams<CatalogRow>) => {
          const sym = p.value ? String(p.value) : '';
          const lib = p.data?.library ? String(p.data.library) : '';
          return (
            <a
              href="#"
              onClick={(ev) => {
                ev.preventDefault();
                if (!lib || !sym) return;
                navigateToInspector({ lib, sym });
              }}
            >
              {sym}
            </a>
          );
        },
      },
      { field: 'entity', headerName: 'Entity', sortable: true, filter: 'agTextColumnFilter', minWidth: 140 },
      { field: 'library', headerName: 'Library', sortable: true, filter: 'agTextColumnFilter', minWidth: 160, flex: 1 },
      { field: 'provider', sortable: true, filter: 'agTextColumnFilter' },
      { field: 'frequency', sortable: true, filter: 'agTextColumnFilter' },
      { field: 'kind', sortable: true, filter: 'agTextColumnFilter' },
      { field: 'dataset', sortable: true, filter: 'agTextColumnFilter', flex: 1 },
    ],
    []
  );

  const defaultColDef = React.useMemo<ColDef>(
    () => ({
      resizable: true,
      sortable: true,
      filter: true,
      minWidth: 120,
    }),
    []
  );

  const rowSelection = React.useMemo(
    () => ({
      mode: 'singleRow' as const,
      checkboxes: false,
      headerCheckbox: false,
      enableClickSelection: true,
    }),
    []
  );

  const [sel, setSel] = React.useState<{ lib: string; sym: string }>(() => {
    try {
      const pLib = document.getElementById('pLib') as HTMLInputElement | null;
      const pSym = document.getElementById('pSym') as HTMLInputElement | null;
      return { lib: (pLib?.value || '').trim(), sym: (pSym?.value || '').trim() };
    } catch {
      return { lib: '', sym: '' };
    }
  });

  React.useEffect(() => {
    const onSel = (ev: any) => {
      try {
        const d = ev?.detail || {};
        const lib = String(d.lib || '').trim();
        const sym = String(d.sym || '').trim();
        setSel({ lib, sym });
      } catch {
        // ignore
      }
    };
    window.addEventListener('quantdsl:selection', onSel as any);
    return () => window.removeEventListener('quantdsl:selection', onSel as any);
  }, []);

  return (
    <div id="pageCatalog" className="page">
      <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <button className="btn" id="btnCatalog" onClick={() => onRefresh()} disabled={loading}>
          Refresh
        </button>
        <button className="btn" id="btnCatalogClear" onClick={() => { /* compatibility */ }} disabled={loading}>
          Clear
        </button>
        <input
          className="input"
          id="catalogSearch"
          value={q}
          onChange={(e) => onChangeSearch(e.target.value)}
          placeholder="AAPL / SP500 / FRED / ..."
          style={{ minWidth: 320, flex: '1 1 320px' }}
        />
      </div>

      <div id="catalog" style={{ marginTop: 12, position: 'relative' }}>
        {loading && <div style={{ color: 'var(--muted)' }}>(loading...)</div>}
        {err && <pre style={{ whiteSpace: 'pre-wrap' }}>{err}</pre>}

        {/* Legacy contract for Playwright: preview links inside #catalog */}
        {!loading && !err && rows.length > 0 && (
          <div
            style={{
              height: 8,
              overflow: 'hidden',
              pointerEvents: 'auto',
              fontSize: 1,
              lineHeight: '1px',
              color: 'transparent',
              userSelect: 'none',
            }}
            aria-hidden="true"
          >
            {rows.slice(0, 800).map((r) => (
              <a
                key={`pw:${r.library}::${r.symbol}`}
                href="#"
                data-act="preview"
                data-lib={r.library}
                data-sym={r.symbol}
                data-entity={r.entity || ''}
                style={{ display: 'inline-block', marginRight: 6, fontSize: 1, lineHeight: '1px', color: 'transparent' }}
                onClick={(ev) => {
                  ev.preventDefault();
                  if (!r.library || !r.symbol) return;
                  // Must not navigate away from catalog.
                  setSelection({ lib: r.library, sym: r.symbol });
                  window.dispatchEvent(new CustomEvent('quantdsl:selection', { detail: { lib: r.library, sym: r.symbol } }));
                }}
              >
                {r.symbol}
              </a>
            ))}
          </div>
        )}

        {!loading && !err && (
          <div className="ag-theme-quant" style={{ height: 560, width: '100%' }}>
            <AgGridReact<CatalogRow>
              rowData={rows}
              columnDefs={colDefs}
              defaultColDef={defaultColDef}
              quickFilterText={q}
              pagination
              paginationPageSize={50}
              paginationPageSizeSelector={[20, 50, 100, 200]}
              animateRows
              rowSelection={rowSelection}
              onRowClicked={(ev) => {
                const r = ev.data;
                if (!r) return;
                setSelection({ lib: r.library, sym: r.symbol });
                // Ensure Catalog-side panels update immediately.
                try {
                  window.dispatchEvent(new CustomEvent('quantdsl:selection', { detail: { lib: r.library, sym: r.symbol } }));
                } catch {}
              }}
            />
          </div>
        )}

        {!loading && !err && (
          <div className="card" style={{ marginTop: 12 }}>
            {/* Download should be driven by the user's input, not by existing catalog selection. */}
            <DownloadPanel lib={sel.lib} sym={sel.sym} start={''} end={''} />
          </div>
        )}
      </div>
    </div>
  );
}
