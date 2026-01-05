import React from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef } from 'ag-grid-community';

import { setSelection } from '../../SelectionBridge';
import { QualityPanel } from './QualityPanel';

type PreviewPayload = {
  rows?: number;
  index_start?: string;
  index_end?: string;
  columns?: string[];
  head?: Array<Record<string, unknown>>;
  tail?: Array<Record<string, unknown>>;
};

type ChartPayload = {
  columns?: string[];
  data?: Array<Record<string, any>>;
};

function readLegacyChartLimit(defaultLimit: number = 1500): number {
  try {
    const el = document.getElementById('pLimit') as HTMLInputElement | null;
    const raw = String(el?.value || '').trim();
    const n = raw ? Number(raw) : NaN;
    if (!Number.isFinite(n)) return defaultLimit;
    return Math.max(50, Math.min(5000, Math.floor(n)));
  } catch {
    return defaultLimit;
  }
}

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

function getOrCreateHiddenInput(id: string): HTMLInputElement {
  let el = document.getElementById(id) as HTMLInputElement | null;
  if (!el) {
    el = document.createElement('input');
    el.id = id;
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  return el;
}

function readQuery(): Record<string, string> {
  const q = new URLSearchParams(window.location.search || '');
  const obj: Record<string, string> = {};
  for (const [k, v] of q.entries()) obj[k] = v;
  return obj;
}

function replaceQuery(patch: Record<string, string | null | undefined>) {
  const cur = new URLSearchParams(window.location.search || '');
  for (const [k, v] of Object.entries(patch || {})) {
    if (v === null || v === undefined || String(v).trim() === '') cur.delete(k);
    else cur.set(k, String(v));
  }
  const next = cur.toString();
  const url = next ? `${window.location.pathname}?${next}` : window.location.pathname;
  try {
    history.replaceState({}, '', url);
  } catch {}
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

function purgePlotly(el: HTMLElement) {
  // Keep as a best-effort cleanup for failures; Plotly.react handles updates.
  try {
    const Plotly = (window as any).Plotly;
    if (Plotly && typeof Plotly.purge === 'function') Plotly.purge(el);
  } catch {}
}

function renderPlotlyFromChartPayload(host: HTMLElement, payload: ChartPayload) {
  const Plotly = (window as any).Plotly;
  if (!Plotly) {
    host.textContent = '(plotly not available)';
    return;
  }

  const cols = Array.isArray(payload?.columns) ? payload.columns : [];
  const data = Array.isArray(payload?.data) ? payload.data : [];
  if (!data.length) {
    host.textContent = '(no data)';
    return;
  }

  // IMPORTANT: `host` must NOT be a React-owned node that React may reconcile.
  // We render into a dedicated inner div and avoid touching the outer React container.
  // Still, keep host clean of placeholder text nodes before Plotly mounts.
  try {
    if (host.childNodes?.length && typeof (host as any).data === 'undefined') {
      host.innerHTML = '';
    }
  } catch {}

  const ts = data.map((r) => (r as any).ts);

  const lowerCols = cols.map((c) => String(c).toLowerCase());
  const idxOpen = lowerCols.indexOf('open');
  const idxHigh = lowerCols.indexOf('high');
  const idxLow = lowerCols.indexOf('low');
  const idxClose = lowerCols.indexOf('close');
  const idxVolume = lowerCols.indexOf('volume');

  const hasOhlc = idxOpen >= 0 && idxHigh >= 0 && idxLow >= 0 && idxClose >= 0;
  const hasVol = idxVolume >= 0;

  let traces: any[] = [];
  if (hasOhlc) {
    const open = data.map((r) => (r as any)[cols[idxOpen]]);
    const high = data.map((r) => (r as any)[cols[idxHigh]]);
    const low = data.map((r) => (r as any)[cols[idxLow]]);
    const close = data.map((r) => (r as any)[cols[idxClose]]);

    const up = '#14f195';
    const down = '#ff3b6b';

    traces = [
      {
        type: 'candlestick',
        x: ts,
        open,
        high,
        low,
        close,
        name: 'OHLC',
        increasing: { line: { color: up, width: 1.2 } },
        decreasing: { line: { color: down, width: 1.2 } },
        hoverlabel: { align: 'left' },
      },
    ];

    if (hasVol) {
      const vol = data.map((r) => (r as any)[cols[idxVolume]]);
      const volColors = close.map((c, i) => (c >= open[i] ? 'rgba(20,241,149,0.35)' : 'rgba(255,59,107,0.35)'));
      traces.push({
        type: 'bar',
        x: ts,
        y: vol,
        name: 'Volume',
        yaxis: 'y2',
        marker: { color: volColors },
      });
    }
  } else {
    const c0 = cols.find((c) => c && String(c).toLowerCase() !== 'ts');
    if (!c0) {
      host.textContent = '(no plottable columns)';
      return;
    }
    const y = data.map((r) => (r as any)[c0]);
    traces = [{ type: 'scatter', mode: 'lines', x: ts, y, name: String(c0), line: { color: '#22d3ee', width: 1.8 } }];

    if (hasVol) {
      const vol = data.map((r) => (r as any)[cols[idxVolume]]);
      traces.push({ type: 'bar', x: ts, y: vol, name: 'Volume', yaxis: 'y2', marker: { color: 'rgba(34,211,238,0.20)' } });
    }
  }

  const rootStyle = getComputedStyle(document.documentElement);
  const bg = rootStyle.getPropertyValue('--bg').trim() || '#0b1020';
  const panel = rootStyle.getPropertyValue('--panel').trim() || '#0f172a';
  const border = rootStyle.getPropertyValue('--border').trim() || '#24304a';
  const text = rootStyle.getPropertyValue('--text').trim() || '#e5e7eb';
  const muted = rootStyle.getPropertyValue('--muted').trim() || '#9aa4b2';

  const layout: any = {
    margin: { l: 55, r: 20, t: 10, b: 40 },
    paper_bgcolor: panel,
    plot_bgcolor: panel,
    font: { color: text },
    xaxis: {
      type: 'date',
      fixedrange: false,
      gridcolor: border,
      tickfont: { color: muted },
      // Keep the interactive range controls enabled (legacy behavior)
      rangeslider: { visible: true, bgcolor: bg, bordercolor: border, thickness: 0.14 },
      rangeselector: {
        bgcolor: panel,
        activecolor: '#7c3aed',
        bordercolor: border,
        borderwidth: 1,
        font: { color: text, size: 11 },
        buttons: [
          { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
          { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
          { count: 1, label: 'YTD', step: 'year', stepmode: 'todate' },
          { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
          { count: 5, label: '5y', step: 'year', stepmode: 'backward' },
          { step: 'all', label: 'All' },
        ],
      },
    },
    yaxis: {
      fixedrange: false,
      gridcolor: border,
      tickfont: { color: muted },
      domain: hasVol ? [0.34, 1] : [0, 1],
      title: '',
    },
    yaxis2: hasVol
      ? {
          fixedrange: false,
          gridcolor: border,
          tickfont: { color: muted },
          domain: [0, 0.30],
          title: 'Vol',
          showticklabels: false,
          zeroline: false,
        }
      : undefined,
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: panel,
      bordercolor: border,
      font: { color: text },
    },
    showlegend: false,
    bargap: 0.05,
  };

  try {
    Plotly.react(
      host,
      traces,
      layout,
      {
        displayModeBar: true,
        responsive: true,
        scrollZoom: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['select2d', 'lasso2d'],
      }
    );

    // Ensure plot is visible after re-render (tab switches)
    try {
      if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(host);
    } catch {}
  } catch {
    // fallback
    purgePlotly(host);
    Plotly.newPlot(host, traces, layout, { displayModeBar: true, responsive: true, scrollZoom: true, displaylogo: false });
  }
}

function PreviewGrid({ title, rows }: { title: string; rows: Array<Record<string, unknown>> }) {
  const columnDefs = React.useMemo<Array<ColDef<Record<string, unknown>>>>(() => {
    const first = rows && rows.length ? rows[0] : {};
    return Object.keys(first || {}).map((k) => ({ field: k, filter: 'agTextColumnFilter', sortable: true, resizable: true }));
  }, [rows]);

  return (
    <div>
      <div style={{ color: 'var(--muted)', margin: '6px 0' }}>
        <b>{title}</b> ({rows.length})
      </div>
      <div className="ag-theme-quant" style={{ height: 260, width: '100%' }}>
        <AgGridReact<Record<string, unknown>>
          rowData={rows}
          columnDefs={columnDefs}
          defaultColDef={{ filter: true, sortable: true, resizable: true, minWidth: 120 }}
          pagination
          paginationPageSize={25}
          paginationPageSizeSelector={[10, 25, 50, 100]}
        />
      </div>
    </div>
  );
}

type WorkspaceTab = 'plot' | 'table' | 'raw';

export function InspectorPageReact() {
  // Ensure legacy selection inputs exist; many other flows rely on them.
  React.useEffect(() => {
    getOrCreateHiddenInput('pLib');
    getOrCreateHiddenInput('pSym');
    // Optional legacy tuning knobs used by smoke tests.
    // If absent, default logic in readLegacyChartLimit() will fall back to 1500.
    getOrCreateHiddenInput('pLimit');
  }, []);

  const qs = React.useMemo(() => readQuery(), []);

  const [lib, setLib] = React.useState<string>(() => String(qs.lib || '').trim());
  const [sym, setSym] = React.useState<string>(() => String(qs.sym || '').trim());
  const [start, setStart] = React.useState<string>('');
  const [end, setEnd] = React.useState<string>('');

  const [preview, setPreview] = React.useState<PreviewPayload | null>(null);
  const [previewErr, setPreviewErr] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [plotLoading, setPlotLoading] = React.useState(false);

  const [workspaceTab, setWorkspaceTab] = React.useState<WorkspaceTab>('plot');

  // Keep last successful chart payload so we can re-render if Plotly drops state after tab switches.
  const [lastChart, setLastChart] = React.useState<ChartPayload | null>(null);

  const plotRef = React.useRef<HTMLDivElement | null>(null);

  const safeResizePlot = React.useCallback(() => {
    const el = plotRef.current;
    if (!el) return;

    const Plotly = (window as any).Plotly;
    if (!Plotly) return;

    // Plotly throws if the element isn't displayed or has no size.
    try {
      const rect = el.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return;

      // offsetParent === null is a decent "not displayed" heuristic.
      if ((el as any).offsetParent === null && getComputedStyle(el).position !== 'fixed') return;

      if (Plotly.Plots && Plotly.Plots.resize) Plotly.Plots.resize(el);
      if (typeof Plotly.redraw === 'function') Plotly.redraw(el);
    } catch {
      // ignore
    }
  }, []);

  // When switching back to Plot tab, Plotly sometimes renders empty if the div was hidden.
  // We force a resize/redraw and (if needed) re-render from lastChart.
  React.useEffect(() => {
    if (workspaceTab !== 'plot') return;
    const el = plotRef.current;
    if (!el) return;

    const Plotly = (window as any).Plotly;
    if (!Plotly) return;

    // If for any reason the plot lost its internal state, re-render from the last chart payload.
    if (lastChart && ((el as any).data == null || (el as any).data?.length === 0)) {
      try {
        renderPlotlyFromChartPayload(el, lastChart);
      } catch {}
    }

    // Always resize/redraw after becoming visible.
    setTimeout(() => safeResizePlot(), 0);
  }, [workspaceTab, lastChart, safeResizePlot]);

  // keep inputs synced from selection events
  React.useEffect(() => {
    const onSel = (ev: any) => {
      try {
        const d = ev?.detail || {};
        const lib2 = String(d.lib || '').trim();
        const sym2 = String(d.sym || '').trim();
        if (lib2) setLib(lib2);
        if (sym2) setSym(sym2);
      } catch {}
    };
    window.addEventListener('quantdsl:selection', onSel as any);
    return () => window.removeEventListener('quantdsl:selection', onSel as any);
  }, []);

  const persistSelection = React.useCallback(
    (lib2: string, sym2: string) => {
      setSelection({ lib: lib2, sym: sym2 });
      patchUiState({ pLib: lib2, pSym: sym2 });
      replaceQuery({ tab: 'inspector', lib: lib2, sym: sym2 });
    },
    []
  );

  const setTab = React.useCallback((t: WorkspaceTab) => {
    setWorkspaceTab(t);
  }, []);

  // Reset to plot when previewing (legacy behavior)
  React.useEffect(() => {
    if (loading) return;
    // if preview just updated successfully, we keep the most recently selected tab.
  }, [loading]);

  const runPreview = React.useCallback(async () => {
    const lib2 = lib.trim();
    const sym2 = sym.trim();
    if (!lib2 || !sym2) {
      setPreviewErr('Missing library/symbol');
      setPreview(null);
      return;
    }

    setLoading(true);
    setPlotLoading(true);
    setPreviewErr(null);

    persistSelection(lib2, sym2);

    // Legacy behavior: switch to plot after preview.
    setWorkspaceTab('plot');

    const plotEl = plotRef.current;
    const stEl = document.getElementById('plotStatus');
    if (stEl) stEl.textContent = '(loading...)';

    // IMPORTANT: never write placeholder text into the plot div after Plotly mounts.
    // Clearing/setting textContent here can destroy Plotly DOM and make the next switch blank.

    try {
      const urlPrev = `/api/catalog/preview/${encodeURIComponent(lib2)}?symbol=${encodeURIComponent(sym2)}&head=12&tail=12`;
      const data = await fetchJson<PreviewPayload>(urlPrev);
      setPreview(data);

      try {
        const params = new URLSearchParams({ symbol: sym2, limit: '1500' });
        // Allow smoke tests (and power-users) to tune chart size via the legacy hidden input.
        // Default stays 1500.
        params.set('limit', String(readLegacyChartLimit(1500)));
         if (start.trim()) params.set('start', start.trim());
         if (end.trim()) params.set('end', end.trim());
        const urlChart = `/api/catalog/chart/${encodeURIComponent(lib2)}?` + params.toString();
        const chart = await fetchJson<ChartPayload>(urlChart);
        setLastChart(chart);
        if (plotEl) {
          renderPlotlyFromChartPayload(plotEl, chart);
          // next tick resize so Plotly lays out correctly
          setTimeout(() => safeResizePlot(), 0);
        }
        if (stEl) stEl.textContent = '(ready)';
      } catch {
        // Keep any previous plot visible; do not clear.
        setLastChart(null);
        if (stEl) stEl.textContent = '(no chart)';
      }
    } catch (e: any) {
      const msg = e?.error?.message || e?.detail || (typeof e === 'string' ? e : JSON.stringify(e));
      setPreviewErr(String(msg));
      setPreview(null);
      setLastChart(null);
      if (stEl) stEl.textContent = '(error)';
      // Keep any previous plot visible; do not clear the canvas.
    } finally {
      setPlotLoading(false);
      setLoading(false);
    }
  }, [lib, sym, start, end, persistSelection, safeResizePlot]);

  return (
    <div id="pageInspector" className="page" style={{ marginTop: 12 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 12, alignItems: 'start' }}>
        <div className="card">
          <div style={{ fontWeight: 650, marginBottom: 8 }}>Inspector</div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <label className="label">
              library
              <input
                id="dLibrary"
                name="dLibrary"
                className="input"
                value={lib}
                onChange={(e) => setLib(e.target.value)}
                style={{ width: '100%' }}
              />
            </label>
            <label className="label">
              symbol
              <input
                id="dSymbol"
                name="dSymbol"
                className="input"
                value={sym}
                onChange={(e) => setSym(e.target.value)}
                style={{ width: '100%' }}
              />
            </label>
            <label className="label">
              start
              <input
                id="dStart"
                name="dStart"
                className="input"
                value={start}
                onChange={(e) => setStart(e.target.value)}
                placeholder="YYYY-MM-DD"
                style={{ width: '100%' }}
              />
            </label>
            <label className="label">
              end
              <input
                id="dEnd"
                name="dEnd"
                className="input"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
                placeholder="YYYY-MM-DD"
                style={{ width: '100%' }}
              />
            </label>
          </div>

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
            <button id="btnPreview" className="btn" onClick={() => void runPreview()} disabled={loading}>
              Preview
            </button>
          </div>

          {previewErr && <pre style={{ whiteSpace: 'pre-wrap', marginTop: 10 }}>{previewErr}</pre>}

          <div style={{ marginTop: 10, color: 'var(--muted)', fontSize: 12 }}>
            {loading ? '(loading...)' : '(ready)'}
          </div>

          {/* Download moved to Catalog under the grid to avoid duplicate contract IDs. */}
          <QualityPanel />
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <div style={{ fontWeight: 650 }}>Workspace</div>
            <div id="canvasTabs" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className="tab"
                id="tabPlot"
                data-tab="plot"
                disabled={workspaceTab === 'plot'}
                onClick={(e) => {
                  e.preventDefault();
                  setTab('plot');
                }}
              >
                Plot
              </button>
              <button
                className="tab"
                id="tabTable"
                data-tab="table"
                disabled={workspaceTab === 'table'}
                onClick={(e) => {
                  e.preventDefault();
                  setTab('table');
                }}
              >
                Table
              </button>
              <button
                className="tab"
                id="tabRaw"
                data-tab="raw"
                disabled={workspaceTab === 'raw'}
                onClick={(e) => {
                  e.preventDefault();
                  setTab('raw');
                }}
              >
                Raw
              </button>
            </div>
          </div>

          {/* Use visibility/height toggling instead of display:none to prevent Plotly from losing its graph state. */}
          <div
            id="plotWrap"
            style={{
              marginTop: 10,
              visibility: workspaceTab === 'plot' ? 'visible' : 'hidden',
              height: workspaceTab === 'plot' ? 'auto' : 0,
              overflow: 'hidden',
            }}
          >
            <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 6 }}>
              <b>Price chart</b> <span id="plotStatus">(idle)</span>
            </div>
            <div
              id="plot"
              data-testid="plotly-chart"
              className="card"
              style={{ height: 360, padding: 6, overflow: 'hidden', position: 'relative' }}
            >
              {/* Plotly mounts into this inner div. React never mutates its children. */}
              <div
                ref={plotRef}
                id="plotInner"
                style={{ width: '100%', height: '100%' }}
              />

              {(lastChart == null || (lastChart?.data || []).length === 0) && !plotLoading ? (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: 12 }}>
                  (no data)
                </div>
              ) : null}

              {plotLoading && (
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(0,0,0,0.25)',
                    color: 'var(--text)',
                    fontSize: 12,
                  }}
                >
                  (loading...)
                </div>
              )}
            </div>
          </div>

          <div id="tableWrap" style={{ marginTop: 10, display: workspaceTab === 'table' ? 'block' : 'none' }}>
            <div id="previewSummary" style={{ marginTop: 6 }}>
              {preview && (
                <>
                  <span style={{ display: 'inline-block', marginRight: 8 }}>
                    rows: <b>{String(preview.rows ?? '')}</b>
                  </span>
                  {preview.index_start ? (
                    <span style={{ marginRight: 8 }}>
                      start: <code>{String(preview.index_start)}</code>
                    </span>
                  ) : null}
                  {preview.index_end ? (
                    <span style={{ marginRight: 8 }}>
                      end: <code>{String(preview.index_end)}</code>
                    </span>
                  ) : null}
                </>
              )}
            </div>

            <div id="previewTables" style={{ marginTop: 10 }}>
              {preview && Array.isArray(preview.head) && preview.head.length > 0 && <PreviewGrid title="head" rows={preview.head} />}
              {preview && Array.isArray(preview.tail) && preview.tail.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <PreviewGrid title="tail" rows={preview.tail} />
                </div>
              )}
              {!preview && !previewErr && <div style={{ color: 'var(--muted)' }}>(no preview)</div>}
            </div>
          </div>

          <div id="rawWrap" style={{ marginTop: 10, display: workspaceTab === 'raw' ? 'block' : 'none' }}>
            <pre id="previewRaw" style={{ overflow: 'auto', maxHeight: 520, whiteSpace: 'pre-wrap' }}>
              {preview ? JSON.stringify(preview, null, 2) : ''}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

