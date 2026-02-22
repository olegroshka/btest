import React from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef } from 'ag-grid-community';
import './RunsPage.css';

type RunStatus = 'pending' | 'running' | 'succeeded' | 'failed';

type RunRecord = {
  run_id: string;
  strategy_id: string;
  strategy_hash: string;
  status: RunStatus;
  submitted_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  duration_s?: number | null;
  metrics?: Record<string, any> | null;
  error?: string | null;
  artifacts_dir?: string | null;
  reports_url?: string | null;
};

type RunsListResponse = {
  runs: RunRecord[];
  total: number;
};

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  const data = (await res.json().catch(() => ({ error: { message: 'non-json response' } }))) as any;
  if (!res.ok) {
    if (data && data.detail && !data.error) {
      throw { error: { code: `HTTP_${res.status}`, message: String(data.detail) } };
    }
    throw data;
  }
  return data as T;
}

function fmtAge(iso: string): string {
  try {
    const dt = new Date(iso);
    const s = Math.max(0, (Date.now() - dt.getTime()) / 1000);
    if (s < 60) return `${Math.floor(s)}s`;
    const m = s / 60;
    if (m < 60) return `${Math.floor(m)}m`;
    const h = m / 60;
    if (h < 48) return `${Math.floor(h)}h`;
    const d = h / 24;
    return `${Math.floor(d)}d`;
  } catch {
    return '';
  }
}

function shortId(runId: string): string {
  const s = String(runId || '');
  return s.length > 8 ? s.slice(0, 8) : s;
}

function statusBadge(status: RunStatus): React.ReactNode {
  const s = (status || 'pending') as RunStatus;
  const cls =
    s === 'succeeded'
      ? 'runStatus runStatus--succeeded'
      : s === 'failed'
        ? 'runStatus runStatus--failed'
        : s === 'running'
          ? 'runStatus runStatus--running'
          : 'runStatus runStatus--pending';
  const label = String(s).toUpperCase();
  return (
    <span className={cls} data-testid={`run-status-${s}`}>
      {label}
    </span>
  );
}

export function RunsPage() {
  const [rows, setRows] = React.useState<RunRecord[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);

  const [strategyId, setStrategyId] = React.useState<string>('');
  const [status, setStatus] = React.useState<string>('');

  const refresh = React.useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const qs = new URLSearchParams();
      qs.set('limit', '50');
      if (strategyId) qs.set('strategy_id', strategyId);
      if (status) qs.set('status', status);
      const data = await fetchJson<RunsListResponse>(`/api/runs?${qs.toString()}`);
      setRows(Array.isArray(data.runs) ? data.runs : []);
    } catch (e: any) {
      const msg = e?.error?.message || e?.detail || (typeof e === 'string' ? e : JSON.stringify(e));
      setErr(String(msg));
    } finally {
      setLoading(false);
    }
  }, [status, strategyId]);

  // Only poll while there is something non-terminal visible.
  React.useEffect(() => {
    let timer: any = null;
    const needsPoll = rows.some((r) => r.status === 'pending' || r.status === 'running');
    if (needsPoll) timer = setInterval(refresh, 3000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [rows, refresh]);

  // Refresh when tab is activated.
  React.useEffect(() => {
    const onTab = (ev: any) => {
      const t = String(ev?.detail?.tab || '').toLowerCase();
      if (t === 'runs') refresh();
    };
    window.addEventListener('quantdsl:tab', onTab as any);
    setTimeout(refresh, 0);
    return () => window.removeEventListener('quantdsl:tab', onTab as any);
  }, [refresh]);

  const colDefs = React.useMemo<Array<ColDef<RunRecord>>>(
    () => [
      {
        field: 'status',
        headerName: 'Status',
        width: 140,
        sortable: true,
        cellRenderer: (p: any) => statusBadge(String(p.value || 'pending') as RunStatus),
      },
      { field: 'strategy_id', headerName: 'Strategy', sortable: true, filter: 'agTextColumnFilter', flex: 2, minWidth: 220 },
      {
        field: 'run_id',
        headerName: 'Run',
        width: 110,
        sortable: true,
        valueGetter: (p: any) => shortId(String(p.data?.run_id || '')),
      },
      {
        field: 'submitted_at',
        headerName: 'Submitted',
        width: 130,
        sortable: true,
        valueGetter: (p: any) => fmtAge(String(p.data?.submitted_at || '')),
      },
      {
        field: 'duration_s',
        headerName: 'Dur(s)',
        width: 110,
        sortable: true,
        valueGetter: (p: any) => {
          const v = p.data?.duration_s;
          if (v === null || v === undefined) return '';
          const n = Number(v);
          if (!Number.isFinite(n)) return '';
          return n.toFixed(2);
        },
      },
      {
        headerName: 'Sharpe',
        width: 110,
        sortable: true,
        valueGetter: (p: any) => (p.data?.metrics && p.data.metrics.sharpe !== undefined ? Number(p.data.metrics.sharpe).toFixed(3) : ''),
      },
      {
        headerName: 'Total Return',
        width: 140,
        sortable: true,
        valueGetter: (p: any) => (p.data?.metrics && p.data.metrics.total_return !== undefined ? Number(p.data.metrics.total_return).toFixed(4) : ''),
      },
      {
        headerName: 'Max DD',
        width: 110,
        sortable: true,
        valueGetter: (p: any) => (p.data?.metrics && p.data.metrics.max_drawdown !== undefined ? Number(p.data.metrics.max_drawdown).toFixed(4) : ''),
      },
      {
        headerName: 'Engine',
        width: 140,
        sortable: true,
        valueGetter: (p: any) => String(p.data?.params?.engine || ''),
      },
      {
        headerName: 'Actions',
        width: 220,
        cellRenderer: (p: any) => {
          const r: RunRecord | undefined = p.data;
          const runId = r?.run_id || '';
          const reportUrl = r?.reports_url || (runId ? `/reports/runs/${runId}/index.html` : '');
          return (
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className="btn"
                data-testid={`btnRunReport-${shortId(runId)}`}
                disabled={!runId}
                onClick={() => {
                  if (!reportUrl) return;
                  window.open(reportUrl, '_blank');
                }}
              >
                View Report
              </button>
              <button
                className="btn"
                data-testid={`btnRunLogs-${shortId(runId)}`}
                disabled={!runId}
                onClick={async () => {
                  if (!runId) return;
                  try {
                    const j = await fetchJson<any>(`/api/runs/${runId}/logs`);
                    alert(String(j?.logs || '').slice(0, 4000));
                  } catch (e: any) {
                    alert(e?.error?.message || 'failed to load logs');
                  }
                }}
              >
                View Logs
              </button>
            </div>
          );
        },
      },
    ],
    []
  );

  const defaultColDef = React.useMemo<ColDef>(
    () => ({
      resizable: true,
      sortable: true,
      filter: true,
      minWidth: 100,
    }),
    []
  );

  const availableStrategies = React.useMemo(() => {
    const s = new Set<string>();
    for (const r of rows) s.add(String(r.strategy_id || ''));
    return Array.from(s).filter(Boolean).sort();
  }, [rows]);

  return (
    <div id="pageRuns" className="page">
      <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <button className="btn" id="btnRunsRefresh" data-testid="btnRunsRefresh" onClick={() => refresh()} disabled={loading}>
          Refresh
        </button>

        <label style={{ color: 'var(--muted)' }}>Strategy</label>
        <select
          className="input"
          id="runsFilterStrategy"
          data-testid="runsFilterStrategy"
          value={strategyId}
          onChange={(e) => setStrategyId(String(e.target.value || ''))}
          disabled={loading}
        >
          <option value="">(all)</option>
          {availableStrategies.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        <label style={{ color: 'var(--muted)' }}>Status</label>
        <select
          className="input"
          id="runsFilterStatus"
          data-testid="runsFilterStatus"
          value={status}
          onChange={(e) => setStatus(String(e.target.value || ''))}
          disabled={loading}
        >
          <option value="">(all)</option>
          <option value="pending">pending</option>
          <option value="running">running</option>
          <option value="succeeded">succeeded</option>
          <option value="failed">failed</option>
        </select>

        {loading && <span style={{ color: 'var(--muted)' }}>(loading...)</span>}
      </div>

      {err && (
        <div className="card" style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 650, marginBottom: 6 }}>Error</div>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{err}</pre>
        </div>
      )}

      {!err && rows.length === 0 && !loading && <div style={{ color: 'var(--muted)', marginTop: 10 }}>(no runs)</div>}

      <div className="card" style={{ marginTop: 10 }}>
        <div style={{ fontWeight: 650, marginBottom: 6 }}>Runs</div>
        <div className="ag-theme-quant" style={{ width: '100%', height: 520 }} data-testid="runs-grid">
          <AgGridReact rowData={rows} columnDefs={colDefs} defaultColDef={defaultColDef} animateRows={false} />
        </div>
      </div>
    </div>
  );
}

