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

  // Prevent overlapping refresh() calls (which can cause loading flicker and stale state races).
  const refreshInFlight = React.useRef(false);
  const lastReqId = React.useRef(0);

  // "busy" means: the user explicitly triggered a refresh/apply/clear and buttons should be disabled.
  // Background polling should NOT set busy (otherwise controls flicker/disable and intercept clicks).
  const [busy, setBusy] = React.useState(false);

  const [strategyId, setStrategyId] = React.useState<string>('');
  const [status, setStatus] = React.useState<string>('');

  // Applied filters (so you can stage changes then Apply, pro UX).
  const [appliedStrategyId, setAppliedStrategyId] = React.useState<string>('');
  const [appliedStatus, setAppliedStatus] = React.useState<string>('');

  const [selectedRun, setSelectedRun] = React.useState<RunRecord | null>(null);

  const [logModal, setLogModal] = React.useState<{ title: string; text: string; runId: string } | null>(null);

  const refresh = React.useCallback(
    async ({ showSpinner }: { showSpinner: boolean } = { showSpinner: true }) => {
      if (refreshInFlight.current) return;
      refreshInFlight.current = true;

      const reqId = ++lastReqId.current;

      if (showSpinner) {
        setLoading(true);
        setBusy(true);
      }
      setErr(null);

      try {
        const qs = new URLSearchParams();
        qs.set('limit', '50');
        if (appliedStrategyId) qs.set('strategy_id', appliedStrategyId);
        if (appliedStatus) qs.set('status', appliedStatus);
        const data = await fetchJson<RunsListResponse>(`/api/runs?${qs.toString()}`);

        // Ignore stale responses.
        if (reqId != lastReqId.current) return;

        const nextRows = Array.isArray(data.runs) ? data.runs : [];
        setRows(nextRows);
        // Keep selectedRun fresh if present.
        if (selectedRun) {
          const upd = nextRows.find((r) => r.run_id === selectedRun.run_id) || null;
          setSelectedRun(upd);
        }
      } catch (e: any) {
        if (reqId != lastReqId.current) return;
        const msg = e?.error?.message || e?.detail || (typeof e === 'string' ? e : JSON.stringify(e));
        setErr(String(msg));
      } finally {
        if (reqId == lastReqId.current) {
          if (showSpinner) {
            setBusy(false);
            setLoading(false);
          }
        }
        refreshInFlight.current = false;
      }
    },
    [appliedStatus, appliedStrategyId, selectedRun]
  );

  // Only poll while there is something non-terminal visible.
  React.useEffect(() => {
    let timer: any = null;
    const needsPoll = rows.some((r) => r.status === 'pending' || r.status === 'running');
    if (needsPoll)
      timer = setInterval(() => {
        // Background refresh: keep controls usable and avoid flicker.
        refresh({ showSpinner: false });
      }, 3000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [rows, refresh]);

  // Refresh when tab is activated.
  React.useEffect(() => {
    const onTab = (ev: any) => {
      const t = String(ev?.detail?.tab || '').toLowerCase();
      if (t === 'runs') refresh({ showSpinner: false });
    };
    window.addEventListener('quantdsl:tab', onTab as any);
    setTimeout(() => refresh({ showSpinner: true }), 0);
    return () => window.removeEventListener('quantdsl:tab', onTab as any);
  }, [refresh]);

  // SSE live tail for the logs modal.
  React.useEffect(() => {
    if (!logModal) return;

    let es: EventSource | null = null;
    let cancelled = false;

    const runId = logModal.runId;

    const fetchFullOnce = async () => {
      try {
        const j = await fetchJson<any>(`/api/runs/${runId}/logs`);
        if (cancelled) return;
        const txt = String(j?.logs || '').slice(0, 200_000);
        setLogModal((prev) => (prev ? { ...prev, text: txt || '(empty)' } : prev));
      } catch {
        // ignore seed failure; SSE/fallback below will handle.
      }
    };

    const fallbackFetch = async (msg?: string) => {
      try {
        const j = await fetchJson<any>(`/api/runs/${runId}/logs`);
        if (cancelled) return;
        setLogModal((prev) => (prev ? { ...prev, text: String(j?.logs || '').slice(0, 200_000) || '(empty)' } : prev));
      } catch (e: any) {
        if (cancelled) return;
        const m = msg || e?.error?.message || 'failed to load logs';
        setLogModal((prev) => (prev ? { ...prev, text: String(m) } : prev));
      }
    };

    // Seed immediately so completed runs show content even if SSE ends quickly.
    fetchFullOnce();

    try {
      es = new EventSource(`/api/runs/${runId}/logs/stream`);

      es.onmessage = (ev) => {
        if (cancelled) return;
        const data = String(ev?.data || '');
        if (!data) return;
        setLogModal((prev) => {
          if (!prev) return prev;
          const next = (prev.text || '') + data + '\n';
          const bounded = next.length > 200_000 ? next.slice(next.length - 200_000) : next;
          return { ...prev, text: bounded };
        });
      };

      es.addEventListener('done', () => {
        try {
          es?.close();
        } catch {}
      });

      es.onerror = () => {
        try {
          es?.close();
        } catch {}
        fallbackFetch('SSE disconnected; loaded full logs instead.');
      };
    } catch {
      fallbackFetch('SSE not available; loaded full logs instead.');
    }

    return () => {
      cancelled = true;
      try {
        es?.close();
      } catch {}
    };
  }, [logModal?.runId]);

  const onApplyFilters = React.useCallback(() => {
    setAppliedStrategyId(strategyId);
    setAppliedStatus(status);
    setSelectedRun(null);
    // Trigger an explicit refresh with spinner => disables buttons briefly.
    setTimeout(() => refresh({ showSpinner: true }), 0);
  }, [refresh, status, strategyId]);

  const onClearFilters = React.useCallback(() => {
    setStrategyId('');
    setStatus('');
    setAppliedStrategyId('');
    setAppliedStatus('');
    setSelectedRun(null);
    setTimeout(() => refresh({ showSpinner: true }), 0);
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
                onClick={() => {
                  if (!runId) return;
                  setLogModal({ title: `Logs — ${shortId(runId)}`, text: '', runId });
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
      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ fontWeight: 650, marginBottom: 6 }}>Filters</div>
        <div className="runFilters">
          <div className="runFiltersLeft">
            <label className="label" style={{ minWidth: 0 }}>
              Strategy
              <select
                className="input runSelect"
                id="runsFilterStrategy"
                data-testid="runsFilterStrategy"
                value={strategyId}
                onChange={(e) => setStrategyId(String(e.target.value || ''))}
                disabled={busy}
              >
                <option value="">(all)</option>
                {availableStrategies.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>

            <label className="label" style={{ minWidth: 0 }}>
              Status
              <select
                className="input runSelect"
                id="runsFilterStatus"
                data-testid="runsFilterStatus"
                value={status}
                onChange={(e) => setStatus(String(e.target.value || ''))}
                disabled={busy}
              >
                <option value="">(all)</option>
                <option value="pending">pending</option>
                <option value="running">running</option>
                <option value="succeeded">succeeded</option>
                <option value="failed">failed</option>
              </select>
            </label>
          </div>

          <div className="runFiltersRight">
            <button className="btn" data-testid="btnRunsApply" onClick={onApplyFilters} disabled={busy}>
              Apply
            </button>
            <button className="btn" data-testid="btnRunsClear" onClick={onClearFilters} disabled={busy}>
              Clear
            </button>
            <button
              className="btn"
              id="btnRunsRefresh"
              data-testid="btnRunsRefresh"
              onClick={() => refresh({ showSpinner: true })}
              disabled={busy}
            >
              Refresh
            </button>
            {loading && (
              <span
                style={{ color: 'var(--muted)', pointerEvents: 'none', userSelect: 'none' }}
                data-testid="runsLoading"
              >
                (loading...)
              </span>
            )}
          </div>
        </div>
        {(appliedStrategyId || appliedStatus) && (
          <div style={{ marginTop: 8, color: 'var(--muted)', fontSize: 12 }} data-testid="runsAppliedFilters">
            Applied: {appliedStrategyId || '(all strategies)'} / {appliedStatus || '(all statuses)'}
          </div>
        )}
      </div>

      {err && (
        <div className="card" style={{ marginTop: 10 }}>
          <div style={{ fontWeight: 650, marginBottom: 6 }}>Error</div>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{err}</pre>
        </div>
      )}

      {!err && rows.length === 0 && !loading && <div style={{ color: 'var(--muted)', marginTop: 10 }}>(no runs)</div>}

      <div style={{ display: 'grid', gridTemplateColumns: selectedRun ? '2fr 1fr' : '1fr', gap: 12, marginTop: 10 }}>
        <div className="card">
          <div style={{ fontWeight: 650, marginBottom: 6 }}>Runs</div>
          <div className="ag-theme-quant" style={{ width: '100%', height: 520 }} data-testid="runs-grid">
            <AgGridReact
              rowData={rows}
              columnDefs={colDefs}
              defaultColDef={defaultColDef}
              animateRows={false}
              rowSelection={'single'}
              onRowClicked={(ev) => {
                const r = ev.data as any;
                if (!r) return;
                setSelectedRun(r as RunRecord);
              }}
            />
          </div>
        </div>

        {selectedRun && (
          <div className="card" data-testid="runDetails">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
              <div style={{ fontWeight: 700 }}>Run Details</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <select
                  className="input"
                  data-testid="runDetailsSelect"
                  value={selectedRun.run_id}
                  onChange={(e) => {
                    const rid = String(e.target.value || '');
                    const r = rows.find((x) => x.run_id === rid) || null;
                    setSelectedRun(r);
                  }}
                >
                  {rows.map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {shortId(r.run_id)}…{String(r.run_id).slice(-4)} — {r.strategy_id}
                    </option>
                  ))}
                </select>
                <button className="btn" data-testid="btnRunDetailsClose" onClick={() => setSelectedRun(null)}>
                  Close
                </button>
              </div>
            </div>
            <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '110px 1fr', gap: 8, fontSize: 12 }}>
              <div style={{ color: 'var(--muted)' }}>Status</div>
              <div>{statusBadge(selectedRun.status)}</div>
              <div style={{ color: 'var(--muted)' }}>Strategy</div>
              <div>{selectedRun.strategy_id}</div>
              <div style={{ color: 'var(--muted)' }}>Run ID</div>
              <div style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' }}>{selectedRun.run_id}</div>
              <div style={{ color: 'var(--muted)' }}>Error</div>
              <div style={{ whiteSpace: 'pre-wrap' }}>{selectedRun.error || ''}</div>
            </div>
            <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className="btn"
                data-testid="btnRunDetailsLogs"
                onClick={() => setLogModal({ title: `Logs — ${shortId(selectedRun.run_id)}`, text: '', runId: selectedRun.run_id })}
              >
                View Logs
              </button>
              <button
                className="btn"
                data-testid="btnRunDetailsReport"
                onClick={() => {
                  const runId = selectedRun.run_id;
                  const reportUrl = selectedRun.reports_url || (runId ? `/reports/runs/${runId}/index.html` : '');
                  if (!reportUrl) return;
                  window.open(reportUrl, '_blank');
                }}
              >
                View Report
              </button>
            </div>
          </div>
        )}
      </div>

      {logModal && (
        <div
          className="runModalBackdrop"
          data-testid="runLogsModal"
          onClick={(e) => {
            // Only close when user clicks the backdrop itself (not inside the modal panel).
            if (e.target === e.currentTarget) setLogModal(null);
          }}
          role="dialog"
          aria-modal="true"
        >
          <div className="runModal" onClick={(e) => e.stopPropagation()}>
            <div className="runModalHeader">
              <div className="runModalTitle">{logModal.title}</div>
              <button className="btn" data-testid="btnRunLogsClose" onClick={() => setLogModal(null)}>
                Close
              </button>
            </div>
            <div className="runModalBody">
              <pre className="runModalPre" data-testid="runLogsText">{logModal.text}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

