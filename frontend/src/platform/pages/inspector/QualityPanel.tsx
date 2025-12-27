import React from 'react';
import { AgGridReact } from 'ag-grid-react';
import type { ColDef } from 'ag-grid-community';

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

function toRows(data: any): Array<Record<string, unknown>> {
  if (Array.isArray(data?.rows)) return data.rows;
  if (Array.isArray(data?.issues)) return data.issues;
  if (Array.isArray(data)) return data;
  if (data && typeof data === 'object') return Object.keys(data).map((k) => ({ key: k, value: (data as any)[k] }));
  return [];
}

function QualityGrid({ rows }: { rows: Array<Record<string, unknown>> }) {
  const colDefs = React.useMemo<Array<ColDef<Record<string, unknown>>>>(() => {
    const first = rows && rows.length ? rows[0] : {};
    return Object.keys(first || {}).map((k) => ({ field: k, filter: 'agTextColumnFilter', sortable: true, resizable: true }));
  }, [rows]);

  if (!rows.length) return <div style={{ color: 'var(--muted)' }}>(no rows)</div>;

  return (
    <div className="ag-theme-quant" style={{ height: 320, width: '100%' }}>
      <AgGridReact<Record<string, unknown>>
        rowData={rows.slice(0, 500)}
        columnDefs={colDefs}
        defaultColDef={{ filter: true, sortable: true, resizable: true, minWidth: 120 }}
        pagination
        paginationPageSize={50}
        paginationPageSizeSelector={[20, 50, 100, 200]}
      />
    </div>
  );
}

export function QualityPanel() {
  const [provider, setProvider] = React.useState('');
  const [freq, setFreq] = React.useState('');
  const [dataset, setDataset] = React.useState('');
  const [kind, setKind] = React.useState('');
  const [entity, setEntity] = React.useState('');
  const [limit, setLimit] = React.useState('200');

  const [rows, setRows] = React.useState<Array<Record<string, unknown>>>([]);
  const [raw, setRaw] = React.useState<any>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const buildParams = React.useCallback(() => {
    const params = new URLSearchParams();
    if (provider.trim()) params.set('provider', provider.trim());
    if (freq.trim()) params.set('frequency', freq.trim());
    if (dataset.trim()) params.set('dataset', dataset.trim());
    if (kind.trim()) params.set('kind', kind.trim());
    if (entity.trim()) params.set('entity', entity.trim());
    if (limit.trim()) params.set('limit', limit.trim());
    return params;
  }, [provider, freq, dataset, kind, entity, limit]);

  const runScan = React.useCallback(async () => {
    setBusy(true);
    setErr(null);
    setRows([]);
    try {
      const params = buildParams();
      const data = await fetchJson<any>('/api/quality/scan?' + params.toString(), { method: 'POST' });
      setRaw(data);
      setRows(toRows(data));
    } catch (e: any) {
      setErr(JSON.stringify(e, null, 2));
    } finally {
      setBusy(false);
    }
  }, [buildParams]);

  const loadIssues = React.useCallback(async () => {
    setBusy(true);
    setErr(null);
    setRows([]);
    try {
      const params = buildParams();
      const data = await fetchJson<any>('/api/quality/issues?' + params.toString());
      const r = Array.isArray(data?.rows) ? data.rows : toRows(data);
      setRaw(data);
      setRows(r);
    } catch (e: any) {
      setErr(JSON.stringify(e, null, 2));
    } finally {
      setBusy(false);
    }
  }, [buildParams]);

  return (
    <div id="qualityPanel" style={{ marginTop: 14, borderTop: '1px solid var(--border)', paddingTop: 12 }}>
      <div style={{ fontWeight: 650, marginBottom: 10 }}>Quality</div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
        <label className="label">provider <input id="qProvider" className="input" value={provider} onChange={(e) => setProvider(e.target.value)} placeholder="PARQUET/FRED/YF" style={{ width: '100%' }} /></label>
        <label className="label">frequency <input id="qFreq" className="input" value={freq} onChange={(e) => setFreq(e.target.value)} placeholder="1d" style={{ width: '100%' }} /></label>
        <label className="label">dataset <input id="qDataset" className="input" value={dataset} onChange={(e) => setDataset(e.target.value)} placeholder="(optional)" style={{ width: '100%' }} /></label>
        <label className="label">kind <input id="qKind" className="input" value={kind} onChange={(e) => setKind(e.target.value)} placeholder="(optional)" style={{ width: '100%' }} /></label>
        <label className="label">entity <input id="qEntity" className="input" value={entity} onChange={(e) => setEntity(e.target.value)} placeholder="(optional)" style={{ width: '100%' }} /></label>
        <label className="label">limit <input id="qLimit" className="input" value={limit} onChange={(e) => setLimit(e.target.value)} style={{ maxWidth: 120 }} /></label>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
        <button id="btnQualityScan" className="btn" onClick={() => void runScan()} disabled={busy}>Scan</button>
        <button id="btnQualityIssues" className="btn" onClick={() => void loadIssues()} disabled={busy}>Issues</button>
      </div>

      <div id="quality" style={{ marginTop: 10 }}>
        {busy && <div style={{ color: 'var(--muted)' }}>(loading...)</div>}
        {err && <pre style={{ whiteSpace: 'pre-wrap' }}>{err}</pre>}
        {!busy && !err && rows.length > 0 && <QualityGrid rows={rows} />}
        {!busy && !err && rows.length === 0 && <div style={{ color: 'var(--muted)' }}>(no rows)</div>}

        <details style={{ marginTop: 8 }}>
          <summary style={{ color: 'var(--muted)' }}>raw</summary>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{raw ? JSON.stringify(raw, null, 2) : ''}</pre>
        </details>
      </div>
    </div>
  );
}

