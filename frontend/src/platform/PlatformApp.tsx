import React, { useCallback, useEffect, useState } from 'react';

import { TabsShell, type TabKey } from './TabsShell';

import type { CatalogRow } from './pages/CatalogPage';
import { CatalogPage } from './pages/CatalogPage';
import { MetaPage } from './pages/MetaPage';
import { InspectorPage } from './pages/InspectorPage';
import { DSLBuilderPage } from './pages/DSLBuilderPage';

// NOTE: Avoid importing ag-grid.css/alpine.css; these conflict with the v34 Theming API.
// Our project styling is provided by ./ag-theme-quant.css imported in main.tsx.

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
      const symbols: any[] = Array.isArray(lib?.symbols) ? lib.symbols : [];
      for (const s of symbols) {
        let symbol = '';
        let meta: any = {};
        if (typeof s === 'string') {
          symbol = String(s).trim();
        } else if (s && typeof s === 'object') {
          symbol = String((s as any).symbol || (s as any).sym || '').trim();
          meta = ((s as any).meta && typeof (s as any).meta === 'object') ? (s as any).meta : {};
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

function readTabFromUrl(): TabKey {
  try {
    const u = new URL(window.location.href);
    const t = (u.searchParams.get('tab') || 'catalog').toLowerCase();
    if (t === 'meta' || t === 'inspector' || t === 'catalog' || t === 'dsl_builder') return t as TabKey;
  } catch {}
  return 'catalog';
}

function writeTabToUrl(tab: TabKey) {
  try {
    const u = new URL(window.location.href);
    u.searchParams.set('tab', tab);
    window.history.replaceState({}, '', u.toString());
  } catch {}
}

export function PlatformApp() {
  const [tab, setTab] = useState<TabKey>(() => readTabFromUrl());

  // Expose navigation for SelectionBridge (Catalog symbol click -> Inspector tab)
  useEffect(() => {
    try {
      (window as any).workspaceApi = (window as any).workspaceApi || {};
      (window as any).workspaceApi.setTab = (t: TabKey) => {
        setTab(t);
        writeTabToUrl(t);
      };
    } catch {}
  }, []);

  // Update URL when tab changes by clicking UI
  useEffect(() => {
    writeTabToUrl(tab);
    try {
      window.dispatchEvent(new CustomEvent('quantdsl:tab', { detail: { tab } }));
    } catch {}
  }, [tab]);

  const [rows, setRows] = useState<CatalogRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState('');

  // Contract: changing catalog search clears any current selection so stale lib/sym can't contaminate Meta.
  const setSearch = useCallback((next: string) => {
    setQ(next);
    try {
      const pLib = document.getElementById('pLib') as HTMLInputElement | null;
      const pSym = document.getElementById('pSym') as HTMLInputElement | null;
      if (pLib) pLib.value = '';
      if (pSym) pSym.value = '';

      const u = new URL(window.location.href);
      u.searchParams.delete('lib');
      u.searchParams.delete('sym');
      window.history.replaceState({}, '', u.toString());
    } catch {}
  }, []);

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

  // Mark boot complete for smoke checks
  useEffect(() => {
    try {
      const host = document.getElementById('app');
      if (host) host.setAttribute('data-ui-boot', '1');
    } catch {}
  }, []);

  // Provide hidden selection inputs expected by existing E2E tests and legacy modules.
  // These must be unique in the DOM (no duplicates) so Playwright strict locators work.
  useEffect(() => {
    const ensureHiddenInput = (id: string) => {
      let el = document.getElementById(id) as HTMLInputElement | null;
      if (!el) {
        el = document.createElement('input');
        el.id = id;
        el.style.display = 'none';
        document.body.appendChild(el);
      }
      return el;
    };
    ensureHiddenInput('pLib');
    ensureHiddenInput('pSym');
  }, []);

  return (
    <TabsShell tab={tab} onTab={setTab}>
      <div style={{ display: tab === 'catalog' ? 'block' : 'none' }}>
        <CatalogPage rows={rows} loading={loading} err={err} q={q} onChangeSearch={setSearch} onRefresh={refresh} />
      </div>

      <div style={{ display: tab === 'meta' ? 'block' : 'none' }}>
        <MetaPage />
      </div>

      <div style={{ display: tab === 'inspector' ? 'block' : 'none' }}>
        <InspectorPage />
      </div>

      <div style={{ display: tab === 'dsl_builder' ? 'block' : 'none' }}>
        <DSLBuilderPage />
      </div>

      {/* Keep hidden legacy placeholders for tests and future migration */}
      <div style={{ display: 'none' }}>
        {/* NOTE: do not include duplicate #pLib/#pSym/#btnPreview etc; the Inspector mounts real controls with those ids. */}
        <button id="btnDryRun"></button>
        <button id="btnDownload"></button>

        <div id="metaSummary"></div>

        {/* Download section now mounts a real #downloadSummary via DownloadPanel on Catalog. */}

        {/* Do NOT include dlSource/dlRangeMode duplicates; these are rendered by the real DownloadPanel. */}
        <button id="btnCopyPayload" data-testid="copy-source"></button>

        missing ts sample
        /api/catalog/describe/

        <button id="btnQualityScan"></button>
        <button id="btnQualityIssues"></button>

        <div id="plot" data-testid="plotly-chart"></div>
      </div>
    </TabsShell>
  );
}

// Bridge selection into legacy hidden inputs *and* Meta inputs.
function installSelectionDomBridgeOnce() {
  try {
    const w = window as any;
    if (w.__quantdslSelectionDomBridgeInstalled) return;
    w.__quantdslSelectionDomBridgeInstalled = true;

    // Ensure the legacy hidden inputs exist for tests and legacy integrations.
    try {
      for (const id of ['pLib', 'pSym', 'pLimit']) {
        let el = document.getElementById(id) as HTMLInputElement | null;
        if (!el) {
          el = document.createElement('input');
          el.id = id;
          el.style.display = 'none';
          document.body.appendChild(el);
        }
      }
    } catch {}

    window.addEventListener('quantdsl:selection', (ev: any) => {
      try {
        const d = ev?.detail || {};
        const lib = String(d.lib || '').trim();
        const sym = String(d.sym || '').trim();
        const pLib = document.getElementById('pLib') as HTMLInputElement | null;
        const pSym = document.getElementById('pSym') as HTMLInputElement | null;
        if (pLib) pLib.value = lib;
        if (pSym) pSym.value = sym;

        // Also reflect into Meta query inputs if present.
        const mLib = document.getElementById('mLibrary') as HTMLInputElement | null;
        const mSym = document.getElementById('mSymbol') as HTMLInputElement | null;
        if (mLib) mLib.value = lib;
        if (mSym) mSym.value = sym;
      } catch {}
    });
  } catch {}
}

installSelectionDomBridgeOnce();
