import React from 'react';

import { InspectorPageReact } from './inspector/InspectorPageReact';

type InspectorModule = { mountInspector?: (containerId?: string) => void };

declare global {
  interface Window {
    __quantdslInspectorModule?: InspectorModule;
    __quantdslInspectorModulePromise?: Promise<InspectorModule>;
    __quantdslInspectorCacheBust?: string;
  }
}

function getCacheBustToken(): string {
  // Stable for this page session; avoids Date.now() causing re-fetch on every mount.
  if (window.__quantdslInspectorCacheBust) return window.__quantdslInspectorCacheBust;
  window.__quantdslInspectorCacheBust = String(Date.now());
  return window.__quantdslInspectorCacheBust;
}

function loadInspectorModule(): Promise<InspectorModule> {
  // Cache across navigations; React may mount/unmount this page.
  if (window.__quantdslInspectorModule) return Promise.resolve(window.__quantdslInspectorModule);
  if (window.__quantdslInspectorModulePromise) return window.__quantdslInspectorModulePromise;

  window.__quantdslInspectorModulePromise = (async () => {
    // Use a dynamic import wrapper so TS/Vite don't require build-time module resolution.
    const dynamicImport = new Function('u', 'return import(u)') as (u: string) => Promise<any>;
    const v = getCacheBustToken();
    const mod = (await dynamicImport(`/static/assets/inspector.js?v=${encodeURIComponent(v)}`)) as InspectorModule;
    window.__quantdslInspectorModule = mod;
    return mod;
  })();

  return window.__quantdslInspectorModulePromise;
}

function shouldUseLegacyInspector(): boolean {
  try {
    const u = new URL(window.location.href);
    const v = (u.searchParams.get('inspector') || '').toLowerCase().trim();
    if (v === 'legacy') return true;
    if (v === 'react') return false;
  } catch {}
  return false;
}

export function InspectorPage() {
  const [useLegacy] = React.useState<boolean>(() => shouldUseLegacyInspector());

  if (!useLegacy) {
    return <InspectorPageReact />;
  }

  // Legacy fallback (temporary during migration)
  const hostRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        if (!hostRef.current) return;
        hostRef.current.id = 'pageInspector';

        // Give immediate feedback while loading; helps debugging in headless runs.
        hostRef.current.textContent = '(loading Inspector...)';

        const mod = await loadInspectorModule();
        if (cancelled) return;

        if (mod && typeof mod.mountInspector === 'function') {
          mod.mountInspector('pageInspector');
        } else {
          hostRef.current.textContent = '(Inspector module missing mountInspector())';
        }
      } catch (e) {
        try {
          if (hostRef.current) hostRef.current.textContent = '(failed to mount Inspector)';
        } catch {
          // ignore
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return <div ref={hostRef} id="pageInspector" className="page" style={{ marginTop: 12 }} />;
}
