import React from 'react';

export type TabKey = 'catalog' | 'meta' | 'inspector';

export function TabsShell({
  tab,
  onTab,
  children,
}: {
  tab: TabKey;
  onTab: (t: TabKey) => void;
  children: React.ReactNode;
}) {
  return (
    <div className="app">
      <div className="appHeader">
        <h1 className="appTitle">Platform UI</h1>
        <div className="appSubtitle">Research workflow</div>
      </div>

      <div id="mainTabs" className="tabs">
        <button className="tab" id="tabCatalog" disabled={tab === 'catalog'} onClick={() => onTab('catalog')}>
          Catalog
        </button>
        <button className="tab" id="tabMeta" disabled={tab === 'meta'} onClick={() => onTab('meta')}>
          Meta
        </button>
        <button className="tab" id="tabInspector" disabled={tab === 'inspector'} onClick={() => onTab('inspector')}>
          Inspector
        </button>
      </div>

      {children}
    </div>
  );
}

