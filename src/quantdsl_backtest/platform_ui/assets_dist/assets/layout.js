import { readQuery, replaceQuery } from './state.js';

export function mountLayout(containerId = 'app') {
  const host = document.getElementById(containerId);
  if (!host) return;

  host.innerHTML = `
    <div class="app">
      <div class="appHeader">
        <h1 class="appTitle">Platform UI</h1>
        <div class="appSubtitle">Research workflow</div>
      </div>

      <div id="mainTabs" class="tabs">
        <button class="tab" data-tab="catalog" id="tabCatalog">Catalog</button>
        <button class="tab" data-tab="meta" id="tabMeta">Meta</button>
        <button class="tab" data-tab="inspector" id="tabInspector">Inspector</button>
      </div>

      <div id="pageCatalog" class="page"></div>
      <div id="pageMeta" class="page" style="display:none"></div>
      <div id="pageInspector" class="page" style="display:none"></div>
    </div>
  `;

  function setTab(name) {
    const isCatalog = name === 'catalog';
    const isMeta = name === 'meta';
    const isInspector = name === 'inspector';

    const pc = document.getElementById('pageCatalog');
    const pm = document.getElementById('pageMeta');
    const pi = document.getElementById('pageInspector');

    if (pc) pc.style.display = isCatalog ? 'block' : 'none';
    if (pm) pm.style.display = isMeta ? 'block' : 'none';
    if (pi) pi.style.display = isInspector ? 'block' : 'none';

    const tc = document.getElementById('tabCatalog');
    const tm = document.getElementById('tabMeta');
    const ti = document.getElementById('tabInspector');

    if (tc) tc.disabled = isCatalog;
    if (tm) tm.disabled = isMeta;
    if (ti) ti.disabled = isInspector;

    replaceQuery({ tab: name });
  }

  // wire
  const tabs = host.querySelectorAll('#mainTabs [data-tab]');
  for (const t of tabs) {
    t.addEventListener('click', (ev) => {
      ev.preventDefault();
      const name = t.getAttribute('data-tab');
      if (name) setTab(name);
    });
  }

  // Default tab from URL
  const q = readQuery();
  const initial = (q && q.tab) ? String(q.tab) : 'catalog';
  if (['catalog','meta','inspector'].includes(initial)) setTab(initial);
  else setTab('catalog');

  return { setTab };
}
