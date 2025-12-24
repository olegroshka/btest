// Entry point for the committed, local-first UI bundles.
// This stays framework-free during migration and is split into modules.

import { mountLayout } from './layout.js';
import { mountCatalog } from './catalog.js';
import { mountInspector } from './inspector.js';
import { mountMeta } from './meta.js';

(function start() {
  try {
    mountLayout('app');
    mountCatalog('pageCatalog');
    mountMeta('pageMeta');
    mountInspector('pageInspector');

    // Mark boot complete (used by smoke checks and debugging)
    try {
      const host = document.getElementById('app');
      if (host) host.setAttribute('data-ui-boot', '1');
    } catch (e2) {}
  } catch (e) {
    try {
      // Make failures visible during smoke runs.
      console.error('UI_BOOT_ERROR', e);
      const host = document.getElementById('app');
      if (host) {
        host.textContent = '(failed to start UI)';
        host.setAttribute('data-ui-boot', '0');
      }
    } catch (e2) {}
  }
})();
