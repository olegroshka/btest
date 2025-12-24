// Tiny UI state helpers: querystring deep-links + localStorage sticky defaults.

const LS_KEY = 'quantdsl.platform_ui.state.v1';

export function getUiState() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return {};
    const obj = JSON.parse(raw);
    return obj && typeof obj === 'object' ? obj : {};
  } catch (e) {
    return {};
  }
}

export function patchUiState(patch) {
  try {
    const st = getUiState();
    const next = { ...st, ...(patch || {}) };
    localStorage.setItem(LS_KEY, JSON.stringify(next));
    return next;
  } catch (e) {
    return {};
  }
}

export function readQuery() {
  const q = new URLSearchParams(window.location.search || '');
  const obj = {};
  for (const [k, v] of q.entries()) obj[k] = v;
  return obj;
}

export function replaceQuery(patch) {
  const cur = new URLSearchParams(window.location.search || '');
  for (const [k, v] of Object.entries(patch || {})) {
    if (v === null || v === undefined || String(v).trim() === '') cur.delete(k);
    else cur.set(k, String(v));
  }
  const next = cur.toString();
  const url = next ? `${window.location.pathname}?${next}` : window.location.pathname;
  try {
    history.replaceState({}, '', url);
  } catch (e) {}
}

