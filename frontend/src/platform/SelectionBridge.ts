export type Selection = { lib: string; sym: string };

export function setSelection(sel: Selection) {
  // write legacy inputs used by the existing non-React Inspector/Meta flows
  const pLib = document.getElementById('pLib') as HTMLInputElement | null;
  const pSym = document.getElementById('pSym') as HTMLInputElement | null;
  if (pLib) pLib.value = sel.lib;
  if (pSym) pSym.value = sel.sym;

  try {
    window.dispatchEvent(new CustomEvent('quantdsl:selection', { detail: { lib: sel.lib, sym: sel.sym } }));
  } catch {
    // ignore
  }
}

export function navigateToInspector(sel: Selection) {
  setSelection(sel);

  // Prefer the React shell navigation API if present
  try {
    const w = window as any;
    if (w.workspaceApi && typeof w.workspaceApi.setTab === 'function') {
      w.workspaceApi.setTab('inspector');
      return;
    }
  } catch {
    // ignore
  }

  // Fallback: manipulate query + attempt to click
  try {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', 'inspector');
    url.searchParams.set('lib', sel.lib);
    url.searchParams.set('sym', sel.sym);
    window.history.replaceState({}, '', url.toString());
  } catch {
    // ignore
  }

  try {
    const btn = document.getElementById('tabInspector') as HTMLButtonElement | null;
    if (btn && typeof btn.click === 'function') btn.click();
  } catch {
    // ignore
  }
}
