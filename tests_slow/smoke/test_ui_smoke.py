from __future__ import annotations

import json
import os
import pathlib
import time

import pytest
from playwright.sync_api import ConsoleMessage, sync_playwright


# Fast local timeouts (UI + API run on localhost). We fail fast and dump diagnostics.
BOOT_TIMEOUT_MS = 6_000
UI_ATTACH_TIMEOUT_MS = 4_000
TAB_TIMEOUT_MS = 5_000
CATALOG_TIMEOUT_MS = 3_000
PANEL_TIMEOUT_MS = 4_000
PLOT_TIMEOUT_MS = 2_000
# Quality UI render should be essentially instant on localhost once the response returns.
QUALITY_TIMEOUT_MS = 2_000


@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.manual
def test_ui_clickthrough_smoke() -> None:
    """Manual smoke test: click through the UI.

    This test is intended to be run against a live server (default http://127.0.0.1:8000/).
    It is marked as 'manual' and skipped by default in automated runs.

    Why the timeouts are aggressive:
    - Everything is localhost.
    - If the UI is broken, waiting 20-30 seconds is wasted developer time.
    """

    return main()


def _snap_text(page, selector: str) -> str | None:
    try:
        return page.inner_text(selector)
    except Exception:
        return None


def _snap_value(page, selector: str) -> str | None:
    try:
        return page.eval_on_selector(selector, "e => e.value")
    except Exception:
        return None


def _read_base_url() -> str:
    root = pathlib.Path(__file__).resolve().parents[2]
    port_file = root / ".platform_ui" / "server.port"
    port = 8000
    try:
        if port_file.exists():
            port = int(port_file.read_text(encoding="utf-8").strip())
    except Exception:
        port = 8000
    return f"http://127.0.0.1:{port}/"


def _wait_until(page, predicate_js: str, *, timeout_ms: int = 1500, label: str = "wait_until") -> None:
    """Fast local wait helper using polling.

    We intentionally avoid page.wait_for_function because this repo has been run
    with different Playwright sync API builds that disagree on its call signature.
    Polling page.evaluate() is small, fast, and reliable for localhost.
    """

    deadline = time.time() + (timeout_ms / 1000.0)
    while True:
        try:
            if bool(page.evaluate(predicate_js)):
                return
        except Exception:
            pass
        if time.time() >= deadline:
            raise TimeoutError(f"{label} timeout after {timeout_ms}ms")
        time.sleep(0.05)


def _wait_catalog_has_rows(page, *, timeout_ms: int = 1500) -> None:
    # Require the new AG Grid catalog (ag-root-wrapper) to be present with at least one row.
    _wait_until(
        page,
        """() => {
          const ag = document.querySelectorAll('.ag-root-wrapper').length > 0;
          const rows = document.querySelectorAll('.ag-center-cols-container .ag-row').length > 0;
          return ag && rows;
        }""",
        timeout_ms=timeout_ms,
    )


def _wait_catalog_has_symbol(page, symbol: str, *, timeout_ms: int = 1500) -> None:
    # Prefer grid row content over #catalog innerText (the legacy container may be empty/slow).
    symbol = str(symbol or "").strip().upper()
    _wait_until(
        page,
        """(sym) => {
          const rows = Array.from(document.querySelectorAll('.ag-center-cols-container .ag-row'));
          if (!rows.length) return false;
          const txt = rows.slice(0, 80).map(r => (r.textContent||'')).join('\n').toUpperCase();
          return txt.includes(sym);
        }""",
        timeout_ms=timeout_ms,
    )


def _wait_selection_populated(page, *, timeout_ms: int = 1500) -> None:
    _wait_until(
        page,
        "() => (document.getElementById('pLib')?.value||'').trim().length>0 && (document.getElementById('pSym')?.value||'').trim().length>0",
        timeout_ms=timeout_ms,
    )


def _wait_plot_request_ok(page, *, timeout_ms: int = 2500) -> None:
    """Wait for the chart API call to complete successfully."""

    # Use expect_response for better typing support.
    with page.expect_response(lambda r: ("/api/catalog/chart/" in (r.url or "")), timeout=timeout_ms) as resp_info:
        pass
    resp = resp_info.value
    if int(resp.status) != 200:
        raise AssertionError(f"Chart request failed: {resp.status} {resp.url}")


def _wait_plot_ready(page, *, timeout_ms: int = 2500) -> None:
    """Wait until Preview completed.

    Strategy (fast & reliable for localhost):
    1) If UI sets #plotStatus to 'ready', accept.
    2) Otherwise ensure the plot container exists and is visible.
    3) Accept if Plotly created a plot div OR svg nodes exist.

    NOTE: Plotly DOM creation can be flaky in headless; for the smoke test we also
    wait for the /api/catalog/chart response before this check.
    """

    _wait_until(
        page,
        """() => {
          const st = (document.getElementById('plotStatus')?.textContent||'').toLowerCase();
          if (st.includes('ready')) return true;
          const el = document.getElementById('plot');
          if (!el) return false;
          // must be displayed
          const style = window.getComputedStyle(el);
          if (style && style.display === 'none') return false;
          if (el.querySelectorAll('.js-plotly-plot').length > 0) return true;
          return el.querySelectorAll('svg').length > 0;
        }""",
        timeout_ms=timeout_ms,
    )


def _wait_meta_summary(page, *, timeout_ms: int = 1500) -> None:
    _wait_until(
        page,
        "() => (document.getElementById('metaSummary')?.innerText||'').toLowerCase().includes('count')",
        timeout_ms=timeout_ms,
    )


def _click_first_catalog_symbol(page) -> None:
    page.locator(".ag-center-cols-container a").first.click()


def _wait_catalog_filter_menu_available(page, *, timeout_ms: int = 1500) -> None:
    _wait_until(
        page,
        """() => {
          const h = document.querySelector('.ag-header');
          return !!h;
        }""",
        timeout_ms=timeout_ms,
    )


def _wait_inspector_panels(page, *, timeout_ms: int = PANEL_TIMEOUT_MS) -> None:
    _wait_until(
        page,
        """() => {
          // New React inspector controls
          const btnPreview = document.getElementById('btnPreview');
          const btnScan = document.getElementById('btnQualityScan');
          const btnIssues = document.getElementById('btnQualityIssues');
          const plot = document.getElementById('plot');
          const quality = document.getElementById('quality');
          const newOk = !!btnPreview && !!btnScan && !!btnIssues && !!plot && !!quality;

          // Legacy layout (note: 'workspace' isn't present in current builds)
          const dl = document.getElementById('downloadPanel');
          const ql = document.getElementById('qualityPanel');
          const legacyOk = !!dl && !!ql;

          return newOk || legacyOk;
        }""",
        timeout_ms=timeout_ms,
        label="inspector_panels",
    )


def _goto_tab(page, name: str) -> None:
    # The UI exposes a supported API for tab switching.
    # Using it is more reliable than manually dispatching popstate events.
    page.evaluate(
        """(name) => {
          const w = window;
          if (w.workspaceApi && typeof w.workspaceApi.setTab === 'function') {
            w.workspaceApi.setTab(String(name));
            return;
          }
          // Fallback: best-effort URL update.
          const u = new URL(window.location.href);
          u.searchParams.set('tab', String(name));
          window.history.replaceState({}, '', u.toString());
        }""",
        name,
    )


def _wait_tab_visible(page, tab: str, *, timeout_ms: int = TAB_TIMEOUT_MS) -> None:
    """Wait until the requested tab is active and its page container is visible.

    We avoid page.wait_for_function arg signature differences by polling via evaluate.
    """

    tab = str(tab)
    deadline = time.time() + (timeout_ms / 1000.0)

    while True:
        ok = page.evaluate(
            """(tab) => {
              const cur = new URL(window.location.href).searchParams.get('tab') || '';
              if (cur !== tab) return false;
              const id = tab === 'catalog' ? 'pageCatalog' : tab === 'meta' ? 'pageMeta' : 'pageInspector';
              const el = document.getElementById(id);
              if (!el) return false;
              const style = window.getComputedStyle(el);
              return !!(style && style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null);
            }""",
            tab,
        )
        if ok:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"tab '{tab}' did not become visible within {timeout_ms}ms")
        time.sleep(0.05)


def _checkpoint(out: dict[str, object], page, console, page_errors, failed_requests, responses_4xx_5xx, all_requests) -> None:
    # Keep last tails to help local debugging when we fail fast.
    out["console_tail"] = console[-80:]
    out["page_errors"] = page_errors[-50:]
    out["failed_requests_tail"] = failed_requests[-50:]
    out["responses_4xx_5xx_tail"] = responses_4xx_5xx[-50:]
    out["all_requests_tail"] = all_requests[-500:]
    try:
        out["url"] = page.url
    except Exception:
        pass


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _timeout_ms(name: str, default: int) -> int:
    """Allow fast iteration by overriding timeouts via env vars."""

    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


# (Place these near other _wait_* helpers, before main())
# --- Quality helpers ---

def _wait_quality_scan_ok(page, *, timeout_ms: int = 8000) -> None:
    """Wait for /api/quality/scan to complete successfully."""

    with page.expect_response(lambda r: ("/api/quality/scan" in (r.url or "")), timeout=timeout_ms) as resp_info:
        pass
    resp = resp_info.value
    if int(resp.status) != 200:
        raise AssertionError(f"Quality scan request failed: {resp.status} {resp.url}")


def _wait_quality_rendered(page, *, timeout_ms: int = QUALITY_TIMEOUT_MS) -> None:
    """Wait until the UI rendered quality results.

    The UI may render a <table>, an AG Grid, or a simple text summary.
    """

    _wait_until(
        page,
        """() => {
          const root = document.getElementById('quality');
          if (!root) return false;
          // Any table
          if (root.querySelectorAll('table').length > 0) return true;
          // AG Grid
          if (root.querySelectorAll('.ag-root-wrapper').length > 0) return true;
          // Any non-empty text indicates some render happened
          const txt = String(root.textContent || '').trim();
          return txt.length > 0 && !txt.toLowerCase().includes('loading');
        }""",
        timeout_ms=timeout_ms,
        label="quality_render",
    )


def main() -> None:
    out: dict[str, object] = {
        "steps": [],
        "errors": [],
        "console_tail": [],
        "failed_requests_tail": [],
        "page_errors": [],
        "responses_4xx_5xx_tail": [],
        "all_requests_tail": [],
        "timing_ms": {},
    }

    timing_ms: dict[str, int] = {}
    out["timing_ms"] = timing_ms

    base_url = _read_base_url()

    t0_all = time.perf_counter()

    ok = False
    try:
        with sync_playwright() as p:
            headless = _env_flag("UI_HEADLESS", True)
            slow_mo = _env_int("UI_SLOW_MO_MS", 0)

            browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)
            page = browser.new_page()

            # Optional: set a larger viewport in headed mode for easier debugging.
            if not headless:
                page.set_viewport_size({"width": 1280, "height": 1024})

            console: list[dict[str, str]] = []
            page_errors: list[str] = []
            failed_requests: list[dict[str, str]] = []
            responses_4xx_5xx: list[dict[str, object]] = []
            all_requests: list[str] = []

            def _on_console(msg: ConsoleMessage) -> None:
                try:
                    console.append({"type": str(msg.type), "text": str(msg.text)})
                except Exception:
                    pass

            page.on("console", _on_console)

            def _on_page_error(exc) -> None:
                try:
                    page_errors.append(str(exc))
                except Exception:
                    pass

            def _on_request_failed(req) -> None:
                try:
                    failed_requests.append({
                        "url": str(req.url),
                        "failure": str(req.failure),
                        "method": str(req.method),
                    })
                except Exception:
                    pass

            def _on_response(resp) -> None:
                try:
                    status = int(resp.status)
                    if status >= 400:
                        responses_4xx_5xx.append({
                            "url": str(resp.url),
                            "status": status,
                            "content_type": str(resp.headers.get('content-type')),
                        })
                except Exception:
                    pass

            def _on_request(req) -> None:
                try:
                    all_requests.append(str(req.url))
                except Exception:
                    pass

            page.on("pageerror", _on_page_error)
            page.on("requestfailed", _on_request_failed)
            page.on("response", _on_response)
            page.on("request", _on_request)

            t_nav = time.perf_counter()
            page.goto(base_url, wait_until="domcontentloaded")
            timing_ms["goto_domcontentloaded"] = int((time.perf_counter() - t_nav) * 1000)

            # Harden: UI must actually boot (not stuck on loading shell)
            try:
                t_boot = time.perf_counter()
                _wait_until(
                    page,
                    """() => {
                      const host = document.getElementById('app');
                      if (!host) return false;
                      const boot = host.getAttribute('data-ui-boot');
                      if (boot === '1') return true;
                      const txt = String(host.textContent || '').trim().toLowerCase();
                      return !!txt && !txt.includes('loading platform ui');
                    }""",
                    timeout_ms=BOOT_TIMEOUT_MS,
                )
                timing_ms["boot"] = int((time.perf_counter() - t_boot) * 1000)
            except Exception as e:
                out["errors"] = [f"UI did not boot (still loading). {e!r}"]
                out["page_html_tail"] = page.content()[-4000:]
                _checkpoint(out, page, console, page_errors, failed_requests, responses_4xx_5xx, all_requests)
                raise

            # Now the UI should have mounted real controls AND tabs.
            try:
                t_attach = time.perf_counter()
                page.wait_for_selector("#btnCatalog", state="attached", timeout=UI_ATTACH_TIMEOUT_MS)
                page.wait_for_selector("#tabCatalog", state="attached", timeout=UI_ATTACH_TIMEOUT_MS)
                page.wait_for_selector("#tabInspector", state="attached", timeout=UI_ATTACH_TIMEOUT_MS)
                timing_ms["ui_attach"] = int((time.perf_counter() - t_attach) * 1000)
            except Exception as e:
                out["errors"] = [f"UI mounted but required controls not attached. {e!r}"]
                try:
                    out["app_html_tail"] = page.inner_html("#app")[-4000:]
                except Exception:
                    pass
                out["page_html_tail"] = page.content()[-4000:]
                _checkpoint(out, page, console, page_errors, failed_requests, responses_4xx_5xx, all_requests)
                raise

            # Harden: catalog grid must render quickly.
            try:
                t_cat = time.perf_counter()
                _wait_catalog_has_rows(page, timeout_ms=CATALOG_TIMEOUT_MS)
                timing_ms["catalog_rows"] = int((time.perf_counter() - t_cat) * 1000)
            except Exception as e:
                out["errors"] = [f"Catalog did not render rows/grid. {e!r}"]
                try:
                    out["app_html_tail"] = page.inner_html("#app")[-4000:]
                except Exception:
                    pass
                _checkpoint(out, page, console, page_errors, failed_requests, responses_4xx_5xx, all_requests)
                raise

            steps: list[dict[str, object]] = out["steps"]  # type: ignore[assignment]

            steps.append(
                {
                    "name": "loaded",
                    "title": page.title(),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            # --- Minimal, high-signal contract flow ---
            # 1) Refresh catalog → rows exist
            steps.append({"name": "before_catalog_refresh"})
            page.click("#btnCatalog")
            _wait_catalog_has_rows(page, timeout_ms=CATALOG_TIMEOUT_MS)

            steps.append(
                {
                    "name": "after_catalog",
                    "catalog_preview_links": len(page.query_selector_all(".ag-center-cols-container a")),
                }
            )

            # 2) Select first row → Inspector visible + selection populated
            steps.append({"name": "before_first_select"})
            _click_first_catalog_symbol(page)
            _wait_selection_populated(page, timeout_ms=2000)
            page.wait_for_selector("#pageInspector", state="visible", timeout=TAB_TIMEOUT_MS)

            steps.append({"name": "after_first_select_values", "pLib": _snap_value(page, "#pLib"), "pSym": _snap_value(page, "#pSym")})

            # 3) Preview → plot ready
            steps.append({"name": "before_preview_click"})
            page.click("#btnPreview")
            # First wait for the chart request to succeed.
            _wait_plot_request_ok(page, timeout_ms=8000)
            # Then quickly wait for UI render markers.
            _wait_plot_ready(page, timeout_ms=PLOT_TIMEOUT_MS)
            assert page.is_visible("#plot")
            assert "error" not in (page.inner_text("#plotStatus") or "").lower()

            # 4) Meta query works (uses current selection)
            steps.append({"name": "before_meta_query"})
            _goto_tab(page, "meta")
            _wait_tab_visible(page, "meta")
            page.click("#btnMetaQuery")
            _wait_meta_summary(page, timeout_ms=1500)
            assert "count" in (page.inner_text("#metaSummary") or "").lower()

            # 5) Back to inspector → Download/Quality panels exist, quality scan renders
            steps.append({"name": "before_quality_scan"})
            _goto_tab(page, "inspector")
            _wait_tab_visible(page, "inspector")
            _wait_inspector_panels(page, timeout_ms=PANEL_TIMEOUT_MS)

            page.click("#btnQualityScan")
            _wait_quality_scan_ok(page, timeout_ms=15_000)
            _wait_quality_rendered(page, timeout_ms=QUALITY_TIMEOUT_MS)

            # Optional: issues view should render *something*
            page.click("#btnQualityIssues")
            _wait_until(page, "() => document.getElementById('quality') !== null", timeout_ms=800)
            assert page.is_visible("#quality")

            timing_ms["total"] = int((time.perf_counter() - t0_all) * 1000)

            _checkpoint(out, page, console, page_errors, failed_requests, responses_4xx_5xx, all_requests)

            ok = True
            browser.close()

    except Exception as e:
        # Ensure we capture state even on very fast failures.
        try:
            _checkpoint(out, page, console, page_errors, failed_requests, responses_4xx_5xx, all_requests)  # type: ignore[name-defined]
        except Exception:
            pass
        errs: list[str] = out["errors"]  # type: ignore[assignment]
        errs.append(repr(e))
    finally:
        # Mark outcome and avoid stale/previous run errors lingering.
        out["ok"] = bool(ok)
        if ok:
            out["errors"] = []
        with open("ui_clickthrough_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    # If we recorded any errors, fail the test so CI/dev sees it.
    if not out.get("ok") or out.get("errors"):
        raise AssertionError(json.dumps({"ok": out.get("ok"), "errors": out.get("errors"), "url": out.get("url")}, indent=2))
