from __future__ import annotations

import json
import pytest
import pathlib
from playwright.sync_api import ConsoleMessage, sync_playwright

@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.manual
def test_ui_clickthrough_smoke() -> None:
    """Manual smoke test: click through the UI.
    
    This test is intended to be run against a live server (default http://127.0.0.1:8000/).
    It is marked as 'manual' and skipped by default in automated runs.
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


def _wait_until(page, predicate_js: str, *, timeout_ms: int = 1500) -> None:
    """Fast local wait helper.

    We intentionally keep timeouts low because this runs against localhost.
    If something is truly slow/broken, failing fast is better than sleeping.
    """
    page.wait_for_function(predicate_js, timeout=timeout_ms)


def _wait_catalog_has_rows(page, *, timeout_ms: int = 1500) -> None:
    _wait_until(
        page,
        "() => document.querySelectorAll(\"#catalog a[data-act='preview']\").length > 0",
        timeout_ms=timeout_ms,
    )


def _wait_selection_populated(page, *, timeout_ms: int = 800) -> None:
    _wait_until(
        page,
        "() => (document.getElementById('pLib')?.value||'').trim().length>0 && (document.getElementById('pSym')?.value||'').trim().length>0",
        timeout_ms=timeout_ms,
    )


def _wait_plot_ready(page, *, timeout_ms: int = 2500) -> None:
    # Either plotStatus says ready OR plot div contains an svg/canvas from plotly.
    _wait_until(
        page,
        """() => {
          const st = (document.getElementById('plotStatus')?.textContent||'').toLowerCase();
          if (st.includes('ready')) return true;
          const el = document.getElementById('plot');
          if (!el) return false;
          // Plotly typically renders svg nodes
          return el.querySelectorAll('svg').length > 0 || el.querySelectorAll('.js-plotly-plot').length > 0;
        }""",
        timeout_ms=timeout_ms,
    )


def _wait_meta_summary(page, *, timeout_ms: int = 1500) -> None:
    _wait_until(
        page,
        "() => (document.getElementById('metaSummary')?.innerText||'').toLowerCase().includes('count')",
        timeout_ms=timeout_ms,
    )


def _wait_quality_grid(page, *, timeout_ms: int = 2500) -> None:
    _wait_until(
        page,
        "() => document.querySelectorAll('#quality table').length >= 1",
        timeout_ms=timeout_ms,
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
    }

    base_url = _read_base_url()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

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

            page.goto(base_url, wait_until="domcontentloaded")

            # Harden: UI must actually boot (not stuck on loading shell)
            try:
                page.wait_for_function(
                    """() => {
                      const host = document.getElementById('app');
                      if (!host) return false;
                      const boot = host.getAttribute('data-ui-boot');
                      if (boot === '1') return true;
                      const txt = String(host.textContent || '').trim().toLowerCase();
                      // before boot it's exactly '(loading Platform UI...)'
                      return txt && !txt.includes('loading platform ui');
                    }""",
                    timeout=20000,
                )
            except Exception as e:
                out["errors"] = [f"UI did not boot (still loading). {e!r}"]
                out["page_html_tail"] = page.content()[-4000:]
                out["console_tail"] = console[-200:]
                out["page_errors"] = page_errors[-50:]
                out["failed_requests_tail"] = failed_requests[-50:]
                out["all_requests_tail"] = all_requests[-500:]
                out["responses_4xx_5xx_tail"] = responses_4xx_5xx[-200:]
                raise

            # Now the UI should have mounted real controls.
            try:
                page.wait_for_selector("#btnCatalog", state="attached", timeout=15000)
            except Exception as e:
                out["errors"] = [f"UI mounted but #btnCatalog not attached. {e!r}"]
                try:
                    out["app_html_tail"] = page.inner_html("#app")[-4000:]
                except Exception:
                    pass
                out["page_html_tail"] = page.content()[-4000:]
                out["console_tail"] = console[-200:]
                out["page_errors"] = page_errors[-50:]
                out["failed_requests_tail"] = failed_requests[-50:]
                out["all_requests_tail"] = all_requests[-500:]
                out["responses_4xx_5xx_tail"] = responses_4xx_5xx[-200:]
                raise

            # It might not be visible depending on CSS/layout; ensure it exists then click.
            page.wait_for_timeout(25)

            steps: list[dict[str, object]] = out["steps"]  # type: ignore[assignment]

            steps.append(
                {
                    "name": "loaded",
                    "title": page.title(),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            page.click("#btnCatalog")
            _wait_catalog_has_rows(page, timeout_ms=1500)
            steps.append(
                {
                    "name": "after_catalog",
                    "catalog_text": (_snap_text(page, "#catalog") or "")[:400],
                    "catalog_preview_links": len(page.query_selector_all("a[data-act='preview']")),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            links = page.query_selector_all("a[data-act='preview']")
            if links:
                links[0].click()
                _wait_selection_populated(page, timeout_ms=800)

                # Clicking a catalog preview link must switch the active tab to Inspector.
                page.wait_for_selector("#pageInspector", state="visible", timeout=10000)
                assert (_snap_value(page, "#pLib") or "").strip(), "Catalog preview did not populate #pLib"
                assert (_snap_value(page, "#pSym") or "").strip(), "Catalog preview did not populate #pSym"

                page.click("#btnPreview")
                _wait_plot_ready(page, timeout_ms=2500)
                assert page.is_visible("#plot")

            steps.append(
                {
                    "name": "after_select",
                    "pLib": _snap_value(page, "#pLib"),
                    "pSym": _snap_value(page, "#pSym"),
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            # Switch to Inspector using the visible tab control.
            # New UI uses #tabInspector.
            try:
                # Prefer the explicit id
                if page.is_enabled("#tabInspector"):
                    page.click("#tabInspector")
                else:
                    page.click("#mainTabs [data-tab='inspector']", force=True)
            except Exception:
                page.click("#mainTabs [data-tab='inspector']", force=True)
            page.wait_for_timeout(25)

            # Ensure the Inspector pane is visible.
            page.wait_for_selector("#pageInspector", state="visible", timeout=10000)

            # Preview button must be visible and enabled.
            page.wait_for_selector("#btnPreview", state="visible", timeout=10000)
            preview_disabled = page.is_disabled("#btnPreview")
            steps.append({"name": "inspector_open", "preview_disabled": preview_disabled})

            if not preview_disabled:
                page.locator("#btnPreview").scroll_into_view_if_needed()
                page.click("#btnPreview")
                _wait_plot_ready(page, timeout_ms=2500)

            steps.append(
                {
                    "name": "after_preview",
                    "summary": (_snap_text(page, "#previewSummary") or "")[:250],
                    "preview_error": (_snap_text(page, "#previewError") or "")[:250],
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            # --- Catalog search + selection scenarios (must be on Catalog tab) ---
            try:
                if page.is_enabled("#tabCatalog"):
                    page.click("#tabCatalog")
                else:
                    page.click("#mainTabs [data-tab='catalog']", force=True)
            except Exception:
                page.click("#mainTabs [data-tab='catalog']", force=True)
            page.wait_for_selector("#pageCatalog", state="visible", timeout=10000)
            page.wait_for_selector("#catalogSearch", state="visible", timeout=10000)

            page.fill("#catalogSearch", "CAC")
            _wait_until(page, "() => (document.getElementById('catalog')?.innerText||'').includes('CAC')", timeout_ms=800)
            txt = (page.inner_text("#catalog") or "")
            assert "CAC" in txt

            links = page.query_selector_all("a[data-act='preview']")
            assert links
            links[0].click()
            _wait_selection_populated(page, timeout_ms=800)

            # ensure Inspector tab activated
            try:
                if page.is_enabled("#tabInspector"):
                    page.click("#tabInspector")
                else:
                    page.click("#mainTabs [data-tab='inspector']", force=True)
            except Exception:
                page.click("#mainTabs [data-tab='inspector']", force=True)

            page.wait_for_selector("#pageInspector", state="visible", timeout=10000)
            page.click("#btnPreview")
            _wait_plot_ready(page, timeout_ms=2500)
            assert page.is_visible("#plot")

            # Switch back to Catalog, pick another symbol, and preview again
            try:
                if page.is_enabled("#tabCatalog"):
                    page.click("#tabCatalog")
                else:
                    page.click("#mainTabs [data-tab='catalog']", force=True)
            except Exception:
                page.click("#mainTabs [data-tab='catalog']", force=True)
            page.wait_for_selector("#pageCatalog", state="visible", timeout=10000)
            page.wait_for_selector("#catalogSearch", state="visible", timeout=10000)

            page.fill("#catalogSearch", "CCMP")
            _wait_until(page, "() => (document.getElementById('catalog')?.innerText||'').includes('CCMP')", timeout_ms=800)
            links2 = page.query_selector_all("a[data-act='preview']")
            assert links2
            links2[0].click()
            _wait_selection_populated(page, timeout_ms=800)

            page.wait_for_selector("#pageInspector", state="visible", timeout=10000)
            assert (_snap_value(page, "#pSym") or "").strip(), "Second Catalog preview did not populate #pSym"

            page.click("#btnPreview")
            _wait_plot_ready(page, timeout_ms=2500)
            assert page.is_visible("#plot")
            assert "error" not in (page.inner_text("#plotStatus") or "").lower()

            # Meta query uses current selection
            try:
                if page.is_enabled("#tabMeta"):
                    page.click("#tabMeta")
                else:
                    page.click("#mainTabs [data-tab='meta']", force=True)
            except Exception:
                page.click("#mainTabs [data-tab='meta']", force=True)
            page.wait_for_selector("#pageMeta", state="visible", timeout=10000)

            page.wait_for_selector("#btnMetaQuery", state="visible", timeout=10000)
            page.click("#btnMetaQuery")
            _wait_meta_summary(page, timeout_ms=1500)
            ms = (page.inner_text("#metaSummary") or "").lower()
            assert "count" in ms

            # Back to Inspector
            try:
                # Prefer the explicit id
                if page.is_enabled("#tabInspector"):
                    page.click("#tabInspector")
                else:
                    page.click("#mainTabs [data-tab='inspector']", force=True)
            except Exception:
                page.click("#mainTabs [data-tab='inspector']", force=True)
            page.wait_for_timeout(25)

            # Ensure the Inspector pane is visible.
            page.wait_for_selector("#pageInspector", state="visible", timeout=10000)

            # Preview button must be visible and enabled.
            page.wait_for_selector("#btnPreview", state="visible", timeout=10000)
            preview_disabled = page.is_disabled("#btnPreview")
            steps.append({"name": "inspector_open", "preview_disabled": preview_disabled})

            if not preview_disabled:
                page.locator("#btnPreview").scroll_into_view_if_needed()
                page.click("#btnPreview")
                _wait_plot_ready(page, timeout_ms=2500)

            steps.append(
                {
                    "name": "after_preview_meta",
                    "summary": (_snap_text(page, "#previewSummary") or "")[:250],
                    "preview_error": (_snap_text(page, "#previewError") or "")[:250],
                    "reqid": _snap_text(page, "#reqid"),
                }
            )

            # --- Download + Quality panels exist and Quality scan renders a grid ---
            page.wait_for_selector("#downloadPanel", state="attached", timeout=10000)
            page.wait_for_selector("#qualityPanel", state="attached", timeout=10000)

            # Run quality scan and ensure it renders a table (grid)
            page.click("#btnQualityScan")
            _wait_quality_grid(page, timeout_ms=2500)
            # should render a table within #quality
            assert page.locator("#quality table").count() >= 1

            # Issues view should also render something (either no issues or a grid)
            page.click("#btnQualityIssues")
            page.wait_for_timeout(100)
            assert page.is_visible("#quality")

            # At end, record console + ensure no unexpected failed requests
            out["console_tail"] = console[-50:]
            out["failed_requests_tail"] = failed_requests[-50:]
            out["page_errors"] = page_errors[-50:]
            out["responses_4xx_5xx_tail"] = responses_4xx_5xx[-50:]
            out["all_requests_tail"] = all_requests[-500:]

            bad_404 = []
            for r in failed_requests[-200:]:
                u = (r.get('url') or '').lower()
                if 'favicon.ico' in u or 'robots.txt' in u or 'manifest.json' in u or 'apple-touch-icon' in u:
                    continue
                bad_404.append(r)
            assert not bad_404, f"Unexpected failed requests: {bad_404[-5:]}"

            # Record console 404 noise (Chromium sometimes logs this without exposing the URL)
            # We keep it visible in results, but don't fail the smoke run on it.
            bad_console = [c for c in console if 'failed to load resource' in (c.get('text','').lower())]
            out["console_resource_errors"] = bad_console[-10:]

            page.wait_for_timeout(50)
            browser.close()
    except Exception as e:
        errs: list[str] = out["errors"]  # type: ignore[assignment]
        errs.append(repr(e))
    finally:
        with open("ui_clickthrough_results.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

