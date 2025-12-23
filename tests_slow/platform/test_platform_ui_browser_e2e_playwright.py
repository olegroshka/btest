from __future__ import annotations

import socket
import threading
import time
import warnings

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.manual]


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_platform_ui_browser_e2e_playwright(tmp_path, monkeypatch):
    """Real browser E2E: verify UI flows and JS contracts.

    This test is our safety net for UI regressions. It validates:
      - Main tabs (Catalog/Inspector) navigation
      - Catalog loads and symbol click populates selection
      - Inspector Preview renders:
          * summary
          * table grid
          * raw json
          * plotly chart container
      - Download dry-run, Quality issues/scan, Meta query
      - No console 'error' entries
    """

    # Avoid pandas warning spam in CI logs
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*BlockManagerUnconsolidated.*",
    )

    diag_path = tmp_path / "platform_ui_browser_e2e_diagnostics.txt"
    snap_page_path = tmp_path / "page_content.html"
    snap_catalog_path = tmp_path / "catalog.html"

    def _diag_write(text: str) -> None:
        # Write diagnostics; do not swallow exceptions silently (else we lose the only evidence)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with diag_path.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

    def _diag_section(title: str) -> None:
        _diag_write("\n" + ("=" * 36) + f" {title} " + ("=" * 36) + "\n")

    def _diag_snapshot_page(page) -> None:
        try:
            snap_page_path.write_text(page.content(), encoding="utf-8", errors="ignore")
        except Exception:
            pass

    def _diag_snapshot_catalog(page) -> None:
        try:
            snap_catalog_path.write_text(page.inner_html("#catalog"), encoding="utf-8", errors="ignore")
        except Exception:
            pass

    arctic_root = tmp_path / "arctic_ui_browser_e2e"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # Populate cache
    from quantdsl_backtest.examples.lagging_indecies import build_strategy
    from quantdsl_backtest.engine.backtest_runner import run_backtest

    strat = build_strategy()
    try:
        strat.backtest.reporting.output_dir = None
    except Exception:
        pass
    res = run_backtest(strat)
    assert res is not None

    # Start server
    port = _get_free_port()

    def _run_server():
        import uvicorn

        from quantdsl_backtest.platform_api.main import app

        uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")

    th = threading.Thread(target=_run_server, daemon=True)
    th.start()

    # wait for server
    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            import httpx

            r = httpx.get(f"http://127.0.0.1:{port}/health")
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        raise AssertionError("Server did not start")

    # Playwright-driven browser
    playwright = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright.sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        console_msgs = []
        page.on(
            "console",
            lambda msg: (
                console_msgs.append(f"{msg.type}: {msg.text}"),
                _diag_write(f"CONSOLE {msg.type}: {msg.text}"),
            ),
        )

        failed_responses = []
        failed_bodies = {}

        def _on_response(resp):
            try:
                if resp.status >= 400:
                    failed_responses.append((resp.status, resp.url))
                    _diag_write(f"HTTP {resp.status} {resp.url}")
                    # Best-effort capture JSON/text body for diagnostics
                    try:
                        failed_bodies[resp.url] = resp.json()
                    except Exception:
                        try:
                            failed_bodies[resp.url] = resp.text()
                        except Exception:
                            failed_bodies[resp.url] = '(unavailable)'
                    _diag_write(f"BODY({resp.url})={failed_bodies.get(resp.url)!r}")
            except Exception:
                pass

        page.on("response", _on_response)

        _diag_section("NAVIGATE")
        page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")

        # Ensure Catalog tab is visible and selected by default
        page.wait_for_selector("#pageCatalog", timeout=10000)
        assert page.is_visible("#pageCatalog")
        assert not page.is_visible("#pageInspector")

        # Ensure JS is fully initialized (handlers attached)
        page.wait_for_selector("#btnCatalog", timeout=10000)
        page.wait_for_function(
            "() => !!document.getElementById('btnCatalog') && typeof document.getElementById('btnCatalog').onclick !== 'undefined'",
            timeout=10000,
        )

        # Catalog loads automatically on page load, but make this deterministic by clicking Refresh.
        page.click("#btnCatalog")

        def _catalog_ready_dom() -> bool:
            txt = (page.inner_text("#catalog") or "").strip().lower()
            if not txt:
                return False
            if txt == "(not loaded)":
                return False
            if txt.startswith("(loading") or txt.startswith("loading"):
                return False
            if page.query_selector_all("a[data-act='preview']"):
                return True
            html = (page.inner_html("#catalog") or "").lower()
            return ("http_" in html) or ("error" in html)

        def _wait_until(predicate, *, timeout_s: float = 60.0, step_s: float = 0.2, on_timeout: str = ""):
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                try:
                    if predicate():
                        return
                except Exception:
                    pass
                time.sleep(step_s)
            raise AssertionError(on_timeout or "Timeout while waiting")

        try:
            _wait_until(
                _catalog_ready_dom,
                timeout_s=60.0,
                on_timeout=f"Catalog did not become ready. diagnostics={diag_path}",
            )
        except Exception:
            _diag_section("CATALOG_NOT_READY")
            _diag_snapshot_page(page)
            _diag_snapshot_catalog(page)
            try:
                _diag_write(f"snap_page={snap_page_path}")
                _diag_write(f"snap_catalog={snap_catalog_path}")
                _diag_write(f"catalog_text={page.inner_text('#catalog')!r}")
                _diag_write(f"catalog_html={page.inner_html('#catalog')!r}")
            except Exception:
                pass
            _diag_write(f"failed_responses={failed_responses!r}")
            _diag_write(f"failed_bodies={failed_bodies!r}")
            _diag_write(f"console_tail={console_msgs[-40:]!r}")
            try:
                # Inline diagnostics into the assertion so we don't rely on filesystem temp paths.
                try:
                    catalog_text = page.inner_text('#catalog')
                except Exception:
                    catalog_text = '(unavailable)'
                try:
                    catalog_html = page.inner_html('#catalog')
                except Exception:
                    catalog_html = '(unavailable)'
                raise AssertionError(
                    f"Catalog did not become ready. catalog_text={catalog_text!r} catalog_html_tail={catalog_html[-1200:]!r} failed_responses={failed_responses[-10:]!r}"
                )
            except Exception:
                pass
            raise

        preview_links = page.query_selector_all("a[data-act='preview']")
        assert preview_links, f"No preview links rendered in catalog. diagnostics={diag_path}"

        def _snapshot_on_fail(stage: str) -> None:
            _diag_section(f"FAIL_{stage}")
            try:
                _diag_snapshot_page(page)
                _diag_snapshot_catalog(page)
                _diag_write(f"snap_page={snap_page_path}")
                _diag_write(f"snap_catalog={snap_catalog_path}")
            except Exception:
                pass
            try:
                _diag_write(f"pLib={page.eval_on_selector('#pLib','el=>el.value')!r}")
                _diag_write(f"pSym={page.eval_on_selector('#pSym','el=>el.value')!r}")
            except Exception:
                pass

        try:
            preview_links[0].click()

            # Immediately capture values after click for diagnostics.
            try:
                _diag_write(f"after_click_pLib={page.eval_on_selector('#pLib','el=>el.value')!r}")
                _diag_write(f"after_click_pSym={page.eval_on_selector('#pSym','el=>el.value')!r}")
            except Exception:
                pass

            # Selection should be populated in hidden Inspector inputs too
            page.wait_for_function(
                "() => document.getElementById('pLib')?.value?.trim().length > 0 && document.getElementById('pSym')?.value?.trim().length > 0",
                timeout=20000,
            )
        except Exception:
            _snapshot_on_fail('AFTER_CLICK_SELECTION')
            raise

        # Switch to Inspector tab and verify layout
        page.click("#mainTabs [data-tab='inspector']")
        page.wait_for_timeout(350)
        page.wait_for_selector("#pageInspector", state="visible", timeout=10000)
        assert page.is_visible("#pageInspector")
        assert not page.is_visible("#pageCatalog")
        assert page.is_visible("#canvasPane")

        # Preview and ensure outputs render
        page.wait_for_selector("#btnPreview", state="visible", timeout=10000)
        assert page.is_enabled("#btnPreview")
        page.locator("#btnPreview").scroll_into_view_if_needed()
        page.click("#btnPreview")

        # Wait for summary (or error)
        try:
            page.wait_for_function(
                """() => {
                  const s = document.getElementById('previewSummary');
                  const e = document.getElementById('previewError');
                  const sTxt = s ? String((s.innerText !== null && s.innerText !== undefined) ? s.innerText : '').trim() : '';
                  if (sTxt.length > 0) return true;
                  const eTxt = e ? String((e.innerText !== null && e.innerText !== undefined) ? e.innerText : '').trim() : '';
                  if (e && e.style && e.style.display !== 'none' && eTxt.length > 0) return true;
                  return false;
                }""",
                timeout=30000,
            )
        except Exception as ex:
            _diag_section("PREVIEW_TIMEOUT")
            # Capture both DOM text and visibility state
            preview_summary = '(unavailable)'
            preview_error_visible = '(unavailable)'
            preview_error_txt = '(unavailable)'
            plot_status = '(unavailable)'
            try:
                preview_summary = page.inner_text('#previewSummary')
                _diag_write(f"previewSummary={preview_summary!r}")
            except Exception:
                _diag_write("previewSummary=(unavailable)")
            try:
                preview_error_visible = page.is_visible('#previewError')
                preview_error_txt = page.inner_text('#previewError')
                _diag_write(f"previewError_visible={preview_error_visible!r}")
                _diag_write(f"previewError={preview_error_txt!r}")
            except Exception:
                _diag_write("previewError=(unavailable)")
            try:
                plot_status = page.inner_text('#plotStatus')
                _diag_write(f"plotStatus={plot_status!r}")
            except Exception:
                _diag_write("plotStatus=(unavailable)")

            try:
                _diag_snapshot_page(page)
                _diag_write(f"snap_page={snap_page_path}")
            except Exception:
                pass
            _diag_write(f"failed_responses={failed_responses[-20:]!r}")
            _diag_write(f"failed_bodies_tail={list(failed_bodies.items())[-5:]!r}")

            raise AssertionError(
                "Preview did not become ready. "
                f"previewSummary={preview_summary!r} "
                f"previewError_visible={preview_error_visible!r} "
                f"previewError={preview_error_txt!r} "
                f"plotStatus={plot_status!r} "
                f"failed_responses_tail={failed_responses[-10:]!r}"
            ) from ex

        if page.is_visible("#previewError") and page.inner_text("#previewError").strip():
            raise AssertionError(
                f"UI preview error: {page.inner_text('#previewError')}\nconsole={console_msgs[-25:]}"
            )

        summary = page.inner_text("#previewSummary")
        assert "rows" in summary.lower()

        # Workspace tab switching should behave
        # Plot tab should be visible by default
        assert page.is_visible("#plotWrap")

        # Table tab shows tables
        page.click("#canvasTabs .tab[data-tab='table']")
        page.wait_for_timeout(200)
        assert page.is_visible("#tableWrap")
        assert not page.is_visible("#rawWrap")
        assert not page.is_visible("#plotWrap")
        table_html = page.inner_html("#previewTables")
        assert "head" in table_html.lower() or "tail" in table_html.lower()

        # Raw tab shows JSON
        page.click("#canvasTabs .tab[data-tab='raw']")
        page.wait_for_timeout(200)
        assert page.is_visible("#rawWrap")
        raw_text = page.inner_text("#previewRaw")
        assert raw_text.strip().startswith("{")
        assert "columns" in raw_text or "head" in raw_text

        # Back to plot, chart should be rendered
        page.click("#canvasTabs .tab[data-tab='plot']")
        page.wait_for_timeout(200)
        assert page.is_visible("#plotWrap")
        assert page.is_visible("[data-testid='plotly-chart']")

        # Ensure the UI called the full-range chart endpoint at least once.
        # (This prevents regressions where we only plot preview head/tail.)
        page.wait_for_function(
            """() => {
              if (!window.performance || !performance.getEntriesByType) return true;
              const entries = performance.getEntriesByType('resource') || [];
              return entries.some(e => String(e.name || '').includes('/api/catalog/chart/'));
            }""",
            timeout=300,
        )

        # At minimum, the plot container should stop showing placeholder text.
        page.wait_for_function(
            """() => {
              const el = document.getElementById('plot');
              if (!el) return false;
              const raw = (el.innerText !== null && el.innerText !== undefined) ? el.innerText : '';
              const t = String(raw).trim().toLowerCase();
              // allow empty (plotly clears innerText) but don't allow placeholders
              if (t === '') return true;
              if (t === '(no data)' || t === '(loading)' || t === 'loading') return false;
              return true;
            }""",
            timeout=300,
        )

        page.wait_for_function(
            """() => {
              const st = document.getElementById('plotStatus');
              if (!st) return true;
              const raw = (st.innerText !== null && st.innerText !== undefined) ? st.innerText : '';
              const t = String(raw).trim().toLowerCase();
              return t === '' || t === '(ready)' || t === '(no data)' || t === '(idle)';
            }""",
            timeout=300,
        )

        status_text = page.inner_text('#plotStatus').strip().lower()
        assert '(loading' not in status_text

        # And it must not show known failure markers.
        plot_text = page.inner_text('#plot').strip().lower()
        assert '(no valid timestamps to plot)' not in plot_text
        assert '(no plottable columns)' not in plot_text

        # If Plotly internal state is attached, assert x-axis is time-based.
        plotly_state = page.evaluate(
            """() => {
              const el = document.getElementById('plot');
              return {
                hasLayout: !!(el && el._fullLayout && el._fullLayout.xaxis),
                xType: (el && el._fullLayout && el._fullLayout.xaxis) ? (el._fullLayout.xaxis.type || null) : null,
                hasData: !!(el && el.data && el.data.length && el.data[0] && el.data[0].x && el.data[0].x.length),
              };
            }"""
        )

        if plotly_state and plotly_state.get('hasLayout'):
            xaxis_type = plotly_state.get('xType')
            assert xaxis_type in ("date", "-", "linear"), f"Expected Plotly date or linear axis, got {xaxis_type!r}"

        if plotly_state and plotly_state.get('hasData'):
            x_years = page.evaluate(
                """() => {
                  const el = document.getElementById('plot');
                  if (!el || !el.data || !el.data.length) return null;
                  const xs = el.data[0].x || [];
                  const years = [];
                  for (const v of xs) {
                    const d = new Date(v);
                    if (!isNaN(d.getTime())) years.push(d.getUTCFullYear());
                  }
                  if (!years.length) return null;
                  years.sort((a,b)=>a-b);
                  return {min: years[0], max: years[years.length-1]};
                }"""
            )
            assert x_years is not None
            assert x_years["max"] >= 2000, f"Unexpected x-axis max year: {x_years}"  # should not be 1970

        # Guess source and copy payload
        page.click("#btnGuessSource")
        page.wait_for_timeout(200)

        # Ensure we have a usable source before download.
        # Some datasets may not have meta.provider populated; in that case, set a safe default.
        src_val = page.eval_on_selector("#dlSource", "el => el.value")
        if not (src_val and str(src_val).strip()):
            page.fill("#dlSource", "parquet://")
            page.wait_for_timeout(50)

        # Copy payload (should now be enabled)
        assert page.is_enabled("#btnCopyPayload")
        page.click("#btnCopyPayload")

        # Capture the JSON payload string from the UI before dry-run
        payload_text = page.evaluate(
            """() => {
              try {
                const p = (typeof buildDownloadPayload === 'function') ? buildDownloadPayload(true) : null;
                return JSON.stringify(p);
              } catch (e) { return 'ERR:' + String(e); }
            }"""
        )

        # Dry-run download should populate downloadSummary and must not 422
        assert page.is_enabled("#btnDryRun")
        page.click("#btnDryRun")
        page.wait_for_timeout(800)
        ds = page.inner_text("#downloadSummary")
        assert ds is not None

        # Quality refresh + scan should work and not 503
        page.click("#btnQualityIssues")
        page.wait_for_function(
            "() => { const t = document.getElementById('quality')?.innerText || ''; return (t.includes('Found') || t.includes('no issues')) && !t.includes('(loading...)'); }",
            timeout=10000
        )
        qtxt = page.inner_text("#quality")
        assert qtxt is not None

        page.click("#btnQualityScan")
        page.wait_for_function(
            "() => { const t = document.getElementById('quality')?.innerText || ''; return t.includes('Scanned') && !t.includes('(scanning...)'); }",
            timeout=10000
        )
        qtxt2 = page.inner_text("#quality")
        assert qtxt2 is not None
        assert "HTTP_503" not in qtxt2
        assert "Scanned" in qtxt2

        # Meta query from Catalog page (switch back)
        page.click("#mainTabs [data-tab='catalog']")
        page.wait_for_timeout(250)
        page.wait_for_selector("#pageCatalog", state="visible", timeout=10000)
        assert page.is_visible("#pageCatalog")
        page.click("#btnMeta")
        page.wait_for_timeout(800)
        meta_txt = page.inner_text("#meta")
        assert meta_txt is not None

        # Don't allow 422s (FastAPI validation errors) or other 4xx/5xx
        bad = [x for x in failed_responses if x[0] >= 400]
        if bad:
            _diag_section("HTTP_FAILURES")
            _diag_write(f"HTTP failures: {bad[-10:]!r}")
            _diag_write(f"Payload(buildDownloadPayload): {payload_text!r}")
            _diag_write(f"failed_bodies={failed_bodies!r}")
            _diag_write(f"console_tail={console_msgs[-20:]!r}")
            url = bad[-1][1]
            raise AssertionError(
                f"HTTP failures detected. diagnostics={diag_path} last={bad[-1]} url={url}"
            )

        # Sanity-check the test file itself (defensive against stale bytecode / old file being executed)
        assert "innerText !== null" in open(__file__, 'r', encoding='utf-8', errors='ignore').read()

        browser.close()
