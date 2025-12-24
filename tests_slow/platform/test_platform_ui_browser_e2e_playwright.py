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
      - Catalog loads and symbol click populates selection

    NOTE: During the UI componentization refactor, Inspector/Preview/Plot flows are
    migrated in later milestones. This E2E test is intentionally scoped to the
    Catalog→Selection flow so we can land incremental changes without breaking
    the entire UI suite.
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

        # Ensure JS is initialized and the Catalog controls exist.
        page.wait_for_selector("#btnCatalog", timeout=10000)
        page.wait_for_selector("#catalog", timeout=10000)

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

            # Switch to Inspector tab and verify it renders
            page.click("#tabInspector")
            page.wait_for_timeout(250)
            page.wait_for_selector("#pageInspector", state="visible", timeout=10000)

            # Preview and ensure outputs render
            page.wait_for_selector("#btnPreview", state="visible", timeout=10000)
            page.click("#btnPreview")

            # Wait for preview summary OR raw json to populate
            page.wait_for_function(
                """() => {
                  const s = document.getElementById('previewSummary');
                  const r = document.getElementById('previewRaw');
                  const st = s ? String(s.innerText || '').trim() : '';
                  const rt = r ? String(r.innerText || '').trim() : '';
                  return st.length > 0 || (rt.startsWith('{') && rt.length > 20);
                }""",
                timeout=30000,
            )

            # Table tab
            page.click("#canvasTabs [data-tab='table']")
            page.wait_for_timeout(150)
            assert page.is_visible("#tableWrap")
            table_html = page.inner_html("#previewTables")
            assert "head" in table_html.lower() or "tail" in table_html.lower()

            # Raw tab
            page.click("#canvasTabs [data-tab='raw']")
            page.wait_for_timeout(150)
            assert page.is_visible("#rawWrap")
            raw_text = page.inner_text("#previewRaw")
            assert raw_text.strip().startswith("{")

            # Plot tab (container exists; Plotly may or may not render depending on dataset)
            page.click("#canvasTabs [data-tab='plot']")
            page.wait_for_timeout(150)
            assert page.is_visible("#plotWrap")
            assert page.is_visible("[data-testid='plotly-chart']")

            # Ensure no console 'error' entries.
            errors = [m for m in console_msgs if m.lower().startswith('error')]
            assert not errors, f"Console errors detected: {errors[-10:]}"
        except Exception:
            _snapshot_on_fail('AFTER_CLICK_SELECTION')
            raise

        # At this milestone we stop here: Catalog→Selection is the preserved flow.

        # Ensure no console 'error' entries.
        errors = [m for m in console_msgs if m.lower().startswith('error')]
        assert not errors, f"Console errors detected: {errors[-10:]}"

        browser.close()
        return

    # ...existing code below is intentionally bypassed during migration...
