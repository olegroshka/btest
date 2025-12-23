from __future__ import annotations

import socket
import threading
import time

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.manual]


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_meta_query_matches_current_catalog_selection(tmp_path, monkeypatch):
    """Repro + guard: after selecting SPX, Query/From selection must not return unrelated CPI/FRED.

    We specifically reproduce the bug where stale localStorage or mismatched filter inputs cause Meta query
    to keep returning FRED/CPI even while SPX is visible/selected.
    """

    # Isolate storage/cache so this test is deterministic.
    arctic_root = tmp_path / "arctic_meta_query_sel"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # Populate a minimal cache/catalog.
    from quantdsl_backtest.examples.lagging_indecies import build_strategy
    from quantdsl_backtest.engine.backtest_runner import run_backtest

    strat = build_strategy()
    try:
        setattr(strat.backtest.reporting, "output_dir", None)
    except Exception:
        pass
    res = run_backtest(strat)
    assert res is not None

    port = _get_free_port()

    def _run_server():
        import uvicorn

        from quantdsl_backtest.platform_api.main import app

        uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")

    th = threading.Thread(target=_run_server, daemon=True)
    th.start()

    # Wait for server
    t0 = time.time()
    while time.time() - t0 < 15:
        try:
            import httpx

            r = httpx.get(f"http://127.0.0.1:{port}/health")
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        raise AssertionError("Server did not start")

    playwright = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright.sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Seed misleading/stale selection to reproduce the issue.
        page.add_init_script(
            """
            () => {
              try {
                const k = 'quantdsl.platform_ui.state.v1';
                const st = {
                  // user previously selected FRED/CPI
                  pLib: 'market_data/FRED/1d',
                  pSym: 'market_bars/fred/CPIAUCSL',
                  fProvider: 'FRED',
                  fEntity: 'CPIAUCSL',
                  lastSelectedLibrary: 'market_data/FRED/1d',
                  lastSelectedSymbol: 'market_bars/fred/CPIAUCSL',
                };
                localStorage.setItem(k, JSON.stringify(st));
              } catch (e) {}
            }
            """
        )

        page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        page.wait_for_selector("#btnCatalog", timeout=10000)

        # Ensure we have a stale state. If not restored, set it manually.
        current_entity = page.eval_on_selector('#fEntity', 'el => el.value')
        if current_entity != 'CPIAUCSL':
            page.evaluate("""() => {
                document.getElementById('pLib').value = 'market_data/FRED/1d';
                document.getElementById('pSym').value = 'market_bars/fred/CPIAUCSL';
                document.getElementById('fProvider').value = 'FRED';
                document.getElementById('fEntity').value = 'CPIAUCSL';
                document.getElementById('pLib').dispatchEvent(new Event('input'));
                document.getElementById('pSym').dispatchEvent(new Event('input'));
                document.getElementById('fEntity').dispatchEvent(new Event('input'));
            }""")
            
        # Type SPX in search.
        page.fill("#catalogSearch", "SPX")
        page.eval_on_selector("#catalogSearch", "el => el.dispatchEvent(new Event('input'))")
            
        # Selection should be cleared because it doesn't match SPX.
        page.wait_for_function("() => !document.getElementById('pSym')?.value", timeout=5000)
        
        # fEntity should follow search.
        page.wait_for_function("() => document.getElementById('fEntity')?.value === 'SPX'", timeout=5000)
        
        # Click Query. It should NOT revert to FRED.
        page.click("#btnMeta")
        # Give it some time to load and render
        page.wait_for_selector("#meta .table-wrap, #meta .muted", timeout=10000)
        
        meta_html = page.inner_html("#meta")
        
        assert "CPIAUCSL" not in meta_html, "Query reverted to stale FRED even though we searched for SPX"
        assert "SPX" in meta_html, "Query did not find SPX"
        
        # Provider should still be empty or SPX (not FRED).
        assert page.eval_on_selector('#fProvider', 'el => el.value').upper() != 'FRED'

        # Now select SPX explicitly.
        # Find the SPX row in the catalog.
        page.wait_for_selector("a[data-act='preview'][data-entity='SPX']", state="attached")
        page.click("a[data-act='preview'][data-entity='SPX']")
        page.wait_for_function("() => document.getElementById('pSym')?.value?.includes('SPX')", timeout=5000)

        # Switch back to catalog tab to see Meta Query panel again.
        page.click("#mainTabs [data-tab='catalog']")
        page.wait_for_selector("#pageCatalog", state="visible")

        # Query should now work and return SPX (or at least NOT FRED/CPI).
        page.click("#btnMeta")
        page.wait_for_timeout(2000)
        meta_html2 = page.inner_html("#meta")

        assert "CPIAUCSL" not in meta_html2
        assert "FRED" not in meta_html2
        # In this environment, it might return several rows including CAC if filter is loose, 
        # but it definitely shouldn't be FRED.
        assert "PARQUET" in meta_html2 or "SPX" in meta_html2

        browser.close()

