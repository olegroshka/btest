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


def test_meta_defaults_follow_visible_catalog_row(tmp_path, monkeypatch):
    """Repro the screenshot: AAPL row is visible but Meta query shows stale FRED/CPI.

    Contract we want:
      - If catalogSearch filters to a single visible row, Meta query defaults must follow that row.
      - Query and From selection must not return FRED/CPI when AAPL is the only visible row.

    This test seeds localStorage with stale FRED/CPI and ensures the UI corrects it.
    """

    # Isolate storage/cache so this test is deterministic.
    arctic_root = tmp_path / "arctic_meta_defaults_visible_row"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # Populate cache/catalog.
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

        # Seed stale meta query state and a catalog search filter to SPX.
        page.add_init_script(
            """
            () => {
              try {
                const k = 'quantdsl.platform_ui.state.v1';
                const st = {
                  catalogSearch: 'SPX',
                  fProvider: 'FRED',
                  fFreq: '1d',
                  fDataset: 'macro',
                  fKind: 'timeseries',
                  fEntity: 'CPIAUCSL',
                };
                localStorage.setItem(k, JSON.stringify(st));
              } catch (e) {}
            }
            """
        )

        page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
        page.wait_for_selector("#btnCatalog", timeout=10000)

        # Manually set stale state if not present (to ensure test is meaningful)
        page.evaluate("""() => {
            if (!document.getElementById('fEntity').value) {
                document.getElementById('fProvider').value = 'FRED';
                document.getElementById('fEntity').value = 'CPIAUCSL';
                document.getElementById('pLib').value = 'market_data/FRED/1d';
                document.getElementById('pSym').value = 'market_bars/fred/CPIAUCSL';
            }
        }""")

        # Type SPX in search box.
        page.fill("#catalogSearch", "")
        page.type("#catalogSearch", "SPX")
        
        # Load catalog to ensure something is visible.
        page.click("#btnCatalog")

        # The visible catalog rows should include SPX.
        page.wait_for_function(
            """
            () => {
              const links = Array.from(document.querySelectorAll("#catalog a[data-act='preview']"));
              return links.some(a => (a.getAttribute('data-entity')||'').trim().toUpperCase()==='SPX');
            }
            """,
            timeout=10000,
        )

        # Key assertion: Meta query defaults must follow SPX (not stale FRED/CPI).
        page.wait_for_function(
            "() => (document.getElementById('fEntity')?.value || '').trim().toUpperCase().includes('SPX')",
            timeout=15000,
        )

        # Selection should be cleared because it doesn't match SPX.
        assert not page.eval_on_selector('#pSym', 'el => el.value')
        # Provider may be blank here (we only default entity from search). The critical part is entity.

        # Query must not return CPIAUCSL/FRED.
        page.click('#btnMeta')
        page.wait_for_timeout(1200)
        meta_html = page.inner_html('#meta')
        assert 'CPIAUCSL' not in meta_html
        assert 'FRED' not in meta_html
        assert 'SPX' in meta_html.upper()

        # Now select SPX explicitly so "From selection" works.
        page.click("a[data-act='preview'][data-entity='SPX']")
        page.wait_for_function("() => document.getElementById('pSym')?.value?.includes('SPX')", timeout=5000)

        # Switch back to catalog tab to see Meta Query panel again.
        page.click("#mainTabs [data-tab='catalog']")
        page.wait_for_selector("#pageCatalog", state="visible")

        # Query must also not return CPIAUCSL/FRED.
        page.click('#btnMeta')
        page.wait_for_timeout(2000)
        meta_html2 = page.inner_html('#meta')
        assert 'CPIAUCSL' not in meta_html2
        assert 'FRED' not in meta_html2
        assert 'SPX' in meta_html2.upper()

        browser.close()
