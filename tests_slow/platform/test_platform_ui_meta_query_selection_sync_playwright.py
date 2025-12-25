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

        # Ensure Meta panel is mounted so its inputs exist.
        page.click("#tabMeta")
        page.wait_for_selector("#pageMeta", state="visible", timeout=10000)

        # Ensure we have a stale state. If not restored, set it manually.
        current_entity = page.eval_on_selector("#mEntity", "el => el.value")
        if current_entity != "CPIAUCSL":
            page.evaluate(
                """() => {
                document.getElementById('pLib').value = 'market_data/FRED/1d';
                document.getElementById('pSym').value = 'market_bars/fred/CPIAUCSL';
                document.getElementById('mProvider').value = 'FRED';
                document.getElementById('mEntity').value = 'CPIAUCSL';
                document.getElementById('pLib').dispatchEvent(new Event('input'));
                document.getElementById('pSym').dispatchEvent(new Event('input'));
                document.getElementById('mEntity').dispatchEvent(new Event('input'));
            }"""
            )

        # Type SPX in search (must be on Catalog tab for the input to be visible/editable).
        page.click("#tabCatalog")
        page.wait_for_selector("#catalogSearch", state="visible", timeout=30000)
        page.fill("#catalogSearch", "SPX")
        page.eval_on_selector("#catalogSearch", "el => el.dispatchEvent(new Event('input'))")

        # Ensure catalog data is loaded (search operates on loaded rows).
        page.click("#btnCatalog")
        page.wait_for_function(
            "() => document.querySelectorAll(\"#catalog a[data-act='preview']\").length > 0",
            timeout=10000,
        )

        # Contract: search clears selection so stale lib/sym can't contaminate Meta.
        page.wait_for_function(
            "() => !new URLSearchParams(window.location.search||'').get('sym')",
            timeout=15000,
        )

        # If the URL isn't updated for some reason, pSym must still be empty.
        page.wait_for_function(
            "() => !document.getElementById('pSym') || !document.getElementById('pSym').value",
            timeout=15000,
        )

        # Switch back to Meta and check it gets a helpful suggested entity.
        page.click("#tabMeta")
        page.wait_for_selector("#pageMeta", state="visible", timeout=10000)
        # Contract: Catalog search must not clobber an existing Meta entity filter coming from
        # localStorage/user. We seeded CPIAUCSL, so it should remain CPIAUCSL.
        page.wait_for_function(
            "() => (document.getElementById('mEntity')?.value || '').trim().toUpperCase() === 'CPIAUCSL'",
            timeout=5000,
        )

        # Click Query. It should NOT revert to stale FRED/CPI.
        page.click("#btnMetaQuery")
        page.wait_for_selector("#metaTable", timeout=10000)
        page.wait_for_function(
            """() => {
              const out = document.getElementById('metaTable');
              const sum = document.getElementById('metaSummary');
              const t = out ? String(out.innerText||'').trim().toLowerCase() : '';
              const s = sum ? String(sum.innerText||'').trim() : '';
              return (s.length > 0) && (!t.includes('(loading'));
            }""",
            timeout=15000,
        )
        meta_html = page.inner_html("#metaTable")

        assert "CPIAUCSL" not in meta_html, "Query reverted to stale CPI result"
        assert "FRED" not in meta_html.upper(), "Query reverted to stale FRED"
        # Under strict contract, we do not overwrite user's existing Meta filters (mProvider/mEntity)
        # just because the Catalog search changed.

        # Now select SPX explicitly (Catalog tab) and ensure Meta uses selection.
        page.click("#tabCatalog")
        page.wait_for_selector("#catalog a[data-act='preview']", state="attached", timeout=30000)
        spx = page.locator("#catalog a[data-act='preview'][data-entity='SPX']")
        if spx.count() == 0:
            spx = page.locator("#catalog a[data-act='preview']", has_text="SPX")
        spx.first.wait_for(state="visible", timeout=30000)
        spx.first.click()

        # Switch to Meta tab again and ensure selection populates Meta symbol.
        page.click("#tabMeta")
        page.wait_for_selector("#pageMeta", state="visible", timeout=10000)
        page.wait_for_function(
            "() => (document.getElementById('mSymbol')?.value||'').toUpperCase().includes('SPX')",
            timeout=5000,
        )

        page.click("#btnMetaQuery")
        page.wait_for_function(
            "() => (document.getElementById('metaSummary')?.innerText||'').length > 0",
            timeout=5000,
        )
        meta_html2 = page.inner_html("#metaTable")

        assert "CPIAUCSL" not in meta_html2
        assert "FRED" not in meta_html2
        assert "FRED" not in meta_html2.upper()

        browser.close()
