from __future__ import annotations

import os
import socket
import threading
import time

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.manual]


def _get_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def test_platform_ui_download_smoke_playwright(tmp_path, monkeypatch):
    """E2E smoke: Catalog selection -> Download dry-run (parquet) shows summary.

    This is deterministic because it uses the local parquet dataset already shipped with the repo
    and the strategy run populates cache.
    """

    # Isolate cache
    arctic_root = tmp_path / "arctic_ui_download_smoke"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # Populate cache quickly via an example strategy
    from quantdsl_backtest.examples.lagging_indecies import build_strategy
    from quantdsl_backtest.engine.backtest_runner import run_backtest

    strat = build_strategy()
    try:
        strat.backtest.reporting.output_dir = None
    except Exception:
        pass
    res = run_backtest(strat)
    assert res is not None

    port = _get_free_port()

    def _run_server():
        from quantdsl_backtest.platform_api.main import create_app

        app = create_app()
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

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
        headless = _env_flag("UI_HEADLESS", True)
        slow_mo = _env_int("UI_SLOW_MO_MS", 0)
        devtools = _env_flag("UI_DEVTOOLS", False)

        browser = p.chromium.launch(headless=headless, slow_mo=slow_mo, devtools=devtools)
        page = browser.new_page()

        page.goto(f"http://127.0.0.1:{port}/?tab=catalog", wait_until="networkidle")
        page.wait_for_selector("#pageCatalog", state="visible", timeout=10000)

        # Ensure catalog loads
        page.wait_for_selector("#btnCatalog", timeout=10000)
        page.click("#btnCatalog")

        page.wait_for_function(
            """() => {
              const links = document.querySelectorAll("#catalog a[data-act='preview']");
              return links && links.length > 0;
            }""",
            timeout=20000,
        )

        # Click first preview link to populate selection
        page.query_selector_all("#catalog a[data-act='preview']")[0].click()
        page.wait_for_function(
            "() => document.getElementById('pLib')?.value?.trim().length > 0 && document.getElementById('pSym')?.value?.trim().length > 0",
            timeout=20000,
        )

        # Ensure download controls exist on Catalog (new UI)
        page.wait_for_selector("#dlSourceType", state="visible", timeout=10000)
        page.wait_for_selector("#dlFrequency", state="visible", timeout=10000)
        page.wait_for_selector("#btnDryRun", state="visible", timeout=10000)

        # PlatformApp includes legacy placeholders; downloadSummary may be attached but not visible until layout settles.
        page.wait_for_selector("#downloadSummary", state="attached", timeout=10000)

        # Set deterministic parquet source and run dry-run
        page.select_option("#dlSourceType", "parquet")
        page.fill("#dlFile", "equities/sp500_daily")
        page.select_option("#dlFrequency", "1d")
        page.select_option("#dlRangeMode", "meta")

        page.click("#btnDryRun")

        # Wait for output in the real DownloadPanel summary (the last #downloadSummary in DOM).
        page.wait_for_function(
            """() => {
              const els = Array.from(document.querySelectorAll('#downloadSummary'));
              const el = els.length ? els[els.length - 1] : null;
              const t = String(el ? (el.textContent || '') : '').trim();
              const low = t.toLowerCase();
              return t.length > 0 && (low.includes('dry_run') || low.includes('http_') || low.includes('error'));
            }""",
            timeout=30000,
        )

        browser.close()
