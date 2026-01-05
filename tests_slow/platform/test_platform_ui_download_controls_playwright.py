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


def test_download_panel_renders_type_and_frequency_controls(tmp_path, monkeypatch):
    """Browser smoke for the new Download UI controls.

    We only verify the controls exist and are interactive. The actual dry-run execution
    is covered by the other download smoke test.
    """

    arctic_root = tmp_path / "arctic_ui_download_controls"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

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

        # core controls
        page.wait_for_selector("#dlSourceType", state="visible", timeout=10000)
        page.wait_for_selector("#dlFrequency", state="visible", timeout=10000)
        page.wait_for_selector("#dlRangeMode", state="visible", timeout=10000)

        # switch source type to YF and verify entity input appears
        page.select_option("#dlSourceType", "yf")
        page.wait_for_selector("#dlSourceText", state="visible", timeout=10000)

        # Enter required parameters for network sources
        page.fill("#dlSourceText", "AAPL")
        page.fill("#dlStart", "2024-01-02")
        page.fill("#dlEnd", "2024-01-06")

        page.click("#btnDryRun")

        # summary should eventually show something (dry_run output or error json)
        page.wait_for_function(
            """() => {
              const el = document.getElementById('downloadSummary');
              const t = String(el ? (el.textContent || '') : '').trim().toLowerCase();
              return t.length > 0;
            }""",
            timeout=30000,
        )

        # switch to parquet and verify file input exists
        page.select_option("#dlSourceType", "parquet")
        page.wait_for_selector("#dlFile", state="visible", timeout=10000)

        browser.close()
