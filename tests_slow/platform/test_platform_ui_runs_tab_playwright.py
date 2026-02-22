from __future__ import annotations

import socket
import threading
import time

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_health(*, port: int, timeout_s: float = 8.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            import httpx

            r = httpx.get(f"http://127.0.0.1:{port}/health")
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("Server did not start")


def _write_parquet_market_data(*, out_path, dates, ticker: str = "AAPL") -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": list(dates) * 1,
            "ticker": [ticker for _ in dates],
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [1_000_000 for _ in dates],
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(out_path, index=False)


@pytest.mark.slow
def test_platform_ui_runs_tab_playwright(tmp_path, monkeypatch):
    """UI E2E: Runs tab shows running/success/failure and report click-through works.

    Scenario (hermetic):
      - Start the FastAPI app in-process (thread + uvicorn)
      - Create a hermetic strategies dir with two strategies:
          1) s_ui_ok: valid parquet -> run should succeed
          2) s_ui_fail: raises RuntimeError in build_strategy() -> run should fail
      - Submit both runs via API
      - Open UI at ?tab=runs and verify:
          - Runs grid renders
          - Both runs appear
          - Status transitions to succeeded/failed within timeout
          - "View Report" opens a page that contains run report HTML
          - Failed run has logs accessible via API, containing Traceback/Exception

    Notes:
      - We avoid flakiness by controlling input data and using generous but bounded timeouts.
      - We validate report availability by actually opening the report URL in a new page.
    """

    # IMPORTANT: ensure the app uses the process-pool runner so strategy code actually executes.
    monkeypatch.setenv("QDSL_PLATFORM_RUNNER_PROCESS_POOL", "1")

    playwright = pytest.importorskip("playwright.sync_api")
    sync_playwright = playwright.sync_playwright

    port = _free_port()

    # Hermetic db + strategies dir so we don't touch developer state.
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    # Tiny deterministic parquet market data for the success strategy.
    data_path = tmp_path / "md.parquet"
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    _write_parquet_market_data(out_path=data_path, dates=dates, ticker="AAPL")
    parquet_uri = "parquet://" + data_path.as_posix()

    # Strategy that should succeed.
    (strategies_dir / "s_ui_ok.py").write_text(
        (
            """
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, EqualWeight, TopN, BottomN, Book
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import CrossSectionRank
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees


def build_strategy() -> Strategy:
    factors = {
        'mom_1': ReturnFactor(name='mom_1', field='close', lookback=1, method='simple')
    }
    signals = {
        'sig': CrossSectionRank(name='sig', factor_name='mom_1')
    }

    long_book = Book(
        name='long_book',
        selector=TopN(factor_name='sig', n=1, mask_name=None),
        weighting=EqualWeight(),
    )
    short_book = Book(
        name='short_book',
        selector=BottomN(factor_name='sig', n=0, mask_name=None),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency='1d',
        signal_delay_bars=1,
    )

    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0),
        volume_limits=VolumeParticipation(max_participation=1.0),
    )

    costs = Costs(
        commission=Commission(type='bps_notional', amount=0.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve='SOFR', spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    return Strategy(
        name='s_ui_ok',
        data=DataConfig(source=r'__PARQUET_URI__', calendar='XNYS', frequency='1d', start='2025-01-01', end='2025-01-07'),
        universe=Universe(name='TEST', static_instruments=['AAPL']),
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=BacktestConfig(engine='event_driven', reporting=Reporting(output_dir=None)),
    )
"""
        )
        .replace("__PARQUET_URI__", parquet_uri)
        .lstrip(),
        encoding="utf-8",
    )

    # Strategy that should fail deterministically.
    (strategies_dir / "s_ui_fail.py").write_text(
        (
            """
from quantdsl_backtest.dsl.strategy import Strategy


def build_strategy() -> Strategy:
    raise RuntimeError('intentional test failure: s_ui_fail')
"""
        ).lstrip(),
        encoding="utf-8",
    )

    # Start server
    def _run_server():
        import uvicorn

        from quantdsl_backtest.platform_api.main import app

        uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")

    th = threading.Thread(target=_run_server, daemon=True)
    th.start()
    _wait_health(port=port)

    # Submit both runs via API
    import httpx

    ok_resp = httpx.post(f"http://127.0.0.1:{port}/api/runs", json={"strategy_id": "s_ui_ok", "params": {}})
    assert ok_resp.status_code == 200, ok_resp.text
    ok_run_id = ok_resp.json()["run_id"]

    fail_resp = httpx.post(f"http://127.0.0.1:{port}/api/runs", json={"strategy_id": "s_ui_fail", "params": {}})
    assert fail_resp.status_code == 200, fail_resp.text
    fail_run_id = fail_resp.json()["run_id"]

    # Helper poll via API to know terminal states (keeps UI waits bounded)
    def _poll_terminal(run_id: str, timeout_s: float = 30.0) -> dict:
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout_s:
            r = httpx.get(f"http://127.0.0.1:{port}/api/runs/{run_id}")
            assert r.status_code == 200
            last = r.json()["run"]
            if last.get("status") in ("succeeded", "failed"):
                return last
            time.sleep(0.05)
        raise AssertionError(f"timeout waiting for terminal status; last={last}")

    ok_terminal = _poll_terminal(ok_run_id, timeout_s=35.0)
    fail_terminal = _poll_terminal(fail_run_id, timeout_s=35.0)

    assert ok_terminal["status"] == "succeeded"
    assert fail_terminal["status"] == "failed"

    # Verify report is on disk via API summary quickly (prevents UI flake)
    sum_ok = httpx.get(f"http://127.0.0.1:{port}/api/runs/{ok_run_id}/summary")
    assert sum_ok.status_code == 200

    # Fetch failed run logs for assertions
    logs_fail = httpx.get(f"http://127.0.0.1:{port}/api/runs/{fail_run_id}/logs")
    assert logs_fail.status_code == 200
    logs_txt = str(logs_fail.json().get("logs") or "")
    assert ("Traceback" in logs_txt) or ("Exception" in logs_txt)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        last_alert_text: dict[str, str] = {"txt": ""}

        def _on_dialog(d):
            try:
                last_alert_text["txt"] = str(d.message or "")
            except Exception:
                last_alert_text["txt"] = ""
            try:
                d.accept()
            except Exception:
                pass

        page.on("dialog", _on_dialog)

        page.goto(f"http://127.0.0.1:{port}/?tab=runs")

        # Wait for boot marker + grid
        page.wait_for_selector("#app[data-ui-boot='1']", timeout=20000)
        page.wait_for_selector("#pageRuns", state="visible", timeout=20000)
        page.wait_for_selector("[data-testid='runs-grid']", state="visible", timeout=20000)

        ok_short = ok_run_id[:8]
        fail_short = fail_run_id[:8]

        # Clicking refresh should not error and should keep grid present.
        page.locator("[data-testid='btnRunsRefresh']").click()
        page.wait_for_selector("[data-testid='runs-grid']", state="visible", timeout=20000)

        # Runs show up
        page.wait_for_function(
            """([a,b]) => document.body && document.body.innerText && document.body.innerText.includes(a) && document.body.innerText.includes(b)""",
            arg=[ok_short, fail_short],
            timeout=20000,
        )

        # Status badges show up
        page.wait_for_function(
            """() => {
              const t = (document.body?.innerText || '').toUpperCase();
              return t.includes('SUCCEEDED') && t.includes('FAILED');
            }""",
            timeout=20000,
        )

        # Filter by status=failed and ensure succeeded run id disappears.
        page.select_option("[data-testid='runsFilterStatus']", "failed")
        # Refresh to apply filter immediately (avoids waiting for poll)
        page.locator("[data-testid='btnRunsRefresh']").click()
        page.wait_for_function(
            """([okPrefix, failPrefix]) => {
              const t = (document.body?.innerText || '');
              return t.includes(failPrefix) && !t.includes(okPrefix);
            }""",
            arg=[ok_short, fail_short],
            timeout=20000,
        )

        # Clear status filter, filter by strategy_id=s_ui_ok and ensure ok run is visible.
        page.select_option("[data-testid='runsFilterStatus']", "")
        page.select_option("[data-testid='runsFilterStrategy']", "s_ui_ok")
        page.locator("[data-testid='btnRunsRefresh']").click()
        page.wait_for_function(
            """([okPrefix, failPrefix]) => {
              const t = (document.body?.innerText || '');
              return t.includes(okPrefix) && !t.includes(failPrefix);
            }""",
            arg=[ok_short, fail_short],
            timeout=20000,
        )

        # Reset filters back to all.
        page.select_option("[data-testid='runsFilterStrategy']", "")
        page.locator("[data-testid='btnRunsRefresh']").click()
        page.wait_for_function(
            """([a,b]) => document.body && document.body.innerText && document.body.innerText.includes(a) && document.body.innerText.includes(b)""",
            arg=[ok_short, fail_short],
            timeout=20000,
        )

        # Click View Logs for failed run -> alert should contain the failure message.
        page.locator(f"[data-testid='btnRunLogs-{fail_short}']").click()

        t0 = time.time()
        while time.time() - t0 < 5.0:
            if (last_alert_text["txt"] or "").strip():
                break
            page.wait_for_timeout(50)
        assert (last_alert_text["txt"] or "").strip(), "Expected alert dialog from View Logs"

        # Since we capture the dialog message in Python, assert it contains our intentional failure.
        assert "intentional test failure" in (last_alert_text["txt"] or "")

        # Click View Report for the succeeded run.
        report_btn = page.locator(f"[data-testid='btnRunReport-{ok_short}']")
        report_btn.wait_for(state="visible", timeout=20000)

        with page.expect_popup() as pop:
            report_btn.click()
        report_page = pop.value

        report_page.wait_for_load_state("domcontentloaded", timeout=20000)
        report_page.wait_for_selector("html", timeout=20000)

        assert f"/reports/runs/{ok_run_id}" in report_page.url

        browser.close()
