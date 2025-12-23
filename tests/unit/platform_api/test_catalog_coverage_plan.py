from __future__ import annotations

import pandas as pd

from quantdsl_backtest.platform_api.services.catalog_coverage import plan_download


def test_plan_download_full_fetch_when_no_meta():
    meta = pd.DataFrame(columns=["symbol", "provider", "frequency", "kind", "dataset", "entity", "start", "end"])
    plan = plan_download(
        request_start="2024-01-01",
        request_end="2024-01-10",
        entities=["AAPL"],
        provider="YF",
        frequency="1d",
        kind="market_bars",
        dataset="sp500",
        meta_df=meta,
    )
    assert plan[0]["action"] == "full_fetch"
    assert plan[0]["symbol"] == "market_bars/sp500/AAPL"


def test_plan_download_cache_hit_and_tail_fetch():
    meta = pd.DataFrame(
        [
            {
                "symbol": "market_bars/sp500/AAPL",
                "provider": "YF",
                "frequency": "1d",
                "kind": "market_bars",
                "dataset": "sp500",
                "entity": "AAPL",
                "start": "2024-01-01",
                "end": "2024-01-10",
            },
            {
                "symbol": "market_bars/sp500/MSFT",
                "provider": "YF",
                "frequency": "1d",
                "kind": "market_bars",
                "dataset": "sp500",
                "entity": "MSFT",
                "start": "2024-01-01",
                "end": "2024-01-05",
            },
        ]
    )

    plan = plan_download(
        request_start="2024-01-01",
        request_end="2024-01-10",
        entities=["AAPL", "MSFT"],
        provider="YF",
        frequency="1d",
        kind="market_bars",
        dataset="sp500",
        meta_df=meta,
    )

    aapl = next(p for p in plan if p["entity"] == "AAPL")
    msft = next(p for p in plan if p["entity"] == "MSFT")

    assert aapl["action"] == "cache_hit"
    assert msft["action"] == "tail_fetch"
    assert msft["fetch_start"].startswith("2024-01-06")
    assert aapl["symbol"] == "market_bars/sp500/AAPL"
    assert msft["symbol"] == "market_bars/sp500/MSFT"


def test_plan_download_intraday_fetch_start_respects_frequency():
    meta = pd.DataFrame(
        [
            {
                "symbol": "market_bars/sp500/AAPL",
                "provider": "YF",
                "frequency": "1h",
                "kind": "market_bars",
                "dataset": "sp500",
                "entity": "AAPL",
                "start": "2024-01-01 09:00:00",
                "end": "2024-01-01 10:00:00",
            }
        ]
    )

    plan = plan_download(
        request_start="2024-01-01 09:00:00",
        request_end="2024-01-01 12:00:00",
        entities=["AAPL"],
        provider="YF",
        frequency="1h",
        kind="market_bars",
        dataset="sp500",
        meta_df=meta,
    )

    assert plan[0]["action"] == "tail_fetch"
    assert plan[0]["fetch_start"].startswith("2024-01-01 11:00")
