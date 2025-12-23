from __future__ import annotations

from quantdsl_backtest.data.requests import DataRequest
from quantdsl_backtest.platform_api.services.catalog_download import download_bundle


def test_download_bundle_dry_run_returns_plan_only():
    req = DataRequest(
        source="fred://CPIAUCSL",
        kind="market_bars",
        start="2024-01-01",
        end="2024-01-10",
        frequency="1d",
        dataset_id="macro",
    )

    out = download_bundle(request=req, universe=None, dry_run=True)
    assert out["dry_run"] is True
    assert out["request"]["source"] == "fred://CPIAUCSL"


def test_download_bundle_dry_run_with_entities_sets_universe_selection():
    req = DataRequest(
        source="yf://",
        kind="market_bars",
        start="2024-01-01",
        end="2024-01-10",
        frequency="1d",
        dataset_id="sp500",
    )

    out = download_bundle(request=req, universe=None, entities=["AAPL", "MSFT"], dry_run=True)
    assert out["dry_run"] is True
    assert out["entities"] == ["AAPL", "MSFT"]
