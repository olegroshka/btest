from __future__ import annotations

from quantdsl_backtest.data.requests import DataRequest
from quantdsl_backtest.platform_api.services.catalog_download_plan import _provider_from_source


def test_provider_from_source():
    assert _provider_from_source("fred://CPI") == "FRED"
    assert _provider_from_source("yf://AAPL") == "YF"
    assert _provider_from_source("parquet://x") == "PARQUET"
    assert _provider_from_source("x://y") == "GLOBAL"


def test_plan_download_for_request_returns_empty_if_no_entities():
    from quantdsl_backtest.platform_api.services.catalog_download_plan import plan_download_for_request

    req = DataRequest(source="yf://", start="2024-01-01", end="2024-01-02", frequency="1d")
    out = plan_download_for_request(request=req, entities=[])
    assert out == []

