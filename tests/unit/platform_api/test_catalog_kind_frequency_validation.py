from __future__ import annotations

import pytest


def test_catalog_download_request_normalizes_and_validates():
    from quantdsl_backtest.platform_api.models.catalog import CatalogDownloadRequest

    r = CatalogDownloadRequest(source="yf://", kind="MARKET_BARS", frequency="daily")
    assert r.kind == "market_bars"
    assert r.frequency == "1d"


@pytest.mark.parametrize("freq", ["2d", "15m", "", "weekly"])
def test_catalog_download_request_rejects_unknown_frequency(freq):
    from quantdsl_backtest.platform_api.models.catalog import CatalogDownloadRequest

    with pytest.raises(ValueError):
        CatalogDownloadRequest(source="yf://", frequency=freq)


@pytest.mark.parametrize("kind", ["prices", "bars", "signals"])
def test_catalog_plan_request_rejects_unknown_kind(kind):
    from quantdsl_backtest.platform_api.models.catalog import CatalogPlanRequest

    with pytest.raises(ValueError):
        CatalogPlanRequest(source="yf://", kind=kind)

