from __future__ import annotations

from quantdsl_backtest.platform_api.models.catalog import CatalogDownloadRequest, CatalogPlanRequest


def test_catalog_download_request_defaults():
    r = CatalogDownloadRequest(source="yf://")
    assert r.kind == "market_bars"
    assert r.frequency == "1d"
    assert r.entities == []
    assert r.dry_run is False


def test_catalog_plan_request_defaults():
    r = CatalogPlanRequest(source="yf://")
    assert r.kind == "market_bars"
    assert r.entities == []

