from __future__ import annotations

import pytest


def test_legacy_validation_rejects_bad_frequency():
    from quantdsl_backtest.data.requests import DataRequest
    from quantdsl_backtest.platform_api.services.catalog_request_validation import normalize_and_validate_request

    req = DataRequest(source="yf://", kind="market_bars", start="", end="", frequency="2d")
    with pytest.raises(ValueError):
        normalize_and_validate_request(req)


def test_legacy_validation_normalizes_aliases():
    from quantdsl_backtest.data.requests import DataRequest
    from quantdsl_backtest.platform_api.services.catalog_request_validation import normalize_and_validate_request

    req = DataRequest(source="yf://", kind="MARKET_BARS", start="", end="", frequency="daily")
    out = normalize_and_validate_request(req)
    assert out.kind == "market_bars"
    assert out.frequency == "1d"

