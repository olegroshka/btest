from __future__ import annotations


def test_health_response_model():
    from quantdsl_backtest.platform_api.models.misc import HealthResponse

    r = HealthResponse(status="ok")
    assert r.status == "ok"


def test_providers_response_model():
    from quantdsl_backtest.platform_api.models.misc import ProvidersResponse

    r = ProvidersResponse(providers=[{"name": "x", "class": "Y"}])
    assert r.providers[0]["name"] == "x"

