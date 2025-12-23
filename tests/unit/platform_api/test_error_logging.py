from __future__ import annotations


def test_error_logging_emits_event(monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()

    error_events = []

    def _log_error(event):
        error_events.append(event)

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.logging_utils.log_error",
        _log_error,
    )

    client = TestClient(app, raise_server_exceptions=False)

    # Force a validation error (invalid frequency)
    r = client.post(
        "/api/catalog/plan_download",
        json={
            "source": "yf://",
            "kind": "market_bars",
            "start": "2024-01-01",
            "end": "2024-01-02",
            "frequency": "2d",
            "entities": [],
        },
        headers={"X-Request-Id": "rid-err"},
    )

    assert r.status_code == 422
    assert len(error_events) >= 1
    e = error_events[-1]
    assert e.request_id == "rid-err"
    assert e.error_code == "INVALID_FREQUENCY"
    assert e.status_code == 422
    assert e.path == "/api/catalog/plan_download"
