from __future__ import annotations


def test_request_logging_emits_event(monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()

    events = []

    def _log_request(event):
        events.append(event)

    # Patch the symbol used by the middleware (imported inside the middleware function)
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.logging_utils.log_request",
        _log_request,
    )

    client = TestClient(app, raise_server_exceptions=False)

    # 1) With a provided request id
    r = client.get("/health", headers={"X-Request-Id": "rid-test"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-Id") == "rid-test"

    assert len(events) == 1
    e = events[-1]
    assert e.request_id == "rid-test"
    assert e.method == "GET"
    assert e.path == "/health"
    assert e.status_code == 200
    assert e.duration_ms >= 0.0

    # Ensure it can be serialized to a dict for structured logging
    d = e.as_dict()
    assert d["request_id"] == "rid-test"

    # 2) Without a provided request id -> server generates one
    events.clear()
    r2 = client.get("/health")
    assert r2.status_code == 200
    assert r2.headers.get("X-Request-Id")

    assert len(events) == 1
    e2 = events[-1]
    assert e2.request_id == r2.headers.get("X-Request-Id")
