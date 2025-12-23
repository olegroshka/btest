from __future__ import annotations


def test_api_error_handlers_registered_and_shape():
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()

    assert Exception in app.exception_handlers
    assert HTTPException in app.exception_handlers

    @app.get("/__test/http_error")
    def _raise_http():
        raise HTTPException(status_code=400, detail="bad")

    @app.get("/__test/unhandled")
    def _raise_unhandled():
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)

    r1 = client.get("/__test/http_error", headers={"X-Request-Id": "rid-1"})
    assert r1.status_code == 400
    assert r1.headers.get("X-Request-Id") == "rid-1"
    j1 = r1.json()
    assert j1["error"]["code"] == "HTTP_400"
    assert j1["error"]["message"] == "bad"
    assert j1["error"]["status"] == 400
    assert j1["error"]["request_id"] == "rid-1"

    # Invalid incoming request id should be sanitized away
    r1b = client.get("/__test/http_error", headers={"X-Request-Id": "bad rid\n"})
    assert r1b.status_code == 400
    assert r1b.headers.get("X-Request-Id")
    assert r1b.headers.get("X-Request-Id") != "bad rid\n"
    assert r1b.json()["error"]["request_id"] == r1b.headers.get("X-Request-Id")

    r2 = client.get("/__test/unhandled")
    assert r2.status_code == 500
    assert r2.headers.get("X-Request-Id")
    j2 = r2.json()
    assert j2["error"]["code"] == "UNHANDLED"
    assert "boom" in j2["error"]["message"]
    assert j2["error"]["status"] == 500
    assert j2["error"]["request_id"]

    # RequestValidationError (Pydantic) should also use our envelope
    r3 = client.post(
        "/api/catalog/plan_download",
        json={
            "source": "yf://",
            "kind": "market_bars",
            "start": "2024-01-01",
            "end": "2024-01-02",
            "frequency": "2d",
            "entities": [],
        },
        headers={"X-Request-Id": "rid-3"},
    )
    if r3.status_code != 422:
        raise AssertionError(f"expected 422 got {r3.status_code} body={r3.text}")
    assert r3.headers.get("X-Request-Id") == "rid-3"
    j3 = r3.json()
    assert j3["error"]["code"] == "INVALID_FREQUENCY"
    assert j3["error"]["status"] == 422
    assert j3["error"]["request_id"] == "rid-3"
    assert "errors" in (j3["error"].get("details") or {})
