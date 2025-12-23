from __future__ import annotations


def test_ui_smoke_openapi_and_docs_load():
    # This is an integration-ish unit test using TestClient (no external server).
    # It catches regressions like the previous OpenAPI generation crash.
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    assert "application/json" in (r_openapi.headers.get("content-type") or "")

    r_docs = client.get("/docs")
    assert r_docs.status_code == 200
    assert "text/html" in (r_docs.headers.get("content-type") or "")

    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "text/html" in (r_ui.headers.get("content-type") or "")
    assert "Platform UI" in r_ui.text

