from __future__ import annotations


def test_ui_index_returns_html():
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in (r.headers.get("content-type") or "")
    assert "Platform UI" in r.text

    # Contract: SPA has a root mount node
    assert "id=\"app\"" in r.text or "id='app'" in r.text

    # Contract: modular entrypoint is referenced (no Node required at runtime)
    assert "/static/assets/main.mjs" in r.text

    # Contract markers used by other tests / workflows
    assert "copy-source" in r.text
    assert "/api/catalog/describe/" in r.text
    assert "missing ts sample" in r.text
