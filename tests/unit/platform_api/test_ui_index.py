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

    # Contract: JS entrypoint is referenced.
    # Current UI shell is React/Vite-driven.
    assert (
        ("/static/assets/main.react.js" in r.text)
        or ("/static/assets/main.js" in r.text)  # fallback/older builds
    )

    # Contract: CSS bundle is referenced (Vite)
    assert ("/static/assets/main.css" in r.text) or ("/static/assets/main-" in r.text)

    # Contract markers used by other tests / workflows
    assert "copy-source" in r.text
    assert "/api/catalog/describe/" in r.text
    assert "missing ts sample" in r.text
