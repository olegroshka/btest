from __future__ import annotations


def test_openapi_generation_does_not_crash():
    from quantdsl_backtest.platform_api.main import create_app

    app = create_app()
    schema = app.openapi()
    assert isinstance(schema, dict)
    assert "paths" in schema
    assert "/openapi.json" not in schema.get("paths", {})

