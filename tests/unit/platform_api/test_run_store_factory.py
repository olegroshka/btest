from __future__ import annotations


def test_create_app_initializes_run_store(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api import main

    # Point default DB path into tmp_path (hermetic)
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = main.create_app()

    assert hasattr(app.state, "run_store")
    store = app.state.run_store
    assert store is not None

    # Smoke: list runs returns empty and doesn't error
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/runs")
    assert r.status_code == 200
    j = r.json()
    assert j["total"] == 0
    assert j["runs"] == []

