from __future__ import annotations

from datetime import datetime


def _seed_store(db_path: str):
    from quantdsl_backtest.platform_api.models.run import RunRecord
    from quantdsl_backtest.platform_api.services.run_store import RunStore

    store = RunStore(db_path=db_path, retention=500, enable_wal=False)
    store.insert_run(
        RunRecord(
            run_id="r1",
            strategy_id="s1",
            strategy_hash="h1",
            status="succeeded",
            submitted_at=datetime(2025, 1, 1, 0, 0, 0),
            params={"engine": "event_driven"},
            metrics={"sharpe": 1.0},
            artifacts_dir="outputs/runs/r1",
            reports_url="/reports/runs/r1/index.html",
        )
    )
    store.insert_run(
        RunRecord(
            run_id="r2",
            strategy_id="s2",
            strategy_hash="h2",
            status="failed",
            submitted_at=datetime(2025, 1, 2, 0, 0, 0),
            params={},
            error="boom",
        )
    )
    return store


def test_runs_list_and_detail(tmp_path):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    db_path = str(tmp_path / "runs.db")
    store = _seed_store(db_path)

    app = create_app()
    app.state.run_store = store

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/api/runs?limit=10")
    assert r.status_code == 200
    assert r.headers.get("X-Request-Id")
    j = r.json()
    assert j["total"] == 2
    assert [x["run_id"] for x in j["runs"]] == ["r2", "r1"]  # newest first

    r1 = client.get("/api/runs/r1")
    assert r1.status_code == 200
    j1 = r1.json()["run"]
    assert j1["run_id"] == "r1"
    assert j1["status"] == "succeeded"
    assert j1["metrics"]["sharpe"] == 1.0

    r404 = client.get("/api/runs/nope")
    assert r404.status_code == 404
    assert r404.json()["error"]["code"] in ("RUN_NOT_FOUND", "HTTP_404")


def test_runs_filter_by_status_and_strategy(tmp_path):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    db_path = str(tmp_path / "runs.db")
    store = _seed_store(db_path)

    app = create_app()
    app.state.run_store = store

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/api/runs?status=failed")
    assert r.status_code == 200
    j = r.json()
    assert j["total"] == 1
    assert j["runs"][0]["run_id"] == "r2"

    r2 = client.get("/api/runs?strategy_id=s1")
    assert r2.status_code == 200
    j2 = r2.json()
    assert j2["total"] == 1
    assert j2["runs"][0]["run_id"] == "r1"

