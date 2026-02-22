from __future__ import annotations

import json


def test_get_run_summary_404s_and_success(tmp_path, monkeypatch):
    from datetime import datetime

    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app
    from quantdsl_backtest.platform_api.models.run import RunRecord
    from quantdsl_backtest.platform_api.services.run_store import RunStore

    # Hermetic db
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    # not found run
    r0 = client.get("/api/runs/nope/summary")
    assert r0.status_code == 404

    # insert run without summary
    store: RunStore = app.state.run_store
    run_dir = tmp_path / "run_artifacts" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)

    store.insert_run(
        RunRecord(
            run_id="r1",
            strategy_id="s1",
            strategy_hash="h1",
            status="succeeded",
            submitted_at=datetime(2025, 1, 1, 0, 0, 0),
            artifacts_dir=str(run_dir),
            reports_url="/reports/runs/r1/index.html",
        )
    )

    r1 = client.get("/api/runs/r1/summary")
    assert r1.status_code == 404
    assert r1.json()["error"]["code"] == "SUMMARY_NOT_FOUND"

    # write summary.json and try again
    (run_dir / "summary.json").write_text(json.dumps({"ok": True, "metrics": {"sharpe": 1.0}}), encoding="utf-8")

    r2 = client.get("/api/runs/r1/summary")
    assert r2.status_code == 200
    j = r2.json()["summary"]
    assert j["ok"] is True
    assert j["metrics"]["sharpe"] == 1.0

