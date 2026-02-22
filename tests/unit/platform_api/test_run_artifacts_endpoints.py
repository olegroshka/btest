from __future__ import annotations

import json


def test_run_artifacts_list_prefers_summary(tmp_path, monkeypatch):
    from datetime import datetime

    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app
    from quantdsl_backtest.platform_api.models.run import RunRecord

    # Hermetic db
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    run_dir = tmp_path / "artifacts" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "a.txt").write_text("a", encoding="utf-8")
    (run_dir / "b.txt").write_text("b", encoding="utf-8")

    # Put summary.json with stable artifact list
    (run_dir / "summary.json").write_text(json.dumps({"artifacts": ["b.txt", "a.txt"]}), encoding="utf-8")

    app.state.run_store.insert_run(
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

    r = client.get("/api/runs/r1/artifacts")
    assert r.status_code == 200
    assert r.json()["artifacts"] == ["b.txt", "a.txt"]


def test_run_artifacts_download_and_traversal_protection(tmp_path, monkeypatch):
    from datetime import datetime

    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app
    from quantdsl_backtest.platform_api.models.run import RunRecord

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    run_dir = tmp_path / "artifacts" / "r2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "file.txt").write_text("hello", encoding="utf-8")

    app.state.run_store.insert_run(
        RunRecord(
            run_id="r2",
            strategy_id="s1",
            strategy_hash="h1",
            status="succeeded",
            submitted_at=datetime(2025, 1, 1, 0, 0, 0),
            artifacts_dir=str(run_dir),
            reports_url="/reports/runs/r2/index.html",
        )
    )

    ok = client.get("/api/runs/r2/artifact/file.txt")
    assert ok.status_code == 200
    assert ok.text == "hello"

    trav = client.get("/api/runs/r2/artifact/../secrets.txt")
    assert trav.status_code == 404

