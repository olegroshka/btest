from __future__ import annotations

from pathlib import Path


def test_reports_serving_and_traversal_protection(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    # point runs root to tmp
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_paths.default_runs_root_dir",
        lambda: runs_root,
    )
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.routes.reports.default_runs_root_dir",
        lambda: runs_root,
    )

    run_id = "r1"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get(f"/reports/runs/{run_id}/index.html")
    assert r.status_code == 200
    assert "ok" in r.text

    # traversal attempt
    r2 = client.get(f"/reports/runs/{run_id}/../secrets.txt")
    assert r2.status_code == 404

    r3 = client.get(f"/reports/runs/{run_id}/does_not_exist.html")
    assert r3.status_code == 404

