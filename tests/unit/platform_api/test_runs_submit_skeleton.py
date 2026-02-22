from __future__ import annotations

import time


def test_post_runs_submits_and_transitions_status(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api import main

    # Hermetic run store DB
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    # Hermetic strategies dir
    strategies_dir = tmp_path / "strategies"
    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    strategies_dir.mkdir(parents=True, exist_ok=True)
    (strategies_dir / "s1.py").write_text("print('hi')\n", encoding="utf-8")

    app = main.create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post(
        "/api/runs",
        json={
            "strategy_id": "s1",
            "params": {"x": 1},
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert "run_id" in j
    assert j["status"] == "pending"

    run_id = j["run_id"]

    # Poll until terminal
    t0 = time.time()
    while True:
        r2 = client.get(f"/api/runs/{run_id}")
        assert r2.status_code == 200
        st = r2.json()["run"]["status"]
        if st in ("succeeded", "failed"):
            break
        if time.time() - t0 > 2.0:
            raise AssertionError(f"run did not finish in time, last status={st}")
        time.sleep(0.01)

    assert st == "succeeded"
    run = r2.json()["run"]
    assert run["strategy_id"] == "s1"
    assert run["source_snapshot"].startswith("print")
    assert run["metrics"] == {"ok": True}


def test_post_runs_backcompat_source_hash_override(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api import main

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = main.create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post(
        "/api/runs",
        json={
            "strategy_id": "s1",
            "source": "print('hi')\n",
            "strategy_hash": "h1",
            "params": {"x": 1},
        },
    )
    assert r.status_code == 200


def test_post_runs_override_requires_both_source_and_hash(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api import main

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = main.create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r1 = client.post(
        "/api/runs",
        json={
            "strategy_id": "s1",
            "source": "print('hi')\n",
            "params": {"x": 1},
        },
    )
    assert r1.status_code == 422

    r2 = client.post(
        "/api/runs",
        json={
            "strategy_id": "s1",
            "strategy_hash": "h1",
            "params": {"x": 1},
        },
    )
    assert r2.status_code == 422


def test_post_runs_validation_errors(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api import main

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.run_store_factory.default_runs_db_path",
        lambda: tmp_path / "runs.db",
    )

    app = main.create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/api/runs", json={})
    assert r.status_code == 422
    assert r.headers.get("X-Request-Id")

    # params must be object
    r2 = client.post(
        "/api/runs",
        json={"strategy_id": "s1", "params": 123},
    )
    assert r2.status_code == 422
