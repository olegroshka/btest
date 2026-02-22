from __future__ import annotations

import pytest


@pytest.mark.parametrize("path", ["/api/strategies", "/api/strategies/s1"])
def test_strategies_endpoints_basic(tmp_path, monkeypatch, path: str):
    """Hermetic strategies endpoints test using a tmp strategies dir."""

    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    strategies_dir = tmp_path / "strategies"

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    strategies_dir.mkdir(parents=True, exist_ok=True)
    (strategies_dir / "s1.py").write_text('"""S1"""\n\nX=1\n', encoding="utf-8")

    r = client.get(path)
    assert r.headers.get("X-Request-Id")

    if path.endswith("/strategies"):
        assert r.status_code == 200
        j = r.json()
        assert "error" not in j
        assert any(s["id"] == "s1" for s in j["strategies"])
    else:
        assert r.status_code == 200
        j = r.json()
        assert "error" not in j
        assert j["strategy"]["id"] == "s1"
        assert "source" in j["strategy"]


def test_create_and_update_strategy(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    strategies_dir = tmp_path / "strategies"

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    # Create
    r = client.post(
        "/api/strategies",
        json={"id": "my_strat", "source": '"""My"""\n\ndef build_strategy():\n    return None\n'},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["id"] == "my_strat"
    assert (strategies_dir / "my_strat.py").exists()

    # Duplicate
    r2 = client.post(
        "/api/strategies",
        json={"id": "my_strat", "source": "print('x')\n"},
    )
    assert r2.status_code == 409

    # Update
    r3 = client.put(
        "/api/strategies/my_strat",
        json={"source": "# updated\n"},
    )
    assert r3.status_code == 200
    j3 = r3.json()
    assert j3["id"] == "my_strat"
    assert "strategy_hash" in j3

    # Update missing
    r4 = client.put(
        "/api/strategies/nope",
        json={"source": "# x\n"},
    )
    assert r4.status_code == 404


def test_get_strategy_404(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quantdsl_backtest.platform_api.main import create_app

    strategies_dir = tmp_path / "strategies"

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/api/strategies/nope")
    assert r.status_code == 404
    assert r.headers.get("X-Request-Id")
    assert r.json()["error"]["code"] in ("STRATEGY_NOT_FOUND", "HTTP_404")
