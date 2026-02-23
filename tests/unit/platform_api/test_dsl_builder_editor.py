"""Tests for DSL Builder editor upgrade (Milestone 3.5.7).

Tests the save/run workflow exercised by the new DSL Builder UI:
  - Save new strategy via POST /api/strategies
  - Update via PUT /api/strategies/{id}
  - Run via POST /api/runs
  - Edge case: run non-existent strategy
"""

from __future__ import annotations

import pytest


def _make_app(tmp_path, monkeypatch):
    """Create an isolated app instance with hermetic strategies dir and run store."""
    from quantdsl_backtest.platform_api.main import create_app
    from quantdsl_backtest.platform_api.services.run_store import RunStore

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "quantdsl_backtest.platform_api.services.strategy_discovery.default_strategies_dir",
        lambda: strategies_dir,
    )

    app = create_app()

    # Use an in-memory run store for isolation.
    store = RunStore(db_path=":memory:", retention=500, enable_wal=False)
    app.state.run_store = store

    # Inject a mock task runner that captures submissions instead of spawning processes.
    class _MockSubmitResult:
        def __init__(self, run_id, status):
            self.run_id = run_id
            self.status = status

    class _MockTaskRunner:
        def __init__(self):
            self.submissions = []

        def submit(self, *, strategy_id, strategy_hash, source_snapshot, params=None):
            import uuid

            rid = uuid.uuid4().hex
            self.submissions.append(
                {
                    "run_id": rid,
                    "strategy_id": strategy_id,
                    "strategy_hash": strategy_hash,
                    "source_snapshot": source_snapshot,
                    "params": params,
                }
            )
            return _MockSubmitResult(run_id=rid, status="pending")

    mock_runner = _MockTaskRunner()
    app.state.task_runner = mock_runner

    return app, strategies_dir, mock_runner


class TestDSLBuilderSaveWorkflow:
    """Tests the strategy save workflow from the DSL Builder UI perspective."""

    def test_create_new_strategy(self, tmp_path, monkeypatch):
        """Happy path: create a new strategy from the editor."""
        from fastapi.testclient import TestClient

        app, strategies_dir, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        source = '"""My test strategy"""\n\ndef build_strategy():\n    return None\n'

        r = client.post("/api/strategies", json={"id": "test_strat", "source": source})
        assert r.status_code == 200
        j = r.json()
        assert j["id"] == "test_strat"
        assert (strategies_dir / "test_strat.py").exists()
        assert (strategies_dir / "test_strat.py").read_text(encoding="utf-8") == source

    def test_update_existing_strategy(self, tmp_path, monkeypatch):
        """Happy path: update an existing strategy."""
        from fastapi.testclient import TestClient

        app, strategies_dir, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        original = "# v1\n"
        (strategies_dir / "my_strat.py").write_text(original, encoding="utf-8")

        updated = "# v2 - updated\n"
        r = client.put("/api/strategies/my_strat", json={"source": updated})
        assert r.status_code == 200
        j = r.json()
        assert j["id"] == "my_strat"
        assert "strategy_hash" in j
        assert (strategies_dir / "my_strat.py").read_text(encoding="utf-8") == updated

    def test_create_duplicate_strategy_409(self, tmp_path, monkeypatch):
        """Edge case: creating a duplicate strategy returns 409."""
        from fastapi.testclient import TestClient

        app, strategies_dir, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        (strategies_dir / "existing.py").write_text("# already here\n", encoding="utf-8")

        r = client.post("/api/strategies", json={"id": "existing", "source": "# new\n"})
        assert r.status_code == 409

    def test_update_nonexistent_strategy_404(self, tmp_path, monkeypatch):
        """Edge case: updating a non-existent strategy returns 404."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        r = client.put("/api/strategies/nope", json={"source": "# x\n"})
        assert r.status_code == 404


class TestDSLBuilderRunWorkflow:
    """Tests the run submission workflow from the DSL Builder UI perspective."""

    def test_run_existing_strategy(self, tmp_path, monkeypatch):
        """Happy path: save then run a strategy."""
        from fastapi.testclient import TestClient

        app, strategies_dir, mock_runner = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        source = '"""Runnable"""\n\ndef build_strategy():\n    return None\n'
        (strategies_dir / "runnable.py").write_text(source, encoding="utf-8")

        r = client.post("/api/runs", json={"strategy_id": "runnable"})
        assert r.status_code == 200
        j = r.json()
        assert "run_id" in j
        assert j["status"] == "pending"

        # Verify the mock runner received the submission.
        assert len(mock_runner.submissions) == 1
        sub = mock_runner.submissions[0]
        assert sub["strategy_id"] == "runnable"
        assert sub["source_snapshot"] == source

    def test_run_nonexistent_strategy_404(self, tmp_path, monkeypatch):
        """Edge case: running a non-existent strategy returns 404."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        r = client.post("/api/runs", json={"strategy_id": "does_not_exist"})
        assert r.status_code == 404

    def test_save_and_run_sequence(self, tmp_path, monkeypatch):
        """Integration: mimics the Save & Run button — create then run."""
        from fastapi.testclient import TestClient

        app, strategies_dir, mock_runner = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        source = "# my strategy\ndef build_strategy():\n    return None\n"

        # Step 1: Save (create)
        r1 = client.post("/api/strategies", json={"id": "my_new", "source": source})
        assert r1.status_code == 200

        # Step 2: Run
        r2 = client.post("/api/runs", json={"strategy_id": "my_new"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "pending"
        assert len(mock_runner.submissions) == 1
        assert mock_runner.submissions[0]["strategy_id"] == "my_new"

    def test_run_without_save_using_source_override(self, tmp_path, monkeypatch):
        """Key scenario: Run button pressed without saving first.

        The UI sends source + strategy_hash directly so the backend doesn't
        need to find the strategy on disk. This must succeed.
        """
        import hashlib

        from fastapi.testclient import TestClient

        app, strategies_dir, mock_runner = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        source = "# unsaved strategy\ndef build_strategy():\n    return None\n"
        source_hash = hashlib.sha256(source.encode()).hexdigest()

        # No file on disk — strategy_id does NOT exist in strategies_dir.
        assert not (strategies_dir / "unsaved_strat.py").exists()

        # Submit with source + strategy_hash override.
        r = client.post("/api/runs", json={
            "strategy_id": "unsaved_strat",
            "source": source,
            "strategy_hash": source_hash,
        })
        assert r.status_code == 200, f"Expected 200 but got {r.status_code}: {r.text}"
        j = r.json()
        assert "run_id" in j
        assert j["status"] == "pending"

        # Verify the runner received the correct source snapshot.
        assert len(mock_runner.submissions) == 1
        sub = mock_runner.submissions[0]
        assert sub["strategy_id"] == "unsaved_strat"
        assert sub["source_snapshot"] == source
        assert sub["strategy_hash"] == source_hash

    def test_run_with_only_source_no_hash_fails(self, tmp_path, monkeypatch):
        """Edge case: providing source without hash (or vice versa) is a 422."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        # source without hash
        r = client.post("/api/runs", json={
            "strategy_id": "x",
            "source": "# code\n",
        })
        assert r.status_code == 422


class TestDSLBuilderGenerateEndpoint:
    """Tests the /api/dsl/generate endpoint used by the form in generated mode."""

    def test_generate_default_config(self, tmp_path, monkeypatch):
        """Happy path: generate code from a minimal config."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        config = {
            "data": {
                "source": "parquet://equities/indicies.parquet",
                "calendar": "XNYS",
                "start_date": "2015-01-01",
                "end_date": "2025-12-31",
            },
            "universe": {"name": "Indices", "filters": []},
            "factors": {"mom_126": {"type": "momentum", "params": {"lookback": 126}}},
            "signals": {"rank_momentum": {"type": "cross_section_rank", "params": {"factor": "mom_126"}}},
            "portfolio": {
                "type": "long_short",
                "long_book": {"selector": "TopN", "weighting": "EqualWeight"},
                "short_book": {"selector": "BottomN", "weighting": "EqualWeight"},
            },
        }

        r = client.post("/api/dsl/generate", json=config)
        assert r.status_code == 200
        j = r.json()
        assert "python_code" in j
        assert "json_config" in j
        assert "ReturnFactor" in j["python_code"]
        assert "Strategy" in j["python_code"]

    def test_generated_code_does_not_call_run_backtest(self, tmp_path, monkeypatch):
        """Generated code must NOT call run_backtest() — the worker does that."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        config = {
            "data": {"source": "test.parquet", "calendar": "XNYS",
                     "start_date": "2020-01-01", "end_date": "2021-01-01"},
            "universe": {"name": "Test"},
            "factors": {"f1": {"type": "momentum", "params": {"lookback": 60}}},
            "signals": {"s1": {"type": "cross_section_rank", "params": {"factor": "f1"}}},
            "portfolio": {"type": "long_short"},
        }

        r = client.post("/api/dsl/generate", json=config)
        assert r.status_code == 200
        code = r.json()["python_code"]
        # There should be no `result = run_backtest(strategy)` line.
        assert "\nresult = run_backtest(" not in code
        # But the `strategy` variable must be defined.
        assert "strategy = Strategy(" in code

    def test_generated_code_uses_signal_for_portfolio_selector(self, tmp_path, monkeypatch):
        """Portfolio selectors must use the actual signal name, not a hardcoded default."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        config = {
            "data": {"source": "test.parquet", "calendar": "XNYS",
                     "start_date": "2020-01-01", "end_date": "2021-01-01"},
            "universe": {"name": "Test"},
            "factors": {"vol_63": {"type": "volatility", "params": {"lookback": 63}}},
            "signals": {"my_custom_signal": {"type": "cross_section_rank", "params": {"factor": "vol_63"}}},
            "portfolio": {"type": "long_short"},
        }

        r = client.post("/api/dsl/generate", json=config)
        assert r.status_code == 200
        code = r.json()["python_code"]
        # Portfolio selectors should reference "my_custom_signal", not "rank_momentum".
        assert 'factor_name="my_custom_signal"' in code
        assert 'factor_name="rank_momentum"' not in code

    def test_generate_empty_factors(self, tmp_path, monkeypatch):
        """Edge case: generate code with no factors."""
        from fastapi.testclient import TestClient

        app, _, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        config = {
            "data": {
                "source": "test.parquet",
                "calendar": "XNYS",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
            },
            "universe": {"name": "Test"},
            "factors": {},
            "signals": {},
        }

        r = client.post("/api/dsl/generate", json=config)
        assert r.status_code == 200
        j = r.json()
        assert "python_code" in j
        # Code should still be valid (even if factors section is empty).
        assert "factors = {" in j["python_code"]


class TestDSLBuilderStrategyLoader:
    """Tests the strategy loading workflow (strategy selector dropdown)."""

    def test_list_strategies(self, tmp_path, monkeypatch):
        """Happy path: list strategies for the dropdown."""
        from fastapi.testclient import TestClient

        app, strategies_dir, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        (strategies_dir / "alpha.py").write_text("# alpha\n", encoding="utf-8")
        (strategies_dir / "beta.py").write_text("# beta\n", encoding="utf-8")

        r = client.get("/api/strategies")
        assert r.status_code == 200
        j = r.json()
        ids = [s["id"] for s in j["strategies"]]
        assert "alpha" in ids
        assert "beta" in ids

    def test_get_strategy_source(self, tmp_path, monkeypatch):
        """Happy path: load a strategy source for the editor."""
        from fastapi.testclient import TestClient

        app, strategies_dir, _ = _make_app(tmp_path, monkeypatch)
        client = TestClient(app, raise_server_exceptions=False)

        source = "# my strategy code\nx = 42\n"
        (strategies_dir / "my_strat.py").write_text(source, encoding="utf-8")

        r = client.get("/api/strategies/my_strat")
        assert r.status_code == 200
        j = r.json()
        assert j["strategy"]["source"] == source
        assert j["strategy"]["id"] == "my_strat"

