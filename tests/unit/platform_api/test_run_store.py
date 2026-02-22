from __future__ import annotations

from datetime import datetime, timedelta

import pytest


def test_run_store_schema_idempotent_and_retention_prunes_to_latest(tmp_path):
    from quantdsl_backtest.platform_api.models.run import RunRecord
    from quantdsl_backtest.platform_api.services.run_store import RunStore

    db_path = str(tmp_path / "runs.db")

    store = RunStore(db_path=db_path, retention=500)
    # Should be safe to call again
    store._ensure_schema()

    base = datetime(2025, 1, 1, 12, 0, 0)

    # Insert 510 runs for one strategy; expect prune to 500 newest.
    for i in range(510):
        rec = RunRecord(
            run_id=f"run-{i:04d}",
            strategy_id="s1",
            strategy_hash=f"hash-{i:04d}",
            status="pending",
            submitted_at=base + timedelta(seconds=i),
            params={"i": i},
        )
        store.insert_run(rec)

    res = store.list_runs(strategy_id="s1", limit=600, offset=0)
    assert res.total == 500
    assert len(res.runs) == 500

    # Newest first
    assert res.runs[0].run_id == "run-0509"
    assert res.runs[-1].run_id == "run-0010"  # first 10 pruned

    # Pagination
    res2 = store.list_runs(strategy_id="s1", limit=10, offset=0)
    assert [r.run_id for r in res2.runs] == [f"run-{i:04d}" for i in range(509, 499, -1)]


def test_run_store_update_and_get_roundtrip(tmp_path):
    from quantdsl_backtest.platform_api.models.run import RunRecord
    from quantdsl_backtest.platform_api.services.run_store import RunStore

    db_path = str(tmp_path / "runs.db")
    store = RunStore(db_path=db_path, retention=500)

    t0 = datetime(2025, 1, 1, 0, 0, 0)
    rec = RunRecord(
        run_id="rid-1",
        strategy_id="s1",
        strategy_hash="h1",
        status="pending",
        submitted_at=t0,
        params={"engine": "event_driven"},
    )
    store.insert_run(rec)

    store.update_run(
        "rid-1",
        status="succeeded",
        started_at=t0,
        ended_at=t0 + timedelta(seconds=2),
        duration_s=2.0,
        metrics={"sharpe": 1.23},
        error=None,
        artifacts_dir="outputs/runs/rid-1",
        reports_url="/reports/runs/rid-1/index.html",
    )

    got = store.get_run("rid-1")
    assert got is not None
    assert got.status == "succeeded"
    assert got.duration_s == 2.0
    assert got.metrics == {"sharpe": 1.23}
    assert got.artifacts_dir == "outputs/runs/rid-1"
    assert got.reports_url.endswith("/reports/runs/rid-1/index.html")


def test_list_strategies_summary_uses_last_run(tmp_path):
    from quantdsl_backtest.platform_api.models.run import RunRecord
    from quantdsl_backtest.platform_api.services.run_store import RunStore

    db_path = str(tmp_path / "runs.db")
    store = RunStore(db_path=db_path, retention=500)

    base = datetime(2025, 1, 1, 0, 0, 0)

    store.insert_run(
        RunRecord(
            run_id="r1",
            strategy_id="s1",
            strategy_hash="h1",
            status="failed",
            submitted_at=base,
            params={},
            metrics=None,
        )
    )
    store.insert_run(
        RunRecord(
            run_id="r2",
            strategy_id="s1",
            strategy_hash="h2",
            status="succeeded",
            submitted_at=base + timedelta(seconds=1),
            params={},
            metrics={"sharpe": 0.5},
        )
    )

    store.insert_run(
        RunRecord(
            run_id="r3",
            strategy_id="s2",
            strategy_hash="h3",
            status="pending",
            submitted_at=base,
            params={},
        )
    )

    sums = store.list_strategies_summary()
    assert [s.strategy_id for s in sums] == ["s1", "s2"]

    s1 = next(s for s in sums if s.strategy_id == "s1")
    assert s1.last_run_id == "r2"
    assert s1.last_status == "succeeded"
    assert s1.last_metrics == {"sharpe": 0.5}

    s2 = next(s for s in sums if s.strategy_id == "s2")
    assert s2.last_run_id == "r3"
    assert s2.last_status == "pending"

