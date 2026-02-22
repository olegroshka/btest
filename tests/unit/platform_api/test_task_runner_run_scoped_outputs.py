from __future__ import annotations

import json
from pathlib import Path


def test_task_runner_creates_run_scoped_output_dir_and_updates_store(tmp_path):
    from quantdsl_backtest.platform_api.models.run import RunRecord
    from quantdsl_backtest.platform_api.services.run_store import RunStore
    from quantdsl_backtest.platform_api.services.task_runner import TaskRunner

    db_path = str(tmp_path / "runs.db")
    store = RunStore(db_path=db_path, retention=500, enable_wal=False)

    # Worker stub to avoid running real backtest in unit test
    def _worker(rec: RunRecord):
        assert rec.artifacts_dir
        out = Path(rec.artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text("<html/>", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"sharpe": 1.0}

    runner = TaskRunner(run_store=store, worker=_worker)

    res = runner.submit(strategy_id="s1", strategy_hash="h1", source_snapshot="print('x')\n")
    rid = res.run_id

    # Poll store until succeeded
    import time

    t0 = time.time()
    while True:
        r = store.get_run(rid)
        assert r is not None
        if r.status in ("succeeded", "failed"):
            break
        if time.time() - t0 > 2.0:
            raise AssertionError("run did not finish")
        time.sleep(0.01)

    r = store.get_run(rid)
    assert r is not None
    if r.status != "succeeded":
        raise AssertionError(f"expected succeeded, got {r.status} error={r.error}")

    assert r.artifacts_dir
    assert Path(r.artifacts_dir).exists()
    assert (Path(r.artifacts_dir) / "index.html").exists()
    assert r.reports_url and r.reports_url.endswith(f"/reports/runs/{rid}/index.html")
