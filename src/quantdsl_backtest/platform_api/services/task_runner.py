from __future__ import annotations

import datetime as _dt
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..models.run import RunRecord, RunStatus
from .run_store import RunStore


@dataclass(frozen=True)
class SubmitResult:
    run_id: str
    status: RunStatus


class TaskRunner:
    """Very small in-process task runner (skeleton).

    MVP scope (this step):
      - submit a run and transition status: pending -> running -> succeeded/failed
      - store transitions in RunStore
      - no real backtest execution yet

    Notes:
      - uses background threads to avoid blocking request handling
      - will be replaced by a ProcessPoolExecutor-based runner later
    """

    def __init__(
        self,
        *,
        run_store: RunStore,
        worker: Callable[[RunRecord], dict[str, Any]] | None = None,
    ) -> None:
        self._store = run_store
        self._worker = worker or (lambda rec: {"ok": True})

    def submit(
        self,
        *,
        strategy_id: str,
        strategy_hash: str,
        source_snapshot: str,
        params: dict[str, Any] | None = None,
        artifacts_dir: str | None = None,
        reports_url: str | None = None,
    ) -> SubmitResult:
        run_id = uuid.uuid4().hex
        now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)

        record = RunRecord(
            run_id=run_id,
            strategy_id=str(strategy_id),
            strategy_hash=str(strategy_hash),
            source_snapshot=str(source_snapshot),
            params=dict(params or {}),
            status="pending",
            submitted_at=now,
            artifacts_dir=artifacts_dir,
            reports_url=reports_url,
        )
        self._store.insert_run(record)

        t = threading.Thread(target=self._run_background, args=(run_id,), daemon=True)
        t.start()

        return SubmitResult(run_id=run_id, status="pending")

    def _run_background(self, run_id: str) -> None:
        started = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
        self._store.update_run(run_id, status="running", started_at=started)

        try:
            rec = self._store.get_run(run_id)
            if rec is None:
                return
            out = self._worker(rec)
            ended = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
            dt_s = max(0.0, (ended - started).total_seconds())
            self._store.update_run(run_id, status="succeeded", ended_at=ended, duration_s=dt_s, metrics=out)
        except Exception as exc:
            ended = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
            dt_s = max(0.0, (ended - started).total_seconds())
            self._store.update_run(run_id, status="failed", ended_at=ended, duration_s=dt_s, error=str(exc))
