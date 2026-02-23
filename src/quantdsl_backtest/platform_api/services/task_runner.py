from __future__ import annotations

import datetime as _dt
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..models.run import RunRecord, RunStatus
from .run_store import RunStore
from .run_paths import run_output_dir, run_reports_url


@dataclass(frozen=True)
class SubmitResult:
    run_id: str
    status: RunStatus


class TaskRunner:
    """Very small in-process task runner (skeleton).

    MVP scope:
      - submit a run and transition status: pending -> running -> succeeded/failed
      - store transitions in RunStore
      - create run-scoped artifacts dir under outputs/runs/<run_id>

    Notes:
      - uses background threads to avoid blocking request handling
      - real backtest execution is a later milestone (process-based runner)
    """

    def __init__(
        self,
        *,
        run_store: RunStore,
        worker: Callable[[RunRecord], dict[str, Any]] | None = None,
        enable_process_pool: bool | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._store = run_store

        # Default worker runs the real backtest via execute_run_in_worker.
        # Only fall back to the simple stub if the real worker can't be imported.
        if worker is not None:
            self._worker = worker
        else:
            try:
                from .run_worker import execute_run_in_worker as _real_worker
                self._worker = _real_worker
            except Exception:
                self._worker = lambda rec: {"ok": True}

        if enable_process_pool is None:
            enable_process_pool = str(os.environ.get("QDSL_PLATFORM_RUNNER_PROCESS_POOL", "0")).strip() in {"1", "true", "True"}
        self._enable_process_pool = bool(enable_process_pool)
        self._max_workers = int(max_workers) if max_workers is not None else 1

        self._pool = None
        if self._enable_process_pool:
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)

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

        out_dir = run_output_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        record = RunRecord(
            run_id=run_id,
            strategy_id=str(strategy_id),
            strategy_hash=str(strategy_hash),
            source_snapshot=str(source_snapshot),
            params=dict(params or {}),
            status="pending",
            submitted_at=now,
            artifacts_dir=str(out_dir),
            reports_url=run_reports_url(run_id),
        )
        self._store.insert_run(record)

        t = threading.Thread(target=self._run_background, args=(run_id,), daemon=True)
        t.start()

        return SubmitResult(run_id=run_id, status="pending")

    def _run_background(self, run_id: str) -> None:
        import traceback as _tb

        started = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
        self._store.update_run(run_id, status="running", started_at=started)

        rec = None
        try:
            rec = self._store.get_run(run_id)
            if rec is None:
                return

            if self._pool is not None:
                from .run_worker import execute_run_in_worker

                fut = self._pool.submit(execute_run_in_worker, rec)
                out = fut.result()
            else:
                out = self._worker(rec)

            ended = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
            dt_s = max(0.0, (ended - started).total_seconds())
            self._store.update_run(run_id, status="succeeded", ended_at=ended, duration_s=dt_s, metrics=out)
        except Exception as exc:
            ended = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
            dt_s = max(0.0, (ended - started).total_seconds())
            err_msg = str(exc)
            # Append traceback to logs.txt (don't overwrite — worker may have written partial logs).
            try:
                artifacts = getattr(rec, "artifacts_dir", None) if rec is not None else None
                if artifacts:
                    from pathlib import Path
                    log_path = Path(artifacts) / "logs.txt"
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n[task_runner] Run failed:\n{_tb.format_exc()}\n")
            except Exception:
                pass
            self._store.update_run(run_id, status="failed", ended_at=ended, duration_s=dt_s, error=err_msg)
