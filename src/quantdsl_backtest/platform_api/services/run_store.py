from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from ..models.run import RunRecord, RunStatus, StrategySummary


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    # ISO-8601, sortable
    return dt.replace(microsecond=0).isoformat()


def _dt_from_str(v: str | None) -> datetime | None:
    if not v:
        return None
    # datetime.fromisoformat supports our output
    return datetime.fromisoformat(v)


@dataclass(frozen=True)
class RunListResult:
    runs: list[RunRecord]
    total: int


class RunStore:
    """Local-first run index backed by SQLite.

    Contract:
      - stdlib-only (sqlite3)
      - safe to create/open repeatedly (schema ensured)
      - deterministic ordering: newest submitted_at first
    """

    def __init__(
        self,
        *,
        db_path: str,
        retention: int = 500,
        connect_timeout_s: float = 5.0,
        enable_wal: bool = True,
    ) -> None:
        self._db_path = str(db_path)
        self._retention = int(retention)
        self._connect_timeout_s = float(connect_timeout_s)
        self._enable_wal = bool(enable_wal)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        # Note: `timeout` handles the common "database is locked" transient.
        conn = sqlite3.connect(self._db_path, timeout=self._connect_timeout_s)
        conn.row_factory = sqlite3.Row

        # WAL improves concurrent read/write behavior for file-backed DBs.
        # It is not supported/meaningful for in-memory DBs.
        try:
            if self._enable_wal and self._db_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            # Best-effort; never break calling code.
            pass

        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id          TEXT PRIMARY KEY,
                  strategy_id     TEXT NOT NULL,
                  strategy_hash   TEXT NOT NULL,
                  source_snapshot TEXT,
                  params_json     TEXT,
                  status          TEXT NOT NULL DEFAULT 'pending',
                  submitted_at    TEXT NOT NULL,
                  started_at      TEXT,
                  ended_at        TEXT,
                  duration_s      REAL,
                  metrics_json    TEXT,
                  error           TEXT,
                  artifacts_dir   TEXT,
                  reports_url     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_strategy   ON runs(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_submitted  ON runs(submitted_at DESC);
                """
            )

    def insert_run(self, record: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                  run_id, strategy_id, strategy_hash, source_snapshot,
                  params_json, status, submitted_at, started_at, ended_at,
                  duration_s, metrics_json, error, artifacts_dir, reports_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.strategy_id,
                    record.strategy_hash,
                    record.source_snapshot,
                    json.dumps(record.params or {}),
                    record.status,
                    _dt_to_str(record.submitted_at),
                    _dt_to_str(record.started_at),
                    _dt_to_str(record.ended_at),
                    record.duration_s,
                    json.dumps(record.metrics) if record.metrics is not None else None,
                    record.error,
                    record.artifacts_dir,
                    record.reports_url,
                ),
            )

        self._prune(strategy_id=record.strategy_id)

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        duration_s: float | None = None,
        metrics: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        artifacts_dir: str | None = None,
        reports_url: str | None = None,
    ) -> None:
        sets: list[str] = []
        vals: list[Any] = []

        def _set(col: str, v: Any) -> None:
            sets.append(f"{col} = ?")
            vals.append(v)

        if status is not None:
            _set("status", status)
        if started_at is not None:
            _set("started_at", _dt_to_str(started_at))
        if ended_at is not None:
            _set("ended_at", _dt_to_str(ended_at))
        if duration_s is not None:
            _set("duration_s", float(duration_s))
        if metrics is not None:
            _set("metrics_json", json.dumps(metrics))
        if error is not None:
            _set("error", error)
        if artifacts_dir is not None:
            _set("artifacts_dir", artifacts_dir)
        if reports_url is not None:
            _set("reports_url", reports_url)

        if not sets:
            return

        vals.append(run_id)

        with self._connect() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_id = ?", tuple(vals))

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row is not None else None

    def list_runs(
        self,
        *,
        strategy_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RunListResult:
        limit_i = max(1, int(limit))
        offset_i = max(0, int(offset))

        where: list[str] = []
        vals: list[Any] = []
        if strategy_id:
            where.append("strategy_id = ?")
            vals.append(strategy_id)
        if status:
            where.append("status = ?")
            vals.append(status)

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) AS n FROM runs{where_sql}", tuple(vals)).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM runs{where_sql} ORDER BY submitted_at DESC LIMIT ? OFFSET ?",
                tuple(vals + [limit_i, offset_i]),
            ).fetchall()

        return RunListResult(runs=[self._row_to_run(r) for r in rows], total=total)

    def list_strategies_summary(self) -> list[StrategySummary]:
        # For each strategy, pick the most recent submitted_at.
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r1.*
                FROM runs r1
                INNER JOIN (
                  SELECT strategy_id, MAX(submitted_at) AS max_submitted
                  FROM runs
                  GROUP BY strategy_id
                ) r2
                ON r1.strategy_id = r2.strategy_id AND r1.submitted_at = r2.max_submitted
                ORDER BY r1.strategy_id ASC
                """
            ).fetchall()

        out: list[StrategySummary] = []
        for r in rows:
            run = self._row_to_run(r)
            out.append(
                StrategySummary(
                    strategy_id=run.strategy_id,
                    last_run_id=run.run_id,
                    last_status=run.status,
                    last_run_at=run.ended_at or run.submitted_at,
                    last_metrics=run.metrics,
                )
            )
        return out

    def _prune(self, *, strategy_id: str) -> None:
        if self._retention <= 0:
            return
        with self._connect() as conn:
            n = int(conn.execute("SELECT COUNT(*) FROM runs WHERE strategy_id = ?", (strategy_id,)).fetchone()[0])
            if n <= self._retention:
                return
            to_delete = n - self._retention
            # Delete oldest by submitted_at
            rows = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE strategy_id = ?
                ORDER BY submitted_at ASC
                LIMIT ?
                """,
                (strategy_id, int(to_delete)),
            ).fetchall()
            if not rows:
                return
            conn.executemany("DELETE FROM runs WHERE run_id = ?", [(r[0],) for r in rows])

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        params_raw = row["params_json"]
        metrics_raw = row["metrics_json"]

        # Defensive: SQLite stores status as TEXT; enforce allowed values.
        raw_status = str(row["status"])
        if raw_status not in ("pending", "running", "succeeded", "failed"):
            raw_status = "failed"

        return RunRecord(
            run_id=str(row["run_id"]),
            strategy_id=str(row["strategy_id"]),
            strategy_hash=str(row["strategy_hash"]),
            source_snapshot=row["source_snapshot"],
            params=json.loads(params_raw) if params_raw else {},
            status=raw_status,
            submitted_at=_dt_from_str(row["submitted_at"]) or datetime.fromtimestamp(0),
            started_at=_dt_from_str(row["started_at"]),
            ended_at=_dt_from_str(row["ended_at"]),
            duration_s=float(row["duration_s"]) if row["duration_s"] is not None else None,
            metrics=json.loads(metrics_raw) if metrics_raw else None,
            error=row["error"],
            artifacts_dir=row["artifacts_dir"],
            reports_url=row["reports_url"],
        )
