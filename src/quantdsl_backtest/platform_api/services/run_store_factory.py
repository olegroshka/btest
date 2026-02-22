from __future__ import annotations

from pathlib import Path

from .run_store import RunStore


def default_runs_db_path() -> Path:
    """Default location for the runs index DB (gitignored)."""

    return Path.cwd() / "local_cache" / "platform_meta" / "runs.db"


def create_default_run_store(*, retention: int = 500) -> RunStore:
    """Create a RunStore at the default path, ensuring parent dirs exist."""

    p = default_runs_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    return RunStore(db_path=str(p), retention=int(retention))

