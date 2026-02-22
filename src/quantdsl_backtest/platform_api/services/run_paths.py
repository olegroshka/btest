from __future__ import annotations

from pathlib import Path


def default_runs_root_dir() -> Path:
    """Root directory for run-scoped artifacts (gitignored).

    Note: unit tests enforce 'no writes under outputs/' to keep the repo clean.
    """

    return Path.cwd() / "local_cache" / "platform_runs" / "runs"


def run_output_dir(run_id: str) -> Path:
    return default_runs_root_dir() / str(run_id)


def run_reports_url(run_id: str) -> str:
    # Served by platform ui route: /reports/*
    return f"/reports/runs/{run_id}/index.html"
