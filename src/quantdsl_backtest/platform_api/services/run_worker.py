from __future__ import annotations

import contextlib
import io
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quantdsl_backtest.platform_api.models.run import RunRecord


def _ensure_reporting_output_dir(strategy: Any, out_dir: Path) -> None:
    """Best-effort set strategy.backtest.reporting.output_dir to out_dir."""

    try:
        bt = getattr(strategy, "backtest", None)
        if bt is None:
            return
        rep = getattr(bt, "reporting", None)
        if rep is None:
            # Minimal holder
            class _Rep:
                output_dir = str(out_dir)

            bt.reporting = _Rep()  # type: ignore[attr-defined]
        else:
            setattr(rep, "output_dir", str(out_dir))
    except Exception:
        return


def _write_config_resolved(strategy: Any, out_dir: Path) -> None:
    p = out_dir / "config_resolved.json"
    try:
        # Strategy is a dataclass in our DSL.
        data = asdict(strategy)  # type: ignore[arg-type]
    except Exception:
        # fallback
        data = {"repr": repr(strategy)}
    p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def execute_run_in_worker(rec: RunRecord) -> dict[str, Any]:
    """Process worker entrypoint.

    Inputs:
      - RunRecord including source_snapshot + artifacts_dir

    Side-effects:
      - Writes logs.txt
      - Writes config_resolved.json
      - Runs engine backtest and writes report artifacts into artifacts_dir

    Returns:
      - dict of metrics (best-effort)

    This function must be top-level picklable.
    """

    if not rec.artifacts_dir:
        raise RuntimeError("RunRecord.artifacts_dir is required")

    out_dir = Path(rec.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "logs.txt"

    buf = io.StringIO()

    # Make sure subprocess has deterministic cwd for relative paths.
    try:
        os.chdir(str(Path.cwd()))
    except Exception:
        pass

    try:
        from .strategy_exec import build_strategy_from_source

        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            strat = build_strategy_from_source(source=rec.source_snapshot or "", strategy_id=rec.strategy_id).strategy
            _ensure_reporting_output_dir(strat, out_dir)
            _write_config_resolved(strat, out_dir)

            from quantdsl_backtest.engine.backtest_runner import run_backtest

            result = run_backtest(strat)

        metrics = dict(getattr(result, "metrics", {}) or {})
        return metrics

    except Exception:
        tb = traceback.format_exc()
        buf.write("\n[run_worker] Exception:\n")
        buf.write(tb)
        # Re-raise with the full traceback so the parent process can mark the run as failed.
        raise

    finally:
        try:
            log_path.write_text(buf.getvalue(), encoding="utf-8", errors="replace")
        except Exception:
            pass

