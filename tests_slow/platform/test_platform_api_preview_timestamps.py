from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = [pytest.mark.slow]


@pytest.mark.unit
def test_catalog_preview_emits_parseable_ts(tmp_path, monkeypatch):
    """Backend contract: preview endpoint must emit a robust `ts` field.

    The UI candlestick chart depends on `head[].ts` being present and in a
    consistent format. We standardize on UTC Zulu timestamps:
      YYYY-MM-DDTHH:MM:SSZ
    """

    # Isolate cache for the test
    arctic_root: Path = tmp_path / "arctic_preview_ts"
    arctic_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QUANTDSL_ARCTIC_URI", f"lmdb://{arctic_root.as_posix()}")

    # Populate cache by running a small example strategy
    from quantdsl_backtest.examples.lagging_indecies import build_strategy
    from quantdsl_backtest.engine.backtest_runner import run_backtest

    strat = build_strategy()
    # best-effort: some configs may not have reporting.output_dir
    try:
        strat.backtest.reporting.output_dir = None  # type: ignore[attr-defined]
    except Exception:
        pass
    run_backtest(strat)

    from fastapi.testclient import TestClient
    from quantdsl_backtest.platform_api.main import app

    client = TestClient(app)

    cat = client.get("/api/catalog").json()
    assert cat.get("libraries"), "Expected catalog to contain at least one library"
    lib0 = cat["libraries"][0]
    library = lib0["library"]
    assert lib0.get("symbols"), "Expected catalog library to contain at least one symbol"
    symbol = lib0["symbols"][0]["symbol"]

    resp = client.get(f"/api/catalog/preview/{library}", params={"symbol": symbol, "head": 6, "tail": 0})
    assert resp.status_code == 200, resp.text
    data = resp.json()

    head = data.get("head") or []
    assert head, "Expected non-empty head from preview"

    row0 = head[0]
    assert "ts" in row0, f"Expected 'ts' in preview row. keys={list(row0.keys())}"

    ts = row0.get("ts")
    assert isinstance(ts, str)

    import re

    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), f"Unexpected ts format: {ts!r}"

