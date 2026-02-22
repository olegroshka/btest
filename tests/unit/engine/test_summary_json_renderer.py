from __future__ import annotations

import json
from pathlib import Path


def test_summary_json_renderer_writes_summary(tmp_path):
    import pandas as pd

    from quantdsl_backtest.engine.backtest_runner import ReportingContext, SummaryJsonRenderer
    from quantdsl_backtest.engine.results import BacktestResult

    idx = pd.to_datetime(["2025-01-01", "2025-01-02"])

    equity = pd.Series([1.0, 1.1], index=idx, name="equity")
    returns = pd.Series([0.0, 0.1], index=idx, name="returns")
    cash = pd.Series([1.0, 1.0], index=idx, name="cash")

    zeros = pd.Series([0.0, 0.0], index=idx)

    positions = pd.DataFrame(index=idx)
    weights = pd.DataFrame(index=idx)
    trades = pd.DataFrame(columns=["datetime", "instrument", "side", "quantity", "price"]).copy()

    result = BacktestResult(
        equity=equity,
        returns=returns,
        cash=cash,
        gross_exposure=zeros,
        net_exposure=zeros,
        long_exposure=zeros,
        short_exposure=zeros,
        leverage=zeros,
        positions=positions,
        weights=weights,
        trades=trades,
        metrics={"total_return": 0.10, "sharpe": 1.23},
        start_date=idx[0],
        end_date=idx[-1],
        benchmark=None,
        metadata={"strategy_name": "s1"},
    )

    class _DummyStrategy:
        name = "s1"

        class backtest:
            engine = "event_driven"

    ctx = ReportingContext(strategy=_DummyStrategy(), output_dir=Path(tmp_path))

    # Create a couple artifacts to ensure artifacts list picks them up
    (tmp_path / "index.html").write_text("<html/>", encoding="utf-8")
    (tmp_path / "equity.parquet").write_bytes(b"x")

    SummaryJsonRenderer().render(result, ctx)

    p = tmp_path / "summary.json"
    assert p.exists()
    j = json.loads(p.read_text(encoding="utf-8"))

    assert j["strategy_name"] == "s1"
    assert j["engine"] == "event_driven"
    assert "metrics" in j
    assert j["metrics"]["sharpe"] == 1.23
    assert "artifacts" in j
    assert "index.html" in j["artifacts"]
