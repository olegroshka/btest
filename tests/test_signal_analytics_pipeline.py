from __future__ import annotations

import numpy as np
import pandas as pd

from quantdsl_backtest.engine.backtest_runner import compute_signal_analytics_and_attribution
from quantdsl_backtest.engine.analytics.types import SignalAnalyticsConfig


def _toy_prices(n: int = 15) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    cols = ["A", "B", "C", "D"]
    base = np.linspace(100.0, 110.0, n)
    prices = pd.DataFrame({c: base + i for i, c in enumerate(cols)}, index=idx)
    return prices


def test_compute_signal_analytics_and_attribution_smoke() -> None:
    prices = _toy_prices(20)

    # simple monotone signal per column; no NaNs
    signal = prices.rank(axis=1)
    signal_panels = {"sig": signal}

    # realized weights: equal weight long
    w = pd.DataFrame(0.25, index=prices.index, columns=prices.columns)

    trades = pd.DataFrame(
        columns=[
            "datetime",
            "instrument",
            "side",
            "quantity",
            "price",
            "notional",
            "slippage_bps",
            "commission",
            "fees",
            "realized_pnl",
        ]
    )

    cfg = SignalAnalyticsConfig(signals=["sig"], horizons=[1, 5], quantiles=4, signal_delay_bars=1)

    reports, attribs = compute_signal_analytics_and_attribution(
        prices=prices,
        signal_panels=signal_panels,
        realized_weights=w,
        trades=trades,
        cfg=cfg,
    )

    assert set(reports.keys()) == {"sig"}
    assert set(attribs.keys()) == {"sig"}

    rep = reports["sig"]
    assert rep.quantile is not None
    assert list(rep.rank_ic.keys()) == [1, 5]
    assert list(rep.mean_fwd_ret_by_q.keys()) == [1, 5]
    assert list(rep.ls_fwd_ret.keys()) == [1, 5]

    # basic shapes / alignment
    assert rep.coverage.index.equals(prices.index)
    assert rep.quantile.shape == prices.shape

    attr = attribs["sig"]
    assert isinstance(attr.contrib_ret_by_q, pd.DataFrame)
    assert len(attr.contrib_ret_ls) == len(attr.contrib_ret_by_q)


def test_compute_signal_analytics_and_attribution_missing_signal_is_ignored() -> None:
    prices = _toy_prices(10)
    signal_panels = {"present": prices.rank(axis=1)}
    w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    trades = pd.DataFrame()

    cfg = SignalAnalyticsConfig(signals=["missing"], horizons=[1], quantiles=4, signal_delay_bars=1)

    reports, attribs = compute_signal_analytics_and_attribution(
        prices=prices,
        signal_panels=signal_panels,
        realized_weights=w,
        trades=trades,
        cfg=cfg,
    )

    assert reports == {}
    assert attribs == {}

