"""Signal analytics for books sold at the next open: forward returns and attribution over the night only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.engine.analytics.attribution import contrib_return_panel
from quantdsl_backtest.engine.analytics.signal_analytics import compute_forward_returns
from quantdsl_backtest.engine.analytics.types import SignalAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import compute_signal_analytics_and_attribution

IDX = pd.date_range("2024-01-01", periods=6, freq="D")
COLS = ["A", "B", "C", "D"]
CLOSE = pd.DataFrame(100.0, index=IDX, columns=COLS)
# Every night A gains 4%, B 2%, C 0%, D loses 2%; the day gives it all back (the close is always 100).
NIGHT = pd.Series({"A": 0.04, "B": 0.02, "C": 0.0, "D": -0.02})
OPEN = pd.DataFrame(np.tile(100.0 * (1.0 + NIGHT.to_numpy()), (len(IDX), 1)), index=IDX, columns=COLS)
SIGNAL = pd.DataFrame(np.tile([4.0, 3.0, 2.0, 1.0], (len(IDX), 1)), index=IDX, columns=COLS)
EMPTY_TRADES = pd.DataFrame(columns=["datetime", "instrument", "side", "quantity", "price", "notional", "commission"])


def test_forward_returns_to_a_later_open():
    fwd = compute_forward_returns(CLOSE, [1, 2], open_=OPEN)
    assert fwd[1].iloc[0].tolist() == pytest.approx(NIGHT.tolist())
    assert fwd[2].iloc[0].tolist() == pytest.approx(NIGHT.tolist())  # the open two bars on, over today's close
    assert fwd[1].iloc[-1].isna().all()
    plain = compute_forward_returns(CLOSE, [1])
    assert plain[1].iloc[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_overnight_contribution_is_the_night_on_yesterdays_weights():
    w = pd.DataFrame(0.0, index=IDX, columns=COLS)
    w["A"] = 1.0
    night = contrib_return_panel(w, CLOSE, open_=OPEN)
    assert night["A"].iloc[1:].tolist() == pytest.approx([0.04] * 5)
    assert night.iloc[0].tolist() == [0.0, 0.0, 0.0, 0.0]  # nothing was held into the first bar
    assert contrib_return_panel(w, CLOSE)["A"].abs().sum() == 0.0  # close to close the price never moves


def _run(cfg: SignalAnalyticsConfig, weights: pd.DataFrame, open_prices=OPEN):
    return compute_signal_analytics_and_attribution(
        prices=CLOSE,
        signal_panels={"sig": SIGNAL},
        realized_weights=weights,
        trades=EMPTY_TRADES,
        cfg=cfg,
        open_prices=open_prices,
    )


def test_the_signal_is_judged_on_the_night():
    w = pd.DataFrame(0.25, index=IDX, columns=COLS)
    night_cfg = SignalAnalyticsConfig(signals=["sig"], horizons=[1], quantiles=4, signal_delay_bars=0, forward_return="close_to_open")
    reports, _ = _run(night_cfg, w)
    rep = reports["sig"]
    assert rep.rank_ic[1].dropna().tolist() == pytest.approx([1.0] * 5)  # the signal orders the night exactly
    by_q = rep.mean_fwd_ret_by_q[1].dropna()
    assert by_q.iloc[0].tolist() == pytest.approx([-0.02, 0.0, 0.02, 0.04])
    assert rep.ls_fwd_ret[1].dropna().iloc[0] == pytest.approx(0.06)

    day_cfg = SignalAnalyticsConfig(signals=["sig"], horizons=[1], quantiles=4, signal_delay_bars=0)
    flat, _ = _run(day_cfg, w, open_prices=None)
    assert flat["sig"].ls_fwd_ret[1].dropna().abs().sum() == 0.0  # close to close there is nothing to find


def test_overnight_attribution_goes_to_the_bucket_the_name_was_bought_in():
    # The signal flips every bar, so today's bucket is not the bucket the position was bought in.
    flip = SIGNAL.copy()
    flip.iloc[1::2] = flip.iloc[1::2, ::-1].to_numpy()
    held = flip.eq(flip.max(axis=1), axis=0).astype(float)  # all in the top name of each bar
    cfg = SignalAnalyticsConfig(signals=["sig"], horizons=[1], quantiles=4, signal_delay_bars=0, forward_return="close_to_open")
    _, attribs = compute_signal_analytics_and_attribution(
        prices=CLOSE, signal_panels={"sig": flip}, realized_weights=held, trades=EMPTY_TRADES, cfg=cfg, open_prices=OPEN
    )
    top = attribs["sig"].contrib_ret_by_q[4]
    # Even bars buy A (+4% a night), odd bars buy D (-2%): the whole night's result sits in the top bucket.
    assert top.iloc[1:].tolist() == pytest.approx([0.04, -0.02, 0.04, -0.02, 0.04])
    assert attribs["sig"].contrib_ret_by_q[1].iloc[1:].fillna(0.0).abs().sum() == 0.0


def test_bad_arguments_are_refused():
    w = pd.DataFrame(0.25, index=IDX, columns=COLS)
    with pytest.raises(ValueError, match="open"):
        _run(SignalAnalyticsConfig(signals=["sig"], horizons=[1], forward_return="close_to_open"), w, open_prices=None)
    with pytest.raises(ValueError, match="forward_return"):
        _run(SignalAnalyticsConfig(signals=["sig"], horizons=[1], forward_return="open_to_close"), w)


def test_only_simulated_bars_are_reported_when_the_signal_has_earlier_history():
    longer = pd.date_range("2023-12-25", periods=len(IDX) + 7, freq="D")
    history = pd.DataFrame(np.tile([4.0, 3.0, 2.0, 1.0], (len(longer), 1)), index=longer, columns=COLS)
    cfg = SignalAnalyticsConfig(signals=["sig"], horizons=[1], quantiles=4, signal_delay_bars=1, forward_return="close_to_open")
    reports, _ = compute_signal_analytics_and_attribution(
        prices=CLOSE,
        signal_panels={"sig": history},
        realized_weights=pd.DataFrame(0.25, index=IDX, columns=COLS),
        trades=EMPTY_TRADES,
        cfg=cfg,
        open_prices=OPEN,
    )
    rep = reports["sig"]
    assert list(rep.coverage.index) == list(IDX)
    assert rep.coverage.iloc[0] == 1.0  # the lagged value on the first simulated bar comes from the history
