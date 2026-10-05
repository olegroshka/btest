"""OptimizedWeight book weighting + the pure optimize_weights() function."""
import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.portfolio import (
    Book,
    BottomN,
    EqualWeight,
    LongShortPortfolio,
    OptimizedWeight,
    TopN,
)
from quantdsl_backtest.engine.portfolio_engine import (
    PRICES_PANEL_KEY,
    compute_target_weights_for_date,
)
from quantdsl_backtest.engine.portfolio_optimizer import (
    optimize_weights,
    optimize_weights_series,
)

skfolio = pytest.importorskip("skfolio", reason="skfolio not installed")


def _returns(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    # A: low vol, B: mid, C: high vol, D: high vol & correlated with C
    a = rng.normal(0.0002, 0.005, n)
    b = rng.normal(0.0003, 0.010, n)
    c = rng.normal(0.0004, 0.020, n)
    d = 0.8 * c + rng.normal(0.0, 0.010, n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=idx)


# --------------------------------------------------------------------------- #
# optimize_weights
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method", ["equal", "inverse_vol", "min_variance", "risk_parity", "hrp", "max_sharpe", "min_cvar"]
)
def test_optimize_weights_sums_to_one_and_long_only(method):
    w = optimize_weights(_returns(), method)
    assert list(w.index) == ["A", "B", "C", "D"]
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_inverse_vol_orders_by_vol():
    w = optimize_weights(_returns(), "inverse_vol")
    assert w["A"] > w["B"] > w["C"]


def test_min_variance_prefers_low_vol_asset():
    w = optimize_weights(_returns(), "min_variance")
    assert w.idxmax() == "A"
    # correlated high-vol pair gets least weight
    assert w["C"] + w["D"] < w["A"]


def test_risk_parity_equalises_risk_contribution():
    r = _returns()
    w = optimize_weights(r, "risk_parity")
    cov = r.cov().values
    rc = w.values * (cov @ w.values)
    rc = rc / rc.sum()
    assert np.allclose(rc, 0.25, atol=0.03)


def test_max_weight_bound_respected():
    w = optimize_weights(_returns(), "min_variance", max_weight=0.35)
    assert (w <= 0.35 + 1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_short_window_falls_back_to_inverse_vol():
    r = _returns(n=10)
    w_fb = optimize_weights(r, "hrp", min_periods=20)
    w_iv = optimize_weights(r, "inverse_vol")
    pd.testing.assert_series_equal(w_fb, w_iv)


def test_all_nan_column_gets_zero_weight():
    r = _returns()
    r["E"] = np.nan
    w = optimize_weights(r, "min_variance")
    assert w["E"] == 0.0
    assert abs(w.drop("E").sum() - 1.0) < 1e-6


def test_single_asset():
    r = _returns()[["A"]]
    w = optimize_weights(r, "risk_parity")
    assert w["A"] == pytest.approx(1.0)


def test_optimize_weights_series_rolls_without_lookahead():
    r = _returns()
    prices = (1 + r).cumprod() * 100
    rebal = prices.index[[100, 200, 300]]
    W = optimize_weights_series(prices, rebal, "inverse_vol", lookback_bars=60)
    assert W.shape == (3, 4)
    # manual check for the middle date: window = rows 141..200 inclusive
    win = prices.pct_change().iloc[141:201]
    pd.testing.assert_series_equal(
        W.loc[rebal[1]], optimize_weights(win, "inverse_vol"), check_names=False
    )


# --------------------------------------------------------------------------- #
# OptimizedWeight inside a Book, through compute_target_weights_for_date
# --------------------------------------------------------------------------- #
def _ls_portfolio(long_w, short_w, delay=0) -> LongShortPortfolio:
    return LongShortPortfolio(
        long_book=Book("long", TopN(factor_name="rank", n=3), long_w),
        short_book=Book("short", BottomN(factor_name="rank", n=1), short_w),
        rebalance_frequency="1d",
        signal_delay_bars=delay,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=1.0,
    )


def _signals_with_prices(r: pd.DataFrame) -> dict:
    prices = (1 + r).cumprod() * 100
    # rank: A,B,C high (long), D lowest (short)
    rank = pd.DataFrame(
        np.tile([0.9, 0.8, 0.7, 0.1], (len(r), 1)), index=r.index, columns=r.columns
    )
    return {
        "rank": rank,
        "long_candidates": pd.DataFrame(True, index=r.index, columns=r.columns),
        "short_candidates": pd.DataFrame(True, index=r.index, columns=r.columns),
        PRICES_PANEL_KEY: prices,
    }


def test_optimized_weight_book_long_side_matches_optimizer():
    r = _returns()
    sigs = _signals_with_prices(r)
    dt = r.index[-1]
    pf = _ls_portfolio(OptimizedWeight(method="inverse_vol", lookback_bars=100), EqualWeight())
    tw = compute_target_weights_for_date(dt, pf, sigs, pd.Series(0.0, index=r.columns))

    # long gross = 1.0, split by inverse vol over the trailing 100 bars ending at dt
    win = sigs[PRICES_PANEL_KEY].pct_change().loc[:dt].iloc[-100:][["A", "B", "C"]]
    expected = optimize_weights(win, "inverse_vol")
    for n in ["A", "B", "C"]:
        assert tw[n] == pytest.approx(expected[n], abs=1e-9)
    assert tw["D"] == pytest.approx(-1.0)
    assert tw.abs().sum() == pytest.approx(2.0)


def test_optimized_weight_respects_signal_delay():
    r = _returns()
    sigs = _signals_with_prices(r)
    dt = r.index[-1]
    pf = _ls_portfolio(OptimizedWeight(method="inverse_vol", lookback_bars=50), EqualWeight(), delay=1)
    tw = compute_target_weights_for_date(dt, pf, sigs, pd.Series(0.0, index=r.columns))

    sig_date = r.index[-2]
    win = sigs[PRICES_PANEL_KEY].pct_change().loc[:sig_date].iloc[-50:][["A", "B", "C"]]
    expected = optimize_weights(win, "inverse_vol")
    for n in ["A", "B", "C"]:
        assert tw[n] == pytest.approx(expected[n], abs=1e-9)


def test_optimized_weight_hrp_runs_in_book():
    r = _returns()
    sigs = _signals_with_prices(r)
    pf = _ls_portfolio(OptimizedWeight(method="hrp", lookback_bars=252, max_weight=0.6), EqualWeight())
    tw = compute_target_weights_for_date(r.index[-1], pf, sigs, pd.Series(0.0, index=r.columns))
    longs = tw[["A", "B", "C"]]
    assert longs.sum() == pytest.approx(1.0)
    assert (longs <= 0.6 + 1e-6).all()
    assert (longs > 0).all()


def test_optimized_weight_without_price_panel_degrades_to_equal():
    r = _returns()
    sigs = _signals_with_prices(r)
    sigs.pop(PRICES_PANEL_KEY)
    pf = _ls_portfolio(OptimizedWeight(method="min_variance"), EqualWeight())
    tw = compute_target_weights_for_date(r.index[-1], pf, sigs, pd.Series(0.0, index=r.columns))
    for n in ["A", "B", "C"]:
        assert tw[n] == pytest.approx(1.0 / 3)


def test_optimized_weight_explicit_returns_signal():
    r = _returns()
    sigs = _signals_with_prices(r)
    sigs.pop(PRICES_PANEL_KEY)
    sigs["my_rets"] = r
    pf = _ls_portfolio(
        OptimizedWeight(method="inverse_vol", lookback_bars=100, returns_signal="my_rets"),
        EqualWeight(),
    )
    dt = r.index[-1]
    tw = compute_target_weights_for_date(dt, pf, sigs, pd.Series(0.0, index=r.columns))
    expected = optimize_weights(r.loc[:dt].iloc[-100:][["A", "B", "C"]], "inverse_vol")
    for n in ["A", "B", "C"]:
        assert tw[n] == pytest.approx(expected[n], abs=1e-9)
