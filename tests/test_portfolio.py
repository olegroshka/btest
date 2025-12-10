import numpy as np
import pandas as pd

from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight, TurnoverLimit
from quantdsl_backtest.engine.portfolio_engine import compute_target_weights_for_date


def _make_signals(index: pd.DatetimeIndex, columns: list[str], rank_rows: list[list[float]],
                  long_mask_rows: list[list[bool]] | None = None,
                  short_mask_rows: list[list[bool]] | None = None) -> dict[str, pd.DataFrame]:
    rank = pd.DataFrame(rank_rows, index=index, columns=columns, dtype="float64")
    if long_mask_rows is None:
        long_mask_rows = [[True] * len(columns) for _ in index]
    if short_mask_rows is None:
        short_mask_rows = [[True] * len(columns) for _ in index]
    long_mask = pd.DataFrame(long_mask_rows, index=index, columns=columns).astype(bool)
    short_mask = pd.DataFrame(short_mask_rows, index=index, columns=columns).astype(bool)
    return {
        "rank": rank,
        "long_candidates": long_mask,
        "short_candidates": short_mask,
    }


def _portfolio(n_long=2, n_short=2, signal_delay=0, gross=2.0, net=0.0, max_abs=1.0, turnover: TurnoverLimit | None = None) -> LongShortPortfolio:
    long_book = Book(name="long", selector=TopN(factor_name="rank", n=n_long), weighting=EqualWeight())
    short_book = Book(name="short", selector=BottomN(factor_name="rank", n=n_short), weighting=EqualWeight())
    return LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        signal_delay_bars=signal_delay,
        target_gross_leverage=gross,
        target_net_exposure=net,
        max_abs_weight_per_name=max_abs,
        turnover_limit=turnover,
    )


def test_compute_target_weights_basic_selection_equal_weights():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    names = ["A", "B", "C", "D"]
    # Clear ranking on last date: A>B>C>D
    ranks = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]]
    sigs = _make_signals(idx, names, ranks)

    portfolio = _portfolio(n_long=2, n_short=2, gross=2.0, net=0.0, max_abs=1.0)
    prev_w = pd.Series(0.0, index=names)

    # Rebalance at last date
    dt = idx[-1]
    tw = compute_target_weights_for_date(dt, portfolio, sigs, prev_w)

    # long_gross=(L+N)/2=1, short_gross=(L-N)/2=1
    # Top 2 by rank: A,B (since row sorted descending A 0.9, B 0.8)
    # Bottom 2: C,D (0.7,0.6)
    expected = pd.Series(0.0, index=names, dtype=float)
    expected[["A", "B"]] = 1.0 / 2
    expected[["C", "D"]] = -1.0 / 2

    pd.testing.assert_series_equal(tw, expected)


def test_compute_target_weights_insufficient_valid_returns_zeros():
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    names = ["A", "B", "C", "D"]
    # Only 3 valid (one NaN) but need 4 (2 long + 2 short) -> zeros
    ranks = [[0.9, 0.8, 0.7, np.nan]]
    sigs = _make_signals(idx, names, ranks)
    portfolio = _portfolio(n_long=2, n_short=2)
    prev_w = pd.Series(0.0, index=names)

    tw = compute_target_weights_for_date(idx[0], portfolio, sigs, prev_w)
    assert (tw == 0.0).all()


def test_turnover_limit_scales_towards_target():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    names = ["A", "B", "C", "D"]
    ranks = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]]
    sigs = _make_signals(idx, names, ranks)

    turnover = TurnoverLimit(window_bars=1, max_fraction=0.2)
    portfolio = _portfolio(n_long=2, n_short=2, gross=2.0, net=0.0, max_abs=1.0, turnover=turnover)

    prev_w = pd.Series(0.0, index=names)
    dt = idx[-1]
    tw = compute_target_weights_for_date(dt, portfolio, sigs, prev_w)

    # Unconstrained target would be +0.5 for A,B and -0.5 for C,D
    unconstrained = pd.Series(0.0, index=names, dtype=float)
    unconstrained[["A", "B"]] = 0.5
    unconstrained[["C", "D"]] = -0.5

    # Turnover from zero = 0.5 * sum(|target|) = 0.5 * 2 = 1.0 -> scale = 0.2
    expected = unconstrained * 0.2
    pd.testing.assert_series_equal(tw, expected)
