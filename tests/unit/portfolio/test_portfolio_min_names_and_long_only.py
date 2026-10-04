"""The minimum-names rule and a long-short portfolio without a short book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.portfolio import (
    Book,
    BottomN,
    EqualWeight,
    LongShortPortfolio,
    SignalWeight,
    TopN,
)
from quantdsl_backtest.engine.portfolio_engine import compute_target_weights_for_date


def _signals(
    rank_row: list[float],
    names: list[str],
    long_mask: list[bool] | None = None,
    short_mask: list[bool] | None = None,
) -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2020-01-01", periods=1, freq="D")
    rank = pd.DataFrame([rank_row], index=idx, columns=names, dtype="float64")
    long = pd.DataFrame([long_mask or [True] * len(names)], index=idx, columns=names)
    short = pd.DataFrame([short_mask or [True] * len(names)], index=idx, columns=names)
    return {"rank": rank, "long_candidates": long, "short_candidates": short}


def test_min_names_per_book_forms_targets_before_the_books_are_full() -> None:
    names = ["A", "B", "C", "D", "E"]
    # four valid ranks; the candidate masks keep the two books apart as a real strategy's do
    sigs = _signals(
        [0.9, 0.8, 0.7, 0.6, np.nan],
        names,
        long_mask=[True, True, False, False, False],
        short_mask=[False, False, True, True, False],
    )
    long_book = Book(
        name="long",
        selector=TopN(factor_name="rank", n=10, fill_from_unmasked=False),
        weighting=EqualWeight(),
    )
    short_book = Book(
        name="short",
        selector=BottomN(factor_name="rank", n=10, fill_from_unmasked=False),
        weighting=EqualWeight(),
    )
    prev = pd.Series(0.0, index=names)
    strict = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        max_abs_weight_per_name=1.0,
    )
    assert (
        float(
            compute_target_weights_for_date(sigs["rank"].index[0], strict, sigs, prev)
            .abs()
            .sum()
        )
        == 0.0
    )  # 4 < 10 + 10
    relaxed = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        max_abs_weight_per_name=1.0,
        min_names_per_book=2,
    )
    tw = compute_target_weights_for_date(sigs["rank"].index[0], relaxed, sigs, prev)
    assert float(tw.abs().sum()) == pytest.approx(2.0) and tw["E"] == 0.0
    assert tw[tw > 0].sum() == pytest.approx(1.0) and tw[tw < 0].sum() == pytest.approx(
        -1.0
    )
    too_strict = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        max_abs_weight_per_name=1.0,
        min_names_per_book=5,
    )
    assert (
        float(
            compute_target_weights_for_date(
                sigs["rank"].index[0], too_strict, sigs, prev
            )
            .abs()
            .sum()
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="min_names_per_book"):
        LongShortPortfolio(
            long_book=long_book,
            short_book=short_book,
            rebalance_frequency="1d",
            min_names_per_book=0,
        )


def test_long_only_portfolio_without_a_short_book() -> None:
    names = ["A", "B", "C", "D"]
    sigs = _signals([0.9, 0.8, 0.7, 0.6], names)
    long_book = Book(
        name="long",
        selector=TopN(factor_name="rank", n=2),
        weighting=SignalWeight(signal_name="rank"),
    )
    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=None,
        rebalance_frequency="1d",
        target_gross_leverage=1.0,
        target_net_exposure=1.0,
        max_abs_weight_per_name=1.0,
    )
    prev = pd.Series(0.0, index=names)
    tw = compute_target_weights_for_date(sigs["rank"].index[0], portfolio, sigs, prev)
    assert (tw >= 0).all() and tw.sum() == pytest.approx(1.0)
    assert (
        tw["A"] == pytest.approx(0.9 / 1.7)
        and tw["B"] == pytest.approx(0.8 / 1.7)
        and tw["C"] == 0.0
    )
    thin = _signals([0.9, np.nan, np.nan, np.nan], names)
    assert (
        float(
            compute_target_weights_for_date(
                thin["rank"].index[0], portfolio, thin, prev
            )
            .abs()
            .sum()
        )
        == 0.0
    )  # 1 < n=2
