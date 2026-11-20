"""Lag, RollingSum, Composite, and CrossSectionRank's midpoint method, descending order and signal input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.signals import (
    Composite,
    CrossSectionRank,
    Greater,
    Lag,
    MaskFromBoolean,
    RollingStd,
    RollingSum,
)
from quantdsl_backtest.engine.signal_engine import SignalEngine

IDX = pd.date_range("2024-01-01", periods=5, freq="D")
COLS = ["A", "B", "C", "D"]


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, np.nan, 2.0, 1.0],
            [2.0, 2.0, 5.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [3.0, 1.0, 4.0, 2.0],
        ],
        index=IDX,
        columns=COLS,
    )


def _run(nodes: dict, use_polars: bool = False, **factors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return SignalEngine(factors or {"f": _panel()}, nodes, use_polars=use_polars).compute_all()


# --------------------------------------------------------------------------- Lag


def test_lag_moves_each_value_to_a_later_bar() -> None:
    f = _panel()
    out = _run({"l1": Lag(base="f", periods=1, name="l1"), "l0": Lag(base="f", periods=0, name="l0")})
    pd.testing.assert_frame_equal(out["l1"], f.shift(1))
    pd.testing.assert_frame_equal(out["l0"], f)
    assert out["l1"].iloc[0].isna().all()


def test_lag_cannot_read_the_future() -> None:
    with pytest.raises(ValueError, match="future"):
        Lag(base="f", periods=-1)


def test_lagged_value_does_not_depend_on_later_data() -> None:
    f = _panel()
    cut = f.copy()
    cut.iloc[3:] = 99.0  # corrupt everything from bar 3 on
    nodes = {"vol": Lag(base=RollingStd(base="f", window=2, min_periods=2), periods=1, name="vol")}
    full = _run(nodes, f=f)["vol"]
    damaged = _run(nodes, f=cut)["vol"]
    pd.testing.assert_frame_equal(full.iloc[:4], damaged.iloc[:4])  # bar 3 uses bars 1 and 2 only


# --------------------------------------------------------------------------- RollingSum


def test_rolling_sum_skips_missing_values_and_respects_min_periods() -> None:
    out = _run(
        {
            "s": RollingSum(base="f", window=3, min_periods=2, name="s"),
            "strict": RollingSum(base="f", window=3, min_periods=3, name="strict"),
        }
    )
    assert out["s"]["A"].tolist()[1:3] == [5.0, 7.0]  # 1+4, 1+4+2
    assert out["s"].loc[IDX[2], "B"] == 4.0  # 2 + (missing) + 2: two values are enough
    assert np.isnan(out["strict"].loc[IDX[2], "B"])  # but not three
    assert np.isnan(out["s"].loc[IDX[0], "A"])  # one value is below min_periods


# --------------------------------------------------------------------------- Composite


def test_composite_is_the_mean_of_the_members_present() -> None:
    a = pd.DataFrame([[1.0, np.nan], [np.nan, np.nan]], index=IDX[:2], columns=["A", "B"])
    b = pd.DataFrame([[3.0, 4.0], [np.nan, 2.0]], index=IDX[:2], columns=["A", "B"])
    c = pd.DataFrame([[5.0, np.nan], [np.nan, 6.0]], index=IDX[:2], columns=["A", "B"])
    out = _run(
        {
            "any": Composite(members=["a", "b", "c"], min_count=1, name="any"),
            "two": Composite(members=["a", "b", "c"], min_count=2, name="two"),
        },
        a=a,
        b=b,
        c=c,
    )
    assert out["any"].loc[IDX[0]].tolist() == [3.0, 4.0]
    assert np.isnan(out["any"].loc[IDX[1], "A"]) and out["any"].loc[IDX[1], "B"] == 4.0
    assert out["two"].loc[IDX[0], "A"] == 3.0
    assert np.isnan(out["two"].loc[IDX[0], "B"])  # one member only
    assert out["two"].loc[IDX[1], "B"] == 4.0


def test_composite_takes_nodes_and_checks_its_arguments() -> None:
    out = _run({"c": Composite(members=["f", Lag(base="f", periods=1)], min_count=2, name="c")})
    f = _panel()
    pd.testing.assert_frame_equal(out["c"], (f + f.shift(1)) / 2.0)
    with pytest.raises(ValueError):
        Composite(members=[])
    with pytest.raises(ValueError):
        Composite(members=["f"], min_count=2)


# --------------------------------------------------------------------------- CrossSectionRank


@pytest.mark.parametrize("use_polars", [False, True])
def test_midpoint_rank_is_rank_minus_a_half_over_n_minus_a_half(use_polars: bool) -> None:
    out = _run({"r": CrossSectionRank(factor_name="f", method="midpoint", name="r")}, use_polars=use_polars)["r"]
    assert out.iloc[0].tolist() == pytest.approx([-0.375, -0.125, 0.125, 0.375])
    # three names: ranks 3, 2, 1 for A, C, D
    assert out.iloc[1][["A", "C", "D"]].tolist() == pytest.approx([1 / 3, 0.0, -1 / 3])
    assert np.isnan(out.iloc[1]["B"])
    # a tie shares the average rank: A and B both at rank 1.5 of 3
    assert out.iloc[2][["A", "B", "C"]].tolist() == pytest.approx([-1 / 6, -1 / 6, 1 / 3])
    assert out.iloc[3].isna().all()
    assert out.sum(axis=1).abs().max() < 1e-12  # the day's scores sum to zero


@pytest.mark.parametrize("use_polars", [False, True])
def test_midpoint_rank_matches_the_pandas_formula(use_polars: bool) -> None:
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(size=(30, 12)), index=pd.date_range("2024-01-01", periods=30), columns=list("ABCDEFGHIJKL"))
    f = f.mask(rng.random(f.shape) < 0.2)
    out = _run({"r": CrossSectionRank(factor_name="f", method="midpoint", name="r")}, use_polars=use_polars, f=f)["r"]
    n = f.notna().sum(axis=1)
    expected = (f.rank(axis=1) - 0.5).div(n.where(n > 0), axis=0) - 0.5
    pd.testing.assert_frame_equal(out, expected, check_names=False, check_freq=False)


@pytest.mark.parametrize("use_polars", [False, True])
def test_descending_rank_is_the_negative_of_the_ascending_one(use_polars: bool) -> None:
    out = _run(
        {
            "up": CrossSectionRank(factor_name="f", method="midpoint", name="up"),
            "down": CrossSectionRank(factor_name="f", method="midpoint", ascending=False, name="down"),
        },
        use_polars=use_polars,
    )
    pd.testing.assert_frame_equal(out["down"], -out["up"])


@pytest.mark.parametrize("use_polars", [False, True])
def test_rank_of_a_signal_within_a_mask(use_polars: bool) -> None:
    nodes = {
        "lagged": Lag(base="f", periods=1, name="lagged"),
        "big": MaskFromBoolean(name="big", expr=Greater(left="f", right=1.5)),
        "r": CrossSectionRank(factor_name="lagged", mask_name="big", method="midpoint", name="r"),
    }
    out = _run(nodes, use_polars=use_polars)["r"]
    # Bar 1: lagged = bar 0 values (1, 2, 3, 4); the mask on bar 1 keeps A (4) and C (2) only.
    assert out.iloc[1][["A", "C"]].tolist() == pytest.approx([-0.25, 0.25])
    assert out.iloc[1][["B", "D"]].isna().all()
