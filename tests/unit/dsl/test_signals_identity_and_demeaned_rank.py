"""Identity signals, the demeaned cross-sectional rank, and a signal engine with no factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.signals import (
    CrossSectionRank,
    Identity,
    MaskFromBoolean,
    NotNull,
)
from quantdsl_backtest.engine.signal_engine import SignalEngine


def _panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, np.nan, 2.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan],
        ],
        index=idx,
        columns=["A", "B", "C", "D"],
    )


def test_identity_returns_the_factor_unchanged_and_detached() -> None:
    f = _panel()
    engine = SignalEngine(
        {"f": f}, {"same": Identity(base="f", name="same")}, use_polars=False
    )
    out = engine.compute_all()["same"]
    pd.testing.assert_frame_equal(out, f)
    out.iloc[0, 0] = 99.0
    assert f.iloc[0, 0] == 1.0  # a copy, not the engine's factor panel


def test_identity_of_a_signal_and_in_an_expression() -> None:
    f = _panel()
    nodes = {
        "same": Identity(base="f", name="same"),
        "valid": MaskFromBoolean(name="valid", expr=NotNull("f")),
        "again": Identity(base="valid", name="again"),
    }
    out = SignalEngine({"f": f}, nodes, use_polars=False).compute_all()
    assert out["valid"].iloc[1].tolist() == [True, False, True, True]
    pd.testing.assert_frame_equal(out["again"], out["valid"])


@pytest.mark.parametrize("use_polars", [False, True])
def test_demeaned_rank_is_percentile_minus_a_half(use_polars: bool) -> None:
    f = _panel()
    nodes = {
        "pct": CrossSectionRank(factor_name="f", method="percentile", name="pct"),
        "dm": CrossSectionRank(factor_name="f", method="demeaned", name="dm"),
    }
    out = SignalEngine({"f": f}, nodes, use_polars=use_polars).compute_all()
    assert out["dm"].iloc[0].tolist() == pytest.approx([-0.5, -1 / 6, 1 / 6, 0.5])
    assert out["dm"].iloc[1]["B"] != out["dm"].iloc[1]["B"]  # NaN stays NaN
    assert out["dm"].iloc[1][["D", "C", "A"]].tolist() == pytest.approx(
        [-0.5, 0.0, 0.5]
    )
    assert out["dm"].iloc[2].isna().all()
    pd.testing.assert_frame_equal(out["dm"], out["pct"] - 0.5)


def test_signal_engine_without_factors_takes_the_panel_shape_from_the_caller() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    engine = SignalEngine(
        {}, {}, index=idx, columns=pd.Index(["A", "B"]), use_polars=False
    )
    assert engine.compute_all() == {}
    assert list(engine.index) == list(idx) and list(engine.columns) == ["A", "B"]
    with pytest.raises(ValueError, match="no factors"):
        SignalEngine({}, {}, use_polars=False)
