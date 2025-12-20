from __future__ import annotations

import pandas as pd

from quantdsl_backtest.dsl.signals import (
    MaskFromBoolean,
    NotNull,
    And,
)
from quantdsl_backtest.engine.signal_engine import SignalEngine


def test_macro_series_broadcast_and_gating():
    # Create a tiny factor panel (2 instruments, 6 dates)
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    cols = ["A", "B"]
    # A simple factor with NaNs on last row to test validity path
    base = pd.DataFrame(
        [[1.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [None, None]],
        index=idx,
        columns=cols,
    )

    factors = {"base": base}

    # Macro risk-on series: off for first 2 days, on afterwards
    macro_series = pd.Series([False, False, True, True, True, True], index=idx)

    # Signals: valid = notnull(base); macro = broadcast(series); long = valid & macro
    valid = MaskFromBoolean(name="valid", expr=NotNull(factor_name="base"))
    macro = MaskFromBoolean(name="macro", expr=macro_series)
    long_sig = MaskFromBoolean(name="long", expr=And(left="valid", right="macro"))

    engine = SignalEngine(factors=factors, signal_nodes={
        "valid": valid,
        "macro": macro,
        "long": long_sig,
    })

    out = engine.compute_all()

    # First two rows gated off by macro
    assert not out["macro"].iloc[0].any()
    assert not out["macro"].iloc[1].any()
    assert out["macro"].iloc[2].all()

    # valid should be False on last row due to NaNs
    assert not out["valid"].iloc[-1].any()

    # long = valid & macro: first two rows False; last row False; middle rows True
    long_df = out["long"]
    assert not long_df.iloc[0].any()
    assert not long_df.iloc[1].any()
    assert long_df.iloc[2].all()
    assert long_df.iloc[3].all()
    assert long_df.iloc[4].all()
    assert not long_df.iloc[5].any()
