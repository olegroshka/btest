import numpy as np
import pandas as pd

from quantdsl_backtest.engine.analytics.signal_analytics import (
    compute_forward_returns,
    assign_quantiles,
    compute_rank_ic,
    mean_forward_return_by_quantile,
    quantile_turnover,
)


def _toy_panel():
    # 4 instruments, 5 dates
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    cols = ["A", "B", "C", "D"]
    # simple increasing across columns to avoid ties
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0],
    ])
    return pd.DataFrame(data, index=idx, columns=cols)


def test_compute_forward_returns_basic():
    close = _toy_panel()
    fwd = compute_forward_returns(close, [1, 2])

    # horizon 1: close[t+1]/close[t] - 1
    exp1 = close.shift(-1) / close - 1.0
    assert (fwd[1].equals(exp1))

    # horizon 2
    exp2 = close.shift(-2) / close - 1.0
    assert (fwd[2].equals(exp2))


def test_assign_quantiles_with_mask_and_no_ties():
    # Construct a panel with no ties per row
    signal = _toy_panel()
    # mask out one name at all times
    mask = pd.DataFrame(True, index=signal.index, columns=signal.columns)
    mask.loc[:, "D"] = False

    qdf = assign_quantiles(signal, q=2, mask=mask)
    # D should be NaN due to mask
    assert qdf["D"].isna().all()
    # Remaining names should be assigned 1..2
    assert set(np.nan_to_num(qdf.iloc[0].values, nan=0.0)) <= {0.0, 1.0, 2.0}
    # Top half (B,C) are higher -> should fall in higher quantile (2) often
    assert int((qdf[["B", "C"]] == 2).sum().sum()) >= int((qdf[["A"]] == 2).sum().sum())


def test_compute_rank_ic_perfect_positive():
    # Two dates; on each, returns rank exactly matches signal rank
    # build signal and returns with same ordering across names
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = ["A", "B", "C"]
    sig = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], index=idx, columns=cols
    )
    rets = pd.DataFrame(
        [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.03, 0.04, 0.05]], index=idx, columns=cols
    )

    ic = compute_rank_ic(sig, rets)
    # Spearman rank correlation should be 1 for all dates
    assert np.isclose(ic.dropna().values, 1.0).all()


def test_mean_forward_return_by_quantile_and_ls():
    # Simple single date case to validate grouping math
    idx = pd.date_range("2020-01-01", periods=1, freq="B")
    cols = ["A", "B", "C", "D"]
    qdf = pd.DataFrame([[1, 1, 2, 2]], index=idx, columns=cols)
    fwd = pd.DataFrame([[0.01, 0.02, -0.01, 0.03]], index=idx, columns=cols)

    m, ls = mean_forward_return_by_quantile(qdf, fwd, q=2)
    # Q1 mean = (0.01+0.02)/2 = 0.015, Q2 mean = (-0.01+0.03)/2 = 0.01
    t0 = idx[0]
    assert np.isclose(float(m.loc[t0, 1]), 0.015)
    assert np.isclose(float(m.loc[t0, 2]), 0.01)
    assert np.isclose(float(ls.loc[t0]), 0.01 - 0.015)


def test_quantile_turnover_fraction_changes():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = ["A", "B", "C"]
    qdf = pd.DataFrame(
        [
            [1, 2, 3],
            [1, 3, 2],  # B and C swapped buckets -> 2/3 changed
            [1, 3, 2],  # no change vs prev -> 0/3
        ],
        index=idx,
        columns=cols,
    )
    turn = quantile_turnover(qdf, q=3)
    assert np.isnan(turn.iloc[0])
    assert np.isclose(turn.iloc[1], 2 / 3)
    assert np.isclose(turn.iloc[2], 0.0)
