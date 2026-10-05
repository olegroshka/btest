"""VolatilityFactor: ewma + garch methods (garch skipped if volforecast/arch missing)."""
import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.factors import VolatilityFactor
from quantdsl_backtest.engine.factor_engine import FactorEngine


def _prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    r = rng.normal(0, 0.01, (n, 2))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=["A", "B"])


def _engine(prices):
    md = {"close": prices}
    return FactorEngine(md, prices)


def test_ewma_vol_matches_pandas():
    px = _prices()
    fe = _engine(px)
    out = fe.compute("v", VolatilityFactor(name="v", lookback=11, method="ewma", min_periods=20))
    r = np.log(px / px.shift(1))
    exp = np.sqrt((r ** 2).ewm(halflife=11.0, min_periods=20, adjust=False).mean()) * np.sqrt(252.0)
    pd.testing.assert_frame_equal(out, exp)


def test_garch_vol_factor_runs_and_is_annualized():
    pytest.importorskip("volforecast")
    pytest.importorskip("arch")
    px = _prices()
    fe = _engine(px)
    out = fe.compute(
        "g", VolatilityFactor(name="g", lookback=500, method="garch", min_periods=250, horizon=5, refit_every=126)
    )
    assert out.shape == px.shape
    assert out.iloc[:249].isna().all().all()
    got = out.iloc[260:].dropna()
    assert len(got) > 0.9 * (len(px) - 260)
    # daily σ≈1% → ≈16% annualised
    assert ((got > 0.05) & (got < 0.40)).all().all()
