"""Tests for the generic TargetWeights portfolio.

TargetWeights is the single primitive that subsumes timing / long-short /
rotation / leverage / hedging: everything reduces to a [date x instrument]
target-weights matrix. These tests cover the weight-resolution path
(`_compute_target_weights_generic`, reached via `compute_target_weights_for_date`)
and an end-to-end run through the event-driven engine.
"""

import math

import pandas as pd
import pytest

from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.costs import (
    BorrowCost,
    Commission,
    Costs,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.execution import (
    Execution,
    LatencyModel,
    OrderPolicy,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import TargetWeights, TurnoverLimit
from quantdsl_backtest.dsl.signals import MaskFromBoolean, NotNull
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.portfolio_engine import compute_target_weights_for_date

INSTRUMENTS = ["TQQQ", "VIXY", "IEF", "DBC"]


def _rotation_matrix(dates):
    """One-hot rotation: TQQQ for first half, IEF for second half."""
    w = pd.DataFrame(0.0, index=dates, columns=INSTRUMENTS)
    half = len(dates) // 2
    w.iloc[:half, w.columns.get_loc("TQQQ")] = 1.0
    w.iloc[half:, w.columns.get_loc("IEF")] = 1.0
    return w


# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #


def test_requires_exactly_one_source():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    w = _rotation_matrix(dates)

    # Neither source -> error
    with pytest.raises(ValueError):
        TargetWeights()

    # Both sources -> error
    with pytest.raises(ValueError):
        TargetWeights(weights=w, weights_signal="foo")

    # Exactly one -> ok
    TargetWeights(weights=w)
    TargetWeights(weights_signal="foo")


# --------------------------------------------------------------------------- #
# Weight resolution
# --------------------------------------------------------------------------- #


def test_precomputed_matrix_no_delay():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    w = _rotation_matrix(dates)
    pf = TargetWeights(weights=w, signal_delay_bars=0)
    prev = pd.Series(0.0, index=INSTRUMENTS)

    # First half -> TQQQ
    t0 = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    assert t0["TQQQ"] == 1.0
    assert t0.drop("TQQQ").abs().sum() == 0.0

    # Second half -> IEF
    t5 = compute_target_weights_for_date(dates[5], pf, {}, prev, None)
    assert t5["IEF"] == 1.0
    assert t5.drop("IEF").abs().sum() == 0.0


def test_signal_delay_shifts_along_matrix_index():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    w = _rotation_matrix(dates)  # half=3: TQQQ on 0,1,2 ; IEF on 3,4,5
    pf = TargetWeights(weights=w, signal_delay_bars=1)
    prev = pd.Series(0.0, index=INSTRUMENTS)

    # On date[3] with delay 1, we read date[2] -> still TQQQ.
    t3 = compute_target_weights_for_date(dates[3], pf, {}, prev, None)
    assert t3["TQQQ"] == 1.0

    # On date[4] with delay 1, we read date[3] -> IEF.
    t4 = compute_target_weights_for_date(dates[4], pf, {}, prev, None)
    assert t4["IEF"] == 1.0


def test_insufficient_history_carries_prev():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    w = _rotation_matrix(dates)
    pf = TargetWeights(weights=w, signal_delay_bars=2)
    prev = pd.Series([0.5, 0.0, 0.0, 0.0], index=INSTRUMENTS)

    # date[1] - delay 2 = -1 -> not enough history -> carry prev forward
    t = compute_target_weights_for_date(dates[1], pf, {}, prev, None)
    pd.testing.assert_series_equal(t, prev)


def test_date_not_in_matrix_carries_prev():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    w = _rotation_matrix(dates)
    pf = TargetWeights(weights=w, signal_delay_bars=0)
    prev = pd.Series([0.3, 0.0, 0.0, 0.0], index=INSTRUMENTS)

    missing = pd.Timestamp("2021-06-01")
    t = compute_target_weights_for_date(missing, pf, {}, prev, None)
    pd.testing.assert_series_equal(t, prev)


def test_gross_leverage_rescale():
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    # A row with gross 1.0 that we want levered to 2.0
    w = pd.DataFrame(0.0, index=dates, columns=INSTRUMENTS)
    w["TQQQ"] = 1.0
    pf = TargetWeights(weights=w, signal_delay_bars=0, target_gross_leverage=2.0)
    prev = pd.Series(0.0, index=INSTRUMENTS)

    t = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    assert t["TQQQ"] == pytest.approx(2.0)
    assert t.abs().sum() == pytest.approx(2.0)


def test_gross_leverage_rescale_mixed_signs():
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    w = pd.DataFrame(0.0, index=dates, columns=INSTRUMENTS)
    w["TQQQ"] = 0.5
    w["VIXY"] = -0.5  # gross 1.0, net 0.0
    pf = TargetWeights(weights=w, signal_delay_bars=0, target_gross_leverage=2.0)
    prev = pd.Series(0.0, index=INSTRUMENTS)

    t = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    assert t["TQQQ"] == pytest.approx(1.0)
    assert t["VIXY"] == pytest.approx(-1.0)
    assert t.abs().sum() == pytest.approx(2.0)
    assert t.sum() == pytest.approx(0.0)


def test_max_abs_weight_clip():
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    w = pd.DataFrame(0.0, index=dates, columns=INSTRUMENTS)
    w["TQQQ"] = 0.9
    w["VIXY"] = 0.1
    pf = TargetWeights(weights=w, signal_delay_bars=0, max_abs_weight_per_name=0.5)
    prev = pd.Series(0.0, index=INSTRUMENTS)

    t = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    assert t["TQQQ"] == pytest.approx(0.5)
    assert t["VIXY"] == pytest.approx(0.1)


def test_turnover_limit_scales_move():
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    w = pd.DataFrame(0.0, index=dates, columns=INSTRUMENTS)
    w["TQQQ"] = 1.0  # full move from flat -> turnover 0.5
    pf = TargetWeights(
        weights=w,
        signal_delay_bars=0,
        turnover_limit=TurnoverLimit(max_fraction=0.25),
    )
    prev = pd.Series(0.0, index=INSTRUMENTS)

    t = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    # turnover capped at 0.25 -> half the intended move
    assert t["TQQQ"] == pytest.approx(0.5)


def test_weights_signal_path():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    panel = _rotation_matrix(dates)
    pf = TargetWeights(weights_signal="rotation_w", signal_delay_bars=0)
    prev = pd.Series(0.0, index=INSTRUMENTS)

    t = compute_target_weights_for_date(dates[0], pf, {"rotation_w": panel}, prev, None)
    assert t["TQQQ"] == 1.0


def test_nan_in_row_is_flat_not_carry():
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    w = pd.DataFrame(
        [[float("nan"), 0.5, float("nan"), 0.0]],
        index=dates,
        columns=INSTRUMENTS,
    )
    pf = TargetWeights(weights=w, signal_delay_bars=0)
    prev = pd.Series([0.9, 0.9, 0.9, 0.9], index=INSTRUMENTS)

    t = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    assert t["TQQQ"] == 0.0  # NaN -> flat, not carried from prev
    assert t["VIXY"] == 0.5


def test_aligns_to_engine_universe():
    """Matrix columns are a subset/superset of the engine universe; result aligns."""
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    # Matrix has an extra column not in the universe, and omits one universe name.
    w = pd.DataFrame(
        [[1.0, 0.0, 0.0]],
        index=dates,
        columns=["TQQQ", "VIXY", "XYZ_NOT_IN_UNIVERSE"],
    )
    pf = TargetWeights(weights=w, signal_delay_bars=0)
    prev = pd.Series(0.0, index=INSTRUMENTS)  # universe = the 4 INSTRUMENTS

    t = compute_target_weights_for_date(dates[0], pf, {}, prev, None)
    assert list(t.index) == INSTRUMENTS  # aligned to universe
    assert t["TQQQ"] == 1.0
    assert "XYZ_NOT_IN_UNIVERSE" not in t.index


# --------------------------------------------------------------------------- #
# End-to-end through the event-driven engine
# --------------------------------------------------------------------------- #


def _toy_prices(dates, instruments):
    data = {}
    for k, inst in enumerate(instruments):
        base = 100 + 10 * k
        data[inst] = [base * (1.0 + 0.001 * i) for i in range(len(dates))]
    return pd.DataFrame(data, index=dates)


def _toy_market_data(dates, instruments, prices, volumes):
    """Build a MarketData with close/volume bars (the engine's factor engine
    needs proper bars, even though precomputed TargetWeights ignores them)."""
    from quantdsl_backtest.data.schema import MarketData

    bars = {}
    for inst in instruments:
        df = pd.DataFrame(index=dates)
        df["close"] = prices[inst]
        df["volume"] = volumes[inst]
        bars[inst] = df
    return MarketData(
        bars=bars,
        instruments=list(instruments),
        fields=["close", "volume"],
        frequency="1d",
        calendar="XNYS",
    )


def test_end_to_end_rotation_event_driven(monkeypatch):
    dates = pd.date_range("2022-01-03", periods=20, freq="B")
    prices = _toy_prices(dates, INSTRUMENTS)
    volumes = pd.DataFrame(1_000_000.0, index=dates, columns=INSTRUMENTS)
    md = _toy_market_data(dates, INSTRUMENTS, prices, volumes)

    import quantdsl_backtest.engine.backtest_runner as br

    def fake_load(strategy):
        return md, prices, volumes

    monkeypatch.setattr(br, "load_data_for_strategy", fake_load)

    w = _rotation_matrix(dates)
    strat = Strategy(
        name="tw_rotation",
        data=DataConfig(
            source="parquet://local/none",
            calendar="XNYS",
            frequency="1d",
            start="2022-01-03",
            end="2022-02-01",
        ),
        universe=Universe(name="ROT", static_instruments=INSTRUMENTS),
        # A precomputed-matrix TargetWeights needs no signals, but the engine
        # eagerly builds the signal engine, which requires >=1 factor. Provide a
        # trivial one; it doesn't drive the (precomputed) weights.
        factors={
            "ret1": ReturnFactor(
                name="ret1", field="close", lookback=1, method="simple"
            )
        },
        signals={"mask_all": MaskFromBoolean(NotNull("ret1"))},
        portfolio=TargetWeights(
            weights=w, rebalance_frequency="1d", signal_delay_bars=1
        ),
        execution=Execution(
            order_policy=OrderPolicy(fill_on="close"),
            latency=LatencyModel(),
            slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0, exponent=1.0),
            volume_limits=VolumeParticipation(
                max_participation=1.0, mode="proportional", min_fill_notional=0.0
            ),
        ),
        costs=Costs(
            commission=Commission(type="bps_notional", amount=1.0),
            borrow=BorrowCost(default_annual_rate=0.0),
            financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
            fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
        ),
        backtest=BacktestConfig(cash_initial=1_000_000.0),
    )
    # Don't write report artifacts into the repo during tests.
    strat.backtest.reporting.output_dir = None

    result = run_backtest(strat)
    assert result is not None
    assert len(result.equity) > 0
    assert math.isfinite(float(result.equity.iloc[-1]))

    # The realized weights should reflect the rotation: TQQQ in the first half,
    # IEF in the second half. (Don't pin exact bars — warmup + signal_delay +
    # ramp-in make the first couple of bars flat; assert on the regimes.)
    realized = result.weights
    half = len(realized) // 2
    assert realized["TQQQ"].iloc[:half].max() > 0.5  # held TQQQ early
    assert realized["IEF"].iloc[-1] > 0.5  # rotated into IEF by the end
    assert realized["IEF"].iloc[:half].abs().max() == 0.0  # no IEF early
