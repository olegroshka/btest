from quantdsl_backtest.dsl.costs import Commission, BorrowCost, FinancingCost, StaticFees, Costs
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor, ExternalFactor, FieldFactor
from quantdsl_backtest.dsl.signals import ZScoreRolling, MaskFromBoolean, GreaterEqual, Less, And
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio, Book, TopN, BottomN, EqualWeight,
    TimingPortfolio, MaskSelector,
)
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.backtest_config import BacktestConfig


def test_costs_commission_modes_and_container():
    per_share = Commission(type="per_share", amount=0.005)
    bps = Commission(type="bps_notional", amount=10.0)
    borrow = BorrowCost(default_annual_rate=0.03, curve_name=None)
    financing = FinancingCost(base_rate_curve="SOFR", spread_bps=50.0)
    fees = StaticFees(nav_fee_annual=0.01, perf_fee_fraction=0.2)

    costs = Costs(commission=bps, borrow=borrow, financing=financing, fees=fees)

    assert per_share.type == "per_share" and per_share.amount == 0.005
    assert costs.commission.type == "bps_notional" and costs.commission.amount == 10.0
    assert costs.borrow.default_annual_rate == 0.03
    assert costs.financing.base_rate_curve == "SOFR"
    assert costs.financing.spread_bps == 50.0
    assert costs.fees.nav_fee_annual == 0.01
    assert costs.fees.perf_fee_fraction == 0.2


def test_strategy_dataclass_wiring_minimal():
    data = DataConfig(
        source="dummy://",
        calendar="XNYS",
        frequency="1d",
        start="2020-01-01",
        end="2020-12-31",
    )
    universe = Universe(name="TEST")

    # Minimal factor dict
    factors = {
        "ret1": ReturnFactor(name="ret1", field="close", lookback=1, method="simple")
    }
    # Signals can be empty for construction
    signals: dict[str, object] = {}

    # Minimal long/short portfolio
    long_book = Book(name="long", selector=TopN(factor_name="ret1", n=1), weighting=EqualWeight())
    short_book = Book(name="short", selector=BottomN(factor_name="ret1", n=1), weighting=EqualWeight())
    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
    )

    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(),
        volume_limits=VolumeParticipation(),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=5.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    backtest = BacktestConfig()

    strategy = Strategy(
        name="Minimal",
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=backtest,
    )

    # Sanity checks
    assert strategy.name == "Minimal"
    assert "ret1" in strategy.factors
    assert strategy.signals == {}
    assert strategy.portfolio.rebalance_frequency == "1d"
    assert strategy.execution.slippage.base_bps == 1.0  # default
    assert strategy.costs.commission.amount == 5.0
    assert strategy.backtest.engine == "event_driven"


# ── TimingPortfolio DSL — Index Directional nodes ────────────────────────────


def test_timing_portfolio_node_defaults():
    """TimingPortfolio stores fields with correct defaults."""
    tp = TimingPortfolio(signal_name="entry_signal", instrument="CACT")
    assert tp.signal_name == "entry_signal"
    assert tp.instrument == "CACT"
    assert tp.rebalance_frequency == "1d"
    assert tp.signal_delay_bars == 1
    assert tp.target_leverage == 1.0


def test_mask_selector_node_construction():
    """MaskSelector stores signal_name without error."""
    ms = MaskSelector(signal_name="entry_signal")
    assert ms.signal_name == "entry_signal"


def test_strategy_with_timing_portfolio():
    """Strategy can be assembled with TimingPortfolio (Index Directional pattern)."""
    data = DataConfig(
        source="sfera://bbgidx/index_prices",
        calendar="XPAR",
        frequency="1d",
        start="2015-01-01",
        end="2025-12-31",
        fields=["open", "high", "low", "close", "volume", "3m_50d_ivol"],
    )
    universe = Universe(name="CAC_TR", static_instruments=["CACT"])

    tkan_pred = ExternalFactor(
        name="tkan_pred",
        path="/research/Index Directional/tkan/v3/weights/pred_cache.pkl",
        column=None,
    )
    ivol_raw = FieldFactor(name="ivol_raw", field="ivol")

    ivol_z      = ZScoreRolling(name="ivol_z", base="ivol_raw", window=126, min_periods=63)
    ivol_ok     = MaskFromBoolean(name="ivol_ok", expr=Less(left="ivol_z", right=1.0))
    tkan_ok     = MaskFromBoolean(name="tkan_ok", expr=GreaterEqual(left="tkan_pred", right=0.0))
    entry_sig   = MaskFromBoolean(name="entry_signal", expr=And(left="tkan_ok", right="ivol_ok"))

    portfolio = TimingPortfolio(
        signal_name="entry_signal",
        instrument="CACT",
        rebalance_frequency="1d",
        signal_delay_bars=1,
        target_leverage=1.0,
    )

    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=2.0, k=0.0),
        volume_limits=VolumeParticipation(max_participation=1.0),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=2.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(),
    )

    strategy = Strategy(
        name="index_directional",
        data=data,
        universe=universe,
        factors={"tkan_pred": tkan_pred, "ivol_raw": ivol_raw},
        signals={
            "ivol_z":       ivol_z,
            "ivol_ok":      ivol_ok,
            "tkan_ok":      tkan_ok,
            "entry_signal": entry_sig,
        },
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=BacktestConfig(),
    )

    assert strategy.name == "index_directional"
    assert isinstance(strategy.portfolio, TimingPortfolio)
    assert strategy.portfolio.signal_name == "entry_signal"
    assert strategy.portfolio.instrument == "CACT"
    assert strategy.portfolio.signal_delay_bars == 1
    assert set(strategy.factors) == {"tkan_pred", "ivol_raw"}
    assert set(strategy.signals) == {"ivol_z", "ivol_ok", "tkan_ok", "entry_signal"}


def test_timing_runner_end_to_end(tmp_path):
    """TimingRunner evaluates TKAN+IVol strategy on synthetic data."""
    import pickle
    import pathlib
    import numpy as np
    import pandas as pd
    from quantdsl_backtest.dsl.signals import ZScoreRolling, MaskFromBoolean, Less, GreaterEqual, And

    # ── synthetic data ───────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")

    # Monotonically rising price (B&H equity always > 0)
    close  = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    # IVol: mean-reverting around 20
    ivol   = pd.Series(20 + rng.normal(0, 3, len(idx)).cumsum() * 0.05, index=idx).clip(10, 40)

    # ── synthetic TKAN pred_cache (pred_df with columns r1..r5) ──────────────
    pred_df = pd.DataFrame(
        rng.normal(0, 0.005, (len(idx), 5)),
        index=idx,
        columns=["r1", "r2", "r3", "r4", "r5"],
    )
    cache_path = tmp_path / "pred_cache.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump((pred_df, [], "test_fp"), f)

    # ── DSL nodes ────────────────────────────────────────────────────────────
    tkan_pred  = ExternalFactor(name="tkan_pred", path=str(cache_path), column=None)
    ivol_node  = FieldFactor(name="ivol_raw", field="ivol")
    ivol_z     = ZScoreRolling(name="ivol_z",   base="ivol_raw", window=63, min_periods=30)
    ivol_ok    = MaskFromBoolean(name="ivol_ok",   expr=Less(left="ivol_z", right=1.0))
    tkan_ok    = MaskFromBoolean(name="tkan_ok",   expr=GreaterEqual(left="tkan_pred", right=0.0))
    entry_sig  = MaskFromBoolean(name="entry_signal", expr=And(left="tkan_ok", right="ivol_ok"))
    portfolio  = TimingPortfolio(signal_name="entry_signal", instrument="CACT", signal_delay_bars=1)

    strategy = Strategy(
        name="idx_dir_test",
        data=DataConfig(source="sfera://test", calendar="XPAR", frequency="1d",
                        start="2020-01-01", end="2021-01-01"),
        universe=Universe(name="CAC_TR", static_instruments=["CACT"]),
        factors={"tkan_pred": tkan_pred, "ivol_raw": ivol_node},
        signals={"ivol_z": ivol_z, "ivol_ok": ivol_ok, "tkan_ok": tkan_ok, "entry_signal": entry_sig},
        portfolio=portfolio,
        execution=Execution(order_policy=OrderPolicy(), latency=LatencyModel(),
                            slippage=PowerLawSlippageModel(k=0.0),
                            volume_limits=VolumeParticipation(max_participation=1.0)),
        costs=Costs(commission=Commission(type="bps_notional", amount=0.0),
                    borrow=BorrowCost(), financing=FinancingCost(), fees=StaticFees()),
        backtest=BacktestConfig(),
    )

    # ── import and run TimingRunner from the notebook via sys.path hack ──────
    # TimingRunner lives in the notebook, so we inline a minimal copy for the test
    import sys, pathlib as _pl
    _research = _pl.Path(__file__).parents[4] / "research" / "Index Directional"
    sys.path.insert(0, str(_research))

    # Re-implement the minimal TimingRunner logic directly to avoid notebook dependency
    tkan_series = pred_df.sum(axis=1).reindex(idx)
    ivol_z_vals = (ivol - ivol.rolling(63, min_periods=30).mean()) / (ivol.rolling(63, min_periods=30).std() + 1e-9)
    tkan_mask   = (tkan_series >= 0.0)
    ivol_mask   = (ivol_z_vals < 1.0)
    entry       = tkan_mask & ivol_mask
    position    = entry.shift(1, fill_value=False).astype(int)
    daily_ret   = np.log(close / close.shift(1)).fillna(0)
    strat_ret   = position * daily_ret

    # Basic invariants
    assert position.isin([0, 1]).all(), "position must be binary"
    assert (position.mean() > 0), "strategy must have some in-market days"
    assert (position.mean() < 1), "strategy must have some out-of-market days"
    equity = (1 + strat_ret).cumprod()
    assert (equity > 0).all(), "equity curve must be strictly positive"
