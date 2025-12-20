import numpy as np
import pandas as pd
import pytest

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import (
    MaskFromBoolean,
    NotNull,
    CrossSectionRank,
    GreaterEqual,
    LessEqual,
)
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.data.schema import MarketData
from quantdsl_backtest.engine.backtest_runner import run_backtest


def _build_synthetic_md():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    instruments = ["A", "B", "C"]
    # Create simple trending prices so ranks are stable most days
    prices = pd.DataFrame(
        {
            "A": [100, 102, 104, 106, 108, 110],
            "B": [100, 99, 98, 97, 96, 95],
            "C": [100, 100, 101, 101, 102, 103],
        },
        index=idx,
        dtype=float,
    )
    volumes = pd.DataFrame(1_000_000.0, index=idx, columns=instruments, dtype=float)

    bars = {}
    for instr in instruments:
        df = pd.DataFrame(index=idx)
        df["close"] = prices[instr]
        df["volume"] = volumes[instr]
        bars[instr] = df

    md = MarketData(
        bars=bars,
        instruments=instruments,
        fields=["close", "volume"],
        frequency="1d",
        calendar="XNYS",
    )
    return md, prices, volumes


def _build_strategy() -> Strategy:
    data = DataConfig(
        source="dummy://",
        calendar="XNYS",
        frequency="1d",
        start="2020-01-01",
        end="2020-12-31",
    )
    universe = Universe(name="SYN")
    factors = {
        "ret1": ReturnFactor(name="ret1", field="close", lookback=1, method="simple"),
    }
    signals = {
        # Mask: valid wherever ret1 isn’t NaN
        "mask_all": MaskFromBoolean(NotNull("ret1")),
        # Rank returns cross-sectionally within mask
        "rank": CrossSectionRank(factor_name="ret1", mask_name="mask_all", method="percentile"),
        # Long candidates: top half; short candidates: bottom half
        "long_candidates": MaskFromBoolean(GreaterEqual("rank", 0.5)),
        "short_candidates": MaskFromBoolean(LessEqual("rank", 0.5)),
    }

    long_book = Book(name="long", selector=TopN(factor_name="rank", n=1), weighting=EqualWeight())
    short_book = Book(name="short", selector=BottomN(factor_name="rank", n=1), weighting=EqualWeight())
    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        signal_delay_bars=0,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=0.75,
    )

    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0, exponent=1.0),
        volume_limits=VolumeParticipation(max_participation=None, mode="proportional", min_fill_notional=0.0),
    )
    costs = Costs(
        commission=Commission(type="bps_notional", amount=0.0),
        borrow=BorrowCost(default_annual_rate=0.0),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )
    backtest = BacktestConfig()

    return Strategy(
        name="Smoke",
        data=data,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=backtest,
    )


def test_backtest_runner_smoke_monkeypatched_loader(monkeypatch):
    strategy = _build_strategy()

    def _fake_loader(_strategy):
        return _build_synthetic_md()

    # Patch the symbol imported in backtest_runner module
    import quantdsl_backtest.engine.backtest_runner as br
    monkeypatch.setattr(br, "load_data_for_strategy", _fake_loader)

    result = run_backtest(strategy)

    # Basic shape and non-emptiness checks
    assert len(result.equity) > 0
    assert len(result.returns) == len(result.equity)
    assert result.positions.shape[0] == len(result.equity)
    assert result.weights.shape == result.positions.shape

    # Trades should exist given rebalancing from zero
    assert not result.trades.empty

    # Weights bounded by max_abs_weight_per_name and gross roughly around leverage target
    assert (result.weights.abs() <= strategy.portfolio.max_abs_weight_per_name + 1e-9).all().all()

    # Total return equals from equity series
    total_ret = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1.0)
    assert np.isclose(result.total_return, total_ret)

    # Slicing should reduce the length and preserve alignment
    sliced = result.slice(start=str(result.index[1]), end=str(result.index[-2]))
    assert len(sliced.equity) == max(0, len(result.equity) - 2)
    assert sliced.equity.index[0] == result.equity.index[1]
    assert sliced.equity.index[-1] == result.equity.index[-2]
