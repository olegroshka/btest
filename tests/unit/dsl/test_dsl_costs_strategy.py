from quantdsl_backtest.dsl.costs import Commission, BorrowCost, FinancingCost, StaticFees, Costs
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight
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
