from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor, VolatilityFactor
from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, Quantile, LessEqual
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight
from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.engine.analytics.types import SignalAnalyticsConfig, StrategyAnalyticsConfig

# 1) Data Configuration
data = DataConfig(
    source="parquet://equities/indicies.parquet",
    calendar="XNYS",
    frequency="1d",
    start="2015-01-01",
    end="2025-12-31",
)

# 2) Universe
universe = Universe(
    name="Indices",
)

# 3) Factors
factors = {
    "mom_126": ReturnFactor(name="mom_126", field="close", lookback=126, method="log"),
}

# 4) Signals
signals = {
    "rank_momentum": CrossSectionRank(factor_name="mom_126", name="rank_momentum"),
}

# 5) Portfolio
portfolio = LongShortPortfolio(
    long_book=Book(name="long", selector=TopN(factor_name="rank_momentum", n=3), weighting=EqualWeight()),
    short_book=Book(name="short", selector=BottomN(factor_name="rank_momentum", n=3), weighting=EqualWeight()),
    rebalance_frequency="1d",
)

# 6) Execution & Costs
execution = Execution(
    order_policy=OrderPolicy(),
    latency=LatencyModel(),
    slippage=PowerLawSlippageModel(base_bps=1.0, k=0.0),
    volume_limits=VolumeParticipation(max_participation=1.0),
)
costs = Costs(
    commission=Commission(type='bps_notional', amount=1.0),
    borrow=BorrowCost(),
    financing=FinancingCost(),
    fees=StaticFees(),
)

# 7) Backtest Config (with reporting for tearsheet + signal analytics)
backtest_config = BacktestConfig(
    engine="event_driven",
    cash_initial=1_000_000,
    reporting=Reporting(
        signal_analytics=SignalAnalyticsConfig(signals=['rank_momentum']),
        strategyAnalytics=StrategyAnalyticsConfig(title="Custom Strategy (QuantDSL)"),
    ),
)

# 8) Compose Strategy
# NOTE: the platform runner calls run_backtest(strategy) automatically.
# Do not call it here; just define the `strategy` object.
strategy = Strategy(
    name="custom_strategy",
    data=data,
    universe=universe,
    factors=factors,
    signals=signals,
    portfolio=portfolio,
    execution=execution,
    costs=costs,
    backtest=backtest_config,
)