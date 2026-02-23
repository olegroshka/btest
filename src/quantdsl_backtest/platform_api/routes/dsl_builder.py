"""
DSL Builder API routes for generating and validating DSL strategies.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def _router():
    try:
        from fastapi import APIRouter as _APIRouter
    except Exception as e:
        raise RuntimeError(
            "Platform API extras are not installed. Install with: pip install -e .[platform]"
        ) from e
    return _APIRouter(prefix="/dsl", tags=["dsl"])


router = _router()


class DataConfigModel(BaseModel):
    source: str
    calendar: str
    start_date: str
    end_date: str


class UniverseModel(BaseModel):
    name: str
    filters: list[str] = []


class FactorModel(BaseModel):
    type: str
    params: dict[str, Any]


class SignalModel(BaseModel):
    type: str
    params: dict[str, Any]


class PortfolioModel(BaseModel):
    type: str  # 'long_short' or 'long_only'
    long_book: dict[str, str] | None = None
    short_book: dict[str, str] | None = None


class DSLConfigRequest(BaseModel):
    data: DataConfigModel | None = None
    universe: UniverseModel | None = None
    factors: dict[str, FactorModel] | None = None
    signals: dict[str, SignalModel] | None = None
    portfolio: PortfolioModel | None = None


class DSLCodeResponse(BaseModel):
    python_code: str
    json_config: str


def generate_dsl_code(config: DSLConfigRequest) -> str:
    """Generate Python DSL code from configuration."""

    lines = [
        "from quantdsl_backtest.dsl.strategy import Strategy",
        "from quantdsl_backtest.dsl.data_config import DataConfig",
        "from quantdsl_backtest.dsl.universe import Universe",
        "from quantdsl_backtest.dsl.factors import ReturnFactor, VolatilityFactor",
        "from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, Quantile, LessEqual",
        "from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight",
        "from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation",
        "from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees",
        "from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting",
        "from quantdsl_backtest.engine.analytics.types import SignalAnalyticsConfig, StrategyAnalyticsConfig",
        "",
        "# 1) Data Configuration",
        "data = DataConfig(",
    ]

    if config.data:
        lines.append(f'    source="{config.data.source}",')
        lines.append(f'    calendar="{config.data.calendar}",')
        lines.append('    frequency="1d",')
        lines.append(f'    start="{config.data.start_date}",')
        lines.append(f'    end="{config.data.end_date}",')

    lines.extend([
        ")",
        "",
        "# 2) Universe",
        "universe = Universe(",
    ])

    if config.universe:
        lines.append(f'    name="{config.universe.name}",')

    lines.extend([
        ")",
        "",
        "# 3) Factors",
        "factors = {",
    ])

    if config.factors:
        for factor_name, factor in config.factors.items():
            if factor.type == "momentum":
                lookback = factor.params.get("lookback", 126)
                lines.append(
                    f'    "{factor_name}": ReturnFactor(name="{factor_name}", field="close", lookback={lookback}, method="log"),'
                )
            elif factor.type == "volatility":
                lookback = factor.params.get("lookback", 63)
                lines.append(
                    f'    "{factor_name}": VolatilityFactor(name="{factor_name}", field="close", lookback={lookback}),'
                )

    lines.extend([
        "}",
        "",
        "# 4) Signals",
        "signals = {",
    ])

    if config.signals:
        for signal_name, signal in config.signals.items():
            if signal.type == "cross_section_rank":
                factor = signal.params.get("factor", "mom_126")
                lines.append(
                    f'    "{signal_name}": CrossSectionRank(factor_name="{factor}", name="{signal_name}"),'
                )

    # Determine which signal name to use for portfolio selectors.
    signal_names = list((config.signals or {}).keys())
    selector_signal = signal_names[0] if signal_names else "rank_momentum"

    # Use reasonable N for portfolio (must be < number of instruments).
    n_select = 3

    # Determine portfolio type from config.
    portfolio_type = (config.portfolio.type if config.portfolio else "long_short")

    lines.extend([
        "}",
        "",
        "# 5) Portfolio",
    ])

    if portfolio_type == "long_only":
        lines.extend([
            "portfolio = LongShortPortfolio(",
            f'    long_book=Book(name="long", selector=TopN(factor_name="{selector_signal}", n={n_select}), weighting=EqualWeight()),',
            '    short_book=None,',
            '    rebalance_frequency="1d",',
            ")",
        ])
    else:
        lines.extend([
            "portfolio = LongShortPortfolio(",
            f'    long_book=Book(name="long", selector=TopN(factor_name="{selector_signal}", n={n_select}), weighting=EqualWeight()),',
            f'    short_book=Book(name="short", selector=BottomN(factor_name="{selector_signal}", n={n_select}), weighting=EqualWeight()),',
            '    rebalance_frequency="1d",',
            ")",
        ])

    lines.extend([
        "",
        "# 6) Execution & Costs",
        "execution = Execution(",
        "    order_policy=OrderPolicy(),",
        "    latency=LatencyModel(),",
        "    slippage=PowerLawSlippageModel(base_bps=1.0, k=0.0),",
        "    volume_limits=VolumeParticipation(max_participation=1.0),",
        ")",
        "costs = Costs(",
        "    commission=Commission(type='bps_notional', amount=1.0),",
        "    borrow=BorrowCost(),",
        "    financing=FinancingCost(),",
        "    fees=StaticFees(),",
        ")",
    ])

    # Build the signal names list for signal_analytics config.
    signal_names_repr = repr(signal_names) if signal_names else "[]"

    lines.extend([
        "",
        "# 7) Backtest Config (with reporting for tearsheet + signal analytics)",
        "backtest_config = BacktestConfig(",
        '    engine="event_driven",',
        "    cash_initial=1_000_000,",
        "    reporting=Reporting(",
        f"        signal_analytics=SignalAnalyticsConfig(signals={signal_names_repr}),",
        '        strategyAnalytics=StrategyAnalyticsConfig(title="Custom Strategy (QuantDSL)"),',
        "    ),",
        ")",
        "",
        "# 8) Compose Strategy",
        "# NOTE: the platform runner calls run_backtest(strategy) automatically.",
        "# Do not call it here; just define the `strategy` object.",
        "strategy = Strategy(",
        '    name="custom_strategy",',
        "    data=data,",
        "    universe=universe,",
        "    factors=factors,",
        "    signals=signals,",
        "    portfolio=portfolio,",
        "    execution=execution,",
        "    costs=costs,",
        "    backtest=backtest_config,",
        ")",
    ])

    return "\n".join(lines)


@router.post("/generate", response_model=DSLCodeResponse)
async def generate_dsl(config: DSLConfigRequest) -> DSLCodeResponse:
    """Generate Python DSL code from configuration."""
    python_code = generate_dsl_code(config)
    
    import json
    json_config = json.dumps(config.model_dump(), indent=2)
    
    return DSLCodeResponse(python_code=python_code, json_config=json_config)


@router.post("/validate")
async def validate_dsl(config: DSLConfigRequest) -> dict[str, Any]:
    """Validate DSL configuration."""
    errors = []
    
    if not config.data:
        errors.append("Data configuration is required")
    elif not config.data.source:
        errors.append("Data source is required")
    
    if not config.universe:
        errors.append("Universe is required")
    elif not config.universe.name:
        errors.append("Universe name is required")
    
    if not config.factors:
        errors.append("At least one factor is required")
    
    if not config.portfolio:
        errors.append("Portfolio configuration is required")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
