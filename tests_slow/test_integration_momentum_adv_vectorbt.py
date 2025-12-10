# tests_slow/test_integration_momentum_adv_vectorbt.py

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import CrossSectionRank, MaskFromBoolean, LessEqual
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio,
    Book,
    TopN,
    BottomN,
    EqualWeight,
    TurnoverLimit,
)
from quantdsl_backtest.dsl.execution import (
    Execution,
    OrderPolicy,
    LatencyModel,
    PowerLawSlippageModel,
    VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import (
    Costs,
    Commission,
    BorrowCost,
    FinancingCost,
    StaticFees,
)
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig,
    MarginConfig,
    RiskChecks,
    Reporting,
)
from quantdsl_backtest.dsl.strategy import Strategy

from quantdsl_backtest.data.adapters import load_market_data
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.accounting import compute_basic_metrics
from quantdsl_backtest.utils.logging import get_logger


# Same data source as other integration test
DATA_SOURCE = "parquet://equities/sp500_daily"


log = get_logger(__name__)


def _build_strategy(
    *,
    signal_delay_bars: int = 0,
    commission_bps: float | None = None,
    commission_per_share: float | None = None,
    turnover_limit: float | None = None,
    max_abs_weight_per_name: float | None = 0.05,
) -> Strategy:
    data_cfg = DataConfig(
        source=DATA_SOURCE,
        calendar="XNYS",
        frequency="1d",
        start="2015-01-01",
        end="2025-01-01",
        price_adjustment="split_dividend",
        fields=["open", "high", "low", "close", "volume"],
    )

    universe = Universe(
        name="SP500",
        filters=[
            HasHistory(min_days=252),
            MinPrice(min_price=5.0),
            MinDollarADV(min_dollar_adv=1_000_000.0),
        ],
        id_field="ticker",
    )

    # Factor: 6m log momentum
    mom_6m = ReturnFactor(
        name="mom_6m",
        field="close",
        lookback=126,
        method="log",
    )
    factors = {"mom_6m": mom_6m}

    rank = CrossSectionRank(
        factor_name="mom_6m",
        mask_name=None,
        method="percentile",
        name="rank",
    )

    long_candidates = MaskFromBoolean(
        name="long_candidates",
        expr=LessEqual(left=0.9, right="rank"),  # rank >= 0.9
    )
    short_candidates = MaskFromBoolean(
        name="short_candidates",
        expr=LessEqual(left="rank", right=0.1),  # rank <= 0.1
    )

    signals = {
        "rank": rank,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    }

    long_book = Book(
        name="long_book",
        selector=TopN(factor_name="rank", n=50, mask_name="long_candidates"),
        weighting=EqualWeight(),
    )
    short_book = Book(
        name="short_book",
        selector=BottomN(factor_name="rank", n=50, mask_name="short_candidates"),
        weighting=EqualWeight(),
    )

    portfolio = LongShortPortfolio(
        rebalance_frequency="1d",
        rebalance_at="market_close",
        signal_delay_bars=signal_delay_bars,
        long_book=long_book,
        short_book=short_book,
        target_gross_leverage=2.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=max_abs_weight_per_name,
        sector_neutral=None,
        turnover_limit=None if turnover_limit is None else TurnoverLimit(max_fraction=turnover_limit),
    )

    execution = Execution(
        order_policy=OrderPolicy(default_order_type="MOC", time_in_force="DAY"),
        latency=LatencyModel(signal_to_order_delay_bars=0, market_latency_ms=0),
        slippage=PowerLawSlippageModel(base_bps=0.0, k=0.0, exponent=1.0, use_intraday_vol=False),
        volume_limits=VolumeParticipation(max_participation=1.0, mode="proportional", min_fill_notional=0.0),
        book_model=None,
    )

    # Costs: configure commission as requested
    commission = Commission(type="per_share", amount=0.0)
    if commission_per_share is not None:
        commission = Commission(type="per_share", amount=float(commission_per_share))
    elif commission_bps is not None:
        commission = Commission(type="bps_notional", amount=float(commission_bps))

    costs = Costs(
        commission=commission,
        borrow=BorrowCost(default_annual_rate=0.0, curve_name=None),
        financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
        fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
    )

    backtest_cfg = BacktestConfig(
        engine="event_driven",
        cash_initial=100_000_000.0,
        margin=MarginConfig(long_initial=0.5, short_initial=0.5, maintenance=0.3),
        risk_checks=RiskChecks(max_drawdown=None, max_gross_leverage=None),
        reporting=Reporting(store_trades=True, store_positions=True, metrics=[]),
    )

    return Strategy(
        name="test_mom_ls_adv",
        data=data_cfg,
        universe=universe,
        factors=factors,
        signals=signals,
        portfolio=portfolio,
        execution=execution,
        costs=costs,
        backtest=backtest_cfg,
    )


def _build_prices_for_vectorbt(strategy: Strategy) -> pd.DataFrame:
    md = load_market_data(strategy.data, strategy.universe)
    instruments = md.instruments
    prices = pd.DataFrame({instr: md.bars[instr]["close"] for instr in instruments}).sort_index()
    return prices


def _select_with_fill(rank_row: pd.Series, mask: pd.Series, n: int, ascending: bool) -> list[str]:
    masked_valid = rank_row[mask.fillna(False) & rank_row.notna()]
    unmasked_valid = rank_row[rank_row.notna()]

    selected = masked_valid.sort_values(ascending=ascending).head(n).index.tolist()
    if len(selected) < n:
        remaining = n - len(selected)
        filler = (
            unmasked_valid.drop(index=selected, errors="ignore")
            .sort_values(ascending=ascending)
            .head(remaining)
            .index.tolist()
        )
        selected.extend(filler)
    return selected


def _compute_baseline_weights(
    prices: pd.DataFrame,
    *,
    lookback: int = 126,
    long_n: int = 50,
    short_n: int = 50,
    target_gross: float = 2.0,
    target_net: float = 0.0,
    signal_delay_bars: int = 0,
    turnover_limit: float | None = None,
    max_abs_weight_per_name: float | None = None,
) -> pd.DataFrame:
    # 6m log momentum panel
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = np.log(prices / prices.shift(lookback))

    dates = prices.index
    cols = prices.columns
    weights = pd.DataFrame(0.0, index=dates, columns=cols, dtype=float)

    long_gross = max(0.0, (target_gross + target_net) / 2.0)
    short_gross = max(0.0, (target_gross - target_net) / 2.0)

    prev = pd.Series(0.0, index=cols, dtype=float)

    for i, dt in enumerate(dates):
        sig_pos = i - signal_delay_bars
        if sig_pos < 0:
            # Before signals are available, keep previous weights as targets
            w = prev.copy()
            weights.iloc[i] = w
            prev = w
            continue

        sig_date = dates[sig_pos]
        rank_row = mom.loc[sig_date]

        # percentile rank as in SignalEngine (_eval_rank)
        valid = rank_row.dropna()

        need = long_n + short_n
        if int(valid.count()) < need:
            w = pd.Series(0.0, index=cols, dtype=float)
        else:
            # compute percentile ranks 0..1 across all valid
            ranks = valid.rank(method="average", ascending=True) - 1
            denom = max(len(valid) - 1, 1)
            ranks = ranks / denom

            # masks
            long_mask = ranks >= 0.9
            short_mask = ranks <= 0.1

            # selection with fill-from-unmasked
            long_names = _select_with_fill(ranks, long_mask, long_n, ascending=False)
            short_names = _select_with_fill(ranks, short_mask, short_n, ascending=True)

            w = pd.Series(0.0, index=cols, dtype=float)
            if len(long_names) > 0 and long_gross > 0:
                w_long = long_gross / len(long_names)
                w.loc[long_names] = w_long
            if len(short_names) > 0 and short_gross > 0:
                w_short = -short_gross / len(short_names)
                w.loc[short_names] = w_short

            if max_abs_weight_per_name is not None and max_abs_weight_per_name > 0:
                w = w.clip(lower=-max_abs_weight_per_name, upper=max_abs_weight_per_name)

        # Apply turnover limit by scaling towards target
        if turnover_limit is not None and turnover_limit > 0:
            delta = w - prev
            turnover = 0.5 * np.abs(delta).sum()
            if turnover > turnover_limit and turnover != 0:
                scale = float(turnover_limit / turnover)
                w = prev + delta * scale

        weights.iloc[i] = w
        prev = w

    return weights


def _build_vbt_from_weights(prices: pd.DataFrame, weights: pd.DataFrame, *, commission_bps: float | None = None) -> vbt.Portfolio:
    fees = 0.0
    if commission_bps is not None and commission_bps > 0:
        # bps of notional: vectorbt supports passing a scalar fraction
        fees = commission_bps / 1e4
    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        init_cash=100_000_000.0,
        fees=fees,
        slippage=0.0,
        cash_sharing=True,
        call_seq="auto",
        freq="1D",
    )
    return pf


@pytest.mark.slow
def test_momentum_ls_with_signal_delay_matches_vectorbt():
    # DSL strategy with 1-bar signal delay
    strategy = _build_strategy(signal_delay_bars=1)
    result = run_backtest(strategy)

    # Baseline with identical logic
    prices = _build_prices_for_vectorbt(strategy)
    weights = _compute_baseline_weights(
        prices,
        signal_delay_bars=1,
        turnover_limit=None,
        max_abs_weight_per_name=0.05,
    )
    pf = _build_vbt_from_weights(prices, weights, commission_bps=None)

    equity_vbt = pf.value(group_by=True)
    returns_vbt = pf.returns(group_by=True)
    weights_vbt = pf.asset_value().div(pf.value(), axis=0).fillna(0.0)

    equity_dsl = result.equity
    returns_dsl = result.returns

    equity_vbt = equity_vbt.reindex(equity_dsl.index).ffill()
    returns_vbt = returns_vbt.reindex(returns_dsl.index).fillna(0.0)

    assert len(equity_dsl) > 0 and len(returns_dsl) > 0

    np.testing.assert_allclose(
        equity_dsl.values,
        equity_vbt.values,
        rtol=1e-6,
        atol=1e-2,
        err_msg="Equity curves differ with signal delay",
    )
    np.testing.assert_allclose(
        returns_dsl.values,
        returns_vbt.values,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Returns differ with signal delay",
    )

    # Log comparable metrics for both frameworks
    metrics_dsl = compute_basic_metrics(returns_dsl, equity_dsl, result.weights)
    metrics_vbt = compute_basic_metrics(returns_vbt, equity_vbt, weights_vbt)
    log.info("[Metrics] DSL (delay): %s", metrics_dsl)
    log.info("[Metrics] vectorbt (delay): %s", metrics_vbt)


@pytest.mark.slow
def test_momentum_ls_with_commission_and_turnover_matches_vectorbt():
    # DSL strategy: 5 bps commission on notional, 10% daily turnover cap
    commission_bps = 5.0
    turnover_cap = 0.10
    strategy = _build_strategy(signal_delay_bars=0, commission_bps=commission_bps, turnover_limit=turnover_cap)
    result = run_backtest(strategy)

    prices = _build_prices_for_vectorbt(strategy)
    weights = _compute_baseline_weights(
        prices,
        signal_delay_bars=0,
        turnover_limit=turnover_cap,
        max_abs_weight_per_name=0.05,
    )
    pf = _build_vbt_from_weights(prices, weights, commission_bps=commission_bps)

    equity_vbt = pf.value(group_by=True)
    returns_vbt = pf.returns(group_by=True)
    weights_vbt = pf.asset_value().div(pf.value(), axis=0).fillna(0.0)

    equity_dsl = result.equity
    returns_dsl = result.returns

    equity_vbt = equity_vbt.reindex(equity_dsl.index).ffill()
    returns_vbt = returns_vbt.reindex(returns_dsl.index).fillna(0.0)

    assert len(equity_dsl) > 0 and len(returns_dsl) > 0

    np.testing.assert_allclose(
        equity_dsl.values,
        equity_vbt.values,
        rtol=1e-6,
        atol=1e4,  # allow small equity drift due to fee timing nuances
        err_msg="Equity curves differ with commission & turnover cap",
    )
    np.testing.assert_allclose(
        returns_dsl.values,
        returns_vbt.values,
        rtol=1e-6,
        atol=2e-5,
        err_msg="Returns differ with commission & turnover cap",
    )

    # Log comparable metrics for both frameworks
    metrics_dsl = compute_basic_metrics(returns_dsl, equity_dsl, result.weights)
    metrics_vbt = compute_basic_metrics(returns_vbt, equity_vbt, weights_vbt)
    log.info("[Metrics] DSL (fees+turnover): %s", metrics_dsl)
    log.info("[Metrics] vectorbt (fees+turnover): %s", metrics_vbt)
