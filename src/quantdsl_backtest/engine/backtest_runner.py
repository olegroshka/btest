# src/quantdsl_backtest/engine/backtest_runner.py

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from ..dsl.strategy import Strategy
from ..engine.results import BacktestResult
from ..utils.logging import get_logger
from .data_loader import load_data_for_strategy
from .factor_engine import FactorEngine
from .signal_engine import SignalEngine
from .portfolio_engine import compute_target_weights_for_date
from .execution_engine import rebalance_to_target_weights
from .accounting import (
    mark_to_market,
    apply_carry_costs,
    compute_exposures,
    compute_basic_metrics,
)
from .analytics.selection_trace import SelectionTraceCollector
from .analytics.types import (
    SignalAnalyticsConfig,
    StrategyAnalyticsConfig,
    SignalTearsheetData,
    PortfolioSignalAttribution,
    SelectionTrace,
)
from .analytics.signal_analytics import (
    compute_forward_returns,
    assign_quantiles,
    compute_rank_ic,
    mean_forward_return_by_quantile,
    quantile_turnover,
)
from .analytics.attribution import (
    contrib_return_panel,
    contrib_by_quantile,
    costs_by_instrument_day,
)
from .analytics.render_tearsheets import (
    render_signal_tearsheet_html,
    render_portfolio_signal_tearsheet_html,
)
from pathlib import Path

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run_backtest(strategy: Strategy) -> BacktestResult:
    """
    Dispatch to the appropriate engine implementation based on
    `strategy.backtest.engine`.

    Supported values:
      - "event_driven" : custom daily event loop (current implementation)
      - "vectorized"   : vectorbt-backed engine in `vectorized_engine.py`
    """
    engine_name = getattr(strategy.backtest, "engine", "event_driven")

    if engine_name == "event_driven":
        return _run_backtest_event_driven(strategy)

    if engine_name == "vectorized":
        from .vectorized_engine import run_backtest_vectorized

        log.info("Running strategy %s with vectorized engine", strategy.name)
        return run_backtest_vectorized(strategy)

    raise ValueError(f"Unknown backtest engine: {engine_name!r}")


# --------------------------------------------------------------------------- #
# Existing event-driven engine (unchanged logic)
# --------------------------------------------------------------------------- #


def _run_backtest_event_driven(strategy: Strategy) -> BacktestResult:
    """
    Run an event-driven daily backtest for the given Strategy.

    This is the original implementation which we keep intact for:
      - fine-grained execution control
      - custom slippage / volume / latency models
      - easier debugging
    """

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    md, prices, volumes = load_data_for_strategy(strategy)
    dates = prices.index
    instruments = prices.columns

    # ------------------------------------------------------------------ #
    # 2. Compute factors & signals
    # ------------------------------------------------------------------ #
    factor_engine = FactorEngine(md, prices)
    factor_panels = factor_engine.compute_all(strategy.factors)

    signal_engine = SignalEngine(factor_panels, strategy.signals)
    signal_panels = signal_engine.compute_all()

    # ------------------------------------------------------------------ #
    # 3. Initialize state containers
    # ------------------------------------------------------------------ #
    equity_series = pd.Series(index=dates, dtype="float64")
    return_series = pd.Series(index=dates, dtype="float64")
    cash_series = pd.Series(index=dates, dtype="float64")
    gross_series = pd.Series(index=dates, dtype="float64")
    net_series = pd.Series(index=dates, dtype="float64")
    long_series = pd.Series(index=dates, dtype="float64")
    short_series = pd.Series(index=dates, dtype="float64")
    lev_series = pd.Series(index=dates, dtype="float64")

    positions_df = pd.DataFrame(index=dates, columns=instruments, dtype="float64").fillna(0.0)
    weights_df = pd.DataFrame(index=dates, columns=instruments, dtype="float64").fillna(0.0)

    all_trades = []
    sel_collector = SelectionTraceCollector()

    init_cash = strategy.backtest.cash_initial
    cash = float(init_cash)
    prev_positions = pd.Series(0.0, index=instruments, dtype="float64")
    prev_prices = prices.iloc[0].ffill()
    prev_equity = cash

    # --- Risk checks context ---
    risk = strategy.backtest.risk_checks
    peak_equity = float(strategy.backtest.cash_initial)
    cooldown_days = 0  # simple "no-risk" cooldown counter
    trading_halted = False  # terminal kill-switch state (hard_kill or latched soft_scale)
    soft_scale_latched = False  # once soft_scale reaches full de-risk, stay flat thereafter

    # ------------------------------------------------------------------ #
    # 4. Main daily loop
    # ------------------------------------------------------------------ #
    for i, dt in enumerate(dates):
        price_t = prices.loc[dt]
        volume_t = volumes.loc[dt]

        # Use effective prices for valuation/P&L: if current price is missing (holiday),
        # carry forward the previous price to avoid artificial PnL spikes.
        price_t_eff = price_t.reindex(prev_positions.index)
        price_t_eff = price_t_eff.where(~price_t_eff.isna(), prev_prices)

        if i == 0:
            # Day 0: no prior PnL, just initialize equity
            equity_before = cash + (prev_positions * price_t_eff).sum()
            price_pnl = 0.0
        else:
            equity_before, price_pnl = mark_to_market(
                prev_positions=prev_positions,
                prev_prices=prev_prices,
                curr_prices=price_t_eff,
                prev_cash=cash,
            )

        # Apply carry costs (borrow + financing)
        cash, borrow_cost, fin_pnl = apply_carry_costs(
            positions=prev_positions,
            prices=price_t_eff,
            cash=cash,
            borrow=strategy.costs.borrow,
            financing=strategy.costs.financing,
        )

        # Optionally management fees (we'll apply continuously on equity_before)
        nav_fee = strategy.costs.fees.nav_fee_annual
        if nav_fee > 0:
            nav_fee_daily = nav_fee / 252.0
            fee_amt = equity_before * nav_fee_daily
            cash -= fee_amt

        # Equity before trades at today's prices
        equity_before = cash + (prev_positions * price_t_eff).sum()

        # Decide if we rebalance today
        do_rebalance = _is_rebalance_date(i, dates, strategy.portfolio)

        trades_today = pd.DataFrame()
        # --- Pre-trade drawdown vs running peak (use equity_before) ---
        dd_pretrade = 0.0 if peak_equity == 0 else (equity_before / peak_equity - 1.0)
        dd_mag = -dd_pretrade if dd_pretrade < 0 else 0.0

        # Determine drawdown policy (backward-compatible):
        # If new policy present, use it; else, if max_drawdown set, treat as hard_kill threshold.
        policy = getattr(risk, "drawdown", None)
        policy_mode = getattr(policy, "mode", None) if policy is not None else None
        # Back-compat mapping
        if policy is None and getattr(risk, "max_drawdown", None) is not None:
            policy_mode = "hard_kill"
            policy_threshold = float(getattr(risk, "max_drawdown"))
        else:
            policy_threshold = float(getattr(policy, "threshold", 0.0) or 0.0)

        # Compute scaling according to policy
        scale = 1.0
        if policy_mode == "hard_kill":
            if dd_mag >= (policy_threshold or 0.0):
                trading_halted = True
                scale = 0.0
        elif policy_mode == "soft_scale":
            start = float(getattr(policy, "start", 0.1) or 0.1)
            full = float(getattr(policy, "full", 0.35) or 0.35)
            curve = getattr(policy, "curve", "linear") or "linear"
            if full <= start:
                # guard: if misconfigured, treat as immediate flat beyond start
                full = start
            # Special-case safeguard: if configured to de-risk effectively at any DD
            # (start <= 0 and full ~ 0), then once we have ever taken risk, latch to flat
            # on subsequent days to satisfy "stay-flat-after-derisk" semantics.
            if not soft_scale_latched and start <= 0.0 and full <= 1e-6:
                try:
                    if i > 0:
                        prev_abs_w_sum = float(np.nansum(np.abs(weights_df.iloc[i - 1].values)))
                    else:
                        prev_abs_w_sum = 0.0
                except Exception:
                    prev_abs_w_sum = 0.0
                if prev_abs_w_sum > 1e-6:
                    soft_scale_latched = True
                    trading_halted = True
                    scale = 0.0
            
            # If we previously latched to flat due to soft_scale, remain halted
            if soft_scale_latched:
                trading_halted = True
                scale = 0.0
            elif dd_mag <= start:
                scale = 1.0
            elif dd_mag >= full:
                # Hit full de-risk threshold: latch and halt for the rest of the run
                scale = 0.0
                soft_scale_latched = True
                trading_halted = True
            else:
                x = (dd_mag - start) / (full - start)  # in (0,1)
                if curve == "quadratic":
                    scale = 1.0 - x * x
                elif curve == "sqrt":
                    scale = 1.0 - np.sqrt(x)
                else:  # linear
                    scale = 1.0 - x
        else:
            # "none" or not set: keep scale 1.0
            pass

        if do_rebalance:
            target_weights = compute_target_weights_for_date(
                date=dt,
                portfolio=strategy.portfolio,
                signals=signal_panels,
                prev_weights=weights_df.iloc[i - 1] if i > 0 else pd.Series(0.0, index=instruments),
                collector=sel_collector,
            )

            # Apply drawdown policy scaling / halting
            if trading_halted:
                target_weights = pd.Series(0.0, index=target_weights.index)
            elif scale < 1.0:
                target_weights = target_weights * scale

            # Decide behavior on empty selection
            # By default, we liquidate to cash when the selector produces no targets.
            # Some users prefer to keep the prior positions in that case to avoid
            # spurious flat days caused by temporary data gaps or strict filters.
            # We expose a configuration knob on BacktestConfig to control this:
            #   backtest.extra["hold_when_no_targets"]: bool (default False)
            # - False (default): liquidate to cash when no targets (legacy behavior)
            # - True: carry forward previous positions when no targets and not halted
            hold_when_no_targets = False
            try:
                extra = getattr(strategy.backtest, "extra", {}) or {}
                hold_when_no_targets = bool(extra.get("hold_when_no_targets", False))
            except Exception:
                hold_when_no_targets = False

            empty_targets = float(np.nansum(np.abs(target_weights.values))) <= 1e-12

            if not trading_halted and hold_when_no_targets and empty_targets:
                # Carry forward existing positions (no trades)
                new_positions = prev_positions
                cash_delta = 0.0
                trades_today = pd.DataFrame()
            else:
                # Rebalance normally (if targets empty and hold_when_no_targets is False,
                # this will liquidate to cash as target_weights are all zeros)
                new_positions, cash_delta, trades_today = rebalance_to_target_weights(
                    date=dt,
                    execution=strategy.execution,
                    commission=strategy.costs.commission,
                    fees=strategy.costs.fees,
                    equity=equity_before,
                    prices=price_t_eff,
                    volumes=volume_t,
                    prev_positions=prev_positions,
                    target_weights=target_weights,
                )

            cash += cash_delta
            cur_positions = new_positions
        else:
            cur_positions = prev_positions

        # If trading is halted (hard_kill or latched soft_scale), enforce flat positions
        if trading_halted:
            cur_positions = pd.Series(0.0, index=instruments)

        # Final equity at end of day t
        equity = cash + (cur_positions * price_t_eff).sum()

        if i == 0:
            ret = 0.0
        else:
            ret = (equity / prev_equity) - 1.0 if prev_equity != 0 else 0.0

        # --- Update peak equity & drawdown (before exposures/storage) ---
        if equity > peak_equity:
            peak_equity = equity
        drawdown = 0.0 if peak_equity == 0 else (equity / peak_equity - 1.0)

        # --- Risk checks: drawdown policy informational logging ---
        if policy_mode == "hard_kill" and trading_halted:
            print(
                f"[RISK] Kill-switch: DD {dd_mag: .2%} >= {policy_threshold: .2%} on {dt.date()}, halting trading and staying in cash."
            )

        # --- Risk checks: max_daily_loss + cooldown ---
        if getattr(risk, "max_daily_loss", None) is not None:
            if ret <= -risk.max_daily_loss:
                cooldown_days = max(cooldown_days, 5)  # stay flat for 5 days

        if cooldown_days > 0:
            cooldown_days -= 1
            # Override: ensure we end the day flat & no new positions next loop
            cur_positions = pd.Series(0.0, index=instruments)
            cash = equity

        # Exposures
        exps = compute_exposures(cur_positions, price_t_eff)
        gross = exps["gross_exposure"]
        net = exps["net_exposure"]
        long_exp = exps["long_exposure"]
        short_exp = exps["short_exposure"]
        lev = gross / equity if equity > 0 else 0.0

        # Store
        equity_series.iloc[i] = equity
        return_series.iloc[i] = ret
        cash_series.iloc[i] = cash
        gross_series.iloc[i] = gross
        net_series.iloc[i] = net
        long_series.iloc[i] = long_exp
        short_series.iloc[i] = short_exp
        lev_series.iloc[i] = lev

        positions_df.iloc[i] = cur_positions
        if equity != 0:
            weights_df.iloc[i] = (cur_positions * price_t_eff) / equity
        else:
            weights_df.iloc[i] = 0.0

        if not trades_today.empty:
            all_trades.append(trades_today)

        prev_prices = price_t_eff
        prev_positions = cur_positions
        prev_equity = equity

    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame(
            columns=[
                "datetime",
                "instrument",
                "side",
                "quantity",
                "price",
                "notional",
                "slippage_bps",
                "commission",
                "fees",
                "realized_pnl",
            ]
        )
    )

    # ------------------------------------------------------------------ #
    # 5. Metrics & BacktestResult
    # ------------------------------------------------------------------ #
    metrics = compute_basic_metrics(return_series, equity_series, weights_df)

    result = BacktestResult(
        equity=equity_series,
        returns=return_series,
        cash=cash_series,
        gross_exposure=gross_series,
        net_exposure=net_series,
        long_exposure=long_series,
        short_exposure=short_series,
        leverage=lev_series,
        positions=positions_df,
        weights=weights_df,
        trades=trades_df,
        metrics=metrics,
        start_date=dates[0],
        end_date=dates[-1],
        benchmark=None,
        metadata={
            "strategy_name": strategy.name,
            "data_source": strategy.data.source,
            "engine": "event_driven",
        },
    )

    # ------------------------------------------------------------------ #
    # 6. Optional: Signal analytics & attribution (Alphalens-style)
    # ------------------------------------------------------------------ #
    try:
        # Prefer configuration under Reporting; then deprecated BacktestConfig field; then legacy extra dict
        cfg_raw = None
        try:
            rep = getattr(strategy.backtest, "reporting", None)
            if rep is not None:
                cfg_raw = getattr(rep, "signal_analytics", None)
        except Exception:
            cfg_raw = None
        if cfg_raw is None:
            cfg_raw = getattr(strategy.backtest, "signal_analytics", None)
        if cfg_raw is None:
            extra = getattr(strategy.backtest, "extra", {}) or {}
            cfg_raw = extra.get("signal_analytics")
        if cfg_raw is not None:
            # Build config
            if isinstance(cfg_raw, SignalAnalyticsConfig):
                cfg = cfg_raw
            elif isinstance(cfg_raw, dict):
                cfg = SignalAnalyticsConfig(**cfg_raw)
            else:
                raise TypeError("signal_analytics must be dict or SignalAnalyticsConfig")

            # Enforce engine’s actual delay
            try:
                cfg.signal_delay_bars = int(strategy.portfolio.signal_delay_bars)
            except Exception:
                pass

            # Reference mask (optional)
            mask_df: Optional[pd.DataFrame] = None
            if cfg.within_mask is not None and cfg.within_mask in signal_panels:
                try:
                    mask_df = signal_panels[cfg.within_mask].astype(bool)
                except Exception:
                    mask_df = signal_panels[cfg.within_mask].notna()

            # Forward returns from prices
            fwd = compute_forward_returns(prices, cfg.horizons)

            reports: Dict[str, SignalTearsheetData] = {}
            attribs: Dict[str, PortfolioSignalAttribution] = {}

            # Contributions (return-space) based on realized weights and asset returns
            contrib_panel = contrib_return_panel(weights_df, prices)

            for sname in cfg.signals:
                if sname not in signal_panels:
                    continue
                panel = signal_panels[sname]
                used_panel = panel.shift(cfg.signal_delay_bars)

                # Optionally cap universe size (Tiering)
                if cfg.max_instruments is not None and used_panel.shape[1] > cfg.max_instruments:
                    used_panel = used_panel.iloc[:, : cfg.max_instruments]
                    if mask_df is not None:
                        mask_df = mask_df.reindex(columns=used_panel.columns)

                # Assign quantiles for QC & attribution
                qdf = assign_quantiles(used_panel, cfg.quantiles, mask=mask_df).astype("float32")

                # Build report
                rep = SignalTearsheetData(name=sname, config=cfg)
                if cfg.store_values:
                    rep.value = used_panel.astype("float32")
                if cfg.store_rank:
                    try:
                        rep.rank = used_panel.rank(axis=1, method="average").astype("float32")
                    except Exception:
                        pass
                if cfg.store_quantile:
                    rep.quantile = qdf

                rep.coverage = used_panel.notna().mean(axis=1)
                rep.xsec_mean = used_panel.mean(axis=1, skipna=True)
                rep.xsec_std = used_panel.std(axis=1, skipna=True)

                # IC & quantile returns per horizon
                for h in cfg.horizons:
                    ic = compute_rank_ic(used_panel, fwd[h], mask=mask_df)
                    rep.rank_ic[h] = ic
                    qret, ls = mean_forward_return_by_quantile(qdf, fwd[h], cfg.quantiles)
                    rep.mean_fwd_ret_by_q[h] = qret
                    rep.ls_fwd_ret[h] = ls

                rep.quantile_turnover = quantile_turnover(qdf, cfg.quantiles)
                reports[sname] = rep

                # Attribution by quantile (return-space)
                contrib_by_q, ls_contrib = contrib_by_quantile(contrib_panel, qdf, cfg.quantiles)

                # Costs by quantile (optional)
                cost_panel = costs_by_instrument_day(trades_df)
                cost_by_q = None
                if cost_panel is not None and not cost_panel.empty:
                    # Convert cost pnl to return space approx by dividing equity
                    eq = result.equity.replace(0.0, np.nan)
                    # align index to contrib dates
                    cost_panel = cost_panel.reindex(contrib_by_q.index).fillna(0.0)
                    cost_ret_panel = cost_panel.div(eq, axis=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    cost_by_q, _ = contrib_by_quantile(cost_ret_panel, qdf, cfg.quantiles)

                attribs[sname] = PortfolioSignalAttribution(
                    contrib_ret_by_q=contrib_by_q,
                    contrib_ret_ls=ls_contrib,
                    cost_pnl_by_q=cost_by_q,
                )

            # Attach to result
            result.signal_reports = reports
            result.signal_attribution = attribs
            result.selection_trace = SelectionTrace(sel_collector.finalize()) if len(sel_collector.rows) > 0 else None

            # Render HTML outputs
            out_dir = Path("outputs") / (strategy.name or "run")
            (out_dir / "signals").mkdir(parents=True, exist_ok=True)
            (out_dir / "attribution").mkdir(parents=True, exist_ok=True)

            for sname, rep in reports.items():
                render_signal_tearsheet_html(
                    rep,
                    output_path=out_dir / "signals" / sname / "signal_tearsheet.html",
                    strategy_name=strategy.name,
                    run_meta=result.metadata,
                )
            for sname, attr in attribs.items():
                render_portfolio_signal_tearsheet_html(
                    signal_name=sname,
                    attribution=attr,
                    output_path=out_dir / "attribution" / sname / "portfolio_signal_tearsheet.html",
                    strategy_name=strategy.name,
                    run_meta=result.metadata,
                )
    except Exception as exc:
        log.warning("Signal analytics generation failed: %s", exc)

    # ------------------------------------------------------------------ #
    # 7. Optional: Strategy-level analytics (QuantStats)
    # ------------------------------------------------------------------ #
    try:
        rep = getattr(strategy.backtest, "reporting", None)
        sa_raw = getattr(rep, "strategyAnalytics", None) if rep is not None else None
        if sa_raw is not None:
            if isinstance(sa_raw, StrategyAnalyticsConfig):
                sa = sa_raw
            elif isinstance(sa_raw, dict):
                sa = StrategyAnalyticsConfig(**sa_raw)
            else:
                raise TypeError("strategyAnalytics must be dict or StrategyAnalyticsConfig")

            if sa.enabled:
                # Resolve benchmark: accept Series; for string fall back to result.benchmark
                bm = None
                try:
                    import pandas as _pd  # local import for isinstance check
                    if isinstance(sa.benchmark, _pd.Series):
                        bm = sa.benchmark
                except Exception:
                    bm = None
                if bm is None and isinstance(getattr(sa, "benchmark", None), str):
                    # TODO: implement lookup by alias/name via data loader; for now, fallback
                    bm = result.benchmark
                if bm is None:
                    bm = result.benchmark

                # Metrics summary
                qs_metrics = result.quantstats_metrics(
                    sa.metrics,
                    benchmark=bm,
                    risk_free=sa.risk_free,
                    prefix=sa.prefix,
                )
                if sa.print_metrics:
                    try:
                        log.info("\n=== QuantStats metrics ===\n%s", qs_metrics.to_string(float_format=lambda x: f"{x:0.4f}"))
                    except Exception:
                        log.info("QuantStats metrics: %s", dict(qs_metrics))

                # HTML tearsheet
                if sa.write_tearsheet:
                    out_dir = Path(sa.output_dir) if sa.output_dir else (Path("outputs") / (strategy.name or "run"))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    title = sa.title or strategy.name or "Strategy"
                    html_path = out_dir / sa.file_name
                    result.quantstats_tearsheet(
                        output=str(html_path),
                        title=title,
                        benchmark=bm,
                        **(sa.html_kwargs or {}),
                    )
                    log.info("QuantStats HTML report written to: %s", html_path)
    except RuntimeError as exc:
        log.info("Strategy analytics skipped: %s", exc)
    except Exception as exc:
        log.warning("Strategy analytics generation failed: %s", exc)

    log.info(
        "Backtest complete (event_driven): total return %.2f%%, Sharpe %.2f, max DD %.2f%%",
        result.total_return * 100.0,
        metrics.get("sharpe", 0.0),
        metrics.get("max_drawdown", 0.0) * 100.0,
    )

    return result


def _is_rebalance_date(
    idx: int,
    dates: pd.DatetimeIndex,
    portfolio,
) -> bool:
    """
    Simple daily rebalance logic for now. Need to extend to weekly/monthly.
    """
    freq = portfolio.rebalance_frequency
    if freq == "1d":
        return True
    # For now just do daily; you can extend later.
    return True
