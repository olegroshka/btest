# src/quantdsl_backtest/engine/analytics/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import pandas as pd

@dataclass(slots=True)
class SignalAnalyticsConfig:
    # which named signals to analyze (keys from strategy.signals / signal_panels)
    signals: List[str]

    # horizons for forward return tests (bars)
    horizons: List[int] = field(default_factory=lambda: [1, 5, 20])

    # quantile bucketing for QC and attribution
    quantiles: int = 5

    # store panels (careful with size)
    store_values: bool = False          # float panel
    store_quantile: bool = True         # int8 panel
    store_rank: bool = False            # float panel

    # optionally apply a mask signal (name) when computing IC/quantiles
    within_mask: Optional[str] = None

    # enforce timing used by portfolio engine
    signal_delay_bars: int = 1          # MUST match portfolio.signal_delay_bars

    # limit storage
    max_instruments: Optional[int] = None   # e.g. 2000
    store_only_traded_names: bool = False   # if you have selection trace


@dataclass(slots=True)
class StrategyAnalyticsConfig:
    """
    Configuration for strategy-level analytics (QuantStats integration).

    Controls printing/logging of a metrics summary and generation of an HTML
    tearsheet using QuantStats, driven by the DSL `Reporting.strategyAnalytics`.
    """

    enabled: bool = True

    # Metrics summary
    # Ordered roughly how a PM/IC would screen a strategy:
    #  1) Return
    #  2) Risk / risk-adjusted
    #  3) Drawdown / tail risk
    #  4) Distribution shape
    #  5) Implementability
    metrics: List[str] = field(
        default_factory=lambda: [
            # --- Return ---
            "total_return",
            "cagr",
            # --- Risk / risk-adjusted ---
            "volatility",
            "sharpe",
            "sortino",
            "calmar",
            # --- Drawdown / tail ---
            "max_drawdown",
            "tail_ratio",
            "ulcer_index",
            "var",
            "cvar",
            # --- Consistency / distribution ---
            "win_rate",
            "skew",
            "kurtosis",
            "profit_factor",
            # --- Implementability / exposure ---
            "turnover",
            "avg_leverage",
            "max_leverage",
            "pct_days_in_market",
        ]
    )
    risk_free: float = 0.0
    prefix: str = "qs_"
    print_metrics: bool = True

    # Tearsheets / files
    write_tearsheet: bool = True
    output_dir: Optional[str] = None  # default: outputs/<strategy>
    file_name: str = "tearsheet.html"
    title: Optional[str] = None
    html_kwargs: Dict[str, object] = field(default_factory=lambda: {"compounded": True, "periods": 252})

    # Benchmark may be provided directly as a returns series, or as a string key/alias
    benchmark: Optional[Union[pd.Series, str]] = None


@dataclass(slots=True)
class SignalTearsheetData:
    name: str
    config: SignalAnalyticsConfig

    # panels (optional)
    value: Optional[pd.DataFrame] = None         # [t x sym]
    rank: Optional[pd.DataFrame] = None          # [t x sym]
    quantile: Optional[pd.DataFrame] = None      # [t x sym] int8-like (1..Q)

    # diagnostics
    coverage: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))  # per date
    xsec_mean: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    xsec_std: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    # IC by horizon
    rank_ic: Dict[int, pd.Series] = field(default_factory=dict)     # horizon -> series
    rank_ic_summary: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())  # horizon x stats

    # quantile returns (mean by date, and summary)
    mean_fwd_ret_by_q: Dict[int, pd.DataFrame] = field(default_factory=dict)  # horizon -> [t x Q]
    ls_fwd_ret: Dict[int, pd.Series] = field(default_factory=dict)           # horizon -> series

    # stability
    quantile_turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))


@dataclass(slots=True)
class SelectionTrace:
    """
    Long table, 1 row per rebalance-date per instrument that was:
      - selected OR near-selected (optional) OR in the candidate universe.
    Keep MVP: only selected rows is enough at first.
    """
    df: pd.DataFrame  # columns: dt, sig_date, book, instrument, selected, score, quantile, target_w, filler, turnover_scale, ...


@dataclass(slots=True)
class PortfolioSignalAttribution:
    # return-space attribution by quantile (cheap, stable)
    contrib_ret_by_q: pd.DataFrame   # [t x Q] and maybe LS column
    contrib_ret_ls: pd.Series        # [t]

    # costs by quantile (optional, depends on trades richness)
    cost_pnl_by_q: Optional[pd.DataFrame] = None

    # optional: constraint shadow diagnostics
    turnover_scale: Optional[pd.Series] = None    # per rebalance date
