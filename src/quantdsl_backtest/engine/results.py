# src/quantdsl_backtest/engine/results.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    """
    Comprehensive backtest result container.

    This is designed to support:
      - Equity curve / returns analysis
      - Exposure & leverage analysis
      - Position & trade-level analysis
      - Easy export to DataFrames / Parquet
      - Light plotting helpers for quick inspection

    Conventions:
      - All time series are indexed by a DatetimeIndex aligned on bar end.
      - `positions` / `weights` are wide DataFrames (columns = instruments).
      - `trades` is a long DataFrame with one row per execution.
    """

    # ---- Core P&L series ---------------------------------------------------
    equity: pd.Series              # NAV over time
    returns: pd.Series             # period returns, aligned with equity.index
    cash: pd.Series                # cash balance

    # ---- Exposure / leverage ----------------------------------------------
    gross_exposure: pd.Series      # |long| + |short|
    net_exposure: pd.Series        # long - short
    long_exposure: pd.Series       # long-only exposure
    short_exposure: pd.Series      # short-only (negative) exposure
    leverage: pd.Series            # gross_exposure / equity

    # ---- Positions & weights -----------------------------------------------
    positions: pd.DataFrame        # units (shares / contracts) per instrument
    weights: pd.DataFrame          # weights (exposure / equity) per instrument

    # ---- Trades -----------------------------------------------------------
    trades: pd.DataFrame
    """
    Expected columns (engine can add more):

        datetime : Timestamp (index or column)
        instrument : str
        side : {"BUY", "SELL", "SHORT", "COVER"} or similar
        quantity : float  (signed or absolute; your choice, but be consistent)
        price : float
        notional : float
        slippage : float      (slippage cost for this trade, in PnL units)
        commission : float
        fees : float
        realized_pnl : float  (if you track it trade-by-trade)
        order_id : optional
        trade_id : optional
    """

    # ---- Metrics & metadata -----------------------------------------------
    metrics: Dict[str, float]      # summary metrics: sharpe, max_dd, etc.
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    benchmark: Optional[pd.Series] = None   # optional benchmark equity/returns

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Free-form metadata, e.g.:
        "strategy_name": str
        "data_source": str
        "parameters": dict
        "run_id": str
        etc.
    """

    # ----------------------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------------------

    @property
    def index(self) -> pd.DatetimeIndex:
        """The primary time index of the backtest (equity index)."""
        return self.equity.index

    @property
    def total_return(self) -> float:
        """Total return over the full period."""
        if len(self.equity) < 2:
            return 0.0
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    # ----------------------------------------------------------------------
    # Slicing / subsetting
    # ----------------------------------------------------------------------

    def slice(self, start: Optional[str] = None, end: Optional[str] = None) -> "BacktestResult":
        """
        Return a new BacktestResult restricted to [start, end].

        `start` and `end` can be anything pandas can parse as a Timestamp.
        """
        if start is not None:
            start_ts = pd.to_datetime(start)
        else:
            start_ts = self.index[0]

        if end is not None:
            end_ts = pd.to_datetime(end)
        else:
            end_ts = self.index[-1]

        mask = (self.index >= start_ts) & (self.index <= end_ts)

        # Helper to slice Series/DataFrames that share the main index
        def _slice_ts(obj):
            return obj.loc[mask] if len(obj) > 0 else obj

        # Slice trades separately by datetime column or index
        trades = self.trades
        if not trades.empty:
            if "datetime" in trades.columns:
                tmask = (trades["datetime"] >= start_ts) & (trades["datetime"] <= end_ts)
                trades = trades.loc[tmask]
            elif isinstance(trades.index, pd.DatetimeIndex):
                trades = trades.loc[start_ts:end_ts]

        return BacktestResult(
            equity=_slice_ts(self.equity),
            returns=_slice_ts(self.returns),
            cash=_slice_ts(self.cash),
            gross_exposure=_slice_ts(self.gross_exposure),
            net_exposure=_slice_ts(self.net_exposure),
            long_exposure=_slice_ts(self.long_exposure),
            short_exposure=_slice_ts(self.short_exposure),
            leverage=_slice_ts(self.leverage),
            positions=_slice_ts(self.positions),
            weights=_slice_ts(self.weights),
            trades=trades,
            metrics=dict(self.metrics),  # metrics are global; you can recompute if needed
            start_date=start_ts,
            end_date=end_ts,
            benchmark=self.benchmark.loc[start_ts:end_ts] if self.benchmark is not None else None,
            metadata=dict(self.metadata),
        )

    # ----------------------------------------------------------------------
    # Export helpers
    # ----------------------------------------------------------------------

    def equity_frame(self) -> pd.DataFrame:
        """Return a DataFrame with the core equity & exposure series."""
        df = pd.DataFrame(
            {
                "equity": self.equity,
                "returns": self.returns,
                "cash": self.cash,
                "gross_exposure": self.gross_exposure,
                "net_exposure": self.net_exposure,
                "long_exposure": self.long_exposure,
                "short_exposure": self.short_exposure,
                "leverage": self.leverage,
            }
        )
        if self.benchmark is not None:
            df["benchmark"] = self.benchmark
        return df

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result into a JSON-serializable dict skeleton.
        (Note: large time-series will be converted via .to_dict())
        """
        return {
            "equity": self.equity.to_dict(),
            "returns": self.returns.to_dict(),
            "cash": self.cash.to_dict(),
            "gross_exposure": self.gross_exposure.to_dict(),
            "net_exposure": self.net_exposure.to_dict(),
            "long_exposure": self.long_exposure.to_dict(),
            "short_exposure": self.short_exposure.to_dict(),
            "leverage": self.leverage.to_dict(),
            "positions": {c: self.positions[c].to_dict() for c in self.positions.columns},
            "weights": {c: self.weights[c].to_dict() for c in self.weights.columns},
            "trades": self.trades.to_dict(orient="records"),
            "metrics": dict(self.metrics),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
            "metadata": self.metadata,
        }

    def to_parquet(self, path: str, include_trades: bool = True, include_positions: bool = True) -> None:
        """
        Save the result to a folder of Parquet files:

            path/
              equity.parquet
              positions.parquet
              weights.parquet
              trades.parquet       (optional)
              metadata.parquet     (key/value as small table)
        """
        import os
        import json

        os.makedirs(path, exist_ok=True)

        self.equity_frame().to_parquet(os.path.join(path, "equity.parquet"))
        if include_positions:
            self.positions.to_parquet(os.path.join(path, "positions.parquet"))
            self.weights.to_parquet(os.path.join(path, "weights.parquet"))
        if include_trades:
            self.trades.to_parquet(os.path.join(path, "trades.parquet"))

        # Save metrics + metadata as a tiny table.
        # Ensure the 'value' column is STRING to avoid pyarrow mixed-type errors.
        def _serialize_value(val) -> str:
            try:
                # Preserve timestamps nicely
                if hasattr(val, "isoformat"):
                    return val.isoformat()
                # Basic scalars
                if isinstance(val, (int, float, bool, str)) or val is None:
                    return json.dumps(val)
                # Try JSON for dict/list/np types
                return json.dumps(val, default=str)
            except Exception:
                return str(val)

        meta_records = []
        for k, v in self.metrics.items():
            meta_records.append({"key": f"metric:{k}", "value": _serialize_value(v)})
        for k, v in self.metadata.items():
            meta_records.append({"key": f"meta:{k}", "value": _serialize_value(v)})

        meta_df = pd.DataFrame(meta_records, columns=["key", "value"]).astype({"key": "string", "value": "string"})
        meta_df.to_parquet(os.path.join(path, "metadata.parquet"))

    # ----------------------------------------------------------------------
    # Summary & plotting
    # ----------------------------------------------------------------------

    def summary(self) -> pd.Series:
        """
        Return a pandas Series with key summary metrics. This is handy
        for printing/logging or quick inspection in a notebook.
        """
        s = pd.Series(self.metrics).copy()
        s["start_date"] = self.start_date
        s["end_date"] = self.end_date
        s["total_return"] = self.total_return
        return s

    def plot_equity(self, ax=None) -> Any:
        """
        Professional-looking equity curve plot.

        - Lazy-imports matplotlib
        - Applies a clean style and grid
        - Uses concise date formatting
        - Returns the provided/new axis without calling plt.show()
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for plot_equity(), "
                "please install it (e.g. `pip install matplotlib`)."
            ) from exc

        # Apply a modern style without requiring seaborn package
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            # Fallback to basic style if not available in older matplotlib
            plt.style.use("default")

        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
            created_fig = True

        # Plot lines with consistent palette and thicker linewidths
        ax.plot(self.equity.index, self.equity.astype(float).values,
                label="Equity", color="#1f77b4", linewidth=2.0)
        if self.benchmark is not None:
            try:
                ax.plot(self.benchmark.index, self.benchmark.astype(float).values,
                        label="Benchmark", color="#2ca02c", linestyle="--", linewidth=1.6)
            except Exception:
                # Be tolerant to odd benchmark types
                self.benchmark.plot(ax=ax, label="Benchmark", linestyle="--", linewidth=1.6)

        # Formatting
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.margins(x=0)

        # Labels and legend
        ax.set_title("Equity Curve", fontsize=12, pad=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.legend(loc="best", frameon=False)

        # Lighten spines for a cleaner look
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if created_fig:
            plt.tight_layout()
        return ax

    def plot_exposure(self, ax=None) -> Any:
        """
        Professional-looking plot of gross/net/long/short exposures over time.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for plot_exposure(), "
                "please install it (e.g. `pip install matplotlib`)."
            ) from exc

        # Style
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("default")

        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
            created_fig = True

        # Consistent colors
        ax.plot(self.long_exposure.index, self.long_exposure.astype(float).values,
                label="Long", color="#2ca02c", linewidth=1.8)
        ax.plot(self.short_exposure.index, self.short_exposure.astype(float).values,
                label="Short", color="#d62728", linewidth=1.8)
        ax.plot(self.net_exposure.index, self.net_exposure.astype(float).values,
                label="Net", color="#9467bd", linewidth=1.8)
        ax.plot(self.gross_exposure.index, self.gross_exposure.astype(float).values,
                label="Gross", color="#7f7f7f", linewidth=1.6)

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.margins(x=0)

        ax.set_title("Exposures", fontsize=12, pad=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Exposure (notional)")
        ax.legend(loc="best", frameon=False, ncol=2)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if created_fig:
            plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Friendly aliases requested by examples
    # ------------------------------------------------------------------

    def plot_equity_curve(self, ax=None) -> Any:
        """Alias for plot_equity()."""
        return self.plot_equity(ax=ax)

    def plot_exposures(self, ax=None) -> Any:
        """Alias for plot_exposure()."""
        return self.plot_exposure(ax=ax)

    def plot_drawdowns(self, ax=None) -> Any:
        """
        Professional drawdown plot derived from the equity curve.

        Drawdown is computed as: equity / rolling_max(equity) - 1.
        Plotted as a filled area below zero for clarity.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required for plot_drawdowns(), "
                "please install it (e.g. `pip install matplotlib`)."
            ) from exc

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("default")

        created_fig = False
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4.5))
            created_fig = True

        eq = self.equity.astype(float)
        roll_max = eq.cummax()
        dd = (eq / roll_max) - 1.0  # negative numbers
        dd_pct = dd * 100.0

        # Filled area under zero for visual impact
        ax.fill_between(dd_pct.index, dd_pct.values, 0.0, color="crimson", alpha=0.30)
        ax.plot(dd_pct.index, dd_pct.values, color="crimson", linewidth=1.2, label="Drawdown (%)")

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.margins(x=0)

        # Y-limits slightly above zero to emphasize downside
        ymin = float(min(-1.0, dd_pct.min() * 1.05))  # cap at -100% or slightly lower than min
        ax.set_ylim(ymin, 2.0)

        ax.set_title("Drawdowns", fontsize=12, pad=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="best", frameon=False)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        if created_fig:
            plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # QuantStats integration
    # ------------------------------------------------------------------

    def quantstats_metrics(
        self,
        metric_names: Sequence[str],
        benchmark: Optional[pd.Series] = None,
        risk_free: float = 0.0,
        prefix: str = "qs_",
    ) -> pd.Series:
        """Compute a configurable set of QuantStats metrics.

        Examples
        --------
        >>> result.quantstats_metrics(
        ...     ["cagr", "sharpe", "sortino", "max_drawdown"]
        ... )
        qs_cagr           0.145
        qs_sharpe         1.45
        qs_sortino        2.10
        qs_max_drawdown  -0.23
        Name: quantstats, dtype: float64
        """
        from .metrics_quantstats import compute_quantstats_metrics

        bench = benchmark if benchmark is not None else self.benchmark
        metrics = compute_quantstats_metrics(
            returns=self.returns,
            metric_names=metric_names,
            benchmark=bench,
            risk_free=risk_free,
        )
        # Optionally prefix keys so they don't collide with core metrics
        keyed = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
        return pd.Series(keyed, name="quantstats")

    def quantstats_tearsheet(
        self,
        output: Optional[str] = None,
        title: Optional[str] = None,
        benchmark: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """Generate a QuantStats HTML tearsheet for this backtest.

        Parameters
        ----------
        output:
            Path to the HTML output file.
        title:
            Optional report title; defaults to strategy name (if present).
        benchmark:
            Optional benchmark returns Series. Falls back to self.benchmark
            if provided there.
        **kwargs:
            Passed through to quantstats.reports.html, e.g.
            compounded=True, periods=252, etc.
        """
        from .metrics_quantstats import generate_quantstats_tearsheet

        bench = benchmark if benchmark is not None else self.benchmark
        if title is None:
            title = self.metadata.get("strategy_name", "Strategy")

        generate_quantstats_tearsheet(
            returns=self.returns,
            benchmark=bench,
            output=output,
            title=title,
            **kwargs,
        )


