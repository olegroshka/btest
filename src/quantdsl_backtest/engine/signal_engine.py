# src/quantdsl_backtest/engine/signal_engine.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pl = None  # lazy optional

from ..dsl.signals import (
    SignalNode,
    NotNull,
    And,
    Or,
    Not,
    LessEqual,
    GreaterEqual,
    Less,
    Greater,
    Quantile,
    CrossSectionAggregate,
    CrossSectionRank,
    MaskFromBoolean,
    TimeSeries,
    EWMMean,
    RollingMean,
    RollingStd,
    Diff,
    PctChange,
    ZScoreRolling,
    RiskMultiplierFromZ,
)
from ..utils.logging import get_logger


log = get_logger(__name__)


class SignalEngine:
    """
    Evaluate SignalNode DSL into wide DataFrames.

    All outputs share the same index/columns as the factor panel(s).
    """

    def __init__(
        self,
        factors: Dict[str, pd.DataFrame],
        signal_nodes: Dict[str, SignalNode],
        use_polars: Optional[bool] = None,
    ):
        self.factors = factors
        self.nodes = signal_nodes
        # This will hold both "named" signals and intermediate results
        self.cache: Dict[str, pd.DataFrame] = {}

        # Assume all factors share the same shape
        any_factor = next(iter(factors.values()))
        self.index = any_factor.index
        self.columns = any_factor.columns

        # Decide whether to use polars-backed path
        if use_polars is None:
            self._use_polars = pl is not None
        else:
            self._use_polars = bool(use_polars and (pl is not None))

        if self._use_polars:
            log.info("SignalEngine: using Polars-accelerated path where available")
        else:
            if use_polars:
                log.warning(
                    "SignalEngine: use_polars requested but polars not available; falling back to pandas"
                )

    # ------------------------------------------------------------------ #

    def compute_all(self) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for name, node in self.nodes.items():
            out[name] = self._compute_named(name, node)
        return out

    # ------------------------------------------------------------------ #
    # Core evaluation
    # ------------------------------------------------------------------ #

    def _compute_named(self, name: str, node: SignalNode) -> pd.DataFrame:
        if name in self.cache:
            return self.cache[name]
        df = self._evaluate(node)
        self.cache[name] = df
        return df

    def _evaluate(self, node: SignalNode) -> pd.DataFrame:
        # Prefer dynamic dispatch via the Signal interface if available
        evaluate = getattr(node, "evaluate", None)
        if callable(evaluate):
            return evaluate(self)
        if isinstance(node, NotNull):
            return self._eval_notnull(node)
        if isinstance(node, And):
            return self._eval_and(node)
        if isinstance(node, Or):
            return self._eval_or(node)
        if isinstance(node, Not):
            return self._eval_not(node)
        if isinstance(node, LessEqual):
            return self._eval_less_equal(node)
        if isinstance(node, GreaterEqual):
            return self._eval_greater_equal(node)
        if isinstance(node, Less):
            return self._eval_less(node)
        if isinstance(node, Greater):
            return self._eval_greater(node)
        if isinstance(node, Quantile):
            return self._eval_quantile(node)
        if isinstance(node, CrossSectionAggregate):
            return self._eval_cross_section_aggregate(node)
        if isinstance(node, CrossSectionRank):
            return self._eval_rank(node)
        if isinstance(node, MaskFromBoolean):
            return self._eval_mask_from_boolean(node)
        if isinstance(node, TimeSeries):
            return self._eval_time_series(node)
        if isinstance(node, EWMMean):
            return self._eval_ewm_mean(node)
        if isinstance(node, RollingMean):
            return self._eval_rolling_mean(node)
        if isinstance(node, RollingStd):
            return self._eval_rolling_std(node)
        if isinstance(node, Diff):
            return self._eval_diff(node)
        if isinstance(node, PctChange):
            return self._eval_pct_change(node)
        if isinstance(node, ZScoreRolling):
            return self._eval_zscore_rolling(node)
        if isinstance(node, RiskMultiplierFromZ):
            return self._eval_risk_multiplier_from_z(node)
        raise TypeError(f"Unsupported SignalNode type: {type(node)}")

    # ------------------------------------------------------------------ #
    # Node evaluators
    # ------------------------------------------------------------------ #

    def _eval_notnull(self, node: NotNull) -> pd.DataFrame:
        factor = self._get_factor(node.factor_name)
        return factor.notna()

    def _eval_and(self, node: And) -> pd.DataFrame:
        left = self._resolve_expr(node.left).astype(bool)
        right = self._resolve_expr(node.right).astype(bool)
        return left & right

    def _eval_or(self, node: Or) -> pd.DataFrame:
        left = self._resolve_expr(node.left).astype(bool)
        right = self._resolve_expr(node.right).astype(bool)
        return left | right

    def _eval_not(self, node: Not) -> pd.DataFrame:
        expr = self._resolve_expr(node.expr).astype(bool)
        return ~expr

    def _eval_less_equal(self, node: LessEqual) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return left <= right

    def _eval_greater_equal(self, node: GreaterEqual) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return left >= right

    def _eval_less(self, node: Less) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return left < right

    def _eval_greater(self, node: Greater) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return left > right

    def _eval_quantile(self, node: Quantile) -> pd.DataFrame:
        # Support quantile over either a factor or another signal
        if node.factor_name in self.factors:
            factor = self._get_factor(node.factor_name)
        else:
            # Treat as signal name if not a factor
            factor = self._get_signal(node.factor_name)

        mask_df: Optional[pd.DataFrame] = None
        if node.within_mask is not None:
            mask_df = self._get_signal(node.within_mask).astype(bool)

        # Fast path with Polars
        if self._use_polars:
            try:
                # Prepare long tables: (date, symbol, value) and optional mask
                vals = factor.copy()
                vals.index.name = "date"
                pl_df_vals = pl.from_pandas(vals.reset_index())
                pl_vals = pl_df_vals.unpivot(
                    index=["date"],
                    on=[c for c in pl_df_vals.columns if c != "date"],
                    variable_name="symbol",
                    value_name="value",
                )

                if mask_df is not None:
                    mdf = mask_df.copy()
                    mdf.index.name = "date"
                    pl_df_mask = pl.from_pandas(mdf.reset_index())
                    pl_mask = pl_df_mask.unpivot(
                        index=["date"],
                        on=[c for c in pl_df_mask.columns if c != "date"],
                        variable_name="symbol",
                        value_name="mask",
                    )
                    pl_joined = pl_vals.join(pl_mask, on=["date", "symbol"], how="left")
                    pl_filtered = pl_joined.filter(
                        pl.col("mask").fill_null(False) & pl.col("value").is_not_null()
                    )
                else:
                    pl_filtered = pl_vals.filter(pl.col("value").is_not_null())

                # Compute per-date quantile using linear interpolation to match pandas
                q = float(node.q)
                per_date_q = (
                    pl_filtered.group_by("date")
                    .agg(pl.col("value").quantile(q, interpolation="linear").alias("q_val"))
                    .sort("date")
                )

                # Expand to wide by broadcasting q_val to all symbols at each date
                # Build pandas Series of q values indexed by date
                q_pd = per_date_q.to_pandas().set_index("date")["q_val"]
                out = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")
                # Vectorized broadcast across columns for matching dates
                idx = q_pd.index.intersection(out.index)
                if len(idx) > 0:
                    vals = q_pd.loc[idx].to_numpy().reshape(-1, 1)
                    out.loc[idx, :] = np.repeat(vals, len(self.columns), axis=1)
                return out
            except Exception as e:  # Fallback safety
                log.warning(f"Polars path failed in Quantile; falling back to pandas: {e}")

        # Pandas reference implementation
        mask = mask_df
        q = node.q
        out = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")
        for ts in self.index:
            row = factor.loc[ts]
            if mask is not None:
                row_mask = mask.loc[ts]
                row = row[row_mask]
            if row.count() == 0:
                continue
            val = float(row.quantile(q))
            out.loc[ts] = val
        return out

    def _eval_rank(self, node: CrossSectionRank) -> pd.DataFrame:
        factor = self._get_factor(node.factor_name)
        if node.mask_name is not None:
            mask = self._get_signal(node.mask_name).astype(bool)
        else:
            mask = pd.DataFrame(True, index=self.index, columns=self.columns)

        # Polars accelerated path
        if self._use_polars:
            try:
                vals = factor.copy()
                vals.index.name = "date"
                pl_df_vals = pl.from_pandas(vals.reset_index())
                pl_vals = pl_df_vals.unpivot(
                    index=["date"],
                    on=[c for c in pl_df_vals.columns if c != "date"],
                    variable_name="symbol",
                    value_name="value",
                )

                mdf = mask.copy()
                mdf.index.name = "date"
                pl_df_mask = pl.from_pandas(mdf.reset_index())
                pl_mask = pl_df_mask.unpivot(
                    index=["date"],
                    on=[c for c in pl_df_mask.columns if c != "date"],
                    variable_name="symbol",
                    value_name="mask",
                )

                df = pl_vals.join(pl_mask, on=["date", "symbol"], how="left").filter(
                    pl.col("mask").fill_null(False) & pl.col("value").is_not_null()
                )

                if node.method == "percentile":
                    # Create denominator column correctly named as "denom"
                    df = df.with_columns(
                        (pl.len().over("date") - 1).alias("denom")
                    ).with_columns(
                        denom_safe=pl.when(pl.col("denom") <= 0)
                        .then(1)
                        .otherwise(pl.col("denom"))
                    ).with_columns(
                        r=(pl.col("value").rank(method="average").over("date") - 1)
                        / pl.col("denom_safe")
                    )
                elif node.method == "zscore":
                    df = df.with_columns(
                        mu=pl.mean("value").over("date"),
                        sigma=pl.std("value").over("date"),
                    ).with_columns(
                        r=pl.when((pl.col("sigma") == 0) | pl.col("sigma").is_nan())
                        .then(0.0)
                        .otherwise((pl.col("value") - pl.col("mu")) / pl.col("sigma"))
                    )
                else:
                    raise ValueError(f"Unknown rank method: {node.method}")

                # Keep only computed ranks and pivot to wide
                long_out = df.select(["date", "symbol", pl.col("r").alias("rank")])
                pl_wide = long_out.pivot(
                    index="date", on="symbol", values="rank", aggregate_function=None
                ).sort("date")
                out = pl_wide.to_pandas()
                out = out.set_index("date") if "date" in out.columns else out
                # Ensure full shape and order
                out = out.reindex(index=self.index, columns=self.columns)
                # Validate: if Polars path produced suspiciously sparse output, fallback to pandas
                try:
                    total_cells = float(len(self.index) * len(self.columns))
                    valid_cells = float(out.notna().sum().sum())
                    if total_cells > 0 and valid_cells / total_cells < 0.2:
                        raise RuntimeError(
                            f"Polars rank produced sparse output ({valid_cells/total_cells:.3%} valid); falling back to pandas"
                        )
                except Exception as _fallback_trigger:
                    log.warning(f"Polars path produced sparse rank; falling back to pandas: {_fallback_trigger}")
                else:
                    return out
            except Exception as e:  # Fallback safety
                log.warning(f"Polars path failed in Rank; falling back to pandas: {e}")

        # Pandas reference implementation
        out = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")
        for ts in self.index:
            row = factor.loc[ts]
            m = mask.loc[ts]
            valid = row[m].dropna()
            if valid.empty:
                # If no valid values exist at this timestamp, emit NaNs.
                # This mirrors the baseline logic used in integration tests
                # (which gate on the count of valid entries) and avoids
                # artificially selecting names when the factor row is empty.
                out.loc[ts] = np.nan
                continue
            if node.method == "percentile":
                ranks = valid.rank(method="average", ascending=True) - 1
                denom = max(len(valid) - 1, 1)
                ranks = ranks / denom
            elif node.method == "zscore":
                mu = valid.mean()
                sigma = valid.std()
                if sigma == 0 or np.isnan(sigma):
                    ranks = pd.Series(0.0, index=valid.index)
                else:
                    ranks = (valid - mu) / sigma
            else:
                raise ValueError(f"Unknown rank method: {node.method}")
            out.loc[ts, ranks.index] = ranks
        return out

    def _eval_mask_from_boolean(self, node: MaskFromBoolean) -> pd.DataFrame:
        expr = self._resolve_expr(node.expr)
        return expr.astype(bool)

    # ---- Time-series primitives ---------------------------------------- #
    def _eval_time_series(self, node: TimeSeries) -> pd.DataFrame:
        # Lazy import to avoid cycles
        try:
            from ..dsl.data_config import DataConfig  # type: ignore
            from ..data.adapters import load_market_data as _load_md  # type: ignore
        except Exception:
            DataConfig = None  # type: ignore
            _load_md = None  # type: ignore

        # Prepare an empty frame in case of failure
        out = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")
        try:
            if _load_md is None:
                return out
            source = node.source
            if DataConfig is not None and isinstance(source, DataConfig):
                cfg = source
            elif isinstance(source, str):
                # Minimal DataConfig: infer calendar/frequency from factor panel
                from ..dsl.data_config import DataConfig as DC  # type: ignore
                cfg = DC(
                    source=source,
                    calendar="XNYS",
                    frequency="1d",
                    start=str(self.index.min().date()),
                    end=str(self.index.max().date()),
                    price_adjustment="none",
                    fields=[node.field],
                )
            else:
                return out

            md = _load_md(cfg)
            sym = md.instruments[0]
            series = md.bars[sym][node.field].astype("float64").ffill()
            # Align to engine index and broadcast across columns
            s = series.reindex(self.index)
            for c in self.columns:
                out[c] = s
            return out
        except Exception:
            # On any failure, return NaNs; downstream logic should handle gracefully
            return out

    def _eval_ewm_mean(self, node: EWMMean) -> pd.DataFrame:
        base = self._resolve_expr(node.base)
        return base.ewm(span=node.span, min_periods=node.min_periods, adjust=node.adjust).mean()

    def _eval_rolling_mean(self, node: RollingMean) -> pd.DataFrame:
        base = self._resolve_expr(node.base)
        return base.rolling(window=node.window, min_periods=node.min_periods).mean()

    def _eval_rolling_std(self, node: RollingStd) -> pd.DataFrame:
        base = self._resolve_expr(node.base)
        return base.rolling(window=node.window, min_periods=node.min_periods).std()

    def _eval_diff(self, node: Diff) -> pd.DataFrame:
        base = self._resolve_expr(node.base)
        return base.diff(node.periods)

    def _eval_pct_change(self, node: PctChange) -> pd.DataFrame:
        base = self._resolve_expr(node.base)
        return base.pct_change(node.periods)

    def _eval_zscore_rolling(self, node: ZScoreRolling) -> pd.DataFrame:
        base = self._resolve_expr(node.base)
        mu = base.rolling(window=node.window, min_periods=node.min_periods).mean()
        sd = base.rolling(window=node.window, min_periods=node.min_periods).std()
        # Avoid division by zero
        out = (base - mu) / sd
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def _eval_risk_multiplier_from_z(self, node: RiskMultiplierFromZ) -> pd.DataFrame:
        z = self._resolve_expr(node.z)
        clipped = z.clip(lower=0.0, upper=float(node.max_z)) / float(node.max_z)
        mult = 1.0 - clipped
        return mult.clip(lower=0.0, upper=1.0)

    def _eval_cross_section_aggregate(self, node: CrossSectionAggregate) -> pd.DataFrame:
        # Source can be a factor or a previously computed signal
        src_df = self._get_factor(node.source) if node.source in self.factors else self._get_signal(node.source)

        if node.mask_name is not None:
            mask_df = self._get_signal(node.mask_name).astype(bool)
        else:
            # default mask: valid where source is non-null
            mask_df = ~src_df.isna()

        out = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")

        if node.op == "mean":
            reducer = pd.Series.mean
        elif node.op == "median":
            reducer = pd.Series.median
        elif node.op == "sum":
            reducer = pd.Series.sum
        elif node.op == "min":
            reducer = pd.Series.min
        elif node.op == "max":
            reducer = pd.Series.max
        else:
            raise ValueError(f"Unknown cross-sectional aggregate op: {node.op}")

        for ts in self.index:
            row = src_df.loc[ts]
            m = mask_df.loc[ts]
            valid = row[m].dropna()
            if valid.empty:
                # emit NaN; downstream comparisons will be False
                out.loc[ts] = np.nan
                continue
            val = reducer(valid)
            out.loc[ts] = float(val)

        return out

    # ------------------------------------------------------------------ #
    # Helpers: resolve expressions
    # ------------------------------------------------------------------ #

    def _get_factor(self, name: str) -> pd.DataFrame:
        try:
            return self.factors[name]
        except KeyError as exc:
            raise KeyError(f"Unknown factor: {name}") from exc

    def _get_signal(self, name: str) -> pd.DataFrame:
        if name in self.cache:
            return self.cache[name]
        if name not in self.nodes:
            raise KeyError(f"Unknown signal: {name}")
        df = self._evaluate(self.nodes[name])
        self.cache[name] = df
        return df

    def _resolve_expr(self, expr: Any) -> pd.DataFrame:
        """
        Resolve an expression into a DataFrame aligned with factor panels.

        - string -> factor or signal
        - SignalNode -> evaluated recursively
        - scalar (int/float) -> broadcast panel
        - DataFrame/Series -> aligned
        """
        # Already a DataFrame
        if isinstance(expr, pd.DataFrame):
            return expr.reindex(index=self.index, columns=self.columns)

        if isinstance(expr, pd.Series):
            # Broadcast series across columns if index is time
            if isinstance(expr.index, pd.DatetimeIndex):
                df = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")
                for col in self.columns:
                    df[col] = expr
                return df
            else:
                # Assume it's per-instrument
                df = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")
                for col in self.columns:
                    if col in expr.index:
                        df[col] = expr[col]
                return df

        # String: name of factor or signal
        if isinstance(expr, str):
            if expr in self.factors:
                return self._get_factor(expr)
            return self._get_signal(expr)

        # Numeric scalar: broadcast
        if isinstance(expr, (int, float, np.number)):
            return pd.DataFrame(
                expr,
                index=self.index,
                columns=self.columns,
                dtype="float64",
            )

        # Another node
        if isinstance(expr, SignalNode):
            return self._evaluate(expr)

        raise TypeError(f"Unsupported expression type in signals: {type(expr)}")
