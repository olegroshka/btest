# src/quantdsl_backtest/engine/factor_engine.py

from __future__ import annotations

import pathlib
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..dsl.factors import (
    FactorNode,
    ReturnFactor,
    VolatilityFactor,
    FiboRetraceFactor,
    OvernightReturnFactor,
    IntradayReturnFactor,
    WinsorizedFactor,
    RatioFactor,
    ExternalFactor,
    FieldFactor,
)
from ..data.schema import MarketData
from ..utils.logging import get_logger


log = get_logger(__name__)


class FactorEngine:
    """
    Evaluate factor DSL nodes into wide DataFrames [datetime x instrument].
    """

    def __init__(
        self,
        market_data: MarketData,
        close: pd.DataFrame,
        btest_root: Optional[pathlib.Path] = None,
    ):
        self.market_data = market_data
        self.close = close
        self.index = close.index
        self.instruments = close.columns
        self._cache: Dict[str, pd.DataFrame] = {}
        self.btest_root = pathlib.Path(btest_root) if btest_root else pathlib.Path.cwd()

    def compute_all(self, factors: Dict[str, FactorNode]) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for name, node in factors.items():
            out[name] = self.compute(name, node)
        return out

    def compute(self, name: str, node: FactorNode) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name]

        if isinstance(node, ReturnFactor):
            df = self._compute_return(node)
        elif isinstance(node, VolatilityFactor):
            df = self._compute_volatility(node)
        elif isinstance(node, FiboRetraceFactor):
            df = self._compute_fibo(node)
        elif isinstance(node, OvernightReturnFactor):
            df = self._compute_overnight(node)
        elif isinstance(node, IntradayReturnFactor):
            df = self._compute_intraday(node)
        elif isinstance(node, WinsorizedFactor):
            base_df = self._dispatch_compute_node(node.base)
            df = self._winsorize_cross_section(base_df, z=node.z)
        elif isinstance(node, RatioFactor):
            num = self._dispatch_compute_node(node.numerator)
            den = self._dispatch_compute_node(node.denominator)
            with np.errstate(divide="ignore", invalid="ignore"):
                df = num / den
        elif isinstance(node, ExternalFactor):
            df = self._compute_external(node)
        elif isinstance(node, FieldFactor):
            df = self._field_panel(node.field)
        else:
            raise TypeError(f"Unsupported factor node type: {type(node)}")

        self._cache[name] = df
        return df

    # ------------------------------------------------------------------ #

    def _compute_return(self, node: ReturnFactor) -> pd.DataFrame:
        """
        Compute log or simple returns over `lookback` for each instrument.
        Result aligned with `self.index`.
        """
        field = node.field
        lookback = node.lookback
        method = node.method

        # Build price panel for requested field
        prices = self._field_panel(field)

        if method == "simple":
            ret = prices / prices.shift(lookback) - 1.0
        elif method == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.log(prices / prices.shift(lookback))
        else:
            raise ValueError(f"Unknown return method: {method}")

        return ret

    def _compute_volatility(self, node: VolatilityFactor) -> pd.DataFrame:
        """
        Realized vol / stdev of 1-bar returns over lookback window.
        """
        field = node.field
        lookback = node.lookback
        method = node.method
        annualize = node.annualize
        min_periods = node.min_periods if getattr(node, 'min_periods', None) is not None else lookback

        prices = self._field_panel(field)
        # 1-period log returns
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(prices / prices.shift(1))

        if method in ("realized", "stdev"):
            vol = r.rolling(lookback, min_periods=min_periods).std()
        else:
            raise ValueError(f"Unknown volatility method: {method}")

        if annualize:
            vol *= np.sqrt(252.0)

        return vol

    def _compute_fibo(self, node: FiboRetraceFactor) -> pd.DataFrame:
        """
        Fibonacci retracement level: low + (high - low) * level over a rolling window.
        """
        high_panel = self._field_panel(node.field_high)
        low_panel = self._field_panel(node.field_low)
        hi = high_panel.rolling(node.lookback, min_periods=node.lookback).max()
        lo = low_panel.rolling(node.lookback, min_periods=node.lookback).min()
        fibo = lo + (hi - lo) * node.level
        return fibo

    def _compute_overnight(self, node: OvernightReturnFactor) -> pd.DataFrame:
        """
        Rolling average of overnight returns: log(open_t / close_{t-1}) or simple equivalent.
        """
        open_px = self._field_panel(node.field_open)
        close_px = self._field_panel(node.field_close)

        if node.method == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = np.log(open_px / close_px.shift(1))
        elif node.method == "simple":
            rets = open_px / close_px.shift(1) - 1.0
        else:
            raise ValueError(f"Unknown return method: {node.method}")

        return rets.rolling(node.lookback, min_periods=node.lookback).mean()

    def _compute_intraday(self, node: IntradayReturnFactor) -> pd.DataFrame:
        """
        Rolling average of intraday returns: log(close_t / open_t) or simple equivalent.
        """
        open_px = self._field_panel(node.field_open)
        close_px = self._field_panel(node.field_close)

        if node.method == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = np.log(close_px / open_px)
        elif node.method == "simple":
            rets = close_px / open_px - 1.0
        else:
            raise ValueError(f"Unknown return method: {node.method}")

        return rets.rolling(node.lookback, min_periods=node.lookback).mean()

    def _field_panel(self, field: str) -> pd.DataFrame:
        """
        Get a wide panel [datetime x instrument] for an arbitrary OHLCV field.
        """
        if field == "close":
            return self.close

        # Build from MarketData.bars on demand
        df = pd.DataFrame(index=self.index, columns=self.instruments, dtype="float64")
        for instr, bars in self.market_data.bars.items():
            if field in bars.columns:
                df[instr] = bars[field].reindex(self.index)
        return df

    # ------------------------------------------------------------------ #
    # Internals
    def _dispatch_compute_node(self, node: FactorNode) -> pd.DataFrame:
        """Compute a node without using the external name cache.
        This allows nested nodes (e.g., WinsorizedFactor(base=...))."""
        if isinstance(node, ReturnFactor):
            return self._compute_return(node)
        if isinstance(node, VolatilityFactor):
            return self._compute_volatility(node)
        if isinstance(node, FiboRetraceFactor):
            return self._compute_fibo(node)
        if isinstance(node, OvernightReturnFactor):
            return self._compute_overnight(node)
        if isinstance(node, IntradayReturnFactor):
            return self._compute_intraday(node)
        if isinstance(node, WinsorizedFactor):
            base_df = self._dispatch_compute_node(node.base)
            return self._winsorize_cross_section(base_df, z=node.z)
        if isinstance(node, RatioFactor):
            with np.errstate(divide="ignore", invalid="ignore"):
                return self._dispatch_compute_node(node.numerator) / self._dispatch_compute_node(node.denominator)
        if isinstance(node, ExternalFactor):
            return self._compute_external(node)
        if isinstance(node, FieldFactor):
            return self._field_panel(node.field)
        raise TypeError(f"Unsupported factor node type: {type(node)}")

    def _compute_external(self, node: ExternalFactor) -> pd.DataFrame:
        """
        Load a pre-computed factor from disk and broadcast across all instruments.

        Standard formats (no loader required):
        - pickle / parquet / CSV of ``pd.Series``       → used directly
        - pickle / parquet / CSV of ``pd.DataFrame``    → column ``node.column`` or first column

        Non-standard formats (ML model outputs, compound objects, etc.):
        - Set ``node.loader = lambda obj: <pd.Series>`` to handle any custom structure.
          The loader receives the raw deserialized object and must return a pd.Series
          with a DatetimeIndex.  This keeps format-specific logic out of the engine.

        Example (TKAN pred_cache tuple)::

            def _tkan_loader(obj):
                pred_df = obj[0]          # (pred_df, retrain_dates, fingerprint)
                return pred_df.sum(axis=1)  # cumulative 5d return prediction

            tkan_pred = ExternalFactor(
                name="tkan_pred",
                path="research/Index Directional/tkan/v3/weights/pred_cache.pkl",
                loader=_tkan_loader,
            )
        """
        path = pathlib.Path(node.path)
        if not path.is_absolute():
            path = self.btest_root / path
        if not path.exists():
            raise FileNotFoundError(
                f"ExternalFactor '{node.name}': file not found: {path}"
            )

        suffix = path.suffix.lower()
        if suffix in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                obj = pickle.load(f)
        elif suffix in (".parquet", ".pq"):
            obj = pd.read_parquet(path)
        elif suffix in (".csv",):
            obj = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            # Fallback: try pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)

        # Custom loader: handles any non-standard format (TKAN tuples, etc.)
        if node.loader is not None:
            series = node.loader(obj)
        elif isinstance(obj, pd.Series):
            series = obj
        elif isinstance(obj, pd.DataFrame):
            col = node.column or obj.columns[0]
            series = obj[col]
        else:
            raise TypeError(
                f"ExternalFactor '{node.name}': unsupported type '{type(obj).__name__}'. "
                f"Set node.loader=<callable> to handle custom formats."
            )

        series = series.copy()
        series.index = pd.DatetimeIndex(series.index)
        aligned = series.reindex(self.index)
        return pd.DataFrame(
            {instr: aligned for instr in self.instruments},
            index=self.index,
        )

    def _winsorize_cross_section(self, df: pd.DataFrame, z: float) -> pd.DataFrame:
        """Clip values per date across instruments to mean±z*std (symmetric).
        Preserves index/columns/NaNs."""
        if df.empty:
            return df
        # Compute row-wise mean and std
        mean = df.mean(axis=1, skipna=True)
        std = df.std(axis=1, skipna=True)
        lower = mean - z * std
        upper = mean + z * std
        # Use axis=0 to align Series to rows
        return df.clip(lower=lower, upper=upper, axis=0)
