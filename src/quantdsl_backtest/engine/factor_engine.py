# src/quantdsl_backtest/engine/factor_engine.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..dsl.factors import FactorNode, ReturnFactor, VolatilityFactor, FiboRetraceFactor
from ..data.schema import MarketData
from ..utils.logging import get_logger


log = get_logger(__name__)


class FactorEngine:
    """
    Evaluate factor DSL nodes into wide DataFrames [datetime x instrument].
    """

    def __init__(self, market_data: MarketData, close: pd.DataFrame):
        self.market_data = market_data
        self.close = close
        self.index = close.index
        self.instruments = close.columns
        self._cache: Dict[str, pd.DataFrame] = {}

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

        prices = self._field_panel(field)
        # 1-period log returns
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(prices / prices.shift(1))

        if method in ("realized", "stdev"):
            vol = r.rolling(lookback, min_periods=lookback).std()
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
