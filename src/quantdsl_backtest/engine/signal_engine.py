# src/quantdsl_backtest/engine/signal_engine.py

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from ..dsl.signals import (
    SignalNode,
    NotNull,
    And,
    Or,
    Not,
    LessEqual,
    GreaterEqual,
    Quantile,
    CrossSectionRank,
    MaskFromBoolean,
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
    ):
        self.factors = factors
        self.nodes = signal_nodes
        # This will hold both "named" signals and intermediate results
        self.cache: Dict[str, pd.DataFrame] = {}

        # Assume all factors share the same shape
        any_factor = next(iter(factors.values()))
        self.index = any_factor.index
        self.columns = any_factor.columns

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
        if isinstance(node, Quantile):
            return self._eval_quantile(node)
        if isinstance(node, CrossSectionRank):
            return self._eval_rank(node)
        if isinstance(node, MaskFromBoolean):
            return self._eval_mask_from_boolean(node)
        raise TypeError(f"Unsupported SignalNode type: {type(node)}")

    # ------------------------------------------------------------------ #
    # Node evaluators
    # ------------------------------------------------------------------ #

    def _eval_notnull(self, node: NotNull) -> pd.DataFrame:
        factor = self._get_factor(node.factor_name)
        return factor.notna()

    def _eval_and(self, node: And) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return (left.astype(bool)) & (right.astype(bool))

    def _eval_or(self, node: Or) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return (left.astype(bool)) | (right.astype(bool))

    def _eval_not(self, node: Not) -> pd.DataFrame:
        expr = self._resolve_expr(node.expr)
        return ~expr.astype(bool)

    def _eval_less_equal(self, node: LessEqual) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return left <= right

    def _eval_greater_equal(self, node: GreaterEqual) -> pd.DataFrame:
        left = self._resolve_expr(node.left)
        right = self._resolve_expr(node.right)
        return left >= right

    def _eval_quantile(self, node: Quantile) -> pd.DataFrame:
        # Support quantile over either a factor or another signal
        if node.factor_name in self.factors:
            factor = self._get_factor(node.factor_name)
        else:
            # Treat as signal name if not a factor
            factor = self._get_signal(node.factor_name)
        mask = None
        if node.within_mask is not None:
            mask = self._get_signal(node.within_mask).astype(bool)

        # For each date, compute quantile across columns
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

        out = pd.DataFrame(index=self.index, columns=self.columns, dtype="float64")

        for ts in self.index:
            row = factor.loc[ts]
            m = mask.loc[ts]

            # Consider only instruments passing mask & non-null
            valid = row[m].dropna()
            if valid.empty:
                continue

            if node.method == "percentile":
                # rank from 0..1
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
