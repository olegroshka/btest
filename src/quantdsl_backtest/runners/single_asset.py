"""
quantdsl_backtest.runners.single_asset
======================================
Single-instrument DSL interpreter for Strategy(portfolio=TimingPortfolio).

Runs on plain pandas — no vectorbt, no wide [time × instrument] DataFrames.

Signal dispatch is done via double-dispatch: each DSL SignalNode calls back
into ``engine._eval_<node_type>(node)`` (the standard protocol defined in
``dsl/signals.py``). This means new signal types require no changes here —
add the node to your strategy and implement ``_eval_*`` on this class only
if you need a new primitive.
"""
from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd

from ..dsl.factors import ExternalFactor, FieldFactor
from ..dsl.portfolio import TimingPortfolio
from ..dsl.signals import SignalNode
from ..dsl.strategy import Strategy
from ..models.costs import build_cost_model
from ..models.slippage import build_slippage_model


class SingleAssetRunner:
    """
    Interpret a Strategy(portfolio=TimingPortfolio) directly on pandas.

    Parameters
    ----------
    strategy      : Strategy with ``portfolio`` of type :class:`TimingPortfolio`
    btest_root    : root directory used to resolve relative ExternalFactor paths
    rate_registry : optional dict mapping rate-curve names to CSV file paths.
                    CSV must have columns ``date`` and ``rate_pct``.
                    Built-in keys: ``"ESTR"`` (auto-detected from known location).
                    Example::

                        SingleAssetRunner(
                            strategy,
                            btest_root,
                            rate_registry={"ESTR": "/path/to/eur_overnight_rate.csv"},
                        )

    Usage::

        runner = SingleAssetRunner(strategy, btest_root)
        result = runner.run(
            price_close = df["close"],
            aux_series  = {"ivol": df["ivol"]},
        )
        # result keys: position, entry, price, daily_ret, strat_ret, gross_ret,
        #              cost_ret, financing_ret, factors, signals
    """

    # Well-known rate-curve CSV paths (override via rate_registry kwarg)
    _DEFAULT_RATE_REGISTRY: dict = {
        "ESTR": r"C:\Personal\Business & Investments\Trading portfolio\ForgeFolio\data\eur_overnight_rate.csv",
    }

    def __init__(self, strategy: Strategy, btest_root, rate_registry: dict | None = None) -> None:
        if not isinstance(strategy.portfolio, TimingPortfolio):
            raise TypeError(
                f"SingleAssetRunner only handles TimingPortfolio, "
                f"got {type(strategy.portfolio).__name__}"
            )
        self.strategy      = strategy
        self.btest_root    = pathlib.Path(btest_root)
        self._rate_registry = {**self._DEFAULT_RATE_REGISTRY, **(rate_registry or {})}
        # Per-run context — populated in run(), cleared after
        self._computed:    dict[str, pd.Series] = {}
        self._signal_vals: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Public

    def run(
        self,
        price_close: pd.Series,
        aux_series: dict[str, pd.Series] | None = None,
    ) -> dict:
        """
        Run the strategy on ``price_close``.

        Parameters
        ----------
        price_close : pd.Series with DatetimeIndex (close prices)
        aux_series  : extra columns required by FieldFactor nodes

        Returns
        -------
        dict with keys: position, entry, price, daily_ret, strat_ret, factors, signals
        """
        port = self.strategy.portfolio
        idx  = price_close.index

        self._computed = {}
        for name, fnode in self.strategy.factors.items():
            self._computed[name] = self._eval_factor(fnode, idx, aux_series or {})

        # Each node dispatches to engine._eval_<name>(node) via evaluate(self).
        # Earlier signals are already in self._signal_vals when later ones resolve
        # string references — so strategy.signals must be in dependency order.
        self._signal_vals = {}
        for name, snode in self.strategy.signals.items():
            self._signal_vals[name] = snode.evaluate(self)

        entry    = self._signal_vals[port.signal_name].reindex(idx).fillna(False).astype(bool)
        position = (
            entry.shift(port.signal_delay_bars)
            .fillna(False)
            .infer_objects(copy=False)
            .astype(int)
        )

        lev       = port.target_leverage          # 1.0 = unlevered, 2.0 = 2x, etc.
        daily_ret = np.log(price_close / price_close.shift(1))
        gross_ret = (position * lev * daily_ret).fillna(0.0)

        # ── Cost & slippage deduction ─────────────────────────────────────────
        cost_model = build_cost_model(self.strategy.costs.commission)
        slip_model = build_slippage_model(self.strategy.execution.slippage)

        trades = position.diff().abs().fillna(0.0)  # 1.0 on entry/exit bars

        # Commission on lev × notional traded
        prices_arr    = price_close.reindex(idx).values
        comm_frac_arr = np.array([
            cost_model.commission_from_trade(qty=1, exec_price=float(p)) / max(float(p), 1e-9)
            for p in prices_arr
        ])
        comm_frac = pd.Series(comm_frac_arr, index=idx)

        # Slippage: participation=0 (no volume data) → only base_bps fires
        slip_frac = slip_model.slippage_bps_from_participation(0.0) / 1e4

        cost_ret  = (trades * lev * (comm_frac + slip_frac)).fillna(0.0)

        # ── Financing drag on borrowed portion ───────────────────────────────
        # Only applies when lev > 1.0. Charged daily on (lev-1) × position.
        fin_cfg = self.strategy.costs.financing if self.strategy.costs is not None else None
        if fin_cfg is not None and lev > 1.0:
            daily_fin_rate  = self._load_rate_curve(fin_cfg, idx)
            financing_ret   = ((lev - 1) * position * daily_fin_rate).fillna(0.0)
        else:
            financing_ret = pd.Series(0.0, index=idx)

        strat_ret = gross_ret - cost_ret - financing_ret

        result = dict(
            position      = position,
            entry         = entry,
            price         = price_close,
            daily_ret     = daily_ret.fillna(0.0),
            strat_ret     = strat_ret,       # net of commission + slippage + financing
            gross_ret     = gross_ret,       # pre-cost, leveraged
            cost_ret      = cost_ret,        # commission + slippage drag per bar
            financing_ret = financing_ret,   # financing cost on borrowed portion
            factors       = dict(self._computed),
            signals       = dict(self._signal_vals),
        )
        self._computed = {}
        self._signal_vals = {}
        return result

    # ------------------------------------------------------------------
    # Expression resolver  (used by all _eval_* methods below)

    def _resolve(self, expr) -> "pd.Series | float":
        """Resolve a DSL Expr: string ref → Series, scalar → scalar, node → dispatch."""
        if isinstance(expr, str):
            if expr in self._signal_vals:
                return self._signal_vals[expr]
            if expr in self._computed:
                return self._computed[expr]
            raise KeyError(
                f"Expression ref '{expr}' not found in factors or signals"
            )
        if isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, SignalNode):
            return expr.evaluate(self)
        raise TypeError(f"_resolve: unexpected expression type {type(expr).__name__}")

    # ------------------------------------------------------------------
    # Factor evaluation

    def _eval_factor(self, fnode, idx, aux_series: dict) -> pd.Series:
        if isinstance(fnode, ExternalFactor):
            return self._load_external(fnode, idx)
        if isinstance(fnode, FieldFactor):
            if fnode.field not in aux_series:
                raise ValueError(
                    f"FieldFactor '{fnode.name}' needs field '{fnode.field}' "
                    f"in aux_series. Provided: {list(aux_series.keys())}"
                )
            return aux_series[fnode.field].reindex(idx)
        raise NotImplementedError(f"_eval_factor: unsupported {type(fnode).__name__}")

    def _load_rate_curve(self, fin_cfg, idx) -> pd.Series:
        """
        Load a daily financing rate series aligned to ``idx``.

        Resolution order:
        1. ``fin_cfg.rate_csv_path`` — explicit CSV path on ``FinancingCost``
        2. ``self._rate_registry[fin_cfg.base_rate_curve]`` — named registry lookup
        3. Flat series of ``spread_bps / 1e4 / 252`` (spread-only, no base rate)

        CSV format: two columns — ``date`` (parseable) and ``rate_pct``
        (annual rate as a percentage, e.g. 2.50 for 2.50%/yr).
        """
        path = getattr(fin_cfg, "rate_csv_path", None) or \
               self._rate_registry.get(fin_cfg.base_rate_curve)

        if path and pathlib.Path(path).exists():
            rates = pd.read_csv(path, parse_dates=["date"], index_col="date")["rate_pct"]
            daily = (rates.reindex(idx, method="ffill") / 100 + fin_cfg.spread_bps / 1e4) / 252
            return daily.fillna(fin_cfg.spread_bps / 1e4 / 252)

        # No file available — use spread only (logs warning)
        import warnings
        warnings.warn(
            f"FinancingCost: rate curve '{fin_cfg.base_rate_curve}' not found in registry "
            f"and no rate_csv_path provided. Using spread_bps={fin_cfg.spread_bps} only.",
            stacklevel=3,
        )
        return pd.Series(fin_cfg.spread_bps / 1e4 / 252, index=idx)

    def _load_external(self, fnode: ExternalFactor, idx) -> pd.Series:
        path = pathlib.Path(fnode.path)
        if not path.is_absolute():
            path = self.btest_root / path
        if not path.exists():
            raise FileNotFoundError(
                f"ExternalFactor '{fnode.name}': file not found: {path}"
            )

        suffix = path.suffix.lower()
        if suffix == ".pkl":
            with open(path, "rb") as f:
                obj = pickle.load(f)
        elif suffix == ".parquet":
            obj = pd.read_parquet(path)
        elif suffix == ".csv":
            obj = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)

        if fnode.loader is not None:
            series = fnode.loader(obj)
        elif isinstance(obj, pd.Series):
            series = obj
        elif isinstance(obj, pd.DataFrame):
            col = getattr(fnode, "column", None) or obj.columns[0]
            series = obj[col]
        else:
            raise TypeError(
                f"ExternalFactor '{fnode.name}': unsupported type "
                f"'{type(obj).__name__}'. Set loader=<callable> to handle "
                f"custom formats."
            )

        series = series.copy()
        series.index = pd.DatetimeIndex(series.index)
        return series.reindex(idx)

    # ------------------------------------------------------------------
    # Signal _eval_* methods — called back by node.evaluate(self)
    #
    # Adding a new DSL signal type = add one _eval_* method here.
    # The strategy script never needs to change.

    def _eval_zscore_rolling(self, node) -> pd.Series:
        base = self._resolve(node.base)
        mp   = node.min_periods
        mu   = base.rolling(node.window, min_periods=mp).mean()
        sd   = base.rolling(node.window, min_periods=mp).std()
        return (base - mu) / (sd + 1e-9)

    def _eval_mask_from_boolean(self, node) -> pd.Series:
        return self._resolve(node.expr).astype(bool)

    def _eval_greater_equal(self, node) -> pd.Series:
        return self._resolve(node.left) >= self._resolve(node.right)

    def _eval_less_equal(self, node) -> pd.Series:
        return self._resolve(node.left) <= self._resolve(node.right)

    def _eval_less(self, node) -> pd.Series:
        return self._resolve(node.left) < self._resolve(node.right)

    def _eval_greater(self, node) -> pd.Series:
        return self._resolve(node.left) > self._resolve(node.right)

    def _eval_and(self, node) -> pd.Series:
        return self._resolve(node.left).astype(bool) & self._resolve(node.right).astype(bool)

    def _eval_or(self, node) -> pd.Series:
        return self._resolve(node.left).astype(bool) | self._resolve(node.right).astype(bool)

    def _eval_not(self, node) -> pd.Series:
        return ~self._resolve(node.expr).astype(bool)

    def _eval_notnull(self, node) -> pd.Series:
        return self._resolve(node.factor_name).notna()

    def _eval_rolling_mean(self, node) -> pd.Series:
        return self._resolve(node.base).rolling(node.window, min_periods=node.min_periods).mean()

    def _eval_rolling_std(self, node) -> pd.Series:
        return self._resolve(node.base).rolling(node.window, min_periods=node.min_periods).std()

    def _eval_ewm_mean(self, node) -> pd.Series:
        return self._resolve(node.base).ewm(
            span=node.span,
            min_periods=node.min_periods,
            adjust=node.adjust,
        ).mean()

    def _eval_diff(self, node) -> pd.Series:
        return self._resolve(node.base).diff(node.periods)

    def _eval_pct_change(self, node) -> pd.Series:
        return self._resolve(node.base).pct_change(node.periods)

    def _eval_risk_multiplier_from_z(self, node) -> pd.Series:
        z = self._resolve(node.z)
        return (1 - z.clip(0, node.max_z) / node.max_z).clip(0, 1)
