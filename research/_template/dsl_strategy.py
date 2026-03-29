"""
<Strategy Name> — QuantDSL Single-Asset Strategy
=================================================
TODO: one-line description of what this strategy does.

Sections
--------
  0.  Setup & imports          [boilerplate — do not change]
  1.  DSL strategy definition  [strategy-specific]
  2.  SingleAssetRunner        [boilerplate — do not change]
  3.  Load data                [strategy-specific]
  4.  Run strategy             [boilerplate — do not change]
  5.  Metrics                  [boilerplate — do not change]
  6.  Diagnostic chart         [pane 1 boilerplate; panes 2-N strategy-specific]
  7.  Parameter sweep          [structure boilerplate; sweep target strategy-specific]
"""

# =============================================================================
# ── 0. Setup & imports  [BOILERPLATE] ────────────────────────────────────────
# =============================================================================

import os, sys, pathlib, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time as _time ; _t0 = _time.perf_counter()
import numpy as np
import pandas as pd

_HERE       = pathlib.Path(__file__).resolve().parent
_BTEST_ROOT = _HERE.parents[1]           # btest/
_WS_ROOT    = _BTEST_ROOT.parent         # workspace root
_OUTPUT     = _BTEST_ROOT / "outputs" / "TODO_strategy_slug"
_OUTPUT.mkdir(parents=True, exist_ok=True)

for _p in [
    str(_WS_ROOT),
    str(_WS_ROOT / "sfera-db"),
    str(_WS_ROOT / "signum"),
    str(_BTEST_ROOT / "src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantdsl_backtest.dsl.strategy        import Strategy
from quantdsl_backtest.dsl.data_config     import DataConfig
from quantdsl_backtest.dsl.universe        import Universe
from quantdsl_backtest.dsl.factors         import ExternalFactor, FieldFactor
from quantdsl_backtest.dsl.signals         import (
    # TODO: import only the signal nodes your strategy uses
    ZScoreRolling, MaskFromBoolean, GreaterEqual, Less, And,
)
from quantdsl_backtest.dsl.portfolio       import TimingPortfolio
from quantdsl_backtest.dsl.execution       import (
    Execution, OrderPolicy, LatencyModel,
    PowerLawSlippageModel, VolumeParticipation,
)
from quantdsl_backtest.dsl.costs           import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting
from quantdsl_backtest.runners             import SingleAssetRunner
from quantdsl_backtest.utils.perf          import compute_series_metrics as compute_metrics

print(f"✅  Imports OK  ({_time.perf_counter()-_t0:.2f}s)")

# =============================================================================
# ── 1. DSL Strategy Definition  [STRATEGY-SPECIFIC] ─────────────────────────
#
# Purely declarative — no data loaded here.
# SingleAssetRunner (section 2) interprets these objects.
# =============================================================================

# ── Strategy constants ── TODO: define your parameters ───────────────────────
BACKTEST_START = "2015-01-01"
# TODO: add signal windows, thresholds, etc.

# 1a. Data source
_data = DataConfig(
    source="sfera://TODO/TODO",         # TODO: sfera table URI
    calendar="XPAR",                    # TODO: exchange calendar
    frequency="1d",
    start=BACKTEST_START,
    end="2025-12-31",
    fields=["open", "high", "low", "close", "volume"],  # TODO: add fields
)

# 1b. Universe
_universe = Universe(
    name="TODO_UNIVERSE",
    static_instruments=["TODO_TICKER"],  # TODO: ticker(s)
)

# 1c. Factors  ── TODO: define ExternalFactor / FieldFactor nodes
# ExternalFactor example (NN model output):
#   def _my_loader(obj): return obj[0].sum(axis=1)
#   _my_factor = ExternalFactor(name="my_pred", path="...", loader=_my_loader)
#
# FieldFactor example (column already in aux_series):
#   _vol_factor = FieldFactor(name="vol_raw", field="vol")

# 1d. Signals  ── TODO: build your signal tree
# Example:
#   _z = ZScoreRolling(name="vol_z", base="vol_raw", window=126, min_periods=63)
#   _ok = MaskFromBoolean(name="entry_signal", expr=Less(left="vol_z", right=1.0))

# 1e. Portfolio
_portfolio = TimingPortfolio(
    signal_name   = "entry_signal",     # TODO: must match the final MaskFromBoolean name
    instrument    = "TODO_TICKER",      # TODO
    rebalance_frequency = "1d",
    rebalance_at  = "market_close",
    signal_delay_bars = 1,              # signal at close[T] → position close[T]→close[T+1]
    target_leverage   = 1.0,
)

# 1f. Execution & costs  (defaults — adjust as needed)
_execution = Execution(
    order_policy  = OrderPolicy(),
    latency       = LatencyModel(),
    slippage      = PowerLawSlippageModel(base_bps=2.0, k=0.0),
    volume_limits = VolumeParticipation(max_participation=1.0),
)
_costs = Costs(
    commission = Commission(type="bps_notional", amount=2.0),
    borrow     = BorrowCost(),
    financing  = FinancingCost(),
    fees       = StaticFees(),
)

# 1g. Compose strategy  ── TODO: fill in factors/signals dicts
strategy = Strategy(
    name      = "TODO_strategy_name",
    data      = _data,
    universe  = _universe,
    factors   = {
        # "my_factor": _my_factor,
    },
    signals   = {
        # "vol_z":        _z,
        # "entry_signal": _ok,
    },
    portfolio  = _portfolio,
    execution  = _execution,
    costs      = _costs,
    backtest   = BacktestConfig(reporting=Reporting(output_dir=str(_OUTPUT))),
)

print("✅  Strategy object built:")
print(f"    factors  : {list(strategy.factors.keys())}")
print(f"    signals  : {list(strategy.signals.keys())}")
print(f"    signal   : {strategy.portfolio.signal_name}  (delay {strategy.portfolio.signal_delay_bars}d)")

# =============================================================================
# ── 2. Runner  [BOILERPLATE] ─────────────────────────────────────────────────
# =============================================================================

runner = SingleAssetRunner(strategy, _BTEST_ROOT)

# =============================================================================
# ── 3. Load data  [STRATEGY-SPECIFIC] ────────────────────────────────────────
#
# Goal: produce a DataFrame `df` with at minimum:
#   df["close"]        — price series (used for returns)
#   df["<factor_col>"] — any columns needed by FieldFactor nodes in aux_series
# Trim to BACKTEST_START at the end.
# =============================================================================

import sfera_db

def _q(sql):
    return (sfera_db.query(sql)
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .set_index("date"))

# TODO: replace with your actual queries
# df = _q("SELECT trade_date AS date, close_price AS close FROM ... WHERE ticker='TODO'")

# TODO: feature engineering (if required by FieldFactor cols)

# TODO: trim to backtest window
# df = df.loc[df.index >= pd.Timestamp(BACKTEST_START)].copy()

print(f"✅  Data loaded: {len(df):,} rows  {df.index[0].date()} → {df.index[-1].date()}")

# =============================================================================
# ── 4. Run strategy  [BOILERPLATE] ───────────────────────────────────────────
# =============================================================================

result = runner.run(
    price_close = df["close"],
    aux_series  = {
        # TODO: map field names to df columns required by FieldFactor nodes
        # "vol": df["vol"],
    },
)

position    = result["position"]
strat_ret   = result["strat_ret"]
daily_ret   = result["daily_ret"]
equity_strat = (1 + strat_ret).cumprod()

# TODO: extract any specific factors/signals you need for charting/sweep
# my_score  = result["factors"]["my_factor"]
# my_signal = result["signals"]["entry_signal"]

print(f"✅  Run complete — in-market: {100 * position.mean():.1f}%  "
      f"({strat_ret.index[0].date()} → {strat_ret.index[-1].date()})")

# =============================================================================
# ── 5. Metrics  [BOILERPLATE] ────────────────────────────────────────────────
# =============================================================================

bh_ret = daily_ret
bh_pos = pd.Series(1, index=bh_ret.index)

def _sig(mask):
    """Boolean mask → lagged integer position (1-bar delay)."""
    return mask.shift(1).fillna(False).infer_objects(copy=False).astype(int)

rows = [
    compute_metrics(bh_ret,   bh_ret, bh_pos,   "Buy & Hold"),
    compute_metrics(strat_ret, bh_ret, position, "Strategy"),
    # TODO: add intermediate signal variants for comparison, e.g.:
    # compute_metrics((_sig(my_signal) * bh_ret).fillna(0), bh_ret, _sig(my_signal), "My signal alone"),
]

metrics_df = pd.DataFrame(rows).set_index("Label")
print("\n" + "=" * 70)
print(f"  {strategy.name} — Signal Comparison")
print("=" * 70)
print(metrics_df.to_string())
print("=" * 70)

metrics_df.to_csv(_OUTPUT / "metrics.csv")
print(f"\n  saved → {_OUTPUT / 'metrics.csv'}")

# =============================================================================
# ── 6. Diagnostic chart  [BOILERPLATE pane 1; remainder STRATEGY-SPECIFIC] ──
# =============================================================================

try:
    from signum import Chart, Dashboard
except ImportError:
    print("⚠   signum not importable — skipping chart")
    Dashboard = None

if Dashboard is not None:
    eq_bh       = (1 + bh_ret).cumprod()
    m_strat     = metrics_df.loc["Strategy"]
    m_bh        = metrics_df.loc["Buy & Hold"]

    # ── Pane 1 — Equity curves  [BOILERPLATE] ────────────────────────────
    pane1 = Chart(watermark="Equity curves — rebased to 1.0", theme="dark", height=300)
    pane1.line(eq_bh.rename("bh"),          name="Buy & Hold",  color="#78909c", width=1)
    pane1.line(equity_strat.rename("strat"), name="Strategy",    color="#66bb6a", width=2)
    # TODO: add intermediate curves if you computed them above
    pane1.stats_legend({
        "── Strategy ──":   "",
        "Total Return":     f"{m_strat['TotalReturn']:.1f}%",
        "CAGR":             f"{m_strat['CAGR']:.1f}%",
        "Sharpe":           f"{m_strat['Sharpe']:.2f}",
        "Max DD":           f"{m_strat['MaxDD']:.1f}%",
        "In-market":        f"{m_strat['InMktPct']:.0f}%",
        "── Buy & Hold ──": "",
        "B&H Total Return": f"{m_bh['TotalReturn']:.1f}%",
        "B&H CAGR":         f"{m_bh['CAGR']:.1f}%",
        "B&H Sharpe":       f"{m_bh['Sharpe']:.2f}",
        "B&H Max DD":       f"{m_bh['MaxDD']:.1f}%",
    }, position="top-left")

    # ── Panes 2-N  [STRATEGY-SPECIFIC] ───────────────────────────────────
    # TODO: add signal-specific panes. Examples:
    #
    # Price + entry shading:
    #   pane2 = Chart(watermark="Price + entries", theme="dark", height=250)
    #   pane2.line(df["close"].rename("close"), name="Price", color="#7cb9e8", width=2)
    #   pane2.shade(pd.DataFrame({"position": position}, index=df.index),
    #               color="#66bb6a", opacity=0.15)
    #
    # Baseline (factor with fill above/below zero):
    #   pane3 = Chart(watermark="My factor score", theme="dark", height=180)
    #   pane3.baseline(my_score.rename("score"), base_value=0.0, ...)
    #
    # Line + threshold:
    #   pane4 = Chart(watermark="Z-score gate", theme="dark", height=160)
    #   pane4.line(my_z.rename("z"), color="#f9a825", width=1)
    #   pane4.price_line(THRESHOLD, title="gate", color="#f9a825")

    Dashboard(
        panes  = [pane1],   # TODO: extend with your panes
        titles = ["1 — Equity Curves"],  # TODO: extend
        theme  = "dark",
    ).show()

# =============================================================================
# ── 7. Parameter sweep  [BOILERPLATE structure; sweep target STRATEGY-SPECIFIC]
# =============================================================================

# TODO: define what to sweep (e.g. signal threshold, lookback window)
# SWEEP_PARAM = np.linspace(...)   or   np.percentile(my_score.dropna(), ...)

sweep_rows = []

# for param_val in SWEEP_PARAM:
#     # TODO: compute mask from param_val
#     mask  = my_score >= param_val
#     pos_s = mask.shift(1).fillna(False).infer_objects(copy=False).astype(int)
#     ret_s = (pos_s * bh_ret).fillna(0)
#     m     = compute_metrics(ret_s, bh_ret, pos_s, f"thr={param_val:.4f}")
#     m["param"] = round(float(param_val), 5)
#     sweep_rows.append(m)

if sweep_rows:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sweep_df = pd.DataFrame(sweep_rows).set_index("Label")
    x = sweep_df["param"].values

    BG, AX, GRN, GREY, WHT = "#131722", "#1e222d", "#66bb6a", "#78909c", "#d1d4dc"

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"{strategy.name} — parameter sensitivity sweep",
                 color=WHT, fontsize=11, y=0.98)

    for ax in axes:
        ax.set_facecolor(AX)
        ax.tick_params(colors=GREY, labelsize=9)
        ax.spines[:].set_color("#2a2e39")
        for spine in ax.spines.values(): spine.set_linewidth(0.5)
        ax.grid(axis="both", color="#2a2e39", linewidth=0.5, linestyle="--")

    axes[0].plot(x, sweep_df["Sharpe"].values,   color=GRN, linewidth=2)
    axes[0].set_ylabel("Sharpe",     color=WHT, fontsize=9)
    axes[1].plot(x, sweep_df["CAGR"].values,     color=GRN, linewidth=2)
    axes[1].set_ylabel("CAGR",       color=WHT, fontsize=9)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    axes[2].plot(x, sweep_df["InMktPct"].values, color=GRN, linewidth=2)
    axes[2].set_ylabel("In-market %", color=WHT, fontsize=9)
    axes[2].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[2].set_xlabel("Parameter value", color=GREY, fontsize=9)

    fig.tight_layout()
    plt.savefig(str(_HERE / "sweep_chart.png"), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"  sweep chart saved → {_HERE / 'sweep_chart.png'}")
