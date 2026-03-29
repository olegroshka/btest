"""
Index Directional — QuantDSL Strategy (standalone script)
==========================================================
Implements TKAN v3 + IVol regime filter using the QuantDSL declarative framework.
Mirrors dsl_strategy.ipynb exactly — run section by section or all at once.

Sections
--------
  0.  Setup & imports
  1.  DSL strategy definition (declarative, no data loaded)
  2.  SingleAssetRunner (from quantdsl_backtest.runners)
  3.  Load data from sfera_db
  4.  Run strategy
  5.  Metrics — compare 4 signal variants
  6.  Diagnostic chart (5 panes)
  7.  Threshold sweep — TKAN cutoff
"""
# ── 0. Setup & imports ────────────────────────────────────────────────────────

import os, sys, pathlib, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

# Resolve paths relative to THIS file's location
_HERE       = pathlib.Path(__file__).resolve().parent          # …/Index Directional/
_BTEST_ROOT = _HERE.parents[1]                                  # btest/
_WS_ROOT    = _BTEST_ROOT.parent                                # workspace root
_TKAN_V3    = _HERE / "tkan" / "v3"
_WEIGHTS    = _TKAN_V3 / "weights"
_OUTPUT     = _BTEST_ROOT / "outputs" / "idx_directional"
_OUTPUT.mkdir(parents=True, exist_ok=True)

import time as _time ; _t0 = _time.perf_counter()

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
    ZScoreRolling, MaskFromBoolean, GreaterEqual, Less, And,
)
from quantdsl_backtest.dsl.portfolio       import TimingPortfolio
from quantdsl_backtest.dsl.execution       import (
    Execution, OrderPolicy, LatencyModel,
    PowerLawSlippageModel, VolumeParticipation,
)
from quantdsl_backtest.dsl.costs           import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig, Reporting

print(f"✅  Imports OK — {_BTEST_ROOT.name}  ({_time.perf_counter()-_t0:.2f}s)")

# ── Strategy constants ────────────────────────────────────────────────────────
BACKTEST_START = "2015-01-01"
IVOL_WINDOW    = 126    # rolling window for IVol z-score (126 trading days ≈ 6 months)
IVOL_Z_THRESH  = 1.0   # z < this → low-fear regime → gate OPEN
TKAN_THRESH    = 0.0   # cumulative 5d prediction ≥ this → bullish
# WINDOW_SIZE / PREDICTION_DAYS / FEATURE_COLS live in the training notebook

# =============================================================================
# ── 1. DSL Strategy Definition ───────────────────────────────────────────────
#
# Purely declarative — no data loaded here.
# TimingRunner (Section 2) interprets these objects.
# =============================================================================

# 1a. Data source (documents intent)
_data = DataConfig(
    source="sfera://bbgidx/index_prices",
    calendar="XPAR",
    frequency="1d",
    start=BACKTEST_START,
    end="2025-12-31",
    fields=["open", "high", "low", "close", "volume", "3m_50d_ivol"],
)

# 1b. Universe
_universe = Universe(name="CAC_TR", static_instruments=["CACT"])

# 1c. Factors
# TKAN pred_cache is saved as a tuple: (pred_df_r1..r5, retrain_dates, fingerprint).
# General NN models should save a plain pd.Series or pd.DataFrame — this loader
# is TKAN-specific and kept here, not in the engine.
def _tkan_loader(obj):
    pred_df = obj[0]                # first element is the per-horizon DataFrame
    return pred_df.sum(axis=1)      # cumulative 5d return prediction

_tkan_pred = ExternalFactor(
    name="tkan_pred",
    path=str(_WEIGHTS / "pred_cache.pkl"),
    loader=_tkan_loader,
)
_ivol_raw = FieldFactor(
    name="ivol_raw",
    field="ivol",
)

# 1d. Signals
_ivol_z = ZScoreRolling(
    name="ivol_z",
    base="ivol_raw",
    window=IVOL_WINDOW,
    min_periods=IVOL_WINDOW // 2,
)
_ivol_ok = MaskFromBoolean(
    name="ivol_ok",
    expr=Less(left="ivol_z", right=IVOL_Z_THRESH),
)
_tkan_ok = MaskFromBoolean(
    name="tkan_ok",
    expr=GreaterEqual(left="tkan_pred", right=TKAN_THRESH),
)
_entry_signal = MaskFromBoolean(
    name="entry_signal",
    expr=And(left="tkan_ok", right="ivol_ok"),
)

# 1e. Portfolio
_portfolio = TimingPortfolio(
    signal_name="entry_signal",
    instrument="CACT",
    rebalance_frequency="1d",
    rebalance_at="market_close",
    signal_delay_bars=1,   # signal at close[T] → position from close[T] to close[T+1]
    target_leverage=1.0,
)

# 1f. Execution & costs
_execution = Execution(
    order_policy=OrderPolicy(),
    latency=LatencyModel(),
    slippage=PowerLawSlippageModel(base_bps=2.0, k=0.0),
    volume_limits=VolumeParticipation(max_participation=1.0),
)
_costs = Costs(
    commission=Commission(type="bps_notional", amount=2.0),
    borrow=BorrowCost(),
    financing=FinancingCost(),
    fees=StaticFees(),
)
_bt = BacktestConfig(
    reporting=Reporting(output_dir=str(_OUTPUT)),
)

# 1g. Compose strategy
strategy = Strategy(
    name="index_directional",
    data=_data,
    universe=_universe,
    factors={
        "tkan_pred": _tkan_pred,
        "ivol_raw":  _ivol_raw,
    },
    signals={
        "ivol_z":       _ivol_z,
        "ivol_ok":      _ivol_ok,
        "tkan_ok":      _tkan_ok,
        "entry_signal": _entry_signal,
    },
    portfolio=_portfolio,
    execution=_execution,
    costs=_costs,
    backtest=_bt,
)

print("✅  Strategy object built:")
print(f"    name         : {strategy.name}")
print(f"    instrument   : {strategy.portfolio.instrument}")
print(f"    signal       : {strategy.portfolio.signal_name}")
print(f"    delay bars   : {strategy.portfolio.signal_delay_bars}")
print(f"    factors      : {list(strategy.factors.keys())}")
print(f"    signals      : {list(strategy.signals.keys())}")
print(f"    portfolio    : {type(strategy.portfolio).__name__}")


# ── 2. TimingRunner (from package) ───────────────────────────────────────────
from quantdsl_backtest.runners import SingleAssetRunner

runner = SingleAssetRunner(strategy, _BTEST_ROOT)


# =============================================================================
# ── 3. Load data from sfera_db ────────────────────────────────────────────────
# =============================================================================

import sfera_db

def _q(sql):
    """Run a sfera query and return a date-indexed DataFrame."""
    return (sfera_db.query(sql)
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .set_index("date"))

cactr       = _q("SELECT trade_date AS date, close_price AS close "
                 "FROM bbgidx.index_total_return WHERE ticker = 'CACT' ORDER BY trade_date")
cac_ohlc    = _q("SELECT trade_date AS date, open_price AS open, high_price AS high, "
                 "low_price AS low, close_price AS cac_close "
                 "FROM bbgidx.index_prices WHERE ticker = 'CAC' ORDER BY trade_date")
ivol_raw_db = _q('SELECT trade_date AS date, "3m_50d_ivol" AS ivol '
                 "FROM bbgidx.index_implied_vol WHERE ticker = 'CAC' ORDER BY trade_date")[["ivol"]]

# Align on common dates
common = cactr.index.intersection(cac_ohlc.index).intersection(ivol_raw_db.index)
df = cac_ohlc.loc[common].copy()
df["close"] = cactr.loc[common, "close"]
df["ivol"]  = ivol_raw_db.loc[common, "ivol"]

# Feature engineering (matches TKAN training notebook exactly)
df["log_return_1d"]  = np.log(df["close"] / df["close"].shift(1))
df["high_low_range"] = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan)
df["close_to_high"]  = (df["high"] - df["cac_close"]) / (df["high"] - df["low"] + 1e-9)
df["sma15"]          = df["close"].rolling(15).mean()
df["close_vs_sma15"] = df["close"] / df["sma15"] - 1
df["return_5d"]      = np.log(df["close"] / df["close"].shift(5))
df["return_20d"]     = np.log(df["close"] / df["close"].shift(20))

hl_sq               = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
park                = np.sqrt((1 / (4 * np.log(2))) * hl_sq.rolling(20).mean() * 252)
close_rvol          = df["log_return_1d"].rolling(20).std() * np.sqrt(252)
df["rvol_park20"]   = park.where(park.notna() & (park > 0), close_rvol)

ivol                     = df["ivol"]
df["ivol_ewma20"]        = ivol.ewm(span=20).mean()
df["ivol_zscore"]        = (ivol - ivol.rolling(IVOL_WINDOW).mean()) / \
                            (ivol.rolling(IVOL_WINDOW).std() + 1e-9)
df["ivol_ema_ratio"]     = ivol / (df["ivol_ewma20"] + 1e-9)
df["ivol_pctl"]          = ivol.rolling(IVOL_WINDOW).apply(
                               lambda x: (x < x[-1]).sum() / (len(x) - 1), raw=True)
df["ivol_roc5"]          = ivol.pct_change(5)
df["rvol_park20_zscore"] = (df["rvol_park20"] - df["rvol_park20"].rolling(IVOL_WINDOW).mean()) / \
                            (df["rvol_park20"].rolling(IVOL_WINDOW).std() + 1e-9)
df["vol_spread"]         = ivol - df["rvol_park20"]

# Trim to backtest window
df = df.loc[df.index >= pd.Timestamp(BACKTEST_START)].copy()

print(f"✅  Data loaded: {len(df):,} rows  "
      f"{df.index[0].date()} → {df.index[-1].date()}")


# =============================================================================
# ── 4. Run DSL strategy via TimingRunner ──────────────────────────────────────
# =============================================================================

result = runner.run(
    price_close = df["close"],
    aux_series  = {"ivol": df["ivol"]},
)

position    = result["position"]
entry       = result["entry"]
strat_ret   = result["strat_ret"]
daily_ret   = result["daily_ret"]
tkan_score  = result["factors"]["tkan_pred"]
ivol_z_vals = result["signals"]["ivol_z"]
equity_strat = (1 + strat_ret).cumprod()

print(f"✅  Strategy run complete")
print(f"    in-market  : {100 * position.mean():.1f}%")
print(f"    data range : {strat_ret.index[0].date()} → {strat_ret.index[-1].date()}")
print(f"    tkan_pred  : {tkan_score.notna().sum():,} non-null rows")
print(f"    tkan_pred range : min={tkan_score.min():.4f}  max={tkan_score.max():.4f}  mean={tkan_score.mean():.4f}")
print(f"    tkan_pred pct>0 : {100*(tkan_score>0).mean():.1f}%  (should not be near 0% or 100% if model works)")
print(f"    weights path    : {_WEIGHTS / 'pred_cache.pkl'}")


# =============================================================================
# ── 5. Metrics — compare 4 signal variants ───────────────────────────────────
# =============================================================================
from quantdsl_backtest.utils.perf import compute_series_metrics as compute_metrics

bh_ret  = daily_ret
bh_pos  = pd.Series(1, index=bh_ret.index)

def _sig(mask):
    """Boolean mask → lagged integer position (signal delay = 1 bar)."""
    return mask.shift(1).fillna(False).infer_objects(copy=False).astype(int)

ivol_sig = _sig(ivol_z_vals < IVOL_Z_THRESH)
tkan_sig = _sig(result["signals"]["tkan_ok"])

rows = [
    compute_metrics(bh_ret,                      bh_ret, bh_pos,   "Buy & Hold"),
    compute_metrics((ivol_sig * bh_ret).fillna(0), bh_ret, ivol_sig, "IVol z-score"),
    compute_metrics((tkan_sig * bh_ret).fillna(0), bh_ret, tkan_sig, "TKAN v3"),
    compute_metrics(strat_ret,                   bh_ret, position, "TKAN + IVol (DSL)"),
]

metrics_df = pd.DataFrame(rows).set_index("Label")
print("\n" + "=" * 70)
print("  Index Directional — Signal Comparison")
print("=" * 70)
print(metrics_df.to_string())
print("=" * 70)

out_path = _OUTPUT / "dsl_metrics.csv"
metrics_df.to_csv(out_path)
print(f"\n  saved → {out_path}")


# =============================================================================
# ── 6. Signum — 4-pane diagnostic chart ──────────────────────────────────────
#
# Pane 1 — CACT price with signal-state shading:
#           ■ green  = TKAN+IVol combined → strategy is long
#           ■ amber  = TKAN is bullish but IVol regime blocked the entry
#           (no shade = neither signal active)
#
# Pane 2 — Raw signal inputs (the "why"):
#           baseline = TKAN 5-day cumulative prediction (green ≥ 0, red < 0)
#           line     = IVol z-score; dashed horizontal = gate threshold
#           Reading: when baseline is green AND line is below the gate → both
#           conditions met → strategy enters (green shade on pane 1)
#
# Pane 3 — Binary activity timeline (when was each component ON?):
#           lane ③ (top)    = combined TKAN+IVol  [green]
#           lane ②          = IVol regime open     [blue]
#           lane ① (bottom) = TKAN signal          [amber]
#           Compare ② vs ① to see how often the IVol gate is the binding
#           constraint vs TKAN being the binding constraint.
#
# Pane 4 — Equity curves — one line per signal layer:
#           grey  = Buy & Hold benchmark
#           amber = IVol regime filter alone (no TKAN)
#           red   = TKAN alone (no IVol gate)
#           green = combined strategy  ← target
# =============================================================================

try:
    from signum import Chart, Dashboard
except ImportError:
    print("⚠   signum not importable — skipping chart (run from btest venv)")
    Dashboard = None

if Dashboard is not None:
    # ── Derived signal states ─────────────────────────────────────────────
    tkan_on  = result["signals"]["tkan_ok"]         # TKAN pred ≥ threshold
    ivol_ok  = result["signals"]["ivol_ok"]         # IVol regime favourable
    combined = tkan_on & ivol_ok                    # both ON → actual long
    blocked  = tkan_on & ~ivol_ok                   # TKAN bullish, IVol says no
    idx      = df.index

    def shade_df(mask):
        """Position DataFrame (0/1) for shade() — accepts DatetimeIndex."""
        return pd.DataFrame(
            {"position": mask.reindex(idx).fillna(False).astype(int)},
            index=idx,
        )

    eq_bh       = (1 + bh_ret).cumprod()
    eq_ivol     = (1 + (ivol_sig * bh_ret).fillna(0)).cumprod()
    eq_tkan     = (1 + (tkan_sig * bh_ret).fillna(0)).cumprod()
    eq_combined = equity_strat

    # Pull metrics for legend annotation
    m_strat = metrics_df.loc["TKAN + IVol (DSL)"]
    m_bh    = metrics_df.loc["Buy & Hold"]

    def step_line(mask, on_val, off_val):
        return pd.Series(
            np.where(mask.reindex(idx).fillna(False), float(on_val), float(off_val)),
            index=idx,
        )

    # ── Pane 1 — Equity curves: result first ─────────────────────────────
    # All curves rebased to 1.0. Strategy (green) vs B&H (grey) is the primary
    # comparison. IVol-only (amber) and TKAN-only (red) isolate each component.
    pane1 = Chart(watermark="Equity curves — rebased to 1.0", theme="dark", height=300)
    pane1.line(eq_bh.rename("bh"),
               name="Buy & Hold",
               color="#78909c", width=1)
    pane1.line(eq_ivol.rename("ivol"),
               name="IVol regime only",
               color="#f9a825", width=1)
    pane1.line(eq_tkan.rename("tkan"),
               name="TKAN v3 only",
               color="#ef5350", width=1)
    pane1.line(eq_combined.rename("combined"),
               name="TKAN + IVol (strategy)",
               color="#66bb6a", width=2)
    pane1.stats_legend({
        "── Strategy ──":    "",
        "Total Return":      f"{m_strat['TotalReturn']:.1f}%",
        "CAGR":              f"{m_strat['CAGR']:.1f}%",
        "Sharpe":            f"{m_strat['Sharpe']:.2f}",
        "Max DD":            f"{m_strat['MaxDD']:.1f}%",
        "In-market":         f"{m_strat['InMktPct']:.0f}%",
        "── Buy & Hold ──":  "",
        "B&H Total Return":  f"{m_bh['TotalReturn']:.1f}%",
        "B&H CAGR":          f"{m_bh['CAGR']:.1f}%",
        "B&H Sharpe":        f"{m_bh['Sharpe']:.2f}",
        "B&H Max DD":        f"{m_bh['MaxDD']:.1f}%",
    }, position="top-left")

    # ── Pane 2 — CACT price + entry shading ──────────────────────────────
    # Green background = strategy is long. Amber = TKAN bullish but IVol blocked it.
    pane2 = Chart(
        watermark="CACT total return  |  green bg = strategy long  |  amber bg = TKAN bullish but IVol regime blocked entry",
        theme="dark", height=250,
    )
    pane2.line(df["close"].rename("close"), name="CACT total return index", color="#7cb9e8", width=2)
    pane2.shade(shade_df(combined), color="#66bb6a", opacity=0.15)
    pane2.shade(shade_df(blocked),  color="#f9a825", opacity=0.12)

    # ── Pane 3 — TKAN 5-day cumulative return prediction ─────────────────
    # Baseline chart: green fill = model predicts positive 5d return (→ entry signal ON).
    # Red fill = model predicts negative return (→ signal OFF).
    # Scale is ~[-0.05, +0.05] log return — looks flat if mixed with IVol (0–8 range).
    pane3 = Chart(
        watermark=f"TKAN v3: 5-day cumulative return prediction  |  green fill = bullish (pred >= {TKAN_THRESH})  |  red fill = bearish",
        theme="dark", height=180,
    )
    pane3.baseline(
        tkan_score.rename("tkan_pred"),
        base_value=float(TKAN_THRESH),
        title=f"TKAN 5d pred (sum r1..r5)",
        topFillColor1="rgba(102,187,106,0.40)", topFillColor2="rgba(102,187,106,0.08)",
        bottomFillColor1="rgba(239,83,80,0.35)", bottomFillColor2="rgba(239,83,80,0.08)",
        topLineColor="#66bb6a", bottomLineColor="#ef5350",
    )
    pane3.price_line(float(TKAN_THRESH), title="entry threshold (0)", color="rgba(255,255,255,0.3)")

    # ── Pane 4 — IVol z-score (regime filter) ────────────────────────────
    # Amber line = IVol z-score relative to 6-month rolling mean.
    # BELOW dashed threshold (z < 1.0) = low-fear regime = IVol gate OPEN → allow entries.
    # ABOVE threshold = elevated fear / vol spike = gate CLOSED → block entries even if TKAN bullish.
    pane4 = Chart(
        watermark=(
            f"IVol z-score  —  (IVol − rolling {IVOL_WINDOW}d mean) / rolling {IVOL_WINDOW}d std  ({IVOL_WINDOW} trading days ≈ 6 months)  |  "
            f"z < {IVOL_Z_THRESH} → IVol is BELOW its recent average → low-fear regime → gate OPEN  |  "
            f"z ≥ {IVOL_Z_THRESH} → IVol spike above average → elevated fear → gate CLOSED"
        ),
        theme="dark", height=160,
    )
    pane4.line(ivol_z_vals.rename("ivol_z"),
               name=f"IVol z-score  (gate opens when z < {IVOL_Z_THRESH})",
               color="#f9a825", width=1)
    pane4.price_line(IVOL_Z_THRESH, title=f"gate threshold z={IVOL_Z_THRESH}", color="#f9a825")
    pane4.price_line(0.0,           title="zero",                              color="rgba(255,255,255,0.2)")

    # ── Pane 5 — Signal ON/OFF timeline ──────────────────────────────────
    # Step lines per component. Line HIGH = signal is ON, LOW = OFF.
    # When green (③) is high → strategy is long.
    # When amber (①) is high but green (③) is low → TKAN bullish but IVol blocked it.
    # When blue (②) is high but green (③) is low → IVol open but TKAN is bearish.
    pane5 = Chart(
        watermark="Signal ON/OFF  |  ① TKAN bullish  ② IVol gate open  ③ Strategy long  |  HIGH=ON  LOW=OFF",
        theme="dark", height=120,
    )
    pane5.line(step_line(combined, 2.8, 2.2).rename("combined"),
               name="③ Strategy LONG (TKAN AND IVol both active)", color="#66bb6a", width=2)
    pane5.line(step_line(ivol_ok,  1.8, 1.2).rename("ivol_gate"),
               name="② IVol regime OK — gate open (z-score < 1.0)", color="#5b9bd5", width=2)
    pane5.line(step_line(tkan_on,  0.8, 0.2).rename("tkan_signal"),
               name="① TKAN bullish — 5d prediction >= 0",           color="#f9a825", width=2)

    Dashboard(
        panes=[pane1, pane2, pane3, pane4, pane5],
        titles=[
            "1 — Equity Curves (strategy result)",
            "2 — CACT Price + Entry Regions",
            "3 — TKAN Signal: 5-day return prediction",
            "4 — IVol Regime Filter: z-score gate",
            "5 — Signal ON/OFF Timeline",
        ],
        theme="dark",
    ).show()


# =============================================================================
# ── 7. Threshold sweep — TKAN cutoff ─────────────────────────────────────────
# =============================================================================

t_range = np.percentile(tkan_score.dropna(), np.arange(5, 76, 5))
sweep_rows = []

for thr in t_range:
    for ivol_gate in [False, True]:
        tkan_mask = tkan_score >= thr
        if ivol_gate:
            mask  = tkan_mask & (ivol_z_vals < IVOL_Z_THRESH)
            label = f"TKAN+IVol  thr={thr:.4f}"
        else:
            mask  = tkan_mask
            label = f"TKAN only  thr={thr:.4f}"
        pos_s = mask.shift(1).fillna(False).infer_objects(copy=False).astype(int)
        ret_s = (pos_s * daily_ret).fillna(0)
        m     = compute_metrics(ret_s, bh_ret, pos_s, label)
        m["thr"]       = round(float(thr), 5)
        m["ivol_gate"] = ivol_gate
        sweep_rows.append(m)

sweep_df = pd.DataFrame(sweep_rows).set_index("Label")
print("\n── Threshold Sweep ──────────────────────────────────────────────────")
print(sweep_df[["thr", "ivol_gate", "Sharpe", "CAGR", "MaxDD", "Calmar", "InMktPct"]].to_string())

best = sweep_df["Sharpe"].idxmax()
print(f"\n  ★ Best Sharpe → {best}  "
      f"(Sharpe={sweep_df.loc[best,'Sharpe']:.3f}, "
      f"CAGR={sweep_df.loc[best,'CAGR']:.1f}%, "
      f"InMkt={sweep_df.loc[best,'InMktPct']:.0f}%)")

# Sweep chart — matplotlib (Signum forces datetime x-axis; sweep needs numeric x)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

tonly = sweep_df[~sweep_df["ivol_gate"]].sort_values("thr")
tcomb = sweep_df[ sweep_df["ivol_gate"]].sort_values("thr")
x     = tonly["thr"].values          # actual threshold values on x-axis

BG   = "#131722"
AX   = "#1e222d"
RED  = "#ef5350"
GRN  = "#66bb6a"
GREY = "#78909c"
WHT  = "#d1d4dc"

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2, 2]})
fig.patch.set_facecolor(BG)
fig.suptitle("TKAN threshold sensitivity sweep  —  red = TKAN only,  green = TKAN + IVol regime filter",
             color=WHT, fontsize=11, y=0.98)

for ax in axes:
    ax.set_facecolor(AX)
    ax.tick_params(colors=GREY, labelsize=9)
    ax.spines[:].set_color("#2a2e39")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(axis="both", color="#2a2e39", linewidth=0.5, linestyle="--")

# ── Panel 1 : Sharpe ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(x, tonly["Sharpe"].values, color=RED, linewidth=2, label="TKAN only")
ax.plot(x, tcomb["Sharpe"].values, color=GRN, linewidth=2, label="TKAN + IVol")
ax.axhline(0, color=GREY, linewidth=0.8, linestyle=":")
# mark the best combined point
best_idx = tcomb["Sharpe"].idxmax()
bx, by = tcomb.loc[best_idx, "thr"], tcomb.loc[best_idx, "Sharpe"]
ax.scatter([bx], [by], color="white", zorder=5, s=60)
ax.annotate(f"  ★ best  thr={bx:.4f}  Sharpe={by:.2f}",
            (bx, by), color="white", fontsize=8, va="center")
ax.set_ylabel("Sharpe ratio", color=WHT, fontsize=9)
ax.legend(loc="lower left", fontsize=8, facecolor=AX, edgecolor="#2a2e39",
          labelcolor=WHT, framealpha=0.85)

# ── Panel 2 : CAGR ───────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(x, tonly["CAGR"].values, color=RED, linewidth=1.5, label="TKAN only")
ax.plot(x, tcomb["CAGR"].values, color=GRN, linewidth=2,   label="TKAN + IVol")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.set_ylabel("CAGR", color=WHT, fontsize=9)

# ── Panel 3 : In-market % ─────────────────────────────────────────────────────
ax = axes[2]
ax.plot(x, tonly["InMktPct"].values, color=RED, linewidth=1.5, label="TKAN only")
ax.plot(x, tcomb["InMktPct"].values, color=GRN, linewidth=2,   label="TKAN + IVol")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylabel("In-market %", color=WHT, fontsize=9)
ax.set_xlabel("TKAN 5d-return threshold  (lower = more permissive, more in-market)",
              color=GREY, fontsize=9)

fig.tight_layout()
plt.savefig(str(_HERE / "sweep_chart.png"), dpi=130, bbox_inches="tight",
            facecolor=BG)
plt.show()
print(f"  sweep chart saved → {_HERE / 'sweep_chart.png'}")
