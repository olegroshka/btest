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
_TKAN_VERSIONS = _HERE / "signals" / "tkan" / "versions"
_WEIGHTS    = _TKAN_VERSIONS / "v3" / "weights"
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
_entry_signal = MaskFromBoolean(
    name="entry_signal",
    expr=And(
        left=GreaterEqual(left="tkan_pred", right=TKAN_THRESH),
        right=Less(left="ivol_z", right=IVOL_Z_THRESH),
    ),
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
        "ivol_z":       _ivol_z,       # continuous z-score — kept for charting & sweep
        "entry_signal": _entry_signal,  # trading signal — TKAN bullish AND IVol regime OK
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
strat_ret   = result["strat_ret"]      # net of commission + slippage
gross_ret   = result["gross_ret"]      # pre-cost
cost_ret    = result["cost_ret"]       # cost drag per bar
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
trade_days  = (position.diff().abs().fillna(0) > 0).sum()
print(f"    trade days : {trade_days}  |  total cost drag : {cost_ret.sum()*100:.2f}%  ({cost_ret[cost_ret>0].mean()*10_000:.1f} bps avg per trade)")

# Per-day ledger — inspect or export
ledger = pd.DataFrame({
    "position":   position,
    "tkan_pred":  tkan_score,
    "ivol_z":     ivol_z_vals,
    "bh_ret":     daily_ret,
    "gross_ret":  gross_ret,
    "cost_ret":   cost_ret,
    "net_ret":    strat_ret,
    "equity":     equity_strat,
})
ledger.index.name = "date"


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
tkan_sig = _sig(tkan_score >= TKAN_THRESH)

rows = [
    compute_metrics(bh_ret,                        bh_ret, bh_pos,   "Buy & Hold"),
    compute_metrics((ivol_sig * bh_ret).fillna(0), bh_ret, ivol_sig, "IVol z-score"),
    compute_metrics((tkan_sig * bh_ret).fillna(0), bh_ret, tkan_sig, "TKAN v3"),
    compute_metrics(gross_ret,                     bh_ret, position, "TKAN + IVol (gross)"),
    compute_metrics(strat_ret,                     bh_ret, position, "TKAN + IVol (net)"),
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
# ── 6. Diagnostic chart ───────────────────────────────────────────────────────
# =============================================================================

try:
    from signum import Chart, Dashboard
except ImportError:
    print("⚠   signum not importable — skipping chart (run from btest venv)")
    Dashboard = None

if Dashboard is not None:
    idx      = df.index
    tkan_on  = tkan_score >= TKAN_THRESH
    ivol_ok  = ivol_z_vals < IVOL_Z_THRESH
    combined = tkan_on & ivol_ok
    blocked  = tkan_on & ~ivol_ok

    def _shade(mask):
        return pd.DataFrame({"position": mask.reindex(idx).fillna(False).astype(int)}, index=idx)

    def _binary(mask):
        return mask.reindex(idx).fillna(False).astype(int)

    def _roll_sharpe(ret_s, w=252):
        mu = ret_s.rolling(w, min_periods=w // 2).mean()
        sd = ret_s.rolling(w, min_periods=w // 2).std()
        return (mu / (sd + 1e-9)) * np.sqrt(252)

    eq_bh   = (1 + bh_ret).cumprod()
    eq_ivol = (1 + (ivol_sig * bh_ret).fillna(0)).cumprod()
    eq_tkan = (1 + (tkan_sig * bh_ret).fillna(0)).cumprod()
    eq_strat = equity_strat

    m  = metrics_df.loc["TKAN + IVol (net)"]
    mb = metrics_df.loc["Buy & Hold"]

    # 1 — Equity curves
    pane1 = Chart(watermark="Equity curves — rebased to 1.0", theme="dark", height=300)
    pane1.line(eq_bh.rename("bh"),       name="Buy & Hold",          color="#78909c", width=1)
    pane1.line(eq_ivol.rename("ivol"),   name="IVol regime only",    color="#f9a825", width=1)
    pane1.line(eq_tkan.rename("tkan"),   name="TKAN v3 only",        color="#ef5350", width=1)
    pane1.line(eq_strat.rename("strat"), name="TKAN + IVol (net)",   color="#66bb6a", width=2)
    pane1.stats_legend({
        "── Strategy ──":   "",
        "Total Return":     f"{m['TotalReturn']:.1f}%",
        "CAGR":             f"{m['CAGR']:.1f}%",
        "Sharpe":           f"{m['Sharpe']:.2f}",
        "Max DD":           f"{m['MaxDD']:.1f}%",
        "In-market":        f"{m['InMktPct']:.0f}%",
        "── Buy & Hold ──": "",
        "B&H Total Return": f"{mb['TotalReturn']:.1f}%",
        "B&H CAGR":         f"{mb['CAGR']:.1f}%",
        "B&H Sharpe":       f"{mb['Sharpe']:.2f}",
        "B&H Max DD":       f"{mb['MaxDD']:.1f}%",
    }, position="top-left")

    # 2 — CACT price + entry shading
    pane2 = Chart(watermark="CACT  |  green = long  |  amber = TKAN bullish, IVol blocked", theme="dark", height=250)
    pane2.line(df["close"].rename("close"), name="CACT TR index", color="#7cb9e8", width=2)
    pane2.shade(_shade(combined), color="#66bb6a", opacity=0.15)
    pane2.shade(_shade(blocked),  color="#f9a825", opacity=0.12)

    # 3 — TKAN prediction
    pane3 = Chart(watermark=f"TKAN 5d cumulative prediction  |  green = bullish (≥ {TKAN_THRESH})  |  red = bearish", theme="dark", height=160)
    pane3.baseline(tkan_score.rename("tkan_pred"), base_value=float(TKAN_THRESH),
                   title="TKAN 5d pred",
                   topFillColor1="rgba(102,187,106,0.40)", topFillColor2="rgba(102,187,106,0.08)",
                   bottomFillColor1="rgba(239,83,80,0.35)", bottomFillColor2="rgba(239,83,80,0.08)",
                   topLineColor="#66bb6a", bottomLineColor="#ef5350")
    pane3.price_line(float(TKAN_THRESH), title=f"threshold {TKAN_THRESH}", color="rgba(255,255,255,0.3)")

    # 4 — IVol z-score
    pane4 = Chart(watermark=f"IVol z-score ({IVOL_WINDOW}d)  |  z < {IVOL_Z_THRESH} = gate OPEN  |  z ≥ {IVOL_Z_THRESH} = gate CLOSED", theme="dark", height=140)
    pane4.line(ivol_z_vals.rename("ivol_z"), name="IVol z-score", color="#f9a825", width=1)
    pane4.price_line(IVOL_Z_THRESH, title=f"gate threshold {IVOL_Z_THRESH}", color="#f9a825")
    pane4.price_line(0.0, title="zero", color="rgba(255,255,255,0.2)")

    # 5a/b/c — Binary signals (0/1 each)
    pane5a = Chart(watermark=f"① TKAN bullish  (pred ≥ {TKAN_THRESH})", theme="dark", height=80)
    pane5a.line(_binary(tkan_on).rename("tkan"), name="TKAN bullish", color="#f9a825", width=1)

    pane5b = Chart(watermark=f"② IVol gate open  (z < {IVOL_Z_THRESH})", theme="dark", height=80)
    pane5b.line(_binary(ivol_ok).rename("ivol_gate"), name="IVol gate open", color="#5b9bd5", width=1)

    pane5c = Chart(watermark="③ Strategy long  (① AND ②)", theme="dark", height=80)
    pane5c.line(_binary(combined).rename("combined"), name="Strategy long", color="#66bb6a", width=1)

    # 6 — Rolling Sharpe
    pane6 = Chart(watermark="Rolling Sharpe (252-day)  |  green = strategy (net)  |  grey = B&H", theme="dark", height=140)
    pane6.line(_roll_sharpe(bh_ret).rename("rs_bh"),     name="Buy & Hold",        color="#78909c", width=1)
    pane6.line(_roll_sharpe(strat_ret).rename("rs_strat"), name="TKAN + IVol (net)", color="#66bb6a", width=2)
    pane6.price_line(0.0, title="zero", color="rgba(255,255,255,0.2)")
    pane6.shade(_shade(combined), color="#66bb6a", opacity=0.08)

    Dashboard(
        panes=[pane1, pane2, pane3, pane4, pane5a, pane5b, pane5c, pane6],
        titles=["1 — Equity Curves", "2 — CACT Price",
                "3 — TKAN Prediction", "4 — IVol z-score",
                "5a — TKAN (0/1)", "5b — IVol gate (0/1)", "5c — Strategy (0/1)",
                "6 — Rolling Sharpe"],
        theme="dark",
    ).show()


# =============================================================================
# ── 7. Threshold sweep — TKAN cutoff ─────────────────────────────────────────
# =============================================================================
from quantdsl_backtest.utils.sweep import threshold_sweep

sweep_df = threshold_sweep(
    factor    = tkan_score,
    gate      = ivol_z_vals < IVOL_Z_THRESH,
    daily_ret = daily_ret,
    bh_ret    = bh_ret,
)
# inspect: sweep_df.sort_values("Sharpe", ascending=False)
