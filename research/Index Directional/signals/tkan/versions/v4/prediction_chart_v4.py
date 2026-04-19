"""TKAN v4 - Prediction Path Chart (% normalized)
Two-panel chart:
  Top    : Full CACT price history + retrain markers + current window highlight
  Bottom : % return from anchor (predicted vs actual) -- shapes are visible here
           Predicted: (d1..d10 ratios - 1)*100
           Actual:    (close[t+k]/close[t] - 1)*100
           Threshold: +1.5% horizontal line
  Title  : shows active cycle (model) + val_loss + signal on/off
"""
import os, sys, pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_HERE, *(['..'] * 7)))
_SFERA_DB = os.path.join(_ROOT, "sfera-db")
if os.path.isdir(_SFERA_DB) and _SFERA_DB not in sys.path:
    sys.path.insert(0, _SFERA_DB)
import sfera_db

WEIGHTS_DIR     = os.path.join(_HERE, "weights")
CACHE_PATH      = os.path.join(WEIGHTS_DIR, "pred_cache.pkl")
MANIFEST_PATH   = os.path.join(WEIGHTS_DIR, "manifest.json")
OUTPUT_HTML     = os.path.join(_HERE, "prediction_chart_v4.html")
ENTRY_THRESHOLD = 1.015
PRED_STEP       = 5

if not os.path.isfile(CACHE_PATH):
    raise FileNotFoundError(f"pred_cache.pkl not found - run TKAN_v4_train.py first")

with open(CACHE_PATH, "rb") as f:
    pred_df = pickle.load(f)
pred_df.index = pd.DatetimeIndex(pred_df.index)
TARGET_COLS = [c for c in pred_df.columns if c.startswith("d")]
n_days = len(TARGET_COLS)
print(f"pred_cache: {len(pred_df):,} rows x {n_days} cols  {pred_df.index[0].date()} -> {pred_df.index[-1].date()}")

cactr = (sfera_db.query(
    "SELECT trade_date AS date, close_price AS close "
    "FROM bbgidx.index_total_return WHERE ticker='CACT' ORDER BY trade_date")
    .assign(date=lambda d: pd.to_datetime(d["date"]))
    .set_index("date"))
print(f"CACT: {cactr.index[0].date()} -> {cactr.index[-1].date()} ({len(cactr):,} rows)")

common    = pred_df.index.intersection(cactr.index)
pred_df   = pred_df.loc[common]
close     = cactr.loc[common, "close"]
all_dates = close.index

cycles = []
if os.path.isfile(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    for key, info in sorted(manifest.get("cycles", {}).items()):
        cycles.append((pd.Timestamp(info["retrain_date"]), key, info.get("val_loss", float("nan"))))

retrain_dates = [c[0] for c in cycles]
print(f"Cycles: {len(cycles)}")

def _cycle_for_date(t):
    active = None
    for rd, key, vloss in cycles:
        if rd <= t:
            active = (rd, key, vloss)
        else:
            break
    return active

slider_idx = list(range(0, len(all_dates), PRED_STEP))
if slider_idx[-1] != len(all_dates) - 1:
    slider_idx.append(len(all_dates) - 1)
print(f"Slider steps: {len(slider_idx)}")

thresh_pct = (ENTRY_THRESHOLD - 1.0) * 100.0

def _step_data(i):
    t      = all_dates[i]
    c0     = float(close.iloc[i])
    ratios = pred_df.iloc[i].values
    fwd_dates, pred_pct, act_pct, hover = [], [], [], []
    for k in range(n_days):
        fi = i + k + 1
        if fi >= len(all_dates):
            break
        fd = all_dates[fi]
        p  = (ratios[k] - 1.0) * 100.0
        a  = (float(close.iloc[fi]) / c0 - 1.0) * 100.0
        fwd_dates.append(fd)
        pred_pct.append(p)
        act_pct.append(a)
        hover.append(f"t+{k+1}  {fd.strftime('%Y-%m-%d')}<br>Pred: {p:+.2f}%<br>Actual: {a:+.2f}%<br>Err: {abs(p-a):.2f}%")
    max_pred_pct = max(pred_pct) if pred_pct else 0.0
    signal_on    = max_pred_pct >= thresh_pct
    ci           = _cycle_for_date(t)
    return dict(t=t, c0=c0, fwd_dates=fwd_dates, pred_pct=pred_pct, act_pct=act_pct,
                hover=hover, max_pred_pct=max_pred_pct, signal_on=signal_on,
                cycle_label=ci[1] if ci else "?", val_loss=ci[2] if ci else float("nan"))

fig = make_subplots(rows=2, cols=1, row_heights=[0.30, 0.70],
    shared_xaxes=False, vertical_spacing=0.10,
    subplot_titles=["CACT Total Return (full history — orange lines = model retrains)",
                    "% Return from Anchor: Predicted (green=entry / red=no entry)  vs  Actual (blue)"])

fig.add_trace(go.Scatter(x=all_dates, y=close.values, mode="lines", name="CACT",
    line=dict(color="#90caf9", width=1.2)), row=1, col=1)

for rd, key, _ in cycles:
    if rd in close.index:
        fig.add_vline(x=rd.timestamp()*1000, line=dict(color="orange", width=1, dash="dot"), row=1, col=1)

first = _step_data(slider_idx[0])

# trace 1: current date vertical on row 1
fig.add_trace(go.Scatter(x=[first["t"], first["t"]],
    y=[float(close.min()), float(close.max())],
    mode="lines", line=dict(color="lime", width=1.5, dash="dash"),
    name="Current date", showlegend=False), row=1, col=1)

# trace 2: anchor dot row 1
fig.add_trace(go.Scatter(x=[first["t"]], y=[first["c0"]], mode="markers",
    marker=dict(size=10, color="lime"), name="Anchor", showlegend=False,
    hovertext=[f"{first['t'].date()}  CACT={first['c0']:,.1f}"], hoverinfo="text"), row=1, col=1)

# trace 3: threshold line row 2
fig.add_trace(go.Scatter(x=first["fwd_dates"], y=[thresh_pct]*len(first["fwd_dates"]),
    mode="lines", name=f"+{thresh_pct:.1f}% threshold",
    line=dict(color="gold", width=1.5, dash="dot"), hoverinfo="skip"), row=2, col=1)

# trace 4: zero line row 2
fig.add_trace(go.Scatter(x=first["fwd_dates"], y=[0.0]*len(first["fwd_dates"]),
    mode="lines", name="0%", line=dict(color="#546e7a", width=1),
    hoverinfo="skip", showlegend=False), row=2, col=1)

# trace 5: actual path row 2
fig.add_trace(go.Scatter(x=first["fwd_dates"], y=first["act_pct"],
    mode="lines+markers", name="Actual path",
    line=dict(color="#42a5f5", width=2), marker=dict(size=6, color="#42a5f5"),
    hoverinfo="skip"), row=2, col=1)

# trace 6: predicted path row 2
sig_col0 = "#69f0ae" if first["signal_on"] else "#ef5350"
fig.add_trace(go.Scatter(x=first["fwd_dates"], y=first["pred_pct"],
    mode="lines+markers", name="Predicted path",
    line=dict(color=sig_col0, width=2.5), marker=dict(size=9, symbol="diamond", color=sig_col0),
    text=first["hover"], hoverinfo="text"), row=2, col=1)

def _title(d):
    sig = "ENTRY SIGNAL" if d["signal_on"] else "no entry"
    return (f"TKAN v4  |  {d['t'].strftime('%Y-%m-%d')}  "
            f"|  Model: {d['cycle_label']}  val_loss={d['val_loss']:.5f}  "
            f"|  Max predicted: {d['max_pred_pct']:+.2f}%  |  {sig}")

steps = []
for i in slider_idx:
    d = _step_data(i)
    sc = "#69f0ae" if d["signal_on"] else "#ef5350"
    steps.append(dict(
        method="update",
        label=d["t"].strftime("%Y-%m-%d"),
        args=[
            {"x": [[d["t"], d["t"]], [d["t"]], d["fwd_dates"], d["fwd_dates"], d["fwd_dates"], d["fwd_dates"]],
             "y": [[float(close.min()), float(close.max())], [d["c0"]],
                   [thresh_pct]*len(d["fwd_dates"]), [0.0]*len(d["fwd_dates"]),
                   d["act_pct"], d["pred_pct"]],
             "text": [None, None, None, None, None, d["hover"]],
             "line.color": ["lime", None, "gold", "#546e7a", "#42a5f5", sc],
             "marker.color": [None, "lime", None, None, "#42a5f5", sc]},
            {"title.text": _title(d)},
            [1, 2, 3, 4, 5, 6],
        ],
    ))

fig.update_layout(
    title=_title(first), height=820, width=1500, template="plotly_dark",
    hovermode="x",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(b=160),
    sliders=[dict(active=0, currentvalue=dict(prefix="Date: ", visible=True, xanchor="center"),
                  pad=dict(t=50, b=10), steps=steps, len=0.95, x=0.025)],
)
fig.update_yaxes(title_text="CACT Level", row=1, col=1)
fig.update_yaxes(title_text="% vs anchor", row=2, col=1,
                 zeroline=True, zerolinecolor="#546e7a", zerolinewidth=1)

fig.write_html(OUTPUT_HTML)
print(f"\nSaved: {OUTPUT_HTML}")
