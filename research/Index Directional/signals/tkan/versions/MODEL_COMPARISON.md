# TKAN Model Comparison: v1 vs v2

## Quick Summary

| | **V1** | **V2** |
|---|---|---|
| **Features** | 5 (raw OHLCV + SMA) | 11 (OHLCV + 6 volatility features) |
| **Data** | LVC only (Excel) | LVC + CAC index + CAC IVol (CSV / Sfera DB) |
| **Target** | Close price (MinMax scaled) | Close price (MinMax scaled) |
| **Training** | Retrain every run if new data | Retrain every run if new data |
| **Walk-forward** | No | Yes, via `train_models.py` (126-day cycles) |
| **Architecture** | 3×TKAN(100) → Dense(1) | 3×TKAN(100) → Dense(1) |
| **Dropout** | None | None |
| **Vol awareness** | None | 6 IVol/RVol features + exit gate |
| **Signal suppression** | None | IVol exit gate (80th pctl, 2d confirm) |
| **Prediction** | 10 business days | 10 business days |

---

## Input Features

### V1 — 5 features (all lagged 1 day)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Prior Close | Previous day's close price |
| 2 | Prior High | Previous day's high |
| 3 | Prior Low | Previous day's low |
| 4 | Prior Volume | Previous day's volume |
| 5 | Prior SMAVG(15) | 15-day simple moving average |

### V2 — 11 features (all lagged 1 day)

| # | Feature | Description |
|---|---------|-------------|
| 1 | prior_close | Previous close price |
| 2 | prior_high | Previous high |
| 3 | prior_low | Previous low |
| 4 | prior_volume | Previous volume |
| 5 | prior_sma15 | 15-day SMA |
| 6 | **prior_ivol** | CAC 3M 50D implied vol |
| 7 | **prior_ivol_ema20** | 20-period EMA of IVol (short-term trend) |
| 8 | **prior_ivol_pctl** | 126-day rolling percentile rank of IVol |
| 9 | **prior_ivol_roc5** | 5-day % change in IVol (momentum) |
| 10 | **prior_rvol_park20** | 20-day Parkinson realized vol (CAC underlying) |
| 11 | **prior_vol_spread** | IVol − RVol (vol risk premium) |

V2 also has a separate walk-forward pipeline (`train_models.py`) that uses **13 stationary features** (log returns, z-scores, ratios instead of raw prices).

---

## Training

### V1 — Simple retrain-on-update
- Single 95/5 train/test split on all available data
- Retrains automatically when new data is detected (weights file older than latest data)
- Saves to `tkan_model_weights.weights.h5`
- **Epochs:** 50, **Batch size:** 32
- **Scaler:** MinMaxScaler on both X and y

### V2 (`TKAN_Index_v2.py`) — Same approach as V1
- Same 95/5 split, auto-retrain on data update
- Saves to `tkan_v2_weights.weights.h5`
- **Epochs:** 50, **Batch size:** 32
- **Scaler:** MinMaxScaler on both X and y

### V2 Walk-Forward (`train_models.py`) — Quarterly retrain
- Retrain every **126 business days** (~6 months) starting from 2015-01-01
- Each cycle trains on all data up to that point (expanding window)
- **23 cycles** covering 2015–2025
- Saves per-cycle weights + scalers to `weights/` directory
- Uses **RobustScaler** on X, no scaler on y (targets are price ratios ≈ 1.0)
- Resumable — manifest tracks completed cycles, skips if config unchanged

---

## Model Architecture

Identical in both versions:

```
Input(batch, 10, n_features)
  → TKAN(100, return_sequences=True)
  → TKAN(100, return_sequences=True)
  → TKAN(100, return_sequences=True)
  → Dense(1)
Output(batch, 10, 1)
```

Only difference: input width (5 features for v1, 11 for v2).

---

## Signal Logic

### V1
1. Predict 10 days of close prices
2. If any predicted price ≥ last actual close + €1.00 → **BUY signal**
3. Signal saved to CSV with confidence score

### V2
1. Predict 10 days of close prices
2. **Check IVol exit gate first:**
   - If CAC 3M IVol > 80th percentile (126-day rolling) for ≥ 2 consecutive days → **signal suppressed**
3. If gate clear and any predicted price ≥ last close + €1.00 → **BUY signal**

---

## IVol Exit Gate (V2 only)

Prevents entering long positions during volatility spikes.

| Parameter | Value |
|-----------|-------|
| Rolling window | 126 business days (~6 months) |
| Percentile threshold | 80th |
| Confirmation days | 2 consecutive days above threshold |

When active, all buy signals are suppressed regardless of price predictions.

---

## Data Sources

### V1
- **LVC prices:** `Data/Archive/LVC_daily.xlsx` (auto-updated via yfinance + web scraper)

### V2
- **LVC prices:** `Data/lvc_ohlcv.csv` (auto-updated via yfinance)
- **CAC OHLCV:** Sfera Postgres `bbgidx.index_prices` (primary) → CSV fallback
- **CAC IVol:** Sfera Postgres `bbgidx.index_implied_vol` (primary) → CSV fallback

---

## Key Takeaway

V1 is a pure price-based model — fast and simple, retrains every run.

V2 adds volatility intelligence: the model sees implied vol regimes through its features, and the exit gate prevents buying into vol spikes. The walk-forward pipeline (`train_models.py`) provides properly out-of-sample weights for backtesting.
