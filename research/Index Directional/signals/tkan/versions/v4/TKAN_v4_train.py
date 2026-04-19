"""TKAN v4 — Walk-Forward Training (10-Day Price Path)
=====================================================
Target   : [close_lev_{t+k}/close_lev_t for k in 1..10] — price ratios on 2× leveraged series
Features : 12 stationary log-based features (all log ratios/z-scores — matches v2 formulas)
Model    : 3 × TKAN(100, dropout=0.2, return_sequences=True) → Dense(1) → (batch, 10, 1)
Loss     : MSE + MAE metric (same as v2)
Key fix  : LEVERAGE_FACTOR=2 → target std ~2× larger → MSE gradients ~4× stronger
           Plain CACT (~0.8% daily) collapses model; 2× series (~1.6%) gives real signal

Entry logic (in dsl_strategy.py)
  Enter if max(pred[d1..d10]) >= 1.015   (+1.5% threshold)
  i.e. "the model expects price to hit +1.5% at least once in the next 10 days"
  This mirrors v2's spirit (any predicted day hits target) but uses a % threshold
  instead of a fixed EUR amount — correct for a large index (CACT ~8,000–15,000).

Output
------
  weights/cycle_NNN.weights.h5       — per-cycle model weights
  weights/cycle_NNN_scaler_X.pkl     — per-cycle RobustScaler (X only)
  weights/manifest.json              — cycle metadata + train times
  weights/pred_cache.pkl             — pd.DataFrame (DatetimeIndex × [d1..d10] ratios)

Signal threshold in dsl_strategy.py
-------------------------------------
  TKAN_VERSION = "v4"
  TKAN_THRESH  = 1.015   # max of 10-day path >= 1.5% gain

Run (ml conda env — tensorflow + tkan required)
---------------------
  cd "c:\Personal\Business & Investments\Python codes\btest"
  $env:PYTHONUTF8 = '1' ; $env:TF_ENABLE_ONEDNN_OPTS = '0'
  conda run -n ml python "research\Index Directional\signals\tkan\versions\v4\TKAN_v4_train.py"
"""
import os, sys, json, pickle, time, hashlib, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tkan import TKAN
from sklearn.preprocessing import RobustScaler

# ── sfera_db path ─────────────────────────────────────────────────────────────
# 7 levels up from this file = "Python codes" root
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_HERE, *(['..'] * 7)))  # Python codes/
_SFERA_DB = os.path.join(_ROOT, 'sfera-db')
if os.path.isdir(_SFERA_DB) and _SFERA_DB not in sys.path:
    sys.path.insert(0, _SFERA_DB)
import sfera_db

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_DIR     = os.path.join(_HERE, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

WINDOW_SIZE      = 10    # context bars fed to TKAN (matches v2 — equal to prediction horizon)
PREDICTION_DAYS  = 10    # predict price ratios for t+1 .. t+10
ENTRY_THRESHOLD  = 1.015 # entry if max(pred[d1..d10]) >= this (+1.5%)
EPOCHS           = 80    # no EarlyStopping — run full 80 epochs per cycle
BATCH_SIZE       = 32
RETRAIN_FREQ     = 252   # trading days between retrains (~1 year)
BACKTEST_START   = '2015-01-01'
IVOL_WINDOW      = 63    # matches v2 (was 126)
DROPOUT_RATE     = 0.2
LEVERAGE_FACTOR  = 2.0   # 2× leveraged synthetic series — 4× stronger MSE signal than plain CACT
VERSION_TAG      = 'v4_logfeat_lev2x'

FEATURE_COLS = [
    # 12 stationary features — same philosophy as v2 (no raw price levels)
    'log_return_1d', 'high_low_range', 'close_to_high',
    'close_vs_sma15',
    'ivol_zscore', 'ivol_ema_ratio', 'ivol_pctl', 'ivol_roc5',
    'rvol_park20_zscore', 'vol_spread',
    'return_5d', 'return_20d',
]  # 12 stationary features

TARGET_COLS = [f'd{k}' for k in range(1, PREDICTION_DAYS + 1)]  # d1..d10


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG FINGERPRINT — cached cycles invalidated if any param changes
# ══════════════════════════════════════════════════════════════════════════════
def _config_hash():
    return hashlib.sha256(json.dumps({
        'version':        VERSION_TAG,
        'WINDOW_SIZE':    WINDOW_SIZE,
        'PREDICTION_DAYS': PREDICTION_DAYS,
        'EPOCHS':         EPOCHS,
        'BATCH_SIZE':     BATCH_SIZE,
        'BACKTEST_START': BACKTEST_START,
        'RETRAIN_FREQ':   RETRAIN_FREQ,
        'IVOL_WINDOW':    IVOL_WINDOW,
        'DROPOUT_RATE':   DROPOUT_RATE,
        'LEVERAGE_FACTOR': LEVERAGE_FACTOR,
        'features':       FEATURE_COLS,
        'model':          '3xTKAN100_all_return_seq_Dense1_12feat_ratio',
        'target':         'price_ratio_lev2x_logfeat',
        'entry_threshold': ENTRY_THRESHOLD,
    }, sort_keys=True).encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    def _q(sql):
        return (sfera_db.query(sql)
                .assign(date=lambda d: pd.to_datetime(d['date']))
                .set_index('date'))

    cactr    = _q("SELECT trade_date AS date, close_price AS close "
                  "FROM bbgidx.index_total_return WHERE ticker='CACT' ORDER BY trade_date")
    cac_ohlc = _q("SELECT trade_date AS date, open_price AS open, "
                  "high_price AS high, low_price AS low, close_price AS cac_close "
                  "FROM bbgidx.index_prices WHERE ticker='CAC' ORDER BY trade_date")
    ivol_raw = _q("SELECT trade_date AS date, \"3m_50d_ivol\" AS ivol "
                  "FROM bbgidx.index_implied_vol WHERE ticker='CAC' ORDER BY trade_date")[['ivol']]

    print(f"  CACT TR:  {cactr.index[0].date()} -> {cactr.index[-1].date()} ({len(cactr)} rows)")
    print(f"  CAC OHLC: {cac_ohlc.index[0].date()} -> {cac_ohlc.index[-1].date()} ({len(cac_ohlc)} rows)")
    print(f"  IVol:     {ivol_raw.index[0].date()} -> {ivol_raw.index[-1].date()} ({len(ivol_raw)} rows)")
    return cactr, cac_ohlc, ivol_raw


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (identical to v3 — verified stationary)
# ══════════════════════════════════════════════════════════════════════════════
def build_features(cactr, cac_ohlc, ivol_raw):
    # Forward-fill ivol over any gaps
    ivol_filled = ivol_raw.reindex(ivol_raw.index.union(cactr.index)).ffill()
    common = cactr.index.intersection(cac_ohlc.index).intersection(ivol_filled.index)

    df = cac_ohlc.loc[common].copy()
    df['close'] = cactr.loc[common, 'close']
    df['ivol']  = ivol_filled.loc[common, 'ivol']

    # ── Synthetic leveraged close (LEVERAGE_FACTOR=2 → 2× ETF daily moves) ──
    log_ret_raw   = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['close_lev'] = df['close'].iloc[0] * np.exp((LEVERAGE_FACTOR * log_ret_raw).cumsum())

    # ── Log-based features — all scale-invariant, matching v2 formulas ────
    # Returns on leveraged series (actual trading moves at 2× exposure)
    df['log_return_1d'] = np.log(df['close_lev'] / df['close_lev'].shift(1))
    df['return_5d']     = np.log(df['close_lev'] / df['close_lev'].shift(5))
    df['return_20d']    = np.log(df['close_lev'] / df['close_lev'].shift(20))

    # Range / position: log ratios from CAC OHLC, scaled by LEVERAGE_FACTOR
    df['high_low_range'] = LEVERAGE_FACTOR * np.log(df['high'] / df['low'].replace(0, np.nan))
    df['close_to_high']  = LEVERAGE_FACTOR * np.log(df['cac_close'] / df['high'].replace(0, np.nan))

    # SMA deviation: log ratio (matches v2: log(close/sma15))
    sma15                = df['close_lev'].rolling(15, min_periods=1).mean()
    df['close_vs_sma15'] = np.log(df['close_lev'] / sma15)

    # Parkinson realised vol — annualised in % units (× 100 matches v2)
    log_hl        = np.log(df['high'] / df['low'].replace(0, np.nan))
    rvol_park20   = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(20).mean() * 252) * 100
    df['rvol_park20'] = rvol_park20

    # Implied vol features
    ivol   = df['ivol']
    ewma20 = ivol.ewm(span=20).mean()
    df['ivol_zscore']        = (ivol - ivol.rolling(IVOL_WINDOW).mean()) / (ivol.rolling(IVOL_WINDOW).std() + 1e-9)
    df['ivol_ema_ratio']     = ivol / (ewma20 + 1e-9)
    df['ivol_pctl']          = ivol.rolling(IVOL_WINDOW, min_periods=20).rank(pct=True)  # matches v2
    df['ivol_roc5']          = ivol.pct_change(5)
    df['rvol_park20_zscore'] = (rvol_park20 - rvol_park20.rolling(IVOL_WINDOW).mean()) / \
                                (rvol_park20.rolling(IVOL_WINDOW).std() + 1e-9)
    df['vol_spread']         = ivol - rvol_park20   # both in % units

    # ── Lag all features by 1 day (no same-day lookahead) ─────────────────
    for col in FEATURE_COLS:
        df[col] = df[col].shift(1)

    df = df.dropna(subset=FEATURE_COLS + ['close', 'close_lev'])
    print(f"  Features built: {len(df):,} rows  "
          f"{df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  CACT daily std: {log_ret_raw.std()*100:.3f}%  →  "
          f"{LEVERAGE_FACTOR}× leveraged: {log_ret_raw.std()*LEVERAGE_FACTOR*100:.3f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, len(FEATURE_COLS))),
        TKAN(100, return_sequences=True,  use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        TKAN(100, return_sequences=True,  use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        TKAN(100, return_sequences=True,  use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(1),  # sequential: (batch,window,1) — price ratios (same as v2)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(X_arr, y_close_arr):
    """Exact v2 approach: anchor = last close in input window.
    Target: fwd_close[t+k] / anchor  (k=1..10), values ~1.0, no y-scaler needed."""
    Xs, ys = [], []
    for i in range(len(X_arr) - 2 * WINDOW_SIZE):
        Xs.append(X_arr[i : i + WINDOW_SIZE])
        anchor     = y_close_arr[i + WINDOW_SIZE - 1]
        fwd_prices = y_close_arr[i + WINDOW_SIZE : i + 2 * WINDOW_SIZE]
        ys.append(fwd_prices / anchor)   # shape (10,), values ~1.0
    return np.array(Xs, dtype='float32'), np.array(ys, dtype='float32')


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_walk_forward(df):
    config_hash   = _config_hash()
    manifest_path = os.path.join(WEIGHTS_DIR, 'manifest.json')

    manifest = {'config_hash': config_hash, 'cycles': {}}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            existing = json.load(f)
        if existing.get('config_hash') == config_hash:
            manifest = existing
            print(f"  Resuming from existing manifest ({len(manifest['cycles'])} cached cycles)")
        else:
            print(f"  Config changed — ignoring cached weights (old hash: {existing.get('config_hash')})")

    all_dates = df.index
    bt_start  = pd.Timestamp(BACKTEST_START)

    # Build retrain trigger dates
    retrain_dates = []
    i = all_dates.searchsorted(bt_start)
    while i < len(all_dates):
        retrain_dates.append(all_dates[i])
        i += RETRAIN_FREQ

    print(f"\n  Walk-forward: {len(retrain_dates)} cycles, "
          f"retrain every {RETRAIN_FREQ} trading days")

    all_preds = {}   # date → predicted ratio (OOS only)

    for cycle_num, retrain_date in enumerate(retrain_dates):
        cycle_key = f'cycle_{cycle_num:03d}'
        w_path    = os.path.join(WEIGHTS_DIR, f'{cycle_key}.weights.h5')
        s_path    = os.path.join(WEIGHTS_DIR, f'{cycle_key}_scaler_X.pkl')

        # OOS window: from retrain_date to next retrain_date (or end of data)
        if cycle_num + 1 < len(retrain_dates):
            oos_end_idx = all_dates.searchsorted(retrain_dates[cycle_num + 1])
        else:
            oos_end_idx = len(all_dates)
        oos_idx_start = all_dates.searchsorted(retrain_date)
        oos_dates     = all_dates[oos_idx_start : oos_end_idx]

        # Training data: all rows strictly before retrain_date
        train_df = df[df.index < retrain_date]
        X_train  = train_df[FEATURE_COLS].values
        y_close  = train_df['close_lev'].values    # leveraged close → 4× stronger MSE signal

        print(f"\n[{cycle_key}]  retrain={retrain_date.date()}  "
              f"train={len(train_df):,}  oos={len(oos_dates)}")

        # ── Load cached or retrain ────────────────────────────────────────────
        if cycle_key in manifest['cycles'] and os.path.isfile(w_path) and os.path.isfile(s_path):
            print(f"  cached — skipping training")
            model = build_model()
            model.load_weights(w_path)
            with open(s_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            if len(train_df) < WINDOW_SIZE + 100:
                print(f"  skip — not enough training data ({len(train_df)} rows)")
                continue

            scaler = RobustScaler()
            X_sc   = scaler.fit_transform(X_train)
            Xs, ys = make_sequences(X_sc, y_close)

            model  = build_model()
            t0     = time.time()
            hist   = model.fit(
                Xs, ys,
                epochs          = EPOCHS,
                batch_size      = BATCH_SIZE,
                validation_split= 0.1,
                verbose         = 0,
            )
            elapsed  = time.time() - t0
            val_loss = min(hist.history.get('val_loss', [float('nan')]))

            model.save_weights(w_path)
            with open(s_path, 'wb') as f:
                pickle.dump(scaler, f)

            manifest['cycles'][cycle_key] = {
                'cycle_num':    cycle_num,
                'retrain_date': str(retrain_date.date()),
                'train_rows':   len(train_df),
                'oos_days':     len(oos_dates),
                'val_loss':     round(val_loss, 8),
                'train_time_s': round(elapsed, 1),
            }
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            print(f"  trained {elapsed:.0f}s  val_loss={val_loss:.6f}")

        # ── OOS predictions ───────────────────────────────────────────────────
        # Scale ALL features with THIS cycle's scaler (no leakage — scaler
        # was fit only on training data)
        full_X  = df[FEATURE_COLS].values
        full_sc = scaler.transform(full_X)  # no clipping — v2 used no clipping

        for date in oos_dates:
            d_idx = all_dates.get_loc(date)
            if d_idx < WINDOW_SIZE:
                continue
            # Use 10 days BEFORE today (d_idx-10 .. d_idx-1). Features are already lagged
            # by 1 in build_features, so this window uses only yesterday-and-prior data.
            # anchor = close[d_idx-1]; pred[k] ≈ close[d_idx+k] / close[d_idx-1]
            # d1 = today's close vs yesterday, d2 = tomorrow, ..., d10 = today+9
            X_win = full_sc[d_idx - WINDOW_SIZE : d_idx].reshape(1, WINDOW_SIZE, -1).astype('float32')
            pred  = model(X_win, training=False).numpy()[0, :, 0]  # (window,1) → (10,) ratios
            all_preds[date] = pred

    # ── Stitch and save pred_cache ────────────────────────────────────────────
    pred_df = pd.DataFrame.from_dict(all_preds, orient='index', columns=TARGET_COLS)
    pred_df.index = pd.DatetimeIndex(pred_df.index)
    pred_df = pred_df.sort_index()

    cache_path = os.path.join(WEIGHTS_DIR, 'pred_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(pred_df, f)

    max_ratio  = pred_df.max(axis=1)
    pct_signal = (max_ratio >= ENTRY_THRESHOLD).mean()
    print(f"\n  pred_cache saved: {len(pred_df):,} rows × {len(TARGET_COLS)} cols → {cache_path}")
    print(f"  d1  ratio mean={pred_df['d1'].mean():.5f}  std={pred_df['d1'].std():.5f}")
    print(f"  d10 ratio mean={pred_df['d10'].mean():.5f}  std={pred_df['d10'].std():.5f}")
    print(f"  d1  pct   mean={(pred_df['d1'].mean()-1)*100:.3f}%  std={pred_df['d1'].std()*100:.3f}%")
    print(f"  signal ON (max ratio >= {ENTRY_THRESHOLD}): {pct_signal:.1%} of days")
    return pred_df


# ══════════════════════════════════════════════════════════════════════════════
# STATE-MACHINE SIGNAL (2× leveraged series)
# ══════════════════════════════════════════════════════════════════════════════
def build_signal(pred_df, close_lev,
                 entry_threshold=ENTRY_THRESHOLD):
    """Convert raw TKAN predictions into a 0/1 position series.

    Rules (all on the 2× leveraged CACT close):
      Entry  : max(d1..d10) >= entry_threshold AND flat → signal = 1,
               anchor = close_lev[T]
      Hold   : signal stays 1 until close_lev >= anchor * entry_threshold
               (no time-stop — hold as long as target not reached)
      Exit   : first bar where close_lev >= anchor * entry_threshold
      Re-entry: on the exit bar itself, if a new prediction fires → enter again
    """
    common = pred_df.index.intersection(close_lev.index)
    pred   = pred_df.loc[common]
    clev   = close_lev.loc[common]

    sig        = pd.Series(0, index=common, dtype=int, name='signal')
    in_trade   = False
    anchor_lev = None

    for i in range(len(common)):
        fires = pred.iloc[i].max() >= entry_threshold

        if in_trade:
            target_hit = clev.iloc[i] >= anchor_lev * entry_threshold

            if target_hit:
                in_trade   = False
                anchor_lev = None
                # immediate re-entry on same bar if new prediction fires
                if fires:
                    sig.iloc[i] = 1
                    in_trade    = True
                    anchor_lev  = clev.iloc[i]
                # else sig.iloc[i] stays 0
            else:
                sig.iloc[i] = 1   # still holding — target not yet reached
        else:
            if fires:
                sig.iloc[i] = 1
                in_trade    = True
                anchor_lev  = clev.iloc[i]

    in_mkt = sig.mean()
    exits  = ((sig.shift(1, fill_value=0) == 1) & (sig == 0)).sum()
    print(f"\n  Signal stats:  in-market={in_mkt:.1%}  "
          f"trades={exits}  "
          f"({common[0].date()} → {common[-1].date()})")
    return sig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("TKAN v4 — 10-Day Price Path Walk-Forward Training")
    print("=" * 60)
    print("\nLoading data from Sfera DB...")
    cactr, cac_ohlc, ivol_raw = load_data()
    df = build_features(cactr, cac_ohlc, ivol_raw)
    pred_df = train_walk_forward(df)

    # ── Build proper state-machine signal and save alongside pred_cache ───
    print("\nBuilding state-machine signal on 2× leveraged series...")
    sig = build_signal(pred_df, df['close_lev'])
    sig_path = os.path.join(WEIGHTS_DIR, 'signal_cache.pkl')
    with open(sig_path, 'wb') as f:
        pickle.dump(sig, f)
    print(f"  signal_cache saved → {sig_path}")

    print("\nDone.")
