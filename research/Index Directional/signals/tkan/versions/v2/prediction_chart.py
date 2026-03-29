"""
TKAN v2.2 — Prediction Visualization
======================================
Loads pre-trained walk-forward model weights (v2.2 stationary features)
and generates an interactive chart showing 10-day neural network predictions
vs actual LVC prices.

Model predicts log returns → converted back to EUR prices for display.
No trading signal / portfolio / IVol logic — just raw predictions.
"""
import os, json, pickle
import tensorflow as tf
from tkan import TKAN
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Config (must match train_models.py) ──────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR         = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR         = os.path.join(ROOT_DIR, "Data")
WEIGHTS_DIR      = os.path.join(SCRIPT_DIR, "weights")
OUTPUT_DIR       = os.path.join(SCRIPT_DIR, "Output")
LVC_CSV          = os.path.join(DATA_DIR, "lvc_ohlcv.csv")
CAC_IVOL         = os.path.join(DATA_DIR, "cac_ivol.csv")
CAC_OHLCV        = os.path.join(DATA_DIR, "cac_ohlcv.csv")
WINDOW_SIZE      = 10
BACKTEST_START   = '2015-01-01'
RETRAIN_FREQ     = 126
IVOL_EXIT_WINDOW = 126
DROPOUT_RATE     = 0.2
PRED_STEP = 5  # slider step resolution in trading days

# ── Feature list (must match train_models.py) ────────────────────────────────
FEATURE_COLS = [
    'log_return_1d', 'high_low_range', 'close_to_high',
    'volume_ratio', 'close_vs_sma15',
    'ivol_zscore', 'ivol_ema_ratio', 'ivol_pctl', 'ivol_roc5',
    'rvol_park20_zscore', 'vol_spread',
    'return_5d', 'return_20d',
]


# ── Data & stationary features (identical to train_models.py) ────────────────
def load_and_build_features():
    lvc = pd.read_csv(LVC_CSV, parse_dates=['date'], index_col='date').sort_index()
    cac_ohlcv = pd.read_csv(CAC_OHLCV, parse_dates=['date'], index_col='date').sort_index()
    ivol_raw = pd.read_csv(CAC_IVOL, parse_dates=['date'], index_col='date').sort_index()

    common = lvc.index.intersection(ivol_raw.index).intersection(cac_ohlcv.index)
    df = pd.DataFrame({
        'open':      lvc.loc[common, 'open'],
        'high':      lvc.loc[common, 'high'],
        'low':       lvc.loc[common, 'low'],
        'close':     lvc.loc[common, 'close'],
        'volume':    lvc.loc[common, 'volume'],
        'ivol':      ivol_raw.loc[common, 'ivol'],
        'cac_high':  cac_ohlcv.loc[common, 'high'],
        'cac_low':   cac_ohlcv.loc[common, 'low'],
        'cac_close': cac_ohlcv.loc[common, 'close'],
    }).sort_index()

    # Stationary features
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = np.log(df['high'] / df['low'])
    df['close_to_high'] = np.log(df['close'] / df['high'])
    vol_ma20 = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_ma20
    sma15 = df['close'].rolling(15, min_periods=1).mean()
    df['close_vs_sma15'] = np.log(df['close'] / sma15)
    ivol_mean63 = df['ivol'].rolling(63).mean()
    ivol_std63 = df['ivol'].rolling(63).std()
    df['ivol_zscore'] = (df['ivol'] - ivol_mean63) / ivol_std63
    ivol_ema20 = df['ivol'].ewm(span=20).mean()
    df['ivol_ema_ratio'] = df['ivol'] / ivol_ema20
    df['ivol_pctl'] = df['ivol'].rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True)
    df['ivol_roc5'] = df['ivol'].pct_change(5)
    log_hl = np.log(df['cac_high'] / df['cac_low'])
    park_factor = 1.0 / (4.0 * np.log(2))
    rvol_park20 = np.sqrt(park_factor * (log_hl ** 2).rolling(20).mean() * 252) * 100
    rvol_mean63 = rvol_park20.rolling(63).mean()
    rvol_std63 = rvol_park20.rolling(63).std()
    df['rvol_park20_zscore'] = (rvol_park20 - rvol_mean63) / rvol_std63
    df['vol_spread'] = df['ivol'] - rvol_park20
    df['return_5d'] = np.log(df['close'] / df['close'].shift(5))
    df['return_20d'] = np.log(df['close'] / df['close'].shift(20))

    # Lag features by 1 day
    lagged = {}
    for col in FEATURE_COLS:
        lagged[col] = df[col].shift(1)
    X = pd.DataFrame(lagged, index=df.index)

    y = df['close']
    df.dropna(inplace=True)
    valid = df.index
    X = X.loc[valid]
    y = y.loc[valid]
    return X, y, df


# ── Model helpers ────────────────────────────────────────────────────────────
def build_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, n_features)),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def load_cycle(cycle_key, n_features):
    model = build_model(n_features)
    model.load_weights(os.path.join(WEIGHTS_DIR, f"{cycle_key}.weights.h5"))
    with open(os.path.join(WEIGHTS_DIR, f"{cycle_key}_scaler_X.pkl"), 'rb') as f:
        scaler_X = pickle.load(f)
    return model, scaler_X


def predict_window(model, scaler_X, X_df, idx, anchor_price):
    """Predict 10-day log returns from idx, convert to EUR prices."""
    X_window = X_df.iloc[idx - WINDOW_SIZE:idx]
    X_sc = scaler_X.transform(X_window)
    X_sc = np.clip(X_sc, -5, 5).reshape(1, WINDOW_SIZE, -1)
    log_returns = model.predict(X_sc, verbose=0).flatten()
    return anchor_price * np.exp(log_returns)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import plotly.graph_objects as go

    manifest_path = os.path.join(WEIGHTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("No trained weights. Run train_models.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    cycles = sorted(manifest['cycles'].items(), key=lambda x: x[1]['cycle_num'])
    print(f"Loaded {len(cycles)} trained cycles")

    X, y, df = load_and_build_features()
    dates = X.index
    n_features = X.shape[1]
    bt_start_idx = dates.searchsorted(pd.Timestamp(BACKTEST_START))
    total = len(dates) - bt_start_idx
    print(f"Backtest: {dates[bt_start_idx].date()} -> {dates[-1].date()} ({total} days)")

    cycle_boundaries = [(info['train_end_idx'], key) for key, info in cycles]

    # ── Generate predictions for every day ───────────────────────────────────
    model, scaler_X = None, None
    cycle_idx = 0
    current_cycle_key = None
    retrain_dates = []

    # Store: index -> (preds_array, cycle_key_used)
    pred_data = {}

    for count, i in enumerate(range(bt_start_idx, len(dates))):
        if cycle_idx < len(cycle_boundaries) and i >= cycle_boundaries[cycle_idx][0]:
            new_key = cycle_boundaries[cycle_idx][1]
            if new_key != current_cycle_key:
                model, scaler_X = load_cycle(new_key, n_features)
                current_cycle_key = new_key
                retrain_dates.append(dates[i])
                print(f"  Loaded {new_key}")
            cycle_idx += 1

        if model is not None and i >= WINDOW_SIZE:
            anchor_price = float(y.iloc[i])
            preds = predict_window(model, scaler_X, X, i, anchor_price)
            pred_data[i] = (preds, current_cycle_key)

        if (count + 1) % 500 == 0:
            print(f"  Progress: {count+1}/{total}")

    print(f"Generated {len(pred_data)} prediction windows")

    # ── Prepare slider data ──────────────────────────────────────────────────
    pred_indices = sorted(pred_data.keys())
    # Subsample for slider steps
    slider_indices = pred_indices[::PRED_STEP]
    if slider_indices[-1] != pred_indices[-1]:
        slider_indices.append(pred_indices[-1])

    print(f"Slider steps: {len(slider_indices)}")

    # ── Build chart ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Trace 0: Actual close price (always visible)
    bt_dates  = dates[bt_start_idx:]
    bt_closes = y.iloc[bt_start_idx:].values
    fig.add_trace(go.Scatter(
        x=bt_dates, y=bt_closes, mode='lines',
        name='LVC Actual Close',
        line=dict(color='royalblue', width=2),
    ))

    # Trace 1: Actual close over the 10-day forecast window (for comparison)
    # Will be restyled by slider
    first_idx = slider_indices[0]
    first_actual_x, first_actual_y = _build_actual_segment(first_idx, dates, y)
    fig.add_trace(go.Scatter(
        x=first_actual_x, y=first_actual_y, mode='lines+markers',
        name='Actual (forecast window)',
        line=dict(color='royalblue', width=2.5, dash='dash'),
        marker=dict(size=5, color='royalblue'),
    ))

    # Trace 2: 10-day forecast curve (restyled by slider)
    first_preds, first_cycle = pred_data[first_idx]
    first_fx, first_fy, first_hover = _build_forecast(first_idx, first_preds, dates, y)
    fig.add_trace(go.Scatter(
        x=first_fx, y=first_fy, mode='lines+markers',
        name='10-Day NN Forecast',
        line=dict(color='crimson', width=2.5, dash='dash'),
        marker=dict(size=7, symbol='diamond', color='crimson'),
        text=first_hover, hoverinfo='text',
    ))

    # Trace 3: Anchor marker (current day)
    fig.add_trace(go.Scatter(
        x=[dates[first_idx]], y=[float(y.iloc[first_idx])],
        mode='markers', name='Forecast Origin',
        marker=dict(size=12, color='green', symbol='star'),
        hoverinfo='text',
        text=[f"Origin: {dates[first_idx].strftime('%Y-%m-%d')}<br>Close: €{y.iloc[first_idx]:.2f}"],
    ))

    # Trace 4: Retrain date markers on the price line
    retrain_y = [float(y.loc[rd]) if rd in y.index else np.nan for rd in retrain_dates]
    fig.add_trace(go.Scatter(
        x=retrain_dates, y=retrain_y,
        mode='markers', name='Model Retrain',
        marker=dict(size=14, color='orange', symbol='x-thin-open', line=dict(width=3)),
        hoverinfo='text',
        text=[f"RETRAIN: {rd.strftime('%Y-%m-%d')}" for rd in retrain_dates],
    ))

    # Retrain vertical lines
    for rd in retrain_dates:
        fig.add_vline(x=rd, line_dash='dot', line_color='orange', opacity=0.3)

    # ── Build slider steps ───────────────────────────────────────────────────
    steps = []
    for idx in slider_indices:
        preds, cycle_key = pred_data[idx]
        date_str = dates[idx].strftime('%Y-%m-%d')

        fx, fy, fhover = _build_forecast(idx, preds, dates, y)
        ax, ay = _build_actual_segment(idx, dates, y)

        # Compute error for this forecast
        errors = []
        for d in range(len(preds)):
            fi = idx + d + 1
            if fi < len(dates):
                errors.append(abs(float(y.iloc[fi]) - float(preds[d])))
        mae = np.mean(errors) if errors else 0

        weights_file = f"{cycle_key}.weights.h5"

        step = dict(
            method='update',
            label=date_str,
            args=[
                # Restyle traces 1,2,3
                {
                    'x': [ax, fx, [str(dates[idx])]],
                    'y': [ay, fy, [float(y.iloc[idx])]],
                    'text': [None, fhover,
                             [f"Origin: {date_str}<br>Close: €{y.iloc[idx]:.2f}<br>Model: {cycle_key}"]],
                },
                # Relayout: update title + annotation
                {
                    'title.text': (
                        f'TKAN v2.2 — 10-Day Forecast from <b>{date_str}</b>  |  '
                        f'10d MAE: €{mae:.2f}'
                    ),
                    'annotations[0].text': (
                        f'Model weights: <b>{weights_file}</b>  |  '
                        f'Scaler: {cycle_key}_scaler_X.pkl'
                    ),
                },
                [1, 2, 3],  # trace indices to update
            ],
        )
        steps.append(step)

    # Initial title
    first_preds_arr, first_ck = pred_data[first_idx]
    first_errors = [abs(float(y.iloc[first_idx+d+1]) - float(first_preds_arr[d]))
                    for d in range(10) if first_idx+d+1 < len(dates)]
    first_mae = np.mean(first_errors) if first_errors else 0

    first_weights = f"{first_ck}.weights.h5"

    fig.update_layout(
        title=(f'TKAN v2.2 — 10-Day Forecast from <b>{dates[first_idx].strftime("%Y-%m-%d")}</b>  |  '
               f'10d MAE: €{first_mae:.2f}'),
        height=750, width=1500,
        template='plotly_white',
        hovermode='x unified',
        yaxis_title='LVC Price (EUR)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(b=120),
        annotations=[
            dict(
                text=(
                    f'Model weights: <b>{first_weights}</b>  |  '
                    f'Scaler: {first_ck}_scaler_X.pkl'
                ),
                xref='paper', yref='paper',
                x=0.5, y=-0.13,
                showarrow=False,
                font=dict(size=12, color='gray'),
                align='center',
            ),
        ],
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='Forecast from: ', font=dict(size=14)),
            pad=dict(t=50),
            steps=steps,
        )],
    )

    output_path = os.path.join(OUTPUT_DIR, "prediction_chart.html")
    fig.write_html(output_path)
    print(f"\nChart saved to: {output_path}")


def _build_forecast(idx, preds, dates, y):
    """Build x/y/hover for the 10-day forecast trace, anchored at actual close."""
    fx = [str(dates[idx])]
    fy = [float(y.iloc[idx])]
    hover = [f"Day 0 (actual): €{y.iloc[idx]:.2f}"]
    for d in range(len(preds)):
        fi = idx + d + 1
        if fi < len(dates):
            actual = float(y.iloc[fi])
            pred = float(preds[d])
            err = pred - actual
            fx.append(str(dates[fi]))
            fy.append(pred)
            hover.append(
                f"Day {d+1}: €{pred:.2f}<br>"
                f"Actual: €{actual:.2f}<br>"
                f"Error: {err:+.2f}"
            )
    return fx, fy, hover


def _build_actual_segment(idx, dates, y):
    """Build x/y for actual close over the 10-day window from idx."""
    ax = []
    ay = []
    for d in range(11):
        fi = idx + d
        if fi < len(dates):
            ax.append(str(dates[fi]))
            ay.append(float(y.iloc[fi]))
    return ax, ay


if __name__ == "__main__":
    main()
