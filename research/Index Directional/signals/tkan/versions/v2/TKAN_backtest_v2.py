"""
TKAN v2 Walk-Forward Backtest with IVol Gate — Interactive
==========================================================
Loads pre-trained model weights (from train_models.py), runs inference,
logs ALL daily predictions, and produces an interactive Plotly HTML chart.

Run train_models.py first to generate the weights.
"""
import os, json, pickle
import tensorflow as tf
from tkan import TKAN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (must match train_models.py)
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR    = os.path.join(ROOT_DIR, "Data")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "Output")

LVC_CSV   = os.path.join(DATA_DIR, "lvc_ohlcv.csv")
CAC_IVOL  = os.path.join(DATA_DIR, "cac_ivol.csv")
CAC_OHLCV = os.path.join(DATA_DIR, "cac_ohlcv.csv")

WINDOW_SIZE     = 10
PREDICTION_DAYS = 10
THRESHOLD       = 1.0
EPOCHS          = 50
BATCH_SIZE      = 32
BACKTEST_START  = '2015-01-01'
RETRAIN_FREQ    = 126

IVOL_EXIT_WINDOW  = 126
IVOL_EXIT_PCTL    = 80
IVOL_EXIT_CONFIRM = 2

INITIAL_CAPITAL = 100_000


# ══════════════════════════════════════════════════════════════════════════════
# DATA & FEATURES
# ══════════════════════════════════════════════════════════════════════════════
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

    df['sma15']      = df['close'].rolling(15, min_periods=1).mean()
    df['ivol_ema20'] = df['ivol'].ewm(span=20).mean()
    df['ivol_pctl']  = df['ivol'].rolling(IVOL_EXIT_WINDOW, min_periods=20).rank(pct=True)
    df['ivol_roc5']  = df['ivol'].pct_change(5)

    log_hl = np.log(df['cac_high'] / df['cac_low'])
    park_factor = 1.0 / (4.0 * np.log(2))
    df['rvol_park20'] = np.sqrt(park_factor * (log_hl ** 2).rolling(20).mean() * 252) * 100
    df['vol_spread']  = df['ivol'] - df['rvol_park20']

    feature_cols = ['close', 'high', 'low', 'volume', 'sma15',
                    'ivol', 'ivol_ema20', 'ivol_pctl', 'ivol_roc5',
                    'rvol_park20', 'vol_spread']
    for col in feature_cols:
        df[f'prior_{col}'] = df[col].shift(1)
    df.dropna(inplace=True)

    prior_cols = [f'prior_{c}' for c in feature_cols]
    X = df[prior_cols]
    y = df['close']
    return X, y, df, ivol_raw


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def build_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(WINDOW_SIZE, n_features)),
        TKAN(100, return_sequences=True, use_bias=True),
        TKAN(100, return_sequences=True, use_bias=True),
        TKAN(100, return_sequences=True, use_bias=True),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def load_cycle(cycle_key, n_features):
    """Load cached weights + scalers for a cycle."""
    weights_path = os.path.join(WEIGHTS_DIR, f"{cycle_key}.weights.h5")
    scaler_x_path = os.path.join(WEIGHTS_DIR, f"{cycle_key}_scaler_X.pkl")
    scaler_y_path = os.path.join(WEIGHTS_DIR, f"{cycle_key}_scaler_y.pkl")

    model = build_model(n_features)
    model.load_weights(weights_path)

    with open(scaler_x_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y


def predict_window(model, scaler_X, scaler_y, X_df, idx):
    """Predict next WINDOW_SIZE days from position idx. Returns array of prices."""
    X_window = X_df.iloc[idx - WINDOW_SIZE:idx]
    X_sc = scaler_X.transform(X_window).reshape(1, WINDOW_SIZE, -1)
    pred_sc = model.predict(X_sc, verbose=0).flatten()
    return scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# IVOL GATE
# ══════════════════════════════════════════════════════════════════════════════
def is_ivol_blocked(ivol_series, as_of_date):
    iv = ivol_series.loc[:as_of_date]
    if len(iv) < 20:
        return False, 0, np.nan

    latest_iv = iv.iloc[-1]
    thr = iv.rolling(IVOL_EXIT_WINDOW, min_periods=20).quantile(IVOL_EXIT_PCTL / 100)
    above = (iv > thr).iloc[-20:]
    consec = 0
    for v in reversed(above.values):
        if v:
            consec += 1
        else:
            break

    blocked = consec >= IVOL_EXIT_CONFIRM
    return blocked, consec, latest_iv


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST  (inference only — loads cached weights)
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest():
    # Load manifest
    manifest_path = os.path.join(WEIGHTS_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: No trained weights found. Run train_models.py first.")
        return None, None, None, None

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    cycles = sorted(manifest['cycles'].items(), key=lambda x: x[1]['cycle_num'])
    print(f"Loaded manifest: {len(cycles)} trained cycles (hash={manifest['config_hash']})")

    X, y, df, ivol_raw = load_and_build_features()
    dates = X.index
    n_features = X.shape[1]
    bt_start_idx = dates.searchsorted(pd.Timestamp(BACKTEST_START))

    print(f"Data: {dates[0].date()} to {dates[-1].date()} ({len(dates)} rows)")
    print(f"Backtest: {dates[bt_start_idx].date()} to {dates[-1].date()} ({len(dates)-bt_start_idx} days)")
    print("=" * 70)

    # Build cycle lookup: for each retrain point, which cycle to use
    cycle_boundaries = []
    for cycle_key, info in cycles:
        cycle_boundaries.append((info['train_end_idx'], cycle_key))

    # State
    capital = INITIAL_CAPITAL
    shares = 0
    holding = False
    buy_price = 0.0
    planned_sell_idx = None

    trades = []
    daily_log = []
    retrain_dates = []

    current_cycle_key = None
    model, scaler_X, scaler_y = None, None, None
    cycle_idx = 0

    for i in range(bt_start_idx, len(dates)):
        current_date = dates[i]
        current_close = y.iloc[i]

        # ── Switch model at retrain points ──
        need_switch = (model is None)
        if cycle_idx < len(cycle_boundaries) and i >= cycle_boundaries[cycle_idx][0]:
            need_switch = True

        if need_switch and cycle_idx < len(cycle_boundaries):
            new_key = cycle_boundaries[cycle_idx][1]
            if new_key != current_cycle_key:
                print(f"  [{current_date.date()}] Loading {new_key}")
                model, scaler_X, scaler_y = load_cycle(new_key, n_features)
                current_cycle_key = new_key
                retrain_dates.append(current_date)
            cycle_idx += 1

        # ── IVol gate ──
        ivol_blocked, ivol_consec, ivol_val = is_ivol_blocked(ivol_raw['ivol'], current_date)

        # ── Predict (every day, whether holding or not) ──
        preds = None
        if model is not None and i >= WINDOW_SIZE:
            preds = predict_window(model, scaler_X, scaler_y, X, i)

        # ── Portfolio value ──
        port_val = capital + shares * current_close

        # ── Determine action ──
        action = 'wait'
        action_detail = ''

        if holding:
            if ivol_blocked:
                sell_price = current_close
                profit = (sell_price - buy_price) * shares
                capital += shares * sell_price
                trades[-1]['sell_date'] = current_date
                trades[-1]['sell_price'] = sell_price
                trades[-1]['profit'] = profit
                trades[-1]['exit_reason'] = 'ivol_gate'
                action = 'sell'
                action_detail = f"IVol gate forced exit ({ivol_consec}d above {IVOL_EXIT_PCTL}p)"
                shares = 0
                holding = False
                planned_sell_idx = None

            elif planned_sell_idx is not None and i >= planned_sell_idx:
                sell_price = current_close
                profit = (sell_price - buy_price) * shares
                capital += shares * sell_price
                trades[-1]['sell_date'] = current_date
                trades[-1]['sell_price'] = sell_price
                trades[-1]['profit'] = profit
                trades[-1]['exit_reason'] = 'planned'
                action = 'sell'
                action_detail = 'Planned exit (prediction horizon reached)'
                shares = 0
                holding = False
                planned_sell_idx = None

            else:
                action = 'hold'
                days_remaining = (planned_sell_idx - i) if planned_sell_idx else 0
                action_detail = f"Holding (exit in {days_remaining}d)"

        elif preds is not None:
            target = current_close + THRESHOLD
            if ivol_blocked:
                action = 'skip'
                action_detail = f"IVol blocked ({ivol_consec}d above {IVOL_EXIT_PCTL}p)"
            elif np.any(preds >= target):
                days_ahead = int(np.argmax(preds >= target)) + 1
                shares_to_buy = int(capital // current_close)
                if shares_to_buy > 0:
                    buy_price = current_close
                    capital -= shares_to_buy * buy_price
                    shares = shares_to_buy
                    holding = True
                    planned_sell_idx = min(i + days_ahead, len(dates) - 1)
                    action = 'buy'
                    action_detail = (f"Pred day{days_ahead}={preds[days_ahead-1]:.2f} >= "
                                     f"target {target:.2f} (+{THRESHOLD})")

                    trades.append({
                        'buy_date': current_date,
                        'buy_price': buy_price,
                        'sell_date': None,
                        'sell_price': None,
                        'shares': shares_to_buy,
                        'profit': None,
                        'predicted_target': preds[days_ahead - 1],
                        'days_ahead': days_ahead,
                        'exit_reason': None,
                    })
                else:
                    action = 'skip'
                    action_detail = 'Signal but insufficient capital'
            else:
                action = 'skip'
                max_pred = preds.max()
                action_detail = f"No pred >= target {target:.2f} (max={max_pred:.2f})"

        # ── Log every day ──
        day_record = {
            'date': current_date,
            'close': current_close,
            'portfolio_value': port_val,
            'ivol': ivol_val,
            'ivol_blocked': ivol_blocked,
            'ivol_consec': ivol_consec,
            'holding': holding if action != 'buy' else True,
            'action': action,
            'action_detail': action_detail,
            'model_cycle': current_cycle_key,
        }
        for d in range(PREDICTION_DAYS):
            day_record[f'pred_day{d+1}'] = preds[d] if preds is not None and d < len(preds) else np.nan
        daily_log.append(day_record)

    # Close any open position
    if holding:
        final_close = y.iloc[-1]
        profit = (final_close - buy_price) * shares
        capital += shares * final_close
        trades[-1]['sell_date'] = dates[-1]
        trades[-1]['sell_price'] = final_close
        trades[-1]['profit'] = profit
        trades[-1]['exit_reason'] = 'end_of_data'

    daily_df = pd.DataFrame(daily_log)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, daily_df, retrain_dates, y


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════
def print_performance(trades_df, daily_df):
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    if trades_df.empty:
        print("No trades executed.")
        return {}

    n_trades = len(trades_df)
    total_profit = trades_df['profit'].sum()
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]
    win_rate = len(winners) / n_trades * 100
    ivol_exits = (trades_df['exit_reason'] == 'ivol_gate').sum()

    final_value = daily_df['portfolio_value'].iloc[-1]
    roi = (final_value / INITIAL_CAPITAL - 1) * 100

    cummax = daily_df['portfolio_value'].cummax()
    drawdown = (daily_df['portfolio_value'] - cummax) / cummax
    max_dd = drawdown.min() * 100

    daily_ret = daily_df['portfolio_value'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    first_close = daily_df['close'].iloc[0]
    last_close = daily_df['close'].iloc[-1]
    bnh_roi = (last_close / first_close - 1) * 100
    bnh_shares = int(INITIAL_CAPITAL // first_close)
    bnh_final = INITIAL_CAPITAL - bnh_shares * first_close + bnh_shares * last_close

    print(f"Trades: {n_trades} ({len(winners)}W / {len(losers)}L) | Win rate: {win_rate:.1f}%")
    print(f"IVol gate forced exits: {ivol_exits}")
    print(f"Total profit: EUR {total_profit:,.2f}")
    print(f"Avg profit/trade: EUR {total_profit/n_trades:,.2f}")
    print(f"Best trade: EUR {trades_df['profit'].max():,.2f} | Worst: EUR {trades_df['profit'].min():,.2f}")
    print(f"\nPortfolio: EUR {INITIAL_CAPITAL:,.0f} -> EUR {final_value:,.0f} (ROI: {roi:+.1f}%)")
    print(f"Max drawdown: {max_dd:.1f}%")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"\nBuy & hold: EUR {INITIAL_CAPITAL:,.0f} -> EUR {bnh_final:,.0f} (ROI: {bnh_roi:+.1f}%)")
    print(f"Strategy alpha: {roi - bnh_roi:+.1f}%")

    return {'n_trades': n_trades, 'win_rate': win_rate, 'total_profit': total_profit,
            'roi': roi, 'max_drawdown': max_dd, 'sharpe': sharpe, 'bnh_roi': bnh_roi}


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLOTLY CHART
# ══════════════════════════════════════════════════════════════════════════════
def build_interactive_chart(trades_df, daily_df, retrain_dates):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        return

    dates = daily_df['date']
    closes = daily_df['close']

    # ── Build hover text with full predictions + action ──
    hover_texts = []
    for _, row in daily_df.iterrows():
        lines = [
            f"<b>{row['date'].strftime('%Y-%m-%d')}</b>",
            f"Close: {row['close']:.2f}",
            f"Model: {row['model_cycle'] or 'N/A'}",
            f"IVol: {row['ivol']:.1f} ({'BLOCKED' if row['ivol_blocked'] else 'OK'})",
            f"Action: <b>{row['action'].upper()}</b>",
            f"{row['action_detail']}",
            "",
            "Predictions (day 1-10):",
        ]
        for d in range(1, 11):
            p = row.get(f'pred_day{d}', np.nan)
            if pd.notna(p):
                lines.append(f"  d{d}: {p:.2f}")
            else:
                lines.append(f"  d{d}: —")
        hover_texts.append("<br>".join(lines))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("LVC Price + Trades", "Portfolio Value", "CAC Implied Volatility"),
    )

    # ── Panel 1: LVC price ──
    fig.add_trace(go.Scatter(
        x=dates, y=closes, mode='lines',
        name='LVC Close', line=dict(color='royalblue', width=1),
        text=hover_texts, hoverinfo='text',
    ), row=1, col=1)

    # Buy markers
    if not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=trades_df['buy_date'], y=trades_df['buy_price'],
            mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            hoverinfo='text',
            text=[f"BUY {r['buy_date'].strftime('%Y-%m-%d')}<br>"
                  f"Price: {r['buy_price']:.2f}<br>"
                  f"Shares: {r['shares']}<br>"
                  f"Target: {r['predicted_target']:.2f} in {r['days_ahead']}d"
                  for _, r in trades_df.iterrows()],
        ), row=1, col=1)

        # Sell markers
        sells = trades_df.dropna(subset=['sell_date'])
        if not sells.empty:
            sell_colors = ['orange' if r['exit_reason'] == 'ivol_gate' else 'red'
                           for _, r in sells.iterrows()]
            fig.add_trace(go.Scatter(
                x=sells['sell_date'], y=sells['sell_price'],
                mode='markers', name='Sell',
                marker=dict(symbol='triangle-down', size=10, color=sell_colors),
                hoverinfo='text',
                text=[f"SELL {r['sell_date'].strftime('%Y-%m-%d')}<br>"
                      f"Price: {r['sell_price']:.2f}<br>"
                      f"P&L: EUR {r['profit']:+,.2f}<br>"
                      f"Exit: {r['exit_reason']}"
                      for _, r in sells.iterrows()],
            ), row=1, col=1)

    # IVol blocked shading on price chart
    blocked = daily_df[daily_df['ivol_blocked']]
    spans = []
    if not blocked.empty:
        blocked_idx = blocked.index.tolist()
        start = blocked_idx[0]
        for j in range(1, len(blocked_idx)):
            if blocked_idx[j] - blocked_idx[j-1] > 1:
                spans.append((start, blocked_idx[j-1]))
                start = blocked_idx[j]
        spans.append((start, blocked_idx[-1]))

        for s, e in spans:
            fig.add_vrect(
                x0=daily_df.loc[s, 'date'], x1=daily_df.loc[e, 'date'],
                fillcolor='red', opacity=0.08, line_width=0,
                row=1, col=1,
            )

    # Retrain vertical lines
    for rd in retrain_dates:
        fig.add_vline(x=rd, line_dash='dot', line_color='gray',
                      opacity=0.4, row=1, col=1)

    # ── Panel 2: Portfolio value ──
    fig.add_trace(go.Scatter(
        x=dates, y=daily_df['portfolio_value'], mode='lines',
        name='Portfolio', line=dict(color='purple', width=1.2),
    ), row=2, col=1)

    first_close = closes.iloc[0]
    bnh_shares = int(INITIAL_CAPITAL // first_close)
    bnh_cash = INITIAL_CAPITAL - bnh_shares * first_close
    bnh_vals = bnh_cash + bnh_shares * closes
    fig.add_trace(go.Scatter(
        x=dates, y=bnh_vals, mode='lines',
        name='Buy & Hold', line=dict(color='gray', width=0.8, dash='dash'),
    ), row=2, col=1)

    fig.add_hline(y=INITIAL_CAPITAL, line_dash='dot', line_color='gray',
                  opacity=0.5, row=2, col=1)

    # ── Panel 3: IVol ──
    fig.add_trace(go.Scatter(
        x=dates, y=daily_df['ivol'], mode='lines',
        name='CAC IVol', line=dict(color='orange', width=1),
    ), row=3, col=1)

    if spans:
        for s, e in spans:
            fig.add_vrect(
                x0=daily_df.loc[s, 'date'], x1=daily_df.loc[e, 'date'],
                fillcolor='red', opacity=0.12, line_width=0,
                row=3, col=1,
            )

    # ── Layout ──
    fig.update_layout(
        title='TKAN v2 Walk-Forward Backtest (Interactive)',
        height=900, width=1400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
    )
    fig.update_yaxes(title_text='LVC Price (EUR)', row=1, col=1)
    fig.update_yaxes(title_text='Portfolio (EUR)', row=2, col=1)
    fig.update_yaxes(title_text='Implied Vol', row=3, col=1)
    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

    output_path = os.path.join(OUTPUT_DIR, "backtest_v2_interactive.html")
    fig.write_html(output_path)
    print(f"\nInteractive chart saved to: {output_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run_backtest()
    if result[0] is None:
        exit(1)

    trades_df, daily_df, retrain_dates, y_full = result
    perf = print_performance(trades_df, daily_df)
    build_interactive_chart(trades_df, daily_df, retrain_dates)

    # Save CSVs
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_v2_trades.csv"), index=False)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_v2_daily.csv"), index=False)
    print(f"Daily log ({len(daily_df)} rows) saved to Output/backtest_v2_daily.csv")