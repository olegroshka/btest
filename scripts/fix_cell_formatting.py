"""Fix cells 6, 10, 12, 16, 18, 31, 35 that lost newlines during restructuring."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "triple_leveraged_etf_strategy.ipynb"

def lines(code: str) -> list[str]:
    """Split a multiline string into notebook source lines (each ending with \\n except last)."""
    raw = code.strip().split("\n")
    result = [l + "\n" for l in raw[:-1]] + [raw[-1]]
    return result

# ── Cell 6: calculate_metrics ────────────────────────────────────────────────
CELL_6 = lines("""\
def calculate_metrics(portfolio_values: pd.Series, risk_free_rate: float = 0.02) -> dict:
    \"\"\"Calculate key performance metrics for a portfolio equity curve.\"\"\"
    daily_returns = portfolio_values.pct_change().dropna()
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    cagr = (1 + total_return) ** (1/years) - 1
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    downside_std = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0
    drawdown = (portfolio_values - portfolio_values.cummax()) / portfolio_values.cummax()
    max_drawdown = drawdown.min()
    mar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')
    return {
        'Years': years, 'Total Return': total_return * 100, 'CAGR': cagr * 100,
        'Volatility': volatility * 100, 'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown * 100, 'MAR Ratio': mar_ratio,
    }""")

# ── Cell 10: Individual Signal Performance ───────────────────────────────────
CELL_10 = lines("""\
# Individual Signal Performance: TQQQ & TMF Standalone
tqqq_equity = (prices['TQQQ'] / prices['TQQQ'].iloc[0]) * INITIAL_CAPITAL
tmf_equity  = (prices['TMF'] / prices['TMF'].iloc[0]) * INITIAL_CAPITAL

tqqq_rets = prices['TQQQ'].pct_change().dropna()
tmf_rets  = prices['TMF'].pct_change().dropna()

tqqq_dd = ((1 + tqqq_rets).cumprod() / (1 + tqqq_rets).cumprod().cummax() - 1) * 100
tmf_dd  = ((1 + tmf_rets).cumprod() / (1 + tmf_rets).cumprod().cummax() - 1) * 100

fig = make_subplots(rows=2, cols=2, vertical_spacing=0.10, horizontal_spacing=0.08,
    subplot_titles=('TQQQ Equity ($100K buy-hold)', 'TMF Equity ($100K buy-hold)',
                    'TQQQ Drawdown', 'TMF Drawdown'))

fig.add_trace(go.Scatter(x=tqqq_equity.index, y=tqqq_equity,
    name='TQQQ', line=dict(color='#2196F3', width=2),
    hovertemplate='$%{y:,.0f}<extra>TQQQ</extra>'), row=1, col=1)
fig.add_trace(go.Scatter(x=tmf_equity.index, y=tmf_equity,
    name='TMF', line=dict(color='#4CAF50', width=2),
    hovertemplate='$%{y:,.0f}<extra>TMF</extra>'), row=1, col=2)
fig.add_trace(go.Scatter(x=tqqq_dd.index, y=tqqq_dd,
    name='TQQQ DD', line=dict(color='#2196F3', width=1),
    fill='tozeroy', fillcolor='rgba(33,150,243,0.2)'), row=2, col=1)
fig.add_trace(go.Scatter(x=tmf_dd.index, y=tmf_dd,
    name='TMF DD', line=dict(color='#4CAF50', width=1),
    fill='tozeroy', fillcolor='rgba(76,175,80,0.2)'), row=2, col=2)

fig.update_yaxes(type='log', title_text='Equity ($)', row=1, col=1)
fig.update_yaxes(type='log', title_text='Equity ($)', row=1, col=2)
fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
fig.update_yaxes(title_text='Drawdown (%)', row=2, col=2)
fig.update_layout(height=550, template='plotly_white', hovermode='x unified',
    title='Individual Signals: TQQQ & TMF Standalone (Buy-Hold from $100K)',
    showlegend=False)
fig.show()

# Per-asset metrics
tqqq_m = calculate_metrics(tqqq_equity)
tmf_m = calculate_metrics(tmf_equity)
pd.DataFrame({'TQQQ (3x Nasdaq)': tqqq_m, 'TMF (3x 20Y Treasury)': tmf_m})""")

# ── Cell 12: Combined Strategy vs Individual Assets ─────────────────────────
CELL_12 = lines("""\
# Combined Strategy vs Individual Assets
bh5050_equity = 0.5 * tqqq_equity + 0.5 * tmf_equity
strat_equity = strategy_results['portfolio_value']

bh5050_rets = bh5050_equity.pct_change().dropna()
strat_rets  = strat_equity.pct_change().dropna()

common_idx = tqqq_rets.index.intersection(tmf_rets.index).intersection(strat_rets.index)
tqqq_rets = tqqq_rets.loc[common_idx]
tmf_rets  = tmf_rets.loc[common_idx]
bh5050_rets = bh5050_rets.loc[common_idx]
strat_rets  = strat_rets.loc[common_idx]

def asset_metrics(rets, rf=0.02):
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    sortino_denom = rets[rets < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf) / sortino_denom if sortino_denom > 0 else 0
    cum = (1 + rets).cumprod()
    dd  = (cum / cum.cummax() - 1).min()
    return {
        'Ann. Return (%)': ann_ret * 100, 'Ann. Vol (%)': ann_vol * 100,
        'Sharpe': sharpe, 'Sortino': sortino, 'Max DD (%)': dd * 100,
        'Hit Rate (%)': (rets > 0).mean() * 100,
        'Total Return (%)': ((1 + rets).prod() - 1) * 100,
    }

metrics_table = pd.DataFrame({
    'TQQQ': asset_metrics(tqqq_rets),
    'TMF': asset_metrics(tmf_rets),
    '50/50 Buy-Hold': asset_metrics(bh5050_rets),
    'Strategy (Rebal+Filter)': asset_metrics(strat_rets),
})
metrics_table""")

# ── Cell 16: Manual Positions & Rebalancing Events ──────────────────────────
CELL_16 = lines("""\
# Manual Strategy: Positions, Weights & Rebalancing Events

# Detect rebalance events (weight snaps back to ~50/50)
invested = strategy_results[~strategy_results['in_cash']].copy()
invested['w_diff'] = (invested['weight_tqqq'] - 0.5).abs()
invested['snap'] = invested['w_diff'] < 0.01
rebal_dates = invested[invested['snap'] & ~invested['snap'].shift(1, fill_value=False)].index

# Detect crash entry/exit events
cash_mask = strategy_results['in_cash']
crash_enter = cash_mask & ~cash_mask.shift(1, fill_value=False)
crash_exit = ~cash_mask & cash_mask.shift(1, fill_value=False)
crash_enter_dates = strategy_results.index[crash_enter]
crash_exit_dates = strategy_results.index[crash_exit]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=('Portfolio Equity & Events', 'TQQQ / TMF Share Counts',
                    'Portfolio Weights (TQQQ / TMF / Cash)'))

# Panel 1: Equity with event markers
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['portfolio_value'],
    name='Portfolio', line=dict(color='#9C27B0', width=2),
    hovertemplate='$%{y:,.0f}<extra></extra>'), row=1, col=1)

# Mark rebalance dates
if len(rebal_dates) > 1:
    rebal_vals = strategy_results.loc[rebal_dates, 'portfolio_value']
    fig.add_trace(go.Scatter(x=rebal_dates, y=rebal_vals, mode='markers',
        name='Rebalance', marker=dict(symbol='diamond', size=8, color='#FF9800'),
        hovertemplate='Rebalance: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>'), row=1, col=1)

# Mark crash events
for dt in crash_enter_dates:
    fig.add_vline(x=dt, line_dash='dash', line_color='red', opacity=0.5, row=1, col=1)
for dt in crash_exit_dates:
    fig.add_vline(x=dt, line_dash='dash', line_color='green', opacity=0.5, row=1, col=1)
fig.update_yaxes(type='log', title_text='Equity ($)', row=1, col=1)

# Panel 2: Share counts
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['shares_tqqq'],
    name='TQQQ shares', line=dict(color='#2196F3', width=1.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['shares_tmf'],
    name='TMF shares', line=dict(color='#4CAF50', width=1.5)), row=2, col=1)
fig.update_yaxes(title_text='Shares', row=2, col=1)

# Panel 3: Stacked weights
w_cash = 1 - strategy_results['weight_tqqq'] - strategy_results['weight_tmf']
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['weight_tqqq']*100,
    name='TQQQ wt%', stackgroup='w', line=dict(width=0.5, color='#2196F3'),
    fillcolor='rgba(33,150,243,0.6)'), row=3, col=1)
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['weight_tmf']*100,
    name='TMF wt%', stackgroup='w', line=dict(width=0.5, color='#4CAF50'),
    fillcolor='rgba(76,175,80,0.6)'), row=3, col=1)
fig.add_trace(go.Scatter(x=strategy_results.index, y=w_cash*100,
    name='Cash wt%', stackgroup='w', line=dict(width=0.5, color='#FFC107'),
    fillcolor='rgba(255,193,7,0.6)'), row=3, col=1)
fig.update_yaxes(title_text='Weight (%)', range=[0, 105], row=3, col=1)

fig.update_layout(height=800, template='plotly_white', hovermode='x unified',
    title='Manual Strategy: Positions, Weights & Rebalancing Events',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
fig.show()

n_rebals = len(rebal_dates) - 1  # subtract initial allocation
n_crashes = len(crash_enter_dates)
print(f"Rebalances: {n_rebals}  |  Crash exits: {n_crashes}  |  "
      f"Avg time between rebals: {(strategy_results.index[-1] - strategy_results.index[0]).days / max(n_rebals,1):.0f} days")""")

# ── Cell 18: Benchmark Metrics ──────────────────────────────────────────────
CELL_18 = lines("""\
# Benchmarks for manual strategy
strategy_metrics = calculate_metrics(strategy_results['portfolio_value'])

tqqq_bh = (prices['TQQQ'] / prices['TQQQ'].iloc[0]) * INITIAL_CAPITAL
tqqq_metrics = calculate_metrics(tqqq_bh)

simple_50_50 = (0.5 * (prices['TQQQ'] / prices['TQQQ'].iloc[0]) +
                0.5 * (prices['TMF'] / prices['TMF'].iloc[0])) * INITIAL_CAPITAL
simple_5050_metrics = calculate_metrics(simple_50_50)

metrics_df = pd.DataFrame({
    'Strategy (Rebal+Filter)': strategy_metrics,
    '50/50 Buy-Hold': simple_5050_metrics,
    'TQQQ Only': tqqq_metrics,
})
metrics_df""")

# ── Cell 31: Build combined Manual vs DSL ────────────────────────────────────
CELL_31 = lines("""\
# Build combined Manual vs DSL comparison DataFrame
manual_eq = strategy_results['portfolio_value'].rename('Manual')

dsl_eq = result.equity_curve()
if isinstance(dsl_eq, pd.DataFrame):
    dsl_eq = dsl_eq.iloc[:, 0]
dsl_eq.name = 'DSL'

combined = pd.concat([manual_eq, dsl_eq], axis=1).dropna()
print(f"Overlap: {len(combined)} days  |  {combined.index.min()} to {combined.index.max()}")

# Metrics comparison
manual_m = calculate_metrics(combined['Manual'])
dsl_m = calculate_metrics(combined['DSL'])
pd.DataFrame({'Manual': manual_m, 'DSL Engine': dsl_m})""")

# ── Cell 35: Positions & Weights Side-by-Side ───────────────────────────────
CELL_35 = lines("""\
# Positions & weights side-by-side comparison (Manual vs DSL)
dsl_weights = result.weights()
if isinstance(dsl_weights, pd.DataFrame) and len(dsl_weights.columns) >= 2:
    dsl_tqqq_w = dsl_weights.iloc[:, 0] * 100
    dsl_tmf_w = dsl_weights.iloc[:, 1] * 100
else:
    dsl_tqqq_w = pd.Series(dtype=float)
    dsl_tmf_w = pd.Series(dtype=float)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=('TQQQ Shares (Manual vs DSL)', 'TMF Shares (Manual vs DSL)',
                    'TQQQ Weight % (Manual vs DSL)'))

# Panel 1: TQQQ shares
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['shares_tqqq'],
    name='Manual TQQQ', line=dict(color='#9C27B0', width=1.5)), row=1, col=1)

# Panel 2: TMF shares
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['shares_tmf'],
    name='Manual TMF', line=dict(color='#9C27B0', width=1.5, dash='dash')), row=2, col=1)

# Panel 3: TQQQ weight %
fig.add_trace(go.Scatter(x=strategy_results.index, y=strategy_results['weight_tqqq']*100,
    name='Manual wt%', line=dict(color='#9C27B0', width=1.5)), row=3, col=1)
if len(dsl_tqqq_w) > 0:
    fig.add_trace(go.Scatter(x=dsl_tqqq_w.index, y=dsl_tqqq_w,
        name='DSL wt%', line=dict(color='#FF9800', width=1.5, dash='dash')), row=3, col=1)

fig.update_yaxes(title_text='Shares', row=1, col=1)
fig.update_yaxes(title_text='Shares', row=2, col=1)
fig.update_yaxes(title_text='TQQQ wt%', row=3, col=1)
fig.update_layout(height=650, template='plotly_white', hovermode='x unified',
    title='Positions & Weights: Manual vs DSL',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
fig.show()""")

# ── Also fix markdown cells that lost newlines ──────────────────────────────

CELL_7_MD = lines("""\
---
## Part 1: Manual Strategy Implementation
50/50 TQQQ/TMF with bimonthly rebalancing and crash filter.""")

CELL_9_MD = lines("""\
### Individual Signals: TQQQ & TMF
Each asset's standalone performance — equity curve, drawdown, and key metrics.""")

CELL_11_MD = lines("""\
### Combined Strategy: 50/50 Rebalanced + Crash Filter
Combining negatively correlated TQQQ and TMF reduces portfolio volatility and improves risk-adjusted returns.""")

CELL_15_MD = lines("""\
### Manual Positions & Rebalancing Events
Share positions, portfolio weights, and markers for every rebalance and crash filter event.""")

CELL_22_MD = lines("""\
---
## Part 2: DSL Engine Implementation
Express the strategy in QuantDSL and run through the backtest engine. **Limitation:** the crash filter (stateful regime switching) can't be expressed in current DSL signals — this tests the core 50/50 daily rebalancing only.""")

CELL_26_MD = lines("""\
### DSL Positions, Weights & Trades""")

CELL_30_MD = lines("""\
---
## Part 3: Manual vs DSL Comparison""")

CELL_36_MD = lines("""\
### Implementation Differences""")

# ── Apply fixes ──────────────────────────────────────────────────────────────
with open(NB_PATH) as f:
    nb = json.load(f)

fixes = {
    6: CELL_6,
    7: CELL_7_MD,
    9: CELL_9_MD,
    10: CELL_10,
    11: CELL_11_MD,
    12: CELL_12,
    15: CELL_15_MD,
    16: CELL_16,
    18: CELL_18,
    22: CELL_22_MD,
    26: CELL_26_MD,
    30: CELL_30_MD,
    31: CELL_31,
    35: CELL_35,
    36: CELL_36_MD,
}

for idx, new_source in fixes.items():
    old_len = len(nb['cells'][idx]['source'])
    nb['cells'][idx]['source'] = new_source
    print(f"Cell {idx}: {old_len} -> {len(new_source)} lines")

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\nDone — fixed {len(fixes)} cells")
