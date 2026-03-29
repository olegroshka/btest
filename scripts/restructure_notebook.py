"""Restructure triple_leveraged_etf_strategy.ipynb into clear sections:
  Part 1: Manual strategy — individual signals, then combined, positions/events
  Part 2: DSL engine — build, run, positions/orders
  Part 3: Comparison — side-by-side charts
  Part 4: Optimization
"""
import json, uuid, copy

NB_PATH = r"notebooks\triple_leveraged_etf_strategy.ipynb"

def make_md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.split("\n"),
    }

def make_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.split("\n"),
        "execution_count": None,
        "outputs": [],
    }

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

old = {c["id"]: c for c in nb["cells"]}

def clear(cell):
    """Return a copy of cell with outputs cleared."""
    c = copy.deepcopy(cell)
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

# ── Existing cells by index (id) ──
# 0  d9222c42  Title markdown
# 1  5ab69980  Imports
# 2  45819638  "Load & Prepare Data"
# 3  01859f56  Load parquet
# 4  012d8dc3  Pivot
# 5  b0aca2f0  Returns + crash days
# 6  90ebc02a  "Manual Strategy" header
# 7  15fdc0f8  Strategy params + run
# 8  d2dbe154  "Per-Asset Signal Analysis"
# 9  c2657def  Per-asset metrics table + equity curves
# 10 8627bea2  Cum returns + rolling sharpe chart
# 11 4e51ff1b  PnL attribution chart
# 12 88295ae9  Annual returns chart
# 13 d29cc794  "Performance Metrics" header
# 14 0b9dd02b  calculate_metrics + benchmarks
# 15 5b2b9253  "Monthly heatmap" header
# 16 9bddef95  Monthly heatmap
# 17 d2bd6a26  DSL header
# 18 8c44f621  sys.path
# 19 45e41a7c  Build DSL
# 20 4404db20  Run DSL
# 21 caaf38a6  "DSL vs Manual Comparison" header
# 22 e44a9f10  Comparison table
# 23 db7ef053  DSL weights chart
# 24 f0a6fe10  "Why ~$700K diff" markdown
# 25 5f7c5613  DSL trades + costs
# 26 b123409a  DSL order flow
# 27 cc6a5a30  "Daily Positions" header
# 28 a2b1ec27  Positions comparison (builds combined df)
# 29 cbb147ce  Equity overlay + COVID
# 30 c669c404  Equity diff chart
# 31 0ac3c319  "Key Differences" header
# 32 bfb85e62  Feature comparison table
# 33 f2a41c37  "Optimization" header
# 34 5e450ab3  Current positions
# 35 2c8fe43d  Param sweep
# 36 8e5c2c79  Heatmaps
# 37 219a048b  Summary

# ===================================================================
# NEW CELLS
# ===================================================================

# --- New: calculate_metrics moved to data section ---
calc_metrics_code = '''def calculate_metrics(portfolio_values: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """Calculate key performance metrics for a portfolio equity curve."""
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
    }'''

# --- New: individual signals chart ---
individual_signals_code = '''# ====== Individual Signal Performance: TQQQ & TMF Standalone ======
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
pd.DataFrame({'TQQQ (3x Nasdaq)': tqqq_m, 'TMF (3x 20Y Treasury)': tmf_m})'''

# --- New: combined section metrics table (replaces old cell 14 which had calc_metrics def) ---
combined_metrics_code = '''# Benchmarks for manual strategy
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
metrics_df'''

# --- New: per-asset metrics + combined returns (replaces old cell 9) ---
per_asset_combined_code = '''# ====== Combined Strategy vs Individual Assets ======
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
metrics_table'''

# --- New: Manual positions & rebalancing events chart ---
manual_positions_code = '''# ====== Manual Strategy: Positions, Weights & Rebalancing Events ======

# Detect rebalance events (weight snaps back to ~50/50)
invested = strategy_results[~strategy_results['in_cash']].copy()
invested['w_diff'] = (invested['weight_tqqq'] - 0.5).abs()
invested['snap'] = invested['w_diff'] < 0.01
rebal_dates = invested[invested['snap'] & ~invested['snap'].shift(1, fill_value=False)].index

# Detect crash entry/exit events
cash_mask = strategy_results['in_cash']
crash_enter = cash_mask & ~cash_mask.shift(1, fill_value=False)  # transition to cash
crash_exit = ~cash_mask & cash_mask.shift(1, fill_value=False)    # transition from cash
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
      f"Avg time between rebals: {(strategy_results.index[-1] - strategy_results.index[0]).days / max(n_rebals,1):.0f} days")'''

# --- New: DSL section header after positions ---
dsl_positions_md = "### DSL Positions, Weights & Trades"

# --- New: Part 3 header ---
part3_md = """---
## Part 3: Manual vs DSL Comparison

Side-by-side comparison of both implementations — equity curves, positions, drawdowns, and the key structural differences that explain the ~$700K gap."""

# --- New: build combined df + comparison metrics ---
comparison_build_code = '''# Build combined Manual vs DSL DataFrame
manual_pos = strategy_results[['shares_tqqq', 'shares_tmf', 'weight_tqqq', 'weight_tmf',
                                'portfolio_value', 'in_cash']].copy()
manual_pos.columns = ['man_shares_TQQQ', 'man_shares_TMF', 'man_w_TQQQ', 'man_w_TMF', 'man_equity', 'man_in_cash']

dsl_pos = pd.DataFrame(index=result.positions.index)
dsl_pos['dsl_shares_TQQQ'] = result.positions['TQQQ']
dsl_pos['dsl_shares_TMF']  = result.positions['TMF']
dsl_pos['dsl_w_TQQQ']      = result.weights['TQQQ']
dsl_pos['dsl_w_TMF']       = result.weights['TMF']
dsl_pos['dsl_equity']       = result.equity

manual_pos.index = pd.to_datetime(manual_pos.index)
dsl_pos.index = pd.to_datetime(dsl_pos.index)
combined = manual_pos.join(dsl_pos, how='inner')
combined['equity_diff'] = combined['dsl_equity'] - combined['man_equity']
combined['equity_diff_pct'] = (combined['equity_diff'] / combined['man_equity']) * 100

# Metrics comparison
dsl_equity = result.equity
dsl_metrics = calculate_metrics(dsl_equity)

comparison = pd.DataFrame({
    'Manual (Rebal+Crash)': strategy_metrics,
    'DSL (Daily Rebal)': dsl_metrics,
    '50/50 Buy-Hold': simple_5050_metrics,
    'TQQQ Only': tqqq_metrics,
})
comparison'''

# ===================================================================
# ASSEMBLE NEW NOTEBOOK
# ===================================================================

cells = []

# ── SECTION 0: INTRO + DATA ──
cells.append(clear(old["d9222c42"]))   # Title
cells.append(clear(old["5ab69980"]))   # Imports
cells.append(clear(old["45819638"]))   # Load data header
cells.append(clear(old["01859f56"]))   # Load parquet
cells.append(clear(old["012d8dc3"]))   # Pivot
cells.append(clear(old["b0aca2f0"]))   # Returns + crash days
cells.append(make_code(calc_metrics_code))  # calculate_metrics defined early

# ── PART 1: MANUAL STRATEGY ──
cells.append(make_md("---\n## Part 1: Manual Strategy Implementation\n\n50/50 TQQQ/TMF with bimonthly rebalancing and crash filter."))
cells.append(clear(old["15fdc0f8"]))   # Strategy params + run

cells.append(make_md("### Individual Signals: TQQQ & TMF\n\nEach asset's standalone performance — equity curve, drawdown, and key metrics."))
cells.append(make_code(individual_signals_code))  # TQQQ/TMF standalone charts

cells.append(make_md("### Combined Strategy: 50/50 Rebalanced + Crash Filter\n\nCombining negatively correlated TQQQ and TMF reduces portfolio volatility and improves risk-adjusted returns."))
cells.append(make_code(per_asset_combined_code))  # Combined metrics table

# Cumulative returns + rolling sharpe chart
cells.append(clear(old["8627bea2"]))  # Cum returns chart

# PnL attribution
cells.append(clear(old["4e51ff1b"]))  # PnL attribution chart

cells.append(make_md("### Manual Positions & Rebalancing Events\n\nShare positions, portfolio weights, and markers for every rebalance and crash filter event."))
cells.append(make_code(manual_positions_code))

# Performance summary
cells.append(make_md("### Performance Summary"))
cells.append(make_code(combined_metrics_code))  # strategy_metrics + benchmarks

# Annual returns
cells.append(clear(old["88295ae9"]))

# Monthly heatmap
cells.append(clear(old["5b2b9253"]))  # header
cells.append(clear(old["9bddef95"]))  # heatmap

# ── PART 2: DSL ENGINE ──
cells.append(make_md("---\n## Part 2: DSL Engine Implementation\n\nExpress the strategy in QuantDSL and run through the backtest engine. **Limitation:** the crash filter (stateful regime switching) can't be expressed in current DSL signals — this tests the core 50/50 daily rebalancing only."))
cells.append(clear(old["8c44f621"]))   # sys.path
cells.append(clear(old["45e41a7c"]))   # Build DSL
cells.append(clear(old["4404db20"]))   # Run DSL

cells.append(make_md("### DSL Positions, Weights & Trades\n\nThe engine tracks every position, weight, and trade with full cost attribution."))
cells.append(clear(old["db7ef053"]))   # DSL weights chart
cells.append(clear(old["5f7c5613"]))   # Trades + costs
cells.append(clear(old["b123409a"]))   # Order flow + turnover

# ── PART 3: COMPARISON ──
cells.append(make_md(part3_md))
cells.append(make_code(comparison_build_code))  # Build combined + metrics

# Why the diff markdown
cells.append(clear(old["f0a6fe10"]))

# Equity overlay + COVID
cells.append(clear(old["cbb147ce"]))

# Equity diff chart
cells.append(clear(old["c669c404"]))

# Positions side-by-side (reuse the chart code from old cell 28, but without the combined df build)
positions_compare_chart = '''# ====== Positions & Weights: Manual vs DSL Side-by-Side ======
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=('TQQQ Shares: Manual vs DSL', 'TMF Shares: Manual vs DSL',
                    'TQQQ Weight (%): Manual vs DSL'))

fig.add_trace(go.Scatter(x=combined.index, y=combined['man_shares_TQQQ'],
    name='Manual TQQQ', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=combined.index, y=combined['dsl_shares_TQQQ'],
    name='DSL TQQQ', line=dict(color='purple', width=1, dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=combined.index, y=combined['man_shares_TMF'],
    name='Manual TMF', line=dict(color='green', width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=combined.index, y=combined['dsl_shares_TMF'],
    name='DSL TMF', line=dict(color='orange', width=1, dash='dot')), row=2, col=1)
fig.add_trace(go.Scatter(x=combined.index, y=combined['man_w_TQQQ']*100,
    name='Manual wt%', line=dict(color='blue', width=1.5)), row=3, col=1)
fig.add_trace(go.Scatter(x=combined.index, y=combined['dsl_w_TQQQ']*100,
    name='DSL wt%', line=dict(color='purple', width=1.5, dash='dot')), row=3, col=1)
fig.add_hline(y=50, line_dash='dot', line_color='gray', opacity=0.5, row=3, col=1)

fig.update_yaxes(title_text='TQQQ Shares', row=1, col=1)
fig.update_yaxes(title_text='TMF Shares', row=2, col=1)
fig.update_yaxes(title_text='Weight (%)', row=3, col=1)
fig.update_layout(height=700, template='plotly_white', hovermode='x unified',
                  title='Daily Positions & Weights: Manual vs DSL Engine')
fig.show()'''
cells.append(make_code(positions_compare_chart))

# Feature comparison table
cells.append(make_md("### Implementation Differences"))
cells.append(clear(old["bfb85e62"]))

# ── PART 4: OPTIMIZATION ──
cells.append(clear(old["f2a41c37"]))   # header
cells.append(clear(old["5e450ab3"]))   # Current positions
cells.append(clear(old["2c8fe43d"]))   # Param sweep
cells.append(clear(old["8e5c2c79"]))   # Heatmaps
cells.append(clear(old["219a048b"]))   # Summary

# ── Write ──
nb["cells"] = cells
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Restructured notebook: {len(cells)} cells written")
print("Sections:")
sec = 0
for i, c in enumerate(cells):
    if c["cell_type"] == "markdown":
        src = "".join(c["source"]).strip()
        if src.startswith("---") or src.startswith("##") or src.startswith("#"):
            first_line = src.split("\n")[0].strip("- ")
            if first_line:
                print(f"  Cell {i:2d}: {first_line[:70]}")
