#!/usr/bin/env python3
"""
Quick Explorer: Load DSL strategy data and explore signals

This is a simple script to inspect the lagging_indecies DSL strategy:
- Load the equity indices data
- Calculate momentum and volatility factors
- Show current signal rankings
- Explore individual index performance vs momentum signal
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from quantdsl_backtest.examples.lagging_indecies import build_strategy
from quantdsl_backtest.engine.data_loader import load_data_for_strategy


def load_and_prepare_data():
    """Load DSL strategy data and prepare for exploration"""
    print("Loading DSL strategy...")
    strategy = build_strategy()
    
    print("Loading market data...")
    market_data, _, _ = load_data_for_strategy(strategy)
    
    # Convert MarketData.bars (Dict[ticker, DataFrame]) to single DataFrame
    dfs = []
    for ticker, df in market_data.bars.items():
        df_copy = df.copy()
        df_copy['ticker'] = ticker
        dfs.append(df_copy)
    
    data_df = pd.concat(dfs, ignore_index=False)  # Keep datetime index
    data_df = data_df.sort_index()
    
    print(f"  [OK] Data shape: {data_df.shape}")
    print(f"  [OK] Tickers: {sorted(data_df['ticker'].unique())}")
    print(f"  [OK] Date range: {data_df.index.min().date()} to {data_df.index.max().date()}")
    
    return data_df, strategy


def add_factors(data_df):
    """Calculate momentum and volatility factors"""
    print("\nCalculating factors...")
    
    # Reset index to make datetime a column for easier groupby operations
    data_df = data_df.reset_index()
    data_df = data_df.rename(columns={'index': 'datetime'})
    
    data_df['log_ret'] = data_df.groupby('ticker')['close'].apply(
        lambda x: np.log(x / x.shift(1))
    ).reset_index(drop=True)
    
    # 126-day momentum
    data_df['mom_126'] = data_df.groupby('ticker')['log_ret'].apply(
        lambda x: x.rolling(126).sum()
    ).reset_index(drop=True)
    
    # Winsorization: clip to ±3σ
    data_df['mom_mean'] = data_df.groupby('ticker')['mom_126'].transform('mean')
    data_df['mom_std'] = data_df.groupby('ticker')['mom_126'].transform('std')
    data_df['mom_126_winsorized'] = np.clip(
        (data_df['mom_126'] - data_df['mom_mean']) / data_df['mom_std'],
        -3, 3
    )
    
    # 63-day volatility
    data_df['volatility_63'] = data_df.groupby('ticker')['log_ret'].apply(
        lambda x: x.rolling(63).std()
    ).reset_index(drop=True)
    
    print("  [OK] Factors calculated: mom_126, mom_126_winsorized, volatility_63")
    
    return data_df


def show_current_rankings(data_df):
    """Show current signal rankings"""
    print("\n" + "="*80)
    print("CURRENT SIGNAL RANKINGS (Most Recent Date)")
    print("="*80)
    
    # Get most recent date
    latest_date = data_df['datetime'].max()
    
    latest_data = data_df[data_df['datetime'] == latest_date].copy()
    
    # Sort by momentum (descending)
    latest_data = latest_data.sort_values('mom_126_winsorized', ascending=False)
    latest_data['rank'] = range(1, len(latest_data) + 1)
    
    # Display
    display_cols = ['rank', 'ticker', 'close', 'mom_126', 'mom_126_winsorized', 'volatility_63']
    print(f"\nAs of: {latest_date.date()}\n")
    print(latest_data[display_cols].to_string(index=False))
    
    print("\n" + "-"*80)
    print(f"Best momentum:  {latest_data.iloc[0]['ticker']} ({latest_data.iloc[0]['mom_126_winsorized']:.4f})")
    print(f"Worst momentum: {latest_data.iloc[-1]['ticker']} ({latest_data.iloc[-1]['mom_126_winsorized']:.4f})")
    
    return latest_data


def show_ticker_analysis(data_df, ticker='DAX'):
    """Show analysis for single ticker"""
    print("\n" + "="*80)
    print(f"TICKER ANALYSIS: {ticker}")
    print("="*80)
    
    ticker_data = data_df[data_df['ticker'] == ticker].copy()
    
    # Get recent data
    recent = ticker_data.tail(10).copy()
    
    display_cols = ['close', 'log_ret', 'mom_126', 'mom_126_winsorized', 'volatility_63']
    print(f"\nLast 10 days:\n")
    print(recent[display_cols].to_string())
    
    # Summary stats
    print(f"\n{ticker} Summary:")
    print(f"  Price: ${ticker_data['close'].iloc[-1]:.2f} (from ${ticker_data['close'].iloc[0]:.2f})")
    print(f"  Total return: {((ticker_data['close'].iloc[-1] / ticker_data['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"  Avg momentum: {ticker_data['mom_126'].mean():.4f}")
    print(f"  Current momentum: {ticker_data['mom_126'].iloc[-1]:.4f}")
    print(f"  Current momentum (winsorized): {ticker_data['mom_126_winsorized'].iloc[-1]:.4f}")
    print(f"  Avg volatility: {ticker_data['volatility_63'].mean():.4f}")


def show_factor_heatmap(data_df, num_days=20):
    """Show recent factor values as heatmap"""
    print("\n" + "="*80)
    print(f"MOMENTUM HEATMAP (Last {num_days} days)")
    print("="*80)
    
    # Get most recent dates
    unique_dates = data_df['datetime'].unique()[-num_days:]
    
    # Create heatmap: rows=tickers, columns=dates
    heatmap_data = []
    for ticker in sorted(data_df['ticker'].unique()):
        ticker_data = data_df[data_df['ticker'] == ticker].copy()
        ticker_data = ticker_data[ticker_data['datetime'].isin(unique_dates)].sort_values('datetime')
        momentum_values = ticker_data['mom_126_winsorized'].values
        heatmap_data.append(momentum_values)
    
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=sorted(data_df['ticker'].unique()),
        columns=[pd.to_datetime(d).date() for d in unique_dates]
    )
    
    print("\nMomentum values (winsorized, +/- 3 sigma):\n")
    print(heatmap_df.to_string())
    
    return heatmap_df


def main():
    print("="*80)
    print("DSL SIGNAL EXPLORER: Lagging Indices Strategy")
    print("="*80)
    print()
    
    # Load data
    data_df, strategy = load_and_prepare_data()
    
    # Add factors
    data_df = add_factors(data_df)
    
    # Show current rankings
    rankings = show_current_rankings(data_df)
    
    # Show analysis for best and worst
    best_ticker = rankings.iloc[0]['ticker']
    worst_ticker = rankings.iloc[-1]['ticker']
    
    show_ticker_analysis(data_df, best_ticker)
    show_ticker_analysis(data_df, worst_ticker)
    
    # Show heatmap
    show_factor_heatmap(data_df, num_days=20)
    
    print("\n" + "="*80)
    print("USAGE IN JUPYTER NOTEBOOK")
    print("="*80)
    print("""
# Load data
from explore_dsl_signals import load_and_prepare_data, add_factors

data_df, strategy = load_and_prepare_data()
data_df = add_factors(data_df)

# Explore
# Pandas operations on data_df
# Example: get last 20 days of DAX
dax_recent = data_df[data_df['ticker'] == 'DAX'].tail(20)
dax_recent[['close', 'mom_126', 'mom_126_winsorized']].plot()
""")
    print()


if __name__ == "__main__":
    main()
