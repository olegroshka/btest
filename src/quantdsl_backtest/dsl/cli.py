"""
Command-line tool to inspect DSL strategies without needing Jupyter.

Usage:
    python -m quantdsl_backtest.dsl.cli config <strategy_json>
    python -m quantdsl_backtest.dsl.cli data <data_path> <ticker>
    python -m quantdsl_backtest.dsl.cli factors <data_path> <tickers...>
    python -m quantdsl_backtest.dsl.cli signals <data_path> <tickers...>
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import click
import pyarrow.parquet as pq

from quantdsl_backtest.dsl.inspector import (
    DSLConfigInspector, DataInspector, FactorInspector, SignalInspector
)


@click.group()
def cli():
    """DSL Inspection Tools - Analyze DSL strategy configuration and data."""
    pass


@cli.command()
@click.argument('strategy_file', type=click.Path(exists=True))
def config(strategy_file: str):
    """Display DSL strategy configuration.
    
    Example:
        python -m quantdsl_backtest.dsl.cli config strategy.json
    """
    # This would deserialize strategy from JSON
    click.echo(f"📋 Loading strategy from {strategy_file}...")
    click.echo("(Configuration inspection requires deserialized Strategy object)")


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--ticker', '-t', help='Filter by ticker')
@click.option('--start', '-s', help='Start date (YYYY-MM-DD)')
@click.option('--end', '-e', help='End date (YYYY-MM-DD)')
def data(data_path: str, ticker: Optional[str] = None, 
         start: Optional[str] = None, end: Optional[str] = None):
    """Inspect market data from parquet file.
    
    Example:
        python -m quantdsl_backtest.dsl.cli data data/equities/sp500_daily -t AAPL
    """
    click.echo(f"📊 Loading data from {data_path}...")
    
    inspector = DataInspector(data_path)
    
    if ticker:
        ticker_data = inspector.get_ticker_data(ticker, start, end)
        click.echo(f"\n{ticker} Data")
        click.echo("-" * 60)
        click.echo(f"Records: {len(ticker_data)}")
        click.echo(f"Date range: {ticker_data.index.min().date()} to {ticker_data.index.max().date()}")
        
        if 'close' in ticker_data.columns:
            close_prices = ticker_data['close']
            click.echo(f"Close price: ${close_prices.mean():.2f} (mean)")
            click.echo(f"            ${close_prices.min():.2f} - ${close_prices.max():.2f}")
        
        if 'volume' in ticker_data.columns:
            click.echo(f"Avg volume: {ticker_data['volume'].mean():,.0f}")
        
        click.echo("\nFirst 5 rows:")
        click.echo(ticker_data.head().to_string())
    else:
        click.echo(inspector.data_summary())


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('tickers', nargs=-1, required=True)
@click.option('--lookback', '-l', type=int, default=126, help='Lookback period (days)')
def factors(data_path: str, tickers: tuple, lookback: int = 126):
    """Analyze factor values for tickers.
    
    Example:
        python -m quantdsl_backtest.dsl.cli factors data/equities/sp500_daily SPX DAX CAC --lookback 126
    """
    click.echo(f"📈 Loading data and computing factors...")
    
    inspector = DataInspector(data_path)
    data = inspector.load_data()
    
    factor_inspector = FactorInspector(
        factors={},
        data=data,
        tickers=list(tickers)
    )
    
    # Compute momentum factor
    mom_vals = factor_inspector.compute_factor('momentum', lookback)
    
    click.echo(f"\n{lookback}-Day Momentum Factor")
    click.echo("=" * 70)
    
    for ticker in tickers:
        if ticker in mom_vals.columns:
            col = mom_vals[ticker].dropna()
            click.echo(
                f"{ticker:8s}: "
                f"mean={col.mean():8.4f}  std={col.std():8.4f}  "
                f"min={col.min():8.4f}  max={col.max():8.4f}"
            )
    
    click.echo(f"\nLatest values ({mom_vals.index[-1].date()}):")
    latest = mom_vals.iloc[-1]
    for ticker in tickers:
        if ticker in latest.index:
            click.echo(f"  {ticker:8s}: {latest[ticker]:8.4f}")


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('tickers', nargs=-1, required=True)
@click.option('--date', '-d', help='Specific date (YYYY-MM-DD, default: latest)')
@click.option('--long-threshold', '-l', type=float, default=0.5, 
              help='Threshold for long positions (0-1)')
def signals(data_path: str, tickers: tuple, date: Optional[str] = None, 
            long_threshold: float = 0.5):
    """Analyze signal rankings and positions.
    
    Example:
        python -m quantdsl_backtest.dsl.cli signals data/equities/sp500_daily SPX DAX CAC
    """
    click.echo(f"📡 Loading data and computing signals...")
    
    inspector = DataInspector(data_path)
    data = inspector.load_data()
    
    factor_inspector = FactorInspector(
        factors={},
        data=data,
        tickers=list(tickers)
    )
    
    # Compute momentum rankings
    mom_vals = factor_inspector.compute_factor('momentum', 126)
    rankings = mom_vals.rank(axis=1, pct=True)
    
    # Get specific date or latest
    if date:
        target_date = pd.Timestamp(date)
        if target_date not in rankings.index:
            click.echo(f"❌ No data for {date}")
            return
    else:
        target_date = rankings.index[-1]
    
    latest_rank = rankings.loc[target_date]
    
    click.echo(f"\nSignal Rankings @ {target_date.date()}")
    click.echo("=" * 70)
    
    positions = {}
    for ticker in tickers:
        rank = latest_rank[ticker]
        
        if rank >= long_threshold:
            pos = "LONG"
            symbol = "🟢"
        elif rank < (1.0 - long_threshold):
            pos = "SHORT"
            symbol = "🔴"
        else:
            pos = "NEUTRAL"
            symbol = "⚪"
        
        positions[ticker] = pos
        click.echo(f"{symbol} {ticker:8s}: {pos:8s} (rank: {rank:.3f})")
    
    # Summary
    pos_counts = pd.Series(list(positions.values())).value_counts()
    click.echo(f"\nPosition Summary:")
    for pos_type, count in pos_counts.items():
        click.echo(f"  {pos_type}: {count}")


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
def validate(data_path: str):
    """Validate data integrity and structure.
    
    Example:
        python -m quantdsl_backtest.dsl.cli validate data/equities/sp500_daily
    """
    click.echo(f"✔️  Validating {data_path}...")
    
    try:
        inspector = DataInspector(data_path)
        data = inspector.load_data()
        
        click.echo(f"✓ File loaded successfully")
        click.echo(f"  Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
        click.echo(f"  Columns: {', '.join(data.columns)}")
        click.echo(f"  Tickers: {data['ticker'].nunique()} unique")
        click.echo(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
        
        # Check for NaN values
        nan_counts = data.isna().sum()
        if nan_counts.sum() > 0:
            click.echo(f"  ⚠️  Missing values:")
            for col, count in nan_counts[nan_counts > 0].items():
                click.echo(f"      {col}: {count:,}")
        else:
            click.echo(f"  ✓ No missing values")
        
        # Check data types
        click.echo(f"  Data types:")
        for col, dtype in data.dtypes.items():
            click.echo(f"      {col}: {dtype}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
