"""
Interactive DSL Explorer: ipywidgets-based UI for exploring DSL configuration,
data, factors, and signals with real-time updates.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable

import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.graph_objects as go

from quantdsl_backtest.dsl.inspector import (
    DataInspector, FactorInspector, SignalInspector,
    plot_data, plot_factor, plot_rankings
)


class InteractiveDSLExplorer:
    """Interactive widget-based explorer for DSL strategies."""
    
    def __init__(self, strategy, data_path: Path | str, tickers: List[str]):
        """Initialize explorer.
        
        Args:
            strategy: Strategy object with all DSL components
            data_path: Path to parquet data file
            tickers: List of tickers to explore
        """
        self.strategy = strategy
        self.data_inspector = DataInspector(data_path)
        self.tickers = tickers
        self.data = self.data_inspector.load_data()
        
        # Initialize inspectors
        self.factor_inspector = FactorInspector(
            factors=strategy.factors,
            data=self.data,
            tickers=tickers
        )
        
        # Cache for factors
        self._factor_cache = {}
        
    def _compute_momentum_factor(self, lookback: int = 126) -> pd.DataFrame:
        """Compute momentum factor for signal inspection."""
        key = f"mom_{lookback}"
        if key in self._factor_cache:
            return self._factor_cache[key]
        
        factor_vals = {}
        for ticker in self.tickers:
            ticker_data = self.data[self.data['ticker'] == ticker].copy()
            if 'close' in ticker_data.columns:
                ticker_data['log_ret'] = np.log(
                    ticker_data['close'] / ticker_data['close'].shift(1)
                )
                factor_vals[ticker] = ticker_data['log_ret'].rolling(lookback).sum()
        
        result = pd.DataFrame(factor_vals)
        self._factor_cache[key] = result
        return result
    
    def create_explorer(self):
        """Create and return interactive explorer UI."""
        
        # ====================================================================
        # TAB 1: Configuration Overview
        # ====================================================================
        
        config_output = widgets.Output()
        
        @config_output.capture()
        def show_config():
            from quantdsl_backtest.dsl.inspector import DSLConfigInspector
            inspector = DSLConfigInspector(self.strategy)
            print(inspector.summary())
        
        show_config()
        
        # ====================================================================
        # TAB 2: Data Explorer
        # ====================================================================
        
        ticker_dropdown_data = widgets.Dropdown(
            options=self.tickers,
            description='Ticker:',
            style={'description_width': '100px'}
        )
        
        date_range_data = widgets.DatetimeRangeSlider(
            min=pd.Timestamp(self.data.index.min()),
            max=pd.Timestamp(self.data.index.max()),
            value=(
                pd.Timestamp(self.data.index.min()),
                pd.Timestamp(self.data.index.max())
            ),
            step=86400000000000,  # 1 day in nanoseconds
            description='Date range:',
            style={'description_width': '100px'}
        )
        
        data_output = widgets.Output()
        
        def on_data_change(change):
            data_output.clear_output(wait=True)
            with data_output:
                ticker = ticker_dropdown_data.value
                ticker_data = self.data_inspector.get_ticker_data(
                    ticker,
                    start_date=date_range_data.value[0],
                    end_date=date_range_data.value[1]
                )
                
                # Show stats
                print(f"\n{ticker} Data Statistics")
                print("=" * 60)
                print(f"Records: {len(ticker_data)}")
                print(f"Date range: {ticker_data.index.min().date()} to {ticker_data.index.max().date()}")
                if 'close' in ticker_data.columns:
                    print(f"Close price: ${ticker_data['close'].mean():.2f} (avg)")
                    print(f"            ${ticker_data['close'].min():.2f} - ${ticker_data['close'].max():.2f}")
                if 'volume' in ticker_data.columns:
                    print(f"Volume: {ticker_data['volume'].mean():,.0f} (avg)")
                print()
                
                # Plot
                fig = plot_data(ticker_data, ticker)
                display(fig)
        
        ticker_dropdown_data.observe(on_data_change, names='value')
        date_range_data.observe(on_data_change, names='value')
        
        data_controls = widgets.VBox([ticker_dropdown_data, date_range_data])
        data_tab = widgets.VBox([data_controls, data_output])
        
        # Trigger initial display
        on_data_change(None)
        
        # ====================================================================
        # TAB 3: Factor Explorer
        # ====================================================================
        
        lookback_slider = widgets.IntSlider(
            min=20, max=252, step=1, value=126,
            description='Lookback (days):',
            style={'description_width': '150px'}
        )
        
        ticker_dropdown_factor = widgets.SelectMultiple(
            options=self.tickers,
            value=tuple(self.tickers[:3]),  # First 3 tickers
            description='Tickers:',
            style={'description_width': '100px'}
        )
        
        factor_output = widgets.Output()
        
        def on_factor_change(change):
            factor_output.clear_output(wait=True)
            with factor_output:
                lookback = lookback_slider.value
                selected_tickers = list(ticker_dropdown_factor.value)
                
                if not selected_tickers:
                    print("Select at least one ticker")
                    return
                
                # Compute factor
                mom_vals = self._compute_momentum_factor(lookback)
                selected_data = mom_vals[selected_tickers].dropna()
                
                print(f"\n6-Month Momentum ({lookback}-day lookback)")
                print("=" * 60)
                for ticker in selected_tickers:
                    if ticker in selected_data.columns:
                        col = selected_data[ticker]
                        print(f"{ticker:8s}: mean={col.mean():8.4f}, std={col.std():8.4f}, "
                              f"min={col.min():8.4f}, max={col.max():8.4f}")
                print()
                
                # Plot
                fig = plot_factor(selected_data.tail(252), selected_tickers,
                                 f"Momentum ({lookback}-day)")
                display(fig)
        
        lookback_slider.observe(on_factor_change, names='value')
        ticker_dropdown_factor.observe(on_factor_change, names='value')
        
        factor_controls = widgets.VBox([lookback_slider, ticker_dropdown_factor])
        factor_tab = widgets.VBox([factor_controls, factor_output])
        
        # Trigger initial display
        on_factor_change(None)
        
        # ====================================================================
        # TAB 4: Signal Inspector
        # ====================================================================
        
        signal_date_picker = widgets.DatePicker(
            description='Date:',
            value=pd.Timestamp(self.data.index.max()).date(),
            style={'description_width': '100px'}
        )
        
        long_threshold = widgets.FloatSlider(
            min=0.0, max=1.0, step=0.05, value=0.5,
            description='Long threshold:',
            style={'description_width': '150px'}
        )
        
        signal_output = widgets.Output()
        
        def on_signal_change(change):
            signal_output.clear_output(wait=True)
            with signal_output:
                # Compute momentum rankings
                mom_vals = self._compute_momentum_factor(126)
                rankings = mom_vals.rank(axis=1, pct=True)
                
                # Get positions at selected date
                selected_date = pd.Timestamp(signal_date_picker.value)
                if selected_date not in rankings.index:
                    print(f"No data for {selected_date}")
                    return
                
                latest_rank = rankings.loc[selected_date]
                threshold = long_threshold.value
                
                print(f"\nSignal Rankings @ {selected_date.date()}")
                print("=" * 60)
                
                positions = {}
                for ticker in self.tickers:
                    rank = latest_rank[ticker]
                    if rank >= threshold:
                        pos = "LONG"
                    elif rank < (1.0 - threshold):
                        pos = "SHORT"
                    else:
                        pos = "NEUTRAL"
                    
                    positions[ticker] = pos
                    color = "🟢" if pos == "LONG" else "🔴" if pos == "SHORT" else "⚪"
                    print(f"{color} {ticker:8s}: {pos:8s} (rank: {rank:.3f})")
                
                # Position summary
                pos_counts = pd.Series(list(positions.values())).value_counts()
                print(f"\nPosition summary:")
                for pos_type, count in pos_counts.items():
                    print(f"  {pos_type}: {count}")
                print()
                
                # Plot rankings heatmap
                fig = plot_rankings(rankings, self.tickers)
                display(fig)
        
        signal_date_picker.observe(on_signal_change, names='value')
        long_threshold.observe(on_signal_change, names='value')
        
        signal_controls = widgets.VBox([signal_date_picker, long_threshold])
        signal_tab = widgets.VBox([signal_controls, signal_output])
        
        # Trigger initial display
        on_signal_change(None)
        
        # ====================================================================
        # Create Tabs
        # ====================================================================
        
        tabs = widgets.Tab(
            children=[config_output, data_tab, factor_tab, signal_tab],
            titles=['🎯 Configuration', '📊 Data Explorer', '📈 Factor Inspector', '📡 Signal Rankings']
        )
        
        return tabs
