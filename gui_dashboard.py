"""
Interactive GUI Dashboard for QuantDSL Backtest

This Streamlit app provides an interactive interface to:
- Explore market data and signals
- Generate and test signals with different parameters
- Run backtests with different parameters
- Visualize signals and their evolution
- Analyze portfolio performance
- Compare multiple strategies

Usage:
    streamlit run gui_dashboard.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os

# Import backtest components
from quantdsl_backtest.examples.lagging_indecies import build_strategy
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.engine.data_loader import load_data_for_strategy
from quantdsl_backtest.engine.results import BacktestResult


# Page config
st.set_page_config(
    page_title="QuantDSL Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stPlotlyChart {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING AND SIGNAL GENERATION
# ============================================================================

@st.cache_data
def load_market_data():
    """Load market data from DSL strategy"""
    strategy = build_strategy()
    market_data, _, _ = load_data_for_strategy(strategy)
    
    # Convert MarketData.bars (Dict[ticker, DataFrame]) to single DataFrame
    dfs = []
    for ticker, df in market_data.bars.items():
        df_copy = df.copy()
        df_copy['ticker'] = ticker
        dfs.append(df_copy)
    
    data_df = pd.concat(dfs, ignore_index=False)
    data_df = data_df.sort_index()
    data_df = data_df.reset_index()
    data_df = data_df.rename(columns={'index': 'datetime'})
    
    return data_df, strategy


@st.cache_data
def calculate_factors(data_df, mom_lookback=126, vol_lookback=63):
    """Calculate momentum and volatility factors"""
    data_df = data_df.copy()
    
    data_df['log_ret'] = data_df.groupby('ticker')['close'].apply(
        lambda x: np.log(x / x.shift(1))
    ).reset_index(drop=True)
    
    # Momentum
    data_df[f'mom_{mom_lookback}'] = data_df.groupby('ticker')['log_ret'].apply(
        lambda x: x.rolling(mom_lookback).sum()
    ).reset_index(drop=True)
    
    # Winsorization: clip to ±3σ
    mom_col = f'mom_{mom_lookback}'
    data_df['mom_mean'] = data_df.groupby('ticker')[mom_col].transform('mean')
    data_df['mom_std'] = data_df.groupby('ticker')[mom_col].transform('std')
    data_df[f'mom_{mom_lookback}_winsorized'] = np.clip(
        (data_df[mom_col] - data_df['mom_mean']) / data_df['mom_std'],
        -3, 3
    )
    
    # Volatility
    data_df[f'vol_{vol_lookback}'] = data_df.groupby('ticker')['log_ret'].apply(
        lambda x: x.rolling(vol_lookback).std()
    ).reset_index(drop=True)
    
    # Cross-sectional ranking (within each date)
    data_df['momentum_rank'] = data_df.groupby('datetime')[f'mom_{mom_lookback}_winsorized'].rank(method='min', ascending=False)
    
    return data_df


@st.cache_data
def load_backtest_result(strategy_name: str) -> BacktestResult:
    """Load or run backtest and cache the result."""
    if strategy_name == "Lagging Indices":
        strategy = build_strategy()
        result = run_backtest(strategy)
        return result
    # Add more strategies here
    return None


def plot_equity_curve(result: BacktestResult) -> go.Figure:
    """Create interactive equity curve plot."""
    fig = go.Figure()
    
    equity = result.equity
    
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_returns_distribution(result: BacktestResult) -> go.Figure:
    """Create returns distribution histogram."""
    returns = result.returns.dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Returns',
        marker=dict(
            color='#1f77b4',
            line=dict(color='white', width=1)
        )
    ))
    
    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    """Create drawdown chart."""
    equity = result.equity
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)'
    ))
    
    fig.update_layout(
        title="Drawdown Chart",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=350
    )
    
    return fig


def plot_signal_heatmap(result: BacktestResult, signal_name: str = "rank_mom") -> go.Figure:
    """Create heatmap of signal values over time."""
    # This would need signal data stored in result
    # Placeholder for demonstration
    fig = go.Figure()
    
    fig.add_annotation(
        text="Signal heatmap visualization<br>(requires signal data export)",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="gray")
    )
    
    fig.update_layout(
        title=f"Signal: {signal_name}",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_weights_evolution(result: BacktestResult) -> go.Figure:
    """Plot portfolio weights evolution over time."""
    weights = result.weights
    
    fig = go.Figure()
    
    for col in weights.columns:
        fig.add_trace(go.Scatter(
            x=weights.index,
            y=weights[col],
            mode='lines',
            name=col,
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Portfolio Weights Evolution",
        xaxis_title="Date",
        yaxis_title="Weight",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def display_metrics(result: BacktestResult):
    """Display key performance metrics."""
    summary = result.summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{summary.get('total_return', 0):.2%}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{summary.get('sharpe', 0):.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{summary.get('max_drawdown', 0):.2%}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Annual Volatility",
            f"{summary.get('volatility', 0):.2%}",
            delta=None
        )


def main():
    # Header
    st.markdown('<h1 class="main-header">📈 QuantDSL Backtest Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Main tabs
    tab_data, tab_signals, tab_backtest = st.tabs(
        ["🔍 Data Explorer", "📊 Signal Generator", "💰 Backtest Results"]
    )
    
    # ======================================================================
    # DATA EXPLORER TAB
    # ======================================================================
    with tab_data:
        st.subheader("Market Data Explorer")
        
        # Load data
        with st.spinner("Loading market data..."):
            data_df, strategy = load_market_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📊 Data Summary:\n- Tickers: {len(data_df['ticker'].unique())}\n- Rows: {len(data_df):,}\n- Date range: {data_df['datetime'].min().date()} to {data_df['datetime'].max().date()}")
        
        with col2:
            ticker_filter = st.multiselect(
                "Select Tickers",
                sorted(data_df['ticker'].unique()),
                default=sorted(data_df['ticker'].unique())[:3]
            )
        
        # Filter data
        filtered_df = data_df[data_df['ticker'].isin(ticker_filter)].copy()
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_last_n = st.slider("Show last N rows", 5, 100, 20)
        with col2:
            if st.button("Show Raw Data"):
                st.dataframe(filtered_df.tail(show_last_n), use_container_width=True, height=400)
        with col3:
            if st.button("Show Statistics"):
                st.dataframe(filtered_df[['ticker', 'close', 'volume']].groupby('ticker').describe(), use_container_width=True)
        
        # Price chart
        st.subheader("Price History")
        price_data = filtered_df.sort_values('datetime')
        fig = px.line(
            price_data,
            x='datetime',
            y='close',
            color='ticker',
            title="Close Prices Over Time",
            labels={'datetime': 'Date', 'close': 'Price ($)', 'ticker': 'Index'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ======================================================================
    # SIGNAL GENERATOR TAB
    # ======================================================================
    with tab_signals:
        st.subheader("Interactive Signal Generator")
        
        # Load data
        with st.spinner("Loading market data..."):
            data_df, strategy = load_market_data()
        
        # Parameter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            mom_lookback = st.slider("Momentum Lookback (days)", 20, 250, 126, step=10)
        with col2:
            vol_lookback = st.slider("Volatility Lookback (days)", 10, 125, 63, step=5)
        with col3:
            if st.button("Generate Signals", type="primary"):
                st.session_state['generate_signals'] = True
        
        if st.session_state.get('generate_signals', False):
            with st.spinner("Calculating factors..."):
                data_df = calculate_factors(data_df, mom_lookback, vol_lookback)
            
            # Get latest rankings
            latest_date = data_df['datetime'].max()
            latest_data = data_df[data_df['datetime'] == latest_date].copy()
            latest_data = latest_data.sort_values(f'mom_{mom_lookback}_winsorized', ascending=False)
            latest_data['rank'] = range(1, len(latest_data) + 1)
            
            # Show rankings
            st.subheader(f"Signal Rankings (As of {latest_date.date()})")
            display_cols = ['rank', 'ticker', 'close', f'mom_{mom_lookback}', f'mom_{mom_lookback}_winsorized', f'vol_{vol_lookback}']
            st.dataframe(latest_data[display_cols], use_container_width=True)
            
            # Show best/worst
            col1, col2, col3 = st.columns(3)
            with col1:
                best_ticker = latest_data.iloc[0]['ticker']
                best_momentum = latest_data.iloc[0][f'mom_{mom_lookback}_winsorized']
                st.metric("Best Momentum", f"{best_ticker}", f"{best_momentum:.4f}")
            with col2:
                worst_ticker = latest_data.iloc[-1]['ticker']
                worst_momentum = latest_data.iloc[-1][f'mom_{mom_lookback}_winsorized']
                st.metric("Worst Momentum", f"{worst_ticker}", f"{worst_momentum:.4f}")
            with col3:
                spread = best_momentum - worst_momentum
                st.metric("Spread", f"", f"{spread:.4f}")
            
            # Momentum heatmap
            st.subheader("Recent Momentum Heatmap")
            num_days = st.slider("Days to show in heatmap", 5, 30, 20)
            
            recent_dates = data_df['datetime'].unique()[-num_days:]
            heatmap_data = []
            for ticker in sorted(data_df['ticker'].unique()):
                ticker_data = data_df[data_df['ticker'] == ticker].copy()
                ticker_data = ticker_data[ticker_data['datetime'].isin(recent_dates)].sort_values('datetime')
                momentum_values = ticker_data[f'mom_{mom_lookback}_winsorized'].values
                heatmap_data.append(momentum_values)
            
            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=sorted(data_df['ticker'].unique()),
                columns=[pd.to_datetime(d).date() for d in recent_dates]
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns,
                y=heatmap_df.index,
                colorscale='RdYlGn_r',
                zmid=0,
                colorbar=dict(title="Momentum")
            ))
            fig.update_layout(height=300, title="Momentum Heatmap (Green=High, Red=Low)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Ticker analysis
            st.subheader("Individual Ticker Analysis")
            selected_ticker = st.selectbox("Select ticker for detailed analysis", sorted(data_df['ticker'].unique()))
            
            ticker_data = data_df[data_df['ticker'] == selected_ticker].sort_values('datetime').tail(30)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${ticker_data['close'].iloc[-1]:.2f}", 
                         f"{((ticker_data['close'].iloc[-1] / ticker_data['close'].iloc[0]) - 1) * 100:.2f}%")
            with col2:
                st.metric("Momentum", f"{ticker_data[f'mom_{mom_lookback}_winsorized'].iloc[-1]:.4f}",
                         f"Volatility: {ticker_data[f'vol_{vol_lookback}'].iloc[-1]:.4f}")
            
            # Price and momentum chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=ticker_data['datetime'], y=ticker_data['close'], name="Price", line=dict(color='blue')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=ticker_data['datetime'], y=ticker_data[f'mom_{mom_lookback}_winsorized'], 
                          name="Momentum", line=dict(color='red')),
                secondary_y=True
            )
            
            fig.update_layout(height=350, title=f"{selected_ticker} - Price vs Momentum")
            fig.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Momentum", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # ======================================================================
    # BACKTEST RESULTS TAB
    # ======================================================================
    with tab_backtest:
        st.subheader("Backtest Results")
        
        # Sidebar for strategy selection
        strategy_name = st.selectbox(
            "Select Strategy",
            ["Lagging Indices", "Momentum L/S SP500", "Custom"]
        )
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner('Running backtest...'):
                result = load_backtest_result(strategy_name)
            
            if result:
                # Metrics row
                st.subheader("Performance Metrics")
                display_metrics(result)
                
                st.markdown("---")
                
                # Charts
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(
                    ["📈 Equity & Returns", "📉 Drawdown & Risk", "⚖️ Portfolio Weights"]
                )
                
                with chart_tab1:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(plot_equity_curve(result), use_container_width=True)
                    with col2:
                        st.plotly_chart(plot_returns_distribution(result), use_container_width=True)
                
                with chart_tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_drawdown(result), use_container_width=True)
                    with col2:
                        st.info("Additional risk metrics coming soon")
                
                with chart_tab3:
                    st.plotly_chart(plot_weights_evolution(result), use_container_width=True)
                
                # Detailed tables
                with st.expander("📋 View Detailed Results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Recent Trades")
                        if hasattr(result, 'trades') and result.trades is not None:
                            st.dataframe(result.trades.tail(20), use_container_width=True)
                        else:
                            st.info("No trade data available")
                    
                    with col2:
                        st.subheader("Current Positions")
                        if hasattr(result, 'positions') and result.positions is not None:
                            st.dataframe(result.positions.tail(10), use_container_width=True)
                        else:
                            st.info("No position data available")
            else:
                st.error("Failed to load backtest results")


if __name__ == "__main__":
    main()
