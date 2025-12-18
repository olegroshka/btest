"""
QuantDSL Backtest GUI - Main Dashboard Application

A professional, dark-themed interactive dashboard for analyzing trading strategies.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os
import sys

# Set working directory to project root (2 levels up from src/gui)
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

# Add src to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from gui.theme import apply_dark_theme, apply_light_theme, get_color_palette
from gui.charts import (
    plot_equity_curve,
    plot_returns_distribution,
    plot_drawdown,
    plot_weights_evolution,
    plot_signal_evolution,
    plot_signal_heatmap,
    plot_signal_ic_analysis
)
from gui.metrics import display_metrics, display_signal_metrics
from gui.utils import load_backtest_result, get_available_strategies


# Page config
st.set_page_config(
    page_title="QuantDSL Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme switcher at the top
col1, col2, col3 = st.columns([6, 1, 1])
with col2:
    st.write("")  # Spacing
with col3:
    theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
    if st.button(f"{theme_icon} Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Apply theme based on selection
if st.session_state.theme == 'dark':
    apply_dark_theme()
else:
    apply_light_theme()


def render_sidebar():
    """Render sidebar configuration."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        strategies = get_available_strategies()
        strategy_name = st.selectbox(
            "Strategy",
            strategies,
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=None)
        with col2:
            end_date = st.date_input("End", value=None)
        
        st.markdown("---")
        
        st.markdown("### 🎨 Display Options")
        show_trades = st.checkbox("Show Trades", value=True)
        show_positions = st.checkbox("Show Positions", value=True)
        
        st.markdown("---")
        
        run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        
        if run_btn:
            st.session_state['run_backtest'] = True
            st.session_state['strategy_name'] = strategy_name
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("Click on charts to zoom. Double-click to reset.")
        
        return strategy_name, start_date, end_date, show_trades, show_positions


def render_overview_tab(result):
    """Render overview tab with main metrics and charts."""
    st.markdown("### 📊 Performance Overview")
    display_metrics(result)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(plot_equity_curve(result), use_container_width=True, key="equity_main")
    
    with col2:
        st.plotly_chart(plot_returns_distribution(result), use_container_width=True, key="returns_dist_main")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_drawdown(result), use_container_width=True, key="drawdown_main")
    
    with col2:
        # Rolling Sharpe
        returns = result.returns.dropna()
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * (252 ** 0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe (1Y)',
            line=dict(color='#00d4ff', width=2)
        ))
        fig.update_layout(
            title="Rolling Sharpe Ratio (1 Year)",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template='plotly_dark',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True, key="sharpe_rolling")


def render_signals_tab(result, strategy):
    """Render signals analysis tab."""
    st.markdown("### 🎯 Signal Analysis")
    
    # Get signal names from strategy
    signal_names = list(strategy.signals.keys()) if hasattr(strategy, 'signals') else []
    
    if not signal_names:
        st.warning("No signals available for this strategy")
        return
    
    # Signal selector
    selected_signal = st.selectbox(
        "Select Signal to Analyze",
        signal_names,
        index=0
    )
    
    st.markdown("---")
    
    # Create tabs for different signal views
    signal_tab1, signal_tab2, signal_tab3 = st.tabs(
        ["📈 Evolution", "🔥 Heatmap", "📊 IC Analysis"]
    )
    
    with signal_tab1:
        st.plotly_chart(
            plot_signal_evolution(result, selected_signal),
            use_container_width=True,
            key=f"signal_evolution_{selected_signal}"
        )
        
        # Display signal statistics
        col1, col2 = st.columns(2)
        with col1:
            display_signal_metrics(result, selected_signal, "cross_sectional")
        with col2:
            display_signal_metrics(result, selected_signal, "time_series")
    
    with signal_tab2:
        st.plotly_chart(
            plot_signal_heatmap(result, selected_signal),
            use_container_width=True,
            key=f"signal_heatmap_{selected_signal}"
        )
    
    with signal_tab3:
        st.plotly_chart(
            plot_signal_ic_analysis(result, selected_signal),
            use_container_width=True,
            key=f"signal_ic_{selected_signal}"
        )
    
    # All signals overview
    st.markdown("---")
    st.markdown("### 📋 All Signals Summary")
    
    cols = st.columns(min(len(signal_names), 4))
    for idx, sig_name in enumerate(signal_names):
        with cols[idx % 4]:
            st.markdown(f"**{sig_name}**")
            # Show small preview chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1, 2],
                y=[0, 1, 0],
                mode='lines',
                line=dict(color='#00d4ff', width=1)
            ))
            fig.update_layout(
                height=80,
                margin=dict(l=0, r=0, t=0, b=0),
                template='plotly_dark',
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"preview_{sig_name}")


def render_portfolio_tab(result):
    """Render portfolio analysis tab."""
    st.markdown("### ⚖️ Portfolio Analysis")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(
            plot_weights_evolution(result),
            use_container_width=True,
            key="weights_evolution"
        )
    
    with col2:
        # Current weights pie chart
        if hasattr(result, 'weights') and result.weights is not None:
            latest_weights = result.weights.iloc[-1].dropna()
            latest_weights = latest_weights[latest_weights.abs() > 0.001]
            
            fig = go.Figure(data=[go.Pie(
                labels=latest_weights.index,
                values=latest_weights.abs().values,
                hole=0.4,
                marker=dict(
                    colors=px.colors.sequential.Viridis
                )
            )])
            fig.update_layout(
                title="Current Position Allocation",
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key="weights_pie")
    
    # Turnover analysis
    st.markdown("---")
    st.markdown("### 🔄 Turnover Analysis")
    
    if hasattr(result, 'weights') and result.weights is not None:
        weights_diff = result.weights.diff().abs().sum(axis=1) / 2
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weights_diff.index,
                y=weights_diff.values * 100,
                mode='lines',
                fill='tozeroy',
                name='Daily Turnover',
                line=dict(color='#ff6b6b', width=1.5)
            ))
            fig.update_layout(
                title="Daily Turnover",
                xaxis_title="Date",
                yaxis_title="Turnover (%)",
                template='plotly_dark',
                height=300,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, key="turnover")
        
        with col2:
            # Turnover distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=weights_diff.dropna().values * 100,
                nbinsx=30,
                marker=dict(color='#ff6b6b')
            ))
            fig.update_layout(
                title="Turnover Distribution",
                xaxis_title="Turnover (%)",
                yaxis_title="Frequency",
                template='plotly_dark',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True, key="turnover_dist")


def render_details_tab(result, show_trades, show_positions):
    """Render detailed data tables."""
    st.markdown("### 📋 Detailed Data")
    
    tab1, tab2, tab3 = st.tabs(["📊 Trades", "💼 Positions", "📈 Returns"])
    
    with tab1:
        if show_trades and hasattr(result, 'trades') and result.trades is not None:
            st.dataframe(
                result.trades.tail(100),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trade data available")
    
    with tab2:
        if show_positions and hasattr(result, 'positions') and result.positions is not None:
            st.dataframe(
                result.positions.tail(50),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No position data available")
    
    with tab3:
        if hasattr(result, 'returns'):
            returns_df = pd.DataFrame({
                'Date': result.returns.index,
                'Return (%)': result.returns.values * 100,
                'Cumulative (%)': (result.returns.cumsum() * 100).values
            })
            st.dataframe(
                returns_df.tail(100),
                use_container_width=True,
                height=400
            )


def main():
    # Header with gradient
    st.markdown("""
        <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0; font-size: 2.5rem;'>
                📈 QuantDSL Backtest Dashboard
            </h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;'>
                Professional Trading Strategy Analysis Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    strategy_name, start_date, end_date, show_trades, show_positions = render_sidebar()
    
    # Main content
    if 'run_backtest' not in st.session_state:
        st.session_state['run_backtest'] = True
        st.session_state['strategy_name'] = strategy_name
    
    if st.session_state.get('run_backtest'):
        with st.spinner('🔄 Running backtest...'):
            result, strategy = load_backtest_result(st.session_state.get('strategy_name', strategy_name))
        
        if result:
            # Main tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overview",
                "🎯 Signals",
                "⚖️ Portfolio",
                "📋 Details"
            ])
            
            with tab1:
                render_overview_tab(result)
            
            with tab2:
                render_signals_tab(result, strategy)
            
            with tab3:
                render_portfolio_tab(result)
            
            with tab4:
                render_details_tab(result, show_trades, show_positions)
        else:
            st.error("❌ Failed to load backtest results")
    else:
        st.info("👈 Configure parameters and click 'Run Backtest' to start")


if __name__ == "__main__":
    main()
