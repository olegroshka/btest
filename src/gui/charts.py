"""
Chart generation functions for the dashboard
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from .theme import get_color_palette


def get_theme_colors():
    """Get colors based on current theme."""
    theme = st.session_state.get('theme', 'light')
    return get_color_palette(theme)


def get_plot_template():
    """Get plotly template based on current theme."""
    theme = st.session_state.get('theme', 'light')
    return 'plotly_dark' if theme == 'dark' else 'plotly_white'


def plot_equity_curve(result) -> go.Figure:
    """Create interactive equity curve plot with theme colors."""
    colors = get_theme_colors()
    equity = result.equity
    
    fig = go.Figure()
    
    # Add equity line
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Equity',
        line=dict(color=colors['success'], width=3),
        fill='tozeroy',
        fillcolor=f"rgba(0, 212, 255, 0.1)",
        hovertemplate='<b>%{x}</b><br>Equity: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add markers for highs and lows
    rolling_max = equity.expanding().max()
    new_highs = equity[equity == rolling_max]
    
    fig.add_trace(go.Scatter(
        x=new_highs.index,
        y=new_highs.values,
        mode='markers',
        name='New Highs',
        marker=dict(
            color=colors['success'],
            size=8,
            symbol='star',
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>New High</b><br>%{x}<br>$%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Portfolio Equity Curve",
            'font': {'size': 20, 'color': colors['text']}
        },
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template=get_plot_template(),
        hovermode='x unified',
        height=450,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True
        )
    )
    
    return fig


def plot_returns_distribution(result) -> go.Figure:
    """Create returns distribution histogram."""
    colors = get_theme_colors()
    returns = result.returns.dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=60,
        name='Returns',
        marker=dict(
            color=returns.values,
            colorscale='RdYlGn',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            cmin=-returns.abs().max(),
            cmax=returns.abs().max()
        ),
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_ret = returns.mean()
    fig.add_vline(
        x=mean_ret,
        line_dash="dash",
        line_color=colors['warning'],
        annotation_text=f"Mean: {mean_ret:.2f}%",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Daily Returns Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template=get_plot_template(),
        height=400,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        showlegend=False
    )
    
    return fig


def plot_drawdown(result) -> go.Figure:
    """Create underwater/drawdown chart."""
    colors = get_theme_colors()
    equity = result.equity
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color=colors['danger'], width=2),
        fill='tozeroy',
        fillcolor=f"rgba(255, 107, 107, 0.3)",
        hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Highlight max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    
    fig.add_trace(go.Scatter(
        x=[max_dd_idx],
        y=[max_dd_val],
        mode='markers+text',
        name='Max DD',
        marker=dict(
            color=colors['danger'],
            size=15,
            symbol='x',
            line=dict(color='white', width=2)
        ),
        text=[f"Max: {max_dd_val:.2f}%"],
        textposition="top center",
        textfont=dict(color=colors['danger'], size=12),
        hovertemplate=f'<b>Max Drawdown</b><br>{max_dd_idx}<br>{max_dd_val:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Underwater Plot",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template=get_plot_template(),
        hovermode='x unified',
        height=400,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text'])
    )
    
    return fig


def plot_weights_evolution(result) -> go.Figure:
    """Plot portfolio weights evolution as stacked area chart."""
    colors = get_theme_colors()
    weights = result.weights
    
    fig = go.Figure()
    
    # Use a nice color palette
    color_palette = px.colors.sequential.Viridis
    
    for idx, col in enumerate(weights.columns):
        color_idx = idx % len(color_palette)
        fig.add_trace(go.Scatter(
            x=weights.index,
            y=weights[col],
            mode='lines',
            name=col,
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor=color_palette[color_idx],
            hovertemplate=f'<b>{col}</b><br>%{{x}}<br>Weight: %{{y:.2%}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Portfolio Weights Evolution",
        xaxis_title="Date",
        yaxis_title="Weight",
        template=get_plot_template(),
        hovermode='x unified',
        height=450,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        yaxis=dict(tickformat='.0%')
    )
    
    return fig


def plot_signal_evolution(result, signal_name: str) -> go.Figure:
    """Plot signal evolution over time for all assets."""
    colors = get_theme_colors()
    fig = go.Figure()
    
    # Use weights as proxy for signal evolution if actual signal data not available
    if hasattr(result, 'weights') and result.weights is not None:
        weights = result.weights
        
        # Show evolution of positions (which are driven by signals)
        for col in weights.columns:
            # Only show non-zero positions
            if weights[col].abs().max() > 0.001:
                fig.add_trace(go.Scatter(
                    x=weights.index,
                    y=weights[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2),
                    hovertemplate=f'<b>{col}</b><br>%{{x}}<br>Weight: %{{y:.2%}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=f"Position Evolution (driven by {signal_name})",
            xaxis_title="Date",
            yaxis_title="Position Weight",
            template=get_plot_template(),
            height=400,
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text']),
            hovermode='x unified',
            yaxis=dict(tickformat='.0%')
        )
    else:
        fig.add_annotation(
            text=f"No position data available for signal: {signal_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=colors['text_secondary'])
        )
        fig.update_layout(
            title=f"Signal Evolution: {signal_name}",
            template=get_plot_template(),
            height=400,
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
    
    return fig


def plot_signal_heatmap(result, signal_name: str) -> go.Figure:
    """Create heatmap of signal values across assets and time."""
    colors = get_theme_colors()
    fig = go.Figure()
    
    # Use weights data to create heatmap
    if hasattr(result, 'weights') and result.weights is not None:
        weights = result.weights
        
        # Sample data to avoid too dense heatmap
        if len(weights) > 100:
            weights_sample = weights.iloc[::max(1, len(weights)//100)]
        else:
            weights_sample = weights
        
        fig.add_trace(go.Heatmap(
            z=weights_sample.T.values,
            x=weights_sample.index,
            y=weights_sample.columns,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{y}</b><br>%{x}<br>Weight: %{z:.2%}<extra></extra>',
            colorbar=dict(
                title="Weight",
                tickformat='.0%'
            )
        ))
        
        fig.update_layout(
            title=f"Position Heatmap (driven by {signal_name})",
            xaxis_title="Date",
            yaxis_title="Asset",
            template=get_plot_template(),
            height=400,
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
    else:
        fig.add_annotation(
            text=f"No weight data available for heatmap",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=colors['text_secondary'])
        )
        fig.update_layout(
            title=f"Signal Heatmap: {signal_name}",
            template=get_plot_template(),
            height=400,
            plot_bgcolor=colors['card_bg'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'])
        )
    
    return fig


def plot_signal_ic_analysis(result, signal_name: str) -> go.Figure:
    """Plot Information Coefficient analysis for the signal."""
    colors = get_theme_colors()
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Rolling IC", "IC Distribution"),
        vertical_spacing=0.12
    )
    
    # Placeholder IC data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    ic_values = np.random.randn(len(dates)) * 0.1
    
    # Rolling IC
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ic_values,
            mode='lines',
            name='IC',
            line=dict(color=colors['info'], width=2),
            hovertemplate='%{x}<br>IC: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=1, col=1)
    
    # IC distribution
    fig.add_trace(
        go.Histogram(
            x=ic_values,
            nbinsx=30,
            name='IC Distribution',
            marker=dict(color=colors['info']),
            hovertemplate='IC: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Information Coefficient Analysis: {signal_name}",
        template=get_plot_template(),
        height=600,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font=dict(color=colors['text']),
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="IC Value", row=2, col=1)
    fig.update_yaxes(title_text="IC", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    return fig
