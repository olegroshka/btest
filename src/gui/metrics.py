"""
Metrics display functions for the dashboard
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from .theme import get_color_palette


colors = get_color_palette()


def display_metrics(result):
    """Display main performance metrics in cards."""
    summary = result.summary() if hasattr(result, 'summary') else {}
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_ret = summary.get('total_return', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: {colors['text_secondary']}; text-transform: uppercase; letter-spacing: 1px;">Total Return</div>
            <div style="font-size: 2rem; font-weight: 700; color: {colors['success'] if total_ret > 0 else colors['danger']}; margin-top: 0.5rem;">{total_ret:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sharpe = summary.get('sharpe', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: {colors['text_secondary']}; text-transform: uppercase; letter-spacing: 1px;">Sharpe Ratio</div>
            <div style="font-size: 2rem; font-weight: 700; color: {colors['info']}; margin-top: 0.5rem;">{sharpe:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_dd = summary.get('max_drawdown', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: {colors['text_secondary']}; text-transform: uppercase; letter-spacing: 1px;">Max Drawdown</div>
            <div style="font-size: 2rem; font-weight: 700; color: {colors['danger']}; margin-top: 0.5rem;">{max_dd:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volatility = summary.get('volatility', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: {colors['text_secondary']}; text-transform: uppercase; letter-spacing: 1px;">Volatility</div>
            <div style="font-size: 2rem; font-weight: 700; color: {colors['warning']}; margin-top: 0.5rem;">{volatility:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        sortino = summary.get('sortino', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: {colors['text_secondary']}; text-transform: uppercase; letter-spacing: 1px;">Sortino Ratio</div>
            <div style="font-size: 2rem; font-weight: 700; color: {colors['primary']}; margin-top: 0.5rem;">{sortino:.2f}</div>
        </div>
        """, unsafe_allow_html=True)


def display_signal_metrics(result, signal_name: str, metric_type: str):
    """Display signal-specific metrics."""
    st.markdown(f"#### {metric_type.replace('_', ' ').title()} Stats")
    
    # Placeholder metrics
    metrics = {
        'Mean': np.random.rand(),
        'Std Dev': np.random.rand() * 0.5,
        'Min': -np.random.rand(),
        'Max': np.random.rand(),
        'Coverage': np.random.rand() * 100
    }
    
    for metric, value in metrics.items():
        if metric == 'Coverage':
            st.metric(metric, f"{value:.1f}%")
        else:
            st.metric(metric, f"{value:.3f}")
