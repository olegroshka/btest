"""
Utility functions for loading and managing backtest data
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import sys

# Add parent to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@st.cache_data
def load_backtest_result(strategy_name: str):
    """Load or run backtest and cache the result."""
    
    if strategy_name == "Lagging Indices":
        from quantdsl_backtest.examples.lagging_indecies import build_strategy
        from quantdsl_backtest.engine.backtest_runner import run_backtest
        
        strategy = build_strategy()
        result = run_backtest(strategy)
        return result, strategy
    
    elif strategy_name == "Momentum L/S SP500":
        from quantdsl_backtest.examples.momentum_long_short_sp500 import build_strategy
        from quantdsl_backtest.engine.backtest_runner import run_backtest
        
        strategy = build_strategy()
        result = run_backtest(strategy)
        return result, strategy
    
    return None, None


def get_available_strategies():
    """Get list of available strategies."""
    return [
        "Lagging Indices",
        "Momentum L/S SP500"
    ]
