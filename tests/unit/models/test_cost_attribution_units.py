from __future__ import annotations

# Force a non-interactive backend for headless test environments.
import matplotlib
matplotlib.use("Agg")

import re
from pathlib import Path

import numpy as np
import pandas as pd

from quantdsl_backtest.engine.analytics.attribution import costs_by_instrument_day
from quantdsl_backtest.engine.analytics.render_tearsheets import render_portfolio_signal_tearsheet_html
from quantdsl_backtest.engine.analytics.types import PortfolioSignalAttribution


def _make_quintile_attrib(*, idx: pd.DatetimeIndex) -> PortfolioSignalAttribution:
    # Minimal attribution object with 5 quantiles
    q_cols = [1, 2, 3, 4, 5]
    contrib_by_q = pd.DataFrame(0.0, index=idx, columns=q_cols)
    ls = pd.Series(0.0, index=idx)
    # Costs are set by caller
    return PortfolioSignalAttribution(contrib_ret_by_q=contrib_by_q, contrib_ret_ls=ls)


def test_costs_by_instrument_day_is_dollars_not_returns():
    # This test guards the unit boundary: costs_by_instrument_day returns $ cost PnL per day per instrument.
    # (Return-space conversion must happen elsewhere.)
    df = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-02 09:30"), pd.Timestamp("2020-01-02 09:31")],
            "instrument": ["A", "B"],
            "commission": [1.0, 2.0],
            "fees": [0.5, 1.5],
            "slippage_bps": [10.0, 20.0],
            "notional": [10000.0, 5000.0],
        }
    )
    piv = costs_by_instrument_day(df)
    d = pd.Timestamp("2020-01-02").normalize()

    # Expected $ costs:
    # A = 1 + 0.5 + 10000*(10bps)= 1.5 + 10 = 11.5
    # B = 2 + 1.5 +  5000*(20bps)= 3.5 + 10 = 13.5
    assert np.isclose(float(piv.loc[d, "A"]), 11.5)
    assert np.isclose(float(piv.loc[d, "B"]), 13.5)


def test_portfolio_tearsheet_labels_cost_table_as_drag_for_small_values(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    attr = _make_quintile_attrib(idx=idx)

    # Return-space-ish values (small)
    cost_by_q = pd.DataFrame(
        {
            1: [0.0, 0.0001, 0.0002],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
            5: [0.0, 0.0001, 0.0],
        },
        index=idx,
        dtype="float64",
    )
    attr.cost_pnl_by_q = cost_by_q

    out = tmp_path / "tearsheet.html"
    render_portfolio_signal_tearsheet_html(signal_name="s", attribution=attr, output_path=out)

    html = out.read_text(encoding="utf-8")
    assert "total_cost_drag" in html
    assert "total_cost_usd" not in html
    assert "Costs are shown in return space" in html


def test_portfolio_tearsheet_labels_cost_table_as_usd_for_large_values(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    attr = _make_quintile_attrib(idx=idx)

    # Dollar-ish values (large)
    cost_by_q = pd.DataFrame(
        {
            1: [0.0, 10.0, 20.0],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
            5: [0.0, 5.0, 0.0],
        },
        index=idx,
        dtype="float64",
    )
    attr.cost_pnl_by_q = cost_by_q

    out = tmp_path / "tearsheet.html"
    render_portfolio_signal_tearsheet_html(signal_name="s", attribution=attr, output_path=out)

    html = out.read_text(encoding="utf-8")
    assert "total_cost_usd" in html
    assert "total_cost_drag" not in html
    assert "Costs are shown in $ PnL" in html

    # Sanity: ensure a representative large value appears in the HTML.
    # Don't depend on exact float formatting.
    assert "20" in html
