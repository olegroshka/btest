import numpy as np
import pandas as pd
import pandas.testing as pdt

from quantdsl_backtest.engine.vectorized_engine import _build_cost_matrices_from_orders
from quantdsl_backtest.dsl.costs import Commission
from quantdsl_backtest.dsl.execution import PowerLawSlippageModel


def _synthetic_inputs():
    # Two assets, five days
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    cols = ["A", "B"]

    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 101, 100],
            "B": [50, 50, 50, 51, 50],
        },
        index=dates,
    ).astype(float)

    volumes = pd.DataFrame(
        {
            "A": [1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000],
            "B": [500_000, 500_000, 500_000, 500_000, 500_000],
        },
        index=dates,
    ).astype(float)

    # orders_df mimics vectorbt's records_readable with Timestamp/Column/Size
    # Create buys and sells on different days and assets
    rows = [
        # Day 1: Buy 10k A, Sell 5k B
        {"Timestamp": dates[1], "Column": 0, "Size": 10_000},
        {"Timestamp": dates[1], "Column": 1, "Size": -5_000},
        # Day 3: Partial split trades on A (net 8k)
        {"Timestamp": dates[3], "Column": 0, "Size": 5_000},
        {"Timestamp": dates[3], "Column": 0, "Size": 3_000},
        # Day 4: Round trip on B within the day (abs>net)
        {"Timestamp": dates[4], "Column": 1, "Size": 2_000},
        {"Timestamp": dates[4], "Column": 1, "Size": -1_500},
    ]
    orders_df = pd.DataFrame(rows)

    commission = Commission(type="per_share", amount=0.003)  # $0.003/share
    slippage = PowerLawSlippageModel(base_bps=1.0, k=25.0, exponent=1.0, use_intraday_vol=False)

    return prices, volumes, orders_df, commission, slippage


def test_cost_matrices_sparse_vs_dense_equal():
    prices, volumes, orders_df, commission, slippage = _synthetic_inputs()

    fees_s, slip_s = _build_cost_matrices_from_orders(
        prices=prices,
        volumes=volumes,
        orders_df=orders_df,
        commission=commission,
        slippage_model=slippage,
        sparse=True,
        numpy_fast=True,
    )
    fees_d, slip_d = _build_cost_matrices_from_orders(
        prices=prices,
        volumes=volumes,
        orders_df=orders_df,
        commission=commission,
        slippage_model=slippage,
        sparse=False,
        numpy_fast=True,
    )

    # Expect exact equality
    if fees_s is not None or fees_d is not None:
        pdt.assert_frame_equal(fees_s, fees_d, check_dtype=False, atol=0.0, rtol=0.0)
    if slip_s is not None or slip_d is not None:
        pdt.assert_frame_equal(slip_s, slip_d, check_dtype=False, atol=0.0, rtol=0.0)


def test_cost_matrices_numpy_vs_pandas_equal():
    prices, volumes, orders_df, commission, slippage = _synthetic_inputs()

    fees_np, slip_np = _build_cost_matrices_from_orders(
        prices=prices,
        volumes=volumes,
        orders_df=orders_df,
        commission=commission,
        slippage_model=slippage,
        sparse=True,
        numpy_fast=True,
    )
    fees_pd, slip_pd = _build_cost_matrices_from_orders(
        prices=prices,
        volumes=volumes,
        orders_df=orders_df,
        commission=commission,
        slippage_model=slippage,
        sparse=True,
        numpy_fast=False,
    )

    # Expect exact equality
    if fees_np is not None or fees_pd is not None:
        pdt.assert_frame_equal(fees_np, fees_pd, check_dtype=False, atol=0.0, rtol=0.0)
    if slip_np is not None or slip_pd is not None:
        pdt.assert_frame_equal(slip_np, slip_pd, check_dtype=False, atol=0.0, rtol=0.0)
