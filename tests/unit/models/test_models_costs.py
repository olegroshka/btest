import numpy as np
import pandas as pd

from quantdsl_backtest.models.costs import (
    PerShareCommission,
    BpsNotionalCommission,
    build_cost_model,
)
from quantdsl_backtest.dsl.costs import Commission


def test_per_share_commission_trade_amount():
    cm = PerShareCommission(amount_per_share=0.02)
    # 100 shares -> $2.00
    assert abs(cm.commission_from_trade(qty=100.0, exec_price=10.0) - 2.0) < 1e-12
    # -50 shares -> $1.00 (absolute)
    assert abs(cm.commission_from_trade(qty=-50.0, exec_price=8.0) - 1.0) < 1e-12


def test_bps_commission_trade_amount():
    cm = BpsNotionalCommission(bps=10.0)  # 10 bps = 0.001
    # 100 shares at $10 -> notional=1000 -> commission=1.0
    assert abs(cm.commission_from_trade(qty=100.0, exec_price=10.0) - 1.0) < 1e-12
    # 20 shares at $50 -> notional=1000 -> commission=1.0
    assert abs(cm.commission_from_trade(qty=-20.0, exec_price=50.0) - 1.0) < 1e-12


def test_per_share_fees_matrix_numpy():
    cm = PerShareCommission(amount_per_share=0.05)
    prices = np.array([[10.0, 0.0], [20.0, 5.0]], dtype=float)
    abs_size = np.array([[100.0, 0.0], [0.0, 10.0]], dtype=float)
    # expected frac: per_share/price on bars with orders (abs_size>0), else 0
    out = cm.build_fees_fraction_matrix_from_orders_numpy(prices_arr=prices, abs_size_arr=abs_size)
    assert out.shape == prices.shape
    assert abs(out[0, 0] - (0.05 / 10.0)) < 1e-12
    assert out[0, 1] == 0.0
    assert out[1, 0] == 0.0
    assert abs(out[1, 1] - (0.05 / 5.0)) < 1e-12


def test_build_cost_model_from_dsl():
    cm = build_cost_model(Commission(type="per_share", amount=0.03))
    assert isinstance(cm, PerShareCommission)
    cm2 = build_cost_model(Commission(type="bps_notional", amount=15.0))
    assert isinstance(cm2, BpsNotionalCommission)
