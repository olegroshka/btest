import numpy as np
import pandas as pd

from quantdsl_backtest.models.slippage import (
    PowerLawSlippage,
    build_slippage_model,
)
from quantdsl_backtest.dsl.execution import PowerLawSlippageModel, Execution, OrderPolicy, LatencyModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Commission, StaticFees
from quantdsl_backtest.engine.execution_engine import rebalance_to_target_weights


def test_powerlaw_slippage_order_bps():
    cfg = PowerLawSlippageModel(base_bps=1.0, k=20.0, exponent=0.5)
    model = build_slippage_model(cfg)

    qty = 100.0
    volume = 1000.0
    part = abs(qty) / volume  # 0.1
    expected_bps = cfg.base_bps + cfg.k * (part ** cfg.exponent)
    got_bps = model.slippage_bps_from_order(qty=qty, volume=volume)
    assert abs(got_bps - expected_bps) < 1e-9


def test_slippage_matrix_numpy_net_scaling():
    # Set up a model with simple parameters
    model = PowerLawSlippage(base_bps=0.0, k=100.0, exponent=1.0)

    # 2 bars, 1 asset
    # Bar 0: net=5, abs=15, vol=100 -> part=0.05, bps=5, scale=5/15=1/3
    # Expect fraction = (5/1e4) * (1/3)
    net = np.array([[5.0], [0.0]])
    abs_tot = np.array([[15.0], [0.0]])
    vol = np.array([[100.0], [100.0]])

    frac = model.build_slippage_fraction_matrix_from_orders_numpy(
        net_size_arr=net,
        abs_size_arr=abs_tot,
        volumes_arr=vol,
    )
    expected_bar0 = (5.0 / 1e4) * (1.0 / 3.0)
    assert frac.shape == (2, 1)
    assert abs(frac[0, 0] - expected_bar0) < 1e-12
    assert frac[1, 0] == 0.0


def test_execution_engine_integration_exec_price_with_slippage():
    # Prepare minimal inputs
    date = pd.Timestamp("2024-01-02")
    instr = "T"

    prices = pd.Series({instr: 10.0})
    volumes = pd.Series({instr: 1000.0})
    prev_positions = pd.Series({instr: 0.0})
    target_weights = pd.Series({instr: 1.0})
    equity = 100.0

    # Commission per-share = 0
    commission = Commission(type="per_share", amount=0.0)
    fees = StaticFees()

    # Slippage: constant 10 bps
    sl_cfg = PowerLawSlippageModel(base_bps=10.0, k=0.0, exponent=1.0)
    execution = Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=sl_cfg,
        volume_limits=VolumeParticipation(max_participation=1.0),
    )

    new_pos, cash_delta, trades = rebalance_to_target_weights(
        date=date,
        execution=execution,
        commission=commission,
        fees=fees,
        equity=equity,
        prices=prices,
        volumes=volumes,
        prev_positions=prev_positions,
        target_weights=target_weights,
    )

    # Check position quantity: 100 notional at price ~10.01 -> qty ~ 10
    assert abs(new_pos[instr] - 10.0) < 1e-9

    # Slippage bps recorded
    assert len(trades) == 1
    tr = trades.iloc[0]
    assert abs(tr["slippage_bps"] - 10.0) < 1e-9

    # Exec price should be 10 * (1 + 0.001) = 10.01
    assert abs(tr["price"] - 10.01) < 1e-9

    # Cash delta should be -notional (no commission)
    assert abs(cash_delta + 10.01 * 10.0) < 1e-9
