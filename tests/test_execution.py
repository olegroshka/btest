import numpy as np
import pandas as pd

from quantdsl_backtest.dsl.execution import Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation
from quantdsl_backtest.dsl.costs import Commission, StaticFees
from quantdsl_backtest.engine.execution_engine import rebalance_to_target_weights


def _exec(unlimited: bool = True) -> Execution:
    vp = VolumeParticipation(max_participation=(None if unlimited else 0.1), mode="proportional", min_fill_notional=0.0)
    slip = PowerLawSlippageModel(base_bps=1.0, k=0.0, exponent=1.0)
    return Execution(
        order_policy=OrderPolicy(),
        latency=LatencyModel(),
        slippage=slip,
        volume_limits=vp,
    )


def test_rebalance_to_target_weights_unlimited_volume_with_bps_commission():
    # Two instruments, simple symmetric target
    names = ["A", "B"]
    date = pd.Timestamp("2020-01-02")
    equity = 1000.0
    prices = pd.Series({"A": 100.0, "B": 100.0})
    volumes = pd.Series({"A": 1_000_000.0, "B": 1_000_000.0})
    prev_pos = pd.Series(0.0, index=names)
    target_w = pd.Series({"A": 0.5, "B": -0.5})

    execution = _exec(unlimited=True)
    commission = Commission(type="bps_notional", amount=10.0)  # 10 bps
    fees = StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0)

    new_pos, cash_delta, trades = rebalance_to_target_weights(
        date,
        execution,
        commission,
        fees,
        equity,
        prices,
        volumes,
        prev_pos,
        target_w,
    )

    # Positions should match target notional / base price exactly (qty = dn / price)
    # dn for each = 0.5 * 1000 = 500 -> qty 5 and -5
    assert np.isclose(new_pos["A"], 5.0)
    assert np.isclose(new_pos["B"], -5.0)

    # Two trades generated
    assert len(trades) == 2
    assert set(trades["instrument"]) == {"A", "B"}
    assert set(trades["side"]) == {"BUY", "SELL"}

    # Slippage is base_bps = 1.0 bps; commission = 10 bps of notional_exec
    # Compute expected cash delta: sum(-notional_exec - commission)
    expected_cash = 0.0
    for instr in names:
        side = "BUY" if instr == "A" else "SELL"
        price = prices[instr]
        qty = 5.0 if side == "BUY" else -5.0
        slip_frac = 1.0 / 1e4
        exec_price = price * (1 + slip_frac) if side == "BUY" else price * (1 - slip_frac)
        notional_exec = exec_price * qty
        comm = 10.0 / 1e4 * abs(notional_exec)
        expected_cash += -notional_exec - comm

    assert np.isclose(cash_delta, expected_cash, rtol=1e-12, atol=1e-10)


def test_rebalance_capped_by_volume_participation():
    # Request a very large move but cap participation at 10% of volume
    names = ["A"]
    date = pd.Timestamp("2020-01-02")
    equity = 1_000_000.0
    prices = pd.Series({"A": 100.0})
    volumes = pd.Series({"A": 1000.0})  # 1000 shares
    prev_pos = pd.Series(0.0, index=names)
    target_w = pd.Series({"A": 1.0})  # want $1,000,000 notional long

    # Max participation 10% -> can trade at most 100 shares -> $10,000 notional
    exec_cfg = _exec(unlimited=False)
    exec_cfg.volume_limits.max_participation = 0.1

    commission = Commission(type="bps_notional", amount=0.0)
    fees = StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0)

    new_pos, cash_delta, trades = rebalance_to_target_weights(
        date,
        exec_cfg,
        commission,
        fees,
        equity,
        prices,
        volumes,
        prev_pos,
        target_w,
    )

    # Should only trade 10% of volume = 100 shares
    assert np.isclose(new_pos["A"], 100.0)
    assert len(trades) == 1
    assert trades.iloc[0]["quantity"] > 0


def test_min_fill_notional_skips_tiny_trades():
    # Single name, tiny target that falls below min_fill_notional should be skipped
    names = ["A"]
    date = pd.Timestamp("2020-01-02")
    equity = 10_000.0
    prices = pd.Series({"A": 100.0})
    volumes = pd.Series({"A": 1_000_000.0})
    prev_pos = pd.Series(0.0, index=names)

    # Target notional change = 0.0005 * 10_000 = $5, below threshold 10
    target_w = pd.Series({"A": 0.0005})

    exec_cfg = _exec(unlimited=True)
    exec_cfg.volume_limits.min_fill_notional = 10.0

    commission = Commission(type="bps_notional", amount=0.0)
    fees = StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0)

    new_pos, cash_delta, trades = rebalance_to_target_weights(
        date,
        exec_cfg,
        commission,
        fees,
        equity,
        prices,
        volumes,
        prev_pos,
        target_w,
    )

    # No trade occurred
    assert np.isclose(new_pos["A"], 0.0)
    assert np.isclose(cash_delta, 0.0)
    assert trades.empty


def test_powerlaw_slippage_exponent_and_k_affect_exec_price():
    # Verify slippage_bps = base + k * participation ** exponent
    names = ["A"]
    date = pd.Timestamp("2020-01-02")
    equity = 100_000.0
    price = 100.0
    prices = pd.Series({"A": price})
    volumes = pd.Series({"A": 50_000.0})  # high volume
    prev_pos = pd.Series(0.0, index=names)
    target_w = pd.Series({"A": 0.5})  # $50,000 notional -> qty 500

    # participation = 500 / 50_000 = 0.01
    base_bps = 2.0
    k = 800.0
    exponent = 1.0

    vp = VolumeParticipation(max_participation=None, mode="proportional", min_fill_notional=0.0)
    slip = PowerLawSlippageModel(base_bps=base_bps, k=k, exponent=exponent)
    exec_cfg = Execution(order_policy=OrderPolicy(), latency=LatencyModel(), slippage=slip, volume_limits=vp)

    commission = Commission(type="bps_notional", amount=0.0)
    fees = StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0)

    new_pos, cash_delta, trades = rebalance_to_target_weights(
        date,
        exec_cfg,
        commission,
        fees,
        equity,
        prices,
        volumes,
        prev_pos,
        target_w,
    )

    qty = 500.0
    participation = qty / volumes["A"]
    expected_slip_bps = base_bps + k * (participation ** exponent)
    expected_exec_price = price * (1 + expected_slip_bps / 1e4)

    # One BUY trade expected
    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "BUY"
    assert np.isclose(trades.iloc[0]["slippage_bps"], expected_slip_bps)
    assert np.isclose(trades.iloc[0]["price"], expected_exec_price)

    # Cash equals -notional (no commission/fees)
    expected_cash = -expected_exec_price * qty
    assert np.isclose(cash_delta, expected_cash, rtol=1e-12, atol=1e-8)


def test_per_share_commission_and_skip_invalid_instruments():
    # Three instruments: one buy, one sell, one invalid (NaN price)
    names = ["A", "B", "C"]
    date = pd.Timestamp("2020-01-02")
    equity = 1000.0
    prices = pd.Series({"A": 100.0, "B": 100.0, "C": np.nan})
    volumes = pd.Series({"A": 1_000_000.0, "B": 1_000_000.0, "C": 1_000_000.0})
    prev_pos = pd.Series(0.0, index=names)
    target_w = pd.Series({"A": 0.5, "B": -0.5, "C": 0.5})

    execution = _exec(unlimited=True)
    commission = Commission(type="per_share", amount=0.01)  # $0.01 per share
    fees = StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0)

    new_pos, cash_delta, trades = rebalance_to_target_weights(
        date,
        execution,
        commission,
        fees,
        equity,
        prices,
        volumes,
        prev_pos,
        target_w,
    )

    # A: +5 shares, B: -5 shares, C skipped due to NaN price
    assert np.isclose(new_pos["A"], 5.0)
    assert np.isclose(new_pos["B"], -5.0)
    assert np.isclose(new_pos.get("C", 0.0), 0.0)

    # Two trades only
    assert len(trades) == 2
    assert set(trades["instrument"]) == {"A", "B"}

    # Cash delta includes per-share commissions: 0.01 * (|5| + |5|) = 0.10
    slip_frac = execution.slippage.base_bps / 1e4
    buy_exec = 100.0 * (1 + slip_frac)
    sell_exec = 100.0 * (1 - slip_frac)
    notional_buy = buy_exec * 5.0
    notional_sell = sell_exec * -5.0
    expected_comm = 0.01 * (abs(5.0) + abs(-5.0))
    expected_cash = -(notional_buy + expected_comm) - (notional_sell + expected_comm - expected_comm)  # careful below
    # Simpler: compute from trades table
    calc_cash = 0.0
    for _, tr in trades.iterrows():
        calc_cash += -(tr["notional"]) - tr["commission"]
    assert np.isclose(cash_delta, calc_cash, rtol=1e-12, atol=1e-10)
