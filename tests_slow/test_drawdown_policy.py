import numpy as np
import pandas as pd

from quantdsl_backtest.examples.lagging_indecies import build_strategy
from quantdsl_backtest.dsl.backtest_config import DrawdownPolicy
from quantdsl_backtest.engine.backtest_runner import run_backtest


def _assert_all_zero_weights(res):
    # All weights should be (close to) zero across the whole period
    w = res.weights.fillna(0.0).abs().values
    assert np.nanmax(w) < 1e-8


def _assert_flat_exposure(res):
    # Gross exposure should be zero (allow tiny float noise)
    gx = res.gross_exposure.fillna(0.0).abs().values
    assert np.nanmax(gx) < 1e-8


def test_drawdown_policy_soft_scale_derisks_and_stays_flat():
    """
    With soft_scale and (start≈0, full≈0), the strategy should de-risk to 0 exposure
    as soon as there is any drawdown, and remain flat thereafter.
    This validates that soft-scaling is applied before rebalance.
    """
    strat = build_strategy()

    # Derisk on any non-zero drawdown
    strat.backtest.risk_checks.drawdown = DrawdownPolicy(
        mode="soft_scale",
        start=0.0,
        full=1e-6,
        curve="linear",
    )

    # Ensure empty selection behavior does not accidentally carry positions.
    # We want liquidation on empty targets for this test.
    try:
        if not hasattr(strat.backtest, "extra") or strat.backtest.extra is None:
            strat.backtest.extra = {}
        strat.backtest.extra["hold_when_no_targets"] = False
    except Exception:
        pass
    try:
        # Also allow overriding via risk_checks.extra for back-compat
        if not hasattr(strat.backtest.risk_checks, "extra") or strat.backtest.risk_checks.extra is None:
            strat.backtest.risk_checks.extra = {}
        strat.backtest.risk_checks.extra["hold_when_no_targets"] = False
    except Exception:
        pass

    res = run_backtest(strat)
    # Sum absolute weights per day
    abs_w_sum = res.weights.fillna(0.0).abs().sum(axis=1)
    # Find first time we actually take risk (non-zero weights)
    started_mask = abs_w_sum > 1e-6
    assert started_mask.any(), "Strategy never took any exposure"
    first_started_ts = started_mask.idxmax()

    # Find first flat day AFTER we've taken exposure at least once
    post_started = abs_w_sum.loc[first_started_ts:]
    flat_after_mask = post_started <= 1e-9
    assert flat_after_mask.any(), "Strategy never de-risked to fully flat under soft_scale"
    first_flat_after_ts = flat_after_mask.idxmax()

    # After first flat day (post-exposure), remain flat thereafter
    assert (abs_w_sum.loc[first_flat_after_ts:] <= 1e-9).all()


def test_drawdown_policy_hard_kill_immediate():
    """
    With hard_kill threshold=0, the strategy should liquidate and halt immediately.
    Verify positions/weights remain zero.
    """
    strat = build_strategy()
    strat.backtest.risk_checks.drawdown = DrawdownPolicy(
        mode="hard_kill",
        threshold=0.0,
    )

    res = run_backtest(strat)
    _assert_all_zero_weights(res)
    _assert_flat_exposure(res)
