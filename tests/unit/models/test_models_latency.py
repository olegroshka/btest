import pandas as pd

from quantdsl_backtest.models.latency import build_latency_model, apply_latency_to_weights


def test_apply_latency_zero_no_change():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.0, -0.1, 0.1]}, index=idx)
    lat = build_latency_model(type("Cfg", (), {"signal_to_order_delay_bars": 0, "market_latency_ms": 0}))
    out = apply_latency_to_weights(df, lat)
    pd.testing.assert_frame_equal(out, df)


def test_apply_latency_shift_two_bars():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, index=idx, dtype=float)
    lat = build_latency_model(type("Cfg", (), {"signal_to_order_delay_bars": 2, "market_latency_ms": 0}))
    out = apply_latency_to_weights(df, lat)
    # Expect shift downwards by 2 rows (later dates take earlier weights), introducing NaNs at top
    expected = df.shift(2)
    pd.testing.assert_frame_equal(out, expected)
