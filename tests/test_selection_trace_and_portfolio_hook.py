import pandas as pd

from quantdsl_backtest.engine.portfolio_engine import compute_target_weights_for_date
from quantdsl_backtest.engine.analytics.selection_trace import SelectionTraceCollector
from quantdsl_backtest.dsl.portfolio import LongShortPortfolio, Book, TopN, BottomN, EqualWeight, TurnoverLimit


def test_selection_trace_records_long_and_short():
    # Build simple signals frame: rank values for two dates and three instruments
    idx = pd.date_range("2021-01-01", periods=3, freq="B")
    cols = ["AAA", "BBB", "CCC"]
    rank = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1],
            [0.5, 0.4, 0.0],
        ],
        index=idx,
        columns=cols,
    )

    signals = {"rank": rank}

    long_book = Book(name="L", selector=TopN(factor_name="rank", n=1), weighting=EqualWeight())
    short_book = Book(name="S", selector=BottomN(factor_name="rank", n=1), weighting=EqualWeight())
    portfolio = LongShortPortfolio(
        long_book=long_book,
        short_book=short_book,
        rebalance_frequency="1d",
        signal_delay_bars=0,
        target_gross_leverage=1.0,
        target_net_exposure=0.0,
        max_abs_weight_per_name=1.0,
        turnover_limit=TurnoverLimit(window_bars=1, max_fraction=1.0),
    )

    prev_w = pd.Series(0.0, index=cols)
    collector = SelectionTraceCollector()

    # Use the second date
    dt = idx[1]
    _ = compute_target_weights_for_date(
        date=dt,
        portfolio=portfolio,
        signals=signals,
        prev_weights=prev_w,
        collector=collector,
    )

    df = collector.finalize()
    # Expect one row for long and one for short
    assert df.shape[0] == 2
    assert set(df["book"]) == {"long", "short"}
    # The selected tickers should be the top and bottom by rank on sig_date
    sig_date = df["sig_date"].iloc[0]
    row = rank.loc[sig_date]
    assert df[df["book"] == "long"]["instrument"].item() == row.idxmax()
    assert df[df["book"] == "short"]["instrument"].item() == row.idxmin()
