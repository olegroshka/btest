# QuantDSL — Agent Reference

> Authoritative cheat sheet for the DSL agent. Read this in full before writing any strategy.
> All class names, fields, and defaults are exact — verified from source.

---

## 1. Canonical Imports

```python
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe, HasHistory, MinPrice, MinDollarADV
from quantdsl_backtest.dsl.factors import (
    ReturnFactor, VolatilityFactor, WinsorizedFactor, RatioFactor,
    OvernightReturnFactor, IntradayReturnFactor, FiboRetraceFactor,
    ExternalFactor, FieldFactor,                 # NEW: external / auxiliary data
)
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank, CrossSectionAggregate,
    Quantile, MaskFromBoolean,
    And, Or, Not,
    LessEqual, GreaterEqual, Less, Greater,
    NotNull,
    EWMMean, RollingMean, RollingStd, Diff, PctChange, ZScoreRolling,
    RiskMultiplierFromZ, TimeSeries,
)
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio, Book, TopN, BottomN, MaskSelector, EqualWeight,
    SectorNeutral, TurnoverLimit,
    TimingPortfolio,                              # NEW: single-instrument timing
)
from quantdsl_backtest.dsl.execution import (
    Execution, OrderPolicy, LatencyModel,
    PowerLawSlippageModel, VolumeParticipation, LimitOrderBookModel,
)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import (
    BacktestConfig, MarginConfig, RiskChecks, DrawdownPolicy, Reporting,
)
from quantdsl_backtest.dsl.transforms import CleaningTransform
from quantdsl_backtest.engine.analytics.types import StrategyAnalyticsConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest
```

---

## 2. Factors

All factors inherit from `FactorNode`. Every factor MUST have a `name: str`.

| Class | Key Fields | Notes |
|-------|-----------|-------|
| `ReturnFactor` | `name, field="close", lookback=1, method="simple"\|"log"` | Most common momentum factor |
| `VolatilityFactor` | `name, field="close", lookback=20, method="realized"\|"stdev", annualize=True, min_periods=None` | For vol-scaling |
| `WinsorizedFactor` | `name, base: FactorNode, z=3.0` | Wraps another factor, clips at ±z·std |
| `RatioFactor` | `name, numerator: FactorNode, denominator: FactorNode` | Risk-adjusted factors |
| `OvernightReturnFactor` | `name, field_open="open", field_close="close", lookback=20, method="log"` | |
| `IntradayReturnFactor` | `name, field_open="open", field_close="close", lookback=20, method="log"` | |
| `FiboRetraceFactor` | `name, field_high="high", field_low="low", lookback=50, level=0.618` | |
| `ExternalFactor` | `name, path: str, column: str\|None=None` | **NEW** — pre-computed ML outputs (e.g. TKAN pickle) |
| `FieldFactor` | `name, field: str` | **NEW** — expose auxiliary data column as a factor (e.g. implied vol) |

**Example — risk-adjusted momentum:**
```python
mom = ReturnFactor(name="mom_126", field="close", lookback=126, method="log")
vol = VolatilityFactor(name="vol_63", field="close", lookback=63, annualize=True)
mom_adj = WinsorizedFactor(
    name="mom_adj",
    base=RatioFactor(name="mom_adj_raw", numerator=mom, denominator=vol),
    z=2.0,
)
```

---

## 3. Signals

All signal nodes accept an optional `name: str`. 
`Expr` means: a string (factor/signal name), another SignalNode, or a numeric constant.

### Cross-sectional
| Class | Key Fields | Notes |
|-------|-----------|-------|
| `CrossSectionRank` | `factor_name, mask_name=None, method="percentile"\|"zscore", name=None` | Returns 0–1 rank per date |
| `Quantile` | `factor_name, q: float, within_mask=None, name=None` | Scalar threshold at quantile q |
| `CrossSectionAggregate` | `source, op="mean"\|"median"\|"sum"\|"min"\|"max", mask_name=None, name=None` | Regime indicators |

### Boolean / masks
| Class | Key Fields | Notes |
|-------|-----------|-------|
| `MaskFromBoolean` | `expr: Expr, name=None` | Stores a True/False mask |
| `And` | `left: Expr, right: Expr, name=None` | Logical AND — nest for 3+ conditions |
| `Or` | `left: Expr, right: Expr, name=None` | Logical OR |
| `Not` | `expr: Expr, name=None` | Logical NOT |
| `NotNull` | `factor_name: str, name=None` | True where factor is non-NaN |

### Comparisons
| Class | Usage |
|-------|-------|
| `LessEqual` | `left <= right` |
| `GreaterEqual` | `left >= right` |
| `Less` | `left < right` |
| `Greater` | `left > right` |

### Time-series transforms (applied per-instrument)
| Class | Key Fields |
|-------|-----------|
| `EWMMean` | `base: Expr, span: int, min_periods=1, adjust=False` |
| `RollingMean` | `base: Expr, window: int, min_periods=1` |
| `RollingStd` | `base: Expr, window: int, min_periods=1` |
| `Diff` | `base: Expr, periods=1` |
| `PctChange` | `base: Expr, periods=1` |
| `ZScoreRolling` | `base: Expr, window: int, min_periods=1` |
| `RiskMultiplierFromZ` | `z: Expr, max_z=2.5` → maps z-score to [0,1] scalar |
| `TimeSeries` | `source: str (e.g. "fred://BAMLH0A0HYM2"), field="close"` |

**Example — composite long signal with regime filter:**
```python
rank = CrossSectionRank(factor_name="mom_126", name="rank")

# Regime: average cross-sectional momentum is positive (risk-on)
avg_mom = CrossSectionAggregate(source="mom_126", op="mean", name="avg_mom")
risk_on = MaskFromBoolean(name="risk_on", expr=GreaterEqual(left="avg_mom", right=0.0))

# Individual validity
valid = MaskFromBoolean(name="valid", expr=NotNull("mom_126"))

# Top decile AND valid AND risk-on
long_mask = MaskFromBoolean(
    name="long_candidates",
    expr=And(
        left="valid",
        right=And(
            left="risk_on",
            right=LessEqual(left=Quantile(factor_name="rank", q=0.9), right="rank"),
        ),
    ),
)
```

---

## 4. Universe

```python
Universe(
    name="SP500",                          # required
    filters=[                              # optional
        HasHistory(min_days=252),
        MinPrice(min_price=5.0),
        MinDollarADV(min_dollar_adv=5_000_000.0),
    ],
    static_instruments=None,               # or list of tickers to hard-code
)
```

---

## 5. Portfolio

### LongShortPortfolio (cross-sectional, multi-instrument)

```python
LongShortPortfolio(
    long_book=Book(
        name="long",
        selector=TopN(factor_name="rank", n=50, mask_name="long_candidates"),
        weighting=EqualWeight(),
    ),
    short_book=Book(
        name="short",
        selector=BottomN(factor_name="rank", n=50, mask_name="short_candidates"),
        weighting=EqualWeight(),
    ),
    rebalance_frequency="1d",              # "1d" | "1w" | "1m"
    rebalance_at="market_close",           # default
    signal_delay_bars=0,                   # 0 = same-day signal
    target_gross_leverage=2.0,             # 2x gross (1x long + 1x short)
    target_net_exposure=0.0,               # dollar-neutral
    max_abs_weight_per_name=0.03,          # 3% max single name
    sector_neutral=None,                   # or SectorNeutral(sector_field="sector")
    turnover_limit=None,                   # or TurnoverLimit(window_bars=5, max_fraction=0.3)
)
```

**Top-N with mask** — preferred over large TopN without mask:
```python
# TopN picks from full universe BUT respects mask when fill_from_unmasked=False
TopN(factor_name="rank", n=50, mask_name="long_candidates", fill_from_unmasked=False)
```

**MaskSelector** — select instruments where a named boolean signal is True (NEW):
```python
# Alternative to TopN/BottomN — use when you want all instruments passing a mask
Book(name="long", selector=MaskSelector(signal_name="entry_signal"), weighting=EqualWeight())
```

### TimingPortfolio (single-instrument market timing) — NEW

Use when the strategy is a **binary enter/exit decision** on one instrument, not a cross-sectional portfolio.

```python
TimingPortfolio(
    signal_name="entry_signal",   # name of a boolean signal: True=long, False=flat
    instrument="CACT",            # ticker to trade (must be in the universe data)
    rebalance_frequency="1d",     # "1d" | "1w" | "1m"
    rebalance_at="market_close",  # default
    signal_delay_bars=1,          # 1 = act on next bar (default)
    target_leverage=1.0,          # 100% invested when signal is on
)
```

**Rules for `TimingPortfolio`:**
- `signal_name` must be a key in `Strategy.signals` with a boolean (True/False) value per date
- `instrument` must appear in the data loaded by `DataConfig`
- The timing runner **does not use** `LongShortPortfolio`-style books or `Book` objects
- `signal_delay_bars=1` means: signal generated at close[T] → position opens at close[T+1]
- `strategy.portfolio` field accepts `Union[LongShortPortfolio, TimingPortfolio]`

---

## 6. Execution

```python
Execution(
    order_policy=OrderPolicy(
        default_order_type="MOC",          # "MKT" | "MOC" | "LIMIT"
        time_in_force="DAY",
    ),
    latency=LatencyModel(
        signal_to_order_delay_bars=0,
        market_latency_ms=0,
    ),
    slippage=PowerLawSlippageModel(
        base_bps=1.0,                      # minimum slippage in bps
        k=20.0,                            # participation scaling coefficient
        exponent=0.5,
    ),
    volume_limits=VolumeParticipation(
        max_participation=0.1,             # max 10% of bar volume
        mode="proportional",
    ),
)
```

**Minimal execution (no slippage, no volume cap):**
```python
Execution(
    order_policy=OrderPolicy(),
    latency=LatencyModel(),
    slippage=PowerLawSlippageModel(base_bps=1.0, k=0.0),
    volume_limits=VolumeParticipation(max_participation=1.0),
)
```

---

## 7. Costs

```python
Costs(
    commission=Commission(type="bps_notional", amount=1.0),  # 1bp per trade
    borrow=BorrowCost(default_annual_rate=0.005),             # 50bps short borrow
    financing=FinancingCost(base_rate_curve="SOFR", spread_bps=0.0),
    fees=StaticFees(nav_fee_annual=0.0, perf_fee_fraction=0.0),
)
```

---

## 8. BacktestConfig

```python
BacktestConfig(
    engine="event_driven",                 # always use this
    cash_initial=1_000_000.0,
    risk_checks=RiskChecks(
        max_gross_leverage=3.0,
        drawdown=DrawdownPolicy(
            mode="soft_scale",             # "none" | "hard_kill" | "soft_scale"
            start=0.10,                    # begin de-risking at 10% DD
            full=0.35,                     # fully flat at 35% DD
            curve="linear",
        ),
    ),
    reporting=Reporting(
        output_dir="outputs/my_strategy",
        store_trades=True,
        store_positions=True,
        strategyAnalytics=StrategyAnalyticsConfig(title="My Strategy"),
    ),
    extra={"hold_when_no_targets": True},  # carry positions when selection is empty
)
```

---

## 9. DataConfig

```python
DataConfig(
    source="parquet://equities/sp500_daily",   # parquet file or folder
    calendar="XNYS",                            # NYSE calendar
    frequency="1d",                             # "1d", "5m", "15m", "1h"
    start="2015-01-01",
    end="2025-01-01",
    price_adjustment="split_dividend",          # default
)
```

### Supported data sources

| URI scheme | Description | Example |
|---|---|---|
| `parquet://path` | Local parquet file or folder | `parquet://equities/sp500_daily` |
| `csv://path.csv` | Local CSV file (single or multi-instrument long format) | `csv://data/c40_ohlcv.csv` |
| `sfera://schema/table` | Sfera PostgreSQL — price data or time-series | `sfera://bbgidx/index_prices` |
| `yf://TICKER` | Yahoo Finance live pull | `yf://SPY` |
| `fred://SERIES` | FRED macro series | `fred://BAMLH0A0HYM2` |

**Sfera notes:**
- One URI scheme for everything — `sfera://schema/table`
- Use `kind="market_bars"` (default) for OHLCV tables → returns MarketBarsBundle
- Use `kind="timeseries"` for macro/event/yield tables → returns TimeSeriesBundle
- `sfera-bars://` still works as a backward-compat alias for `kind="market_bars"`
- Available schemas: `bbgidx` (index prices/vol), `mxbdprc` (bond data), `ecocal`, `mxent`

**CSV notes:**
- Long-format (multiple tickers per file): needs a `ticker` column (auto-detected)
- Single-instrument file: instrument name = filename stem (e.g. `c40_ohlcv.csv` → `c40_ohlcv`)
- Override columns via query params: `csv://data/file.csv?ticker_col=symbol&date_col=trade_date`

---

## 10. Full Strategy Assembly

```python
strategy = Strategy(
    name="my_strategy",          # used for output folder naming
    data=data,
    universe=universe,
    factors={"mom_126": mom_126, "vol_63": vol_63},   # dict[str, FactorNode]
    signals={                                           # dict[str, SignalNode]
        "rank": rank,
        "long_candidates": long_candidates,
        "short_candidates": short_candidates,
    },
    portfolio=portfolio,
    execution=execution,
    costs=costs,
    backtest=bt,
)
result = run_backtest(strategy)
print(result.summary())
```

**Rules:**
- Factor dict key MUST match the `name` field of the FactorNode
- Signal dict key MUST match the `name` field of the SignalNode
- `factors` and `signals` dicts are separate — don't mix them
- `CrossSectionRank.factor_name` must be a key in the `factors` dict
- `Quantile.factor_name` refers to a **signal** name (typically the rank signal), not a raw factor

---

## 11. Run Command

```powershell
# From workspace root:
cd "c:\Personal\Business & Investments\Python codes\btest"
uv run python strategies\my_strategy.py

# Or activate btest venv first, then:
python strategies\my_strategy.py
```

Output goes to `btest/outputs/<strategy_name>/`.

---

## 12. Common Patterns

### Pure long-only (no shorts)
```python
portfolio = LongShortPortfolio(
    long_book=Book(name="long", selector=TopN(factor_name="rank", n=20), weighting=EqualWeight()),
    short_book=Book(name="short", selector=BottomN(factor_name="rank", n=0), weighting=EqualWeight()),
    rebalance_frequency="1w",
    target_gross_leverage=1.0,
    target_net_exposure=1.0,
)
```

### Momentum + volatility filter (quality momentum)
```python
# Factor: momentum / vol = risk-adjusted momentum
mom = ReturnFactor(name="mom_126", field="close", lookback=126, method="log")
vol = VolatilityFactor(name="vol_63", lookback=63, annualize=True)
mom_adj = WinsorizedFactor(name="mom_adj", base=RatioFactor(name="mom_adj_raw", numerator=mom, denominator=vol), z=2.0)
rank = CrossSectionRank(factor_name="mom_adj", name="rank")
```

### Regime filter (trend-following with SPY MA filter)
```python
# When avg cross-sectional momentum is negative → go flat
avg_mom = CrossSectionAggregate(source="mom_126", op="mean", name="avg_mom")
regime_ok = MaskFromBoolean(name="regime_ok", expr=GreaterEqual(left="avg_mom", right=0.0))
long_candidates = MaskFromBoolean(
    name="long_candidates",
    expr=And(left="regime_ok", right=LessEqual(left=Quantile(factor_name="rank", q=0.8), right="rank")),
)
```

### Mean reversion (low RSI / low short-term return)
```python
# Short-term reversal: short recent winners, long recent losers
ret_5 = ReturnFactor(name="ret_5", field="close", lookback=5, method="log")
rank_rev = CrossSectionRank(factor_name="ret_5", name="rank_rev")
# Long = bottom decile of 5d return (most oversold)
long_candidates = MaskFromBoolean(
    name="long_candidates",
    expr=LessEqual(left="rank_rev", right=Quantile(factor_name="rank_rev", q=0.1)),
)
```

### Single-instrument market timing with ML signal (NEW)

Use `ExternalFactor` for ML model outputs, `FieldFactor` for auxiliary data fields,
and `TimingPortfolio` to describe the long/flat entry logic.  A `TimingRunner`
in the notebook interprets these DSL objects directly.

```python
# 1. Data: sfera with implied-vol field in addition to OHLCV
data = DataConfig(
    source="sfera://bbgidx/index_prices",
    calendar="XPAR",
    frequency="1d",
    start="2015-01-01",
    end="2025-12-31",
    fields=["open", "high", "low", "close", "volume", "3m_50d_ivol"],
)

# 2. Universe: single instrument (hard-coded)
universe = Universe(name="CAC_TR", static_instruments=["CACT"])

# 3. Factors
tkan_pred = ExternalFactor(
    name="tkan_pred",
    path="research/Index Directional/tkan/v3/weights/pred_cache.pkl",
    column=None,   # file is a plain pd.Series indexed by date
)
ivol = FieldFactor(name="ivol_3m", field="3m_50d_ivol")

# 4. Signals: IVol z-score regime gate + TKAN cumulative prediction
ivol_z = ZScoreRolling(base="ivol_3m", window=126, name="ivol_z")
tkan_positive = MaskFromBoolean(
    name="tkan_positive",
    expr=GreaterEqual(left="tkan_pred", right=0.0),
)
regime_ok = MaskFromBoolean(
    name="regime_ok",
    expr=Less(left="ivol_z", right=1.0),   # low implied-vol regime
)
entry_signal = MaskFromBoolean(
    name="entry_signal",
    expr=And(left="tkan_positive", right="regime_ok"),
)

# 5. Portfolio: timing, not cross-sectional
portfolio = TimingPortfolio(
    signal_name="entry_signal",
    instrument="CACT",
    rebalance_frequency="1d",
    signal_delay_bars=1,
    target_leverage=1.0,
)
```

**`ExternalFactor` file formats and loading:**
```python
# Pickle (pd.Series indexed by DatetimeIndex) — most common for TKAN cache:
series = pd.read_pickle(factor.path)           # Series directly

# Pickle (pd.DataFrame with a named column):
df = pd.read_pickle(factor.path)
series = df[factor.column]

# Parquet / CSV: same pattern but use pd.read_parquet / pd.read_csv
```

**`FieldFactor` loading:**
```python
# Assumes the DataConfig loaded a DataFrame with field as a column:
series = data_df[factor.field]     # e.g. data_df["3m_50d_ivol"]
```

---

## 13. Parameter Sweeps — Best Practices

### Data caching (built-in, automatic)
`engine/data_loader.py` has a module-level `_DATA_CACHE` dict keyed on `(source, start, end)`.
Identical strategies in the same process share one parquet load automatically.
Call `from quantdsl_backtest.engine.data_loader import clear_data_cache` to free it.

### Parallelization with ProcessPoolExecutor
Each worker process gets its own `_DATA_CACHE`, so data is loaded once per worker.
Worker functions **must be at module level** (not lambdas or nested defs) to be picklable.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

N_WORKERS = max(1, os.cpu_count() // 2)

def _run_single(args: tuple) -> dict:          # module-level — picklable
    yw, n = args
    import sys; sys.path.insert(0, "src")      # Windows spawn needs this
    # load strategy + run_backtest here ...

grid    = [(yw, n) for yw in YIELD_WEIGHTS for n in LONG_NS]
rows    = {}
with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(_run_single, a): a for a in grid}
    for fut in as_completed(futures):
        rows[futures[fut]] = fut.result()
```

### ExternalFactor with on-the-fly composite (no temp files)
Avoid temp parquets for parameter sweeps — use the `loader` parameter instead:

```python
def _make_loader(yw: float, growth_path: str):
    def _loader(yield_df):                      # obj = loaded yield_rank.parquet
        import pandas as pd
        growth = pd.read_parquet(growth_path)
        return yw * yield_df + (1 - yw) * growth
    return _loader

composite_raw = ExternalFactor(
    name="composite_raw",
    path="data/yield_rank.parquet",
    per_instrument=True,
    loader=_make_loader(yield_weight, str(DATA_DIR / "growth_rank.parquet")),
)
```

### GPU / further acceleration
- The Polars-accelerated path in `SignalEngine` already vectorizes cross-sectional ranking.
- The event loop (`_run_backtest_event_driven`) is CPU-bound; GPU is not beneficial here.
- For 10–100 run sweeps: `ProcessPoolExecutor` with `N_WORKERS = cpu_count // 2` is optimal.
- For 1000+ run sweeps: consider Optuna with `n_jobs=-1` (uses joblib internally).

