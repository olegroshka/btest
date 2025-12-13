quantdsl-backtest
==================

Experimental internal DSL for designing and backtesting systematic trading strategies. The DSL lets you declaratively define data sources, universes, factors, signals, portfolio construction, execution models, and costs — then run an event‑driven daily backtest to produce results and metrics.

This repository also includes a fully worked example: a cross‑sectional momentum long/short strategy on the S&P 500.


TL;DR: a tiny DSL backtest
--------------------------
A minimal end‑to‑end example showing the core idea. It builds a simple cross‑sectional momentum long/short and runs a daily backtest.

Note: this assumes you’ve prepared the S&P 500 parquet dataset at `equities/sp500_daily` (see “Get the data” below).

```
from quantdsl_backtest.dsl.strategy import Strategy
from quantdsl_backtest.dsl.data_config import DataConfig
from quantdsl_backtest.dsl.universe import Universe
from quantdsl_backtest.dsl.factors import ReturnFactor
from quantdsl_backtest.dsl.signals import (
    CrossSectionRank, Quantile, LessEqual, MaskFromBoolean,
)
from quantdsl_backtest.dsl.portfolio import (
    LongShortPortfolio, Book, TopN, BottomN, EqualWeight,
)
from quantdsl_backtest.dsl.execution import (
    Execution, OrderPolicy, LatencyModel, PowerLawSlippageModel, VolumeParticipation,
)
from quantdsl_backtest.dsl.costs import Costs, Commission, BorrowCost, FinancingCost, StaticFees
from quantdsl_backtest.dsl.backtest_config import BacktestConfig
from quantdsl_backtest.engine.backtest_runner import run_backtest

# 1) Data & universe
data = DataConfig(
    source="parquet://equities/sp500_daily", calendar="XNYS", frequency="1d",
    start="2015-01-01", end="2025-01-01",
)
universe = Universe(name="SP500")

# 2) Factor: 6‑month momentum on close prices
mom_126 = ReturnFactor(name="mom_126", field="close", lookback=126, method="log")

# 3) Signals: rank momentum; pick top/bottom 50 names
rank = CrossSectionRank(factor_name="mom_126", name="rank")
long_candidates = MaskFromBoolean(
    name="long_candidates",
    expr=LessEqual(left=Quantile(factor_name="rank", q=0.9), right="rank"),
)
short_candidates = MaskFromBoolean(
    name="short_candidates",
    expr=LessEqual(left="rank", right=Quantile(factor_name="rank", q=0.1)),
)

factors = {"mom_126": mom_126}
signals = {
    "rank": rank,
    "long_candidates": long_candidates,
    "short_candidates": short_candidates,
}

# 4) Portfolio: equal‑weight long/short, daily rebalance
portfolio = LongShortPortfolio(
    long_book=Book(name="long", selector=TopN(factor_name="rank", n=50), weighting=EqualWeight()),
    short_book=Book(name="short", selector=BottomN(factor_name="rank", n=50), weighting=EqualWeight()),
    rebalance_frequency="1d",
)

# 5) Execution & costs (kept simple)
execution = Execution(
    order_policy=OrderPolicy(),
    latency=LatencyModel(),
    slippage=PowerLawSlippageModel(base_bps=1.0, k=0.0),
    volume_limits=VolumeParticipation(max_participation=1.0),  # no strict cap
)
costs = Costs(
    commission=Commission(type="bps_notional", amount=1.0),  # 1bp per trade
    borrow=BorrowCost(), financing=FinancingCost(), fees=StaticFees(),
)

# 6) Backtest runtime
bt = BacktestConfig(engine="event_driven", cash_initial=1_000_000)

# 7) Compose and run
strategy = Strategy(
    name="tiny_momentum_ls",
    data=data,
    universe=universe,
    factors=factors,
    signals=signals,
    portfolio=portfolio,
    execution=execution,
    costs=costs,
    backtest=bt,
)

result = run_backtest(strategy)
print(result.summary())
```


Key features
------------
- Declarative Strategy DSL: `Data`, `Universe`, `Factors`, `Signals`, `Portfolio`, `Execution`, `Costs`, `Backtest`.
- Event‑driven daily backtester with mark‑to‑market, carry/financing/fees, and rebalancing.
- Pluggable data adapters (parquet expected by default for the example).
- Example strategy: momentum long/short on S&P 500 with sector neutrality and turnover limits.


Requirements
------------
- Python 3.11.x (strict)
- OS: Windows, macOS, or Linux
- Core deps (managed via `pyproject.toml`): pandas, numpy, vectorbt, yfinance, plotly, pyarrow, lxml


Install (recommended)
---------------------
Using pip in a virtual environment:

1) Create and activate a virtual environment

- PowerShell (Windows):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install the package (editable) with dependencies
```
pip install -e .
```

Alternatively, using uv (if you prefer):
```
uv sync
```


Get the data (S&P 500 daily OHLCV to parquet)
---------------------------------------------
The example strategy expects a parquet dataset at `equities/sp500_daily` in a long format (date, ticker, ohlcv[, sector]). A helper script is provided to download and prepare data using Yahoo Finance via vectorbt.

PowerShell (Windows):
```
python scripts/download_sp500_to_parquet.py --start 2015-01-01 --end 2025-01-01 --out equities/sp500_daily
```

macOS/Linux:
```
python3 scripts/download_sp500_to_parquet.py --start 2015-01-01 --end 2025-01-01 --out equities/sp500_daily
```

Notes:
- If you want to avoid the Wikipedia/lxml dependency or pin the universe, provide your own tickers CSV (columns: `ticker[,sector]`):
```
python scripts/download_sp500_to_parquet.py --tickers-csv data/sp500_tickers.csv --out equities/sp500_daily
```
- The example and defaults are aligned with `DataConfig(source="parquet://equities/sp500_daily")`.


Quickstart: run the sample strategy
-----------------------------------
After installing dependencies and preparing data, run the example module:

- PowerShell (Windows):
```
python -m quantdsl_backtest.examples.momentum_long_short_sp500
```

- macOS/Linux:
```
python3 -m quantdsl_backtest.examples.momentum_long_short_sp500
```

What it does:
- Builds the DSL `Strategy` object (momentum long/short, sector‑neutral, top/bottom N selection, equal weights, turnover cap).
- Runs the event‑driven backtest via `engine.backtest_runner.run_backtest(strategy)`.
- Prints a concise summary (total return, Sharpe, max drawdown, turnover). You can extend plotting/export from the returned `BacktestResult`.


Engines: vectorized vs event‑driven
-----------------------------------
There are two interchangeable backtest engines. Choose them via `BacktestConfig(engine=...)`:

- `"vectorized"` — wrapper around `vectorbt` using `Portfolio.from_orders` with `size_type="targetpercent"`.
  - Pros: very fast for large universes/long histories; great for parameter sweeps and research iterations.
  - Cons: fewer features; cannot enforce per‑bar volume participation limits (< 100%) with target‑percent sizing; some cost models are approximations.
- `"event_driven"` — custom daily event loop.
  - Pros: full feature set; fine‑grained execution control (volume participation, min notional, power‑law slippage, per‑share/nominal commissions), easier to debug and extend.
  - Cons: slower than the vectorized path.

Feature differences (brief):
- Sizing/rebalance
  - Both build daily target weights from signals/portfolio; vectorized applies them via vectorbt; event‑driven applies them trade‑by‑trade.
- Costs
  - Commissions: both support bps‑of‑notional; per‑share commissions in vectorized are emulated via a two‑pass run (first pass to infer order sizes, second pass to apply costs) or an approximate single‑pass option.
  - Slippage: event‑driven supports power‑law slippage by participation; vectorized can approximate it (two‑pass exact per bar from pass‑1 orders, or approximate single‑pass).
- Volume limits
  - Event‑driven enforces max participation (< 100%) and min fill notional per instrument.
  - Vectorized cannot enforce participation < 100% with target‑percent sizing; when such limits are configured, it automatically falls back to the event‑driven engine for parity.

When to use which:
- Use `vectorized` for speed when you don’t require strict intra‑bar execution constraints (e.g., no volume participation < 100%) and you’re OK with approximate frictions or the two‑pass cost model.
- Use `event_driven` when execution details matter (volume constraints, custom slippage/fees, more realistic fills) or for debugging/validation.

How to select the engine:
```
from quantdsl_backtest.dsl import BacktestConfig

backtest_cfg = BacktestConfig(
    cash_initial=1_000_000,
    engine="vectorized",  # or "event_driven"
)
```

Vectorized engine tuning (optional):
- You can influence the vectorized path via `BacktestConfig.extra["vectorized_engine"]` or environment variables.
  - Keys: `sparse_cost_mats` (default True), `numpy_fast_path` (True), `approx_single_pass` (False), `enable_caching` (False), `timing` (False).
  - Example:
    ```
    backtest_cfg = BacktestConfig(
        cash_initial=1_000_000,
        engine="vectorized",
        extra={
            "vectorized_engine": {
                "approx_single_pass": True,
                "enable_caching": True,
            }
        },
    )
    ```


Build a wheel (optional)
------------------------
If you want a distributable wheel/sdist:
```
pip install build
python -m build
```
Artifacts will be in `dist/`.


Project layout
--------------
- `src/quantdsl_backtest/dsl/` — DSL primitives: data config, universe, factors, signals, portfolio, execution, costs, backtest config.
- `src/quantdsl_backtest/engine/` — Backtest orchestration: data loading, factor/signal engines, portfolio/execution engines, accounting, results.
- `src/quantdsl_backtest/examples/` — Example strategies (S&P 500 momentum sample).
- `scripts/` — Utilities for data acquisition and preparation.
- `equities/sp500_daily/` — Expected local parquet dataset path for the sample (created by the script).
- `tests/` — Unit/E2E tests for factors, signals, execution, and the example.


Run tests
---------
With the virtual environment active:
```
pytest -q
```


Run slow integration tests
--------------------------
The slower end-to-end/integration tests live under `tests_slow/`.

- Run only the slow test suite:
```
uv run pytest tests_slow -q
```

- Run slow tests with live logging to the console (INFO level):
```
uv run pytest -q tests_slow -m slow -o log_cli=true --log-cli-level=INFO
```


Troubleshooting
---------------
- Import errors: ensure the venv is active and you installed with `pip install -e .`.
- Python version: must be 3.11.x as specified in `pyproject.toml`.
- Missing data: run the download script and verify output under `equities/sp500_daily`.
- Yahoo rate limiting: re‑run the script later or reduce the ticker set via `--tickers-csv`.


License
-------
This project is licensed under the terms of the MIT License (see `LICENSE`).
