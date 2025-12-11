quantdsl-backtest
==================

Experimental internal DSL for designing and backtesting systematic trading strategies. The DSL lets you declaratively define data sources, universes, factors, signals, portfolio construction, execution models, and costs — then run an event‑driven daily backtest to produce results and metrics.

This repository also includes a fully worked example: a cross‑sectional momentum long/short strategy on the S&P 500.


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
