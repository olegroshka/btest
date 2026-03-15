# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quantdsl-backtest** is an event-driven backtesting framework for systematic trading strategies using a declarative Python DSL. The main package is `quantdsl_backtest` under `src/`.

Requires **Python 3.11.x** exactly. Use `uv` as the package manager.

## Commands

### Setup
```bash
uv sync --extra platform --extra dev --extra e2e
```

### Running Tests
```bash
# Default: unit + slow (automated, no server required)
uv run python scripts/run_tests.py

# Specific suites
uv run python scripts/run_tests.py --unit          # fast unit tests only
uv run python scripts/run_tests.py --slow          # slow integration tests
uv run python scripts/run_tests.py --platform      # API e2e + Playwright
uv run python scripts/run_tests.py --smoke         # live-server smoke (manual)
uv run python scripts/run_tests.py --all           # everything

# Parallel execution
uv run python scripts/run_tests.py --parallel

# Pass pytest args through
uv run python scripts/run_tests.py -- -k signal_engine
uv run python scripts/run_tests.py -- --maxfail=1 -x

# Run a single test directly
uv run pytest tests/unit/engine/test_signal_engine.py -q
uv run pytest tests/unit/ -k "test_name" -q
```

Default `testpaths` in pytest is `tests/unit`. Slow tests live in `tests_slow/` and must be run explicitly. Test logs are written to `.test_logs/`.

### Platform Server
```bash
uv run python scripts/run_platform_ui.py           # port 8000
uv run python scripts/run_platform_ui.py --port 8001
# PowerShell wrappers:
.\scripts\run_platform_ui.ps1
.\scripts\stop_platform_ui.ps1
```

### Frontend (React/Vite)
```bash
cd frontend
npm install
npm run dev      # dev server (proxies to FastAPI)
npm run build    # builds to frontend/dist/
```

After `npm run build`, copy assets using:
```bash
uv run python scripts/rebuild_platform_ui_assets.py
```
This copies the built assets into `src/quantdsl_backtest/platform_ui/assets_dist/` which is **committed to git** and served by FastAPI at runtime.

### Data Management
```bash
uv run python scripts/download_sp500_to_parquet.py --start 2015-01-01 --end 2025-01-01 --out equities/sp500_daily
uv run python scripts/reset_arctic_cache.py --dry-run
uv run python scripts/reset_arctic_cache.py --backup
```

## Architecture

### Package Layout (`src/quantdsl_backtest/`)

| Module | Role |
|--------|------|
| `dsl/` | Declarative strategy specification (dataclasses only, no logic) |
| `engine/` | Backtesting orchestration, factor/signal/portfolio/execution engines |
| `data/` | Pluggable data loading with ArcticDB caching |
| `models/` | Reusable execution and cost model implementations |
| `platform_api/` | FastAPI REST backend |
| `platform_ui/` | Committed React build artifacts served by FastAPI |
| `utils/` | Logging, time, type utilities |
| `examples/` | Ready-to-run strategy examples |

### DSL → Engine Flow

```
Strategy (DSL dataclasses)
  └─ DataConfig, Universe, factors{}, signals{}, LongShortPortfolio, Execution, Costs, BacktestConfig
       ↓ run_backtest(strategy)
  engine/backtest_runner.py  (main orchestrator ~2600 LOC)
       ↓
  data_loader → factor_engine → signal_engine → portfolio_engine
       ↓
  event-driven OR vectorized execution engine
       ↓
  accounting → analytics → BacktestResult
       ↓
  HTML tearsheets (quantstats) + analytics reports
```

**Two interchangeable engines** (selected via `BacktestConfig(engine="event_driven"|"vectorized")`):
- **Event-driven** (default): custom daily loop, full cost/slippage modeling, trade-by-trade precision
- **Vectorized**: wraps `vectorbt`; fast for parameter sweeps; some cost approximations; auto-falls back to event-driven for unsupported features (e.g. volume participation < 100%)

### DSL Design
All strategy specification is in `dsl/`. These are pure dataclasses — no execution logic. Key composable pieces:
- **Factors**: `ReturnFactor`, `VolatilityFactor`, `FiboRetraceFactor`, `WinsorizedFactor`, `RatioFactor`, …
- **Signals**: `CrossSectionRank`, `Quantile`, `NotNull`, `And`, `LessEqual`, `MaskFromBoolean`, …
- **Portfolio**: `LongShortPortfolio` → composed of `Book` objects with `TopN`/`BottomN`, `EqualWeight`, `SectorNeutral`, `TurnoverLimit`
- **Execution**: `OrderPolicy`, `PowerLawSlippageModel`, `VolumeParticipation`, `LimitOrderBookModel`
- **Costs**: `Commission` (per-share or bps), `BorrowCost`, `FinancingCost`, `StaticFees`
- **Risk**: `BacktestConfig(margin=MarginConfig(...), risk=RiskChecks(...), drawdown_policy=DrawdownPolicy(...))`

### Data Sources
URL scheme dispatched by `data/sources/registry.py`:
- `parquet://local/path` → local parquet files
- `yf://TICKER1,TICKER2` → Yahoo Finance
- `fred://SERIES_ID` → FRED economic data

All sources cached via **ArcticDB** (`data/cache_arctic.py`). Cache is fail-safe — corruption doesn't break runs.

### Platform API (`platform_api/`)
FastAPI app with routes for:
- Strategy discovery (`strategies.py`), backtest runs CRUD (`runs.py`), reports (`reports.py`)
- Data catalog browsing, metadata refresh, download planning
- DSL builder operations (`dsl_builder.py`)
- SSE log streaming (`log_streamer.py`)
- UI bootstrap — serves React assets from `platform_ui/assets_dist/`

Run state stored on disk via `services/run_store.py`. Runs execute via `services/run_worker.py`.

### Frontend (`frontend/`)
React 18 + TypeScript + Vite + AG Grid + Monaco Editor. Main pages:
- **CatalogPage**: data catalog browser
- **RunsPage**: composed of DSL tab (Monaco editor), Run tab (execute + monitor via SSE), Report tab (tearsheet links)
- **DSLBuilderPage**: strategy builder UI

Built artifacts are committed at `src/quantdsl_backtest/platform_ui/assets_dist/` — do not edit files there directly.

### Test Organization
- `tests/unit/` — fast unit tests, mirrors package structure; run by default
- `tests_slow/engine/` — multi-engine consistency, backtest correctness
- `tests_slow/integration/` — full strategy runs, data source integration
- `tests_slow/platform/` — FastAPI endpoint tests + Playwright browser E2E
- `tests_slow/smoke/` — live-server smoke tests (marked `manual`, require running server)

Playwright tests require a running platform server and are marked `manual`. The `.platform_ui/server.port` file stores the active port.

## Code Style

- Python 3.11 strict
- Type hints on all public functions and methods
- Google-style docstrings
- No hardcoded parameters — all tunables read from YAML config via Pydantic
- `ruff` for formatting and linting
- Imports: stdlib → third-party → local, separated by blank lines

## Architecture Rules

- All new SMIM code goes under `src/quantdsl_backtest/smim/`
- Existing `dsl/` and `engine/` code: read freely, modify only with explicit discussion
- SMIM components implement protocols defined in `smim/interfaces.py`
- Data adapters follow the existing FRED adapter pattern — read it first before writing new ones
- Bridge signals (SMIM → btest) go in `smim/signals/` and follow `dsl/signals.py` conventions
- Experiment configs are YAML files in `experiments/` parsed by `smim/config.py`

## Testing Rules

- Every new public function needs a test
- Unit tests in `tests/unit/smim/` mirror the `smim/` directory structure
- Integration tests in `tests_slow/smim/`
- Matrix/numerical code: test against known analytical solutions where possible
- Data adapters: test with cached fixtures, never hit live APIs in unit tests

## SMIM-Specific Context

Read `src/quantdsl_backtest/smim/CLAUDE.md` for mathematical notation, standing
assumptions, milestone status, and implementation patterns specific to the SMIM framework.

## Reference Documents (read when needed, not upfront)

- `docs/smim/PROPOSAL_SUMMARY.md` — condensed research proposal (mathematical architecture, work packages)
- `docs/smim/IMPLEMENTATION_PLAN.md` — milestones, quality gates, acceptance criteria
- `docs/smim/TASK_REGISTRY.md` — Claude Code task decomposition with current status
- `docs/smim/DECISIONS.md` — architectural decision log (append after each gate)
- `docs/smim/ADAPTER_GUIDE.md` — how to write a new data adapter (created in M1.2-T1)
- `smim/interfaces.py` — all Protocol definitions (read this before implementing anything)
- `smim/config.py` — Pydantic config models (all tuneable parameters)
- `experiments/mvp_energy_us_uk.yaml` — sample experiment config for the first build

## Git Conventions

- Branch per feature: `smim/m2.1-granger-edges`
- Commit format: `[SMIM M{wp}.{ms}-T{task}] Brief description`
- Example: `[SMIM M2.1-T3] Implement NarrativeEdgeEstimator`
- Run `uv run pytest -q` before every commit — existing tests must not break
