## Plan: Modern component-based Platform UI (Roadmap)

Refactor `src/quantdsl_backtest/platform_ui/site.py` from a giant HTML+JS string into a coherent component-based SPA, while:
- keeping existing API endpoints (`/api/catalog/*`, `/api/quality/*`) unchanged
- staying local-first (no CDN)
- preserving E2E selectors / legacy contract IDs (e.g., `btnPreview`, `pLib`, `pSym`, etc.)
- shipping compiled assets via the existing `/static/*` route (FastAPI changes minimal)

This plan is designed to be used as a “prompt” for future chats.

---

### Decisions (locked in)
- **Built assets are committed** (`src/quantdsl_backtest/platform_ui/assets_dist/`) so `pytest -q` stays fast and doesn’t require Node.
- **Deep-linkable routes are supported** via query params: `?tab=...&lib=...&sym=...&start=...&end=...`.
- **Grid = AG Grid Community** (no AG Grid Enterprise modules / licensing).
- **Plotting = Plotly** served locally via `/static/plotly.min.js`.

---

## Phase 0 — Freeze contracts (DONE)
**Goal:** don’t break tests or existing integration points while refactoring.
- Inventory required DOM selectors from tests (`tests/unit/*`, `tests_slow/*`).
- Freeze API usage under `/api/catalog/*` and `/api/quality/*`.
- Preserve stable IDs and `data-testid` attributes used by automation.

---

## Phase 1 — React shell + asset pipeline (DONE)
**Goal:** ship a React SPA under existing `/static/*` plumbing.
- Vite+React build output committed under `platform_ui/assets_dist/`.
- `/` serves an `index.html` shell that loads the React bundle.
- `/static/{path}` serves committed assets.

---

## Phase 2 — “Option B” UI implementation (IN PROGRESS / stabilizing)
**Goal:** coherent React component model across all tabs, with no regression in features.

### Current state (what should work today)
- **Catalog tab**: AG Grid Community, quick filter search box, row click selection.
  - **Download panel lives on Catalog** (not Inspector) and is **input-driven** (not gated by selecting an existing cached dataset).
  - Download layout: inputs on the left, output/progress/errors on the right.
- **Meta tab**: React implementation with AG Grid for results.
- **Inspector tab**: React implementation with:
  - Preview (head/tail grids)
  - Plotly chart (with range buttons like 1m/6m/YTD/1y/5y/All)
  - Quality panel
  - (Download is no longer the primary workflow here)

### Download: current contract + behavior
- UI is driven by supported sources returned from:
  - `GET /api/catalog/sources` (returns source ids + capabilities like file-based)
- **Parquet** (file-based): requires a file/path input; dates optional.
- **YF / FRED** (network sources): require:
  - entity (e.g. `AAPL` / `CPIAUCSL`)
  - start + end dates
  - frequency (e.g. `1d`)
  - UI formats provider URI (`yf://AAPL`) behind the scenes; user does not type schemes.
- Buttons are enabled only when required fields are present, and disabled while request is in progress.

### Tests / contracts
- **Manual UI smoke contract**: `tests_slow/smoke/test_ui_smoke.py -m manual` (server-required click-through).
- Playwright E2E remains under `tests_slow/platform/`.
- Optional network-dependent verification:
  - YF real download test exists but is **skipped by default** unless `QUANTDSL_RUN_NET=1`.

### Phase 2 cleanup tasks (keep doing until stable)
1. **Stabilize Plotly lifecycle**
   - Ensure switching Plot/Table/Raw doesn’t blank the plot.
   - Avoid React touching Plotly-owned DOM nodes.
   - Ensure resize/redraw on tab switch.

2. **AG Grid consistency**
   - Community-only features; do not accidentally rely on enterprise modules.
   - Remove unused UI (e.g., checkbox selection column) but keep row-selection highlight.

3. **DOM/Forms hygiene**
   - Avoid duplicate IDs between hidden placeholders and real components.
   - Ensure form inputs have ids/names where it matters.

4. **Legacy placeholder strategy**
   - Keep only the minimal hidden placeholders required by tests.
   - Keep placeholders unique / non-duplicating.

5. **Performance + developer loop**
   - Keep the UI smoke test fast and informative.
   - Prefer deterministic waits and clear error reporting.

Exit criteria for Phase 2:
- No console errors on hard refresh of Catalog/Meta/Inspector.
- Plot remains stable when switching tabs.
- UI smoke test passes reliably and quickly.

---

## Phase 3 — Data UX + contracts (NEXT)
**Goal:** make the platform UI a strong data workbench: explore, validate, and understand catalog datasets.

### Deliverables
1. **Catalog enhancements (still AG Grid Community)**
   - Better column sizing/pinning defaults.
   - (Optional) client-side categorical filtering controls (must remain community-only).
   - Persist user grid preferences (column hide/order/width) in localStorage.

2. **Meta enhancements**
   - Make Meta query UX robust (clear empty states, loading/error states).
   - Add “copy query” / “share link” affordances (URL is source-of-truth).

3. **Inspector data ergonomics**
   - Tighten Preview/Plot UX and error states.
   - Quality: better presentation of scan summary and issues list.

4. **Diagnostics panel (optional)**
   - Show backend request-id / status for latest calls.
   - Minimal “copy debug bundle” for support.

Exit criteria:
- Data exploration flows feel complete without reaching for scripts.
- Deep-links reproduce the same state and are shareable.

---

## Phase 3.5 — Strategy Runs & Compare

**Goal:** let a quant edit a strategy in the web UI, submit it for execution, inspect run results (including the existing HTML report site), and compare multiple runs — all without leaving the browser.

This phase focuses on **task-run infrastructure** and the **web-based strategy editor + runs workflow**. Server-side notebooks (JupyterLab / Colab-style) are deferred to Phase 4.5 and will layer on top of the run infrastructure built here.

**Why this ordering (context for implementers):**
The original plan called this "Notebook-first." Codebase analysis showed that:
- The core value (edit, run, inspect, compare) does **not** require notebooks.
- Notebooks add heavy dependencies (kernel management, JupyterLab) orthogonal to the run registry.
- The existing DSL Builder tab already generates strategy Python code — extending it with a proper editor and a Run button is natural and low-risk.
- A tight web-only loop ships faster and validates the run infrastructure before notebooks layer on.

---

### Codebase context (read before implementing)

> This section tells the coding agent *where things are* so it can navigate confidently.

| Concept | Current location | Notes |
|---|---|---|
| FastAPI app factory | `src/quantdsl_backtest/platform_api/main.py` then `create_app()` | All routers registered here. New routers must follow same pattern. |
| Existing route modules | `src/quantdsl_backtest/platform_api/routes/*.py` | Each file creates `router = _router()` via `APIRouter`. Follow this pattern. |
| Service layer | `src/quantdsl_backtest/platform_api/services/*.py` | Business logic lives here, routes are thin. Follow this pattern. |
| Pydantic models | `src/quantdsl_backtest/platform_api/models/*.py` | Request/response shapes. Follow this pattern. |
| Strategy DSL dataclass | `src/quantdsl_backtest/dsl/strategy.py` then `Strategy` | `@dataclass(slots=True)`. Serializable via `dataclasses.asdict()`. |
| Backtest entry point | `src/quantdsl_backtest/engine/backtest_runner.py` then `run_backtest(strategy)` | Synchronous, CPU-bound. Returns `BacktestResult`. |
| Reporting pipeline | `src/quantdsl_backtest/engine/backtest_runner.py` then `ReportingPipeline`, renderers | Composable renderer list. New renderers added here. |
| Output dir resolution | `backtest_runner.py` then `_resolve_reporting_output_dir()` | Defaults to `outputs/<strategy.name>`. Must be patched to run-scoped path. |
| BacktestResult | `src/quantdsl_backtest/engine/results.py` | Has `.metrics`, `.metadata`, `.to_parquet()`, `.equity`, `.returns`. |
| Example strategies | `src/quantdsl_backtest/examples/*.py` | Each has `build_strategy() -> Strategy` + `main()`. |
| Frontend app shell | `frontend/src/platform/PlatformApp.tsx` | Renders tabs, manages state. |
| Tab definition | `frontend/src/platform/TabsShell.tsx` | `TabKey` union type; tab buttons rendered here. |
| Page components | `frontend/src/platform/pages/*.tsx` | One file per tab page. |
| Current DSL Builder | `frontend/src/platform/pages/DSLBuilderPage.tsx` + `routes/dsl_builder.py` | Form then generated Python code. No execution capability. |
| Frontend deps | `frontend/package.json` | React 18, AG Grid Community 34, Vite 6. No editor library yet. |
| Static asset serving | `routes/ui.py` then `GET /static/{path}` | Serves from `platform_ui/assets_dist/`. |
| Existing outputs | `outputs/` directory | Flat: `outputs/<strategy_name>/`. Contains HTML reports + parquet. **No run scoping.** |
| Test structure | `tests/unit/` (fast, default), `tests_slow/` (integration, E2E) | Unit tests must not require server. Integration tests in `tests_slow/`. |
| Local cache | `local_cache/` | gitignored. Used for ArcticDB cache. Good place for SQLite run index. |

---

### 3.5.0 — Guardrails and implementation standards

**Non-negotiable rules for all Phase 3.5 work:**

1. **Additive only** — do NOT modify existing `/api/catalog/*`, `/api/quality/*`, `/api/dsl/*` contracts or route files.
2. **Existing UI must keep working** — Catalog, Meta, Inspector, DSL Builder tabs must render and function exactly as before. All existing E2E and unit tests must continue to pass.
3. **Error handling contract** — all new API endpoints must return errors using the existing `to_api_error()` helper from `platform_api/errors.py` with the `X-Request-Id` header pattern (see `main.py` exception handlers for reference).
4. **No new runtime dependencies unless explicitly listed** — the only new pip dependency allowed is none (we use stdlib `sqlite3`, `concurrent.futures`, `subprocess`). New npm dependencies: only `@monaco-editor/react` (in M3).
5. **Every new backend module must have unit tests** — test files go under `tests/unit/platform_api/` or `tests/unit/engine/` following existing naming: `test_<module_name>.py`.
6. **Every new API route must have an integration test** — using `httpx.AsyncClient` with `app` (in-process, no server needed). See existing tests in `tests/unit/platform_api/test_*.py` for the pattern.

**Testing contract for each milestone:**

| Layer | Tool | Location | What to test |
|---|---|---|---|
| Service unit tests | pytest | `tests/unit/platform_api/test_run_store.py`, `test_strategy_discovery.py`, `test_task_runner.py` | Pure logic: insert/query/update, file scanning, job lifecycle |
| API integration tests | pytest + httpx | `tests/unit/platform_api/test_runs_endpoint.py`, `test_strategies_endpoint.py` | Request to response shape, status codes, error cases |
| Renderer unit tests | pytest | `tests/unit/engine/test_summary_json_renderer.py` | Renderer writes correct JSON given a synthetic `BacktestResult` |
| Frontend smoke | Playwright | `tests_slow/platform/test_platform_ui_runs_tab_playwright.py` | Tab renders, grid loads, status badges visible |

---

### 3.5.1 — Strategy storage and discovery

**Strategies live on the filesystem** in a `strategies/` directory at project root.

```
strategies/                         # gitignored (user workspace, not committed)
  momentum_long_short_sp500.py      # copied from examples/ on first boot
  tiny_momentum_ls.py
  lagging_indecies.py
  my_experiment.py                  # user-created
```

**Bootstrap behavior:** on server startup, if `strategies/` is empty or does not exist, copy all `.py` files from `src/quantdsl_backtest/examples/` into it. This gives new users working strategies out of the box. Log a message when bootstrapping occurs.

**Convention:** each file MUST expose a `build_strategy() -> Strategy` callable. Files that don't are skipped during discovery (log a warning).

**Strategy identity model:**
- `strategy_id`: filename stem (e.g. `momentum_long_short_sp500`). Must be unique within `strategies/`.
- `strategy_hash`: SHA-256 hex digest of file content at the moment a run is submitted. Stored per run for reproducibility.
- `source_snapshot`: full Python source text, stored per run in the run index. This means even if the file is later edited, you can always see what code produced a given run.

**Why filesystem + gitignored (not in-repo examples):**
- Strategies are user workspace — they evolve, they are experimental, they should not pollute git history.
- The `examples/` directory remains the canonical set of reference strategies (committed, tested).
- Bootstrap-on-first-run gives a smooth onboarding experience without requiring git-tracked user files.

**Implementation targets:**

| File to create | Purpose |
|---|---|
| `src/quantdsl_backtest/platform_api/services/strategy_discovery.py` | `discover_strategies(dir) -> list[StrategyInfo]`, `read_strategy_source(id) -> str`, `write_strategy_source(id, source)`, `bootstrap_strategies(examples_dir, target_dir)` |
| `src/quantdsl_backtest/platform_api/models/strategy.py` | Pydantic models: `StrategyInfo`, `StrategyDetail`, `StrategySaveRequest` |
| `tests/unit/platform_api/test_strategy_discovery.py` | Tests against a tmp_path fixture with `.py` files |

**Testing guidance:**
- Create test strategy files in `tmp_path` with valid/invalid `build_strategy()` signatures.
- Verify discovery skips files without `build_strategy()`.
- Verify `strategy_hash` changes when file content changes.
- Verify bootstrap copies files and is idempotent.

---

### 3.5.2 — Task runner infrastructure

**Design:** a lightweight in-process async job runner using `concurrent.futures.ProcessPoolExecutor`.

**Why `ProcessPoolExecutor`:**
- `run_backtest()` is CPU-bound (NumPy/Pandas heavy) — a thread would hold the GIL and block the FastAPI event loop.
- Process isolation: a crashing/OOM backtest cannot take down the API server.
- Single worker process for MVP (configurable later). Matches single-user local-first model.
- stdlib — no Celery, no Redis, no external queue.

**Job lifecycle state machine:**
```
PENDING --submit--> RUNNING --success--> SUCCEEDED
                       |
                       +--failure--> FAILED
```

**Execution flow (detailed, for implementer):**

1. **Submit** (`POST /api/runs`):
   - Read strategy source from `strategies/<strategy_id>.py`.
   - Compute `strategy_hash` (SHA-256 of source bytes).
   - Generate `run_id` = `uuid4()` as hex string.
   - Create `artifacts_dir` = `outputs/runs/<run_id>/` (mkdir).
   - Insert `RunRecord(status='pending')` into SQLite run index.
   - Dispatch to `ProcessPoolExecutor` via `asyncio.get_event_loop().run_in_executor(...)`.
   - Return `{ run_id, status: "pending" }` immediately.

2. **Worker function** (runs in a child process):
   - Redirect stdout/stderr to `outputs/runs/<run_id>/logs.txt` (use `contextlib.redirect_stdout/stderr`).
   - Write `config_resolved.json` = `json.dumps(dataclasses.asdict(strategy))` before starting.
   - Patch `strategy.backtest.reporting.output_dir` to `outputs/runs/<run_id>/`.
   - Call `run_backtest(strategy)` which returns `BacktestResult`.
   - Extract metrics from `result.metrics` and write `summary.json`.
   - Return `(run_id, "succeeded", metrics_dict)` or `(run_id, "failed", traceback_str)`.

3. **Completion callback** (back in main process, on the event loop):
   - Update `RunRecord` in SQLite: set `status`, `ended_at`, `duration_s`, `metrics_json` or `error`.

**Log streaming (SSE — Server-Sent Events):**
Live log tailing is important for long-running backtests. Implement `GET /api/runs/{id}/logs/stream` as an SSE endpoint:
- Open `logs.txt`, seek to end of what has been sent, yield new lines as `data:` events.
- Close when run status transitions to a terminal state (`succeeded` or `failed`).
- Fallback: `GET /api/runs/{id}/logs` returns the full `logs.txt` content as a string (for completed runs or clients that don't support SSE).
- Use `starlette.responses.StreamingResponse` with `media_type="text/event-stream"`.

**Implementation targets:**

| File to create | Purpose |
|---|---|
| `src/quantdsl_backtest/platform_api/services/task_runner.py` | `TaskRunner` class: manages `ProcessPoolExecutor`, `submit_run()`, `get_status()`, completion callbacks. Singleton lifecycle tied to FastAPI `lifespan`. |
| `src/quantdsl_backtest/platform_api/services/run_worker.py` | `execute_run(strategy_id, run_id, source, artifacts_dir, params) -> WorkerResult` — the function that runs in the child process. Must be importable at module level (pickle requirement for ProcessPoolExecutor). |
| `tests/unit/platform_api/test_task_runner.py` | Test job submission, status transitions, completion callback. Use a mock worker that returns immediately. |

**Testing guidance:**
- The worker function must be testable in-process (call it directly, verify it writes `summary.json` and `logs.txt`).
- The `TaskRunner` should be testable with a 1-worker pool and a trivial callable.
- Test failure path: worker raises an exception, verify `RunRecord.status == "failed"` and `error` contains traceback.
- Test timeout: if a run exceeds a configurable timeout (default 10 minutes), mark as failed. (Nice-to-have for MVP.)

---

### 3.5.3 — Run-scoped output directory and artifact contract

**Hard rule:** a run MUST write into a unique directory and MUST NOT overwrite prior outputs.

**Directory layout per run:**
```
outputs/runs/<run_id>/
  summary.json              # NEW — machine-readable key stats
  config_resolved.json      # NEW — full strategy config snapshot (dataclasses.asdict)
  logs.txt                  # NEW — captured stdout/stderr
  index.html                # existing ReportSiteIndexRenderer
  tearsheet.html            # existing QuantStatsHtmlRenderer
  signals/<name>/           # existing SignalAnalyticsHtmlRenderer
  attribution/<name>/       # existing SignalAnalyticsHtmlRenderer
  equity.parquet            # existing ParquetArtifactsRenderer
  returns.parquet           # (need to add — currently only equity.parquet is written)
  weights.parquet           # existing ParquetArtifactsRenderer
  trades.parquet            # existing ParquetArtifactsRenderer
  positions.parquet         # existing ParquetArtifactsRenderer
```

**`summary.json` schema:**
```json
{
  "run_id": "a1b2c3d4",
  "strategy_id": "momentum_long_short_sp500",
  "strategy_name": "xsec_momentum_long_short_sp500",
  "strategy_hash": "sha256:abcdef",
  "engine": "event_driven",
  "start_date": "2015-01-01",
  "end_date": "2025-01-01",
  "duration_s": 42.3,
  "metrics": {
    "total_return": 0.42,
    "cagr": 0.036,
    "sharpe": 0.85,
    "sortino": 1.12,
    "max_drawdown": -0.18,
    "volatility": 0.12,
    "turnover_annual": 4.2,
    "calmar": 0.20
  },
  "artifacts": ["index.html", "tearsheet.html", "equity.parquet", "weights.parquet", "trades.parquet"]
}
```

**Implementation — changes to existing files:**

| File to modify | Change |
|---|---|
| `src/quantdsl_backtest/engine/backtest_runner.py` | Add `SummaryJsonRenderer` class (follows existing `ResultsRenderer` protocol). Add it to `build_reporting_pipeline()` — append after all other renderers so `artifacts` list can enumerate what was written. |
| `src/quantdsl_backtest/engine/backtest_runner.py` | In `build_reporting_pipeline()`, make `ParquetArtifactsRenderer` default-on when `output_dir` is set (currently gated on `parquet.enabled` config flag). |

| File to create | Purpose |
|---|---|
| `tests/unit/engine/test_summary_json_renderer.py` | Create a synthetic `BacktestResult` (use existing test fixtures from `tests/unit/engine/`), render via `SummaryJsonRenderer`, assert JSON structure and values. |

**Testing guidance:**
- `SummaryJsonRenderer.render()` must be safe when called with minimal `BacktestResult` (empty trades, no signal reports).
- Test that `artifacts` list accurately reflects files present in `output_dir` after all renderers run.
- Test that `summary.json` is valid JSON and all values are JSON-serializable (no numpy types leaking).

---

### 3.5.4 — Run index persistence (SQLite)

**Location:** `local_cache/platform_meta/runs.db` (gitignored, auto-created).

**Why SQLite:**
- Need efficient filtering (by strategy, status, date) and sorting (by submitted_at, metrics) for the Runs grid.
- JSONL would require full scan into memory for every query.
- SQLite is stdlib (`sqlite3`), zero-config, single-file, ACID.
- Plays well with future features: full-text search on logs, pagination, retention policies.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS runs (
  run_id          TEXT PRIMARY KEY,
  strategy_id     TEXT NOT NULL,
  strategy_hash   TEXT NOT NULL,
  source_snapshot TEXT,
  params_json     TEXT,
  status          TEXT NOT NULL DEFAULT 'pending',
  submitted_at    TEXT NOT NULL,
  started_at      TEXT,
  ended_at        TEXT,
  duration_s      REAL,
  metrics_json    TEXT,
  error           TEXT,
  artifacts_dir   TEXT,
  reports_url     TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_strategy   ON runs(strategy_id);
CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_submitted  ON runs(submitted_at DESC);
```

**Retention:** keep last N=500 run records (configurable). Prune oldest on insert when count exceeds limit. Artifact directories on disk are NOT auto-deleted (user manages disk space).

**Implementation targets:**

| File to create | Purpose |
|---|---|
| `src/quantdsl_backtest/platform_api/services/run_store.py` | `RunStore` class: `__init__(db_path)`, `insert_run(record)`, `update_run(run_id, **fields)`, `get_run(run_id) -> RunRecord or None`, `list_runs(strategy_id?, status?, limit, offset) -> list[RunRecord]`, `list_strategies_summary() -> list[StrategySummary]` (derived: distinct strategy_ids + last run snapshot), `_ensure_schema()`, `_prune()` |
| `src/quantdsl_backtest/platform_api/models/run.py` | Pydantic models: `RunRecord`, `RunSubmitRequest`, `RunSubmitResponse`, `RunListResponse`, `StrategySummary` |
| `tests/unit/platform_api/test_run_store.py` | Full CRUD tests against in-memory SQLite (`:memory:`). Test insert, update status, query filters, ordering, pagination, retention pruning. |

**Testing guidance:**
- Use `:memory:` SQLite for unit tests — fast, no cleanup needed.
- Test `list_strategies_summary()` returns correct `last_run_*` fields when multiple runs exist per strategy.
- Test concurrent inserts (simulate with threads) — SQLite handles this but verify no data corruption.
- Test that `_ensure_schema()` is idempotent (call twice, no error).
- Test retention: insert 510 records, verify only 500 remain after prune.

---

### 3.5.5 — Backend API routes

All new routes live under `/api/`. Do NOT touch existing route files.

**New route files to create:**

| File | Prefix | Purpose |
|---|---|---|
| `src/quantdsl_backtest/platform_api/routes/strategies.py` | `/api/strategies` | Strategy listing, detail, save |
| `src/quantdsl_backtest/platform_api/routes/runs.py` | `/api/runs` | Run submission, listing, detail, logs, delete |

**Register in `main.py`** (add after existing router registrations):
```python
from .routes import strategies, runs
app.include_router(strategies.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
```

**Strategy endpoints:**

| Method | Path | Purpose | Response shape |
|---|---|---|---|
| `GET` | `/api/strategies` | List discovered strategies + last-run snapshot | `{ strategies: [{ id, name, path, description, last_run_id, last_status, last_metrics, last_run_at }] }` |
| `GET` | `/api/strategies/{id}` | Strategy detail + full source code | `{ id, name, path, source, description, runs_count }` |
| `POST` | `/api/strategies` | Create new strategy file | Body: `{ id, source }`. Writes to `strategies/<id>.py`. Returns `{ id, path }`. |
| `PUT` | `/api/strategies/{id}` | Update strategy source | Body: `{ source }`. Overwrites file. Returns `{ id, hash }`. |

**Run endpoints:**

| Method | Path | Purpose | Response shape |
|---|---|---|---|
| `POST` | `/api/runs` | Submit a run | Body: `{ strategy_id, params: { start, end, engine } }`. Returns `{ run_id, status: "pending" }`. |
| `GET` | `/api/runs` | List runs | Query: `strategy_id`, `status`, `limit`, `offset`. Returns `{ runs: [RunRecord], total: int }`. |
| `GET` | `/api/runs/{id}` | Single run detail | Returns full `RunRecord` (metrics, error, reports_url, etc.) |
| `GET` | `/api/runs/{id}/logs` | Get full log content | Returns `{ logs: string }` (content of `logs.txt`). |
| `GET` | `/api/runs/{id}/logs/stream` | SSE live log tail | `StreamingResponse` with `text/event-stream`. Yields new log lines as `data:` events. Closes when run reaches terminal status. |
| `GET` | `/api/runs/{id}/source` | Get source snapshot | Returns `{ source: string, strategy_id, strategy_hash }`. |
| `GET` | `/api/runs/{id}/equity` | Equity timeseries as JSON | Reads `equity.parquet`, returns `{ dates: [...], values: [...] }`. Used by Compare. |
| `DELETE` | `/api/runs/{id}` | Delete run + artifacts | Deletes SQLite row + `shutil.rmtree(artifacts_dir)`. Returns `{ deleted: true }`. |

**Report serving (static files):**

Mount `outputs/runs/` as static files so the UI can link to HTML reports:
```python
from starlette.staticfiles import StaticFiles
app.mount("/reports/runs", StaticFiles(directory="outputs/runs"), name="run_reports")
```

This gives URLs like `/reports/runs/<run_id>/index.html`, `/reports/runs/<run_id>/tearsheet.html`.

**Implementation targets:**

| File to create | Purpose |
|---|---|
| `src/quantdsl_backtest/platform_api/routes/strategies.py` | Strategy CRUD routes |
| `src/quantdsl_backtest/platform_api/routes/runs.py` | Run lifecycle routes |
| `tests/unit/platform_api/test_strategies_endpoint.py` | In-process httpx tests for all strategy endpoints |
| `tests/unit/platform_api/test_runs_endpoint.py` | In-process httpx tests for all run endpoints |

**Testing guidance:**
- Use `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))` for in-process API tests (no server process needed). See existing test files for the pattern.
- Test error cases: submit run for nonexistent strategy returns 404. Submit duplicate strategy id returns 409. Delete nonexistent run returns 404.
- Test that `POST /api/runs` returns immediately (does not block until backtest completes).
- Test `GET /api/runs/{id}` returns `status: "pending"` right after submit, then eventually `"succeeded"` or `"failed"` after worker completes (poll in test with short timeout).
- For fast tests, mock the worker to return instantly (do not run a real backtest in unit tests).

---

### 3.5.6 — Runs tab (frontend, Milestone M2)

A new top-level tab: **Runs**.

**Implementation targets:**

| File to modify | Change |
|---|---|
| `frontend/src/platform/TabsShell.tsx` | Add `'runs'` to `TabKey` union. Add Runs tab button. |
| `frontend/src/platform/PlatformApp.tsx` | Import and render `RunsPage` component. |

| File to create | Purpose |
|---|---|
| `frontend/src/platform/pages/RunsPage.tsx` | Full Runs tab implementation |

**Runs tab specification:**

**Grid (AG Grid Community):**
- Columns: `status` (icon badge), `strategy` (text), `run_id` (short 8-char), `submitted` (relative time), `duration` (seconds), `sharpe`, `total_return`, `max_drawdown`, `turnover`, `engine`
- Default sort: `submitted` descending (newest first)
- Row selection: single-click highlights, checkbox for multi-select (compare)

**Status badges:**
- `pending` = grey clock icon
- `running` = blue spinner (CSS animation pulse)
- `succeeded` = green checkmark
- `failed` = red X icon

**Row actions (buttons or context menu):**
- **View Report** opens `/reports/runs/<run_id>/index.html` in a new browser tab (full HTML report site)
- **View Logs** expands a detail panel below the row (or modal) showing log content; use SSE endpoint `GET /api/runs/{id}/logs/stream` for live tailing of in-progress runs, fall back to `GET /api/runs/{id}/logs` for completed runs
- **View Source** expands a detail panel showing the Python source snapshot (read-only Monaco or `<pre>` block) fetched from `GET /api/runs/{id}/source`
- **Delete** shows confirmation dialog then calls `DELETE /api/runs/{id}` and removes the row from grid

**Auto-refresh:**
- Poll `GET /api/runs?limit=50` every 3 seconds **only** while at least one run in the current data has `status === "pending"` or `status === "running"`.
- Stop polling when all visible runs are terminal (`succeeded` or `failed`).
- A "Refresh" button is always available for manual refresh.

**Filter bar (above grid):**
- Dropdown: filter by strategy (populated from distinct strategy names in runs)
- Dropdown: filter by status (all / pending / running / succeeded / failed)
- These set query params on `GET /api/runs`

**Deep-linking:** `?tab=runs&run_id=<id>` opens Runs tab, highlights and scrolls to that run, expands its detail panel.

**Testing guidance:**
- Playwright test: navigate to `?tab=runs`, verify grid renders with correct column headers.
- Playwright test: mock API to return a run with `status: "succeeded"`, verify green badge renders.
- Playwright test: click "View Report" button, verify a new window opens with the report URL.

---

### 3.5.7 — DSL Editor upgrade (frontend, Milestone M3)

Transform the existing DSL Builder tab into a real strategy editor with code editing and execution capability.

**New npm dependency:** `@monaco-editor/react` (MIT, lazy-loads Monaco Editor approximately 2MB).

Install:
```bash
cd frontend && npm install @monaco-editor/react
```

**Implementation targets:**

| File to modify | Change |
|---|---|
| `frontend/src/platform/pages/DSLBuilderPage.tsx` | Major rewrite — two-panel layout with Monaco editor |
| `frontend/package.json` | Add `@monaco-editor/react` dependency |

**Layout (two-panel, side by side):**

**Left panel — Strategy Form** (existing, improved):
- Keep current form controls: Data Config, Universe, Factors, Signals, Portfolio
- Add "Add Factor" / "Remove Factor" buttons for dynamic factor list
- Add "Add Signal" / "Remove Signal" buttons for dynamic signal list
- Add more factor types to the dropdown (all types from `dsl/factors.py`: `ReturnFactor`, `VolatilityFactor`, `OvernightReturnFactor`, `IntradayReturnFactor`, `FiboRetraceFactor`)
- Add more signal types to the dropdown (from `dsl/signals.py`)
- Form changes auto-update the code in the right panel (existing behavior, keep)

**Right panel — Code Editor** (new):
- Monaco Editor instance configured for Python: language="python", theme="vs-dark", minimap disabled, wordWrap on, fontSize 13, lineNumbers on
- **Two modes** (toggle button above editor):
  - **Generated** (default): form changes auto-update the code. Editor is editable but changes are overwritten on next form change.
  - **Free-edit**: form is disabled/greyed. User edits code freely. Code is the source of truth.
- **Strategy selector** (dropdown above editor): pick an existing strategy from `GET /api/strategies` to load its source into the editor.

**Action buttons (toolbar above the editor):**
- **Save** calls `PUT /api/strategies/{id}` (or `POST /api/strategies` if new). Source = current editor content. Shows success/error toast.
- **Run** calls `POST /api/runs { strategy_id }`. On success, switch to Runs tab (`?tab=runs&run_id=<new_id>`).
- **Save and Run** does Save then Run in sequence.

**Testing guidance:**
- Playwright test: navigate to `?tab=dsl_builder`, verify Monaco editor renders (look for `.monaco-editor` CSS class).
- Playwright test: type code in editor, click Save, verify `PUT /api/strategies/{id}` is called.
- Playwright test: click Run, verify redirect to Runs tab with the new run_id in the URL.
- Unit test (API level): `POST /api/strategies` with valid Python source, verify file created on disk.

---

### 3.5.8 — Compare workflow (frontend, Milestone M4)

**Access:** a "Compare" button appears in the Runs tab toolbar when 2 or more runs are checked.

**Compare view** (full-width panel that replaces the grid, or a new sub-view within Runs tab):

1. **Metrics comparison table:**
   - Rows: metric names (total_return, sharpe, sortino, max_drawdown, cagr, volatility, turnover, calmar)
   - Columns: one per selected run (header = strategy name + run_id short)
   - Cell formatting: percentages for return/drawdown metrics, 2-3 decimal places for ratios
   - Highlight: green for best value per row, red for worst
   - Data source: `metrics` from `RunRecord` (already in SQLite, returned by `GET /api/runs/{id}`)

2. **Equity curve overlay (Plotly):**
   - Fetch equity timeseries for each selected run via `GET /api/runs/{id}/equity`
   - Normalize all curves to start at 1.0
   - One trace per run, colored distinctly, with legend
   - Standard Plotly range buttons (1m, 6m, YTD, 1y, 5y, All) matching Inspector tab pattern

3. **Report links:** for each compared run, a button to open its full HTML report site in a new tab.

4. **"Back to runs"** button to return to the grid view.

**Implementation targets:**

| File to create | Purpose |
|---|---|
| `frontend/src/platform/pages/ComparePanel.tsx` | Compare view component (metrics table + Plotly chart) |

**Testing guidance:**
- Playwright test: select 2 runs via checkboxes, click Compare, verify metrics table renders with correct number of columns.
- API test: `GET /api/runs/{id}/equity` returns valid JSON with `dates` and `values` arrays.

---

### 3.5.9 — Integration with existing reporting pipeline

**Principle:** minimal, additive changes to existing backtest engine code.

**Changes to `backtest_runner.py`:**

1. **Add `SummaryJsonRenderer`** (new class, follows `ResultsRenderer` protocol):
   - `name = "summary_json"`
   - `render()`: scans `ctx.output_dir` for existing files, builds `summary.json` dict, writes it.
   - Added to `build_reporting_pipeline()` as the **last** renderer (so `artifacts` list is complete).

2. **Make `ParquetArtifactsRenderer` default-on** when `output_dir` is set:
   - Currently: only added when `rep.parquet.enabled == True`.
   - Change: always add when `output_dir is not None`, unless explicitly disabled.
   - This ensures every run produces parquet artifacts for comparison.

3. **`returns.parquet` output**: currently `to_parquet()` writes `equity.parquet` but not a standalone `returns.parquet`. Verify and add if missing (the `BacktestResult.returns` Series should be saved alongside equity).

**No changes to:**
- `Strategy` dataclass — no new fields
- `BacktestResult` dataclass — no new fields
- `ReportingPipeline` protocol — no signature changes
- Any existing renderer — no modifications

**How the task runner integrates:**
The worker function (`run_worker.py`) patches `strategy.backtest.reporting.output_dir` to the run-scoped path before calling `run_backtest()`. All existing renderers then automatically write to `outputs/runs/<run_id>/`. The only new renderer is `SummaryJsonRenderer`. The worker also writes `config_resolved.json` and captures logs independently of the reporting pipeline.

---

### Milestone sequence and dependencies

```
M1: Run infrastructure (backend)     <-- no frontend changes, testable via httpx + pytest
 |
 |-- 3.5.1 Strategy discovery
 |-- 3.5.3 SummaryJsonRenderer
 |-- 3.5.4 Run store (SQLite)
 |-- 3.5.2 Task runner
 |-- 3.5.5 API routes + static file mount
 +-- 3.5.9 Reporting pipeline integration
 |
M2: Runs tab (frontend)              <-- depends on M1 API being available
 |
 +-- 3.5.6 RunsPage.tsx + TabsShell update
 |
M3: DSL Editor upgrade (frontend)    <-- depends on M1 (save/run endpoints) + M2 (navigate to Runs tab)
 |
 +-- 3.5.7 DSLBuilderPage rewrite + Monaco
 |
M4: Compare MVP (frontend)           <-- depends on M2 (runs grid with multi-select)
 |
 +-- 3.5.8 ComparePanel.tsx + equity endpoint
```

Each milestone is independently deployable and testable. M1 can be fully validated with pytest before any frontend work begins.

### Exit criteria (Phase 3.5)
A quant can:
1. Open the DSL Builder tab, load an existing strategy (or write one from scratch), save it, and click Run.
2. See the run appear in the Runs tab with live status updates (pending to running to succeeded/failed).
3. When succeeded: click "View Report" to open the full HTML report site (index.html, tearsheet, signals, attribution) in a new tab.
4. When failed: see the error traceback and full logs in the Runs tab.
5. Select 2+ completed runs and compare their metrics side-by-side with equity curve overlays.
6. All existing tabs (Catalog, Meta, Inspector) continue to work unchanged.
7. All existing unit and E2E tests continue to pass.

---

## Phase 4 — Signals UI
**Goal:** introduce first-class UX around signals computed from data.

### Scope
- A new top-level tab or section: **Signals**
- Define a *minimal* API contract (if adding endpoints is required, keep it incremental and written down).

### Deliverables
- List available signals (name, universe, lookback, frequency, description).
- Signal explorer:
  - pick signal
  - pick universe / entity
  - preview chart + table
  - show meta (coverage / missingness)
- Ability to deep-link to a signal view.

Exit criteria:
- Signals are explorable similarly to raw data (catalog) with consistent UI patterns.

---

## Phase 4.5 — Server-side Notebooks
**Goal:** add Jupyter/Colab-style notebook support that layers on top of the run infrastructure from Phase 3.5.

This phase is intentionally separated from Phase 3.5 because it introduces heavy dependencies (JupyterLab server, kernel management, `nbformat`) and the core edit/run/compare loop is fully functional without it.

### Scope
- Launch a JupyterLab server alongside (or embedded within) the platform API
- Strategy notebook templates: load strategy, dry-run, submit official run via `POST /api/runs`, load artifacts
- Compare notebook templates: accept `run_ids`, overlay equity curves, compare metrics
- API endpoints: `GET /api/notebooks/open?strategy_id=...`, `GET /api/notebooks/compare?run_ids=...`

### Key design decisions
- Notebooks are **not** the execution substrate — they call `POST /api/runs` exactly like the web UI does
- The run registry + artifacts (from Phase 3.5) are the single source of truth
- Notebooks are a power-user analysis tool layered on top, not a replacement for the web workflow

Exit criteria:
- A quant can open a notebook from the UI, submit a run via API, and analyze artifacts
- Compare notebook can load N runs and overlay results
- All Phase 3.5 functionality continues to work unchanged

---

## Phase 5 — Visual Strategy Builder
**Goal:** evolve from the text-based DSL editor to a visual graph/node-based strategy builder, without losing reproducibility.

### Deliverables
- Guided visual editor for Strategy DSL (graph/nodes for factors to signals to portfolio)
- Parameters UI + validation + dry-run checks
- Bidirectional sync: visual graph and Python DSL code
- Run strategy from the visual editor (reuses Phase 3.5 run infrastructure)

Exit criteria:
- A newcomer can:
  1. Build a strategy visually without writing Python
  2. Run it and inspect results end-to-end
  3. Export the visual strategy as Python code
