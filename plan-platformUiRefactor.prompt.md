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

## Phase 3.5 — Notebook-first Strategy UI (NEW / preferred next step)
**Goal:** enable a quant-friendly workflow to edit/run/compare strategies via Jupyter notebooks, while keeping runs reproducible and comparable.

This phase intentionally leans on notebooks for *interaction* and *analysis*, but keeps the **backend run registry + artifacts** as the source of truth.

### Product shape (Phase 3.5)
- A new top-level tab/section: **Strategies**
  - Strategies grid: list available strategies (name, tags, last run, key metrics)
  - Actions: **Open Notebook**, Clone, Run, Compare, Reports
- A new top-level tab/section: **Runs**
  - Runs grid: run id, strategy version/hash, dataset version, start/end, status, key metrics
  - Actions: open reports, open analysis notebook, add to compare basket
- Compare (MVP): select 2–10 runs → show metrics table + a couple key overlays; also provide **Open compare notebook**.

### Guardrails (avoid breaking existing functionality)
- Phase 3.5 must be **additive**:
  - do not modify existing `/api/catalog/*` contracts
  - UI smoke tests continue to pass (Catalog/Meta/Inspector)
- Notebook-first must not turn into “ad-hoc stateful runs”:
  - notebooks may do exploration, but **official runs** must still record artifacts + metrics predictably.

### Where do strategy HTML reports live?
We already generate static HTML reports under `outputs/` today. For Phase 3.5 we should keep this convention, but make it **run-scoped** and **indexable**.

**Hard rule:** a run must write into a unique directory and must NOT overwrite prior outputs.

- Per-run directory:
  - `outputs/runs/<run_id>/` (preferred)
  - OR `outputs/<strategy_name>/<run_id>/` (acceptable if it matches current naming)
- Reports remain static HTML (e.g. `index.html`, `tearsheet.html`, attribution pages).
- The run record stores:
  - `reports_root` (filesystem path)
  - `reports_url` (served URL prefix) and/or an explicit list of known report entrypoints

> Implementation detail (later): serve `outputs/` (or `outputs/runs/`) via the platform server as static files so the UI can link directly.

### Minimal stats storage (MVP)
We do **not** need full historical experiment tracking on day one.

**Strategy-level snapshot (enables Strategies grid):**
- `last_run_id`
- `last_status` (success/fail)
- `last_run_at`
- `last_metrics` (a compact JSON: Sharpe, CAGR, maxDD, turnover, etc.)
- `last_reports_url` / `last_reports_entrypoints`

**Run-level minimal index (enables Compare and Runs grid):**
- Keep only the most recent **N** runs (configurable; start with N=200).
- Each run index row stores: `run_id`, `strategy_id`, `strategy_hash`, `params`, `status`, `ended_at`, `metrics`, `reports_url`.

**Persistence choice (MVP):**
- prefer simple local-first persistence under ignored folders:
  - JSONL file under `.platform_ui/` or `local_cache/` (easiest)
  - OR SQLite under `.platform_ui/` (slightly more structure)

### Key design decisions (for reproducibility)
- **Notebooks are not the official execution substrate** for full runs.
  - Notebooks submit runs to a backend job runner/worker.
  - Worker executes in a clean environment; notebook polls status and loads artifacts.
- Each run is recorded with:
  - strategy identifier + strategy version hash
  - dataset/source + dataset version tag (best-effort at first)
  - resolved params (start/end/frequency + overrides)
  - environment hash (optional early)
  - links to artifacts + reports

### Minimal backend/API contracts (add incrementally)
> Keep these small and explicit; this is the glue between Web UI, notebooks, and the backtest runner.

1) Strategies
- `GET /api/strategies` → list strategies (id, name, path, tags, last_run_id, last_metrics)
- `POST /api/strategies/clone` → clone strategy (source_id/path → new_id/path)
- `GET /api/strategies/{id}` → metadata + resolved path

2) Runs
- `POST /api/runs` → submit run (strategy_id, params, optional overrides)
- `GET /api/runs/{id}` → status + metrics + artifact index + report links
- `GET /api/runs` → list runs (filters by strategy/date/status/tag)

3) Notebook integration
- `GET /api/notebooks/open?strategy_id=...` → returns a URL/path to open in JupyterLab
- `GET /api/notebooks/compare?run_ids=...` → returns a prefilled compare notebook URL/path

### Artifact contract (must be machine-readable)
HTML reports remain great for humans, but comparison requires structured outputs.
Emit per-run (minimum):
- `summary.json` (key stats)
- `returns.parquet`
- `weights.parquet`
- `trades.parquet` (if available)
- `signal_attribution.parquet` and/or `factor_pnl.parquet` (as available)
- `config_resolved.json`
- `logs.txt`
- HTML reports (static): `index.html`, `tearsheet.html`, etc.

### Notebook UX (templates)
- Per strategy: a starter notebook template that:
  - loads the strategy
  - runs a small dry-run sample (fast feedback)
  - submits an official run via the run API
  - loads artifacts for deeper analysis
- Compare notebook template:
  - accepts run_ids
  - overlays equity/returns
  - ranks summary metrics
  - compares attribution/exposures

### Suggested milestone breakdown (keeps complexity controlled)
1) **Run directory + artifact index (no notebook yet)**
   - Introduce `run_id` + run-scoped output directory.
   - Emit `summary.json` + keep existing HTML reports.
   - Add minimal run index persistence (JSONL/SQLite).

2) **Strategies + Runs grids (web UI)**
   - Show strategies list with last-run snapshot.
   - Show runs list from the run index.
   - Link to HTML reports.

3) **Compare MVP (web)**
   - Compare table reads `summary.json` or run index metrics.
   - Minimal overlays (equity/returns) from artifacts.

4) **Notebook integration**
   - “Open notebook” and “open compare notebook” endpoints.
   - Notebook templates submit runs via API and load artifacts.

### Exit criteria
- A quant can:
  1) run a strategy and get run-scoped HTML reports
  2) see last-run metrics in Strategies grid
  3) select multiple runs and compare metrics + open reports
  4) open a notebook that can submit runs and load outputs

---

## Phase 4 — Signals UI (NEXT after Phase 3.5)
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

## Phase 5 — Strategies UI (AFTER notebook-first; custom DSL editor later)
**Goal:** evolve from notebook-first to a custom visual DSL editor + rich strategy UI, without losing reproducibility.

### Deliverables
- Guided editor for Strategy DSL (graph/nodes for factors/signals/etc.)
- Parameters UI + validation + dry-run checks
- Run strategy (locally) and show:
  - performance / equity curve
  - attribution tear sheet links (existing `outputs/*`)
  - diagnostics (turnover, costs, drawdowns)
- Persist runs and link to outputs.

Exit criteria:
- A newcomer can:
  1) populate the cache
  2) run a strategy
  3) open UI and inspect results end-to-end.
