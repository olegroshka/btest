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

## Phase 4 — Signals UI (NEXT after Phase 3)
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

## Phase 5 — Strategies UI (NEXT after Signals)
**Goal:** run strategies and visualize outputs in the same Platform UI.

### Deliverables
- Strategy picker + parameters
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

---

## Engineering guardrails (apply to all phases)
- Keep APIs stable unless a new feature truly needs a new route.
- If adding a route, document it here (inputs/outputs, error modes, IDs used by tests).
- Keep Playwright E2E under `tests_slow/` only.
- Keep `assets_dist/` committed and rebuild via the repo script.
- Prefer small steps + keep tests green.

## Testing & commands (reference)
- Fast unit tests:
  - `uv run pytest -q`
- UI manual/smoke contract:
  - `uv run pytest -q tests_slow/smoke/test_ui_smoke.py -m manual`
- Slow suite gate:
  - `uv run pytest -q -n auto tests_slow -m "slow or manual"`
