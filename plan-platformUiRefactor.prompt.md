## Plan: Modern component-based Platform UI

Refactor `src/quantdsl_backtest/platform_ui/site.py` from a single giant HTML+JS string into a component-based SPA optimized for large catalog tables, interactive profiling, and charting—while keeping existing API endpoints (`/api/catalog/*`, `/api/quality/*`) unchanged, staying local-first (no CDN), and preserving unit/E2E test selectors (IDs like `btnPreview`). Ship compiled assets via the existing `/static/*` route and keep FastAPI route changes minimal (mostly just returning the new `index.html` and serving built assets).

**Decisions (locked in)**
- **Built assets are committed** (recommended for local-first + keeping `pytest -q` fast without requiring Node).
- **Deep-linkable routes are supported** (at least `?tab=...&lib=...&sym=...&start=...&end=...`, and optionally hash routes).
- **Grid: AG Grid Community** (no enterprise features / licensing).

### Steps 1) Baseline contracts & constraints (freeze what must not change)
1. Inventory required DOM selectors from tests: `tests/unit/platform_api/test_ui_index.py`, `tests_slow/**` (e.g., `#btnPreview`, `#catalogSearch`, `data-testid='copy-source'`).
2. Document the current UI “feature contract” from `site.py`: Catalog listing/search, Inspector selection/preview/plot/table/raw, Quality scan/issues panes, localStorage state.
3. Freeze API contract usage from `site.py`: exact endpoints and query params used under `/api/catalog/*` and `/api/quality/*` (no new “v2” routes).

### Steps 2) Choose UI stack (React + Vite) and justify versus alternatives
1. Use **React + Vite** (TypeScript) as the SPA baseline, built into static files served locally via `/static/…`.
2. Rationale (why React+Vite here):
   - Fast DX and simple build artifacts (one `dist/` folder).
   - Strong ecosystem for data-heavy UX: AG Grid, URL-state, Plotly wrappers.
   - Clean component boundaries for plotting + grids + analytics panels.
3. Alternatives and why not (repo-specific):
   - Svelte/Vue: fine but React has the widest “data grid + analytics UI” ecosystem.
   - HTMX/alpine: too brittle for reactive grids and cross-panel state.
   - Another vanilla rewrite: repeats the current maintenance issues.

### Steps 3) Serving strategy: keep `/static` and minimize FastAPI changes
1. Keep `src/quantdsl_backtest/platform_api/routes/ui.py` routes as the integration boundary:
   - `/` returns HTML (a small `index.html` shell that loads SPA bundles from `/static/...`).
   - `/static/{path:path}` continues to serve assets locally (no CDN).
2. Add a committed build output directory:
   - `src/quantdsl_backtest/platform_ui/assets_dist/` containing Vite output (hashed JS/CSS, `assets/*`, etc).
3. Update `/static/{path:path}` resolution order:
   1) serve from `platform_ui/assets_dist/` (SPA bundles)
   2) fallback to `platform_ui/assets/` (legacy/static misc)
   3) keep the special-case `plotly.min.js` behavior (served from installed `plotly` package if available)

### Steps 4) URL-driven state & deep links (avoid later refactors)
Implement URL state early so we don’t have to retrofit it after adding analytics:
1. Support query params as the stable public “deep link” format:
   - `?tab=catalog|inspector`
   - `&lib=...&sym=...`
   - `&start=YYYY-MM-DD&end=YYYY-MM-DD`
   - optionally: `&view=plot|table|raw`
2. One-way sync on load:
   - On first mount, URL overrides localStorage defaults.
3. Two-way sync after:
   - When user changes selection/tab/date range, update URL via `history.replaceState()`.
4. Keep localStorage (`quantdsl.platform_ui.state.v1`) for “sticky” defaults, but treat URL as shareable source of truth.

### Steps 5) Incremental migration (milestone-driven, preserves functionality)
**Goal:** always keep `/` usable and keep unit tests passing between milestones.

1. Milestone A — “New shell + test contracts preserved”
   - Replace `html_index()` with a *small* HTML page that:
     - includes the required IDs and `data-testid` strings (can be in placeholder DOM)
     - loads `/static/assets_dist/...` bundles
   - This keeps `tests/unit/platform_api/test_ui_index.py` fast and stable.

2. Milestone B — Catalog as React components
   - Implement `CatalogPage` with AG Grid Community for:
     - sortable columns
     - quick filter (search)
     - row click → selection
   - Preserve IDs: `btnCatalog`, `btnCatalogClear`, `catalogSearch`, and a stable host node for the grid.
   - Keep endpoint usage identical: `GET /api/catalog`.

3. Milestone C — Inspector selection + workspace
   - Implement Inspector controls and workspace tabs:
     - `Preview` → `/api/catalog/preview/{lib}?symbol=...&head=..&tail=..`
     - Chart → `/api/catalog/chart/{lib}?symbol=...&limit=...&start=...&end=...`
     - Raw JSON view
   - Keep IDs: `btnPreview`, `pLib`, `pSym`, `dStart`, `dEnd`, `plot` (or equivalent with `data-testid='plotly-chart'`).

4. Milestone D — Download panel
   - Move download payload construction into React (but keep payload schema the same).
   - Keep IDs: `dlSource`, `dlRangeMode`, `btnGuessSource`, `btnCopyPayload`, `btnDryRun`, `btnDownload`, `downloadSummary`, `data-testid='copy-source'`.

5. Milestone E — Quality panel
   - Wire Quality Scan/Issues to `/api/quality/scan` and `/api/quality/issues`.
   - Keep IDs: `btnQualityScan`, `btnQualityIssues`.

6. Milestone F — remove legacy JS
   - Delete or significantly shrink the monolithic inline JS once React covers all flows.
   - If unit tests still assert inline `function ...` signatures, migrate those tests to assert DOM-only / contract IDs instead (keep fast).

### Steps 6) Data-grid + plotting integration (AG Grid Community + Plotly)
1. Standardize on **AG Grid Community** for:
   - Catalog grid (large lists, filtering)
   - Preview grid (head/tail)
   - Potential future analytics tables (coverage stats, missingness, etc.)
2. Plotly:
   - Keep local-first Plotly by continuing to serve `/static/plotly.min.js`.
   - Wrap rendering into a `PlotlyChart` React component.
   - Preserve `data-testid='plotly-chart'` (apply to the plot container div).

### Steps 7) Preserve unit tests and keep `pytest -q` fast
1. Keep `/` HTML containing required ID strings so `tests/unit/platform_api/test_ui_index.py` remains a cheap string assertion test.
2. Do **not** require Node/Vite build during unit tests:
   - Commit `assets_dist` output.
   - Optionally add a developer-only script to rebuild assets.
3. Keep Playwright E2E under `tests_slow/` only, per guardrails.

### Further Considerations (now resolved)
- Built assets: **commit `assets_dist/`**.
- Deep links: **do it early**, treat URL as shareable state.
- Grid licensing: **Community only** (no AG Grid Enterprise features).
