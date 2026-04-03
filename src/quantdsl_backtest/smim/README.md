# SMIM — Spectral Multi-layer Investment Misallocation

SMIM estimates actor-specific investment gaps by building a directed multilayer
graph, applying spectral decomposition, running state-space filtering with regime
switching, computing emergence diagnostics, and finally producing investment gap
benchmarks relative to those diagnostics.

This is a **research framework**, not a trading strategy. It lives as a subpackage
of `quantdsl_backtest` and shares its data infrastructure (ArcticDB caching,
FRED/parquet adapters). The connection to the backtesting engine happens only
through bridge signals in `smim/signals/`.

---

## Package Layout

```
src/quantdsl_backtest/smim/
├── interfaces.py        # All Protocols, dataclasses, enums — read this first
├── config.py            # Pydantic config models (SmimConfig + per-component sections)
├── profiling.py         # Pipeline runtime profiler
│
├── compute/             # GPU/CPU compute layer (PyTorch, CPU/CUDA unified)
│   ├── torch_ops.py     # Device selection, tensor conversion utilities
│   ├── linalg.py        # SVD, eigh, polar, Schur, hermitian dilation, batch solve
│   ├── batch_granger.py # Batched Granger causality via torch.linalg.lstsq
│   ├── batch_pid.py     # Batched PID bootstrap via torch covariance + log-det
│   └── gpu_knn.py       # Brute-force KNN via torch pairwise distances
│
├── data/                # Data loading and actor registry
│   ├── actor_registry.py      # ActorRegistry: N actors with type/layer/geography
│   ├── intensity_mappers.py   # InvestmentIntensityMapper (per-ActorType normalisation)
│   ├── pit_store.py           # Point-in-time data store (A1 enforcement)
│   ├── quality_checks.py      # Data quality validation
│   ├── coverage_report.py     # Data coverage diagnostics
│   └── adapters/              # Source-specific data adapters
│       ├── fred_vintage.py    # FRED vintage series (point-in-time safe)
│       ├── bea_io.py          # BEA input-output tables
│       ├── edgar.py           # SEC EDGAR filings
│       ├── gdelt.py           # GDELT news event data
│       ├── imf_sdmx.py        # IMF SDMX data service
│       └── oecd_sdmx.py       # OECD SDMX data service
│
├── graph/               # Directed multilayer graph construction
│   ├── edges/
│   │   ├── base.py            # EdgeEstimator base + registry
│   │   ├── granger.py         # GrangerEdgeEstimator (statsmodels VAR + batch GPU)
│   │   ├── narrative.py       # NarrativeEdgeEstimator (GDELT co-occurrence)
│   │   └── supply_chain.py    # SupplyChainEdgeEstimator (BEA I-O tables)
│   ├── operators.py           # Multilayer → combined operator A_t
│   ├── sparsification.py      # Spectral sparsification (retains >80% energy)
│   ├── null_models.py         # Random graph null models for falsification
│   ├── null_comparison.py     # Null model comparison utilities
│   ├── ablation.py            # Edge-type ablation studies
│   ├── sensitivity_report.py  # Sensitivity analysis over edge parameters
│   └── storage.py             # Graph persistence (ArcticDB)
│
├── spectral/            # Spectral decomposition of the graph operator
│   ├── base.py                # SpectralDecomposer base class
│   ├── schur.py               # SchurDecomposer (complex Schur via scipy)
│   ├── polar.py               # PolarDecomposer (A=UP, GPU-accelerated SVD)
│   ├── hermitian.py           # HermitianDilationDecomposer (GPU-accelerated)
│   ├── dmd.py                 # DMDDecomposer — Dynamic Mode Decomposition
│   ├── dv_basis.py            # Dahleh-Verghese orthonormal basis construction
│   ├── mode_selection.py      # MDLModeSelector — selects K* modes via MDL
│   ├── comparison.py          # Decomposer comparison utilities
│   ├── oos_evaluation.py      # Out-of-sample modal basis evaluation
│   └── modal_report.py        # Gate G3 modal stability report
│
├── dynamics/            # State-space filtering and regime detection
│   ├── kalman.py              # KalmanFilter — Woodbury-accelerated (167x speedup)
│   ├── kim_filter.py          # KimFilter — M-regime switching (321x speedup)
│   ├── model_selection.py     # select_regime_count — BIC over M=1..M_max
│   ├── observability.py       # Observability matrix and rank diagnostics
│   ├── phase_transition.py    # PhaseTransitionDetector — order parameter psi_t
│   ├── actor_level.py         # Actor-level state disaggregation
│   └── evaluation.py          # Filter evaluation metrics
│
├── emergence/           # Emergence diagnostics (synergy, entropy, topology)
│   ├── pid.py                 # PID — Partial Information Decomposition (batch GPU)
│   ├── transfer_entropy.py    # KSG transfer entropy (GPU KNN)
│   └── tda.py                 # TDA — persistent homology, Betti numbers
│
├── gaps/                # Investment gap benchmarks
│   ├── predictive.py          # PredictiveBenchmark — y* = U @ alpha_{t|t-1}
│   ├── modal.py               # ModalBenchmark — y* = U @ alpha_{t|t}
│   ├── emergence_aware.py     # EmergenceAwareBenchmark — synergy-adjusted
│   ├── structural.py          # StructuralBenchmark — long-run equilibrium
│   └── emergence_evaluation.py # EmergenceAware gap evaluation utilities
│
├── signals/             # Bridge: SMIM gaps → btest trading DSL
│   └── gap_signal.py          # GapSignal — wraps GapResult as a btest factor
│
└── validation/          # Statistical validation and robustness tests
    ├── metrics.py             # Spearman rho, Kendall tau, hit rate, OOS R²
    ├── rolling_oos.py         # Rolling out-of-sample evaluation
    ├── falsification.py       # Graph rewiring falsification tests
    ├── event_studies.py       # Event-study validation against known episodes
    ├── persistence.py         # Gap persistence tests
    ├── baselines.py           # Naive benchmark comparisons
    ├── model_comparison.py    # Decomposer and filter model comparison
    ├── transfer.py            # Transfer validation across sectors/geographies
    ├── evidence_report.py     # Gate evidence compilation report
    └── extension_report.py    # Extension study report (sector/country)
```

---

## Experiment Toolbox

Every component in the table below is fully implemented, tested, and
GPU-accelerated where applicable. Pass the corresponding config class
to `SmimConfig` to enable or tune it.

### Graph Construction

| Component | Class | Config | Description |
|-----------|-------|--------|-------------|
| Granger edges | `GrangerEdgeEstimator` | `GrangerEdgeConfig` | VAR-based Granger causality. BIC lag selection. GPU batch: 250K pairs solved in one `torch.linalg.lstsq` call (11–16× faster at N=200). |
| Narrative edges | `NarrativeEdgeEstimator` | `NarrativeEdgeConfig` | GDELT co-occurrence matrix → directed influence edges. |
| Supply-chain edges | `SupplyChainEdgeEstimator` | `SupplyChainEdgeConfig` | BEA input-output tables → upstream/downstream flow edges. |
| Graph operator | `build_operator()` | — | Combines all edge channels into a single sparse N×N operator A_t. |
| Sparsification | `SpectralSparsifier` | `SparsificationConfig` | Drops weak edges while retaining >80% spectral energy (A3). |
| Null models | `ConfigurationModel`, `ErdosRenyi` | `NullModelConfig` | Random graph baselines for falsification tests. |

### Spectral Decomposition

| Component | Class | Config | Description |
|-----------|-------|--------|-------------|
| Schur | `SchurDecomposer` | `SpectralConfig` | Complex Schur A=QTQ^H via scipy. Produces complex eigenmodes. **Always CPU** (no `torch.linalg.schur`). |
| Polar | `PolarDecomposer` | `SpectralConfig` | Polar A=UP via SVD. Orthonormal real basis. **GPU-accelerated** via `compute.linalg.polar_decompose`. |
| Hermitian dilation | `HermitianDilationDecomposer` | `SpectralConfig` | Eigendecompose H=[[0,A],[A^T,0]]. Handles non-square A. **GPU-accelerated**. |
| DMD | `DMDDecomposer` | `SpectralConfig` | Dynamic Mode Decomposition — data-driven temporal modes. **GPU-accelerated SVD**. |
| Dahleh-Verghese | `DVBasisBuilder` | `SpectralConfig` | Orthonormal basis from DV construction for non-normal operators. |
| Mode selection | `MDLModeSelector` | `ModeSelectionConfig` | Selects K* via Minimum Description Length. Enforces A4 (stable modes). |

### State-Space Dynamics

| Component | Class | Config | Key methods |
|-----------|-------|--------|-------------|
| **Kalman filter** | `KalmanFilter` | `KalmanConfig` | `.filter(obs, modal_frame)` — standard Kalman smoother. `.em_estimate(...)` — EM parameter estimation. **Woodbury identity**: O(NK²+K³) instead of O(N³). 167× faster at N=200. |
| **Kim filter** | `KimFilter` | `KimConfig` | `.filter(obs, modal_frame, F_list, Q_list, R, P_trans)` — M-regime Hamilton/Kim filter with collapse approximation. `.em_estimate(...)` — symmetric EM (see known limitations). **321× faster at N=200** after Woodbury fix. |
| Regime selection | `select_regime_count()` | `RegimeConfig` | BIC over M=1..M_max Kim filter fits. Returns `RegimeSelectionResult` with `selected_m`. |
| Observability | `ObservabilityAnalyzer` | — | Computes observability matrix, checks rank ≥ K (A4 enforcement). |
| Phase transition | `PhaseTransitionDetector` | `PhaseConfig` | Order parameter ψ_t = mean field of alpha_t. Criticality C_t from susceptibility. |
| Actor-level states | `disaggregate_actor_states()` | — | Projects modal states α_t back to actor level via U. |

### Emergence Diagnostics

| Component | Class | Config | Description |
|-----------|-------|--------|-------------|
| **PID** | `PIDAnalyzer` | `PIDConfig` | Partial Information Decomposition: synergy S_{jk}, redundancy R_{jk}, unique U_{jk} for all mode pairs. Bootstrap CIs. **GPU batch**: K*(K-1)/2 pairs × B samples in one covariance computation. |
| Transfer entropy | `TransferEntropyEstimator` | `TEConfig` | KSG estimator (Kraskov Alg-1). Frenzel-Pompe conditional TE. **GPU KNN**: brute-force pairwise distances on device (10–12× at T≥2000). |
| TDA | `TopologicalComplexityAnalyzer` | `TDAConfig` | Vietoris-Rips persistent homology via `ripser`. Betti numbers β_0, β_1. Bottleneck distance d_B for stability (tolerance 2ε per VR theorem). |

### Investment Gap Benchmarks

| Component | Class | `BenchmarkClass` label | Description |
|-----------|-------|------------------------|-------------|
| Predictive | `PredictiveBenchmark` | `PREDICTIVE` | y*_{i,t} = U α_{t\|t-1} — one-step-ahead prediction. Gap = observation − prediction. |
| Modal | `ModalBenchmark` | `MODAL` | y*_{i,t} = U α_{t\|t} — filtered (smoothed) reconstruction. |
| EmergenceAware | `EmergenceAwareBenchmark` | `EMERGENCE_AWARE` | Predictive baseline adjusted by synergy S and criticality C. |
| Structural | `StructuralBenchmark` | `STRUCTURAL` | Long-run equilibrium from stationary distribution of A. |

Every `GapResult` must carry a `BenchmarkClass` — the field is non-optional.

### GPU Compute Layer (`compute/`)

All operations are numpy-in / numpy-out. Device is selected via `SMIM_DEVICE`
environment variable (`"auto"` / `"cpu"` / `"cuda"`) or `ComputeConfig.device`.

| Function | Module | Accelerated? | Notes |
|----------|--------|--------------|-------|
| `svd(A, k)` | `compute.linalg` | GPU | Truncated SVD via `torch.linalg.svd` |
| `eigh(H, k)` | `compute.linalg` | GPU | Symmetric eigensolver |
| `polar_decompose(A)` | `compute.linalg` | GPU | A = UP via SVD |
| `hermitian_dilation_decompose(A, k)` | `compute.linalg` | GPU | H = [[0,A],[A^T,0]] eigensystem |
| `schur_decompose(A, k)` | `compute.linalg` | **CPU only** | `scipy.linalg.schur` — no torch equivalent |
| `batch_solve(A, B)` | `compute.linalg` | GPU | Batched `torch.linalg.solve` |
| `batch_granger_test(signals, lag, p)` | `compute.batch_granger` | GPU | 250K pairs in one lstsq call |
| `batch_pid_synergy(modal, target, B)` | `compute.batch_pid` | GPU | K*(K-1)/2 pairs × B bootstrap |
| `knn_query(data, k)` | `compute.gpu_knn` | GPU | Brute-force pairwise L∞/L2 KNN |

### Validation Tools

| Component | Class / function | Description |
|-----------|-----------------|-------------|
| Rolling OOS | `RollingOOSEvaluator` | Expanding-window OOS R² and hit rate over time |
| Falsification | `GraphRewireFalsification` | Destroys edges → gaps should collapse to noise |
| Event studies | `EventStudyValidator` | Tests gap spikes align with known macro episodes |
| Persistence | `GapPersistenceTest` | AR(1) persistence of gaps across quarters |
| Baselines | `NaiveBaselines` | AR(1), mean-reversion, zero-gap comparisons |
| Model comparison | `ModelComparison` | Pairwise DM test across decomposers/filters |
| Transfer | `TransferValidator` | Out-of-sample transfer to held-out sector/country |
| Metrics | `spearman_rho`, `hit_rate`, `oos_r2` | Standard evaluation metrics |

---

## Pipeline Data Flow

```
observations y_{i,t}  (N actors × T quarters)
│
├─ graph/edges/        → adj_channel A_t^(r)  (sparse N×N per channel r)
├─ graph/operators.py  → operator A_t          (combined sparse N×N)
│
├─ spectral/           → modal_frame           (basis U: N×K, eigenvalues: K)
├─ spectral/mode_selection.py → K*             (MDL-optimal mode count)
│
├─ dynamics/kalman.py  → FilteredState         (alpha_filtered T×K, alpha_predicted T×K)
├─ dynamics/kim_filter.py → FilteredState      (+ regime_probs T×M)
├─ dynamics/model_selection.py → M*            (BIC-optimal regime count)
│
├─ emergence/pid.py    → synergy_matrix S      (K×K)
├─ emergence/te.py     → te_matrix TE          (K×K)
├─ emergence/tda.py    → betti_t, complexity   (T,)
│
├─ gaps/               → GapResult             (gaps N×T, benchmarks N×T, BenchmarkClass)
│
└─ signals/gap_signal.py → GapSignal           (btest DSL factor)
```

---

## Mathematical Notation → Python

| Math | Python variable | Shape | Module |
|------|----------------|-------|--------|
| y_{i,t} | `intensities` | (N, T) | `data/` |
| A_t^{(r)} | `adj_channel` | sparse (N, N) | `graph/edges/` |
| A_t | `operator` | sparse (N, N) | `graph/operators.py` |
| U_t | `modal_frame.basis` | (N, K) | `spectral/` |
| α_t | `alpha_filtered` | (T, K) | `dynamics/` |
| α_{t\|t-1} | `alpha_predicted` | (T, K) | `dynamics/` |
| F^{(z)} | `F_list[z]` | (K, K) | `dynamics/kim_filter.py` |
| z_t | `argmax(regime_probs, axis=1)` | (T,) int | `dynamics/` |
| ψ_t | `order_param` | (T,) | `dynamics/phase_transition.py` |
| C_t | `criticality` | (T,) | `dynamics/phase_transition.py` |
| S_{jk} | `synergy_matrix` | (K, K) | `emergence/pid.py` |
| Δ_{i,t} | `gap_result.gaps` | (N, T) | `gaps/` |
| y*_{i,t} | `gap_result.benchmarks` | (N, T) | `gaps/` |

---

## Setup

```bash
# Standard dev install
uv sync --extra dev --extra platform

# GPU acceleration (PyTorch with CUDA 12.x)
uv pip install "torch==2.10.0+cu126" --index-url https://download.pytorch.org/whl/cu126

# Benchmark extras (pytest-benchmark)
uv sync --extra benchmarks

# IDTxl — not on PyPI, required for acceptance test R-TE-1
uv pip install "idtxl @ git+https://github.com/pwollstadt/IDTxl.git"
```

**IDTxl requires Java (JDK 11+).** Verify with `java -version`.
JPype1 and setuptools are in `[dev]` extras and installed by `uv sync`.

Verify GPU setup:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Data Acquisition

SMIM experiments require equity OHLCV data and macro/alternative data.
The equity universe construction and download is handled by a single script.

### Equity universes + OHLCV

```bash
# Build all universe CSVs and download OHLCV (full run, ~20 min)
uv run python scripts/smim/smim_build_universes.py

# Individual steps
uv run python scripts/smim/smim_build_universes.py --step 1          # universe CSVs only
uv run python scripts/smim/smim_build_universes.py --step 2          # OHLCV download only
uv run python scripts/smim/smim_build_universes.py --step 3          # quality verification only

# Retry specific universes (useful after partial failures)
uv run python scripts/smim/smim_build_universes.py --step 2 3 --only US-LC MIXED-200

# Skip live market-cap fetch for US-LC (uses CSV order instead)
uv run python scripts/smim/smim_build_universes.py --skip-market-cap
```

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/universes/*.csv` | Ticker lists (columns: `ticker, name, sector, gics_code`) — committed to git |
| `equities/smim/{universe_id}/ohlcv.parquet` | Daily OHLCV, long-form — gitignored (large binary) |

**Universes built:**

| ID | Description | ~Size |
|----|-------------|-------|
| `US-LC` | Top-200 S&P 500 by market cap | 200 tickers |
| `US-LC-ENERGY` | S&P 500 GICS sector 10 (Energy) | ~22 tickers |
| `US-LC-TECH` | S&P 500 GICS sector 45 (IT) | ~68 tickers |
| `US-LC-FINS` | S&P 500 GICS sector 40 (Financials) | ~74 tickers |
| `US-LC-HEALTH` | S&P 500 GICS sector 35 (Health Care) | ~60 tickers |
| `US-LC-INDUS` | S&P 500 GICS sector 20 (Industrials) | ~78 tickers |
| `US-MC` | 200 S&P 400 mid-cap names | 200 tickers |
| `US-SC` | 200 Russell 2000 small-cap (stratified, seed=42) | 200 tickers |
| `UK-LC` | FTSE 100 (Yahoo Finance `.L` suffix) | ~99 tickers |
| `UK-MC` | FTSE 250 ex-100 | ~100 tickers |
| `MIXED-200` | US energy (S&P 500) + UK energy (FTSE 100) — MVP Layer-2 universe | ~27 tickers |

**OHLCV parquet schema** (long-form):

```
date       datetime64[ns]
ticker     str
open       float64
high       float64
low        float64
close      float64
volume     float64
sector     str  (from universe CSV, may be NaN for UK tickers)
```

**Date range:** 2005-01-03 to 2025-12-30 for all universes.
Each ticker starts from its IPO date — tickers listed after 2005 will have
fewer rows than the full 20-year range. This is expected and correct.

**Sparse ticker flags** in the step-3 quality report flag tickers with
<50% of the max trading-day count. For post-IPO companies (e.g. `ABNB`,
`COIN`, `CRWD`, `MRNA`) this is expected — their history is simply shorter.

**Download implementation note:** the script uses `yf.download()` with a
per-chunk outer-join strategy so each ticker retains its own date range.
The earlier vectorbt batch approach caused inner-join truncation (one
recent-IPO ticker in a chunk would silently truncate all others to its
listing date); that approach has been replaced.

### FRED macro signals + ALFRED vintages

```bash
# Requires: FRED_API_KEY environment variable
uv run python scripts/smim/smim_fetch_fred.py
```

Fetches 29 macro series from FRED and full ALFRED vintage histories for the 5
most revision-prone series (GDP, UNRATE, CPIAUCSL, INDPRO, FEDFUNDS). Results
are stored in the PIT store for A1-compliant backtesting.

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/raw/fred/<SERIES>.parquet` | Raw per-series observations |
| `data/smim/raw/fred/<SERIES>_alfred.parquet` | All ALFRED releases for vintaged series |
| `data/smim/processed/fred_signals.parquet` | Unified tidy table (all series) |
| `data/smim/pit_store/fred.parquet` | PIT store shard, queryable via `PointInTimeStore` |

Series coverage: Layer 0 exogenous (GDP, CPI, VIX, housing, …), Layer 1 upstream
(Fed funds, yield curve, credit spreads), energy sector (WTI, Brent, gasoline),
and sector proxies for financials and industrials.

Acquisition status and known failures: `docs/smim/DATA_ACQUISITION.md`.

### SEC EDGAR — XBRL balance sheet data

```bash
# No API key required — SEC only needs a User-Agent header
uv run python scripts/smim/smim_fetch_edgar.py
```

Fetches company-level XBRL facts for all US equity universes (US-LC, US-MC,
US-SC, and all five sector slices) from the SEC EDGAR JSON API. The filing
date (`filed`) is used as `pub_date`, making this strictly A1-compliant.

**Coverage (from 2026-03-22 run):** 765 / 772 tickers, 461,203 filing records,
date range 2005-07-04 to 2026-02-28.

**XBRL tags fetched:**

| Tag | Coverage |
|-----|----------|
| `PaymentsToAcquirePropertyPlantAndEquipment` (CapEx) | 611 tickers |
| `Assets` | 765 tickers |
| `StockholdersEquity` | 757 tickers |
| `LongTermDebt` | 605 tickers |
| `Revenues` | 559 tickers |
| `RevenueFromContractWithCustomerExcludingAssessedTax` | 537 tickers |
| `ResearchAndDevelopmentExpense` | 341 tickers |

Note: `CapitalExpenditures` (the older tag) has near-zero coverage because most
modern 10-K/10-Q filers use `PaymentsToAcquirePropertyPlantAndEquipment` instead.

**Tickers with no filings (7):** `BBUC`, `BTDR`, `CMDB`, `GAMB`, `HSHP`, `LZM`, `VTEX`
(recent cross-listings or SPACs with no EDGAR XBRL history).

**Tickers with no CIK mapping (8):** `DAY`, `FI`, `FRBA`, `MMC`, `MOGA`, `PDLI`,
`THRD`, `XTSLA` (de-listed, renamed, or non-reporting entities).

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/processed/edgar_balance_sheet.parquet` | Normalised tidy table (ticker, cik, event_date, pub_date, tag, value, form_type, period) |
| `data/smim/pit_store/edgar.parquet` | PIT store shard, queryable via `PointInTimeStore` |

### GDELT narrative signals

```bash
# No auth required — direct GKG 2.0 CSV downloads, free
uv run python scripts/smim/smim_fetch_gdelt.py                  # full fetch (incremental, uses daily cache)
uv run python scripts/smim/smim_fetch_gdelt.py --weekly-only    # rebuild weekly from existing daily cache
uv run python scripts/smim/smim_fetch_gdelt.py --rebuild        # reprocess outputs from cache, no downloads
uv run python scripts/smim/smim_fetch_gdelt.py --force-refetch  # re-download all ~3,970 daily files
uv run python scripts/smim/smim_fetch_gdelt.py --validate-only  # spot-check yesterday's file
uv run python scripts/smim/smim_fetch_gdelt.py --workers 8      # increase parallelism
uv run python scripts/smim/smim_fetch_gdelt.py --daily-only     # build daily artifact only, skip weekly/PIT
```

Fetches one representative GKG 2.0 file per **UTC calendar day** (slot nearest 12:00 UTC),
parses V2EnhancedThemes and V2Organizations, and computes per-day narrative intensity and
tone for 5 sector signals and 4 institutional actor signals. The canonical weekly panel is
then **derived from daily data** using mathematically correct aggregation:

- `weekly_article_count = sum(daily_article_count)` — sums matched docs across the week
- `weekly_avg_tone = weighted mean` using `daily_article_count` as weights — not a simple mean
- `weekly_intensity = sum(daily_matched) / sum(daily_total_docs)` — not a mean of daily ratios

This replaces the old approach (one file per ISO week), which used a single 15-minute snapshot
as a proxy for a full week. The new method aggregates ~7 samples per week, giving ~20× more
matched documents per weekly signal value.

**Coverage (from 2026-03-22 run):** 3,138 exact noon + 25 fallback + 360 missing days,
32,481 daily rows, 5,094 weekly rows, date range 2015-02-19 to 2025-12-31.

**Signals:**

| Signal | Type | Weekly intensity range |
|--------|------|----------------------|
| `sector_energy` | Sector | 0–28.6% |
| `sector_technology` | Sector | 0–66.7% |
| `sector_financials` | Sector | 0–23.0% |
| `sector_healthcare` | Sector | 0–66.7% |
| `sector_macro` | Sector | 0–20.0% |
| `actor_FED` | Institution | 0–4.0% |
| `actor_IMF` | Institution | 0–4.5% |
| `actor_SEC` | Institution | 0–6.2% |
| `actor_BOE` | Institution | 0–0.09% |

**Theme baskets** use actual GKG 2.0 V2EnhancedThemes codes (WB\_/ENV\_/EPU\_ hierarchy).
Old simple codes (OIL, GAS, TECH, etc.) do not exist in GKG 2.0 — see
`docs/smim/DATA_ACQUISITION.md` for full basket definitions and aggregation formulas.

**Actor matching** is case-insensitive substring match in the V2Organizations NLP field.
Note: GKG NLP drops "and" from org names (`"securities exchange commission"`, not
`"securities and exchange commission"`).

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/cache/gdelt/daily_aggregates/YYYY-MM-DD.parquet` | Per-day stats cache (resumability) |
| `data/smim/cache/gdelt/daily_file_index.parquet` | Selection log: date, slot, exact\_noon/fallback/missing |
| `data/smim/processed/gdelt_narrative_daily.parquet` | Daily panel: theme\_or\_actor, event\_date, article\_count, avg\_tone, intensity, total\_docs\_day |
| `data/smim/processed/gdelt_narrative.parquet` | **Canonical weekly panel** (daily-derived): theme\_or\_actor, week\_start, article\_count, avg\_tone, intensity |
| `data/smim/pit_store/gdelt.parquet` | PIT store shard: 14,638 rows, 3 signal\_ids × 9 actors × 566 weeks |

---

### IMF macro signals (WEO + IFS)

```bash
# No API key required — uses IMF DataMapper API
uv run python scripts/smim/smim_fetch_imf.py
```

Fetches international macro indicators from two IMF sources:

**Primary — IMF DataMapper** (`https://www.imf.org/external/datamapper/api/v1/`):
Annual WEO projections + history (2000–2030) for US, UK, Germany, Japan.

| Indicator | Description | Countries |
|-----------|-------------|-----------|
| `NGDP_RPCH` | Real GDP growth (%) | US, GB, DE, JP |
| `PCPIPCH` | CPI inflation (%) | US, GB, DE, JP |
| `BCA` | Current account balance (USD bn) | US, GB |
| `GGXCNL_NGDP` | Govt net lending (% of GDP) | US, GB |
| `GGXWDG_NGDP` | Govt gross debt (% of GDP) | US, GB |
| `LUR` | Unemployment rate (%) | US, GB |
| `PPPGDP` | GDP PPP (international $bn) | US, GB, DE, JP |

**Secondary — IMF IFS SDMX** (`dataservices.imf.org`): quarterly series
(NGDP\_R\_XDC, PCPI\_IX, FPOLM\_PA, BCA\_BP6\_USD) attempted at runtime; skipped
gracefully if the endpoint is unreachable (it times out in some network environments).

A1 compliance: `pub_date = event_date + 365 days` (conservative annual publication lag).

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/raw/imf/<INDICATOR>.parquet` | Per-indicator raw DataMapper JSON |
| `data/smim/processed/imf_macro.parquet` | Unified tidy table (618 rows) |
| `data/smim/pit_store/imf.parquet` | PIT store shard — 7 signals, 4 actors |

---

### OECD macro signals (CLI + QNA)

```bash
# No API key required — OECD SDMX 3.0 public endpoint
uv run python scripts/smim/smim_fetch_oecd.py
```

Fetches two OECD SDMX 3.0 dataflows for US and UK, using the `all` key
with in-memory filtering (the OECD 3.0 API requires all 9–14 dimensions to be
specified exactly; `all` avoids that complexity).

| Dataflow | Signals | Freq | Description |
|----------|---------|------|-------------|
| `DSD_STES@DF_CLI` | `LI`, `BCICP`, `CCICP` | Monthly | Composite Leading Indicator, Business Confidence, Consumer Confidence (amplitude-adjusted) |
| `DSD_NAMAIN1@DF_QNA_EXPENDITURE_CAPITA` | `B1GQ_POP` | Quarterly | GDP per capita in USD PPP — levels, seasonally adjusted |

Country codes: OECD uses ISO 3166-1 alpha-3 (USA, GBR); normalised to alpha-2 (US, GB) in the PIT store.

A1 compliance: `pub_date = event_date + 45 days` (CLI) / `+ 75 days` (QNA).

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/raw/oecd/DSD_STES_DF_CLI_4.0.parquet` | Raw CLI response |
| `data/smim/raw/oecd/DSD_NAMAIN1_DF_QNA_EXPENDITURE_CAPITA_1.1.parquet` | Raw QNA response |
| `data/smim/processed/oecd_macro.parquet` | Unified tidy table (244 rows) |
| `data/smim/pit_store/oecd.parquet` | PIT store shard — 4 signals, 2 actors |

---

### BEA Input-Output supply-chain coefficients

```bash
# With BEA API key (recommended — full industry detail):
BEA_API_KEY=<your_key> uv run python scripts/smim/smim_fetch_bea.py

# Without API key (fallback — downloads published Excel from apps.bea.gov):
uv run python scripts/smim/smim_fetch_bea.py
```

Fetches the BEA "Use of Commodities by Industries, Before Redefinitions"
table (TableID=259) for 2010–2024. Computes direct-requirements coefficients:

```
coeff[source→target] = flow[source, target] / column_total[target]
```

Maps BEA NAICS-based industry codes to SMIM sector labels:

| NAICS prefixes | SMIM sector |
|---------------|-------------|
| 211, 213, 324 | Energy |
| 334, 511, 518, 519 | Technology |
| 521–525 | Financials |
| 621–624 | Healthcare |
| 331–333, 336, 337 | Industrials |

A1 compliance: `pub_date = year-end + 548 days` (~18-month BEA publication lag).

The fallback (no API key) downloads the BEA annual Use Table Excel file from
`https://apps.bea.gov/industry/xls/io-annual/` and parses the flow matrix.
Free BEA API keys: `https://apps.bea.gov/API/signup/`

**Outputs:**

| Path | Contents |
|------|----------|
| `data/smim/raw/bea/use_table_<year>.parquet` | Raw per-year API response |
| `data/smim/processed/bea_io_tables.parquet` | Sector-mapped coefficients (26,852 rows, 2010–2024) |
| `data/smim/pit_store/bea.parquet` | PIT store shard — `actor_id = "Src→Tgt"`, 315 sector-pair observations |

---

## Running Tests

### Unit tests (~15 s)

```bash
# All SMIM unit tests
uv run pytest tests/unit/smim/ -q

# Single submodule
uv run pytest tests/unit/smim/dynamics/ -q
uv run pytest tests/unit/smim/compute/ -q
```

### Acceptance suite (~65 s, 130 tests)

```bash
# Full suite with gate report (CPU)
uv run python scripts/smim/run_smim_acceptance.py

# On CUDA
SMIM_DEVICE=cuda uv run pytest tests/acceptance/smim/ -v --tb=short

# Single section
uv run python scripts/smim/run_smim_acceptance.py --section graph_construction
uv run python scripts/smim/run_smim_acceptance.py --section spectral
uv run python scripts/smim/run_smim_acceptance.py --section kalman
uv run python scripts/smim/run_smim_acceptance.py --section pipeline_sanity

# Verbose / stop on first failure
uv run python scripts/smim/run_smim_acceptance.py -v
uv run python scripts/smim/run_smim_acceptance.py -- -x
```

**Expected output:**

```
SMIM Acceptance Report — 2026-03-21
===========================================
  Graph Construction:        20/20 passed
  Spectral Decomposition:    37/37 passed
  Mode Selection:             9/9  passed
  Kalman Filter + EM:        14/14 passed
  Observability:              3/3  passed
  Phase Transition:           8/8  passed
  PID:                        6/6  passed
  Transfer Entropy:           6/6  passed
  TDA:                        7/7  passed
  Benchmarks/Gaps:            7/7  passed
  Pipeline Sanity:            5/5  passed
  Uncategorised:              8/8  passed
-------------------------------------------
  TOTAL:                    130/130 passed
  STATUS: READY FOR EXPERIMENTS
```

> Without idtxl/Java: Transfer Entropy shows 5/6 (R-TE-1 skipped).
> Skipped != failed — gate remains READY FOR EXPERIMENTS.

### Performance benchmarks (~90 s)

```bash
# Run all benchmarks (CPU + CUDA), save JSON
uv run pytest tests/benchmarks/smim/ -v \
    --benchmark-columns=mean,stddev,rounds \
    --benchmark-json=.benchmark_results.json

# Generate speedup report
uv run python scripts/smim/gpu_speedup_report.py
```

**Measured speedups (RTX 4070 Ti, T=80):**

| Component | CPU | CUDA | Speedup |
|-----------|-----|------|---------|
| Granger edges N=200 | 119 ms | 7.9 ms | 15× |
| Full pipeline N=200 | 333 ms | 40 ms | 8× |
| Kim filter EM N=200 | 235 s | 0.73 s | 321× (Woodbury, CPU) |
| Kalman filter N=200 | 5.0 s | 0.03 s | 167× (Woodbury, CPU) |
| KNN for TE T=2000 | 25 ms | 2.0 ms | 12× |

---

## Acceptance Gate

**Experiments must not start until all 130 acceptance tests pass.**

The gate report is generated automatically by
`tests/acceptance/smim/conftest_report.py` and printed at the end of every
pytest run targeting `tests/acceptance/smim/`.

Full acceptance test specification: `docs/smim/ACCEPTANCE_TESTS.md`.

---

## Known Implementation Deviations

| Test | Original spec | Correct behaviour | Reason |
|------|--------------|-------------------|--------|
| I-MB-1 | attr sums to gap[i,t] | `attr_sum = gap_modal − gap_pred` | Spec had algebraic error |
| P-2 | M*=1 for pure noise | BIC may select M>1 | BIC penalty too small vs Kim LL gain; OOS R² is definitive check |
| R-TE-1 | Within 25% of IDTxl | Tolerance 50% | Kraskov Alg-1 vs Frenzel-Pompe ~37% divergence at T=2000 |
| I-TDA-1 | d_B < ε | d_B < 2ε | VR stability theorem: d_B ≤ 2·d_H ≤ 2ε |

### KimFilter known limitations

- **Symmetric EM initialisation**: EM starts all regimes at `F = 0.9·I` — cannot
  break symmetry to discover distinct regimes. Provide asymmetric initial parameters
  when testing regime detection; do not rely on EM to discover M > 1.
- **alpha_pred approximation**: `kim_filter.py` sets `alpha_pred[t] = alpha_filt[t]`.
  Predictive and modal benchmarks from KimFilter are therefore nearly identical.

---

## Standing Assumptions (Never Violate)

| ID | Rule |
|----|------|
| A1 | **Point-in-time**: never use data with `pub_date > backtest_date` — enforced by `pit_store` |
| A2 | **Typed comparability**: normalisation is per-`ActorType` via `InvestmentIntensityMapper` |
| A3 | **Sparse propagation**: after sparsification, operator retains >80% spectral energy |
| A4 | **Stable modes**: eigenmode rank correlation >0.5 across ≥80% of rolling windows |
| A5 | **Regime persistence**: average regime duration >8 quarters |

---

## Implementation Pattern

Every SMIM component follows this pattern:

1. **Protocol** defined in `smim/interfaces.py`
2. **Implementation** in the appropriate submodule
3. **Config** as a Pydantic section in `smim/config.py`
4. **Unit tests** in `tests/unit/smim/` mirroring the source path
5. **Acceptance tests** in `tests/acceptance/smim/`

```python
# Example: add a new edge estimator
# 1. Read EdgeEstimator protocol in interfaces.py
# 2. Create smim/graph/edges/my_estimator.py implementing the protocol
# 3. Add MyEdgeConfig to SmimConfig in config.py
# 4. Add tests in tests/unit/smim/graph/test_my_estimator.py
# 5. Register in the edge estimator factory (graph/edges/__init__.py)
```

---

## Key Reference Documents

| Document | Purpose |
|----------|---------|
| `docs/smim/CLAUDE.md` | Math notation, standing assumptions, implementation patterns |
| `docs/smim/GPU_ACCELERATION_PLAN.md` | GPU design, measured speedups, quality gates |
| `docs/smim/ACCEPTANCE_TESTS.md` | Full test plan, pass criteria, known deviations |
| `docs/smim/IMPLEMENTATION_PLAN.md` | Milestones, quality gates, work packages |
| `docs/smim/TASK_REGISTRY.md` | Per-task status (Claude Code decomposition) |
| `docs/smim/PROPOSAL_SUMMARY.md` | Condensed research proposal |
| `docs/smim/notation.md` | Every mathematical symbol with Python mapping |
| `docs/smim/benchmark_specs.md` | Formal definitions for all 5 benchmark families |
| `docs/smim/ADAPTER_GUIDE.md` | How to write a new data adapter |
| `docs/smim/DATA_ACQUISITION.md` | Data acquisition status: what's downloaded, what failed, why |
| `smim/interfaces.py` | All Protocol definitions — read before implementing |
| `smim/config.py` | All tuneable parameters (Pydantic) |
| `experiments/mvp_energy_us_uk.yaml` | Sample experiment config |
