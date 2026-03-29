# SMIM Experiment Execution Plan

## Companion to: EXPERIMENT_OBJECTIVES.md
## Assumes: All development milestones (WP0–WP6) complete, pipeline operational

---

## Experiment Programme Structure

The programme has four phases, executed in order. Each phase produces results that
inform the next phase's configuration. Total estimated duration: 8–12 weeks of
compute + analysis time.

```
Phase A: Anchor Experiments (weeks 1–3)
  → Establish baseline on the MVP domain, validate the measurement stack

Phase B: Ablation Experiments (weeks 3–6)
  → Component value decomposition, signal attribution, robustness boundaries

Phase C: Transfer Experiments (weeks 6–9)
  → Cross-sector, cross-cap, cross-geography, cross-period

Phase D: Economic Validation (weeks 9–12)
  → Persistence, correction, diffusion, event alignment, expert comparison
```

---

## Data Universe Definitions

### Equity Actor Universes

All equity universes use SEC EDGAR XBRL (US) or Companies House (UK) for
balance-sheet data, Yahoo Finance / exchange feeds for market data, and the
btest parquet pipeline for OHLCV.

| Universe ID | Description | N (approx) | Index/Filter | Data Regime | Coverage |
|-------------|-------------|------------|--------------|-------------|----------|
| `US-LC` | US large cap | 200 | S&P 500, top 200 by market cap | Gold | 2005–2025 |
| `US-MC` | US mid cap | 200 | S&P 400 constituents | Silver | 2007–2025 |
| `US-SC` | US small cap | 200 | Russell 2000, random stratified sample | Bronze | 2010–2025 |
| `UK-LC` | UK large cap | 100 | FTSE 100 | Gold | 2005–2025 |
| `UK-MC` | UK mid cap | 100 | FTSE 250 (ex-FTSE 100) | Silver | 2008–2025 |
| `US-LC-ENERGY` | US large cap, energy sector | 22 | S&P 500 Energy (GICS 10) — actual count | Gold | 2005–2025 |
| `US-LC-TECH` | US large cap, technology | 68 | S&P 500 IT (GICS 45) — actual count | Gold | 2005–2025 |
| `US-LC-FINS` | US large cap, financials | 74 | S&P 500 Financials (GICS 40) — actual count | Gold | 2005–2025 |
| `US-LC-HEALTH` | US large cap, healthcare | 60 | S&P 500 Health Care (GICS 35) — actual count | Gold | 2005–2025 |
| `US-LC-INDUS` | US large cap, industrials | 78 | S&P 500 Industrials (GICS 20) — actual count | Gold | 2005–2025 |
| `MIXED-200` | Multi-sector, cross-geography MVP universe | 103 (91 equity, 12 institutional) | 6 sectors (energy, tech, fins, health, industrials, diversified), US+UK, all layers (L0–L2); expanded 2026-03-29 via RP3 | Gold/Silver mix | 2005–2025 |

### Institutional Actor Sets (Layers 0–1)

| Set ID | Actors | Data Sources |
|--------|--------|--------------|
| `INST-US` | Fed, SEC, Treasury, FDIC, EPA, DOE, top 10 US think tanks | FRED, Federal Register, GDELT, central bank speech corpora |
| `INST-UK` | BoE, FCA, OFGEM, HMRC, top 5 UK think tanks | BoE API, Companies House, GDELT |
| `INST-INTL` | IMF, OECD, BIS, World Bank, ECB | IMF SDMX, OECD SDMX, BIS SDMX |
| `INST-MINIMAL` | Fed + BoE + IMF only (minimal set for fast experiments) | FRED, BoE, IMF |

### Signal Feed Configurations

| Feed ID | Signal Families Included | Data Sources | Notes |
|---------|------------------------|--------------|-------|
| `FULL` | All 6 families (macro, policy, market, balance sheet, narrative, network) | FRED, EDGAR, GDELT, BEA I/O, Yahoo Finance, central bank APIs | Maximum signal coverage |
| `NO-NARRATIVE` | All except narrative | FRED, EDGAR, BEA I/O, Yahoo Finance | Tests narrative marginal value |
| `NO-MARKET` | All except market pricing | FRED, EDGAR, GDELT, BEA I/O | Tests whether market signals dominate |
| `NO-NETWORK` | All except network position | FRED, EDGAR, GDELT, Yahoo Finance | Tests supply-chain edge value |
| `MACRO-ONLY` | Macro/fundamental only | FRED, IMF, OECD | Minimal signal baseline |
| `MACRO+MARKET` | Macro + market pricing | FRED, Yahoo Finance | Common quant baseline |
| `MACRO+NARRATIVE` | Macro + narrative | FRED, GDELT | Tests narrative on top of macro |
| `BALANCE+MACRO` | Balance sheet + macro | FRED, EDGAR | Fundamental-only baseline |

### Time Period Configurations

| Period ID | Train Window | Test Window | Notes |
|-----------|-------------|-------------|-------|
| `FULL-ROLL` | Rolling 10yr | 1yr ahead | Primary evaluation; 10 non-overlapping test windows |
| `PRE-GFC` | 2005–2007 | 2008–2009 | Stress test: can pre-crisis data predict crisis? |
| `POST-GFC` | 2010–2014 | 2015–2019 | Stable expansion test |
| `PRE-COVID` | 2015–2019 | 2020–2021 | Regime break: COVID |
| `POST-COVID` | 2020–2022 | 2023–2025 | Post-crisis normalisation |
| `RECENT` | 2018–2023 | 2024–2025 | Most recent OOS, minimal look-ahead risk |

---

## Phase A: Anchor Experiments (Weeks 1–3)

**Goal**: Establish baseline performance on the MVP domain. Validate that the
measurement stack works end-to-end. Produce the reference numbers all subsequent
experiments compare against.

### Experiment A1: MVP Full Pipeline

```yaml
experiment_id: A1-MVP-FULL
description: "Full pipeline on MVP domain — the reference benchmark"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL
pipeline_depth: full  # all components active
spectral_method: [best from WP3]  # use the G3-selected method
regimes: [1, 2, 3]  # compare
emergence: true
benchmarks: [predictive, modal, emergence_aware]
falsification: all_7_tests
measurements: [L1, L2, L3, L4, L5]  # full measurement stack
```

**Expected output**: baseline OOS R², component ΔR²s, falsification results,
runtime profile. This is the "full score" the framework achieves under ideal conditions.

**Decision**: if A1 OOS R² ≤ 0 against random walk, stop and diagnose before
proceeding. The framework doesn't work on its own MVP domain.

### Experiment A2: Naïve Baselines

```yaml
experiment_id: A2-BASELINES
description: "All baseline models on the same data as A1"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL
models:
  - historical_mean
  - random_walk
  - sector_mean
  - ar1_per_actor
  - dynamic_factor_model_k5
  - dynamic_factor_model_k10
  - var_bic
  - symmetric_laplacian_spectral
measurements: [L1]
```

**Expected output**: baseline performance for every comparison model.
All subsequent ΔR² numbers are computed against these.

### Experiment A3: Measurement Stack Validation

```yaml
experiment_id: A3-STACK-VALIDATION
description: "Verify measurement stack produces sensible numbers"
universe: US-LC (subset of 50)
institutions: INST-MINIMAL
signals: MACRO+MARKET
period: RECENT
pipeline_depth: full
checks:
  - verify_all_L1_metrics_finite
  - verify_component_delta_r2_sums_approximately
  - verify_falsification_runs_complete_for_B100
  - verify_runtime_profiling_captures_all_components
  - verify_results_database_schema_correct
```

**Expected output**: confirmation that the measurement machinery works. Fix any
bugs before Phase B. This experiment is fast (50 actors, short period, minimal
institutions).

### Experiment A4: Computational Scaling Profile

```yaml
experiment_id: A4-SCALING
description: "Runtime and memory as a function of N"
universe: [US-LC subset N=20, N=50, N=100, N=200, N=500]
institutions: INST-MINIMAL
signals: MACRO+MARKET
period: RECENT (single window)
pipeline_depth: full
measurements: [L5]  # computational efficiency only
```

**Expected output**: scaling exponents for each component. Determines whether
Phase B/C experiments are computationally feasible at planned N.

---

## Phase B: Ablation Experiments (Weeks 3–6)

**Goal**: Determine which components earn their complexity, which signals
matter, and where the framework breaks.

### Experiment B1: Component Layer Ablation

The core experiment. Runs the pipeline at every depth level and measures
incremental value.

```yaml
experiment_id: B1-COMPONENT-ABLATION
description: "Layer-by-layer pipeline depth ablation"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL

iterations:
  - depth: L0  # naïve baselines (from A2, reused)
  - depth: L1  # graph only (spectral factors from graph, no state-space)
    config: {pipeline: graph_factors_only}
  - depth: L2  # graph + spectral compression
    config: {pipeline: spectral_factors, regimes: 1, emergence: false}
  - depth: L3  # + regime switching
    config: {pipeline: spectral_factors, regimes: best_from_A1, emergence: false}
  - depth: L4  # + emergence diagnostics
    config: {pipeline: spectral_factors, regimes: best_from_A1, emergence: true, phase_transition: false}
  - depth: L5  # + phase transition (full pipeline = A1)
    config: {pipeline: full}

measurements: [L1, L2]
```

**Expected output**: the component value table:

| Layer Added | OOS R² | ΔR² | p-value (DM test vs previous) |
|-------------|--------|------|-------------------------------|
| L0 (naïve) | — | — | — |
| L1 (+graph) | ? | ? | ? |
| L2 (+spectral) | ? | ? | ? |
| ... | | | |

**Decision**: any layer with ΔR² < 0.5pp and DM p > 0.10 is flagged as
"marginal" for this condition. Still tested in other conditions.

### Experiment B2: Spectral Method Comparison

```yaml
experiment_id: B2-SPECTRAL-METHODS
description: "Head-to-head spectral decomposition comparison"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL

iterations:
  - spectral_method: schur
  - spectral_method: polar
  - spectral_method: hermitian_dilation
  - spectral_method: directed_variation
  - spectral_method: dmd
  - spectral_method: extended_dmd
  - spectral_method: pca  # symmetric baseline

# For each: same pipeline (single regime first, then best regime count)
measurements: [L1, L2, L5]
```

**Expected output**: method × metric comparison table.
Tests H2a (directed > symmetric) and H2b (polar/dilation > Schur on ill-conditioned).

### Experiment B3: Signal Family Leave-One-Out

```yaml
experiment_id: B3-SIGNAL-LOO
description: "Leave-one-out signal family ablation"
universe: MIXED-200
institutions: INST-US + INST-UK
period: FULL-ROLL
pipeline_depth: full

iterations:
  - signals: FULL                 # reference
  - signals: NO-NARRATIVE         # drop narrative
  - signals: NO-MARKET            # drop market
  - signals: NO-NETWORK           # drop network position
  - signals: FULL minus policy    # drop policy (custom config)
  - signals: FULL minus balance   # drop balance sheet (custom config)
  - signals: FULL minus macro     # drop macro (custom config)

measurements: [L1, L2]
```

**Expected output**: dispensability table — R² loss when each family is removed.
Tests Q4: which data sources drive performance?

### Experiment B4: Signal Family Leave-One-In

```yaml
experiment_id: B4-SIGNAL-LOI
description: "Leave-one-in signal family marginal value"
universe: MIXED-200
institutions: INST-US + INST-UK
period: FULL-ROLL
pipeline_depth: full

iterations:
  - signals: MACRO-ONLY
  - signals: MACRO+MARKET
  - signals: MACRO+NARRATIVE
  - signals: BALANCE+MACRO
  - signals: MACRO-ONLY + network position only
  - signals: MACRO-ONLY + policy only

measurements: [L1, L2]
```

**Expected output**: marginal value table — R² gain when each family is added
to the macro baseline. Combined with B3, reveals both marginal value and dispensability.

### Experiment B5: Signal × Component Interaction

```yaml
experiment_id: B5-SIGNAL-COMPONENT-INTERACTION
description: "Does narrative only matter with emergence? Does market only matter with regimes?"
universe: MIXED-200
institutions: INST-US + INST-UK
period: FULL-ROLL

# 2×2×2 factorial: narrative(yes/no) × emergence(yes/no) × regime(yes/no)
iterations:
  - signals: FULL,            emergence: true,  regimes: best
  - signals: FULL,            emergence: true,  regimes: 1
  - signals: FULL,            emergence: false, regimes: best
  - signals: FULL,            emergence: false, regimes: 1
  - signals: NO-NARRATIVE,    emergence: true,  regimes: best
  - signals: NO-NARRATIVE,    emergence: true,  regimes: 1
  - signals: NO-NARRATIVE,    emergence: false, regimes: best
  - signals: NO-NARRATIVE,    emergence: false, regimes: 1
  # Repeat for NO-MARKET if compute allows

measurements: [L1, L2]
```

**Expected output**: interaction effects. Tests H1c (narrative × regime interaction)
and H4a (emergence × signal interaction).

### Experiment B6: Robustness — Actor Count Sweep

```yaml
experiment_id: B6-N-SWEEP
description: "Performance as a function of actor universe size"
institutions: INST-MINIMAL
signals: MACRO+MARKET
period: RECENT
pipeline_depth: full

iterations:
  - universe: US-LC subset N=20
  - universe: US-LC subset N=50
  - universe: US-LC subset N=100
  - universe: US-LC (full, N=200)
  - universe: US-LC + US-MC (N=400)

measurements: [L1, L3, L5]
```

**Expected output**: performance and runtime as functions of N.
Identifies the "cliff" where components break (Q5).

### Experiment B7: Robustness — Time Series Length Sweep

```yaml
experiment_id: B7-T-SWEEP
description: "Performance as a function of training window length"
universe: US-LC (N=200)
institutions: INST-US
signals: FULL
pipeline_depth: full

iterations:
  - period: train=5yr,  test=2024-2025
  - period: train=8yr,  test=2024-2025
  - period: train=10yr, test=2024-2025
  - period: train=15yr, test=2024-2025
  - period: train=20yr, test=2024-2025

measurements: [L1, L3]
```

**Expected output**: minimum training window for each component.
H3b: regime switching should add nothing with short stable windows.

### Experiment B8: Robustness — Noise Injection

```yaml
experiment_id: B8-NOISE
description: "Performance under increasing signal noise"
universe: US-LC (N=200)
institutions: INST-US
signals: FULL
period: FULL-ROLL
pipeline_depth: full

iterations:
  - noise_level: 0.0   # clean (= A1 reference)
  - noise_level: 0.1   # 10% Gaussian noise added to all signals
  - noise_level: 0.2
  - noise_level: 0.3
  - noise_level: 0.5
  - noise_level: 1.0   # noise equal to signal magnitude

measurements: [L1, L3]
```

**Expected output**: degradation curve. Is it graceful or cliff?
At what noise level does each component stop adding value?

### Experiment B9: Robustness — Edge Degradation

```yaml
experiment_id: B9-EDGE-DEGRADE
description: "Performance as graph edges are progressively corrupted"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL

iterations:
  - edge_corruption: 0.0   # true estimated edges
  - edge_corruption: 0.1   # randomly flip 10% of edges
  - edge_corruption: 0.2
  - edge_corruption: 0.3
  - edge_corruption: 0.5
  - edge_corruption: 1.0   # completely random graph

measurements: [L1, L2, L3]
```

**Expected output**: how sensitive is performance to graph misspecification?
If performance degrades only slightly at 30% corruption, the framework is robust.
If it collapses at 10%, graph estimation quality is critical.

### Experiment B10: Regime Count Sensitivity

```yaml
experiment_id: B10-REGIME-SWEEP
description: "Performance across regime counts, with and without MDL selection"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL

iterations:
  - regimes: 1
  - regimes: 2
  - regimes: 3
  - regimes: 4
  - regimes: 5
  - regimes: 6
  - regimes: auto_mdl  # MDL-selected (= A1 reference)

measurements: [L1, L2]
```

**Expected output**: OOS R² vs regime count curve. Validates MDL selection.
Tests H3a/H3b: regime switching value is period-dependent.

---

## Phase C: Transfer Experiments (Weeks 6–9)

**Goal**: Test generalisation across sectors, geographies, cap tiers, and time periods.

### Experiment C1: Cross-Sector Transfer (Zero-Shot)

Train on energy, test on other sectors WITHOUT re-estimating parameters.

```yaml
experiment_id: C1-SECTOR-TRANSFER-ZERO
description: "Apply energy-trained model to other sectors without re-training"
train_universe: US-LC-ENERGY
train_institutions: INST-US
train_signals: FULL
train_period: 2010-2022

test_iterations:
  - test_universe: US-LC-TECH,   test_period: 2023-2025
  - test_universe: US-LC-FINS,   test_period: 2023-2025
  - test_universe: US-LC-HEALTH, test_period: 2023-2025
  - test_universe: US-LC-INDUS,  test_period: 2023-2025

transfer_mode: zero_shot  # no re-estimation
measurements: [L1, L3]
```

**Expected output**: transfer score per sector. Tests H5a.

### Experiment C2: Cross-Sector Transfer (Fine-Tuned)

Same as C1 but allow re-estimation of edge weights and channel weights
while keeping modal structure and regime dynamics fixed.

```yaml
experiment_id: C2-SECTOR-TRANSFER-FINETUNE
description: "Transfer with selective re-estimation"
# Same setup as C1, but:
transfer_mode: fine_tune
re_estimate: [edge_weights, channel_weights]
freeze: [modal_structure, regime_dynamics, K_star, M_star]
```

**Expected output**: how much does fine-tuning help? Which components needed
re-estimation? This produces the "transfer recipe" for new sectors.

### Experiment C3: Cross-Cap Transfer

```yaml
experiment_id: C3-CAP-TRANSFER
description: "Large cap model applied to mid and small cap"
train_universe: US-LC
train_period: 2010-2022

test_iterations:
  - test_universe: US-MC, transfer_mode: zero_shot
  - test_universe: US-MC, transfer_mode: fine_tune
  - test_universe: US-SC, transfer_mode: zero_shot
  - test_universe: US-SC, transfer_mode: fine_tune

measurements: [L1, L3]
```

**Expected output**: tests H5b (mid cap ≥70% retention) and H5c (small cap <50%
due to data degradation). The gap between zero-shot and fine-tuned reveals
what breaks across cap tiers.

### Experiment C4: Cross-Geography Transfer

```yaml
experiment_id: C4-GEO-TRANSFER
description: "US model applied to UK"
train_universe: US-LC
train_institutions: INST-US
train_period: 2010-2022

test_iterations:
  - test_universe: UK-LC, test_institutions: INST-UK, transfer_mode: zero_shot
  - test_universe: UK-LC, test_institutions: INST-UK, transfer_mode: fine_tune

measurements: [L1, L3]
```

**Expected output**: which components are geography-specific?
Expectation: macro regimes transfer, edge weights don't, institutional actors
need full re-specification.

### Experiment C5: Cross-Period Transfer (Structural Break)

```yaml
experiment_id: C5-PERIOD-TRANSFER
description: "Models trained in one era applied to another"

iterations:
  - train: PRE-GFC,    test: POST-GFC     # pre-crisis → post-crisis
  - train: POST-GFC,   test: PRE-COVID    # expansion → late expansion
  - train: PRE-COVID,  test: POST-COVID   # pre-COVID → pandemic
  - train: POST-COVID, test: RECENT       # pandemic → normalisation

universe: US-LC
institutions: INST-US
signals: FULL
transfer_mode: zero_shot
measurements: [L1, L3]
```

**Expected output**: which era transitions cause performance collapse?
This is the harshest transfer test — regime dynamics should fail across
structural breaks, but modal structure might survive.

### Experiment C6: Data Regime Degradation

Simulates moving from gold to bronze data by progressively removing signals.

```yaml
experiment_id: C6-DATA-REGIME
description: "Performance under progressively degraded data coverage"
universe: US-LC (N=200)
institutions: INST-US
period: FULL-ROLL
pipeline_depth: full

iterations:
  - data_regime: gold    # all signals, daily+ frequency
    signals: FULL
  - data_regime: silver  # remove daily market data, keep quarterly+
    signals: FULL minus intraday, quarterly frequency only
  - data_regime: bronze  # remove balance sheet detail, keep macro + basic market
    signals: MACRO+MARKET, quarterly
  - data_regime: sparse  # macro only, annual frequency
    signals: MACRO-ONLY, annual

measurements: [L1, L2, L3]
```

**Expected output**: the data regime boundary for each component.
Tests Q2: at what coverage level does each layer stop helping?

---

## Phase D: Economic Validation (Weeks 9–12)

**Goal**: Determine whether gaps are economically meaningful, not just
statistically significant.

### Experiment D1: Gap Persistence Analysis

```yaml
experiment_id: D1-PERSISTENCE
description: "Do gaps mean-revert, and how fast?"
universe: US-LC
institutions: INST-US
signals: FULL
period: FULL-ROLL
pipeline_depth: full
benchmarks: [predictive, structural, emergence_aware]

analysis:
  - estimate_half_life_ar1  # fit AR(1) on gap series per actor
  - estimate_half_life_panel  # panel AR(1) with actor fixed effects
  - top_decile_tracking  # track top 10% |Δ| actors over next 8 quarters
  - quintile_transition_matrix  # gap quintile persistence quarter to quarter

measurements: [L4]
```

**Expected output**: half-life estimates per benchmark class. Tests H6a (4–12Q half-life).
If half-life < 2Q, gaps are noise. If > 16Q, gaps are structural and uninteresting for timing.

### Experiment D2: Correction Prediction

```yaml
experiment_id: D2-CORRECTION
description: "Do large gaps predict subsequent repricing or capex revision?"
universe: US-LC
institutions: INST-US
signals: FULL
period: FULL-ROLL
pipeline_depth: full

outcome_variables:
  - next_4q_capex_revision    # change in CapEx/Assets in next 4 quarters
  - next_4q_total_return      # equity total return
  - next_4q_credit_spread     # CDS or credit spread change
  - next_4q_analyst_revision  # consensus EPS revision

analysis:
  - portfolio_sort  # sort actors into gap quintiles, track outcome variable
  - regression      # panel regression: outcome = β₀ + β₁ Δ_{i,t} + controls
  - logistic        # P(correction > median | large gap) vs P(correction > median | small gap)

measurements: [L4]
```

**Expected output**: does the top gap quintile predict worse subsequent outcomes?
Tests H6b. The key test is whether the gap adds information beyond simple momentum
or value signals.

### Experiment D3: Graph Diffusion Prediction

```yaml
experiment_id: D3-DIFFUSION
description: "Do upstream gaps predict downstream gaps along graph edges?"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL
pipeline_depth: full

analysis:
  - layer_lead_lag  # does Layer 1 gap at t predict Layer 2 gap at t+1?
  - neighbour_contagion  # does actor i's gap predict graph-neighbour j's gap?
  - channel_specific_diffusion  # which edge channel carries gap diffusion?
  - diffusion_speed  # lag (quarters) from upstream to downstream gap emergence

measurements: [L4]
```

**Expected output**: evidence for top-down propagation of misallocation.
If gaps in Layer 1 (central banks, regulators) predict Layer 2 (firms) gaps
1–2 quarters ahead, the framework captures something real about transmission.

### Experiment D4: Historical Event Alignment

```yaml
experiment_id: D4-EVENTS
description: "Do gaps spike around known misallocation episodes?"
universe: US-LC
institutions: INST-US
signals: FULL
pipeline_depth: full
benchmarks: [predictive, emergence_aware]

events:
  - name: "Energy over-investment 2012-2014"
    date: 2013-Q2
    sector: energy
    expected_sign: positive  # over-investment
    window_quarters: 4

  - name: "Pre-GFC bank leverage"
    date: 2007-Q2
    sector: financials
    expected_sign: positive
    window_quarters: 4

  - name: "Post-GFC tech under-investment"
    date: 2010-Q1
    sector: technology
    expected_sign: negative  # under-investment
    window_quarters: 4

  - name: "COVID capital reallocation"
    date: 2020-Q2
    sector: all
    expected_sign: mixed  # both over and under
    window_quarters: 2

  - name: "2022 energy super-profits"
    date: 2022-Q2
    sector: energy
    expected_sign: positive
    window_quarters: 4

  - name: "Fed tightening 2022"
    date: 2022-Q3
    sector: all
    expected_sign: mixed
    window_quarters: 4

  - name: "AI investment surge 2023"
    date: 2023-Q3
    sector: technology
    expected_sign: positive
    window_quarters: 4

  - name: "Regional bank stress 2023"
    date: 2023-Q1
    sector: financials
    expected_sign: negative
    window_quarters: 2

analysis:
  - gap_level_around_event  # average |Δ| in sector around event vs control periods
  - gap_sign_alignment      # does Δ sign match expected direction?
  - emergence_spike         # do C_t or T_t spike near events?
  - modal_attribution       # which modes activate around events?

measurements: [L4]
```

**Expected output**: event study table with hit rate. If ≥6/8 events show
significant gap alignment, the framework captures known misallocation patterns.

### Experiment D5: Benchmark Divergence

```yaml
experiment_id: D5-BENCHMARK-DIVERGENCE
description: "How different are the benchmark classes from each other?"
universe: US-LC
institutions: INST-US
signals: FULL
period: FULL-ROLL
pipeline_depth: full

analysis:
  - pairwise_correlation  # corr(Δ^pred, Δ^str), corr(Δ^pred, Δ^em), etc.
  - mean_absolute_divergence  # mean |Δ^pred - Δ^str| per actor type
  - divergence_by_regime  # does divergence increase during crises?
  - actors_with_sign_disagreement  # how often do benchmarks disagree on direction?

measurements: [L4]
```

**Expected output**: if all benchmarks produce the same gaps (correlation >0.95),
the benchmark distinction is academic and only the predictive benchmark matters.
Tests H6c (divergence > 0.3σ). If divergence increases during crises, the
structural benchmark captures something real about distortion channels.

### Experiment D6: Emergence Timing

```yaml
experiment_id: D6-EMERGENCE-TIMING
description: "Do emergence diagnostics provide genuine early warning?"
universe: MIXED-200
institutions: INST-US + INST-UK
signals: FULL
period: FULL-ROLL
pipeline_depth: full

analysis:
  - criticality_lead_time
    # For each known regime transition, measure:
    # 1. When C_t first exceeds 2× its rolling median
    # 2. When VIX first exceeds 2× its rolling median
    # 3. Lead time = (VIX signal date) - (C_t signal date)

  - synergy_clustering
    # Does total synergy (sum of S matrix) increase before misallocation clusters?
    # Compare S_total in [-8Q, -1Q] before cluster vs [-16Q, -9Q]

  - tda_structural_break
    # Does T_t or Wasserstein distance spike before structural breaks?
    # Compare against standard change-point detection (CUSUM, Bai-Perron)

  - emergence_vs_volatility
    # Head-to-head: C_t vs VIX, T_t vs realised vol, synergy vs correlation
    # Do emergence diagnostics provide information beyond standard risk measures?

measurements: [L4]
```

**Expected output**: lead time in quarters for each diagnostic vs each standard measure.
Tests H3c, H4a, H4b, H4c. This is the hardest test — if emergence diagnostics
merely replicate VIX/volatility signals, they don't justify their complexity.

---

## Experiment Execution Matrix (Summary)

| ID | Phase | Universe | Signals | Iterations | Est. Pipeline Runs | Priority |
|----|-------|----------|---------|------------|-------------------|----------|
| A1 | A | MIXED-200 | FULL | 1 (×10 windows) | 10 | Critical |
| A2 | A | MIXED-200 | FULL | 8 baselines × 10 | 80 | Critical |
| A3 | A | US-LC-50 | MACRO+MKT | 1 | 1 | Critical |
| A4 | A | US-LC varying | MACRO+MKT | 5 | 5 | High |
| B1 | B | MIXED-200 | FULL | 5 depths × 10 | 50 | Critical |
| B2 | B | MIXED-200 | FULL | 7 methods × 10 | 70 | High |
| B3 | B | MIXED-200 | varies | 6 configs × 10 | 60 | High |
| B4 | B | MIXED-200 | varies | 6 configs × 10 | 60 | Medium |
| B5 | B | MIXED-200 | varies | 8 factorial × 10 | 80 | Medium |
| B6 | B | US-LC varying | MACRO+MKT | 5 | 5 | High |
| B7 | B | US-LC | FULL | 5 | 5 | Medium |
| B8 | B | US-LC | FULL | 6 × 10 | 60 | Medium |
| B9 | B | MIXED-200 | FULL | 6 × 10 | 60 | High |
| B10 | B | MIXED-200 | FULL | 7 × 10 | 70 | Medium |
| C1 | C | Cross-sector | FULL | 4 | 4 | High |
| C2 | C | Cross-sector | FULL | 4 | 4 | High |
| C3 | C | Cross-cap | FULL | 4 | 4 | High |
| C4 | C | Cross-geo | FULL | 2 | 2 | Medium |
| C5 | C | US-LC | FULL | 4 | 4 | High |
| C6 | C | US-LC | varies | 4 × 10 | 40 | High |
| D1 | D | US-LC | FULL | 1 | analysis only | Critical |
| D2 | D | US-LC | FULL | 1 | analysis only | Critical |
| D3 | D | MIXED-200 | FULL | 1 | analysis only | Critical |
| D4 | D | US-LC | FULL | 8 events | analysis only | High |
| D5 | D | US-LC | FULL | 1 | analysis only | High |
| D6 | D | MIXED-200 | FULL | 1 | analysis only | High |
| **Total** | | | | | **~680 pipeline runs** | |

At ~5 minutes per pipeline run (N=200, T=80Q), total compute ≈ 57 hours.
At ~15 minutes (N=500 or full signals), ≈ 170 hours. Parallelisable across runs.

---

## Results Storage and Reporting

### Results Database Schema

```
results/
├── experiments.parquet        # experiment metadata (ID, config hash, timestamp)
├── metrics/
│   ├── level1_{exp_id}.parquet  # statistical performance per window
│   ├── level2_{exp_id}.parquet  # component attribution
│   ├── level3_{exp_id}.parquet  # robustness metrics
│   ├── level4_{exp_id}.parquet  # economic validity
│   └── level5_{exp_id}.parquet  # computational efficiency
├── configs/
│   └── {exp_id}.yaml            # exact config used (reproducibility)
└── reports/
    ├── complexity_map.md         # domain × data_regime → recommended depth
    ├── failure_map.md            # domain × data_regime × component → status
    ├── signal_attribution.md     # signal family value ranking
    ├── transfer_scorecard.md     # cross-domain transfer results
    └── economic_evidence.md      # D-series results synthesis
```

### Automated Report Outputs

After all experiments complete, the report generator produces:

1. **Complexity Map**: a table showing recommended pipeline depth for each
   {sector, cap_tier, data_regime} combination
2. **Failure Map**: component status (✅/⚠️/❌/💥) across all conditions
3. **Signal Ranking**: ordered list of signal families by marginal value
4. **Transfer Scorecard**: which components transfer where
5. **Economic Evidence Summary**: which gaps predict what, with what lead time
6. **Hypothesis Scorecard**: H1–H6 with verdict (supported/refuted/inconclusive)
7. **Operating Envelope**: the boundary conditions within which the framework works
