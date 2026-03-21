# SMIM Experimental Objectives and Measurement Framework

## Purpose

With the core pipeline operational (WP0–WP4), we need a systematic experiment
programme that answers six fundamental questions before committing to WP5's
full validation. This document formalises the objectives, defines what exactly
we measure, and specifies how to interpret results — so the subsequent experiment
plan can be designed against concrete targets.

---

## The Six Questions

### Q1: Component Value Decomposition
**"Which components of the framework earn their complexity?"**

The SMIM pipeline stacks five layers of complexity on top of a simple baseline.
Each layer must justify itself:

```
Layer 0: Naïve baseline (historical mean, random walk)
Layer 1: + Graph structure (directed multilayer operator)
Layer 2: + Spectral compression (modal representation)
Layer 3: + Regime switching (Kim filter vs single-regime)
Layer 4: + Emergence diagnostics (PID, TE, criticality, TDA)
Layer 5: + Phase-transition dynamics (GL landscape, order parameters)
```

**What we measure**: incremental OOS R² at each layer. If Layer N adds < 0.5pp
over Layer N-1, it doesn't justify its complexity for that experimental condition.

**Why this matters**: the framework is modular by design. Some conditions may only
need Layers 0-2; others may need the full stack. Knowing this avoids over-engineering
and under-engineering simultaneously.

### Q2: Data Regime Sensitivity
**"How does performance vary across data quality regimes?"**

The actor universe has radically different data characteristics depending on what
you look at:

| Regime | Coverage | Frequency | Timeliness | Example |
|--------|----------|-----------|------------|---------|
| **Gold** | >95% signals available, quarterly+ | Daily–quarterly | <1 day pub lag | S&P 500 large caps, Fed, major central banks |
| **Silver** | 70–95% available | Quarterly | 1–30 day lag | Mid-cap equities, regional banks, OECD data |
| **Bronze** | 40–70% available | Quarterly–annual | 30–90 day lag | Small caps, emerging market actors, municipal data |
| **Sparse** | <40% available | Annual or irregular | >90 day lag | Think tanks, SMEs, some international orgs |

**What we measure**: framework performance (all metrics) stratified by data regime.
Specifically:
- Does the graph add value when you only have "silver" or "bronze" data?
- Does regime switching help when temporal resolution is only quarterly?
- Do emergence diagnostics require "gold" data to be meaningful?
- What is the minimum data regime where the framework outperforms naïve baselines?

**Why this matters**: determines the practical scope of the framework. If it only
works on "gold" data, the claim of covering "heterogeneous actors including SMEs and
municipalities" is hollow.

### Q3: Cross-Domain Transferability
**"Does the framework generalise, or is it curve-fitted to one domain?"**

The MVP is energy sector, US+UK. But the proposal claims applicability to
heterogeneous economic actors across domains. We need to test this without
waiting for WP6.

**Dimensions of transfer**:
- **Cross-sector**: energy → technology, financials, healthcare, industrials
- **Cross-geography**: US+UK → EU, Japan, emerging markets
- **Cross-cap**: large cap → mid cap → small cap (same sector)
- **Cross-period**: pre-2015 → post-2015 (structural break in data availability)

**What we measure**: for each transfer, which pipeline components retain their
parameters and which require re-estimation:
- Graph structure: do edge channels transfer? Do channel weights transfer?
- Modal structure: does the number of retained modes K* transfer?
- Regime dynamics: does the transition matrix P transfer? Does M* transfer?
- Emergence: do synergy patterns transfer across domains?

**Transfer score**: fraction of components that perform within 80% of
in-domain accuracy when applied out-of-domain without re-training.

### Q4: Signal Attribution
**"Which data sources and signal families drive performance?"**

The framework ingests six signal families (macro, policy, market, balance sheet,
narrative, network position) from multiple APIs. Not all contribute equally.

**What we measure**:
- **Marginal value**: for each signal family, the OOS R² gain when it's added
  to a model trained on all other families (leave-one-in)
- **Dispensability**: for each signal family, the OOS R² loss when it's removed
  from the full model (leave-one-out)
- **Signal family × component interaction**: does the narrative signal family
  only matter when emergence diagnostics are active? Does the market signal
  family only matter with regime switching?

**Why this matters**: determines which data sources are worth the engineering
cost to maintain. If narrative signals add nothing, GDELT integration is wasted
effort. If market signals dominate, the framework may be capturing price
information that simpler models already exploit.

### Q5: Robustness and Failure Modes
**"Where does the framework fail, and how does it fail?"**

Beyond the 7 falsification tests (WP5), we need to understand failure modes
across conditions:

**Failure categories**:
- **Silent failure**: framework produces gaps that look plausible but have
  no predictive value (most dangerous)
- **Loud failure**: numerical instability, degenerate regimes, ill-conditioned
  eigendecomposition (detectable and fixable)
- **Graceful degradation**: performance decreases smoothly as conditions worsen
  (acceptable)
- **Cliff failure**: performance is fine until a threshold, then collapses
  (need to know where the cliff is)

**What we measure**:
- Performance as a function of actor count N (10, 50, 100, 200, 500)
- Performance as a function of time series length T (20Q, 40Q, 60Q, 80Q)
- Performance as a function of noise level (inject increasing noise into signals)
- Performance as a function of graph misspecification (progressively degrade edges)
- Performance under structural breaks (train on one regime, test on another)

**Boundary mapping**: for each component, find the condition threshold below
which it stops adding value. This defines the framework's "operating envelope."

### Q6: Economic Meaningfulness
**"Do the gaps mean anything, or are they statistical artefacts?"**

This is the hardest question. Statistical significance is necessary but not
sufficient. The gaps must correspond to something economically real.

**What we measure**:
- **Gap persistence**: do large gaps at time t predict gaps at t+h? (half-life)
- **Gap correction**: do gaps predict subsequent repricing, capex revision, or
  credit spread changes?
- **Gap diffusion**: do gaps in Layer 2 actors predict downstream gaps in
  Layer 3 actors connected via the graph?
- **Gap and known events**: do gaps spike around known misallocation events
  (energy over-investment pre-2015, COVID capital reallocation, etc.)?
- **Benchmark sensitivity**: how much do gaps change across benchmark classes?
  If Δ^{pred} ≈ Δ^{str} ≈ Δ^{em}, the benchmark distinction is academic.
- **Expert alignment**: for a subset of actors with known over/under-investment
  (e.g., from analyst consensus), do gaps align?

---

## The Measurement Stack

Every experiment produces a results vector measured at five levels:

### Level 1: Statistical Performance (computed automatically)

| Metric | Formula | Target | Granularity |
|--------|---------|--------|-------------|
| OOS R² | 1 - SSE/SST on holdout | > 0 (minimum); > 0.05 (good) | Per actor, per period, per benchmark class |
| RMSE | √(mean squared error) | < σ(y) | Per actor type |
| Spearman ρ | Rank correlation of gaps vs realised deviations | > 0.1 | Cross-sectional per period |
| NDCG@10 | Ranking accuracy for top/bottom decile actors | > 0.5 | Cross-sectional per period |
| Hit rate | Fraction of actors where gap sign matches direction of subsequent move | > 0.55 | Binary, per actor |
| DM test p-value | Diebold-Mariano vs each baseline | < 0.10 | Per baseline pair |

### Level 2: Component Attribution (computed per layer)

| Metric | What it captures |
|--------|-----------------|
| ΔR²(graph) | R² with graph − R² without graph (DFM baseline) |
| ΔR²(spectral) | R² with chosen spectral method − R² with PCA |
| ΔR²(regime) | R² with switching − R² with single regime |
| ΔR²(emergence) | R² with emergence corrections − R² without |
| Falsification survival | How many of 7 tests passed per component layer |

### Level 3: Robustness (computed across conditions)

| Metric | What it captures |
|--------|-----------------|
| Cross-domain transfer score | % of performance retained out-of-domain |
| Stability σ | Std dev of OOS R² across rolling windows |
| Cliff threshold | Minimum N, T, or coverage where R² > 0 |
| Regime-break resilience | R² when trained on regime A, tested on regime B |

### Level 4: Economic Validity (computed on realised outcomes)

| Metric | What it captures |
|--------|-----------------|
| Gap half-life | How quickly large gaps mean-revert (quarters) |
| Correction prediction | Does |Δ| > 1.5σ predict repricing in 2–8 quarters? |
| Diffusion prediction | Do upstream gaps predict downstream gaps? |
| Event alignment | Gap spike coincidence with known misallocation events |
| Benchmark divergence | Mean |Δ^{pred} − Δ^{str}| — do benchmarks differ meaningfully? |

### Level 5: Computational Efficiency (computed for scaling decisions)

| Metric | What it captures |
|--------|-----------------|
| Pipeline runtime | End-to-end seconds for one estimation window |
| Per-component runtime | Seconds per: edge estimation, eigendecomposition, Kim filter, PID |
| Memory peak | GB at peak during pipeline run |
| Scaling exponent | How runtime grows with N (fit N^α) |

---

## Interpretation Framework

### The Value Ladder

For each experimental condition, components are ranked by their incremental
contribution. The goal is to find the **minimum viable complexity** for each
operating regime:

```
If ΔR²(graph) < 0.5pp:          → use DFM baseline, skip graph
If ΔR²(spectral) < 0.5pp:       → use PCA, skip directed spectral methods
If ΔR²(regime) < 0.5pp:         → use single-regime model
If ΔR²(emergence) < 0.5pp:      → skip emergence corrections
If ΔR²(phase_transition) < 0.5pp: → skip GL landscape
```

This produces a **complexity map**: domain × data_regime → recommended pipeline depth.

### The Failure Map

For each experimental condition, record whether each component:
- ✅ Adds significant value (ΔR² > 0.5pp, survives falsification)
- ⚠️ Marginal value (0 < ΔR² < 0.5pp, mixed falsification)
- ❌ No value or negative value (ΔR² ≤ 0, fails falsification)
- 💥 Numerical failure (ill-conditioned, divergent, overflow)

This produces a **failure map**: domain × data_regime × component → status.

### Statistical Discipline

- **Multiple comparison correction**: with many conditions × metrics, apply
  Benjamini-Hochberg FDR correction at 5% level
- **Effect size reporting**: always report Cohen's d alongside p-values
- **Confidence intervals**: bootstrap 95% CIs on all key metrics
- **Pre-registration**: the experiment plan specifies hypotheses and analysis
  methods BEFORE running experiments. Post-hoc exploration is labelled as such.

---

## What Success Looks Like

### Minimum success (publishable negative result)
The framework works for zero conditions, but the failure analysis is thorough
enough to explain why and inform future work.

### Baseline success (publishable positive result)
The framework outperforms naïve baselines on the MVP domain (energy, US+UK,
gold/silver data regime) with ≥4/7 falsification tests passed, and at least
one component beyond DFM adds demonstrable value.

### Strong success
Multiple components add value; the framework transfers to ≥1 additional domain
with ≥80% performance retention; emergence diagnostics provide genuine early-warning
signal; gaps predict economically meaningful corrections.

### Exceptional success
The complexity map reveals that different domains need different pipeline depths,
providing a practical "recipe" for practitioners. The failure map is well-characterised.
The framework is useful even in degraded data regimes.

---

## Dependencies and Timing

This measurement framework should be **implemented before running systematic
experiments**, so that every experiment automatically produces the full measurement
stack. The implementation tasks are:

1. **Metrics module** (extends M5.1-T2): implement all Level 1–5 metrics
2. **Component ablation harness**: automated layer-by-layer ablation
3. **Cross-domain runner**: parameterised pipeline that accepts different configs
4. **Results database**: structured storage for all experiment results
5. **Report generator**: automated comparison tables and failure maps

These should be Claude Code tasks in a new milestone block (M-EXP) that sits
between WP4 and WP5, or as an extension of M5.1.

---

## Pre-Registration of Hypotheses

Before running experiments, we commit to the following directional hypotheses
(to be tested with the measurement stack above):

### H1: Graph Value
**H1a**: Directed graph structure adds ΔR² > 0.5pp over DFM for energy sector,
gold data regime, US+UK.
**H1b**: Graph value decreases monotonically with data regime quality (gold > silver > bronze).
**H1c**: Narrative edges (C4) add more marginal value than supply-chain edges (C5)
during policy-activism regimes.

### H2: Spectral Value
**H2a**: Directed spectral methods outperform symmetric PCA by ΔR² > 0.3pp.
**H2b**: Polar or Hermitian dilation outperform Schur decomposition on ill-conditioned operators.
**H2c**: The optimal K* is between 5 and 15 for N~200 actors.

### H3: Regime Value
**H3a**: Regime switching (M*≥2) adds ΔR² > 0.5pp over single-regime for energy
during periods containing at least one known crisis.
**H3b**: Regime switching adds NO value during stable expansion periods (2013-2019).
**H3c**: The criticality index C_t provides ≥1 quarter lead time over VIX for
at least one regime transition.

### H4: Emergence Value
**H4a**: PID synergy between macro and narrative modes is significantly positive
during crisis periods.
**H4b**: Transfer entropy anomalies (direct Layer 1→3 bypassing Layer 2) precede
misallocation clusters.
**H4c**: Topological complexity T_t rises ≥6 months before structural breaks.

### H5: Cross-Domain
**H5a**: Energy → technology transfer retains ≥60% of in-domain R².
**H5b**: Large cap → mid cap transfer retains ≥70% of R².
**H5c**: Large cap → small cap transfer retains <50% (data regime degrades too much).

### H6: Economic Meaning
**H6a**: Gap half-life is between 4 and 12 quarters (not too fast, not too slow).
**H6b**: Top-decile |Δ| at t predicts correction at t+4Q with p < 0.10.
**H6c**: Δ^{pred} and Δ^{str} differ by >0.3σ on average (benchmarks are distinct).

---

## Reporting Standard

Every experiment result must include:
1. **Condition label**: {domain, geography, cap_tier, data_regime, period, config_hash}
2. **Full measurement stack**: all Level 1–5 metrics
3. **Component attribution**: layer-by-layer ΔR²
4. **Falsification status**: which of 7 tests passed
5. **Hypothesis status**: which pre-registered hypotheses are supported/refuted
6. **Benchmark labels**: every gap tagged with its BenchmarkClass (mandatory rule)

Results are stored in a structured database (Parquet or SQLite) for cross-experiment
comparison. The report generator produces the complexity map and failure map automatically.
