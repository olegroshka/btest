# Paper 2: Nonlinear Structure and Emergence in Cross-Sectional Investment Dynamics

> Created: 2026-04-02 | Revised: 2026-04-03 (v2 — reframed as standalone Paper 2)
> Paper 1: "Regularised Spectral State-Space Models..." — submitted to SSRN, arXiv pending
> Paper 1 baseline: rolling SMIM R² = 0.702 (nested CV), 0.765 (holdout) at quarterly
> This plan: test whether nonlinear/emergent structure exists at higher frequencies

---

## 1. Motivation: Why a Second Paper

Paper 1 established three facts about cross-sectional investment dynamics:

1. **8 linear rotating modes explain 70% of variance.** The DMD spectral basis
   rotates at 26°/quarter while maintaining fixed dimensionality.

2. **Dual regularisation is essential.** Spherical R (1 param from N²) and
   near-identity F (1 param from K²) eliminate overparameterisation.

3. **Emergence is absent at quarterly frequency.** PID synergy, TDA complexity,
   EDMD, MI operators, and directed TE operators all fail because T=20 quarterly
   observations cannot support nonlinear estimation (EDMD T/P=0.45; PID needs T>30).

Fact 3 leaves a critical question open: **Is emergence absent because the system
is fundamentally linear, or because quarterly resolution is too coarse to detect it?**

This question cannot be answered by Paper 1's data. It requires a dedicated
multi-frequency investigation with sufficient statistical power for nonlinear
estimation. That is Paper 2.

### What Paper 2 Contributes Regardless of Outcome

| Outcome | Contribution | Venue fit |
|---------|-------------|-----------|
| Emergence detected at daily | First evidence of emergent structure in investment dynamics | Complexity economics, econophysics |
| Nonlinear > linear at daily | Koopman/MI methods outperform linear DMD on economic panels | Computational finance, JBES |
| All null at daily | Definitive: investment dynamics are fundamentally linear with rotating modes | Strong negative result — q-fin, JASA |

A null result at daily T=1260 with genuine temporal variation is **not** the same
as a null at quarterly T=20. It closes the question rather than leaving it open.

---

## 2. The Effective Sample Size Problem

### Why Step-Interpolation Fails for Emergence Testing

The original plan (v1) proposed step-interpolating quarterly CapEx rank to daily
frequency: each stock inherits its last quarterly rank for ~63 consecutive days.
This creates an ILLUSION of T=1260 daily observations.

**The problem**: step-interpolated data has autocorrelation = 1.0 within quarters.
The effective degrees of freedom for nonlinear estimation remain ~20 (the number
of quarterly transitions), not 1260.

| Method | Nominal T | Effective T (step-interpolated) | Required T |
|--------|-----------|--------------------------------|-----------|
| EDMD degree 2 (P=44) | 1260 | ~20 (same as quarterly) | T/P > 5 → T > 220 |
| KSG MI (k=5) | 1260 | ~20 | T > 50 for reliable MI |
| PID synergy (28 pairs) | 1260 | ~20 | T > 100 for stable MI |
| TDA sliding window (w=20) | 62 windows/yr | ~4 windows/yr | >20 windows for persistence |

Step-interpolation is valid for the LINEAR Kalman filter (it learns that observation
noise is near-zero within quarters), but NOT for any method that requires independent
observations for nonlinear estimation.

**Conclusion**: Paper 2 requires signals with **genuine daily temporal variation**.

### What Counts as Genuine Daily Variation

A daily signal has genuine temporal variation if:
- Its 1-day autocorrelation is substantially below 1.0 (< 0.99)
- The cross-sectional rank changes materially on most trading days
- The information content per observation is non-trivial (not copied from a lower-frequency source)

Examples:
- ✅ 60-day cross-sectional momentum rank: changes daily as returns enter/leave the window
- ✅ FRED VIX: changes daily, reflects new market information
- ✅ GDELT sector tone: changes daily from new articles
- ❌ Step-interpolated quarterly CapEx rank: constant for ~63 days

---

## 3. Data Inventory

### Already Available

| Dataset | Actors | Frequency | Coverage | Location |
|---------|--------|-----------|----------|----------|
| US-LC OHLCV | 199 tickers | Daily | 2005-01-03 to 2025-12-30 | `equities/smim/US-LC/ohlcv.parquet` |
| EDGAR balance sheet | 611 CapEx tickers | Quarterly | 2005-2026 | `data/smim/processed/edgar_balance_sheet.parquet` |
| FRED daily | 8 daily series | Daily | 2000-2026 | `data/smim/processed/fred_signals.parquet` |
| GDELT narrative | 9 daily actors | Daily | 2015-02-19 to 2025-12-31 | `data/smim/processed/gdelt_narrative_daily.parquet` |

**Key fact**: 155 US-LC tickers have both daily OHLCV and EDGAR CapEx data.
No new data download is needed for the stock-level panel.

### To Download (small, quick)

| Dataset | Actors | Source | Script | Est. Time |
|---------|--------|--------|--------|-----------|
| GICS Sector ETFs | 11 ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC) | Yahoo Finance | `yf://` | 5 min |

Note: XLRE (Real Estate) inception 2015-10-08; XLC (Communication) inception 2018-06-18.
Use only the 9 ETFs that existed before 2010 for full-period coverage, or handle
entries correctly with NaN padding.

---

## 4. Daily Intensity Construction

### 4.1 Primary Signal: Cross-Sectional Momentum Rank

For each US-LC stock i on trading day t:

```
r_i,t = cumulative return of stock i from day t-L to day t-1   (point-in-time: excludes day t)
y_i,t = cross-sectional percentile rank of r_i,t across all N stocks on day t
```

This is the daily investment intensity proxy. It measures the market's revealed
preference about capital allocation direction: high momentum rank = capital flowing
toward this firm.

**Lookback L**: test L ∈ {20, 60, 120} days in inner CV. Default: L=60 (standard
institutional momentum lookback).

**Why momentum rank works for SMIM**:
- Changes daily — genuine T=1260 observations per 5yr window
- Cross-sectionally ranked in [0,1] — same scale as quarterly CapEx rank
- Has well-documented factor structure (Fama-French, sector rotation, crowding)
- Cross-sectional dynamics are the natural target for spectral decomposition
- NOT the same as "return-based intensity" that failed in Paper 1:
  - Paper 1 used 12-month return ranked quarterly (4 obs/year) → no dynamics
  - Here: 60-day rolling momentum ranked daily (252 obs/year) → rich dynamics

**Why this is NOT the same failure mode as Paper 1's return intensity (R²=-0.15)**:
Paper 1's return-based intensity used 12-month trailing returns, ranked cross-sectionally
at quarterly frequency. This produced only 4 cross-sectional snapshots per year. The
R²=-0.15 result was driven by insufficient temporal resolution for spectral dynamics,
not by returns being uninformative. At daily frequency with 252 snapshots per year,
cross-sectional momentum structure is well-established in the empirical finance
literature and provides exactly the rich temporal dynamics that quarterly return
intensity lacked.

### 4.2 Validation: Momentum Rank as Investment Proxy

Before running any SMIM experiments, validate signal quality and investment relevance:

**Signal quality gates** (must pass before any SMIM runs):

| Gate | Test | Threshold | If fails |
|------|------|----------|----------|
| V-1 | Per-actor 1-day autocorrelation of momentum rank | Median ρ < 0.995 | Signal too slow; try L=20 |
| V-2 | Daily cross-sectional rank turnover (fraction of actors changing rank by ≥0.01) | > 50% of days | Signal too static; try L=20 |
| V-3 | Effective degrees of freedom: T_eff = T × (1 - ρ_median) per 5yr window | T_eff > 100 | Insufficient for EDMD/MI |

**Autocorrelation note**: With L=60, each day shares 59/60=98.3% of returns with
the previous day. Per-actor ρ will be high (~0.97-0.99). But what matters for SMIM
is the CROSS-SECTIONAL rank change rate: even if individual actors barely move,
stocks near rank boundaries reshuffle daily. V-2 measures this directly.

**If V-1 or V-2 fail at L=60**: retest at L=20 (only 19/20=95% overlap).
**If still fails at L=20**: fall back to 5-day (weekly) aggregation of momentum rank.
This gives T≈252 per 5yr — still 12× more than quarterly.

**Investment relevance gates** (inform framing, not hard gates):

| Gate | Test | Threshold | Interpretation |
|------|------|----------|----------------|
| V-4 | Cross-sectional Spearman(momentum_rank, CapEx_rank) per quarter | > 0.10 | Momentum reflects investment position |
| V-5 | Momentum rank predicts next-Q CapEx rank change | t > 2.0 | Forward-looking investment content |

**If V-4 and V-5 fail**: momentum does not proxy investment. Paper 2 is reframed as
"emergence in cross-sectional equity dynamics" — still publishable, but different claim.
**If V-4 or V-5 pass**: the investment framing is supported, strengthening the Paper 1 connection.

### 4.3 Step-Interpolated CapEx: Diagnostic Baseline Only

We still compute step-interpolated quarterly CapEx rank at daily frequency — but ONLY
as a diagnostic comparison. It provides an upper bound on the linear Kalman filter's
performance (since within-quarter observation noise is nearly zero) and a lower bound
on what emergence methods can extract (since effective T ≈ 20).

This diagnostic helps interpret the results:
- If momentum SMIM > step-CapEx SMIM: daily dynamics contain genuine new information
- If momentum SMIM < step-CapEx SMIM: momentum is noisier than CapEx for linear prediction
- For nonlinear methods, ONLY the momentum panel is meaningful (genuine daily T)

### 4.4 Secondary Signals (Phase 2, after Phase 1 validation)

If Phase 1 (momentum-only) shows cross-sectional spectral structure:

| Signal | Per actor? | Construction | Role |
|--------|-----------|-------------|------|
| 60-day momentum rank | Per stock | Rolling return, xs-ranked | Primary (daily variation) |
| Sector ETF relative strength | Per sector | ETF return rank within sector universe | Sector-level allocation flow |
| GDELT sector tone | Per sector | Daily normalised tone | Narrative investment sentiment |
| FRED daily macro | Global | Cross-sectional rank across 8 series | Regime indicators |

Combination method: **equal-weight average**, then cross-sectional rank.
No weight optimisation — this avoids overfitting the proxy construction.

---

## 5. Two Evaluation Panels

### Panel A: Stock-Level (Primary — for linear SMIM + EDMD)

| Parameter | Value |
|-----------|-------|
| Actors | ~155 US-LC stocks with EDGAR CapEx overlap |
| Signal | 60-day cross-sectional momentum rank |
| Frequency | Daily (252 trading days/year) |
| Coverage | 2010-01-01 to 2025-12-31 |
| Training | 5yr rolling (T=1260 trading days) |
| Test | 1yr (252 trading days) |
| Windows | W2015 through W2024 (10 windows, matching Paper 1) |
| Inner CV | Last 2yr of train; selects K ∈ {3, 5, 8, 10} and L ∈ {20, 60, 120} |
| Holdout | W2023-W2024 (frozen, no tuning — same as Paper 1) |

**Why N=155 works**: T/N = 1260/155 ≈ 8.1. DMD is well-conditioned. For EDMD
with P=44 lifted features: T/P = 1260/44 = 28.6 — properly identified.

### Panel B: Sector/Macro (Secondary — for emergence diagnostics)

| Parameter | Value |
|-----------|-------|
| Actors | 11 sector ETFs + 8 FRED daily + 9 GDELT daily = 28 actors |
| Signal | Within-type normalisation (see below) |
| Frequency | Daily |
| Coverage | 2015-02-19 to 2025-12-31 (limited by GDELT) |
| Training | 5yr (T=1260) |
| Test | 1yr (252 days) |
| Windows | W2020 through W2024 (5 windows) |

**Normalisation**: Cross-sectional ranking of 28 heterogeneous actors (VIX vs
GDELT tone vs XLK return) is meaningless. Instead, normalise WITHIN actor type:
- 11 sector ETFs: cross-sectional rank within ETF universe per day → [0,1]
- 8 FRED daily: rolling 252-day z-score, then CDF → [0,1]
- 9 GDELT: rolling 252-day z-score, then CDF → [0,1]

All actors then live in [0,1] with comparable scales but type-appropriate normalisation.

**Why Panel B for emergence**: All 28 actors have genuine daily variation. No
step-interpolation, no momentum approximation. If emergence exists in the
cross-sectional dynamics of sector allocation, macro conditions, and narrative
sentiment, Panel B will detect it. N=28 gives at most 28 spectral modes; at
T=1260, DMD estimation is rock-solid (T/N ≈ 45).

**Inference limitation**: 5 windows is underpowered for window-level sign tests
(need 5/5 wins for p=0.031). Use actor-day-level Diebold-Mariano tests instead
(many thousands of observations per window), same approach as Paper 1.

---

## 6. Emergence Hypotheses (Precisely Defined)

Paper 2 tests four distinct emergence hypotheses. Each has a specific statistical
test and a clear falsification criterion.

### H_NL: Nonlinear Mode Coupling

**Claim**: The K spectral modes interact nonlinearly. Quadratic cross-terms
(α_j · α_k) contain predictive information beyond linear dynamics (F · α_t).

**Test**: Polynomial autoregression on the K-dimensional alpha trajectory.
Rather than full Koopman lifting (which has a consistency problem — predicted
lifted features must match the quadratic of predicted alpha), use polynomial AR:

```
α_{t+1} = F_lin · α_t + F_quad · vech(α_t ⊗ α_t) + ε_t
```

This directly tests whether quadratic interactions (α_j · α_k) predict future α.
Dimension of predictors: P = K + K(K+1)/2 = 8 + 36 = 44 (at K=8).
At daily T=1260: T/P = 28.6 — well-conditioned.

- Ridge regularisation on [F_lin, F_quad] jointly (λ selected by inner CV)
- Compare: R²(poly AR) vs R²(linear AR, i.e., F_lin only)
- Diagonal-only variant: P = K + K = 16 (only self-interactions α_j²)

**Metric**: ΔR² = R²(poly AR) - R²(linear AR), evaluated on same test windows.
**Falsification**: ΔR² ≤ 0 in majority (>50%) of windows → modes are linearly decoupled.
**Multiple-testing**: This is hypothesis 1 of 4; adjusted α = 0.0125 (Holm-Bonferroni).

### H_MI: Information-Theoretic Cross-Sectional Dependence

**Claim**: The cross-sectional dependence structure between actors contains
nonlinear components that Pearson correlation misses. An MI-based operator
produces a spectral basis that better captures these dependencies.

This is a DIFFERENT type of nonlinearity from H_NL:
- **H_NL** tests nonlinear MODE dynamics: α_{t+1} = f_nonlinear(α_t)
- **H_MI** tests nonlinear CROSS-SECTIONAL dependence: how actors relate to each other

Both can exist independently. Daily T=1260 makes MI estimation reliable for the
first time (Paper 1's quarterly MI failed because KSG with T=40 was noise-dominated).

**Test** (three tiers, from safest to most ambitious):

**(a) Symmetric MI operator** (primary — tests nonlinear dependence):
- KSG MI(y_i, y_j) for all N(N-1)/2 pairs, T=1260 daily observations, k=5
- Build symmetric operator W[i,j] = MI(y_i, y_j)
- Spectral decomposition of W → basis U_MI
- Run full SMIM pipeline (Kalman + dual reg) with U_MI
- Compare: R²(MI basis) vs R²(correlation-DMD basis)

For near-Gaussian data, MI ∝ -0.5·log(1 - ρ²), so MI is a monotonic but NONLINEAR
transformation of correlation. The MI operator's eigenvectors differ from correlation's
because the transformation is element-wise, not matrix-level. The difference grows
with non-Gaussianity — and daily momentum ranks are bounded in [0,1], which is
non-Gaussian by construction.

**(b) MI-weighted DMD** (novel — combines MI with temporal dynamics):
- Weight the DMD snapshot matrices by MI importance: rows with high total MI
  (strongly connected actors) get upweighted in the SVD
- This produces a spectral basis that emphasises actors in dense MI clusters
- Compare: R²(MI-weighted DMD) vs R²(standard DMD)

**(c) Directed CMI operator** (tests information flow asymmetry):
- CMI(y_{i,t+1}; y_{j,t} | y_{i,t}) for all N(N-1) pairs
- Directed operator → Schur decomposition (asymmetric → Schur ≠ PCA)
- Compare: R²(directed CMI Schur) vs R²(symmetric MI)
- If directed > symmetric: information flow IS asymmetric
- If directed ≈ symmetric: cross-sectional dependence is symmetric but nonlinear

**Metric**: ΔR² = R²(best MI variant) - R²(correlation-DMD basis).
**Falsification**: All three MI tiers ≤ correlation-DMD → dependence structure IS
well-captured by linear correlation at daily frequency.
**Multiple-testing**: This is hypothesis 2 of 4; adjusted α = 0.0125 (Holm-Bonferroni).
Within H_MI, the three tiers are hierarchical (not independent), so no further correction.

**Computational cost at daily T=1260**:
- Panel A (N=155): 11,935 MI calls × ~0.08s = ~16 min
- Panel A CMI: 23,870 CMI calls × ~0.12s = ~48 min
- Panel B (N=28): 378 MI calls × ~0.08s = ~30 sec
- Panel B CMI: 756 CMI calls × ~0.12s = ~90 sec
- All highly feasible at daily frequency

### H_SYNC: Emergent Mode Synchronisation

**Claim**: Spectral modes exhibit information-theoretic synergy — knowing multiple
modes jointly predicts better than knowing them independently. This synergy is
the hallmark of emergence in complex systems.

**Test**: PID (Partial Information Decomposition) on daily alpha modes:
- For each mode pair (j, k), compute synergy S_{jk} using Gaussian MI
- T=1260 points per pair → reliable MI estimation
- Bootstrap: resample T=1260 observations 1000 times → 95% CI for each S_{jk}

**Metric**: Fraction of mode pairs where S_{jk} > 2× bootstrap CI (significant synergy).
**Falsification**: 0/C(K,2) pairs show significant synergy → modes are informationally independent.
**Multiple-testing**: Bonferroni across C(K,2) pairs within this hypothesis.

### H_TOPO: Topological Structure Evolution

**Claim**: The topological complexity of the alpha trajectory (measured by
persistent homology) varies over time, and complexity changes correlate with
structural market events or forecast degradation.

**Test**: Sliding-window TDA on the daily alpha trajectory:
- Window = 60 days, stride = 20 days → ~60 windows per 5yr period
- Compute persistence diagrams (H0, H1) per window
- Track: persistence entropy, total persistence, Betti-1 count
- Correlate complexity time series with (a) market events, (b) ΔR² per window

**Metric**: Spearman correlation between TDA complexity and ΔR² across windows.
**Falsification**: |ρ| < 0.2 → topological complexity is uncorrelated with dynamics.

---

## 7. Experiment Programme

### Phase 0: Data Construction + Validation (Session 1, ~2 hours)

| ID | Task | Gate |
|----|------|------|
| P0-1 | Compute 60-day cross-sectional momentum rank for 155 US-LC stocks, daily, 2010-2025 | Panel shape (155, ~3780) |
| P0-2 | Download 11 sector ETFs (5 min via yf://) | 9+ ETFs with full coverage |
| P0-3 | Construct Panel B: ETF + FRED + GDELT, cross-sectionally ranked per day | Panel shape (28, ~2520) |
| P0-4 | Validation gates V-1 through V-5 for momentum rank | See §4.2 |
| P0-5 | Compute step-interpolated daily CapEx rank (diagnostic only) | Coverage matches P0-1 |

**STOP if V-1 through V-3 all fail at ALL lookbacks (L=60, L=20, weekly)**: signal
construction fundamentally broken. Reframe Paper 2 around Panel B only.
See §12 (Fallback Paths) for detailed decision tree.

### Phase 1: Linear SMIM at Daily Frequency (Session 1, ~2 hours)

| ID | Experiment | Hypothesis | Panel |
|----|-----------|-----------|-------|
| E3-1a | SMIM rolling (dual reg, DMD K from inner CV) on momentum rank | Framework works at daily | A |
| E3-1b | SMIM rolling on Panel B (sector/macro) | Framework works on genuine daily data | B |
| E3-1c | SMIM rolling on step-interpolated CapEx (diagnostic) | Upper bound on linear performance | A |
| E3-1d | Daily AR(1) per actor on momentum rank | Baseline for daily frequency | A |

**Metrics**: R², ΔR² vs AR(1), DM test.
**Gate**: E3-1a R² > daily AR(1) R² → spectral model adds value at daily.
**If gate fails**: momentum rank has no cross-sectional spectral structure. Report
as negative result and focus on Panel B.

### Phase 2: Nonlinear Extensions — H_NL (Session 2, ~3 hours)

| ID | Experiment | Predictors P | T/P | Panel |
|----|-----------|-------------|-----|-------|
| E3-2a | Poly AR degree 2 on daily alpha (Panel A) | 44 (K=8) | 28.6 | A |
| E3-2b | Poly AR degree 3 on daily alpha | 164 | 7.7 | A |
| E3-2c | Diagonal-only quadratic (self-interactions α_j²) | 16 | 78.8 | A |
| E3-2d | Poly AR degree 2 on Panel B alpha | 44 | 28.6 | B |
| E3-2e | Best poly AR + dual-regularised Kalman filter | varies | — | A |

**Ridge regularisation**: λ selected by inner CV (see §13).
This prevents overfitting in the polynomial space even at T/P=7.7.

**Gate (H_NL)**: ΔR²(EDMD) > 0 in ≥6/10 windows AND mean ΔR² significant at
adjusted α=0.0125 (bootstrap CI excludes 0).

### Phase 3: Information-Theoretic + Emergence — H_MI, H_SYNC, H_TOPO (Session 2-3, ~4 hours)

**Priority order** (by expected information value per compute hour):

| Priority | ID | Test | Hypothesis | Est. Time | Panel |
|----------|-----|------|-----------|-----------|-------|
| **1** | **E3-3a** | **Symmetric MI operator (KSG, k=5)** | **H_MI(a)** | **20 min** | **A, B** |
| 2 | E3-3b | MI-weighted DMD | H_MI(b) | 20 min | A |
| 3 | E3-3c | Multi-resolution DMD divergence (60d vs 252d vs 1260d) | emergence | 30 min | A |
| 4 | E3-3d | PID synergy on daily alpha modes (C(K,2)=28 pairs) | H_SYNC | 30 min | A, B |
| 5 | E3-3e | TDA sliding-window persistence on daily alpha | H_TOPO | 30 min | A |
| 6 | E3-3f | CMI directed operator | H_MI(c) | 45 min | B |
| 7 | E3-3g | Daily basis rotation tracking (monthly steps, 120 DMDs) | structural | 30 min | A |

**Note on E3-3a (multi-resolution)**: At daily frequency, the fast window (60 days)
has T=60 — equivalent to a full quarterly window. The slow window (252 days) has
T=252. Both are well-conditioned for DMD. Unlike the quarterly test (where the
8-quarter fast window gave noise-dominated angles at 47°), daily multi-resolution
can separate genuine scale-dependent structure from estimation noise.

**Note on E3-3f (daily rotation)**: We use MONTHLY steps (not daily) for computational
efficiency: 120 DMD recomputations across 10 years. This reveals whether the
26°/quarter rotation found in Paper 1 is smooth (gradual monthly increments) or
abrupt (concentrated in specific weeks).

### Phase 4: Combined + Synthesis (Session 3, ~2 hours)

| ID | Task |
|----|------|
| E3-4a | Stack all innovations with ΔR² > 0 |
| E3-4b | DM test of combined vs linear SMIM on all windows |
| E3-4c | DM test of combined vs daily AR(1) on all windows |
| E3-4d | Produce verdict table for H_NL, H_MI, H_SYNC, H_TOPO |
| E3-4e | Generate Paper 2 key figures |

---

## 8. Execution Dependencies

```
Phase 0: Data construction + validation     [no deps]
  |
  +--> Phase 1: Linear SMIM daily           [depends on Phase 0]
  |      |
  |      +--> Phase 2: EDMD (H_NL)          [depends on Phase 1 alpha trajectories]
  |      |
  |      +--> Phase 3: Emergence            [depends on Phase 1 alpha + daily panels]
  |             |
  |             |    E3-3a multi-res   ─┐
  |             |    E3-3b MI operator  │ can run in parallel
  |             |    E3-3c PID synergy  │
  |             |    E3-3d TDA          │
  |             |    E3-3e CMI directed ─┘
  |             |    E3-3f rotation    ─── (independent)
  |
  +--> Phase 4: Combined                    [depends on Phases 2-3]
```

Phases 2 and 3 can start as soon as Phase 1 produces alpha trajectories.
Within Phase 3, all experiments are independent and can run in parallel.

---

## 9. Computational Budget

| Experiment | Dominant cost | Est. time | GPU accelerable? |
|-----------|--------------|-----------|-----------------|
| P0 Data construction | Momentum rank computation (155 stocks × 3780 days) | 15 min | No (I/O bound) |
| E3-1 Linear SMIM (10 windows × 2 panels) | Kalman filter N=155, T=252 test | 30 min | Yes |
| E3-2 EDMD (5 configs × 10 windows) | Koopman SVD in lifted space | 30 min | Yes |
| E3-3a MI operator (symmetric) | 11,935 KSG calls (Panel A) + 378 (Panel B) | 20 min | No |
| E3-3b MI-weighted DMD | Re-weight DMD SVD by MI row sums | 15 min | Yes |
| E3-3c Multi-resolution DMD | 3 DMD per window × 10 windows | 15 min | Yes |
| E3-3d PID synergy | 28 pairs × bootstrap × 2 panels | 30 min | No |
| E3-3e TDA | 60 windows × Ripser | 30 min | No |
| E3-3f CMI directed | 23,870 CMI calls (Panel A) or 756 (Panel B) | 50 min | No |
| E3-3g Rotation tracking | 120 DMD recomputations | 20 min | Yes |
| E3-4 Combined | Re-run best configs | 30 min | Yes |
| **TOTAL** | | **~4-5 hours** | |

Buffer for debugging and unexpected issues: **+50% → 6-8 hours total**.

---

## 10. Success Criteria

### Per-Hypothesis Verdicts (Holm-Bonferroni corrected, FWER = 0.05)

| Hypothesis | SUPPORTED if | REFUTED if |
|-----------|-------------|-----------|
| H_NL | ΔR²(EDMD) > 0, CI excludes 0 at adjusted α=0.0125 | ΔR² ≤ 0 in >5/10 windows |
| H_MI | ΔR²(any MI variant) > 0, CI excludes 0 at adjusted α=0.0125 | All MI variants ≤ correlation-DMD |
| H_SYNC | >15% of mode pairs show synergy > 2× bootstrap CI | 0/28 pairs significant |
| H_TOPO | TDA complexity ρ with ΔR² > 0.3, p < adjusted α | |ρ| < 0.2 |

### Overall Paper 2 Verdict

| Level | Criterion | Paper 2 conclusion |
|-------|----------|-------------------|
| BRONZE | Daily linear SMIM R² > daily AR(1) | Spectral framework generalises to daily |
| SILVER | Any one hypothesis supported | Nonlinear/emergent structure exists at daily |
| GOLD | Two or more hypotheses supported | Multi-faceted emergence |
| PLATINUM | Combined model > linear by >3pp with DM p < 0.01 | Emergence has forecasting value |
| NULL | All four hypotheses refuted at daily T=1260 | **Definitive**: dynamics are linear with rotating modes |

**The NULL result is as valuable as PLATINUM** — it definitively resolves an open
question from Paper 1 and establishes the boundary of linearity in investment dynamics.

---

## 11. Power Analysis and Minimum Detectable Effects

With 10 windows (Panel A) or 5 windows (Panel B), what ΔR² can we detect?

From Paper 1: the rolling basis gain (+14.3pp) had std=5.0pp across 10 windows.
The bootstrap CI was [11.1, 17.2]pp. So the SE of the mean ΔR² with 10 windows
is approximately σ/√n = 5.0/√10 ≈ 1.6pp. A 95% CI width is ±3.1pp.

**Minimum detectable effect (MDE) at 80% power, α=0.0125 (Holm-adjusted)**:
- Panel A (10 windows): MDE ≈ 3.5pp (assuming σ ≈ 3pp for typical ΔR² across windows)
- Panel B (5 windows): MDE ≈ 5.5pp (wider CI from fewer windows)
- Actor-day DM test (N ≈ 40,000 obs): MDE ≈ 0.3pp (much more powerful)

**Implication**: Window-level tests can detect effects ≥ 3.5pp. Effects of 1-2pp
will be invisible at window level but potentially detectable at actor-day level.
We report BOTH: window-level for interpretability, actor-day for power.

If EDMD gives a genuine but small improvement (~1pp), it will show as "promising
but inconclusive at window level" and may reach significance at actor-day level.
This is honest and publishable either way.

---

## 12. Fallback Paths

### If BRONZE Fails (Daily SMIM R² < Daily AR(1))

Daily momentum rank may lack exploitable cross-sectional spectral structure.
Before declaring failure:

1. **Retest with shorter lookback (L=20)**: more daily variation, more noise
2. **Retest with weekly aggregation**: T=252 per 5yr, still 12× quarterly
3. **Try sector-level aggregation**: average momentum rank per GICS sector (N=11)
   — fewer actors but much smoother signal
4. **Focus on Panel B**: sector/macro panel has genuine daily structure by construction

If ALL of the above fail, the BRONZE failure IS the result: "The SMIM spectral
framework does not generalise to equity momentum dynamics at daily frequency.
Cross-sectional investment structure is detectable only through fundamental data
(CapEx) at quarterly frequency."

### If Phase 0 Validation Fails (High Autocorrelation)

If V-1 fails at L=60 (ρ > 0.995) and L=20 (ρ > 0.99):
- Use **5-day (weekly) aggregated momentum rank** as primary signal
- T=252 per 5yr window; T/P=252/44=5.7 for EDMD — marginal but feasible
- Rerun all experiments at weekly frequency
- Label results as "weekly" not "daily"

---

## 13. Inner Cross-Validation Protocol

Same nested structure as Paper 1, adapted for daily:

**Outer folds**: 10 annual test windows (Panel A) or 5 (Panel B).
**Inner folds**: last 2 years (504 trading days) of each outer-train period.

Inner CV selects:

| Hyperparameter | Search space | Used by |
|---------------|-------------|---------|
| K (mode count) | {3, 5, 8, 10, 15} | All experiments |
| L (momentum lookback) | {20, 60, 120} days | Panel A signal construction |
| λ_ridge (EDMD regularisation) | {0.001, 0.01, 0.1, 1.0, 10.0} | H_NL experiments |

**Fixed (not tuned)**: F=0.99I, Q₀=0.5I, τ=8, λ_Q=0.3 — same defaults as Paper 1.
These are stable across frequencies (Paper 1 showed zero-shot transfer works).

**Holdout**: W2023-W2024 (Panel A) or W2024 (Panel B) — frozen, no tuning exposure.

---

## 14. Microstructure and Robustness

Daily equity data contains bid-ask bounce, overnight gaps, and index rebalancing
effects. Stocks with very similar momentum cluster near rank boundaries and
randomly reshuffle from microstructure noise.

**Robustness checks** (run after main experiments, ~30 min):

| Check | Method | Purpose |
|-------|--------|---------|
| Weekly aggregation | 5-day average momentum rank | Filter microstructure noise |
| Sector-level | GICS-sector average momentum | Eliminate stock-level noise |
| Winsorised returns | Cap daily returns at ±5σ | Remove outlier contamination |
| Volume-weighted momentum | Weight returns by dollar volume | Reduce small-cap noise |

These are reported as sensitivity analysis, not main results.

---

## 15. Risk Matrix (updated)

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|-----------|
| **Momentum rank has no cross-sectional spectral structure** | Medium | Phase 1 fails | V-1 through V-4 gates catch early; fall back to Panel B |
| **Momentum proxy doesn't relate to investment** | Medium | Paper 2 reframed | V-1, V-2 gates; honest reframing as "equity dynamics" |
| **EDMD overfits even at T/P=28.6** | Low-Med | H_NL null | Ridge regularisation; diagonal-only fallback (T/P=78.8) |
| **MI operator ≈ correlation (near-Gaussian data)** | Medium | H_MI null | Expected for Gaussian; value is in tails |
| **PID estimation unreliable at T=1260** | Low | H_SYNC uncertain | Gaussian MI (analytical, no KSG needed); bootstrap CIs |
| **TDA trivial on smooth trajectories** | Medium | H_TOPO null | Valid negative — trajectory is topologically simple |
| **Daily DMD modes very unstable (A4 violated)** | Medium | K selection hard | Inner CV for K; report stability diagnostic |
| **All emergence null at daily** | Possible | NULL verdict | **Publishable**: definitive negative, closes Paper 1's open question |
| **Computational cost exceeds budget** | Low | Delays | Priority ordering; drop E3-3e (CMI) if needed |
| **Sector ETF download fails or is incomplete** | Very low | Panel B smaller | Use 9 pre-2010 ETFs; exclude XLRE and XLC |

---

## 16. Relation to Paper 1

| Dimension | Paper 1 | Paper 2 |
|-----------|---------|---------|
| Question | Can you forecast investment intensity? | Does emergence exist in the dynamics? |
| Data frequency | Quarterly | Daily (+ monthly intermediate) |
| Signal | CapEx/Assets rank | Momentum rank (daily proxy) |
| Key methods | DMD + Kalman + dual reg | MI operator + Poly AR + PID + TDA |
| Main contribution | Dual regularisation + rolling basis | Multi-frequency emergence test |
| Baseline model | Per-actor AR(1) | Paper 1's linear SMIM at daily |
| If positive | Forecasting result | First evidence of emergence |
| If negative | Emergence "future work" | Definitively linear |

Paper 2 CITES Paper 1 as the baseline framework and extends it. Paper 2's
linear SMIM experiments use the exact same pipeline (dual reg, rolling basis)
— only the data frequency and intensity signal change.

**Critical separation**: Paper 1's contribution stands independently. Paper 2's
results (positive or negative) do not change Paper 1's claims.

---

## 17. Monthly Intermediate Frequency (Quick Validation)

Before committing to daily experiments, run a 1-hour monthly validation:

| ID | Test | Monthly T | Interpretation |
|----|------|----------|---------------|
| M-1 | SMIM on monthly momentum rank (T=60/5yr) | 60 | Does monthly add over quarterly? |
| M-2 | EDMD degree 2 on monthly alpha (P=44, T/P=1.36) | 60 | Tight but first genuine test of H_NL |
| M-3 | Monthly AR(1) baseline | 60 | Baseline at monthly frequency |

**If M-1 R² > monthly AR(1)**: encouraging for daily.
**If M-2 shows any positive ΔR²**: strong signal that daily will work (T/P improves 21×).
**If both null**: daily may also be null, but proceed (daily has much more statistical power).

Monthly is NOT a gate — it's an early signal to calibrate expectations.

---

## 18. What We Do NOT Attempt

- Do NOT use step-interpolated CapEx as primary signal for emergence testing (effective T ≈ 20)
- Do NOT compare daily R² numerically with quarterly R²=0.691 (different scales, different targets)
- Do NOT optimise composite signal weights (use equal-weight to avoid overfitting)
- Do NOT skip Panel A validation gates (V-1 through V-4)
- Do NOT test sub-daily (intraday) frequency — not meaningful for investment dynamics
- Do NOT use return-based intensity at quarterly frequency (proven dead end)
- Do NOT add Paper 2 results to Paper 1 — keep publications separate
- Do NOT claim daily results "replace" the quarterly contribution
- Do NOT run TE on Panel A (22K pairs × T=1260 is 4+ hours; use Panel B instead)
- Do NOT attempt more than 4 emergence hypotheses — multiple testing already demanding

---

## 19. Session Plan

### Session 1 (~4 hours): Data + Linear Baseline

```
Hour 1: Phase 0 (data construction + validation)
  P0-1: Compute momentum rank panels (L=60, also L=20 for fallback)
  P0-2: Download sector ETFs
  P0-3: Construct Panel B (within-type normalisation per §5)
  P0-4: Validation gates V-1 to V-5 (see §4.2)
  P0-5: Step-interpolated CapEx diagnostic
  → DECISION: proceed with Panel A, adjust L, or fall back (see §12)

Hour 2: Monthly intermediate (Phase M, see §17)
  M-1: Monthly SMIM on momentum rank
  M-2: Monthly poly AR degree 2
  M-3: Monthly AR(1) baseline
  → SIGNAL: calibrate daily expectations

Hours 3-4: Phase 1 (daily linear SMIM)
  E3-1a: SMIM rolling on momentum rank (10 windows)
  E3-1b: SMIM rolling on Panel B (5 windows)
  E3-1c: Step-interpolated diagnostic
  E3-1d: Daily AR(1) baseline
  → GATE: E3-1a R² > daily AR(1)?
  → If fails: execute fallback path (§12) before proceeding
```

### Session 2 (~4 hours): Nonlinear + Emergence

```
Hours 1-2: Phase 2 (poly AR — H_NL)
  E3-2a through E3-2e: polynomial AR variants on both panels
  Inner CV for ridge λ on each window
  → VERDICT: H_NL supported or refuted

Hours 3-4: Phase 3 (information-theoretic + emergence)
  E3-3a: Symmetric MI operator on BOTH panels (H_MI — TOP PRIORITY)
  E3-3b: MI-weighted DMD (H_MI)
  E3-3c: Multi-resolution DMD divergence
  E3-3d: PID synergy on daily alpha modes (H_SYNC)
  E3-3e: TDA sliding-window (H_TOPO)
  E3-3g: Daily rotation tracking (structural)
  [E3-3f: CMI directed — if time permits]
  → VERDICT: H_MI, H_SYNC, H_TOPO supported or refuted
```

### Session 3 (~2 hours): Combined + Robustness + Paper Draft

```
Hour 1: Phase 4 (combined + robustness)
  E3-4a: Stack winners
  E3-4b-c: DM tests (window-level AND actor-day-level per §11)
  E3-4d: Verdict table (Holm-Bonferroni corrected)
  Robustness: weekly aggregation, sector-level, winsorised (§14)

Hour 2: Paper 2 scaffolding
  E3-4e: Key figures
  Outline Paper 2 sections
  Update STATUS.md and EXPERIMENT_RESULTS.md
```

---

## 20. Session 1 Results (2026-04-03)

### Phase 0: Data Construction + Validation

Panels built successfully:
- Panel A: 4023 days x 140 tickers (L=20), 139 (L=60, L=120), 2010-2025
- Panel B: 2726 days x 28 actors (11 sectors + 8 FRED + 9 GDELT), 2015-2025
- Weekly Panel A: 835 weeks x 140 tickers

Validation gate results:

| Gate | L=20 | L=60 | L=120 | Verdict |
|------|------|------|-------|---------|
| V-1 (rho < 0.995) | 0.934 PASS | 0.976 PASS | 0.987 PASS | All pass |
| V-2 (turnover > 50%) | 83% PASS | 74% PASS | 66% PASS | All pass |
| V-3 (T_eff > 100) | 84 FAIL | 30 FAIL | 16 FAIL | **All fail** |
| V-4 (Spearman > 0.10) | -0.028 FAIL | -0.047 FAIL | -0.043 FAIL | **All fail** |
| V-5 (t > 2.0) | t=0.27 FAIL | t=1.28 FAIL | t=1.11 FAIL | **All fail** |

**Key findings:**
- Momentum rank does NOT proxy investment (V-4, V-5) → Paper 2 reframed as "equity dynamics"
- T_eff below threshold for all lookbacks → emergence methods need careful interpretation
- L=20 is best candidate (lowest autocorrelation, highest turnover)

### Phase 1: Linear SMIM at Daily Frequency

| Experiment | Mean R² | AR(1) R² | Delta | Wins |
|-----------|---------|----------|-------|------|
| E3-1a: Panel A raw (L=20, K=8) | 0.162 | 0.870 | -0.708 | 0/10 |
| E3-1b: Panel B (K=8) | 0.578 | 0.780 | -0.203 | 0/5 |
| E3-1c: Step-interpolated CapEx | 0.875 | 0.994 | -0.119 | 0/10 |
| E3-1e: Pre-whitened (AR1 + spectral) | 0.852 | 0.870 | -0.018 | 0/10 |
| E3-1f: Pre-whitened K=1 | 0.865 | 0.870 | -0.006 | 0/10 |

**BRONZE gate: FAIL (0/10 windows).** Monotonic degradation with K (even K=1 hurts).

### Emergence Tests on Panel B

**H_NL (polynomial AR):** Mean MSE ratio poly/lin = 1.015. Poly wins 2/5 windows.
**NOT SUPPORTED.** No nonlinear mode coupling.

**H_MI (MI operator):** Structurally striking, predictively useless.

| Basis | Static R² | Rolling R² | vs AR(1) |
|-------|-----------|-----------|----------|
| Correlation (PCA) | 0.217 | — | -0.562 |
| DMD | 0.169 | **0.507** | -0.273 |
| **MI** | **0.035** | **0.383** | **-0.397** |
| MI-weighted DMD | — | 0.499 | -0.281 |
| AR(1) | — | — | **0.780** |

MI and correlation bases are **68 degrees apart** (near-orthogonal). The nonlinear
dependence MI captures is REAL (r(|corr|, MI) = 0.17) but orthogonal to the
predictively useful structure. MI basis is strictly worse in every window (0/5).

### Overall Verdict

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| BRONZE (SMIM > AR1 at daily) | **FAIL** | 0/10 wins on Panel A, 0/5 on Panel B |
| H_NL (nonlinear modes) | **NOT SUPPORTED** | MSE ratio 1.015, 2/5 |
| H_MI (MI basis > correlation) | **NOT SUPPORTED** | Delta -0.124, 0/5 |
| H_SYNC (PID synergy) | NOT TESTED | BRONZE failure makes this moot |
| H_TOPO (TDA evolution) | NOT TESTED | BRONZE failure makes this moot |

**Paper 2 conclusion: The SMIM spectral framework is frequency- and signal-specific.**
Its value comes from the cross-sectional structure of quarterly CapEx intensity,
not from any general property of economic panel dynamics. Daily equity/sector data
lacks exploitable spectral structure beyond per-actor AR(1). Nonlinear dependence
exists (MI ≠ correlation) but is orthogonal to predictive dynamics.

### Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/smim/run_smim_iter3_data.py` | Phase 0: data construction + validation gates |
| `scripts/smim/run_smim_iter3_phase1.py` | Phase 1: daily linear SMIM (E3-1a/b/c/d) |
| `scripts/smim/run_smim_iter3_emergence.py` | Panel B: K sweep + MI operator + poly AR |
| `scripts/smim/run_smim_iter3_mi_drilldown.py` | MI drilldown: MI/corr/DMD basis comparison |

### Data Created

| File | Contents |
|------|----------|
| `data/smim/intensities/iter3_panel_a_L{20,60,120}.parquet` | Panel A momentum rank |
| `data/smim/intensities/iter3_panel_a_weekly.parquet` | Weekly aggregated Panel A |
| `data/smim/intensities/iter3_panel_b.parquet` | Panel B sector/macro daily |
| `data/smim/intensities/iter3_panel_b_weekly.parquet` | Weekly aggregated Panel B |
| `data/smim/intensities/iter3_capex_step.parquet` | Step-interpolated CapEx diagnostic |
| `results/metrics/iter3_*.parquet` | All experiment results |
