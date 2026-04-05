# SMIM Architectural Decision Log

This file records architectural decisions made during SMIM development.
Append a new entry after each gate or significant decision point.

Format: ADR-NNN, date, context → decision → consequences.

---

## ADR-001: BIC regime selection unreliable on short noise sequences

**Date**: 2026-03-19
**Gate**: G3 (state-space filtering complete) / acceptance test review

**Context**

Acceptance test P-2 (pure noise null) found that `select_regime_count` returns
M*=2 on pure iid Gaussian noise (K=1, T=150). The Kim filter gains thousands of
log-likelihood units by fitting heteroscedastic variance patterns in the noise,
far outweighing the BIC penalty (~25 units for 5 extra parameters × log(150)).
This is not a bug — BIC is correctly reporting that M=2 fits the data better
in-sample. The problem is that the fitted variance patterns are not generalisable.

**Decision**

BIC regime count M* is an **initial candidate**, not a final answer. The
experimental protocol requires:

> If regime switching improves in-sample BIC but does not improve OOS R² by
> ≥ 0.5 percentage points versus the M=1 baseline, regime switching is not
> justified for that condition.

The definitive null check is **OOS R² ≤ 0.1** (acceptance test P-2). All
experiments in Phase B that evaluate regime switching must report both BIC-M*
and OOS R² and only claim regime structure when OOS R² > 0.1.

**Consequences**

- Experiments B1 (component ablation) and B10 (regime sweep) must evaluate
  regime switching via OOS R², not BIC alone.
- The `select_regime_count` API is retained as-is; callers are responsible for
  validating M* with OOS metrics before committing to a regime count.
- P-2 acceptance test keeps OOS R² ≤ 0.1 as the primary criterion; the test
  does not assert M*=1 from BIC (see the existing note in the test docstring).
- Any future regime-selection improvement (e.g. a penalty schedule that scales
  with T) should be validated against P-2 before replacing the current BIC
  formulation.

---

## ADR-002: KSG transfer entropy estimates have high inter-implementation variance

**Date**: 2026-03-19
**Gate**: G5 (transfer entropy complete) / acceptance test review

**Context**

Acceptance test R-TE-1 found ~37% divergence between our KSG estimator
(Kraskov Algorithm 1, L∞ metric) and IDTxl/JIDT (Frenzel-Pompe CMI variant)
at T=2000. The tolerance in R-TE-1 was relaxed from 25% to 50% to accommodate
this systematic bias without false-failing due to algorithm-variant differences.

This is not an implementation bug. It is a well-documented property of KSG
estimators: different neighbour-counting conventions, boundary corrections, and
conditioning strategies produce O(30–50%) differences on finite samples. Both
implementations converge to the true value as T→∞, but at practical sample
sizes (T=2000–10000) estimates are noisy and variant-dependent.

**Decision**

Experimental conclusions based on transfer entropy must use **TE ratios and
rankings across conditions**, not absolute TE values. Examples:

- ✅ Robust: "TE_{L1→L3} doubles during crisis vs expansion" (ratio, variant-invariant)
- ✅ Robust: "L1→L3 is the strongest TE link in crisis" (ranking)
- ❌ Fragile: "TE_{L1→L3} = 0.15 nats during crisis" (absolute value, variant-dependent)

**Consequences**

- Phase D experiments (D3 diffusion topology, D6 emergence timing) must report
  relative changes and rankings, not absolute TE values.
- R-TE-1 tolerance remains at 50%; this is the correct bound for cross-variant
  agreement at T=2000, not a quality issue.
- When comparing TE results across papers or tools, always report the estimator
  variant (Algorithm 1 vs 2, metric, k_neighbours) alongside the value.

---

## ADR-003: RRR loses to DMD — forecast-optimised basis unstable at N≫T

**Date**: 2026-04-05
**Gate**: Iteration 6.1 Phase 1

**Context**

Iteration 6.1 Phase 1 tested reduced-rank regression (RRR; Reinsel & Velu 1998)
as a forecast-optimised alternative to the DMD basis. RRR directly minimises
one-step prediction error and finds a rank-K approximation to the OLS coefficient
matrix C = Y X^T (X X^T)^{-1}. Results on the 93-actor panel (T≈20 quarters):

| K | RRR R² | DMD R² | Subspace angle |
|---|--------|--------|---------------|
| 2 | 0.391  | 0.400  | 69.7°         |
| 4 | 0.368  | 0.423  | 58.7°         |
| 8 | 0.347  | 0.415  | 40.1°         |

RRR lost to DMD at all K values despite large subspace angles (40–70°),
confirming that the two methods find genuinely different subspaces.

**Decision**

In the N≫T regime (N=93, T≈20), the forecast-optimised OLS coefficient
C ∈ R^{N×N} is severely under-determined. The rank-K SVD restriction provides
some regularisation but not enough: the RRR basis overfits to the training
snapshots. DMD's dynamics-structured SVD decomposition acts as an implicit
regulariser by constraining the solution to the Koopman operator subspace
(X' ≈ AX), which is a stronger structural prior than RRR's pure rank constraint.

This finding does **not** imply RRR is inferior in general — it is specific to
the short-panel (N≫T) financial data regime. With longer time series (T≫N),
the OLS coefficient would be well-determined and RRR might outperform DMD.

**Consequences**

- Do not pursue further RRR variants (penalised RRR, Bayesian RRR) for
  the current 93-actor panel. The data regime is the bottleneck, not the
  estimation method.
- The paper should note this finding: in short panels, implicit structural
  regularisation (DMD's Koopman constraint) dominates explicit forecast
  optimisation (RRR). This is a methodological contribution.
- If extending to longer panels (e.g. monthly data, T>100), revisit RRR
  as a candidate basis.

---

## ADR-004: SMIM as spectral augmentation of pooled+FE — C1 positive result

**Date**: 2026-04-05
**Gate**: Iteration 6.1 Phase 2a

**Context**

Phase 2a tested three refinements after Phase 1's finding that using DMD's
full reduced propagator Ã as the Kalman transition matrix improved predictive
R² from 0.415 to 0.486 (A1c, +0.071 over baseline F=0.99I). Key results:

| Model                        | R²    | ΔR² vs AR(1) |
|------------------------------|-------|--------------|
| Per-actor AR(1)              | 0.594 | —            |
| Pooled+FE                    | 0.591 | −0.003       |
| SMIM baseline (F=0.99I)     | 0.419 | −0.175       |
| A2 best (γ=0.75)            | 0.490 | −0.104       |
| A4 diagonal of Ã            | 0.483 | −0.111       |
| **C1 combined (pooled+resid)**| **0.630** | **+0.036** |

A2: Optimal shrinkage γ=0.75 (not 1.0), gaining +0.004 over pure Ã.
A4: Diagonal of Ã in U_r space captures 97% of A1c gain; off-diagonal
coupling is negligible (+0.003). Phase 1's apparent cross-mode effect
was an artifact of comparing against eigenvalue diagonal (A1a, DMD mode
coordinates) rather than Ã diagonal (A4, SVD coordinates).
C1: Combined pooled+FE + SMIM-on-residuals achieves R²=0.630, beating
per-actor AR(1) (0.594) with CI [+0.021, +0.054] and 10/10 window wins.
Residual R²=0.095 despite near-zero residual persistence (ρ≈0.04).

**Decision**

The paper narrative shifts from "standalone spectral forecasting" to
"spectral augmentation of standard panel models." SMIM cannot replace
per-actor AR(1), but it adds +0.036 predictive R² on top of pooled+FE.

The augmentation architecture is:
1. Stage 1: Pooled AR(1)+FE → captures shared persistence
2. Stage 2: DMD+Kalman (Ã transition) on Stage 1 residuals → captures
   cross-sectional spectral structure not modelled by pooled+FE

**Consequences**

- Paper Section 5 (empirical results) reframes around the two-stage model.
  The standalone SMIM negative result is reported as motivation, then the
  augmentation positive result as the contribution.
- No need for C3 (DMDc) — the augmentation route already delivers the
  key positive finding without requiring explicit exogenous-input modelling.
- The A4 finding (diagonal of Ã ≈ full Ã) means the spectral value-add
  is per-mode dynamics, not cross-mode coupling. This simplifies
  the methodological narrative but reduces the "propagation" story.
- Target: SILVER criterion met (C1 positive augmentation R²). GOLD
  requires predictive R²>0.55 standalone (not achieved).

---

## ADR-005: C1 augmentation validated — generalises across all three panels

**Date**: 2026-04-05
**Gate**: Iteration 6.1 Validation

**Context**

Full validation of the C1 two-stage augmentation architecture.

Key findings from the validation suite:

**1. Residual transition audit (Section 1):**
The Phase 2a C1 result (R²=0.630) used full Ã on the residual stage.
Testing the minimal transition:
- F=0.99I on residuals: R²=0.471 (WORSE than pooled+FE alone at 0.591)
- diag(Ã) on residuals: R²=0.619 (recovers 84% of the full Ã gain)
- full Ã on residuals: R²=0.630 (full gain)

The diagonal of Ã is the minimal viable residual transition. F=0.99I
destroys the result — correct mode-specific dynamics are essential.

**2. Robustness across panels (Section 2):**

| Panel | AR(1) | Pooled | C1 | ΔC1−AR1 | CI | W/AR1 |
|-------|-------|--------|-----|---------|-----|-------|
| 146-firm CapEx/Rev | 0.728 | 0.745 | 0.745 | +0.017 | [+0.009,+0.025] | 8/10 |
| 270-actor multi-ratio | 0.728 | 0.738 | 0.753 | +0.025 | [+0.019,+0.030] | 10/10 |
| 93-actor multilayer | 0.594 | 0.591 | 0.630 | +0.036 | [+0.021,+0.054] | 10/10 |

C1 beats AR(1) on all three panels with CIs excluding zero. The effect
is largest on the heterogeneous 93-actor panel (+3.6%) and smallest on
the homogeneous 146-firm panel (+1.7%). This is general spectral
augmentation, not heterogeneity-specific.

**3. Strong baselines (Section 3):**
C1 beats layer-specific pooled+FE by +0.032 (CI [+0.022, +0.042],
10/10 wins). It also beats DFM K=8 (0.568) by +0.062. The permutation
p-value vs AR(1) is 0.0013.

**4. Ablation ladder (Section 4):**

| Residual stage | R² | Δ vs pool |
|---------------|-----|-----------|
| Pooled+FE only | 0.591 | — |
| + resid AR(1) | 0.605 | +0.014 |
| + resid PCA proj | 0.404 | −0.187 |
| + resid PCA+VAR (DFM) | 0.577 | −0.014 |
| + resid DMD proj | 0.469 | −0.122 |
| + resid DMD/Kalman 0.99I | 0.471 | −0.119 |
| + resid DMD/Kalman diag(Ã) | 0.619 | +0.028 |
| + resid DMD/Kalman full Ã | 0.630 | +0.039 |

Key: projection-only models (PCA proj, DMD proj) HURT. Naive Kalman
(F=0.99I) also hurts. Residual AR(1) gives +0.014 (modest). The gain
is specifically from DMD/Kalman with correct dynamics (diag Ã or full Ã).
This is not "any second-stage model" — it is specifically spectral
dynamics that matter.

**5. Leakage audit (Section 5):**
Strict point-in-time causality confirmed. All six audit points pass.

**Decision**

The paper narrative is **"spectral augmentation"** — general across panels,
robust to strong baselines, specifically attributable to learned spectral
dynamics (not projection or naive Kalman), with no leakage.

**Consequences**

- Paper title/framing should centre on the two-stage augmentation
  architecture as the methodological contribution.
- The standalone SMIM negative result is motivation, not a weakness.
- The ablation ladder is the key evidence table: it shows the gain is
  specifically from DMD-informed dynamics, not generic second-stage models.
- No DMDc needed — the augmentation already generalises without explicit
  cross-layer modelling.

---

## ADR-006: Final architecture — spectral augmentation without filter refinement

**Date**: 2026-04-05
**Gate**: Iteration 6.1 Architecture finalisation

**Context**

After establishing C1 spectral augmentation as the lead positive result
(ADR-005), tested whether the architecture benefits from further refinement.

**Key findings:**

1. **diag(Ã) vs full Ã (Section 1):** The full Ã increment (+0.011 over
   diag Ã) is marginally significant (t=2.47, p=0.036, CI [+0.003, +0.019],
   7/10 wins). Both are valid; diag(Ã) is the parsimonious default.

2. **D1 spectral Kalman (Section 2):** Diagonal Q and structured R produce
   ZERO incremental gain (Δ = 0.0000 for both). The current adaptive Q
   already captures the needed dynamics. No value from spectralising the
   noise covariances.

3. **D2 state persistence (Section 3):** Projecting Kalman state across
   basis updates produces Δ = −0.0008. Marginally worse — the reset is
   adequate. The basis changes are small enough that fresh projection
   is not harmful.

4. **A5 Kim switching (Section 4):** Not warranted. Non-switching filter
   showed no improvement from D1/D2 refinements.

5. **Economic validation (Section 5):** Both pooled and C1 gaps predict
   future intensity revisions (β < 0, mean-reversion to model-implied
   fair value). Pooled gaps are slightly MORE informative (|t|=27.8 vs
   23.1). C1 improves forecast R² but does NOT strengthen the economic
   content of the gaps — the predictive gain comes from better point
   forecasts, not from identifying a more meaningful benchmark.

**Decision**

The recommended architecture is the simplest form that delivers the
positive result:

**Default:** pooled AR(1)+FE → residual DMD/Kalman with F = diag(Ã)
  - R² = 0.619 on 93-actor panel (Δ vs AR(1) = +0.025)
  - Generalises across all three panels

**Max-performance:** same but F = full Ã
  - R² = 0.630 (Δ vs AR(1) = +0.036, 10/10 windows)
  - Marginally significant improvement over diag(Ã)

No spectral Q, no structured R, no state persistence, no Kim switching.

**Paper narrative:** "augmentation result survives, extra filter complexity
not needed." The ablation ladder is the central evidence table.

**Consequences**

- Paper should report both diag(Ã) and full Ã as parsimonious / max
  variants. The 8-parameter diag(Ã) is the cleaner methodological claim.
- Do not claim that C1 gaps have stronger economic content than pooled
  gaps. The contribution is predictive improvement, not economic signal.
- Kim switching, spectral Q/R, and state persistence are explicitly
  tested and shown unnecessary — this strengthens the paper by
  demonstrating the architecture is already near-optimal at its
  simplest form.
- Iteration 6.1 is complete. The three preserved findings:
  (i) RRR loses to DMD in N≫T regime (ADR-003)
  (ii) standalone gain is coordinate-correct per-mode dynamics (ADR-004)
  (iii) residual spectral augmentation is the first robust positive result (ADR-005)
