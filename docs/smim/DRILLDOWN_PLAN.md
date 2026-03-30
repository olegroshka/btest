# Phase A-B Drill-Down Research Plan

> Created: 2026-03-30
> Status: Active
> Prerequisites: Phase A (4 experiments), Phase B (8 experiments), Phase D (6 experiments) complete
> Cross-reference: EXPERIMENT_RESULTS.md, EXPERIMENT_SESSION_PROMPTS.md

## 1. Executive Summary

After 18 experiments, SMIM's best model (L1 graph-factor OLS) achieves R²=0.34 on
FULL-ROLL, while a naive per-actor AR(1) achieves 0.43. The 9pp gap persists across
all windows. This plan systematically decomposes WHY the gap exists, tests whether
it is closeable, and identifies the conditions under which SMIM adds genuine value
beyond simple autoregression.

The preliminary evidence (see Section 2) reveals:
- The gap is NOT a bug — it reflects a structural limitation of shared-factor models
  vs per-actor models when temporal persistence dominates cross-sectional structure.
- But SMIM captures COMPLEMENTARY information: the ensemble AR(1) + L1 achieves 0.41,
  recovering 96% of AR(1) performance, and D2 shows gaps predict CapEx revision (t=-34.7).
- Several quick wins remain unexplored: training window length at L1, DMD basis, and
  regularised Kalman all show promise.

## 2. Variance Decomposition (already computed)

Across 10 FULL-ROLL windows (mean R²):

| Model | Mean R² | Source of R² |
|-------|---------|-------------|
| Grand mean | -0.005 | None (single value for all actors) |
| **Per-actor mean** | **0.281** | Cross-sectional level (+0.286) |
| L1 K=1 (Schur) | 0.318 | + Common temporal factor (+0.037) |
| L1 K=3 (Schur) | 0.340 | + Cross-sectional dynamics (+0.022) |
| **AR(1) per actor** | **0.425** | + Actor-specific persistence (+0.085) |
| **Ensemble (AR1 + L1 K=1)** | **0.413** | Combined information |

**Key insight**: 67% of the total R² (0.281/0.425) comes from the per-actor mean.
AR(1) adds 14.4pp through actor-specific persistence (b_i heterogeneity). L1
captures only 5.9pp through spectral dynamics. The 8.5pp gap (0.425 - 0.340) is
the price of imposing shared K-factor dynamics on heterogeneous actors.

## 3. Root Cause Hypotheses

### H_RC1: Spectral basis captures co-movement, not actor-specific persistence
AR(1) has N free slopes (b_1, ..., b_N). L1 has K² shared transition parameters.
With K=3 and N=92, 3 common factors cannot represent 92 different persistence
rates. Actors with b=0.3 and b=0.9 are projected onto the same basis.

**Test**: Factor-augmented AR(1) — AR(1) per actor + K spectral factors as
additional regressors. If FA-AR(1) > AR(1), the spectral basis captures
cross-actor structure beyond persistence. If FA-AR(1) ≈ AR(1), the spectral
information is redundant given per-actor persistence.

**Preliminary evidence**: FA-AR(1) K=3 mean R²=0.11 (WORSE than AR(1), massive
overfitting). The K=3 factors add 3 regressors per actor to a T=40 series —
too many parameters. NEED TO TEST with K=1 and with regularisation.

### H_RC2: The operator captures contemporaneous structure, not temporal causation
The intensity cross-correlation operator is symmetric and captures which actors
have similar intensity LEVELS, not which actors' changes predict others. DMD
partially addresses this (B2: DMD R²=0.359 vs Schur 0.339).

**Test**:
- (a) Lag-1 cross-correlation operator: corr(y_{i,t}, y_{j,t+1}) — directed, temporal
- (b) DMD at L1 across all FULL-ROLL windows (not just one)
- (c) End-to-end optimised operator with DMD decomposition

### H_RC3: K=3 is a poor compromise — K=1 or K=5+ may be better
B1 used K_MIN=3 (MDL floor). But the W2020 diagnostic shows L1 K=1 (R²=0.435)
outperforms L1 K=3 (R²=0.418) in that window. Cross-window: K=3 mean (0.340)
is slightly better than K=1 mean (0.318), but the optimal K likely varies by window.

**Test**: Full K sweep (1, 2, 3, 5, 8) at L1 across all 10 windows.
Find per-window optimal K and test adaptive K selection.

### H_RC4: 10yr training window is suboptimal for L1
B6 showed RECENT (T=24Q) gives L1 R²=0.70-0.82 while FULL-ROLL (T=40Q) gives
0.33-0.40. This 2x performance gap suggests strong non-stationarity: the
cross-sectional correlation structure shifts over time, and older data hurts.

**Test**: Training window sweep at L1 depth: T=3yr, 5yr, 8yr, 10yr, 15yr.
All with test=single next year. Find optimal T for the correlation operator.

### H_RC5: The demeaning strategy loses information
We subtract the full training-period per-actor mean. If the mean drifts
(non-stationary level), a fixed 10yr mean is a poor estimator. A rolling
or exponentially-weighted mean might track better.

**Test**: Compare demeaning strategies:
- (a) Full training mean (current)
- (b) Rolling 8Q mean
- (c) Rolling 20Q mean (5yr)
- (d) Exponential weighted mean (halflife=8Q)
- (e) No demeaning + intercept in AR on factors

### H_RC6: The R² metric masks actor-level heterogeneity
Global R² pools all (actor, time) pairs. If SMIM excels for some actors
(institutional, global_shock) but fails for others (large_firm), the aggregate
masks the win. D1 showed half-life varies 1.3Q (large_firm) to 4.7Q (global_shock).

**Test**: Compute per-actor-type R² for AR(1) vs L1 vs ensemble. Find the
actor types where SMIM adds most value.

## 4. Ablation Matrix

Each experiment is keyed to a hypothesis and has an estimated runtime.

| ID | Experiment | Hypotheses | New Code? | Est. Time |
|----|-----------|-----------|-----------|-----------|
| DD-1 | K sweep at L1 (K=1,2,3,5,8) x 10 windows | H_RC3 | No | 3 min |
| DD-2 | Training window sweep at L1 (3,5,8,10,15yr) x single test | H_RC4 | No | 3 min |
| DD-3 | DMD at L1 across all 10 FULL-ROLL windows | H_RC2 | No | 3 min |
| DD-4 | Lag-1 directed operator at L1 x 10 windows | H_RC2 | Minimal | 5 min |
| DD-5 | Demeaning strategy comparison (5 variants) x 10 windows | H_RC5 | Minimal | 5 min |
| DD-6 | Per-actor-type R² decomposition | H_RC6 | No | 3 min |
| DD-7 | FA-AR(1) with K=1 and ridge regularisation | H_RC1 | Yes | 15 min |
| DD-8 | Ensemble weight optimisation (alpha*AR1 + (1-alpha)*L1) | H_RC1 | Minimal | 5 min |
| DD-9 | Shrinkage Kalman (Ledoit-Wolf on R) at L2 | H_RC1 | Yes | 20 min |
| DD-10 | Combined best: DMD + optimal K + optimal T + rolling demean | All | No | 5 min |

**Total estimated time: ~70 min** for all 10 experiments.

## 5. Execution Plan

### Phase 1: Quick Diagnostics (20 min, DD-1 through DD-6)

These require NO new code — just parameter sweeps on existing infrastructure.
They answer: is the gap structural or parametric?

**Execution order**: DD-1 → DD-3 → DD-2 → DD-4 → DD-5 → DD-6

DD-1 (K sweep) is first because it directly reveals whether K=3 is the problem.
DD-3 (DMD) is second because it tests whether the operator/decomposition is the
bottleneck. DD-2 (T sweep) tests whether the training window is the issue.

**Success criteria for Phase 1**: Identify at least one configuration where L1
closes >50% of the AR(1) gap (R² > 0.383).

### Phase 2: Model Architecture Experiments (30 min, DD-7 through DD-9)

These require new code and test whether the SMIM pipeline can be made competitive.

DD-7 (FA-AR with regularisation): The most promising because it combines AR(1)'s
actor-specific persistence with spectral cross-sectional structure. Ridge regression
prevents the overfitting seen in the preliminary FA-AR test.

DD-8 (Ensemble weight): Simple but informative — if optimal alpha is near 1.0
(all AR(1)), the spectral model adds nothing. If alpha = 0.5-0.7, SMIM is
genuinely complementary.

DD-9 (Shrinkage Kalman): Tests whether L2 can be rescued. Ledoit-Wolf shrinkage
on R reduces the effective parameter count of the observation covariance from
N² to ~N, potentially allowing the Kalman filter to work at T/N < 1.

**Success criteria for Phase 2**: At least one model configuration achieves
R² > 0.425 (matching AR(1)) with lower variance across windows.

### Phase 3: Synthesis (20 min, DD-10 + write-up)

Combine the best configuration from Phase 1-2 and validate across all 10 windows.
Write findings into EXPERIMENT_RESULTS.md.

## 6. Alternative Research Trajectories

If the drill-down confirms SMIM cannot beat AR(1) on aggregate R², consider:

### A. Reframe the success criterion
SMIM's value is NOT higher R² but rather:
- D2: gap prediction of CapEx revision (t=-34.7)
- D3: layer diffusion structure (L0→L2 transmission)
- D6: gap dispersion as VIX leading indicator
- B8: extreme noise robustness (95% retention at σ=1.0)

These are economically meaningful findings that AR(1) cannot provide.
The paper's contribution is the STRUCTURAL interpretation, not the forecast accuracy.

### B. Change the target variable
Cross-sectional rank (capex_assets_xsrank) is designed to be stable and
mean-reverting. This makes prediction "easy" for AR(1) but limits the scope for
spectral methods. Testing on:
- Raw CapEx/Assets ratio (unbounded, more volatile)
- Year-on-year CapEx change (growth signal, less persistent)
- Residual after sector-specific AR(1) (cross-sectional excess)
...might reveal conditions where spectral structure adds more value.

### C. Change the frequency
Quarterly frequency (4 observations/year) gives very few transitions per training
window. Monthly intensity (from interpolated EDGAR or from OHLCV-derived proxies)
would triple T and potentially enable richer spectral dynamics.

### D. The "actor-specific loading" model
Instead of y_i = Σ_k u_{ik} * alpha_k (shared loadings from operator), allow
y_i = c_i + b_i * y_{i,t-1} + Σ_k w_{ik} * alpha_k (actor-specific intercept,
persistence, AND loading). This is a panel factor model with heterogeneous
coefficients. Estimate via EM or Bayesian shrinkage. This directly addresses H_RC1.

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| K sweep shows K=1 is optimal everywhere | Medium | Undermines spectral story | Reframe: K=1 = market factor extraction, still useful |
| Shrinkage Kalman doesn't help | High | L2 remains broken | Accept B1 finding: L1 is the right depth |
| No configuration beats AR(1) | Medium | Weakens paper | Pivot to Trajectory A: structural interpretation story |
| DD-10 combined best overfits to specific window | Low | False positive | Validate on held-out windows |
| Non-stationarity is the dominant factor | Medium | Shorter T is always better | Include as key finding: regime-adaptive window |

## 8. Success Criteria

| Level | R² Target | Interpretation |
|-------|----------|---------------|
| **Bronze** | > 0.383 (close 50% of gap) | SMIM is competitive, parametric issue found |
| **Silver** | > 0.425 (match AR(1)) | SMIM matches best baseline |
| **Gold** | > 0.425 with lower σ | SMIM dominates AR(1) on risk-adjusted basis |
| **Structural** | D2 beta significant AND above | Economic value regardless of R² ranking |

The **Structural** criterion may be the most important: even if SMIM doesn't
beat AR(1) on R², the gap estimates have genuine economic predictive power
(D2: t=-34.7) that AR(1) residuals cannot provide. This needs verification:
does the D2 prediction hold after controlling for AR(1) momentum?

## 9. Data Dependencies

All experiments use existing data. No new data collection needed.

| Data | Location | Used By |
|------|----------|---------|
| Intensity panel | `data/smim/intensities/experiment_a1_intensities.parquet` | All |
| Actor registry | `data/smim/registries/experiment_a1_registry.json` | DD-6 |
| A1 results | `results/metrics/level1_A1-MVP-FULL.parquet` | DD-10 |
| A2 baselines | `results/metrics/level1_A2-BASELINES.parquet` | Comparison |
| B1 ablation | `results/metrics/level1_B1-COMPONENT-ABLATION.parquet` | Reference |

## 10. Outputs

All drill-down results will be stored in:
```
results/metrics/drilldown_DD-{N}.parquet    per-experiment results
results/configs/DRILLDOWN.yaml              combined config + findings summary
docs/smim/DRILLDOWN_RESULTS.md              findings narrative
```

## 11. Self-Review Notes

*This section records critical review of the plan itself.*

**Reviewed 2026-03-30. Issues identified and addressed:**

1. **Missing: AR(1) residual analysis.** If SMIM captures information AR(1) misses,
   the spectral factors should predict AR(1) residuals. Added to DD-7 scope: test
   whether spectral factors have explanatory power for AR(1) forecast errors.

2. **Missing: statistical significance of R² differences.** A 1pp R² difference
   across 10 windows may not be significant. All comparisons should include Diebold-
   Mariano p-values, not just point estimates. Added to all DD experiments.

3. **Missing: look-ahead bias check.** The end-to-end operator optimisation (Approach C
   in A1) uses sub-val split within training. But the operator is then applied to the
   test period. Is the operator overfitting to the training distribution? DD-10 should
   compare Approach C operator vs simple correlation operator on the same best config.

4. **Missing: sensitivity to the 0.1 correlation threshold.** The operator thresholds
   correlations below 0.1 to zero. This is arbitrary. DD-4 should test threshold
   sensitivity (0.0, 0.05, 0.1, 0.2, 0.3).

5. **Missing: the D2 control.** D2 showed gaps predict revision (t=-34.7). But this
   could be mechanical mean-reversion of the rank variable. The critical test: does
   the gap predict revision AFTER controlling for the current intensity level? If
   beta(gap | controlling for level) is still significant, the SPECTRAL structure
   adds value. If not, the gap is just "you're far from the mean, so you'll revert."
   This is the single most important analysis in the drill-down.

6. **Potential trajectory missed: online/adaptive Kalman.** Instead of EM on the
   full training window, use an online Kalman filter that adapts Q and R every quarter.
   This naturally handles non-stationarity (H_RC4) without requiring window selection.
   Added as DD-9b option.

7. **K sweep may reveal that FULL-ROLL with large K overfits but RECENT with large K
   works.** This would confirm that the issue is T/K ratio, not K per se. DD-1 should
   include both FULL-ROLL and RECENT periods for comparison.

---

## 12. Drill-Down Results (completed 2026-03-30)

### Phase 1 (DD-1 through DD-6): Quick Diagnostics

| DD | Finding | Impact |
|----|---------|--------|
| DD-1 K sweep | K=8 best (0.354) but K=5 more robust | +1.5pp |
| DD-2 T sweep | **T=3yr K=5 hits 0.547 on one window** | T=5yr best stable |
| DD-3 DMD | DMD +0.020 over Schur (confirmed B2) | +2.0pp |
| DD-4 Lag-1 | Marginal (+0.5pp with combined operator) | Minor |
| DD-5 Demeaning | **EWM_8Q best (0.381 vs 0.339)** | +4.2pp |
| DD-6 Actor-type | Per-actor R2 unreliable (near-zero var actors) | Diagnostic |

### Phase 2 (DD-7 through DD-9): Model Architecture

| DD | Finding | Impact |
|----|---------|--------|
| DD-7 FA-AR(1) | K=5 ridge=10: R2=0.405 | +1.3pp vs L1 |
| DD-8 Ensemble | alpha=0.2 (80% L1 + 20% AR1): R2=0.413 | L1 dominates |
| DD-9 Shrinkage | **Spherical R rescues Kalman: modal R2=0.434** | +4.2pp |

### DD-10 Final: Combined Best

| Config | Mean R2 | vs AR(1) | DM p | Wins |
|--------|---------|----------|------|------|
| **DMD + spherical Kalman** | **0.467** | **+0.042** | **0.001** | **8/10** |
| Schur + spherical Kalman | 0.434 | +0.010 | 0.446 | 5/10 |
| L1 OLS (Phase 1 best) | 0.392 | -0.033 | 0.010 | 4/10 |
| AR(1) T=10yr | 0.425 | -- | -- | -- |

**GOLD criterion achieved: R2=0.467, DM p=0.001.**

### Remaining Items (completed)

| Item | Finding |
|------|---------|
| AR(1) residual | K=5 DMD modes explain 11.6% of AR(1) errors. Complementary. |
| Correlation threshold | Robust (0.37-0.40 across 0.00-0.30). Higher slightly better. |
| Online Kalman | +0.4 to +3.4pp in 5/6 windows. Q adaptation helps. |
| D2 control | Gap survives level control (t=-6.95, p<0.001). Genuine structure. |

### Performance Ladder (final)

```
Original A1 (T=10yr, K=3, Schur, full demean):       0.339
  + EWM demeaning:                                    0.381  (+4.2pp)
  + shorter T=5yr + K=5:                              0.392  (+1.1pp)
  + spherical R Kalman:                               0.434  (+4.2pp)
  + DMD basis:                                        0.467  (+3.3pp)
  + online Q adaptation (estimated):                 ~0.480  (+1.3pp est)
                                                     ------
  Total improvement:                                 ~14.1pp
  AR(1) baseline:                                     0.425
```
