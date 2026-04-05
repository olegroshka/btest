# Iteration 5.3: Pooled AR(1)+FE and Dynamic Factor Model Baselines

> Created: 2026-04-05
> Status: PLANNED
> Predecessor: Iteration 5.2 (parameter space exploration) + Paper R1-R5 reviews
> Current headline: nested CV R^2=0.711, delta=+0.042 vs AR(1), 8/8 wins
> Motivation: Six independent reviewers unanimously requested stronger baselines

---

## 1. Motivation

The paper currently compares SMIM against per-actor AR(1) (no pooling) and PCA
variants (contemporaneous pooling). The gap between "no pooling" and "spectral
pooling" has no intermediate benchmark. Reviewers demand:

1. **Pooled AR(1) + firm FE** -- the simplest cross-sectional pooling model.
   Shares a single rho across all firms while retaining firm intercepts.
   If this matches SMIM, the spectral machinery is unnecessary.

2. **Dynamic Factor Model (DFM)** -- the standard econometric approach for
   exactly this problem (Stock & Watson 2002, Doz et al. 2012). PCA factor
   extraction + VAR(1) dynamics on K=2 factors. This is the closest
   econometric cousin to SMIM and the baseline a JoE referee will demand.

The truncated PCA baseline (R4) ruled out SVD regularisation as the source
of DMD's advantage. These two new baselines test whether the advantage
survives against standard pooled econometric methods.

---

## 2. Baselines to Implement

### Baseline 1: Pooled AR(1) with Firm Fixed Effects

**Model:**
```
y_{i,t} = alpha_i + rho * y_{i,t-1} + eps_{i,t}
```
- Single shared rho (pooled across all N firms)
- Firm-specific intercepts alpha_i
- Estimated via within-transformation (firm-demean, then OLS)
- Prediction: y_hat_{i,t} = alpha_i + rho * y_{i,t-1}

**Key property:** Pools information via shared rho while retaining
firm heterogeneity. This is the simplest intermediate between per-actor
AR(1) (N separate rho_i) and SMIM (spectral pooling).

### Baseline 2: Dynamic Factor Model (PCA + VAR(1))

**Model:**
```
Step 1: EWM demeaning (same tau as SMIM)
Step 2: PCA extraction --> Lambda in R^{N x K}, f_t = Lambda^T * y_tilde_t
Step 3: VAR(1) on factors: f_{t+1} = A * f_t + eta_t  (A in R^{K x K})
Step 4: Forecast: y_hat_t = mu_hat + Lambda * A * f_{t-1}
```
- Uses same EWM demeaning as SMIM
- K=2 factors (matching SMIM)
- Rolling basis update each quarter (matching SMIM)
- VAR(1) coefficient matrix A estimated by OLS (4 parameters at K=2)

**Key property:** Standard econometric dynamic factor model. Tests whether
the DMD temporal-pair formulation captures something VAR(1) on PCA factors
misses.

---

## 3. Evaluation Protocol

**Must match existing pipeline exactly:**
- Same panel: 146-firm CapEx/Revenue
- Same test years: 2015-2024 (10 windows)
- Same rolling update: expand training each quarter, re-estimate everything
- Same R^2: pooled over actor-quarter pairs per window (Eq. 1)
- Same output: per-window R^2, mean, delta vs AR(1), wins

**Configurations to run:**
1. Fixed config T=3yr, tau=12Q, K=2 (matches existing Table 3 middle block)
2. Fixed config T=2yr, tau=8Q, K=2 (matches Table 3 robustness block)
3. Bootstrap CI and exact permutation test for SMIM-vs-new-baseline deltas

---

## 4. Implementation

**Script:** `scripts/smim/run_baselines_iter5_3.py`

Based on the existing `run_smim_iter5_1_cv2.py` pipeline:
- Uses `build_panel()` from iter5_1_cv2 for the same 146-firm panel
- Uses `ewm_demean()` from iter5_1_cv2 for matched demeaning
- Uses `oos_r_squared` from `smim.validation.metrics`
- Adds PooledAR1FE and DFM classes
- Runs both at T=3yr and T=2yr configs
- Compares against existing SMIM and AR(1) results from saved parquets
- Saves results to `results/metrics/iter5_3_*.parquet`

**Quality gates:**
- QG1: AR(1) from new script reproduces existing 0.699 (T=3yr) and 0.671 (T=2yr)
- QG2: Pooled rho should be ~0.28 (median per-actor rho for CapEx/Revenue)
- QG3: Pooled AR(1)+FE R^2 >= per-actor AR(1) (pooling should not hurt)
- QG4: DFM R^2 >= PCA projection-only R^2 (VAR dynamics should help vs static)
- QG5: VAR(1) A matrix eigenvalues inside unit circle (stable dynamics)
- QG6: No NaN/Inf in any prediction

---

## 5. Paper Updates

After running, add to Table 3:
```
Fixed configuration (K=2, tau=12Q, T=3yr):
  ...existing...
  Pooled AR(1) + firm FE    10 windows   [R^2]  [delta]  [wins]
  DFM (PCA + VAR(1))        10 windows   [R^2]  [delta]  [wins]

Robustness (K=2, tau=8Q, T=2yr):
  ...existing...
  Pooled AR(1) + firm FE    10 windows   [R^2]  [delta]  [wins]
  DFM (PCA + VAR(1))        10 windows   [R^2]  [delta]  [wins]
```

Add to Table 4 (inference) if SMIM vs new baseline is significant.

Add paragraph to Section 4.1 interpreting results (see scenario guide below).

---

## 6. Expected Scenarios

**A: Pooled+FE ~ AR(1), DFM ~ PCA, both << SMIM**
Best case. Simple pooling doesn't help; standard DFM doesn't capture what
DMD captures. Confirms temporal snapshot formulation is the key.

**B: Pooled+FE > AR(1) but << SMIM, DFM between PCA and SMIM**
Good. Pooling helps marginally, DFM captures some dynamics, but SMIM's
DMD formulation adds value beyond both.

**C: DFM ~ SMIM**
Challenging. Standard DFM captures the same information. Paper's contribution
shifts from "DMD is better" to "DMD achieves DFM-level performance with
simpler estimation (no VAR, just fixed scalars)."

**D: DFM >> SMIM**
Unlikely but would mean SMIM's regularisation is too aggressive. Would need
to consider loosening F=0.99 constraint.

---

## 7. Success Criteria

- BRONZE: Both baselines implemented, evaluated, and added to Table 3
- SILVER: SMIM beats both in mean R^2 at both T=3yr and T=2yr
- GOLD: SMIM beats both with CI excluding zero at T=2yr
- PLATINUM: DFM provides informative decomposition (e.g., DFM ~ PCA+Kalman,
  confirming VAR(1) ~ regularised Kalman at K=2)

---

## 8. References

- Main pipeline: `scripts/smim/run_smim_iter5_1_cv2.py`
- PCA baseline: `scripts/smim/run_pca_baseline.py`
- Panel construction: `build_panel()` in iter5_1_cv2.py (146-firm CapEx/Revenue)
- EWM demeaning: `ewm_demean()` in iter5_1_cv2.py
- R^2 metric: `src/quantdsl_backtest/smim/validation/metrics.py::oos_r_squared`
- Existing results: `results/metrics/iter5_1v2_*.parquet`
- Paper: `docs/smim/paper/smim_paper.tex` (Table 3 at lines 395-420)
- SMIM CLAUDE.md: `docs/smim/CLAUDE.md`
