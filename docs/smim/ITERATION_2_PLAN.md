# SMIM Experiment Iteration 2: Emergence & Directed Operators

> Created: 2026-04-01
> Status: Active
> Baseline: GOLD+ R²=0.524 (DMD K=8, spherical R Kalman, EWM, online Q)
> Target: BRONZE = emergence fires, SILVER = directed > symmetric, GOLD = R²>0.55

## 1. Executive Summary

Iteration 1 achieved R²=0.524 through regularisation innovations (spherical R,
online Q, EWM demeaning) and DMD basis. However, two core SMIM claims remain
unvalidated:

1. **Emergence does not fire** — PID synergy contributes zero delta-R² because
   T=20 gives unreliable MI estimates for K=8 mode pairs.
2. **Directed operators collapse to PCA** — the correlation operator is symmetric,
   so all directional decompositions (Schur, Polar, Hermitian) produce identical
   results.

This iteration targets both weaknesses through 6 focused experiments.

## 2. Experiment Plan

| ID | Experiment | Hypothesis | Est. Time | Priority |
|----|-----------|-----------|-----------|----------|
| E2-1 | Transfer Entropy Operator | TE asymmetry breaks PCA collapse | 15 min | HIGH |
| E2-2 | Granger on Intensity | Granger on primary data creates directed edges | 10 min | HIGH |
| E2-3 | Actor-Specific Loadings | Per-actor w_i captures heterogeneity | 10 min | HIGH |
| E2-4 | Economic Emergence Signals | Observable features beat abstract PID | 15 min | HIGH |
| E2-5 | Kim Filter K-means Init | Asymmetric init enables regime switching | 10 min | MED |
| E2-6 | Combined Best | Stack winning innovations | 10 min | — |

## 3. Experiment Details

### E2-1: Transfer Entropy Operator

**Hypothesis**: TE(i→j) ≠ TE(j→i) by construction. A TE-based operator produces
genuinely asymmetric spectral decompositions where Schur ≠ PCA.

**Method**:
1. For each training window, compute pairwise TE from intensity series
   - Use T=10yr (40Q, n=39 points) for reliable KSG estimation
   - ksg_transfer_entropy(source=y_j, target=y_i, lag=1, k_neighbours=5)
2. Build directed operator W[i,j] = TE(j→i) (how much j influences i's future)
3. Decompose W with Schur (complex, ordered by eigenvalue modulus)
4. Run GOLD+ pipeline with TE-derived basis U_TE
5. Compare: TE-Schur vs TE-PCA vs correlation-DMD (GOLD+)

**Metrics**: R², DM test vs GOLD+, symmetry measure |W - W^T|/|W|

**Risk**: KSG with n=39 points and k=5 has high variance. Mitigated by using
the TE matrix as structure (relative magnitudes), not absolute values.

### E2-2: Granger on Intensity Operator

**Hypothesis**: Granger causality between intensity series reveals lagged
cross-prediction structure that the contemporaneous correlation misses.

**Method**:
1. For each pair (i,j), test: does y_j,1:t-1 predict y_i,t given y_i,1:t-1?
   - OLS: y_{i,t} = a + b*y_{i,t-1} + c*y_{j,t-1} + eps
   - Test c != 0 with F-test, record |c| as edge weight
2. Build directed adjacency: W[i,j] = |c_{j→i}| if significant, 0 otherwise
3. Decompose and evaluate as in E2-1

**Difference from B3**: B3 tested OHLCV signals as Granger inputs. This tests
INTENSITY itself as Granger source. Different data, different question.

### E2-3: Actor-Specific Loadings

**Hypothesis**: The 14.4pp gap between AR(1) (0.425) and per-actor mean (0.281)
comes from heterogeneous persistence. Per-actor weights on spectral factors can
capture this.

**Method**:
1. Run GOLD+ pipeline to get basis U and filtered alpha_t
2. For each actor i, estimate w_i via OLS on training data:
   y_{i,t} = mu_i + w_i * (U[i,:] @ alpha_t) + eps
3. On test data: pred_i = mu_i + w_i * (U[i,:] @ alpha_pred_t)
4. Compare R² vs GOLD+ (shared w_i = 1 for all actors)

**Note**: This adds N parameters (one per actor). Ridge regularisation on w_i
prevents overfitting. Test with lambda = {0, 0.01, 0.1, 1.0, 10.0}.

### E2-4: Economic Emergence Signals

**Hypothesis**: Observable economic features (dispersion, rotation) capture
emergence better than abstract PID synergy between spectral modes.

**Method**: Compute three emergence metrics per quarter:
1. **Cross-sectional dispersion**: sigma_t = std(y_{i,t}) across actors
   - D6 showed this leads VIX by 1-4Q → economically meaningful
2. **Sector rotation velocity**: |d/dt mean_sector_intensity| averaged across sectors
   - Measures how fast capital reallocates between sectors
3. **Concentration ratio**: Herfindahl of abs(y_{i,t} - mean_t)
   - High concentration = few actors diverge = potential misallocation

Add as correction: y_pred_i,t = GOLD+_pred + beta * [sigma_t, rotation_t, HHI_t]
Estimate beta via leave-one-window-out CV.

### E2-5: Kim Filter K-means Init

**Hypothesis**: Symmetric F initialization prevents regime separation. K-means
on the alpha trajectory provides asymmetric starting points that EM can exploit.

**Method**:
1. Run GOLD+ Kalman filter on training data to get alpha_filtered
2. K-means on alpha trajectory → M=2 clusters
3. Estimate cluster-specific F matrices: F[m] = VAR(1) on cluster m's alpha
4. Initialize KimFilter with these asymmetric F[0], F[1]
5. Run Kim EM estimation
6. Compare M=2-kmeans vs M=2-symmetric vs M=1

### E2-6: Combined Best

Stack all experiments that produced positive delta-R². Run on all 10 windows.
Validate with DM tests vs GOLD+ and vs AR(1).

## 4. Execution Order

E2-1 and E2-2 are independent (both build directed operators). Run sequentially.
E2-3 and E2-4 build on GOLD+ baseline. Can run after E2-1/E2-2.
E2-5 is independent.
E2-6 combines the best.

## 5. Success Criteria

| Level | Criterion | What it means |
|-------|----------|---------------|
| BRONZE | Any emergence signal gives delta-R² > 0 | Emergence is measurable |
| SILVER | Directed operator outperforms symmetric | Schur ≠ PCA with real impact |
| GOLD | R² > 0.55 with emergence active | Full framework validated |

## 6. Dead Ends (do not retry)

- PID synergy with T=20 quarterly data (unreliable MI at K=8)
- External signals at L1 depth (B3: intensity correlation is sufficient)
- Kim M>1 with symmetric initialization (known failure mode)
- Longer training window T (SMIM needs short T for current structure)
- Return-based intensity (R²=-0.15, no structure to exploit)
- TE/Granger operators as spectral basis (Kalman diverges at K>=5)
- Actor-specific loadings (DMD basis already optimal)
- Emergence features as additive correction (redundant with DMD-Kalman)
- EM estimation of F (overfits; F near-identity is strictly better)

## 7. Results (completed 2026-04-01)

### Per-Experiment Outcomes

| ID | Result | R² | Delta vs GOLD+ | Verdict |
|----|--------|-----|----------------|---------|
| E2-1 | TE operator IS asymmetric (1.17) but noisy | 0.36 (K=3) | -0.16 | NEGATIVE |
| E2-2 | Granger directed but diverges at K>=5 | 0.37 (K=3) | -0.15 | NEGATIVE |
| E2-3 | Actor loadings hurt | 0.519 | -0.005 | NEGATIVE |
| E2-4 | Emergence features redundant | 0.521 | -0.003 | BRONZE FAIL |
| E2-4b | Dispersion weighting hurts | 0.524 | 0.000 | NEGATIVE |
| E2-5 | Kim M=2 helps (+1.4pp) via F effect | 0.537 | +0.014 | INSIGHT |
| **E2-5b** | **F=0.99*I beats EM F, 10/10 wins** | **0.538** | **+0.014** | **KEY FINDING** |
| **E2-6** | **F_reg + Q=0.5: PLATINUM config** | **0.543** | **+0.019** | **NEW BEST** |

### Success Criteria Assessment

| Level | Criterion | Outcome |
|-------|----------|---------|
| BRONZE | Emergence delta > 0 | **FAIL** -- no emergence variant adds R² |
| SILVER | Directed > symmetric | **FAIL** -- directed operators worse as bases |
| GOLD | R² > 0.55 | **FAIL** -- reached 0.543, close but below |
| **Bonus** | **Pipeline simplification** | **PASS** -- EM F removed, +1.9pp** |

### Key Insight: Dual Regularisation

Iteration 1 discovered that the observation covariance R must be regularised
(spherical R eliminates N^2 parameters). Iteration 2 discovered the exact same
principle applies to the transition matrix F: EM-estimated F (K^2 parameters)
overfits, and F=0.99*I (1 parameter) is strictly better.

The winning pipeline regularises at BOTH levels:
- R: spherical (1 parameter vs N^2)
- F: near-identity (1 parameter vs K^2)
- Q: online adaptation handles all temporal dynamics

This "dual regularisation" is the combined technical contribution of iterations 1+2.

### Updated Performance Ladder

```
Original A1 (T=10yr, K=3, Schur, full demean):       0.339
  + EWM demeaning:                                    0.381  (+4.2pp)
  + shorter T=5yr + K=5:                              0.392  (+1.1pp)
  + spherical R Kalman:                               0.434  (+4.2pp)
  + DMD basis:                                        0.467  (+3.3pp)
  + online Q adaptation + K=8:                        0.524  (+5.7pp)
  + F regularisation (F=0.99*I, no EM):               0.538  (+1.4pp)
  + Q=0.5*I (higher initial Q):                       0.543  (+0.5pp)
                                                     ------
  Total improvement:                                 +20.4pp
  AR(1) baseline (T=10yr):                            0.425

  FINAL BEST (PLATINUM): F_reg + Q_0.5 + DMD K=8 + sph R = 0.543
  vs AR(1): +11.8pp, wins 10/10 windows
  Peak: W2020 R2=0.659
```
