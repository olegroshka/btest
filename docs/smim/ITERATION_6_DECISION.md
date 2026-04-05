# Iteration 6 Decision Memo

> Date: 2026-04-05
> Decision: Reframe paper as structural/methods contribution; strip forecasting claims

---

## Summary of findings

Three tests were run to determine whether SMIM's spectral state-space
machinery adds forecasting value beyond pooled AR(1) with fixed effects.
All three produced negative results. Two additional tests (sector-structured
panel, directed operators) were skipped per the early-termination rule.

## Quantitative results

### Test 1: Multi-ratio panel (270 actors)

The paper's reported +2.8pp advantage of SMIM over AR(1) on the multi-ratio
panel is fully explained by heterogeneous persistence across ratio types.
A pooled AR(1) with two ratio-specific rho values (rho_capex=0.153,
rho_revass=-0.059) achieves R^2=0.739, matching SMIM K=2's R^2=0.737
(gap=0.001, CI [-0.008, +0.004]). SMIM K=4 is worse (R^2=0.722).

**Implication**: The "+2.8pp" figure in the paper is not evidence for
spectral methods. It is evidence that CapEx/Revenue and Revenue/Assets
have different persistence, a fact captured by two parameters.

### Test 3: 93-actor multilayer panel (the decisive test)

SMIM was designed for this panel: 93 heterogeneous actors across macro
shocks (7), institutions (4), and firms (82) with different persistence
profiles. Results at T=5yr:

| Model | R^2 | Delta vs AR(1) | Wins |
|-------|-----|----------------|------|
| Per-actor AR(1) | 0.594 | --- | --- |
| Pooled+FE (single rho) | 0.591 | -0.003 | 4/10 |
| Layer-specific pooled+FE | 0.598 | +0.004 | 5/10 |
| SMIM K=8 (predictive) | 0.415 | -0.179 | 0/10 |
| SMIM K=8 (modal) | 0.692 | --- | --- |

The modal-predictive gap of 0.277 is the central finding: the 8 spectral
modes reconstruct the current cross-section well (modal R^2=0.69) but
carry no one-step-ahead predictive information. The spectral structure is
descriptive, not forecastable.

### Test 4: Regime-break diagnostic

Basis rotation angle shows no positive correlation with SMIM advantage
(Pearson r=-0.24, p=0.50). SMIM does not benefit from structural breaks.
High-rotation windows (2018 tariff war, 2022 Fed tightening) actually
show slightly worse SMIM performance.

## Why SMIM fails

1. **The Kalman predict step destroys spectral information.** With F=0.99I,
   the predicted state alpha_{t|t-1} = 0.99 * alpha_{t|t} shrinks toward
   zero. After projection through U, this produces a prediction that is
   ~99% of the EWM mean plus ~1% of the spectral structure. Per-actor
   AR(1) does better because it uses the actual lag-1 correlation (rho=0.6)
   rather than the conservative F=0.99.

2. **Heterogeneous normalisation across actors.** The 93-actor panel mixes
   minmax-normalised macro series (trending, rho~0.88) with cross-sectionally
   ranked firm intensities (mean-reverting, rho~0.60). Pooling these in a
   shared spectral basis conflates different distributional regimes.

3. **Layer 2 dominance (88% of actors).** The heterogeneity argument requires
   meaningful minority layers. With 82/93 actors in Layer 2, pooled rho is
   not dramatically wrong for the majority. Layer-specific rho adds only
   +0.004 over single rho.

4. **All factor models fail, not just SMIM.** DFM (PCA+VAR) also loses to
   per-actor AR(1), suggesting the problem is dimension reduction itself,
   not the specific spectral method.

## Decision

### What survives

The following contributions are genuine and do not depend on forecasting:

- **Spectral decomposition method**: DMD + dual regularisation (spherical R,
  F=0.99I) + rolling quarterly basis provides a principled, automated
  pipeline for cross-sectional decomposition of panel data.

- **Modal analysis**: Modal R^2=0.69 at K=8 on the 93-actor panel. The
  spectral reconstruction of current-period cross-sectional structure is
  strong. This is a descriptive/structural contribution.

- **Rotation and ablation**: Basis rotation (25.8 deg/Q), the ablation
  ladder showing rolling > static > no-Kalman, and the gap decomposition
  are genuine structural findings about how cross-sectional investment
  patterns evolve.

- **Negative result on forecasting**: The modal-predictive gap of 0.28
  is itself an informative finding. It quantifies the limits of spectral
  forecasting on investment data and explains WHY: spectral structure
  describes contemporaneous co-movement, not lagged dynamics.

### What must change in the paper

1. **Strip all forecasting superiority claims.** SMIM does not beat pooled+FE
   on point forecasts. The abstract, introduction, and conclusion must
   reflect this honestly.

2. **Add pooled+FE baselines to all forecast tables.** Already partially
   done in Iter 5.3. Extend to multi-ratio and 93-actor panels.

3. **Reframe the contribution** as structural/methods:
   - "We develop a spectral state-space pipeline for cross-sectional
     investment analysis and show that it provides strong descriptive
     decomposition (modal R^2=0.69) with interpretable structure
     (rotation, ablation, gap decomposition). However, this structural
     decomposition does not translate into forecasting advantage over
     simple pooled econometric baselines."

4. **Present the modal-predictive gap as a key finding**, not a limitation.
   It answers the question: "Is spectral structure in investment data
   forecastable?" Answer: mostly not, on US data 2010-2024.

5. **Keep all structural analysis sections.** The ablation, rotation,
   zero-shot transfer, and emergence diagnostics are valid as descriptive
   analysis. They just don't carry forecasting implications.

### Target venue

With honest negative framing, the paper is suitable for:
- **Journal of Business & Economic Statistics (JBES)**: methods + negative result
- **Computational Statistics & Data Analysis**: methodological contribution
- **Journal of Financial Econometrics**: if the structural analysis is foregrounded

Not suitable (with current results) for:
- Top finance journals (JF, JFE, RFS) — no positive forecasting result
- Econometrica / REStud — insufficient theoretical contribution

### What NOT to do

- Do not run additional tests hoping for a different answer. The decisive
  test (93-actor panel) was pre-specified as the make-or-break test, and
  the result is unambiguous (0/10 wins, -0.18 delta).
- Do not cherry-pick subsets of actors or time periods.
- Do not change the evaluation metric post-hoc.
- Do not abandon the paper entirely — the negative result and the structural
  analysis have genuine value.
