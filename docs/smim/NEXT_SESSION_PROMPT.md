# Iteration 6.1 — Methodological Ablation Sessions

Read `docs/smim/ITERATION_6_1_PLAN.md` first. It contains the full rationale,
hypotheses, kill criteria, and combinatorial matrix.

---

# SESSION 1: Phase 1 Core Diagnostics (A1 + C4 + B2 + E1)

## Context you need to know

Iteration 6 found a **modal-predictive gap of 0.277** on the 93-actor panel:
modal R²=0.69 (good reconstruction) but predictive R²=0.42 (loses to AR(1)
at 0.59). The spectral structure is real but we destroy it during prediction
by replacing DMD's learned dynamics with F=0.99I.

This session runs the four highest-information experiments from the plan.
Each attacks a different hypothesis about why prediction fails.

## What to do

### Experiment A1: DMD-Eigenvalue Diagonal F (attacks H1: transition mis-specification)

**The most important experiment.** DMD already computes eigenvalues λ_k that
encode mode-specific growth, decay, and oscillation. Currently we throw them
away. Test using them as the transition matrix.

Write `scripts/smim/run_iter6_1_phase1.py`. For the 93-actor panel at
T=5yr, K=8, τ=12Q, implement:

1. **Baseline:** Current SMIM (F=0.99I). Must reproduce R²=0.415 (±0.002).

2. **A1a (simple diagonal):** After `dmd_basis()`, extract eigenvalues from
   `mf.metadata["Atilde"]` via `np.linalg.eig(Atilde)`. Build F as:
   - For real eigenvalues: F_kk = clip(|λ_k|, 0.01, 0.99)
   - For complex conjugate pairs λ = r·e^{iθ}: 2×2 rotation-scaling block
     F_block = clip(r, 0.01, 0.99) · [[cos θ, -sin θ], [sin θ, cos θ]]
   This is the `_to_real_modes` structure already in `dmd.py`.

3. **A1b (stability-clipped):** Same as A1a but clip all |λ_k| to max 0.95
   (more conservative).

4. **A1c (full Ã):** Use F = Atilde directly (the full reduced propagator,
   not just eigenvalues). Clip spectral radius to 0.99 via:
   `eigenvalues, W = eig(Atilde); scale = min(1, 0.99/max(|eigenvalues|));
   F = Atilde * scale`

**Critical implementation detail:** The DMD basis and eigenvalues are
re-estimated every quarter (rolling). After each basis update, extract the
NEW Ã and rebuild F from its eigenvalues. F changes every quarter along
with U.

### Experiment C4: Per-Mode Predictive R² Decomposition (diagnostic)

For the baseline SMIM run and for A1a, decompose the prediction into per-mode
contributions:

For each mode k=1,...,K:
- Compute per-mode prediction: ŷ_k = U[:,k] · α_{k,t|t-1}
- Compute per-mode actual: a_k = U[:,k] · α_{k,t|t} (filtered)
- R²_pred(k) = correlation between predicted and actual mode amplitudes

Report a table: mode k, |λ_k|, R²_modal(k), R²_pred(k), gap(k).

This reveals WHICH modes are forecastable. Hypothesis: slow modes (|λ|≈1)
should be more forecastable than fast modes (|λ|≈0.5).

### Experiment B2: Reduced-Rank Regression (attacks H2: wrong basis)

Implement RRR as an alternative to DMD:

```python
def rrr_basis(otr, k=2):
    """Reduced-rank regression basis (Reinsel & Velu 1998)."""
    X = otr[:-1].T   # (N, T-1) — predictors
    Y = otr[1:].T    # (N, T-1) — targets
    # OLS coefficient: C = Y X^T (X X^T)^{-1}
    XXT_inv = np.linalg.solve(X @ X.T + 1e-6*np.eye(N), np.eye(N))
    C = Y @ X.T @ XXT_inv
    # Rank-K approximation via SVD
    L, S, Rt = np.linalg.svd(C, full_matrices=False)
    U = L[:, :k]       # forecast-optimised basis
    B = np.diag(S[:k]) @ Rt[:k, :]  # transition
    return U, B
```

Prediction: ŷ_{t+1} = μ̂ + U · B · U^T · (y_t - μ̂)

Run with K=2,4,8 matching SMIM. Report:
- R² for each K
- **Subspace angle** between RRR and DMD bases (use `scipy.linalg.subspace_angles`)
- If angle > 30°: the reconstruction-prediction mismatch is real

### Experiment E1: Uniform Normalisation (attacks H3: heterogeneous pooling)

The 93-actor panel mixes minmax (macro, ρ≈0.88) and xsrank (firms, ρ≈0.60).
Test whether this confounds the spectral decomposition.

Load the raw intensities. Re-normalise ALL actors using cross-sectional
percentile rank per quarter (matching the 146-firm methodology). Then re-run
baseline SMIM and A1a on the re-normalised panel.

```python
# For each quarter, rank all 93 actors cross-sectionally
panel_ranked = panel.rank(axis=1, method='average', pct=True)
```

Report how per-layer ρ changes after uniform ranking.

### Output format

```
PHASE 1 RESULTS (93-actor panel, T=5yr, K=8):

A1 — TRANSITION DYNAMICS:
  Variant               R²     ΔR² vs AR1  ΔR² vs baseline  Wins
  Baseline (F=0.99I)    0.415  -0.179       ---              0/10
  A1a (eigenvalue F)    0.XXX  +0.XXX       +0.XXX           X/10
  A1b (clipped 0.95)    0.XXX  +0.XXX       +0.XXX           X/10
  A1c (full Ã)          0.XXX  +0.XXX       +0.XXX           X/10

  DMD eigenvalues (mean across windows):
  Mode   |λ|    θ(deg)  R²_modal  R²_pred  Gap
  1      0.XXX  XX.X    0.XXX     0.XXX    0.XXX
  ...
  8      0.XXX  XX.X    0.XXX     0.XXX    0.XXX

B2 — REDUCED-RANK REGRESSION:
  K    RRR R²   DMD R²   Subspace angle
  2    0.XXX    0.XXX    XX.X°
  4    0.XXX    0.XXX    XX.X°
  8    0.XXX    0.XXX    XX.X°

E1 — UNIFORM NORMALISATION:
  Model (on re-ranked panel)    R²     ΔR² vs AR1
  Per-actor AR(1)               0.XXX
  Pooled+FE                     0.XXX  +0.XXX
  SMIM baseline (F=0.99I)       0.XXX  +0.XXX
  SMIM A1a (eigenvalue F)       0.XXX  +0.XXX

PHASE 1 DECISION:
  [A1 gained ≥+0.05 → Phase 2a: refine F]
  [B2 beats DMD → Phase 2a: use RRR basis]
  [E1 helps → re-run A1 on fixed panel]
  [All fail → Phase 2b: architectural changes]
```

Save results to `results/metrics/iter6_1_phase1_*.parquet`.

### Quality gates

- QG1: Baseline SMIM reproduces R²=0.415 (±0.002)
- QG2: Per-actor AR(1) reproduces R²=0.594 (±0.002)
- QG3: No NaN/Inf in any predictions
- QG4: Report max |λ_k| for DMD eigenvalues in each window
- QG5: Modal R² should not degrade under A1 variants (check separately)

### Key files

| File | Role |
|------|------|
| `docs/smim/ITERATION_6_1_PLAN.md` | Full plan — read first |
| `scripts/smim/run_iter6_test3.py` | Template: 93-actor baseline runs |
| `src/quantdsl_backtest/smim/spectral/dmd.py` | DMD impl — Ã in metadata |
| `data/smim/intensities/experiment_a1_intensities.parquet` | 93-actor panel |
| `data/smim/registries/experiment_a1_registry.json` | Layer labels |
| `results/metrics/iter6_test3_t5yr.parquet` | Iter 6.0 baseline results |

### What NOT to do

- Do NOT run Phase 2 experiments yet — Phase 1 results determine the path
- Do NOT modify existing scripts or results
- Do NOT change the evaluation metric or panel
- Do NOT implement Kim filter, Hankel-DMD, or DMDc in this session
- Do NOT update the paper — just collect results

---

# SESSION 2: Phase 2a — Refine the Best Variant

## Context

Read Phase 1 results before starting. The path depends on which experiments
succeeded.

## If A1 gained ≥+0.05 (eigenvalue F helps)

Run experiments A2, A4, B4, D1 from the plan:

1. **A2 (Shrinkage):** F = γ·F_DMD + (1-γ)·0.99I. Sweep γ ∈ {0, 0.1, 0.25,
   0.5, 0.75, 1.0}. Report optimal γ.

2. **A4 (Low-rank+diagonal F):** F = D + uv^T where D = eigenvalue diagonal.
   Estimate the rank-1 perturbation from modal innovations via ridge regression.

3. **B4 (OptDMD → A1):** Implement OptDMD using variable projection (or use
   BOP-DMD bagging). Extract more robust eigenvalues. Feed into A1-style F.
   Report eigenvalue uncertainty σ(|λ_k|).

4. **D1 (Spectral Kalman):** Combine eigenvalue F + diagonal Q (mode-specific
   innovation variance) + structured R (basis-aware observation noise). This
   is the full spectral-aware filter.

5. **D2 (State persistence):** Test projecting Kalman state into new basis
   instead of resetting. Apply to whichever F variant is best.

## If B2 won (RRR basis beats DMD)

1. Run A1 variants using the RRR basis instead of DMD basis.
2. Compare: RRR + eigenvalue-F vs RRR + OLS-VAR-F.
3. Report subspace angle evolution across rolling windows.

## If E1 helped (uniform normalisation matters)

1. Re-run A1 and B2 on the uniformly-normalised panel.
2. Report whether the gap closes further.

## If all Phase 1 experiments failed

Proceed to Phase 2b (Session 3).

### Output: Same format as Session 1, extended with new variants.

---

# SESSION 3: Phase 2b — Architectural Redesign

## Context

Only reach this session if Phase 1 showed no transition or basis improvement.
This tests whether SMIM can add value *on top of* pooled+FE rather than
replacing it, and whether layer-specific decomposition helps.

## What to do

### Experiment C1: DMD on Pooled+FE Residuals

Two-stage model:
1. Compute pooled+FE predictions (reuse from iter5_3 code)
2. Compute residuals: r_t = y_t - ŷ_pool_t
3. Run DMD + Kalman on residual panel
4. Final: ŷ = ŷ_pool + γ · U_resid · α^{resid}_{t|t-1}

The residuals have lower persistence and are demeaned — potentially a better
input for DMD. Cross-validate γ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}.

**Kill:** R² on residuals ≤ 0.0 (no spectral structure in residuals).
**Win:** Combined model beats pooled+FE.

### Experiment C2: Block-SMIM / Input-Output SMIM

Separate DMD per layer:
- Layer 0 (7 macro): K_0=2
- Layer 2 (82 firms): K_2=4

Cross-layer transition: α^(2)_{t+1} = F_{22}·α^(2)_t + G·[α^(0)_t] + η

This directly implements the proposal's Eq. 11 (exogenous macro inputs).
Ridge-regularise G toward zero.

### Experiment C3: DMDc (DMD with Control)

Stack macro as inputs: Ω = [X_firms; X_macro], Y = X'_firms.
DMDc gives [Ã|B̃] = Û*·Y·V̂·Ŝ^{-1}.

**Quality gate:** B matrix entries should be non-trivial (F-test).

### Output format

```
PHASE 2b RESULTS:

C1 — SPECTRAL AUGMENTATION:
  Residual DMD R²:     0.XXX  (>0 = spectral structure exists in residuals)
  Combined R²:         0.XXX  (ΔR² vs pooled+FE = +0.XXX)
  Optimal blend γ:     X.X

C2 — BLOCK-SMIM:
  Layer 0 R² (K=2):    0.XXX
  Layer 2 R² (K=4):    0.XXX
  Cross-layer G norm:  0.XXX  (>0 = propagation exists)
  Combined R²:         0.XXX

C3 — DMDc:
  A matrix spec radius: 0.XXX
  B matrix norm:        0.XXX
  DMDc R²:              0.XXX

VERDICT: [AUGMENTATION WORKS / PROPAGATION FOUND / DEFINITIVE NEGATIVE]
```

---

# SESSION 4: Paper Revision (After All Experiments)

## Context

All Phase 1 and Phase 2 experiments are complete. Read all results.

## Decision tree

Read `docs/smim/ITERATION_6_1_PLAN.md` Section 12 for the paper outcomes.

### If any experiment reached SILVER or above

Update the paper:
- Add the successful variant(s) to the results
- Reframe the contribution around the methodological finding
- Keep honest about what works and what doesn't
- Add a "methodology ablation" section

### If HONEST FAIL

Update the paper with the stronger negative result:
- "Robust to a wide class of spectral, state-space, and operator variants"
- Include the per-mode decomposition (C4) as diagnostic
- Include subspace angle analysis (B2) as evidence
- The negative result is now definitive at three levels:
  dynamics (A1), basis (B2), and residual structure (C1)

### In either case

- Update `docs/smim/ITERATION_6_1_PLAN.md` with final status
- Write `docs/smim/ITERATION_6_1_DECISION.md`
- Update ITERATION_6_PLAN.md to reference 6.1 results
