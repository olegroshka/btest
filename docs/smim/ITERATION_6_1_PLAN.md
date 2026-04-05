# Iteration 6.1: Methodological Ablation Programme 

> Date: 2026-04-05
> Status: PROPOSED
> Predecessor: Iteration 6 (SMIM loses to pooled+FE on all panels)
> Objective: Determine whether SMIM's forecasting failure is intrinsic to
>   spectral compression on financial panel data, or specific to the current
>   DMD + regularised Kalman implementation.

---

## 0. Why This Iteration Exists

Iteration 6 answered a panel-selection question. The answer was negative:
on every available panel, pooled AR(1)+FE matches or beats SMIM on point
forecasts, while SMIM remains strong only as a structural decomposition
method (modal R²=0.69 at K=8).

Iteration 6.1 is **not** a search for a friendlier dataset, metric, or
subsample. It is a **methodology ablation programme** on the **same
benchmark panels** and **same forecasting metric**, designed to answer:

> Did SMIM fail because spectral methods are the wrong tool for
> cross-sectional investment forecasting, or because the current
> implementation destroys the very spectral information it extracts?

If the answer is "wrong tool," then the honest negative result is broader
and more publishable than our current implementation-specific one. If the
answer is "fixable implementation," the paper regains a positive claim.
Either outcome is strictly better than where we are now.

---

## 1. The Core Diagnosis — Four Testable Hypotheses

Iteration 6 revealed a 0.277 modal-predictive gap (modal R²=0.69 →
predictive R²=0.42). Four concrete failure mechanisms can explain this:

### H1. Transition Mis-specification (the smoking gun)

DMD computes eigenvalues λ_k encoding mode-specific growth, decay, and
oscillation — and we **throw them away**, replacing the full spectral
dynamics with F=0.99I. The predict step α_{t|t-1} = 0.99·α_{t|t}
shrinks every mode identically toward zero. Per-actor AR(1) uses ρ≈0.6;
our Kalman uses 0.99 uniformly. We never tested whether DMD's own
temporal dynamics could improve *prediction* as opposed to
*reconstruction*.

    α_{t|t}  (filtered)     → modal R² = 0.69  ← spectral info PRESENT
        ↓
    α_{t|t-1} = 0.99·α_{t|t}  → uniform shrinkage destroys mode-specific dynamics
        ↓
    ŷ = U·α_{t|t-1} + μ̂   → predictive R² = 0.42  ← spectral info GONE

### H2. Wrong Modal Geometry

Raw DMD modes are non-orthogonal. For non-normal operators, left and right
eigenvectors differ. The current implementation uses U = Re(Φ) (real part of
right modes) and projects via U^T — but for non-orthogonal modes, the
correct projection requires biorthogonal duality (left modes Ψ such that
Ψ*Φ = I). Additionally, the spherical observation covariance R = cI may be
wrong in modal space even if reasonable in actor space.

### H3. Heterogeneous Observation Pooling

The 93-actor panel mixes minmax-normalised macro series (trending, ρ≈0.88)
with cross-sectionally ranked firm intensities (mean-reverting, ρ≈0.60). A
single spectral basis conflates different distributional regimes. The
proposal envisioned macro actors as exogenous inputs G^(z)·u_t (Eq. 11),
not as interchangeable peers in the same observation equation.

### H4. Under-Modelled Temporal Order

Standard DMD operates on one-step snapshot pairs (y_t, y_{t+1}). If
cross-layer propagation involves multi-quarter lags (oil at t-2 → Fed at
t-1 → firm CapEx at t), DMD's single-step pairs cannot capture this.
Delay embeddings or subspace methods may recover the hidden lag structure.

---

## 2. Fixed Testbed

### Panels (unchanged from Iteration 6)
1. **146-firm CapEx/Revenue** — homogeneous negative control
2. **270-actor multi-ratio** — heterogeneous persistence test
3. **93-actor multilayer** — main stress test (the decisive panel)

### Primary metric
Mean OOS **predictive R²** over the existing rolling windows.

### Secondary metrics (for diagnosis, not success claims)
- Modal R² (must not degrade under transition changes)
- Modal-predictive gap per mode (which modes are forecastable?)
- Basis rotation stability
- Innovation whiteness / calibration
- Subspace angle between competing bases (DMD vs RRR vs PCA)

### Baselines (run for every test)
- Per-actor AR(1)
- Pooled AR(1)+FE (single ρ)
- Layer-specific pooled AR(1)+FE (3 ρ's)
- DFM (PCA+VAR), K matched to SMIM's K

### What this iteration will NOT do
- Hunt for new panels where SMIM happens to win
- Change the target metric after seeing results
- Claim success from structural interpretability if forecasting still fails
- Add complex nonlinear/emergence machinery before fixing the linear core
- Cherry-pick windows, actor subsets, or evaluation horizons

---

## 3. Stream A — Fix the Transition Dynamics (attacks H1)

**Goal:** Test whether the predictive failure comes from F=0.99I rather
than from spectral decomposition itself.

### A1. DMD-Eigenvalue Diagonal F ★★★★★

Replace F=0.99I with F=diag(|λ_1|,...,|λ_K|) from DMD eigenvalues.

For complex conjugate pairs λ_k = r_k·e^{iθ_k}, use real 2×2 blocks:

    F_k = r_k · [[cos θ_k, -sin θ_k], [sin θ_k, cos θ_k]]

Real eigenvalues stay as 1×1 blocks. This preserves mode-specific decay,
oscillation, and rotation — the dynamics DMD was designed to extract.

**Sub-variants:**
- **A1a (simple):** F = diag(|λ_1|,...,|λ_K|), ignoring oscillatory
  components. Pure mode-specific shrinkage. Zero additional parameters.
- **A1b (proper):** Full real block-diagonal form with rotation-scaling
  blocks. Preserves oscillatory DMD dynamics.
- **A1c (stability-clipped):** Clip all |λ_k| to max 0.99 to prevent
  explosive modes. This is the safe variant if raw eigenvalues exceed unity.

**Zero additional parameters.** The eigenvalues are already computed.

**Kill:** F_DMD produces R² ≤ F=0.99I on the 93-actor panel.

**Win:** Predictive R² increases by ≥0.05 (from 0.415 to ≥0.465).

**Time:** 3h. **Priority:** Run first. Everything else depends on this.

### A2. Shrinkage Toward DMD Dynamics ★★★★

    F = γ·F_DMD + (1-γ)·(0.99·I)

Sweep γ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} by nested CV. At γ=0, this is
the current SMIM; at γ=1, this is A1. The optimal γ quantifies how much
DMD dynamics can be trusted.

**Why not just A1?** If DMD eigenvalues at T=20 are noisy, the optimal γ
may be 0.3 (use some but not all dynamics). Shrinkage is the principled
way to interpolate between the data-driven and prior-based extremes.

**Analogy:** Ledoit-Wolf shrinkage for covariance, applied to F.

**Time:** 3h.

### A3. VAR(1) on Modal Amplitudes ★★★

After extracting modal amplitudes α_t = U^T·ỹ_t for t=1,...,T_train,
estimate F via OLS: F_OLS = (Σ α_{t+1}α_t^T)(Σ α_tα_t^T)^{-1}. Apply
Ledoit-Wolf shrinkage: F_shrunk = γ·F_OLS + (1-γ)·(0.99·I).

**Why this differs from DFM:** DFM uses PCA basis + VAR. Here we use DMD
basis + VAR. The basis matters: if DMD captures different subspace than
PCA, the modal VAR coefficients may be more forecastable. DFM failed
(R²=0.568, Δ=-0.026), but with a different basis the VAR might succeed.

**Diagnostic value:** Comparing F_OLS, F_DMD (Ã), and F_EM tells us
whether the transition dynamics are estimable at all from T=20 quarters.
If all three agree → robust dynamics. If they disagree → insufficient data.

**Time:** 3h.

### A4. Low-Rank-Plus-Diagonal F ★★★

    F = D + UV^T

where D = diag(|λ_1|,...,|λ_K|) is the DMD-diagonal and UV^T is a rank-1
or rank-2 perturbation capturing the strongest cross-mode interaction.
Estimate UV^T via ridge-penalised regression on the modal innovations.

**Why:** If mode j at t predicts mode k at t+1 (e.g., macro mode drives
firm mode with one-quarter lag), a purely diagonal F cannot capture this.
But full K×K F overfits at T=20. Rank-1 perturbation tests for the
single strongest cross-mode channel.

**Time:** 3h.

### A5. Kim Filter (Regime-Switching F) — CONDITIONAL ★★★

Only after A1-A4, and only if some non-switching F shows improvement.

Run M=2 Kim filter with regime-specific F^(z) and Q^(z):
- Regime 1: F^(1) = F_DMD (normal spectral dynamics)
- Regime 2: F^(2) = 0.5·I (crisis: aggressive mean-reversion)

Regime count controlled by MDL/BIC. F^(z) constrained to be diagonal
(no cross-mode coupling within regime) to keep parameters manageable.

**Rule:** No regime switching until a non-switching spectral model is
already competitive. Otherwise we are adding flexibility to a broken core.

**Time:** 6h.

---

## 4. Stream B — Fix the Modal Geometry and Basis (attacks H2 + H4)

### B1. Biorthogonal DMD Projection ★★★★

For non-normal operators, left eigenvectors Ψ ≠ Φ (right eigenvectors).
The correct modal decomposition uses:
- Right modes Φ for reconstruction: ŷ = Φ·α
- Left modes Ψ for projection: α = Ψ^*·ỹ (where Ψ^*Φ = I)

Currently we use U = Re(Φ) for both, which is mathematically incorrect for
non-orthogonal modes. The biorthogonal projection may produce cleaner modal
amplitudes and reduce the innovation variance.

**Ablation:** Compare biorthogonal vs QR-orthonormalised DMD subspace.
If QR performs equally, the gain comes from the subspace, not the exact
coordinates; if biorthogonal wins, the non-orthogonal geometry matters.

**Time:** 3h.

### B2. Reduced-Rank Regression (Forecast-Optimised Basis) ★★★★★

Instead of DMD's objective (min ||X'-AX||_F), directly optimise for
one-step-ahead prediction:

    C = Y·X^T·(X·X^T)^{-1}     (OLS coefficient)
    C = L·Σ·R^T                  (SVD)
    C_K = L_K·Σ_K·R_K^T          (rank-K approximation)

Set U = L_K, B = Σ_K·R_K^T. Prediction: ŷ_{t+1} = μ̂ + U·B·U^T·ỹ_t.

**Why this is critical:** DMD optimises reconstruction; RRR optimises
prediction. If the best-reconstructing directions differ from the
best-predicting directions, DMD finds the wrong subspace. RRR is the
reduced-rank VAR(1) of Reinsel & Velu (1998) — cited in the proposal
but never implemented.

**Diagnostic:** Measure the subspace angle between DMD and RRR bases.
If the angle is large (>30°), the reconstruction-prediction mismatch is
real and the basis choice is the primary bottleneck.

**Time:** 4h.

### B3. Hankel-DMD (Time-Delay Embedding) ★★★★

Augment snapshots with d time-delayed copies before DMD:

    H = [[y_1, ..., y_{T-d}],
         [y_2, ..., y_{T-d+1}],
         ...
         [y_d, ..., y_T]]

DMD on the (N·d × (T-d)) Hankel matrix captures VAR(d)-equivalent dynamics.

**Why:** Standard DMD's single-step pairs cannot capture multi-quarter
lagged effects. If oil at t-2 predicts firm CapEx at t, Hankel with d=2
embeds this lag. Arbabi & Mezić (2017) proved convergence to Koopman
eigenvalues.

**Constraint:** At N=93, d=2, T=20 → H is 186×18 (underdetermined in
snapshots). Use d ∈ {2,3} with inner CV. The rank of the DMD is still
min(N·d, T-d) = T-d, so the effective rank doesn't increase — but the
embedding lets DMD learn lagged cross-actor structure.

**Time:** 5h.

### B4. Optimized DMD (OptDMD) ★★★★

OptDMD (Askham & Kutz 2018) uses variable projection to directly fit
exponential time dynamics y_k ≈ Σ φ_i·λ_i^k·b_i, jointly optimising
modes, eigenvalues, and amplitudes.

**Why:** Exact DMD eigenvalues are biased by measurement noise on short
panels. OptDMD's variable projection minimises reconstruction error over
the eigenvalue manifold, producing more accurate eigenvalues. These then
serve as more reliable transition parameters for A1/A2.

**Natural combination:** OptDMD → A1 (eigenvalue F) is the cleanest
"use DMD's own dynamics for prediction" test.

**Bagging extension (BOP-DMD):** Draw B=100 bootstrap samples of the T
columns, run OptDMD on each, average. Provides UQ over eigenvalues:
if σ(|λ_k|) > 0.2, eigenvalues are too noisy for F. This is diagnostic
even if R² doesn't improve.

**Time:** 5h (including bagging).

### B5. Subspace DMD ★★★

Project future snapshots onto the space of past snapshots before Koopman
estimation. Specifically designed for stochastic systems with observation
noise — exactly our setting.

**Why:** Standard DMD assumes noise-free dynamics (X'≈AX). Subspace DMD
and TLS-DMD explicitly model observation noise, potentially yielding
a better operator estimate on noisy quarterly financial data.

**Time:** 3h.

### B6. Forward-Backward DMD ★★★

Compute DMD forward (X'≈AX) and backward (X≈A^{-1}X'), combine:
λ_fb = √(λ_fwd/λ_bwd). Corrects eigenvalue bias at zero data cost.

**Time:** 2h. Quick test, run alongside B4.

---

## 5. Stream C — Architectural Redesign (attacks H3)

### C1. DMD on Residuals (Spectral Augmentation) ★★★★★

Two-stage model:
1. Pooled AR(1)+FE → ŷ^{AR}_t (captures shared persistence)
2. DMD+Kalman on residuals r_t = y_t - ŷ^{AR}_t

Final prediction: ŷ_t = ŷ^{AR}_t + U_r·α^{resid}_{t|t-1}

**Why this changes the strategic question:** Instead of SMIM vs pooled+FE,
we test whether SMIM adds value *on top of* pooled+FE. Any positive R² on
the residuals = pure spectral value-add. The residuals are demeaned by
construction and have lower persistence, which may actually be a *better*
input for DMD.

Variant: two-stage stacking with cross-validated blend weight γ:
ŷ = ŷ_pooled + γ·U_r·α^{resid}_{t|t-1}

**Kill:** R² on residuals ≤ 0.0 (no spectral structure in the residuals).

**Win:** R² > 0.0 on residuals AND combined model beats pooled+FE with CI
excluding zero.

**Time:** 4h.

### C2. Block-SMIM (Multi-View Spectral Model) ★★★★

Estimate **separate bases per layer**, couple via cross-block transition:

    U^(0) for macro (K_0=2), U^(1) for institutions (K_1=1), U^(2) for firms (K_2=4)

Cross-layer transition:

    α^(g)_{t+1} = F_{gg}·α^(g)_t + Σ_{h≠g} F_{gh}·α^(h)_t + η^(g)_t

Preserves heterogeneity while allowing cross-layer propagation. F_{gh}
matrices are small (K_g × K_h) and ridge-regularised toward zero.

**Why this is better than single-basis DMD:** A single basis forces
minmax-normalised macro series and xsrank-normalised firms into the same
spectral subspace. Block-SMIM respects layer-specific distributions while
still modelling the cross-layer dynamics that are SMIM's raison d'être.

**Stricter variant (Input-Output SMIM):** Layers 0/1 are **drivers**
(exogenous inputs u_t), Layer 2 is the **predicted output**. The system
becomes:

    α^{(2)}_{t+1} = F_{22}·α^{(2)}_t + G·u_t + η_t

where u_t = [α^{(0)}_t; α^{(1)}_t] are macro/institutional modal states.
This matches the proposal's original design (Eq. 11: G^(z)·u_t).

**Time:** 6h.

### C3. DMD with Control (DMDc) ★★★★

Treat Layer 0 (7 macro actors) as exogenous inputs u_t and Layer 2
(82 firms) as the system x_t. DMDc solves X' ≈ AX + BU:

    Ω = [X; U]  (89 × T-1)
    SVD: Ω = ÛŜV̂*
    [Ã|B̃] = Û*·Y·V̂·Ŝ^{-1}

The B matrix captures cross-layer propagation: if oil at t predicts firm
CapEx at t+1, B encodes this. Pooled+FE cannot.

**Relationship to C2:** DMDc is the formal operator-estimation version of
Block-SMIM's input-output variant. C2 estimates separate bases then
couples them; C3 estimates the coupled system jointly.

**Quality gate:** The estimated B matrix should have non-trivial entries
(macro variables actually matter). If B≈0, cross-layer propagation is not
present in the data.

**Time:** 5h.

### C4. Mode-Filtered Prediction (Per-Mode Diagnostic) ★★★★

Not all K=8 modes are forecastable. Decompose the modal-predictive gap
per mode:

    R²_pred(mode k) = 1 - Σ_t(α_{k,t} - α_{k,t|t-1})² / Σ_t α²_{k,t}

Use only the M ≤ K modes with positive predictive R² in the forecast.
For non-forecastable modes, use α_k = 0 (drop from prediction, keep for
structural analysis).

**This is diagnostic even if R² doesn't improve globally.** It reveals
the structure of the gap: do slow/macro modes predict while fast/local
modes don't? If so, the paper gains: "spectral methods predict at low
frequencies; simple baselines suffice at high frequencies."

**Time:** 2h. Run alongside A1.

---

## 6. Stream D — Filter and Estimation Refinements

### D1. Spectral Kalman Filter (Novel) ★★★★

Combine three spectral-aware modifications:
1. **Mode-specific F** from DMD eigenvalues (= A1)
2. **Diagonal Q** = diag(q_1,...,q_K): mode-specific innovation variance,
   computed from Kalman innovations per mode. Persistent modes (high |λ|)
   get low q; volatile modes get high q.
3. **Structured R** = U·D·U^T + σ²I: actors well-represented by the basis
   get lower observation noise; poorly represented actors get higher noise.

**Novel combination.** Individual pieces exist (DMD-Kalman: Nonomura et al.
2018; structured R: Ledoit-Wolf shrinkage). Combining them for spectral
state-space prediction would be a methodological contribution.

**Time:** 5h. Run after A1 validates the DMD-eigenvalue F.

### D2. Kalman State Persistence Across Basis Updates ★★★

Currently, every quarterly basis update **resets** the Kalman state:
α = U_new^T·ỹ_t, P = I. This discards all accumulated filtering
information. Fix: project the old state into the new basis:

    M = U_new^T · U_old          (K×K rotation)
    α_new = M · α_old
    P_new = M · P_old · M^T

Do NOT reset Q. This preserves the Kalman filter's accumulated knowledge
about state uncertainty and process noise across basis changes.

**Time:** 2h.

### D3. Observation Covariance in Modal Space ★★★

Instead of spherical R = cI in actor space, test:
- Diagonal R_α in modal coordinates (anisotropic in the spectral direction)
- Block-diagonal R_α for conjugate mode pairs
- Per-layer R: R = diag(r_{layer(i)}) with 3 parameters

**Why:** If the main error is anisotropic in modal space (some modes fit
well, others don't), spherical actor-space R throws away useful info.

**Time:** 2h.

### D4. Numerical Sanity: Square-Root Filtering ★★

Use square-root Kalman on the best variant. Check whether the reported
shrinkage/conditioning issues are partly numerical.

**Time:** 1h.

---

## 7. Stream E — Data and Evaluation Fixes

### E1. Uniform Normalisation ★★★★

Re-normalise the 93-actor panel with a single method:
- Option A: Cross-sectional percentile rank for ALL actors (including macro)
- Option B: Minmax normalisation for ALL actors
- Option C: Within-type normalisation (rank within macro, rank within firms),
  then standardise to common scale

Test all three. Report persistence (ρ) distribution under each scheme.

**Time:** 3h.

### E2. Multi-Horizon Evaluation ★★★

Evaluate at h=1,2,4 quarters. SMIM's spectral modes may predict medium-term
mean-reversion better than AR(1), even if they lose at h=1. At longer
horizons, AR(1) predictions converge to the mean (ρ^h → 0); SMIM's modal
structure may capture structural patterns only visible over multiple quarters.

**Variant (forecast-optimised horizon):** Apply DMD on time-shifted pairs
(y_t, y_{t+h}) instead of (y_t, y_{t+1}). The resulting modes are optimised
for h-step prediction. Inner CV selects h.

**Time:** 3h.

### E3. Extended DMD on Macro Actors Only ★★

Extended DMD with polynomial lifting (degree=2) failed at K=8 because
P=44 >> T=20. But applied to ONLY the 7 macro actors:
P = (7+1)(7+2)/2 = 36, T=20 → barely feasible.

Captures nonlinear macro dynamics (e.g., VIX² term). Feed the lifted
Koopman modes into DMDc-style cross-layer prediction for firms.

**Time:** 3h. Conditional on C3 showing cross-layer signal.

---

## 8. Execution Phases

### Phase 1: Core Diagnostics (10h)

Run the tests that give maximum information per hour. Decision point at end.

| Test | Time | Hypothesis | What it answers |
|------|------|-----------|----------------|
| **A1** DMD-eigenvalue F | 3h | H1 | Is F=0.99I the bottleneck? |
| **C4** Mode-filtered prediction | 2h | H1 | Which modes are forecastable? |
| **B2** Reduced-rank regression | 4h | H2 | Is DMD optimising the wrong objective? |
| **E1** Uniform normalisation | 3h | H3 | Is mixed normalisation a confound? |
| **E2** Multi-horizon | 2h | H4 | Is the prediction horizon wrong? |

*Some tests overlap in calendar time; total wall-clock ~10h.*

**Decision gate after Phase 1:**

| Outcome | Action |
|---------|--------|
| A1 gains ≥+0.05 | Spectral dynamics help. Proceed to Phase 2a (refine F). |
| B2 materially beats DMD basis | Basis was the bottleneck. Proceed with RRR basis + A1 F. |
| E1 reduces the gap substantially | Data confound confirmed. Re-run A1 on fixed panel. |
| A1 ≤ +0.01 AND C4 shows no forecastable modes | Spectral dynamics unestimable at T=20. Go to C1 (augmentation). |
| Everything fails | Proceed to Phase 2b (architectural changes) as last resort. |

### Phase 2a: Refine the Best Transition (10h, conditional on A1 success)

| Test | Time | Purpose |
|------|------|---------|
| **A2** Shrinkage F | 3h | Find optimal blend DMD↔prior |
| **B4** OptDMD + A1 | 5h | Noise-robust eigenvalues → better F |
| **B3** Hankel-DMD | 5h | Multi-lag dynamics in F |
| **D1** Spectral Kalman | 5h | Full spectral-aware filter |

### Phase 2b: Architectural Redesign (10h, if Phase 1 shows no F improvement)

| Test | Time | Purpose |
|------|------|---------|
| **C1** DMD on residuals | 4h | Augmentation — can SMIM add to pooled+FE? |
| **C2** Block-SMIM / IO-SMIM | 6h | Respect layer heterogeneity |
| **C3** DMDc | 5h | Formal cross-layer propagation |

### Phase 3: Deep Dives (10h, conditional on any Phase 2 win)

| Test | Time | Condition |
|------|------|-----------|
| **A3** VAR on modes | 3h | If A1 worked: compare F_OLS vs F_DMD vs F_EM |
| **A4** Low-rank+diagonal F | 3h | If A1 worked: test cross-mode coupling |
| **A5** Kim filter | 6h | If D1 worked: add regime switching |
| **B1** Biorthogonal DMD | 3h | If B2 showed subspace angle matters |
| **D2** State persistence | 2h | Always useful if any variant works |
| **E3** Extended DMD on macro | 3h | If C3 showed cross-layer signal |

---

## 9. Combinatorial Matrix

Tests are not independent. The most powerful configurations are likely
combinations:

| Combination | Description | Synergy |
|-------------|-------------|---------|
| E1 + A1 | Fix normalisation + eigenvalue F | Remove confound, then test dynamics |
| A1 + B4 | OptDMD eigenvalues as F | Better eigenvalues → better F |
| A1 + B3 | Hankel-DMD eigenvalues as F | Multi-lag dynamics in F |
| A1 + D1 | DMD F + diagonal Q + structured R | Full spectral Kalman |
| B2 + A1 | RRR basis + DMD-eigenvalue transition | Best-predicting subspace + data-informed dynamics |
| C1 + A1 | Residuals → DMD → eigenvalue F | Augmentation with best prediction |
| C2 + A1 | Block-SMIM + eigenvalue F | Layer-specific bases with proper dynamics |
| C3 + A1 | DMDc + eigenvalue F | Cross-layer propagation with spectral transition |

**The golden combination (if it works):**
E1 (fix normalisation) → B4 (OptDMD for robust eigenvalues) → D1 (spectral
Kalman with mode-specific F, diagonal Q, structured R) → C1 (augment
pooled+FE with spectral residuals).

---

## 10. Stream-Specific Kill Criteria

### Transition kill (after A1-A4)
If no F variant beats F=0.99I by ≥+0.03 on the 93-actor panel, stop
investing in transition dynamics. The issue lies upstream (basis or
measurement), not in the predict step.

### Basis kill (after B2-B6)
If RRR, Hankel-DMD, OptDMD, and Subspace DMD do not improve over plain
DMD, stop exploring DMD variants. The data may not contain exploitable
low-rank lagged structure at this T.

### Heterogeneity kill (after C2, C3, E1)
If Block-SMIM, DMDc, and uniform normalisation do not improve the
93-actor panel, the mixed-normalisation explanation is not the main culprit.

### Regime kill
No Kim/switching models unless a non-switching spectral model is already
competitive. Adding flexibility to a broken core wastes effort.

### Global kill (after Phase 2)
If NO variant beats current SMIM by ≥+0.03 predictive R² on the 93-actor
panel AND no variant matches layer-specific pooled+FE within 0.01:
**the failure is not implementation-specific.** Spectral forecasting on
these data is a genuinely hard negative result.

---

## 11. Success Criteria

### BRONZE
All Phase 1 tests completed with quality gates passing. Clear per-mode
decomposition of which spectral features are forecastable. Decision memo
on methodological limits. Subspace angle between DMD and RRR quantified.

### SILVER
C1 (DMD on residuals) shows positive spectral augmentation R², OR
A1 closes ≥25% of the modal-predictive gap, OR B2 (RRR) outperforms
DMD basis by ≥+0.03. Paper reframed as: "Spectral augmentation of
standard panel models" or "Forecast-optimised spectral bases."

### GOLD
A combination from §9 closes ≥50% of the modal-predictive gap
(predictive R² > 0.55 on the 93-actor panel). Spectral Kalman (D1) or
RRR (B2) is a standalone methodological contribution.

### PLATINUM
A variant beats pooled+FE with CI excluding zero on the 93-actor panel.
Paper regains a positive forecasting claim, now grounded in proper
spectral dynamics and demonstrated against the strongest baselines.

### HONEST FAIL
No variant clears the global kill. The paper explicitly states:

> The negative result is robust not just to panel choice but to a wide
> class of spectral, state-space, and operator-estimation variants.
> Cross-sectional spectral structure in US investment data is
> descriptive, not forecastable, regardless of the DMD variant,
> transition specification, or filter architecture.

This is a **stronger and more publishable** negative result than the
current implementation-specific finding.

---

## 12. What The Paper Becomes Under Each Outcome

### If Phase 1 or 2 succeeds
The paper stays structurally framed but gains a methodological contribution:
- The original regularised SMIM is structurally strong but predictively weak
- A [Koopman-consistent / RRR / spectral-Kalman / augmentation] variant
  recovers forecastability
- The key lesson is **which spectral-state-space design actually works**
  (and why the naive version fails)
- Target venues: JBES, J. Financial Econometrics, Computational Statistics

### If everything fails
The paper becomes cleaner, not weaker:
- Structural decomposition survives (modal R²=0.69, rotation, ablation)
- Forecasting edge does not survive even after aggressive methodology repair
- The contribution is a **disciplined negative result** with precise diagnostic:
  the modal-predictive gap is X% transition-attributable, Y% basis-attributable,
  Z% intrinsic (from C4's per-mode decomposition)
- Target venues: same as before, but with stronger negative result framing

---

## 13. Falsification Commitment

The single most important test is **A1 (DMD-eigenvalue F)** because it
tests the deepest thesis: that DMD's own temporal dynamics can improve
spectral prediction. This is the minimum viable test of H1.

The second most important test is **B2 (RRR)** because it tests whether
the basis itself is wrong for prediction. If both A1 and B2 fail, the
spectral forecasting thesis has been tested at both the dynamics and basis
levels and found wanting.

The third most important test is **C1 (DMD on residuals)** because it is
the most forgiving formulation: SMIM only needs to find *any* structure
beyond pooled+FE.

If A1, B2, AND C1 all fail, the negative result is definitive at three
levels: wrong dynamics, wrong basis, and no residual structure. No amount
of further DMD-variant exploration will change this, and we should accept
the structural framing with full confidence.

---

## 14. Recommended First Experiment

If only one thing is done next, it should be:

**A1 + C4 hybrid (3h):** Run DMD-eigenvalue F on the 93-actor panel AND
compute the per-mode predictive R² decomposition.

This single experiment simultaneously:
1. Tests whether DMD's own dynamics improve prediction (H1)
2. Reveals which of the 8 modes carry forecastable information
3. Determines whether the benefit (if any) comes from slow/macro modes
   (supporting the propagation story) or fast/local modes

If that experiment fails, the next move is **C1 (augmentation)** —
testing whether SMIM can add to pooled+FE rather than replace it.

If both fail, we proceed to **B2 (RRR)** — testing whether DMD is
extracting the wrong subspace entirely.

Three negative results from {A1, C1, B2} = definitive negative. Proceed
to the honest-fail paper with confidence.

---

## 15. Quality Gates

### Universal (all experiments)
- QG-U1: Baseline SMIM reproduces R²=0.415 from Iter 6.0 (±0.002)
- QG-U2: No NaN/Inf in any predictions
- QG-U3: Modal R² not degraded by transition changes (check separately)
- QG-U4: Report condition number of F for stability check
- QG-U5: Report max |λ_k| for DMD eigenvalues (all should be ≤1 after clipping)

### Stream-specific
- **A1:** Report DMD Ã eigenvalues and whether improvement is monotonic across modes
- **A2:** Report optimal γ; is the improvement monotonic in γ?
- **B1:** Report Ψ^*Φ matrix condition number (biorthogonality quality)
- **B2:** Report subspace angle between RRR and DMD bases; F-test on RRR coefficients
- **B3:** Report Hankel matrix rank vs d; optimal d from inner CV
- **C1:** Report residual panel persistence (ρ) — should be lower than raw panel
- **C2:** Report cross-layer F_{gh} entries; F-test on A_{20} (macro → firms)
- **C3:** Report B matrix entries and significance (macro inputs matter?)

---

## 16. Files

| Existing File | Role |
|---------------|------|
| `src/.../smim/spectral/dmd.py` | Modify: return Ã and eigenvalues; add OptDMD, Hankel-DMD |
| `src/.../smim/dynamics/kim_filter.py` | A5 (Kim filter) |
| `src/.../smim/dynamics/kalman.py` | D1 (spectral Kalman), D2 (state persistence) |
| `src/.../smim/validation/metrics.py` | OOS R², DM test (reuse) |
| `scripts/smim/run_iter6_test3.py` | Template for 93-actor predictive R² |
| `scripts/smim/run_baselines_iter5_3.py` | Pooled+FE baselines (for C1 residuals) |
| `data/smim/intensities/experiment_a1_intensities.parquet` | 93-actor panel |

| New File | Role |
|----------|------|
| `src/.../smim/spectral/optdmd.py` | OptDMD (B4) |
| `src/.../smim/spectral/hankel_dmd.py` | Hankel-DMD (B3) |
| `src/.../smim/spectral/rrr.py` | Reduced-rank regression (B2) |
| `src/.../smim/dynamics/spectral_kalman.py` | Spectral Kalman (D1) |
| `scripts/smim/run_iter6_1_phase1.py` | Phase 1: A1, C4, B2, E1, E2 |
| `scripts/smim/run_iter6_1_phase2.py` | Phase 2: depends on Phase 1 outcome |
| `results/metrics/iter6_1_*.parquet` | Per-window results |
| `docs/smim/ITERATION_6_1_DECISION.md` | Decision memo |