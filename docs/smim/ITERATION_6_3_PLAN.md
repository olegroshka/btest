# Iteration 6.3: Forecasting the Rotating Geometry

> Date: 2026-04-06
> Status: PROPOSED (final synthesis)
> Predecessor: Iteration 6.2 (DMD ≈ PCA ≈ Ridge — method-agnostic ceiling)
> Core idea: The forecastable object is the SUBSPACE ORIENTATION, not the
>   actor values. Change the prediction target from y_{t+1} to (U_{t+1}, α_{t+1}).

---

## 0. Why This Iteration Is Different

Iterations 6.0–6.2 all asked the same question with increasing precision:
"Can spectral methods predict actor-level investment intensity better than
simple baselines?" The answer is consistently no — all methods converge
to the same ~0.630 R² ceiling.

But the structural results tell a different story: an 8-dimensional spectral
basis genuinely rotates at 25.8°/Q (σ=6.5°), with stable dimensionality
across 52 quarterly transitions. Modal R²=0.69 confirms the structure is
real. The rotation speed varies meaningfully across economic regimes
(2018 tariff war: 37°, 2022 Fed tightening: 38°, COVID 2020: 22°).

**The diagnosis:** Every method we have tested predicts in a FIXED coordinate
frame — the basis U_t estimated from training data through time t. By the
time t+1 arrives, the true co-movement structure has rotated ~26°. We
predict in yesterday's coordinate system. This is analogous to:

- Trying to demodulate a radio signal without a phase-locked loop
- Predicting particle motion in the lab frame instead of the rotating frame
- Parallel-transporting a vector along a curve without accounting for curvature

No amount of improving the amplitude prediction (α_{t+1|t}) within the stale
frame U_t can fix this — the frame itself is wrong. Ridge, PCA, and DMD all
suffer equally because none of them predicts the frame.

**The 6.3 hypothesis:** If the rotation is partially predictable, then
predicting U_{t+1} and α_{t+1} separately — geometry and amplitude —
may break through the 0.630 ceiling where all flat-space methods plateau.

**This is NOT another forecasting horse race.** It is a geometry-targeted
iteration. The object being forecast changes from actor values to the
rotating subspace itself. If the subspace evolution is forecastable but
does not improve actor-level prediction, that is still a novel structural
finding worth reporting. If it does improve actor-level prediction, it is
a genuinely new contribution that no flat-space method can replicate.

---

## 1. Mathematical Framework

### 1.1 The Fiber Bundle Structure

The full state of the system at time t is a point on a fiber bundle:

    (U_t, α_t)  ∈  Gr(K, N) × R^K

where:
- **Base space:** Gr(K, N) = the Grassmannian manifold of K-dimensional
  subspaces of R^N. This encodes WHICH actors co-move (the orientation
  of the spectral basis).
- **Fiber:** R^K = the modal amplitudes within the current subspace.
  This encodes HOW MUCH each mode is activated.

The observation model is:

    y_t = μ_t + U_t · α_t + ε_t

Current methods treat U_t as given (from training) and predict only α_{t+1}.
Iteration 6.3 predicts BOTH components:

    Û_{t+1|t} = geometry model(U_1, ..., U_t)
    α̂_{t+1|t} = amplitude model(α_1, ..., α_t, Û_{t+1|t})
    ŷ_{t+1|t} = μ̂ + Û_{t+1|t} · α̂_{t+1|t}

### 1.2 The Rotation Sequence

Between consecutive quarters, the subspace rotates. The rotation is
captured by the Procrustes alignment:

    R_t = argmin_{R ∈ O(K)}  ||U_t − U_{t-1} · R||_F

Computed via SVD of U_{t-1}^T U_t = VΣW^T → R_t = VW^T.

R_t ∈ O(K) is a K×K rotation matrix with K(K−1)/2 = 28 free parameters
on SO(8). The rotation angle (geodesic distance on Gr(K,N)) is:

    d(U_t, U_{t-1}) = ||θ||_2 = √(Σ_k θ_k²)

where θ_1, ..., θ_K are the principal angles between subspaces.

We already know this quantity: 25.8°/Q (σ=6.5°).

**Key data advantage:** The rotation sequence R_1, ..., R_T uses ALL T≈52
quarterly transitions (2010Q1–2024Q4). This is 52 data points for a
28-dimensional rotation — much more favourable than the T≈20 training
windows used for actor-level prediction. The rotation dynamics may be
estimable even when actor-level dynamics are not.

### 1.3 Tangent-Space Representation

The Grassmannian Gr(K,N) is a smooth manifold with tangent space at U:

    T_U Gr(K,N) = {V ∈ R^{N×K} : U^T V = 0}

This has dimension K(N−K) = 8×85 = 680 — too high for direct estimation.
But the logarithmic map Log_U(U') gives the tangent vector connecting
two nearby subspaces:

    Δ_t = Log_{U_{t-1}}(U_t)  ∈  T_{U_{t-1}} Gr(K,N)

This tangent vector encodes the direction and magnitude of the rotation.
If {Δ_t} is autocorrelated, the rotation is predictable.

**Dimensionality reduction in tangent space:** The 680-dimensional tangent
vector Δ_t has most of its energy in a few principal directions (the
dominant principal angles). Compressing Δ_t to its leading d ≤ K
components gives a d-dimensional time series that tracks the subspace
evolution. At d=K=8, this is an 8-dimensional time series over T=52
quarters — entirely feasible for AR/VAR modelling.

### 1.4 Projector Representation and Hilbert-Schmidt Space

Each subspace can be represented as its orthogonal projector:

    P_t = U_t U_t^T  ∈  R^{N×N}

The space of N×N symmetric matrices with the Hilbert-Schmidt (Frobenius)
inner product ⟨A, B⟩_HS = tr(A^T B) is a Hilbert space. The projector
trajectory P_1, ..., P_T lives in this Hilbert space, and linear
operations (averaging, regression) are well-defined.

**Key property:** P_t has rank K=8, so it lives in a K(2N−K+1)/2-dimensional
affine subspace of the symmetric matrix space. Direct HS-linear regression
on P_t is equivalent to working with the K×N representation U_t, avoiding
the N²=8,649 dimensionality.

### 1.5 Phase-Amplitude Decomposition

DMD modes come in conjugate pairs (λ_k, λ_k*). For each pair, the
real-valued modal amplitude oscillates:

    α_k(t) = A_k(t) · cos(ω_k t + φ_k(t))

where A_k(t) is the slowly-varying envelope and φ_k(t) is the phase.
The analytic signal via Hilbert transform decomposes this:

    z_k(t) = α_k(t) + i·H[α_k](t) = A_k(t) · e^{i(ω_k t + φ_k(t))}

This gives instantaneous amplitude A_k = |z_k| and phase φ_k = arg(z_k).

**The key hypothesis:** Amplitude evolves slowly (A_{k,t+1} ≈ ρ_A · A_{k,t}).
Phase advances at approximately constant rate (φ_{k,t+1} ≈ φ_{k,t} + ω_k).
Predicting each separately is EASIER than predicting their nonlinear product
α_k = A·cos(φ). The mapping (A, φ) → α is nonlinear — exactly the
separation that linear methods (DMD, PCA, Ridge) cannot perform.

### 1.6 Connection to Existing Literature

**Grassmannian subspace tracking** (Saad-Falcon et al. 2024, Sasfi et al.
2024/2025 "GREAT", Bharadwaj et al. 2025 "GeRoST"): Algorithms for
tracking evolving subspaces on the Grassmannian with convergence guarantees.

**Dynamic subspace models** (Vaswani et al. 2018, Narayanamurthy & Vaswani
2019 "Dynamic Robust PCA"): Treats the evolving subspace as the primary
latent variable.

**Koopman on manifolds** (Brunton et al. 2017 "HAVOK", Williams et al. 2015):
When dynamics live on a manifold, Koopman eigenfunctions must respect the
manifold structure.

**Coupled oscillator theory** (Kuramoto 1984): If spectral modes behave as
coupled oscillators with phase-locking, the Kuramoto model provides the
canonical mathematical framework.

---

## 2. Gate A — Is the Rotation Predictable?

### Goal

Determine whether the subspace rotation R_1, ..., R_T has temporal structure
(autocorrelation, mean-reversion, regime dependence) that enables prediction
beyond persistence. If the rotation is white noise, no geometric method
can help.

### 2.1 Rotation Diagnostics

Compute for each quarterly transition t=1,...,T−1:

**Rotation matrix:** R_t via Procrustes on U_{t-1}^T U_t.

**Principal angles:** θ_1(t), ..., θ_K(t) from SVD of U_{t-1}^T U_t.

**Geodesic distance:** d_t = ||θ(t)||_2 (overall rotation magnitude).

**Rotation axis stability:** Eigendecomposition of R_t gives rotation
planes. The dominant eigenvector identifies which subspace direction
rotates most. Track axis stability across quarters:
- Compute the dominant rotation plane for each t
- Measure the angle between consecutive dominant planes
- If this angle is small (< 20°), the rotation has a stable axis and
  the effective dimensionality drops to ~2 (rate + minor perturbation)

**Tangent vector:** Δ_t = Log_{U_{t-1}}(U_t), compressed to K principal
directions via SVD of Δ_t.

### 2.2 Temporal Structure Tests

For the scalar rotation magnitude d_t (T≈52 observations):
- Autocorrelation function ACF(d_t) at lags 1, 2, 4
- Ljung-Box test for white noise (H0: d_t is iid)
- Correlation with macro indicators (VIX, Fed funds rate, credit spreads)
- Mean and variance by economic regime (expansion, tightening, crisis)
- Variance ratio test (is d_t mean-reverting or persistent?)

For the K-dimensional principal angle vector θ(t):
- Per-angle autocorrelation: ACF(θ_k(t)) for k=1,...,K
- Cross-correlation between principal angles
- VAR(1) on θ(t): is the lag-1 coefficient matrix significantly nonzero?

For the rotation axis:
- Stability metric: mean cosine similarity between consecutive dominant
  rotation planes
- If stable (cosine > 0.8): effective 2D dynamics, predict rate only
- If unstable (cosine < 0.5): full SO(8) dynamics needed

### 2.3 Subspace Prediction Models

Compare predictions for U_{t+1|t} (all evaluated on the 52-quarter
trajectory, leave-one-out or rolling):

| # | Model | Description | Params |
|---|-------|-------------|--------|
| P0 | Persistence | Û_{t+1|t} = U_t | 0 |
| P1 | Last-rotation extrapolation | Û_{t+1|t} = U_t · R_t | 0 |
| P2 | Mean-rotation extrapolation | Û_{t+1|t} = U_t · R̄ | 28 |
| P3 | EWM-rotation extrapolation | Û_{t+1|t} = U_t · R̄_ewm(τ) | 29 |
| P4 | Tangent-space AR(1) | Δ̂_{t+1} = A·Δ_t → Exp on Gr | ≤K² |
| P5 | Principal-angle AR(1) | θ̂_k,t+1 = φ_k·θ_k,t | K |
| P6 | Phase-tracking (per conj. pair) | θ̂_k,t+1 = θ_k,t + ω̂_k | K/2 |
| P7 | Euclidean projector average (w=4Q) | P̂ = (1/w)Σ P_{t-s} | 1 |
| P8 | Karcher mean on Gr(K,N) (w=4Q) | Fréchet mean of recent U_t | 1 |
| P9 | HS-linear forecast on projectors | P̂_{t+1} = Σ_k c_k P_{t+1-k} | lags |

**On P7/P8:** The Euclidean average of rank-K projectors is generally NOT
rank-K; the Karcher mean IS a proper point on Gr(K,N). If the Karcher mean
beats Euclidean average, manifold-awareness matters. If the Euclidean average
suffices, simple HS-space operations are adequate.

**On P9:** HS-linear forecast treats P_t as a vector in the Hilbert space
of symmetric matrices and fits AR with Frobenius loss. Operationally, since
P_t = U_t U_t^T has rank K, this is equivalent to regression on the
vectorised U_t with HS-norm loss. Distinct from P4 (tangent-space AR, which
uses geodesic distance as loss).

### 2.4 Metrics

**Subspace prediction quality (all computed per quarter, averaged):**

- Projector Frobenius error: ||P_{t+1} − P̂_{t+1|t}||_F
- Chordal distance: d_chord = ||sin(θ)||_2 where θ are principal angles
  between predicted and actual subspace
- Principal-angle RMSE: √(mean(θ²))
- Subspace correlation: 1 − d²_chord/K (1=perfect, 0=orthogonal)

**Critical comparison:** P0 (persistence) is the null. Any geometric method
must beat persistence on at least one metric with paired-t CI excluding zero.

### 2.5 Kill Rule A

If no model P1–P9 beats persistence P0 on projector Frobenius error or
chordal distance with paired-t CI excluding zero over the 52-quarter
trajectory: **the rotation is structurally real but temporally
unpredictable.** Stop geometric forecasting.

The paper reports the rotation diagnostics (ACF, regime dependence, axis
stability) as structural findings but does not claim geometric forecasting
value.

### Time: 4h

---

## 3. Gate B — Does Predicted Geometry Improve Actor Forecasts?

### Goal

Test whether explicitly forecasting U_{t+1} and α_{t+1} separately
produces better actor-level predictions than predicting in the stale
frame U_t.

**Only run if Gate A passes** (at least one geometric predictor beats
persistence).

### 3.1 Parallel Transport of Amplitudes

When predicting in a new frame Û_{t+1|t}, the amplitude α̂ estimated in
the old frame U_t must be transported to the new frame. Naive re-projection
(α_new = Û^T · U · α̂) mixes geometry and amplitude. The correct operation
is parallel transport along the geodesic from U_t to Û_{t+1|t}:

    α̂_transported = Γ(U_t → Û_{t+1|t}) · α̂_{t+1|t}

For small rotations (~26°), this is approximately:

    Γ ≈ R_{t+1|t}^T    (inverse of the predicted rotation)

The predicted-frame reconstruction is:

    ŷ^{geo}_{t+1} = μ̂ + Û_{t+1|t} · R_{t+1|t}^T · F · α_{t|t}

### 3.2 Model Comparison

All models on the 93-actor panel, same rolling protocol:

| # | Model | Geometry | Amplitude | Frame |
|---|-------|----------|-----------|-------|
| S0 | Pooled+FE only | — | — | — |
| S1 | Current best (6.1 C1) | Stale U_t | diag(Ã)/full Ã | Stale |
| S2 | Geo-aware (mean R) | U_t · R̄ | same as S1 | Predicted |
| S3 | Geo-aware (last R) | U_t · R_t | same as S1 | Predicted |
| S4 | Geo-aware (best Gate A) | best Û_{t+1|t} | same as S1 | Predicted |
| S5 | Full separation | best Û_{t+1|t} | phase-amplitude (Gate C) | Predicted |
| R1 | Ridge (reference) | implicit | N×N | N/A |
| R2 | PCA+diag (reference) | stale PCA | diag AR | Stale |

**Primary endpoint:** S4 vs S1 — geometry-aware vs stale frame, same
amplitude model. This isolates the geometric prediction value.

**Critical comparison:** S4 vs R1 — does geometry-aware spectral prediction
break through the Ridge ceiling of 0.632?

### 3.3 Geometry-vs-Amplitude Attribution Diagnostic

Decompose the R² difference between S4 and S1 into three components:

    ΔR²_total = ΔR²_geometry + ΔR²_amplitude + ΔR²_interaction

Operationally:

- **ΔR²_geometry** = R²(predicted U, zero α correction) − R²(stale U, zero α)
  How much does the better frame help, holding amplitudes at their
  stale-frame predicted values?

- **ΔR²_amplitude** = R²(stale U, best α prediction) − R²(stale U, baseline α)
  How much does amplitude prediction help, holding frame fixed?

- **ΔR²_interaction** = ΔR²_total − ΔR²_geometry − ΔR²_amplitude
  Is there synergy between better geometry and better amplitudes?

If ΔR²_geometry ≈ 0 but ΔR²_amplitude > 0: the frame doesn't matter.
If ΔR²_geometry > 0: genuine geometric forecasting value exists.
If ΔR²_interaction > 0: geometry and amplitude forecasting are synergistic
— the predicted frame specifically helps the amplitude prediction, not
just the reconstruction.

### 3.4 Two-Stage Augmentation with Geometry

Apply geometric prediction within the two-stage architecture:

    Stage 1: ŷ^{pool} = pooled AR(1)+FE (unchanged)
    Stage 2: ŷ^{geo-resid} = Û^{resid}_{t+1|t} · α̂^{resid}_{t+1|t}

where the residual-stage geometry is predicted separately from the
residual-stage amplitudes.

### 3.5 Kill Rule B

If S4 ≤ S1 + 0.005 on the 93-actor panel: **the rotation is predictable
(Gate A passed) but not useful for actor-level forecasting.**

This means the actor-level conditional mean is insensitive to frame choice
— the ~26° rotation per quarter is small relative to amplitude variation,
so stale-frame prediction is adequate.

### Time: 5h

---

## 4. Gate C — Phase-Amplitude Decomposition

### Goal

Test whether decomposing modal dynamics into slowly-varying amplitude and
steadily-advancing phase improves prediction within the (possibly predicted)
subspace frame.

**Run if Gate A passes.** Can run regardless of Gate B outcome — phase-
amplitude decomposition may improve amplitude prediction even in the stale
frame.

### 4.1 Analytic Signal Construction

For each conjugate-pair mode (modes come in pairs from DMD):

1. Extract the real modal amplitude series α_k(1), ..., α_k(T)
2. Compute the analytic signal: z_k(t) = α_k(t) + i·H[α_k](t)
   where H is the Hilbert transform
3. Decompose: A_k(t) = |z_k(t)|, φ_k(t) = unwrap(arg(z_k(t)))

For real-eigenvalue modes (non-oscillatory), skip phase decomposition
and use standard AR.

### 4.2 Phase-Amplitude Prediction

**Amplitude prediction:**
    A_k(t+1) = ρ_A · A_k(t)        (simple AR(1) on the envelope)

**Phase prediction:**
    φ_k(t+1) = φ_k(t) + ω̂_k       (constant angular velocity)

where ω̂_k is estimated from the mean phase advance:
    ω̂_k = mean(φ_k(t) − φ_k(t−1))

**Reconstruction:**
    α̂_k(t+1) = Â_k(t+1) · cos(φ̂_k(t+1))

### 4.3 Phase-Locking Analysis (Inter-Mode Phase Coupling)

For each pair of conjugate modes (i, j), compute the phase coherence:

    C_{ij} = |mean(e^{i(φ_i(t) − φ_j(t))})|

C_{ij} = 1 means perfect phase-locking (modes rotate in lockstep).
C_{ij} = 0 means independent phases.

**If any C_{ij} > 0.5:** Strong phase-locking exists. Test coupled
phase prediction via the Kuramoto model:

    φ̂_i(t+1) = φ_i(t) + ω̂_i + Σ_j γ_{ij} · sin(φ_j(t) − φ_i(t))

where γ_{ij} is the coupling strength, estimated from the phase trajectory.

**Interpretation:** If DMD modes behave as coupled oscillators with
sector-interpretable phase-locking (e.g., tech rotation phase-locked to
energy rotation with a lag), this connects spectral investment dynamics
to oscillator theory — a genuinely novel finding regardless of R².

**Diagnostic even if R² doesn't improve:** Report the phase coherence
matrix C as a structural finding. Which modes are phase-locked? Do the
phase-locked pairs correspond to economically related sectors?

### 4.4 Why This Is Genuinely Different from Anything Tested

The mapping (A, φ) → α = A·cos(φ) is **nonlinear**. Linear methods
(DMD, PCA, Ridge) model α_{t+1} = f·α_t. But if the true dynamics are
"amplitude persists, phase advances at constant rate," the linear model
must simultaneously track the cosine oscillation — which it does via
complex eigenvalues (λ = |λ|·e^{iω}). The problem: at T≈20, the complex
eigenvalues are poorly estimated, and the real-valued projection Re(Φ)
loses the phase information.

Phase-amplitude decomposition recovers this by working in the natural
coordinate system where the dynamics ARE linear (amplitude decays, phase
advances). The nonlinearity is confined to the reconstruction step.

### 4.5 Feasibility

Oscillatory modes with |θ| ≈ 45–90° per quarter have oscillation periods
of 4–8 quarters. At T=52 quarters over the full sample, we have 6–13
full cycles — sufficient for phase estimation. Within each rolling window
(T≈20), 2.5–5 cycles — marginal but feasible.

### 4.6 Model Comparison

| # | Model | Amplitude | Phase | Params per pair |
|---|-------|-----------|-------|----------------|
| C0 | Linear (current) | F·α | implicit in λ | K or K² |
| C1 | Phase-amplitude (independent) | AR(1) on |z| | constant ω | 3 |
| C2 | Phase-amplitude (adaptive ω) | AR(1) on |z| | AR(1) on dφ/dt | 4 |
| C3 | Kuramoto coupled | AR(1) on |z| | coupled ω + γ_{ij} | 3 + K_pairs |

Test C0–C2 unconditionally. Test C3 only if phase coherence C_{ij} > 0.5
for at least one pair.

### 4.7 Kill Rule C

If phase-amplitude prediction (C1/C2) ≤ linear prediction (C0) within
the same frame: **the oscillatory structure does not provide additional
forecastable information beyond what eigenvalue-based linear dynamics
capture.** The phase decomposition is descriptively interesting but not
predictively useful.

### Time: 4h

---

## 5. Gate D — Kernel Koopman in RKHS (conditional)

### Goal

Test whether nonlinear dynamics in the K-dimensional modal space —
specifically, the rotation interaction that is nonlinear in Cartesian
but linear in polar coordinates — can be captured by kernel methods.

**Only run if Gate C fails** (phase-amplitude doesn't help). If phase-
amplitude works, kernel methods are unnecessary — the nonlinearity has
already been captured by the correct coordinate transformation.

### 5.1 Kernel Extended DMD

**This is a lightweight probe on K=8 modal amplitudes, NOT a massive
RKHS model.** The kernel matrix is T×T ≈ 20×20 — perfectly feasible.

Apply EDMD with Gaussian RBF kernel to the modal amplitude trajectory:

    K_ij = exp(−||α_i − α_j||² / σ²)

Koopman operator in RKHS:

    K_op = G^{-1} A    where G_ij = K(α_i, α_j), A_ij = K(α_i, α'_j)

Kernel bandwidth σ selected by inner CV from {0.5, 1.0, 2.0} × median
pairwise distance.

### 5.2 Fair Comparison

Against kernel Ridge regression on the same K-dimensional modal inputs.
If kernel EDMD beats kernel Ridge → Koopman eigenfunction structure matters.
If kernel Ridge matches → generic nonlinearity, not Koopman-specific.

### 5.3 Kill Rule D

If kernel EDMD ≤ kernel Ridge AND both ≤ linear methods: no exploitable
nonlinear modal dynamics exist. The dynamics are genuinely linear in
modal space.

### Time: 3h

---

## 6. Gate E — Graph-Informed Geometry (conditional)

**Only if Gates A+B succeed.** Use graph structure as a prior on the
subspace evolution, not as a standalone prediction method.

### 6.1 Sector-Constrained Rotation

If rotation primarily involves sector-rotation patterns (from the 6.1
mode interpretation: tech/healthcare vs financials), constrain the
predicted rotation to respect sector structure:

    R̂_{t+1|t} ∈ {R ∈ SO(K) : R preserves sector block structure}

Reduces rotation degrees of freedom from 28 (full SO(8)) to the number
of inter-sector rotation parameters.

### 6.2 Laplacian-Smoothed Subspace

Use a sector or supply-chain graph Laplacian L to regularise the predicted
subspace: columns of Û_{t+1|t} should be smooth with respect to L.

**Use simple fixed graphs only:**
- GICS sector graph (actors in same sector connected)
- Institution/firm block graph (layer structure)
- Supply-chain graph (if already in BEA I/O data)

**Do NOT use:** deep graph embeddings, learned graph structures, or any
method requiring more data than T≈52.

### 6.3 Kill Rule E

If graph regularisation does not improve subspace prediction over
unconstrained geometric models: the rotation is not sector-structured,
or the graph is too crude to capture the structure.

### Time: 4h

---

## 7. Execution Plan

### Phase 1: Rotation Diagnostics + Gate A (4h)

| Step | Time | What it answers |
|------|------|----------------|
| Compute R_t, θ_t, d_t for t=1,...,52 | 1h | Raw rotation data |
| Temporal structure (ACF, Ljung-Box, axis stability, regime dependence) | 1h | Is rotation predictable? |
| Subspace prediction models P0–P9 | 2h | Can we beat persistence? |

**Gate A decision:**
- If any P1–P9 beats P0 with CI excluding zero → proceed to Gate B
- If all null → rotation is unpredictable. Report diagnostics, stop.

### Phase 2: Actor-Level Reconstruction + Gate B (5h, conditional)

| Step | Time | What it answers |
|------|------|----------------|
| Geometry-aware reconstruction S1–S4 | 2h | Does predicted geometry help? |
| Geometry-vs-amplitude attribution | 1h | Is the gain from frame or amplitude? |
| Two-stage augmentation with geometry | 2h | Does it break the 0.630 ceiling? |

**Gate B decision:**
- S4 > S1 + 0.005 → proceed to Gates C and E
- S4 ≤ S1 + 0.005 → rotation predictable but not useful for this target.
  Still proceed to Gate C (phase-amplitude may help amplitude prediction
  even in stale frame).

### Phase 3: Phase-Amplitude + Phase-Locking + Gate C (4h)

| Step | Time | What it answers |
|------|------|----------------|
| Hilbert transform + phase extraction | 1h | Phase/amplitude decomposition |
| Per-mode phase-amplitude prediction | 1h | Does separation improve prediction? |
| Phase coherence matrix C_{ij} | 1h | Are modes phase-locked? |
| Kuramoto coupled prediction (if warranted) | 1h | Does coupling improve prediction? |

**Gate C decision:**
- Phase-amplitude > linear → major finding, proceed to full system
- Phase coherence C_{ij} > 0.5 for any pair → structural finding even
  if R² doesn't improve
- Both null → try Gate D (kernel) as last resort

### Phase 4: Kernel Koopman + Gate D (3h, conditional)

Only if Gate C fails.

### Phase 5: Graph-Informed + Gate E (4h, conditional)

Only if Gates A+B succeed.

### Total time estimates

| Path | Phases | Time |
|------|--------|------|
| Gate A null (rotation unpredictable) | Phase 1 only | 4h |
| Gate A pass, B null, C null | Phases 1–3 | 13h |
| Gate A pass, B pass, C pass | Phases 1–3, 5 | 17h |
| Everything passes | Phases 1–5 | 20h |
| Most likely: A pass, B borderline, C interesting | Phases 1–3 | 13h |

---

## 8. Success Criteria

### BRONZE
Gate A complete. Clean characterisation of rotation dynamics: ACF, regime
dependence, axis stability, predictability tests, Karcher mean vs
persistence. These are novel structural findings regardless of forecasting
value: "the cross-sectional investment basis rotates with [predictable /
regime-dependent / stable-axis] temporal dynamics."

### SILVER
Gate A passes AND Gate B shows geometry-aware prediction exceeds the 0.630
ceiling (S4 > R1). OR Gate C shows phase-amplitude decomposition improves
amplitude prediction even in the stale frame. Either would be the first
evidence of forecasting value from the geometric/oscillatory structure.

### GOLD
Silver plus phase coherence analysis reveals interpretable phase-locking
between sector-rotation modes (C_{ij} > 0.5 for economically related
pairs). The paper reports: "Investment sector rotation patterns exhibit
oscillator-like dynamics with measurable phase-locking between
tech/healthcare and financial rotation cycles."

### PLATINUM
Full geometry+phase system generalises across panels. Paper claims:
"Cross-sectional investment dynamics are best understood as amplitude
modulation on a rotating spectral basis. Forecasting requires tracking
both the frame evolution and the signal within the frame."

### HONEST COMPLETION
Gate A passes (rotation is partially predictable) but neither Gate B
nor Gate C improves actor-level forecasting. Paper reports:

> We characterise the evolution of the cross-sectional spectral basis
> as a smooth trajectory on the Grassmannian with [autocorrelated /
> regime-dependent] rotation dynamics. While the subspace evolution is
> partially predictable, explicit geometric prediction does not improve
> actor-level forecasts at the quarterly horizon, suggesting that
> amplitude variation dominates frame variation for this prediction target.

This is publishable as a structural/geometric contribution alongside
the two-stage augmentation result from 6.1.

---

## 9. Falsification Commitment

**Gate A is diagnostic and almost certainly passes.** The rotation at
25.8° ± 6.5° has a coefficient of variation of 0.25 — not white noise.
But passing Gate A does not imply practical value.

**Gate B is the decisive actor-level test.** If geometry-aware prediction
does not beat stale-frame prediction by ≥0.005, the geometric approach is
dead for point forecasting. The geometry-vs-amplitude attribution
diagnostic will reveal exactly why: either the frame correction is too
small relative to amplitude noise, or the predicted frame is not accurate
enough.

**Gate C is the deepest theoretical probe.** Phase-amplitude decomposition
tests genuine nonlinearity. If it fails, the dynamics are truly linear in
modal space and the rotation is fully captured by complex eigenvalues.

**If all gates produce nulls:** the paper has a complete story. The
iteration history (6.0→6.1→6.2→6.3) is comprehensive:
- 6.0: Standalone spectral fails (panel choice)
- 6.1: Transition repair + augmentation works (architecture)
- 6.2: DMD ≈ PCA ≈ Ridge (method equivalence)
- 6.3: Rotation is real and [predictable/unpredictable] but not useful
  for actor-level point forecasting (geometric structure)

The 0.630 ceiling is a data property — the amount of cross-sectional
predictability in quarterly US investment panels — not a methodological
limitation.

---

## 10. What NOT To Do

- Do not reopen actor-level method horse races (settled in 6.2)
- Do not reopen Kim filter, spectral Q/R (settled in 6.1)
- Do not reopen standalone SMIM (settled in 6.0)
- Do not use deep graph neural networks (T≈52 is insufficient)
- Do not use methods requiring T > 100 (Takens embedding needs ~10×
  attractor dimension; Grassmannian methods work at T≈52)
- Do not claim rotation prediction improves actor forecasting without
  demonstrating it in Gate B
- Do not use "massive RKHS models" — Gate D is a lightweight kernel
  probe on K=8 modal coordinates, not a high-dimensional kernel method

---

## 11. Connection to the Proposal

The research proposal (v5) envisioned several of these ideas:

- **Phase-transition dynamics** (§5.6): Ginzburg-Landau order parameter
  from spectral modes — rotation rate is itself an order parameter
- **Information geometry** (§6.5): Fisher information on the parameter
  manifold — Grassmannian prediction is Fisher-geometric on subspace manifold
- **Joint time-vertex processing** (§5.4): The fiber bundle decomposition
  is the state-space analogue

What the proposal did NOT envision: that the primary forecasting challenge
would be tracking the FRAME rather than the SIGNAL within the frame.
This reframing — from "improve the signal model" to "track the evolving
coordinate system" — is the key insight of 6.3 and would be novel in the
investment dynamics literature.

---

## 12. Files

| File | Role |
|------|------|
| `scripts/smim/run_iter6_3_gate_a.py` | Rotation diagnostics + subspace prediction |
| `scripts/smim/run_iter6_3_gate_b.py` | Geometry-aware actor reconstruction |
| `scripts/smim/run_iter6_3_gate_c.py` | Phase-amplitude + phase-locking analysis |
| `scripts/smim/run_iter6_3_gate_d.py` | Kernel Koopman (conditional) |
| `scripts/smim/run_iter6_3_gate_e.py` | Graph-informed geometry (conditional) |
| `src/.../smim/geometry/grassmannian.py` | Log, Exp, parallel transport, Karcher mean |
| `src/.../smim/geometry/rotation.py` | Procrustes extraction, SO(K) operations |
| `src/.../smim/geometry/phase_amplitude.py` | Hilbert transform, phase tracking, coherence |
| `results/metrics/iter6_3_*.parquet` | Per-gate results |
| `docs/smim/ITERATION_6_3_PLAN.md` | This file |
| `docs/smim/ITERATION_6_3_DECISION.md` | Decision memo (after execution) |