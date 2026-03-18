# SMIM Component Acceptance Test Plan

## Purpose

Before running ANY experiment, we must verify that every mathematical primitive
in the pipeline produces correct results. This is not integration testing — this
is **black-box verification** of each transformation against known ground truth.

A component passes acceptance only when ALL tests in its section pass. A single
failure blocks the entire experiment programme until fixed and re-verified.

## Important

Do not feet the test code to the existing implementation, rather make sure that the current implementation is correct.

---

## Verification Strategy

Each component is tested at four levels:

| Level | What it verifies | How |
|-------|-----------------|-----|
| **Analytical** | Exact correctness on problems with known closed-form solutions | Compare output against hand-computed or textbook answers to machine precision |
| **Invariant** | Mathematical properties that must hold regardless of input | Check algebraic identities, conservation laws, symmetries, bounds |
| **Reference** | Agreement with trusted external implementations | Compare against scipy, statsmodels, ripser, MATLAB, R — established libraries with published test suites |
| **Adversarial** | Correct behaviour on pathological inputs | Singular matrices, unit roots, zero variance, identical rows, extreme sparsity, numerical edge cases |

**Tolerance conventions:**
- "Exact" means `np.allclose(a, b, atol=1e-12, rtol=1e-12)` (machine precision)
- "Tight" means `np.allclose(a, b, atol=1e-8, rtol=1e-8)`
- "Loose" means `np.allclose(a, b, atol=1e-4, rtol=1e-4)` (acceptable for iterative methods)
- "Directional" means sign and order match, magnitude within 20%

---

## 1. Graph Construction Primitives

### 1.1 Granger Edge Estimation

**A-GR-1 (Analytical — known VAR(1) system):**
Generate data from a known VAR(1): y_t = A y_{t-1} + ε, where A is a 4×4 matrix
with A[1,0] = 0.6 (actor 0 causes actor 1) and A[2,3] = 0.5 (actor 3 causes actor 2),
all other off-diagonal = 0. T = 2000.
- **Pass**: estimator detects edges (0→1) and (3→2) with p < 0.01
- **Pass**: estimator does NOT detect edges (1→0) or (2→3) at p < 0.05
- **Pass**: estimated F-statistics for true edges are > 10× those for false edges

**A-GR-2 (Analytical — no causality):**
Generate 4 independent random walks (no causal structure). T = 2000.
- **Pass**: no edge has p < 0.01 (no false positives at this threshold)
- **Pass**: edge density < 10% at p < 0.05 threshold (expected ~5% by chance)

**I-GR-1 (Invariant — directionality):**
For any estimated adjacency A, if A[j,i] > 0 (i causes j), verify that the
Granger test was run in the direction i → j, not j → i.
- **Pass**: manually trace the test direction for 10 random edges

**I-GR-2 (Invariant — point-in-time):**
Feed data with dates up to 2020-Q4. Set date_range.end = 2019-Q4.
- **Pass**: estimator uses only data up to 2019-Q4 (verify by checking the
  effective sample size equals the expected truncated length)

**D-GR-1 (Adversarial — collinear series):**
Two actors with y2 = 2*y1 + small noise (near-perfect collinearity).
- **Pass**: estimator handles without crash; VARs should still be estimable

**D-GR-2 (Adversarial — constant series):**
One actor with constant y = 5.0 (zero variance).
- **Pass**: no edges to/from this actor; no division by zero or NaN

### 1.2 Narrative Edge Estimation

**A-NE-1 (Analytical — known similarity):**
Three actors. Actor 1 documents: ["energy policy reform", "renewable energy targets"].
Actor 2 documents: ["energy sector regulation", "green energy policy"].
Actor 3 documents: ["machine learning in healthcare", "neural network drug discovery"].
- **Pass**: similarity(1,2) > 0.3; similarity(1,3) < 0.1; similarity(2,3) < 0.1

**I-NE-1 (Invariant — symmetry of undirected similarity):**
For cosine similarity (before temporal directionality), verify sim(i,j) = sim(j,i).
- **Pass**: |sim(i,j) - sim(j,i)| < 1e-10 for all pairs

**I-NE-2 (Invariant — bounds):**
- **Pass**: all similarity values in [0, 1] for TF-IDF cosine
- **Pass**: all edge weights in A are ≥ 0

**D-NE-1 (Adversarial — empty document):**
One actor has no documents.
- **Pass**: no edges to/from this actor; no NaN; no crash

### 1.3 Aggregate Operator

**A-AO-1 (Analytical — linearity):**
A1 = [[0,1],[0,0]], A2 = [[0,0],[1,0]], weights ω1=0.3, ω2=0.7.
- **Pass**: result = [[0, 0.3], [0.7, 0]] (exact)

**I-AO-1 (Invariant — uniform weights):**
With R channels and uniform weights (ω_r = 1/R):
- **Pass**: aggregate = (A1 + A2 + ... + AR) / R (exact)

**I-AO-2 (Invariant — spectral energy):**
- **Pass**: spectral_energy(aggregate) ≤ Σ_r ω_r × spectral_energy(A_r)
  (sub-additivity from triangle inequality)

### 1.4 Sparsification

**A-SP-1 (Analytical — threshold sparsification):**
Dense matrix with known values. Threshold = 0.5.
- **Pass**: all entries < 0.5 are exactly zero; all entries ≥ 0.5 unchanged

**I-SP-1 (Invariant — density control):**
Apply L1 sparsification with target_density = 0.15.
- **Pass**: resulting density ≤ 0.15

**I-SP-2 (Invariant — spectral energy retention):**
- **Pass**: spectral_energy_retained() is in [0, 1]
- **Pass**: spectral_energy_retained() is monotonically non-decreasing as
  target_density increases from 0.01 to 1.0

### 1.5 Null Model Rewiring

**I-NM-1 (Invariant — degree preservation):**
For degree-preserving rewiring:
- **Pass**: in-degree per node is identical before and after (exact integer match)
- **Pass**: out-degree per node is identical before and after (exact integer match)
- **Pass**: total edge count unchanged

**I-NM-2 (Invariant — block preservation):**
For block-preserving rewiring with 4 layer blocks:
- **Pass**: no new cross-block edges created
- **Pass**: within-block degree sequence preserved

**I-NM-3 (Invariant — randomness):**
Two calls to degree_preserving_rewire with different seeds:
- **Pass**: results differ (not deterministic without seed)

**A-NM-1 (Statistical — rewiring distribution):**
Generate 1000 rewirings of a 20-node graph. Count how many times each possible
edge appears. For a uniform random rewiring conditional on degree sequence:
- **Pass**: no single edge appears in >50% of rewirings (distribution is spread)
- **Pass**: edge frequency distribution has entropy > 80% of maximum

---

## 2. Spectral Decomposition Primitives

### 2.1 Schur Decomposition

**A-SC-1 (Analytical — known eigenvalues):**
A = [[2, 1], [0, 3]]. Known eigenvalues: 2, 3.
- **Pass**: diagonal of T equals {2, 3} (or {3, 2}) to machine precision
- **Pass**: T is upper triangular: all entries below diagonal < 1e-12
- **Pass**: Q is unitary: ||Q^H Q - I|| < 1e-12
- **Pass**: reconstruction: ||QTQ^H - A|| < 1e-12

**A-SC-2 (Analytical — symmetric matrix reduces to eigendecomposition):**
A = [[4, 1], [1, 3]]. Known eigenvalues: (7±√5)/2.
- **Pass**: Schur eigenvalues match known eigenvalues to tight tolerance
- **Pass**: for symmetric input, T should be diagonal (Schur = eigendecomp)

**A-SC-3 (Analytical — 5×5 companion matrix):**
Companion matrix for polynomial x⁵ - 1. Known eigenvalues: 5th roots of unity.
- **Pass**: Schur eigenvalues match 5th roots of unity (modulus tight, angle tight)

**I-SC-1 (Invariant — unitarity of Q):**
Random 100×100 non-symmetric matrix.
- **Pass**: ||Q^H Q - I||_F < 1e-10
- **Pass**: ||Q Q^H - I||_F < 1e-10

**I-SC-2 (Invariant — triangularity of T):**
Random 100×100 matrix.
- **Pass**: max(|T[i,j]| for i > j) < 1e-10

**I-SC-3 (Invariant — reconstruction):**
Random 100×100 matrix.
- **Pass**: ||QTQ^H - A||_F / ||A||_F < 1e-10

**R-SC-1 (Reference — scipy agreement):**
Random 50×50 matrix. Compare our Schur output against scipy.linalg.schur directly.
- **Pass**: eigenvalues from T match scipy eigenvalues to machine precision
- (Note: Q and T themselves may differ by unitary transformation — eigenvalues are the invariant)

**D-SC-1 (Adversarial — near-defective matrix):**
A = [[1, 1], [0, 1+1e-10]] (nearly defective, eigenvalues nearly coincide).
- **Pass**: no crash; eigenvalues close to 1; condition number reported in metadata

**D-SC-2 (Adversarial — zero matrix):**
A = 0 (all zeros).
- **Pass**: all eigenvalues exactly 0; no NaN; no crash

### 2.2 Polar Decomposition

**A-PL-1 (Analytical — rotation matrix):**
A = [[cos θ, -sin θ], [sin θ, cos θ]] for θ = π/4.
- **Pass**: U = A (the matrix is already orthogonal)
- **Pass**: P = I (identity)
- **Pass**: U @ P = A (exact)

**A-PL-2 (Analytical — scaling matrix):**
A = [[3, 0], [0, 5]] (diagonal positive).
- **Pass**: U = I (identity)
- **Pass**: P = A = [[3,0],[0,5]]

**A-PL-3 (Analytical — rotation + scaling):**
A = [[3cos θ, -3sin θ], [5sin θ, 5cos θ]] — NOT a valid polar decomposition input
Use A = R @ S where R is known rotation, S is known symmetric PSD.
- **Pass**: recovered U ≈ R, recovered P ≈ S

**I-PL-1 (Invariant — U is orthogonal):**
Random 100×100 matrix.
- **Pass**: ||U^T U - I||_F < 1e-10

**I-PL-2 (Invariant — P is symmetric PSD):**
Random 100×100 matrix.
- **Pass**: ||P - P^T||_F < 1e-12
- **Pass**: all eigenvalues of P are ≥ -1e-10

**I-PL-3 (Invariant — reconstruction):**
Random 100×100 matrix.
- **Pass**: ||UP - A||_F / ||A||_F < 1e-10

**R-PL-1 (Reference — scipy agreement):**
Compare against scipy.linalg.polar.
- **Pass**: ||U_ours - U_scipy||_F < 1e-10; ||P_ours - P_scipy||_F < 1e-10

**D-PL-1 (Adversarial — singular matrix):**
A = [[1, 0], [0, 0]] (rank deficient).
- **Pass**: decomposition succeeds; P has one zero eigenvalue; U is still orthogonal

### 2.3 Hermitian Dilation

**A-HD-1 (Analytical — known SVD):**
A = [[3, 0], [0, 5]]. Known SVD: singular values {3, 5}.
Hermitian dilation H = [[0, A], [A^T, 0]].
- **Pass**: eigenvalues of H are {-5, -3, 3, 5} (the ±σ_k pairs)
- **Pass**: left singular vectors recovered from H eigenvectors match U_L from SVD
- **Pass**: right singular vectors recovered match U_R from SVD

**A-HD-2 (Analytical — rank-1 matrix):**
A = [[1], [2], [3]] @ [[4, 5]] = [[4,5],[8,10],[12,15]]. Single singular value σ = √(1²+2²+3²)×√(4²+5²).
- **Pass**: H has eigenvalues ±σ (plus zeros)
- **Pass**: non-zero eigenvalues match the one non-zero singular value

**I-HD-1 (Invariant — eigenvalue pairing):**
Random 50×50 matrix. Form H (100×100).
- **Pass**: for every eigenvalue λ of H, -λ is also an eigenvalue
- **Pass**: eigenvalues of H sorted by magnitude match [σ_1, σ_1, σ_2, σ_2, ...] (each doubled as ±)

**I-HD-2 (Invariant — H is Hermitian):**
- **Pass**: ||H - H^T||_F < 1e-12

**I-HD-3 (Invariant — singular value recovery):**
Random 50×50 A. Compute SVD via np.linalg.svd. Compute eigenvalues of H.
- **Pass**: sorted positive eigenvalues of H match sorted singular values of A to tight tolerance

**R-HD-1 (Reference — numpy SVD agreement):**
Random 50×50 A.
- **Pass**: singular values recovered from H match np.linalg.svd(A) to machine precision

**D-HD-1 (Adversarial — rectangular matrix):**
A is 30×50 (non-square). H is 80×80.
- **Pass**: works correctly; H has 30 non-zero eigenvalue pairs + 20 zeros

### 2.4 Directed Variation Basis

**I-DV-1 (Invariant — orthonormality):**
Output basis U from any input.
- **Pass**: ||U^T U - I||_F < 1e-8 (optimisation tolerance)

**I-DV-2 (Invariant — DV monotonicity):**
Compute DV(u_k) for each basis vector u_k.
- **Pass**: DV(u_1) ≤ DV(u_2) ≤ ... ≤ DV(u_K) (monotonically non-decreasing)

**I-DV-3 (Invariant — DV non-negativity):**
- **Pass**: DV(u_k) ≥ 0 for all k (by definition, it's a sum of squared terms)

**R-DV-1 (Reference — agrees with standard eigenbasis on symmetric input):**
If A is symmetric, directed variation reduces to standard graph total variation.
The DV-optimised basis should approximate the Laplacian eigenbasis.
- **Pass**: on symmetric A, basis vectors have correlation > 0.9 with Laplacian eigenvectors
  (up to sign and ordering)

### 2.5 DMD (Exact)

**A-DMD-1 (Analytical — known linear system):**
Generate data from y_{k+1} = A y_k + ε where A = [[0.9, 0.1], [0, 0.8]], ε ~ N(0, 0.01).
Known eigenvalues: 0.9, 0.8. T = 50 (both modes decay to noise floor after ~20 steps;
using T > 50 makes the smaller eigenvalue unrecoverable).
- **Pass**: DMD eigenvalue magnitudes within 15% of {0.9, 0.8}
- **Pass**: DMD modes span the same subspace as eigenvectors of A (angle < 10°)

**A-DMD-2 (Analytical — oscillatory system):**
A has eigenvalues 0.95 × exp(±iπ/6) (damped oscillation at period 12).
T = 500.
- **Pass**: DMD eigenvalue magnitudes within 5% of 0.95
- **Pass**: DMD eigenvalue angles within 5% of ±π/6
- **Pass**: reconstructed oscillation period within 10% of 12

**A-DMD-3 (Analytical — reconstruction convergence):**
Known 5-mode system. Retain k = 1, 2, 3, 4, 5 modes.
- **Pass**: reconstruction error ||y - y_reconstructed|| decreases monotonically with k
- **Pass**: at k=5, reconstruction error < 1% of signal energy

**I-DMD-1 (Invariant — reconstruction fidelity):**
Random 20-dimensional **symmetric** stable system (spectral radius 0.9), T=200.
Symmetric A is used so all eigenvalues are real and DMD modes (stored as .real) contain
no truncation error, making the reconstruction formula valid.
- **Pass**: ||Y - Φ diag(λ^k) B||_F / ||Y||_F < 0.1 with all modes retained

**R-DMD-1 (Reference — PyDMD agreement):**
Same data through our implementation and PyDMD (`pydmd` is a required dev dependency).
- **Pass**: eigenvalue magnitudes agree to tight tolerance
- **Pass**: modes span same subspace (principal angles < 5°)

**D-DMD-1 (Adversarial — short time series):**
T = 5 observations, N = 20 dimensions (underdetermined).
- **Pass**: graceful handling (truncated SVD, reduced rank, no crash)

**D-DMD-2 (Adversarial — constant signal):**
All snapshots identical: y_k = y_0 for all k.
- **Pass**: single mode with eigenvalue 1.0; no NaN; no crash

---

## 3. Mode Selection Primitives

### 3.1 MDL Criterion

**A-MDL-1 (Analytical — known rank):**
Generate signal: y = U_3 @ alpha + noise, where U_3 has rank 3 and noise is small.
Offer K = 1, 2, 3, 4, 5, 10 modes.
- **Pass**: MDL selects K* = 3

**A-MDL-2 (Analytical — pure noise):**
Generate y as pure iid Gaussian noise.
- **Pass**: MDL selects K* = 0 or K* = 1 (minimum)

**I-MDL-1 (Invariant — monotonicity of description length):**
For fixed data, compute DL(k) for k = 1, ..., 20.
- **Pass**: L(data|model) decreases with k (more modes = better fit)
- **Pass**: L(model) increases with k (more modes = higher complexity)
- **Pass**: total DL has a minimum (not monotonically decreasing)

### 3.2 Lempel-Ziv Compressibility

**A-LZ-1 (Analytical — constant sequence):**
Sequence: [1, 1, 1, ..., 1] (length 1000).
- **Pass**: LZ complexity = 1 (or 2 with boundary effects); ρ > 0.99

**A-LZ-2 (Analytical — periodic sequence):**
Sequence: [1, 2, 3, 1, 2, 3, ...] (length 1000).
- **Pass**: LZ complexity = O(log n); ρ > 0.5

**A-LZ-3 (Analytical — random sequence):**
Sequence: iid uniform integers from {0, ..., 255} (length 1000).
- **Pass**: ρ < 0.15 (nearly incompressible)

**I-LZ-1 (Invariant — bounds):**
- **Pass**: ρ ∈ [0, 1] for any input
- **Pass**: ρ(constant) > ρ(periodic) > ρ(random) on same-length sequences

### 3.3 RG Relevance

**A-RG-1 (Analytical — global mode):**
Mode with uniform loadings across all layers: u = [1,1,1,...,1]/√N.
- **Pass**: RG relevance score is high (mode persists under coarse-graining)

**A-RG-2 (Analytical — local mode):**
Mode with loadings only on Layer 3 actors: u = [0,...,0,1,...,1,0,...,0]/√(N_3).
- **Pass**: RG relevance score is low (mode vanishes under coarse-graining to Layer 0-1)

---

## 4. State-Space Filtering Primitives

### 4.1 Kalman Filter

**A-KF-1 (Analytical — steady-state gain):**
Known F, Q, R, H for a stable 2D system. Compute the steady-state Kalman gain
by solving the Discrete Algebraic Riccati Equation (DARE):
P = F P F^T + Q - F P H^T (H P H^T + R)^{-1} H P F^T
K = P H^T (H P H^T + R)^{-1}
Run filter for 500 steps.
- **Pass**: filter gain K_t converges to K_DARE to tight tolerance by step 50
- **Pass**: filtered covariance P_t converges to P_DARE to tight tolerance

**A-KF-2 (Analytical — perfect observation):**
R = 0 (no measurement noise), H = I (full observation).
- **Pass**: filtered state = observation exactly (after first step)
- **Pass**: filtered covariance = 0

**A-KF-3 (Analytical — known trajectory):**
Generate y_t from known F, Q, R with known true state trajectory x_t.
T = 1000.
- **Pass**: mean(||x_filtered - x_true||) < mean(||y_observed - x_true||)
  (filter improves over raw observations)
- **Pass**: filtered state within 2σ of true state for >95% of time steps

**I-KF-1 (Invariant — covariance positive definiteness):**
Run filter on random system for 500 steps.
- **Pass**: all eigenvalues of P_t|t > 0 at every step
- **Pass**: all eigenvalues of P_t|t-1 > 0 at every step

**I-KF-2 (Invariant — log-likelihood computation):**
- **Pass**: log-likelihood = Σ_t log p(y_t | y_{1:t-1}) computed via innovation
  sequence matches direct computation using multivariate normal logpdf
  (within 1e-6 relative tolerance)

**I-KF-3 (Invariant — innovation whiteness):**
For a correctly specified model, the innovation sequence ν_t = y_t - H x_{t|t-1}
should be approximately white noise.
- **Pass**: Ljung-Box test on innovations fails to reject white noise (p > 0.05)
- **Pass**: autocorrelation of innovations at lags 1-10 all < 0.1

**R-KF-1 (Reference — statsmodels agreement):**
Same model and data through our Kalman filter and statsmodels.tsa.statespace.
- **Pass**: filtered states agree to tight tolerance
- **Pass**: log-likelihoods agree to tight tolerance

**R-KF-2 (Reference — scipy DARE agreement):**
Steady-state gain from our filter vs scipy.linalg.solve_discrete_are.
- **Pass**: agree to machine precision

**D-KF-1 (Adversarial — unstable system):**
F has spectral radius > 1 (eigenvalue 1.1).
- **Pass**: filter runs without overflow for at least 100 steps
  (covariance may grow, but should not produce NaN/Inf)

**D-KF-2 (Adversarial — near-singular R):**
R = diag(1e-12, 1e-12) (near-perfect observation).
- **Pass**: no crash; filtered state very close to observation

### 4.2 EM Estimation (Single Regime)

**A-EM-1 (Analytical — recover known parameters):**
Generate data from known F, Q, R (stable 2D system, T=2000).
Run EM from random initialisation.
- **Pass**: estimated F within 10% of true F (element-wise)
- **Pass**: estimated Q within 20% of true Q
- **Pass**: estimated R within 20% of true R

**I-EM-1 (Invariant — monotone log-likelihood):**
Run EM for 200 iterations.
- **Pass**: log-likelihood is monotonically NON-DECREASING across ALL iterations
  (any decrease, even by 1e-10, is a bug)

**I-EM-2 (Invariant — convergence):**
- **Pass**: |LL(iter) - LL(iter-1)| < tol for some iter < max_iter
  (EM actually converges, doesn't just hit the iteration limit)

**D-EM-1 (Adversarial — bad initialisation):**
Initialise F = 2I (spectral radius 2, unstable), Q = 0.001I (too small).
- **Pass**: EM recovers reasonable parameters (F stable, Q appropriate)
  or terminates gracefully with a warning

### 4.3 Kim Filter (Regime-Switching)

**A-KIM-1 (Analytical — known regime sequence):**
Generate 2-regime data: regime 1 for t=1..50 (F1, Q1), regime 2 for t=51..100 (F2, Q2).
F1 and F2 are very different (e.g., F1 = 0.9I, F2 = -0.5I). T = 100.
- **Pass**: smoothed regime probability P(z_t=1|Y_T) > 0.8 for t ∈ [10, 45]
- **Pass**: smoothed regime probability P(z_t=2|Y_T) > 0.8 for t ∈ [55, 95]
- **Pass**: transition detected within ±5 steps of true switch at t=50

**A-KIM-2 (Analytical — single-regime equivalence):**
Generate single-regime data (M=1). Run Kim filter with M=1.
- **Pass**: Kim filter output matches plain Kalman filter output to tight tolerance
  (they should be identical for M=1)

**I-KIM-1 (Invariant — regime probability normalisation):**
At every time step t:
- **Pass**: Σ_j P(z_t=j|Y_t) = 1.0 to machine precision
- **Pass**: P(z_t=j|Y_t) ≥ 0 for all j
- **Pass**: Σ_j P(z_t=j|Y_T) = 1.0 (smoothed probabilities too)

**I-KIM-2 (Invariant — moment-matching collapse preserves moments):**
After the M²→M collapse step at each time step:
- **Pass**: collapsed mean = weighted average of branch means (exact)
- **Pass**: collapsed covariance = weighted average of branch covariances
  + weighted average of squared mean deviations (exact, by definition)

**I-KIM-3 (Invariant — transition matrix properties):**
Estimated transition matrix P:
- **Pass**: all entries in [0, 1]
- **Pass**: each row sums to 1.0 to machine precision
- **Pass**: no row is all-zeros (every regime has a successor)

**I-KIM-4 (Invariant — EM log-likelihood monotonicity):**
- **Pass**: log-likelihood is monotonically non-decreasing across EM iterations
  (same as single-regime EM, but more critical here)

**R-KIM-1 (Reference — degenerates to Hamilton filter):**
For a model with observed states (H=I) and no state dynamics (F=I),
the Kim filter should produce regime probabilities matching the Hamilton (1989)
filter. Implement Hamilton filter independently and compare.
- **Pass**: regime probabilities agree to tight tolerance

**D-KIM-1 (Adversarial — degenerate regimes):**
Initialise with F1 = F2 (identical regime dynamics). Run EM.
- **Pass**: EM converges; either selects M*=1 or identifies that regimes are
  indistinguishable; does NOT produce NaN

**D-KIM-2 (Adversarial — rare regime):**
One regime active for only 5 of 100 time steps.
- **Pass**: filter does not crash; rare regime may have high parameter uncertainty
  but produces finite estimates

### 4.4 Observability Matrix

**A-OB-1 (Analytical — observable system):**
F = [[0.9, 0.1], [0, 0.8]], C = [[1, 0]]. Observability matrix:
O = [C; CF; CF²; CF³] — known to have rank 2 (fully observable).
- **Pass**: computed rank = 2; condition number < 100

**A-OB-2 (Analytical — unobservable system):**
F = [[0.9, 0], [0, 0.8]], C = [[1, 0]]. Second state is unobservable.
O has rank 1.
- **Pass**: computed rank = 1 (or effective rank 1 via singular value gap)
- **Pass**: condition number > 10^6 (or Inf)

**I-OB-1 (Invariant — dimensions):**
For K* modes and K* block rows:
- **Pass**: O has shape (N × K*, K*) where N is observation dimension

---

## 5. Phase Transition Primitives

### 5.1 Order Parameter Extraction

**I-OP-1 (Invariant — dimensionality):**
Input: alpha_filtered (T, K). Parameter d.
- **Pass**: output shape is (T, d) for d = 1, 2, 3

**I-OP-2 (Invariant — bounded):**
- **Pass**: all output values are finite; max |ψ| < 100 × max |α|

### 5.2 Ginzburg-Landau Landscape

**A-GL-1 (Analytical — double-well potential):**
Generate ψ_t from the stochastic gradient flow dψ = -(aψ + bψ³)dt + σdW
with a = -1, b = 1 (double well at ψ = ±1). Long run T = 10000.
- **Pass**: fitted a < 0 (negative, indicating bistability)
- **Pass**: fitted b > 0 (positive, for stability)
- **Pass**: fitted potential has minima near ±1

**A-GL-2 (Analytical — single-well potential):**
Same with a = 1, b = 1 (single well at ψ = 0).
- **Pass**: fitted a > 0; potential minimum near 0

**I-GL-1 (Invariant — potential at minima):**
At any fitted minimum ψ*: dF/dψ|_{ψ*} ≈ 0.
- **Pass**: gradient at fitted minima < 0.1 × max gradient anywhere

### 5.3 Criticality Index

**A-CI-1 (Analytical — approaching unit root):**
Generate AR(1) process with ρ_t increasing from 0.5 to 0.99 over T=400 steps.
Compute C_t.
- **Pass**: C_t is significantly higher in the last quarter than the first quarter
- **Pass**: Kendall τ correlation between C_t and true ρ_t is > 0.3

**A-CI-2 (Analytical — stationary process):**
Generate AR(1) with constant ρ = 0.5. T = 400.
- **Pass**: C_t ≈ 1.0 throughout (variance ratio ≈ 1, ACF ratio ≈ 1)
- **Pass**: std(C_t) < 0.3 (low variability when process is stationary)

**I-CI-1 (Invariant — non-negativity):**
- **Pass**: C_t ≥ 0 for all t (it's a product of variance and ACF ratios, both ≥ 0)

---

## 6. Emergence Diagnostic Primitives

### 6.1 PID (Partial Information Decomposition)

**A-PID-1 (Analytical — Gaussian MMI, redundant sources):**
X1 = X2 = Z, Y = Z + noise. (Two copies of the same source.)
- **Pass**: redundancy R > 0; R ≈ I(X1; Y)
- **Pass**: unique information U1 ≈ U2 ≈ 0
- **Pass**: synergy S ≈ 0
- **Pass**: R + U1 + U2 + S ≈ I(X1, X2; Y) (decomposition sums correctly)

**A-PID-2 (Analytical — Gaussian MMI, XOR-like synergy):**
X1, X2 independent Gaussian. Y = X1 × X2 + small noise (interaction term).
- **Pass**: synergy S > 0 (joint information not available from either alone)
- **Pass**: S is significantly larger than in A-PID-1

**A-PID-3 (Analytical — independent sources):**
X1, X2 independent. Y independent of both.
- **Pass**: R ≈ 0, U1 ≈ 0, U2 ≈ 0, S ≈ 0
- **Pass**: I(X1, X2; Y) ≈ 0

**I-PID-1 (Invariant — non-negativity):**
- **Pass**: R ≥ 0, U1 ≥ 0, U2 ≥ 0, S ≥ 0 (for MMI measure)

**I-PID-2 (Invariant — decomposition identity):**
- **Pass**: |R + U1 + U2 + S - I(X1, X2; Y)| < 1e-6 for every test case

**I-PID-3 (Invariant — synergy matrix symmetry):**
- **Pass**: |S_matrix[j,k] - S_matrix[k,j]| < 1e-10 for all j, k

### 6.2 Transfer Entropy

**A-TE-1 (Analytical — coupled AR, Gaussian equivalence):**
X_t = 0.8 X_{t-1} + ε_x. Y_t = 0.5 Y_{t-1} + 0.4 X_{t-1} + ε_y.
For Gaussian case, TE_{X→Y} = 0.5 × ln(Var(ε_Y) / Var(ε_{Y|X})).
Compute analytically, then via KSG estimator with T = 10000.
- **Pass**: KSG estimate within 20% of analytical value
- **Pass**: TE_{X→Y} > 0 (X causes Y)
- **Pass**: TE_{Y→X} < 0.5 × TE_{X→Y} (Y does not cause X, or much weaker)

**A-TE-2 (Analytical — independent processes):**
X, Y independent AR(1) processes. T = 10000.
- **Pass**: TE_{X→Y} < 0.05 (near zero)
- **Pass**: TE_{Y→X} < 0.05

**A-TE-3 (Analytical — Granger equivalence for Gaussian):**
Same coupled system as A-TE-1. Also run Granger causality test.
- **Pass**: TE and Granger F-test agree in direction (both detect X→Y)

**I-TE-1 (Invariant — non-negativity):**
- **Pass**: TE ≥ 0 for all pairs (TE is a conditional mutual information, non-negative by definition)

**I-TE-2 (Invariant — conditional reduces TE for mediated paths):**
Chain: X→Z→Y (X causes Z, Z causes Y, no direct X→Y link).
- **Pass**: TE_{X→Y} > 0 (unconditional)
- **Pass**: TE_{X→Y|Z} ≈ 0 (conditioning on mediator removes the effect)

**R-TE-1 (Reference — IDTxl or JIDT agreement):**
Same data through our implementation and IDTxl.
- **Pass**: TE values agree within 20% (KSG estimators have variance)

### 6.3 Persistent Homology / TDA

**A-TDA-1 (Analytical — circle):**
100 points uniformly sampled from a unit circle + small noise (σ=0.05).
- **Pass**: persistence diagram has exactly 1 long-lived H1 feature (one loop)
- **Pass**: the persistence (death - birth) of this H1 feature is > 0.5
- **Pass**: β_0 = 1 at large scale (one connected component)

**A-TDA-2 (Analytical — two clusters):**
50 points around (0,0), 50 points around (5,0), σ=0.3 each.
- **Pass**: at small ε: β_0 = 2 (two components)
- **Pass**: at large ε: β_0 = 1 (merged into one)
- **Pass**: no significant H1 features (no loops)

**A-TDA-3 (Analytical — sphere):**
200 points on unit sphere in R³ + small noise.
- **Pass**: one significant H2 feature (one void)
- **Pass**: β_0 = 1 at appropriate scale

**I-TDA-1 (Invariant — stability theorem):**
Add small perturbation (ε=0.01) to point cloud from A-TDA-1. Recompute diagram.
- **Pass**: bottleneck distance between original and perturbed diagrams < ε
  (stability theorem guarantee)

**I-TDA-2 (Invariant — topological complexity non-negativity):**
- **Pass**: T_t ≥ 0 for all windows

**I-TDA-3 (Invariant — Wasserstein distance properties):**
- **Pass**: W(D, D) = 0 (distance to self is zero)
- **Pass**: W(D1, D2) = W(D2, D1) (symmetry)
- **Pass**: W(D1, D2) ≥ 0 (non-negativity)

**R-TDA-1 (Reference — ripser agreement):**
Same point cloud through our implementation and raw ripser.
- **Pass**: persistence diagrams are identical (we should be wrapping ripser)

---

## 7. Benchmark / Gap Computation

### 7.1 Predictive Benchmark

**A-PB-1 (Analytical — gap equals innovation):**
For a correctly specified model, Δ^{pred}_{i,t} = y_{i,t} - ŷ_{i,t|t-1}
should equal the Kalman innovation ν_t (projected back to actor space).
- **Pass**: ||Δ^{pred} - U @ ν|| < 1e-8

**I-PB-1 (Invariant — benchmark label):**
- **Pass**: every GapResult has benchmark_class == BenchmarkClass.PREDICTIVE

**I-PB-2 (Invariant — gap + benchmark = observation):**
- **Pass**: gap + benchmark = observation for all (i, t), to machine precision

### 7.2 Modal Benchmark

**I-MB-1 (Invariant — attribution sums to total gap):**
- **Pass**: Σ_k modal_attribution[i, t, k] ≈ gap[i, t] for all (i, t)
  (within 1e-6, accounting for residual term)

**I-MB-2 (Invariant — benchmark label):**
- **Pass**: benchmark_class == BenchmarkClass.MODAL

### 7.3 Emergence-Aware Benchmark

**A-EB-1 (Analytical — zero emergence equals predictive):**
Set synergy_matrix = 0, criticality = 0, topo_complexity = 0.
- **Pass**: Δ^{em} = Δ^{pred} to tight tolerance

**A-EB-2 (Analytical — synergy correction direction):**
If S[j,k] > 0 and α_j × α_k > 0, the synergistic correction should increase the benchmark.
- **Pass**: benchmark^{em} > benchmark^{pred} for actors with positive mode interaction loadings

---

## 8. Pipeline-Level Sanity Checks

These test the full pipeline end-to-end on synthetic data with known ground truth.

**P-1 (Round-trip — known DGP):**
Generate a full synthetic system:
- 20 actors, 4 layers, 3 edge channels with known adjacency
- 5 true spectral modes
- 2 regimes with known transition matrix
- Known F, Q, R per regime
Run the full pipeline. Verify:
- **Pass**: estimated K* ∈ [3, 7] (close to true 5)
- **Pass**: estimated M* = 2
- **Pass**: gaps have correct sign for deliberately over/under-invested actors

**P-2 (Null — pure noise DGP):**
Generate 20 actors with iid Gaussian noise (no structure).
- **Pass**: graph has no significant edges (null-model p > 0.05)
- **Pass**: MDL selects K* = 0 or 1
- **Pass**: regime switching not justified (M* = 1)
- **Pass**: OOS R² ≤ 0

**P-3 (Determinism — same input same output):**
Run pipeline twice with same config, same data, same random seed.
- **Pass**: all outputs are bitwise identical

**P-4 (Monotonicity — more data helps):**
Run pipeline with T = 20, 40, 60, 80 quarters (same DGP).
- **Pass**: OOS R² is non-decreasing with T (or at least: R²(80) > R²(20))

---

## Execution and Reporting

### Test Runner

All acceptance tests live in `tests/acceptance/smim/` (separate from unit tests).
They are tagged and can be run by section:

```bash
# All acceptance tests (takes 30-60 minutes)
uv run pytest tests/acceptance/smim/ -v --tb=short

# By section
uv run pytest tests/acceptance/smim/ -v -k "A_GR or I_GR or D_GR"  # Graph
uv run pytest tests/acceptance/smim/ -v -k "A_SC or I_SC or R_SC"  # Schur
uv run pytest tests/acceptance/smim/ -v -k "KF"                     # Kalman
uv run pytest tests/acceptance/smim/ -v -k "KIM"                    # Kim filter
uv run pytest tests/acceptance/smim/ -v -k "PID"                    # PID
uv run pytest tests/acceptance/smim/ -v -k "TDA"                    # TDA
uv run pytest tests/acceptance/smim/ -v -k "P_"                     # Pipeline
```

### Acceptance Gate

**The experiment programme MUST NOT start until all tests pass.**

The acceptance report is generated automatically:

```
SMIM Acceptance Report — [date]
===========================================
Graph Construction:     20/20 passed ✅
Spectral Decomposition: 35/35 passed ✅
Mode Selection:          9/9  passed ✅
Kalman Filter + EM:     14/14 passed ✅
Kim Filter:              9/9  passed ✅
Observability:           3/3  passed ✅
Phase Transition:        8/8  passed ✅
PID:                     6/6  passed ✅
Transfer Entropy:        6/6  passed ✅
TDA:                     7/7  passed ✅
Benchmarks/Gaps:         7/7  passed ✅
Pipeline Sanity:         4/4  passed ✅
-------------------------------------------
TOTAL:                 128/128 passed ✅
STATUS: READY FOR EXPERIMENTS
```

Any single failure → STATUS: BLOCKED. Fix and re-run full suite.
