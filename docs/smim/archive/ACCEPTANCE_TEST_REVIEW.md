# Acceptance Test Implementation Review

## Summary

The implementation adapted 128 → 119 tests (9 Kim filter tests counted differently in
the report) with 18 documented deviations from the original spec. I've classified each
deviation into one of four categories:

- ✅ **Correct fix**: the original spec had a mathematical error; the implementation is right
- 👍 **Acceptable pragmatic**: the original spec was theoretically correct but impractical;
  the relaxation is justified and doesn't compromise scientific validity
- ⚠️ **Needs scrutiny**: the relaxation might mask a real implementation problem
- ❌ **Tighten back up**: the relaxation goes too far; the original criterion was achievable

---

## Deviation-by-Deviation Analysis

### 1. I-GR-1: Directionality check — planted edges, ≥4/5 instead of 10 random

**Category: 👍 Acceptable pragmatic**

Using planted edges instead of random sampling is actually stronger — it tests
against known ground truth. The ≥4/5 tolerance for a borderline case where the
F-statistic is similar in both directions is reasonable for Granger causality, which
is inherently noisy on individual pairs.

**Verdict**: Keep as-is.

---

### 2. I-SP-1: Density tolerance — `density ≤ 0.15 + 1/N²`

**Category: ✅ Correct fix**

L1 soft-thresholding operates on continuous weights but density is a discrete
count. At the boundary, one entry can push density over by exactly 1/N². The
slack is the minimum possible for a discrete-to-continuous bridge.

**Verdict**: Keep as-is. The slack is negligible (0.25% for N=20).

---

### 3. A-NM-1: Circulant graph instead of ER for rewiring distribution

**Category: ✅ Correct fix**

This is a genuine improvement. ER graphs have heterogeneous degree sequences
that create "congested" positions where certain edges are forced to appear in
most rewirings. Testing a structural invariant on a vertex-transitive graph
is the right call — it separates "is the rewiring algorithm correct" from
"does the degree sequence allow uniform sampling."

**Verdict**: Keep as-is. Better than the original spec.

---

### 4. A-PL-1: TIGHT tolerance (1e-8) instead of machine precision

**Category: 👍 Acceptable pragmatic**

The rotation matrix test goes through LAPACK SVD which introduces O(ε_mach × ‖A‖)
error. For ‖A‖=1, actual error is ~1e-16, so 1e-8 is generous but harmless.
The important thing is that the test EXISTS and catches real bugs.

**Verdict**: Keep as-is, but note this for documentation: "tolerance 1e-8 due
to LAPACK SVD intermediate computation."

---

### 5. A-PL-3: Sign flip handling — `min(‖U-R‖, ‖U+R‖)`

**Category: ✅ Correct fix**

Polar decomposition is unique for invertible A, but the SVD branch cut can
introduce a global det(U)=−1 vs +1 ambiguity. Checking both signs is the
correct mathematical approach.

**Verdict**: Keep as-is.

---

### 6. A-HD-1: Singular vector recovery checks omitted

**Category: ⚠️ Needs scrutiny**

The original spec required recovering U_L and U_R from H's eigenvectors —
this tests the CORE PURPOSE of Hermitian dilation (preserving directional
upstream/downstream structure). Dropping this to "just check eigenvalue pairs"
significantly weakens the test.

The dilation is not useful if it correctly produces ±σ_k eigenvalues but you
can't actually extract the left/right singular vectors from the eigenvectors.
That extraction is the whole point — it's how the proposal recovers directional
mode structure from a symmetric eigensolver.

**Verdict: ADD BACK the U_L/U_R recovery test.** The implementation note says
"I-HD-1/3 cover this" but they don't — I-HD-1 checks eigenvalue pairing and
I-HD-3 checks singular VALUES. Neither checks that the singular VECTORS are
correctly recoverable from H's eigenvectors.

Create a new test or extend A-HD-1:
```
eigvals, eigvecs = eigh(H)
# For eigenvalue +σ_k, eigenvector is [u_L; u_R]/√2
# For eigenvalue -σ_k, eigenvector is [u_L; -u_R]/√2
# Recover: u_L = √2 * eigvec[:N], u_R = √2 * eigvec[N:] (for +σ_k)
# Verify: recovered u_L ≈ true left singular vector (up to sign)
# Verify: recovered u_R ≈ true right singular vector (up to sign)
```

---

### 7. A-HD-2 and D-HD-1: Direct np.block construction bypassing decomposer

**Category: 👍 Acceptable pragmatic**

The decomposer's `_validate_operator` enforcing square input is a reasonable API
constraint (the SMIM pipeline always uses square operators). Testing the mathematics
of H(A) on rectangular A at the linear algebra level is valid — it verifies the
mathematical property without conflating it with the API validation.

**Verdict**: Keep as-is.

---

### 8. D-HD-1: Scaled zero threshold `1e-9 × ‖A‖_F` instead of fixed 1e-10

**Category: ✅ Correct fix**

For a 30×50 standard-normal matrix, singular values are O(√50) ≈ 7. A fixed
1e-10 threshold would misclassify numerical zeros that are O(ε_mach × 7) ≈ 1e-15
as genuine eigenvalues. Scaling by ‖A‖_F is standard practice in numerical
linear algebra.

**Verdict**: Keep as-is.

---

### 9. A-DMD-1: T=50 and 15% tolerance instead of T=500 and 5%

**Category: ⚠️ Needs scrutiny**

The justification (decaying eigenvalues bury the signal) is correct in principle,
but T=50 with 15% tolerance is very loose for a test of a fundamental algorithm.

The issue is that 0.8^20 ≈ 0.01 with noise σ=0.01, so the second mode IS
genuinely hard to recover at long horizons. But DMD on a clean linear system
should be nearly exact at SHORT horizons. The right fix isn't "accept 15% error"
but rather "reduce noise or test on a less challenging eigenvalue pair."

**Verdict: Adjust the test design instead of loosening tolerance.** Try:
- A_true = [[0.95, 0.1], [0, 0.90]] (both eigenvalues decay slowly)
- σ = 0.001 (low noise)
- T = 200
- Tolerance: 5% on eigenvalues

This tests DMD accuracy without the confound of an eigenvalue falling below the
noise floor. Keep the current T=50/15% version as a SEPARATE stress test (D-DMD-3)
that documents the noise-floor limitation.

---

### 10. I-DMD-1: Symmetric A instead of generic random

**Category: ⚠️ Needs scrutiny**

The implementation note explains that `ModalFrame.basis` stores only the real
part of DMD modes, causing reconstruction failure for non-symmetric A. This is
a **real problem with the ModalFrame design**, not just a test issue.

DMD modes for non-symmetric systems are genuinely complex. If `ModalFrame.basis`
discards the imaginary part, the DMD decomposer cannot correctly represent
oscillatory dynamics — which are the MAIN use case for DMD in the SMIM framework
(economic cycles have complex eigenvalues).

**Verdict: This reveals a design issue to fix.** Options:
1. Store complex modes in ModalFrame (add a `complex_basis` field)
2. Store modes as pairs of real/imaginary parts
3. Convert complex conjugate pairs to real-valued sinusoidal modes

Option 3 is standard in DMD practice. For now, **accept the test as-is** but
open a tracked issue: "ModalFrame must support complex modes for DMD." This is
a pre-experiment blocker if DMD is a comparison method in the spectral experiments.

---

### 11. A-MDL-1: K* ∈ {2,3,4} instead of exactly 3; MDL formula change

**Category: ✅ Correct fix (the formula) + 👍 Acceptable (the K* range)**

The original MDL formula `T·N·log(res_var) + k·log(T)` was indeed incorrect —
the penalty term must scale with the number of free parameters per mode (N
components per eigenvector), not just k. The corrected
`T·log(res_var) + k·N·log(T)/2` is the BIC-consistent MDL penalty for
subspace estimation. This is a genuine bug in my original spec.

Accepting K* ∈ {2,3,4} is reasonable — MDL is a consistent selector (converges
to truth as T→∞) but at finite T=200 it can be off by ±1.

**Verdict**: Keep the formula fix. Accept the K* range.

---

### 12. A-LZ-3: ρ < 0.95 instead of ρ < 0.15

**Category: ⚠️ Needs scrutiny**

This is concerning. The original spec expected random sequences to have
ρ ≈ 0.05 (nearly incompressible); the implementation gets ρ ≈ 0.90 (highly
compressible?). The issue is the normalisation formula.

The original spec assumed the standard LZ normalisation:
`ρ = 1 - c(s)·log₂(n)/n` where c(s) is the number of distinct phrases.
For random data, c(s) ≈ n/log₂(n), so ρ ≈ 0.

The implementation uses `ρ = 1 - c(s)/n` which gives ρ ≈ 1 - 1/log₂(n) ≈ 0.90
for n=1000. This means the implementation's compressibility measure does NOT
correctly reflect algorithmic complexity.

**This matters** because the mode selection pipeline uses ρ_min to filter noise
modes. If ρ ≈ 0.90 for random noise, and the threshold is ρ_min = 0.1, then
NOTHING gets filtered — the compressibility criterion becomes vacuous.

**Verdict: Fix the normalisation.** Use the standard formula:
`ρ = 1 - c(s)·log₂(n)/n`
Then the original thresholds work: constant→0.999, periodic→~0.5, random→~0.05.
The acceptance threshold ρ < 0.15 for random data should be restored.

If for some reason the alternative normalisation is preferred, then the
threshold ρ_min in the config must be recalibrated to ~0.95 (not 0.1), and
the entire mode selection pipeline needs re-verification.

---

### 13. A-EM-1: Q tolerance 25% instead of 20%; fixed init instead of random

**Category: 👍 Acceptable pragmatic**

Q estimation has higher variance than F or R because it requires the smoothed
state covariance (a second-order quantity). 25% is reasonable for T=2000.
Fixed init (F=0.9I, Q=R=0.1I) is fine — the test verifies parameter recovery,
not robustness to initialisation (that's D-EM-1's job).

**Verdict**: Keep as-is.

---

### 14. I-EM-1: LL monotonicity tolerance -1e-6 instead of -1e-10

**Category: 👍 Acceptable pragmatic**

The implementation note documents two genuine M-step bugs that were found and
fixed during testing — this is EXACTLY what the acceptance suite is for.

The cross-covariance bug (using `P_s[t] @ F` instead of the Shumway-Stoffer
lag-one formula) and the observation noise bias (missing `U@P_s@U.T` term) are
significant implementation errors. The fact that the suite caught them is a
major win.

The 1e-6 tolerance (vs 1e-10) accounts for floating-point accumulation across
the forward-backward recursion. After the bugs are fixed, the observed jitter
is 1e-8, so 1e-6 provides 100× margin.

**Verdict**: Keep as-is. The two bug fixes are more valuable than a tighter tolerance.

---

### 15. A-KF-2: R=1e-10·I instead of R=0; tolerance 1e-5

**Category: 👍 Acceptable pragmatic**

R=0 causes a singular innovation covariance S = H P H^T + R = H P H^T, which
requires a pseudoinverse. Using R=1e-10·I keeps the matrix invertible and the
filtered state is within 1e-5 of the observation, which is "perfect observation"
for all practical purposes.

**Verdict**: Keep as-is.

---

### 16. A-KF-3: Per-component 2σ check, 90% instead of 95%

**Category: ✅ Correct fix**

The original spec's "within 2σ for >95% of time steps" didn't account for the
multivariate case. For K=3 independent components, the probability that ALL
three are within 2σ simultaneously is 0.9545³ ≈ 0.87. Requiring >95% of
JOINT steps would reject a correct filter.

Per-component checking at >90% is the right approach.

**Verdict**: Keep as-is.

---

### 17. P-2: M*=1 not required; OOS R² ≤ 0.1 instead of ≤ 0

**Category: ⚠️ Needs scrutiny**

The explanation (BIC penalty too small relative to Kim filter's ability to fit
noise heteroscedasticity) is plausible, but this is an important null-hypothesis
test. If the framework selects M*=2 on pure noise, regime switching is not a
reliable diagnostic on its own.

The OOS R² ≤ 0.1 criterion is reasonable — on pure noise, even a model that
overfits in-sample should have OOS R² near 0 (and 0.1 is generous enough to
account for lucky splits).

However, the BIC behaviour deserves documentation: "On pure noise with K=1, T=150,
BIC selects M=2 due to heteroscedasticity fitting. This is a known limitation;
regime selection should always be validated OOS."

**Verdict**: Accept OOS R² ≤ 0.1 as the primary criterion, but **add a note in
DECISIONS.md** that BIC regime selection on short noise sequences is unreliable.
Also **document this in the experiment plan**: when interpreting regime counts,
always cross-validate with OOS R² before trusting M*.

---

### 18. A-CI-1: Redesigned from gradual ramp to abrupt phase change

**Category: ✅ Correct fix**

The criticality index C_t measures the RATIO of statistics in adjacent windows.
With a slow linear ramp in ρ, adjacent windows have nearly identical statistics,
so C_t ≈ 1 throughout. The Kendall-τ test would fail for a correct implementation.

The redesigned test (abrupt jump from ρ=0.5 to ρ=0.95) creates a sharp
transition where C_t should spike — this is exactly what critical slowing down
theory predicts near a phase boundary.

**Verdict**: Keep as-is. Better test of the actual phenomenon.

---

### 19. A-CI-2: ρ=0.7, w=50, T=2000; median ∈ (0.1, 20) instead of C_t ≈ 1

**Category: 👍 Acceptable pragmatic**

The original spec's expectation (C_t ≈ 1 with std < 0.3) assumed perfect
estimation of variance ratios and ACF ratios, which requires large windows.
With w=8, sampling noise dominates. The relaxation to w=50 and a wide
acceptance band is practical.

**Verdict**: Keep as-is, but the wide band (0.1, 20) is very loose. Consider
tightening to (0.5, 5) once the implementation is stable.

---

### 20. A-PID-2: Linear additive construction instead of product

**Category: ✅ Correct fix**

This is an important correction. The product construction Y = X1 × X2 has
zero Gaussian pairwise correlation (E[X1² X2] = E[X1²]E[X2] = 0 for
independent X1, X2). Under Gaussian MMI, which operates on the covariance
matrix only, this construction shows zero synergy — not because synergy
doesn't exist, but because Gaussian MMI can't detect non-linear interactions.

The linear additive construction Y = X1 + X2 + noise has analytically known
PID decomposition under Gaussian MMI and correctly produces S > 0.

**Verdict**: Keep as-is. This is a better test.

---

### 21. I-PID-2: Tolerance 0.02 nats instead of 1e-6

**Category: 👍 Acceptable pragmatic**

The PID decomposition identity holds exactly for the population quantities, but
sample estimation introduces O(p/T) bias. At T=5000, this is ~0.001 nats.
The 0.02 threshold provides 20× margin over expected bias while still catching
systematic errors (which would be O(0.1) nats or larger).

**Verdict**: Keep as-is.

---

### 22. R-TE-1: 50% tolerance instead of 25%

**Category: ⚠️ Needs scrutiny**

The implementation note cites a documented ~37% divergence between Kraskov
Algorithm 1 and Frenzel-Pompe CMI. This is a known issue in the TE estimation
literature — different KSG variants DO produce different results on finite data.

However, 50% is very loose for a reference comparison. If two implementations
diverge by 50%, you can't tell whether a 30% change in TE due to experimental
conditions is real or an estimator artefact.

**Verdict**: Keep 50% for now but **add a documented limitation**: "KSG-based TE
estimates have O(30-50%) inter-implementation variance. Experimental conclusions
should be based on TE RATIOS and RANKINGS across conditions, not absolute values."
This is important context for the experiment programme.

---

### 23. I-TDA-1: `d_B < 2ε` instead of `d_B < ε`

**Category: ✅ Correct fix**

The VR stability theorem gives d_B(Dgm(X), Dgm(Y)) ≤ 2·d_H(X,Y).
For per-coordinate perturbation ε, d_H ≤ ε√d ≤ 2ε (in R²).
The original bound of ε was off by a factor of 2.

**Verdict**: Keep as-is.

---

### 24. I-MB-1: Attribution decomposes benchmark difference, not total gap

**Category: ✅ Correct fix**

The original spec said "attribution sums to gap[i,t]" which is algebraically
wrong. The modal attribution decomposes the difference between modal and
predictive benchmarks: `attr_sum = bench_pred - bench_modal`. This is correct
because modal and predictive benchmarks use different projections of the same
filtered state.

**Verdict**: Keep as-is. The original spec had an error.

---

## Blocking Issues (Must Fix Before Experiments)

### BLOCKER 1: LZ Compressibility Normalisation (Deviation #12)

The mode selection pipeline's compressibility filter is effectively disabled if
random noise has ρ ≈ 0.90 under the implemented normalisation. Either:
- Fix the normalisation to use `ρ = 1 - c(s)·log₂(n)/n`, or
- Recalibrate ρ_min in config to distinguish structured from random under the
  current normalisation (e.g., ρ_min = 0.97)

This affects experiments B1 (component ablation) and B5 (mode selection tests).

### BLOCKER 2: Complex DMD Modes (Deviation #10)

If DMD is used as a comparison method in spectral experiments (B2), the
ModalFrame must support complex modes. Otherwise DMD cannot represent
oscillatory dynamics, making the comparison unfair.

Options: store complex basis, or convert conjugate pairs to real sinusoidal form.

### BLOCKER 3: Hermitian Dilation Vector Recovery (Deviation #6)

The singular vector recovery test must be added back. Without it, the
Hermitian dilation decomposer could produce correct eigenvalues but fail to
recover the directional mode structure — which is the entire reason the method
exists.

---

## Should-Fix (Before Experiments, Moderate Priority)

### SF-1: A-DMD-1 Test Design (Deviation #9)

Add a clean low-noise DMD test alongside the current challenging one:
- Clean test: well-separated eigenvalues, low noise, tight tolerance (5%)
- Stress test: decaying modes, higher noise, loose tolerance (15%)

### SF-2: P-2 BIC Regime Selection Documentation (Deviation #17)

Add to DECISIONS.md: BIC regime selection on short pure-noise sequences is
unreliable. Experimental protocol should always cross-validate M* with OOS R².

### SF-3: TE Estimation Variance Documentation (Deviation #22)

Document in the experiment plan: KSG TE estimates have ~30-50% inter-implementation
variance. Use TE ratios and rankings, not absolute values.

### SF-4: A-CI-2 Tighter Acceptance Band (Deviation #19)

Once implementation is stable, tighten the stationary criticality band from
(0.1, 20) to (0.5, 5).

---

## Verdict Summary

| # | Deviation | Category | Action |
|---|-----------|----------|--------|
| 1 | I-GR-1 planted edges | 👍 Acceptable | Keep |
| 2 | I-SP-1 density slack | ✅ Correct fix | Keep |
| 3 | A-NM-1 circulant graph | ✅ Correct fix | Keep |
| 4 | A-PL-1 TIGHT tolerance | 👍 Acceptable | Keep |
| 5 | A-PL-3 sign flip | ✅ Correct fix | Keep |
| 6 | A-HD-1 vector recovery dropped | ⚠️ Scrutiny | **ADD BACK** ❌ |
| 7 | A-HD-2/D-HD-1 direct construction | 👍 Acceptable | Keep |
| 8 | D-HD-1 scaled threshold | ✅ Correct fix | Keep |
| 9 | A-DMD-1 T=50, 15% | ⚠️ Scrutiny | **Redesign** ⚠️ |
| 10 | I-DMD-1 symmetric only | ⚠️ Scrutiny | **Track issue** ⚠️ |
| 11 | A-MDL-1 formula + range | ✅ Correct fix | Keep |
| 12 | A-LZ-3 ρ < 0.95 | ⚠️ Scrutiny | **FIX NORMALISATION** ❌ |
| 13 | A-EM-1 Q tol 25% | 👍 Acceptable | Keep |
| 14 | I-EM-1 LL tol -1e-6 | 👍 Acceptable | Keep |
| 15 | A-KF-2 R=1e-10 | 👍 Acceptable | Keep |
| 16 | A-KF-3 per-component 2σ | ✅ Correct fix | Keep |
| 17 | P-2 M* not required | ⚠️ Scrutiny | Keep + **document** |
| 18 | A-CI-1 redesigned | ✅ Correct fix | Keep |
| 19 | A-CI-2 wide band | 👍 Acceptable | Keep, tighten later |
| 20 | A-PID-2 linear construct | ✅ Correct fix | Keep |
| 21 | I-PID-2 tol 0.02 | 👍 Acceptable | Keep |
| 22 | R-TE-1 50% tolerance | ⚠️ Scrutiny | Keep + **document** |
| 23 | I-TDA-1 2ε bound | ✅ Correct fix | Keep |
| 24 | I-MB-1 attribution identity | ✅ Correct fix | Keep |

**Score: 9 correct fixes, 8 acceptable, 6 need scrutiny (3 blocking)**

**Bottom line**: the implementation quality is GOOD — the acceptance suite
caught two genuine EM bugs and several spec errors. But three issues
(LZ normalisation, complex DMD modes, Hermitian vector recovery) must be
resolved before experiments can start.
