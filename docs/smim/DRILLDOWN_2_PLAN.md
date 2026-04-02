# SMIM Drilldown #2: Nonlinear Dynamics, Frequency Upgrade, Emergence

> Created: 2026-04-02
> Status: v4 (restructured around 4 priority venues)
> Baseline: PLATINUM R^2=0.543 (DMD K=8, F=0.99*I, Q=0.5*I, sph R, EWM, online Q)
> Goal: R^2 > 0.55 AND emergence fires AND nonlinear structure detected

---

## 1. Executive Summary

Four research venues in priority order. Venues 1-3 run on EXISTING quarterly
data immediately. Venue 4 requires new data construction.

| Priority | Venue | Effort | Data needed | Tests |
|----------|-------|--------|-------------|-------|
| **1** | **Extended DMD (Koopman on alpha)** | Low | Existing quarterly | Nonlinear mode coupling |
| **2** | **Recursive / streaming DMD** | Medium | Existing quarterly | Emergence as basis change |
| **3** | **Multi-resolution DMD divergence** | Medium | Existing quarterly | Fast/slow timescale emergence |
| **4** | **Higher-frequency intensity** | High | New daily/monthly | Dense data for all methods |

Venues 1-3 answer: "Does nonlinear / emergent structure exist in the CURRENT
data that our linear pipeline misses?" If yes, we know what to look for at
higher frequency. If no, Venue 4 tests whether it's a frequency problem.

## 2. Why Emergence Fails: Diagnosis

### H1: Data too sparse (LIKELY)
T=20 quarterly points. PID needs reliable MI between K=8 mode pairs (28 pairs
from 20 points). TE needs T>30. TDA gets 1 sliding window. Mechanically
impossible to detect nonlinear structure.

### H3: Emergence absent at this resolution (POSSIBLE)
93 actors updating quarterly is not a complex system. Cross-sectional structure
is well-explained by 8 linear modes. Emergence requires interaction density
that quarterly/93 may lack.

### H2: Pipeline already captures it (UNLIKELY but untested)
E2-4 showed economic emergence features redundant with DMD-Kalman. But we've
never tested nonlinear prediction methods (EDMD, kernel DMD) on this data.
H2 cannot be evaluated until we've tried nonlinear methods.

---

## 3. VENUE 1: Extended DMD (Koopman on Modal Alpha)

**Priority: HIGHEST. Code exists. Can run today.**

### Rationale

Standard DMD extracts linear modes: alpha_{t+1} = F @ alpha_t. But if modes
interact nonlinearly (alpha_j * alpha_k predicts future alpha_j), standard
DMD misses this. Extended DMD lifts the state to polynomial observables,
approximating the Koopman operator in the lifted space.

### The key trick: lift in K-space, not N-space

The existing `ExtendedDMDDecomposer` lifts the full N=93 dimensional state:
- Degree 2 polynomial: P = C(95,2) = 4465 features. T/P = 20/4465 = 0.004.
  **Completely hopeless.**

Instead: two-stage approach:
1. Standard DMD on intensity Y -> alpha trajectory (T, K) where K=8
2. Polynomial lift of alpha -> Theta (T, P) where P = K + K(K+1)/2 = 44
3. Linear DMD on Theta -> nonlinear mode dynamics in Koopman space
4. T/P = 20/44 = 0.45 (tight but feasible). With online Q: workable.

**What this captures**: quadratic mode interactions. If energy-sector modes
and interest-rate modes interact multiplicatively (investment accelerates
when both are high), the alpha_j * alpha_k cross-term will appear in the
Koopman operator.

### Experiments

| ID | Test | Hypothesis | Metric |
|----|------|-----------|--------|
| V1-1 | EDMD degree 2 on alpha (P=44) | Quadratic coupling exists | R^2 > 0.543 |
| V1-2 | EDMD degree 3 on alpha (P=164) | Higher-order interactions | R^2 delta |
| V1-3 | Diagonal-only quadratic (P=16) | Self-interaction only | R^2 delta |
| V1-4 | EDMD + PLATINUM Kalman | Koopman basis + regularised filter | Combined R^2 |

### Implementation

Two approaches:
**(a)** Feed alpha trajectory (K, T) to `ExtendedDMDDecomposer` directly.
The decomposer treats each column as a state vector and lifts it. The output
basis is in lifted P-space; project back to K-space for Kalman filtering.

**(b)** Manual: lift alpha to Theta, estimate linear F_Theta on Theta via OLS,
predict by: alpha_pred -> lift -> F_Theta @ lifted -> extract K components.
More transparent, easier to debug.

Approach (b) is cleaner. Here's the math:
```
theta_t = [alpha_t; vech(alpha_t @ alpha_t')]   # (P,) = (K + K(K+1)/2,)
F_theta = argmin ||Theta[1:] - Theta[:-1] @ F||  # OLS on lifted trajectory
theta_{t+1|t} = F_theta @ theta_t
alpha_{t+1|t} = theta_{t+1|t}[:K]                # first K components
```

### Falsification

If V1-1 through V1-4 all give R^2 <= 0.543 (no improvement over linear DMD),
then modes do NOT interact nonlinearly at quarterly frequency. This doesn't
rule out nonlinear structure at higher frequency (Venue 4).

---

## 4. VENUE 2: Recursive / Streaming DMD

**Priority: HIGH. Detects emergence as basis change.**

### Rationale

Currently DMD is computed ONCE on the 5yr training window. The spectral basis
U is frozen. But the cross-sectional structure evolves. Recursive DMD updates
U at each time step (or each quarter), tracking how the basis rotates.

Emergence manifests as **abrupt basis change**: a new mode appears (eigenvalue
enters significance), or existing modes reorient (subspace angle jumps). If
the basis is frozen, these structural changes are invisible to the pipeline.

### Method: Incremental SVD for streaming DMD

At each new observation y_t:
1. Project y_t onto current basis: residual = y_t - U @ (U' @ y_t)
2. If ||residual|| > threshold: rank-1 update to SVD (add new direction)
3. If eigenvalue magnitude drops below threshold: mode death (remove mode)
4. Track K_effective(t): number of significant modes over time

This is online/streaming DMD via incremental SVD updates (Brand 2002).

### Experiments

| ID | Test | Hypothesis | Metric |
|----|------|-----------|--------|
| V2-1 | Recursive DMD (update U each quarter) | Basis adapts better | R^2 > frozen DMD |
| V2-2 | Track K_effective over time | Mode count varies | K_eff non-constant |
| V2-3 | Basis rotation speed as predictor | Fast rotation = regime change | Rotation predicts R^2 |
| V2-4 | Mode birth/death events | Structural breaks visible | Events align with market shocks |

### Emergence detection via V2-2 and V2-4

If K_effective changes over time (e.g., K=6 during calm periods, K=10 during
crises), this is DIRECT evidence of emergence: new cross-sectional structure
appearing that wasn't present before. The mode birth IS the emergence.

### Implementation

```python
# Recursive DMD pseudo-code
U, S, V = initial_DMD(Y_train)  # standard DMD on training window
K_eff = [rank(S > threshold)]

for t in test_period:
    y_new = observations[t]
    residual = y_new - U @ (U.T @ y_new)
    if np.linalg.norm(residual) > threshold:
        # Rank-1 SVD update (Brand 2002)
        U, S, V = incremental_svd_update(U, S, V, y_new)
    K_eff.append(np.sum(S > threshold))
    alpha[t] = U.T @ y_new  # project onto current (evolving) basis
```

### Falsification

If U barely changes across quarters (subspace angle between U_t and U_{t+1}
is < 5 degrees consistently), the cross-sectional structure is static and
emergence is absent at quarterly frequency.

---

## 5. VENUE 3: Multi-Resolution DMD Divergence

**Priority: HIGH. Novel emergence signal that works even at T=20.**

### Rationale

This is the most conceptually exciting venue. Compute DMD at multiple
timescales and measure DIVERGENCE between them:

- **Slow basis** U_slow: DMD on full 5yr (20Q) training window
- **Fast basis** U_fast: DMD on most recent 2yr (8Q)

When U_fast suddenly disagrees with U_slow (large subspace angle), something
new is happening at the fast timescale that the slow structure didn't predict.
This divergence IS emergence: a pattern visible only at high resolution that
the coarse-grained description misses.

### Why this works even at T=20

Unlike PID/TE which need T>30 for reliable estimation, subspace angle between
two DMD bases needs only the bases themselves (which are computed from their
respective windows). The COMPARISON is robust even if each individual DMD is
noisy, because systematic divergence will appear across multiple windows while
noise-driven divergence will be random.

### Experiments

| ID | Test | Hypothesis | Metric |
|----|------|-----------|--------|
| V3-1 | Compute subspace angle(U_fast, U_slow) per window | Angle varies | Non-constant angle |
| V3-2 | Angle predicts next-window R^2 change | High angle = regime shift | Correlation(angle, delta_R^2) |
| V3-3 | Use angle as regime indicator in prediction | Weight spectral vs mean by angle | R^2 improvement |
| V3-4 | Three-scale analysis (2yr/5yr/10yr) | Multi-scale structure | Emergence across scales |

### Timescale combinations to test

| Fast window | Slow window | What it detects |
|------------|------------|----------------|
| 2yr (8Q) | 5yr (20Q) | Short-term structural shift |
| 1yr (4Q) | 5yr (20Q) | Rapid reorientation (crisis) |
| 5yr (20Q) | 10yr (40Q) | Secular structural change |

### Implementation

```python
def subspace_angle(U1, U2, k=None):
    """Principal angle between column spaces of U1 and U2."""
    if k is None:
        k = min(U1.shape[1], U2.shape[1])
    Q1, _ = np.linalg.qr(U1[:, :k])
    Q2, _ = np.linalg.qr(U2[:, :k])
    svals = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    angles = np.arccos(np.clip(svals, -1, 1))
    return np.mean(angles)  # mean principal angle in radians
```

### Falsification

If subspace angle between fast and slow bases is consistently < 10 degrees
across all windows, the cross-sectional structure is scale-invariant (no
multi-scale emergence). If angle is large but uncorrelated with R^2 or
market events, it's noise-driven basis instability, not emergence.

---

## 6. VENUE 4: Higher-Frequency Intensity Construction

**Priority: MEDIUM-HIGH. Highest potential but highest risk.**

### Rationale

Venues 1-3 push the limits of what T=20 quarterly data can support. If they
all produce null results, the conclusion is: "emergence is undetectable at
quarterly frequency." Venue 4 tests whether higher frequency resolves this.

### Two frequency targets

**Monthly (T=60 per 5yr)**: triple the data. PID/TE become marginally reliable.
Construct from EDGAR quarterly via step interpolation + monthly indicators.

**Daily (T=1260 per 5yr)**: 63x more data. All nonlinear methods are well-
conditioned. But requires proxy construction.

### Daily intensity construction (v3 corrected approach)

Primary signal: **step-interpolated quarterly CapEx/Assets rank.**
- Each actor's daily intensity = last-reported quarterly CapEx/Assets rank
- Updates 4x per year when new EDGAR filing arrives (point-in-time correct)
- Cross-sectional structure identical to proven quarterly methodology
- Kalman filter sees 252 daily observations of slowly-moving quarterly state

Supplementary daily signals (multi-measure panel per actor):
- 60-day momentum rank (market's view of actor's investment trajectory)
- Sector ETF relative flow (sector-level capital allocation signal)
- GDELT sector tone (narrative investment sentiment)

### Actor space expansion (v3 corrected: equity-only first)

Step 1: Equity-only panel (N=150 SP500 stocks with EDGAR coverage)
- Validates that daily frequency helps without confounding actor-type effects
- Clean comparison: same actors, different frequency

Step 2: Add heterogeneous actors (if step 1 succeeds)
- 8 FRED daily macro series (VIX, credit spreads, rates)
- 9 GDELT daily narrative themes
- 11 sector ETFs
- Total: ~178 actors across 4 layers

### Validation gates

| Gate | Test | Threshold | If fails |
|------|------|----------|----------|
| V4-G1 | Daily AR(1) per actor R^2 > 0 | All actors positive | Proxy invalid |
| V4-G2 | Cross-sectional rank correlation with quarterly | Spearman > 0.5 | Proxy doesn't track CapEx |
| V4-G3 | PLATINUM pipeline R^2 > 0.10 at daily | Positive OOS | Pipeline works at daily |
| V4-G4 | EDMD > DMD at daily | delta R^2 > 0 | Nonlinear structure at daily |

---

## 7. Execution Plan

### Sprint 1: Venues 1-3 on existing data (can start NOW)

```
Day 1 (2-3 hours):
  V1-1: EDMD degree 2 on alpha trajectory
  V1-3: Diagonal-only quadratic
  V1-4: EDMD + PLATINUM Kalman
  -> GATE: nonlinear mode coupling?

  V3-1: Subspace angle(U_fast, U_slow) across 10 windows
  V3-2: Angle vs next-window R^2
  V3-3: Angle-weighted prediction
  -> GATE: multi-resolution emergence?

Day 2 (3-4 hours):
  V2-1: Recursive DMD with quarterly updates
  V2-2: Track K_effective over time
  V2-3: Basis rotation speed
  V2-4: Mode birth/death events
  -> GATE: dynamic basis structure?

  V1-2: EDMD degree 3 (if V1-1 positive)
  V3-4: Three-scale analysis (if V3-1 positive)
```

### Sprint 2: Venue 4 (only if Sprint 1 shows promise OR all null)

```
Day 3 (2-3 hours):
  V4-prep: Download SP500 daily, construct step-interpolated panel
  V4-G1/G2: Validate daily proxy
  V4-G3: PLATINUM at daily
  -> GATE: daily proxy valid?

Day 4 (3-4 hours):
  V4-G4: EDMD at daily
  Rerun V1-V3 experiments at daily frequency
  PID/TE/TDA at daily (emergence redux)
  -> GATE: emergence at daily?
```

## 8. Success Criteria

| Level | Criterion | What it proves |
|-------|----------|---------------|
| BRONZE | EDMD > DMD by any positive delta | Nonlinear mode coupling exists |
| SILVER | Multi-res divergence predicts R^2 | Emergence detected as scale disagreement |
| GOLD | Recursive DMD shows mode birth/death | Structural emergence directly observed |
| PLATINUM | R^2 > 0.55 with nonlinear/emergence | Framework fully validated |
| DIAMOND | All of the above + daily frequency | Full vision realised |

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|-----------|
| EDMD on alpha overfits (P=44, T=20) | High | V1 null | Ridge on Koopman; test P=16 diagonal |
| Subspace angle noise at T=20 | Medium | V3 noisy | Average across multiple fast windows |
| Recursive DMD numerically unstable | Medium | V2 fails | Use fixed K, only track rotation |
| Step-interpolated daily is just constant | Low | V4 trivial | Add OHLCV noise overlay |
| All venues null | Medium | No emergence | Valid science: publish as negative result |

## 10. Falsification Matrix

| Venue | Emergence EXISTS if | Emergence ABSENT if |
|-------|-------------------|-------------------|
| V1 EDMD | Quadratic terms improve R^2 by >0.5pp | All EDMD configs <= DMD |
| V2 Recursive | K_effective varies, rotation > 10 deg/yr | K stable, rotation < 5 deg/yr |
| V3 Multi-res | Angle corr with delta_R^2 > 0.3 | Angle uncorrelated or constant |
| V4 Daily | PID synergy > 2x bootstrap CI | All emergence null at T=1260 |

**Nuclear falsification**: if V1-V3 are all null on quarterly AND V4 is null
on daily, the conclusion is definitive: cross-sectional investment dynamics
are fundamentally linear with 8 modes, dual regularisation is optimal, and
emergence is not a feature of this system. This IS a publishable finding.

## 11. Sprint 1 Results (2026-04-02)

### V1: Extended DMD (Koopman on Alpha) -- NEGATIVE

All EDMD configurations are WORSE than PLATINUM:

| Config | P | T/P | Mean R^2 | Delta vs PLATINUM |
|--------|---|-----|----------|-------------------|
| PLATINUM (baseline) | -- | -- | 0.543 | -- |
| EDMD degree 2 full | 44 | 0.45 | 0.192 | -0.351 |
| EDMD degree 3 full | 164 | 0.12 | 0.302 | -0.241 |
| EDMD degree 2 diag | 16 | 1.25 | -2.114 | -2.657 |

Even 70/30 blends with PLATINUM degrade performance (0.51 vs 0.54).

**Root cause**: T=20 quarterly observations is too few for stable Koopman
operator estimation, even at P=16. Multi-step prediction (4 quarters)
compounds small estimation errors exponentially. The quadratic terms
alpha_j * alpha_k are noise-dominated at this sample size.

**Interpretation**: STRONG evidence for H1. Nonlinear mode coupling is
not detectable at quarterly frequency because the estimation problem is
underdetermined. This does NOT mean nonlinear structure is absent -- it
means T=20 cannot support its estimation. At daily T=1260, T/P=28.6
for P=44, which is well-conditioned.

### V3: Multi-Resolution DMD Divergence -- MIXED (informative null)

Subspace angles between fast (2yr) and slow (5yr) DMD bases:

| Fast window | K | Mean angle | Std | Range |
|-------------|---|-----------|-----|-------|
| 1yr | 5 | 45.4 deg | 8.1 | 31-55 |
| 2yr | 5 | 47.5 deg | 11.5 | 23-61 |
| 3yr | 5 | 47.5 deg | 9.3 | 33-62 |

The bases ARE substantially different across timescales (47 degrees is
more than halfway to orthogonal). However:

- V3-2: Angle does NOT predict R^2 (Spearman r=0.19, p=0.60)
- V3-2 lag: Angle(t) does NOT predict R^2(t+1) (r=0.07, p=0.87)
- V3-3: Angle-weighted prediction HURTS in all configurations

**Interpretation**: The 47-degree angle is close to the RANDOM expectation
for subspace angle between two independently estimated K=5 bases from
T=8 and T=20 observations respectively. With such short samples, DMD
estimation noise dominates genuine structural change. The angles likely
reflect estimation uncertainty, not multi-scale emergence.

**To resolve**: at daily frequency, the 2yr window has T=504 (not 8Q=8).
DMD estimation is much more stable. Any remaining angle > 15 degrees
would be genuine structural change, not estimation noise.

### Sprint 1 Verdict

Both venues produce NEGATIVE results on quarterly data, but for the
SAME reason: T=20 is too few for ANY method that needs to estimate
structure beyond simple linear regression. This is H1 confirmed:
**the data is too sparse for nonlinear/emergence detection.**

The results STRENGTHEN the case for Venue 4 (daily frequency). The
PLATINUM linear pipeline is at the ceiling of what quarterly data can
support. To go further, we need more temporal resolution.

### Key numbers for the paper

1. EDMD(deg2, P=44) at quarterly: R^2=0.19 vs linear DMD: 0.54.
   **Nonlinear lifting hurts by -35pp at T=20.**
2. Subspace angle between 2yr/5yr DMD: mean 47.5 degrees.
   **Consistent with estimation noise, not structural emergence.**
3. Angle-R^2 correlation: r=0.19, p=0.60.
   **No predictive relationship between scale divergence and accuracy.**

These support the paper's narrative: "quarterly investment intensity
dynamics are well-described by a linear 8-mode spectral model with
dual regularisation. Nonlinear extensions are data-starved at T=20."

## 12. What We Do NOT Attempt

- Do NOT skip Venues 1-3 for Venue 4. Quick experiments first.
- Do NOT use pure return-based intensity (proven dead end).
- Do NOT lift full N-space in EDMD (P=4465 >> T=20). Always K-space.
- Do NOT add heterogeneous actors before validating equity-only daily.
- Do NOT claim daily results replace the quarterly paper contribution.
- Do NOT skip falsification -- null results are valid science.
