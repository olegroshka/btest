# Iteration 6.4c: The Parsimony Frontier

> Date: 2026-04-07
> Status: PROPOSED (revised after review)
> Predecessor: Iteration 6.4b (mixture architecture, +0.047)
> Scope: Joint T×K sweep on existing architecture. No new methods.

---

## 0. Motivation

The current paper reports one operating point: T=5yr, K=8 globally,
K_b ∈ {2,4} locally. But Iteration 5 demonstrated R²≈0.711 on the
146-firm panel with K=2, T=2yr — a result the current paper does not
engage with. The paper says "the 146-firm panel shows no benefit"
(Δ=−0.002 for the mixture), but this was tested only at T=5yr with
K=8/K_b=4. It is silent on whether:

1. Low-K local models improve the mixture on the 93-actor panel
2. The optimal K_b scales with T (shorter T → fewer modes)
3. Standalone spectral with low K works on homogeneous panels at short T

A practitioner with a short-history dataset (new asset class, emerging
market, recently-listed sector) needs to know which (T, K) operating
point is right for their data regime. The paper currently gives no
guidance.

---

## 1. Pre-Registered Predictions (with falsifiable thresholds)

Written before running any experiment. Threshold: 0.005 (comparable
to placebo std = 0.007).

**P1.** On the 93-actor mixture at T=5yr:
|R²(K_b=2) − R²(K_b=4)| < 0.005.
The K_b choice sits in a flat region of the parsimony frontier.

**P2.** On the 93-actor mixture at T=2yr:
R²(K_b=2) > R²(K_b=4) by at least 0.005.
Low K provides necessary regularisation when T≈8 quarters.
Also: K_b ≥ 3 will degrade sharply on the larger blocks (tech/health
N=25) because T < K_b in the relevant sample-size sense.

**P3.** On the 93-actor mixture at T=3yr:
R²(K_b=2) and R²(K_b=3) within 0.003 of each other; both > R²(K_b=6).

**P4.** On the 146-firm panel, standalone spectral at T=2yr, K=2:
R²(standalone) > R²(AR(1)) by at least 0.010.
(Reproducing the Iteration 5 finding.)

**P5.** On the 146-firm panel, standalone spectral at T=5yr, K=8:
|R²(standalone) − R²(AR(1))| < 0.010.
(Matching the existing finding that spectral ≈ AR(1) at the current
operating point.)

---

## 2. Experiment Design

### Grid (revised — dropped 146-firm mixture control)

| Panel | Architecture | T (years) | K or K_b | Cells |
|-------|-------------|-----------|----------|-------|
| 93-actor | Mixture M2 | {2, 3, 5} | {2, 3, 4, 6} | 12 |
| 93-actor | Global augmentation G1 | {2, 3, 5} | {2, 3, 4, 6, 8} | 15 |
| 146-firm | Standalone spectral | {2, 3, 5} | {2, 3, 4, 6, 8} | 15 |

Total: 42 cells. Each cell is 10 rolling OOS windows.

**Why no 146-firm mixture:** The 146-firm panel is homogeneous. Testing
the mixture on it requires introducing a new block-discovery method
(k-means) that the rest of the paper doesn't use. If k-means produces
a gain, it contradicts Section 5.2 using an untested method. If it
doesn't, it adds complexity for nothing. The 146-firm question is
purely about standalone spectral parsimony, not about the mixture.

### Block assignment for short T

Block structure is pre-specified (sector/layer), unchanged across T
values. Only K_b varies. At T=2yr with 8 training quarters, K_b ≥ 3
on a 25-actor block means estimating a 25×3 basis from 8 observations
of a 25-dimensional vector — likely degenerate. The grid captures this
directly.

### Baselines per cell

For each (T, K) cell, also compute per-actor AR(1) R² at the same T.
This gives the gain at each operating point.

### Per-block decomposition

For each 93-actor mixture cell, report BOTH the aggregate Δ(M2−G1)
AND the per-block contribution from MERGED_tech_health specifically.
This distinguishes "the mixture is K_b-robust" (aggregate) from
"tech/health needs K_b=4 specifically" (block-specific). The
tech/health block is the most consequential cell because it
contributes the dominant share of the headline gain.

---

## 3. Expected Outputs

### Table A: Parsimony frontier (93-actor mixture)

| T | K_b | AR(1) | G1 | M2 | Δ(M2−G1) | M2 tech/health |
|---|-----|-------|-----|-----|---------|----------------|
| 2 | 2 | ? | ? | ? | ? | ? |
| 2 | 3 | ? | ? | ? | ? | ? |
| 2 | 4 | ? | ? | ? | ? | ? |
| 2 | 6 | ? | ? | ? | ? | ? |
| 3 | 2 | ? | ? | ? | ? | ? |
| ... | | | | | | |
| 5 | 4 | ? | ? | ? | ? | ? |

### Table B: 146-firm standalone spectral

| T | K | AR(1) | Spectral | Δ |
|---|---|-------|----------|---|
| 2 | 2 | ? | ? | ? |
| 2 | 3 | ? | ? | ? |
| ... | | | | |
| 5 | 8 | ? | ? | ? |

### Figure: Two side-by-side heatmaps

Left: Δ(M2−G1) by (T, K_b) — aggregate mixture gain.
Right: M2 tech/health R² by (T, K_b) — per-block view.
Current operating point (T=5, K_b=4) marked with a star.

---

## 4. Integration into the Paper

### If the parsimony frontier is real (P1–P3 confirmed):

Add Section 7.5: **"Configuration for Short-History Applications"**

- Report the (T, K_b) parsimony frontier as a small table
- State that the paper's pre-specified K_b values sit in a flat
  region of the frontier — K_b ∈ {2, 4} produce statistically
  indistinguishable mixture gains at T=5yr, indicating the result
  is robust to local-rank choice
- Note that at shorter T, lower K_b is needed (natural
  regularisation)
- Derive the empirical heuristic from the data (not from a prior
  formula)

### If the 146-firm standalone result confirms Iteration 5:

Keep Section 5.2 (mixture-null finding) intact. Add one footnote
in Section 5.2:

*"See Section 7.5 for a separate finding: low-K standalone spectral
models can match AR(1) on this panel under short training windows.
This spectral parsimony benefit is distinct from the mixture
architectural benefit documented here."*

Add the 146-firm standalone result in Section 7.5 as a complementary
observation. The two findings are about different things:
- Section 5.2: the ARCHITECTURAL gain requires heterogeneity
- Section 7.5: the SPECTRAL gain works on homogeneous panels at
  low K/short T

### If the hypothesis is NOT confirmed:

Add three sentences to Limitation 5:

*"A joint (T, K_b) sweep shows the mixture gain is stable across
K_b ∈ {2, 3, 4, 6} at T=5yr (all within ±0.003). The paper's
pre-specified K_b values are not sensitive to this choice."*

---

## 5. Kill Rules

**Kill the parsimony-frontier story if:**
- K_b has no effect: all K_b values within ±0.003 of each other
  at EVERY T value tested
- The T×K_b interaction is non-monotone and uninterpretable

**Kill the 146-firm standalone story if:**
- Standalone spectral at K=2, T=2yr does NOT beat AR(1) by at
  least 0.010 on the 146-firm panel

**In either kill case:** report briefly in limitations, no new
subsection.

---

## 6. Timeline

| Step | Time | What |
|------|------|------|
| Implement T×K grid runner | 1h | Parameterise existing pipeline |
| Run 93-actor grid (27 cells) | 5 min | |
| Run 146-firm grid (15 cells) | 3 min | |
| Analyse + generate tables | 1h | |
| Write subsection (if positive) | 1h | |

**Total: ~3 hours.**

---

## 7. Files

| File | Role |
|------|------|
| `scripts/smim/run_iter6_4c_parsimony.py` | T×K grid sweep |
| `results/metrics/iter6_4c_parsimony.parquet` | Grid results |
| `docs/smim/ITERATION_6_4c_PLAN.md` | This file |
