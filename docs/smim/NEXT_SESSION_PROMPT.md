# SMIM Iteration 5.3: Pooled AR(1)+FE and DFM Baselines

## What you need to know

Six independent reviews of the SMIM paper (`docs/smim/paper/smim_paper.tex`)
unanimously identified two missing baselines as the single biggest weakness
for journal submission. The paper currently beats per-actor AR(1) and PCA
variants, but has no **pooled econometric** comparator. This iteration adds them.

## Current state

- Paper: 33 pages, 5 revision rounds complete, all quality gates pass
- Headline: nested CV R^2=0.711, delta=+0.042 vs AR(1), 8/8 wins (146-firm CapEx/Revenue)
- PCA baseline: R^2=0.702 (indistinguishable from AR(1)); truncated SVD PCA identical
- Key finding from R4: SVD regularisation ruled out; advantage is from DMD's temporal snapshot pairs

## Your task

Implement and evaluate two new baselines, then update the paper.

### 1. Implement baselines

Read the plan: `docs/smim/ITERATION_5_3_PLAN.md`

Write: `scripts/smim/run_baselines_iter5_3.py`

The script must:
- Use the SAME 146-firm panel as `run_smim_iter5_1_cv2.py` (call `build_panel()`)
- Use the SAME `ewm_demean()` function
- Use the SAME `oos_r_squared` metric
- Run at BOTH configs: T=3yr/tau=12Q AND T=2yr/tau=8Q
- Output per-window R^2 for both baselines + AR(1) + comparison to saved SMIM results
- Save to `results/metrics/iter5_3_*.parquet`

**Baseline A: Pooled AR(1) + Firm FE**
```
y_{i,t} = alpha_i + rho * y_{i,t-1} + eps
```
Single shared rho, firm intercepts. Estimated via within-transformation.
Rolling: re-estimate each quarter during test year (matching SMIM).

**Baseline B: Dynamic Factor Model (PCA + VAR(1))**
```
Step 1: EWM demean (same tau)
Step 2: PCA -> Lambda (N x K), f_t = Lambda^T * y_tilde_t
Step 3: VAR(1): f_{t+1} = A * f_t + eta
Step 4: Forecast: y_hat = mu + Lambda * A * f_{t-1}
```
K=2 factors, rolling update each quarter. This is Stock & Watson (2002).

### 2. Verify

Before trusting results, check quality gates:
- QG1: AR(1) from new script = 0.699 (T=3yr) and 0.671 (T=2yr)
- QG2: Pooled rho ~ 0.28 (median per-actor rho)
- QG3: Pooled+FE R^2 >= per-actor AR(1) (pooling should not hurt)
- QG4: DFM R^2 >= PCA projection R^2 (VAR dynamics should help)
- QG5: VAR(1) eigenvalues inside unit circle
- QG6: No NaN/Inf predictions

### 3. Update paper

Add rows to Table 3 (both config blocks) and Table 4 (if significant).
Add interpretation paragraph to Section 4.1 after PCA discussion.
Update abstract/conclusion if results change the story.

## Key files

| File | Role |
|------|------|
| `docs/smim/ITERATION_5_3_PLAN.md` | Full plan with scenarios |
| `docs/smim/paper/smim_paper.tex` | Paper (Table 3 at lines ~395-420) |
| `scripts/smim/run_smim_iter5_1_cv2.py` | Main pipeline (panel, AR(1), SMIM) |
| `scripts/smim/run_pca_baseline.py` | PCA baseline (structure to follow) |
| `results/metrics/iter5_1v2_phase_a.parquet` | SMIM fixed-config results |
| `results/metrics/pca_kalman_fixed.parquet` | PCA results for comparison |
| `results/metrics/pca_t2yr_fixed.parquet` | PCA at T=2yr |
| `docs/smim/CLAUDE.md` | SMIM-specific conventions and test commands |
| `CLAUDE.md` | Project-wide conventions |

## Critical conventions

- All intensity values are cross-sectional percentile ranks in [0,1]
- The panel is stored as a pandas DataFrame: index=quarterly dates, columns=tickers
- `otr` in the pipeline means "observations for training" -- shape (T_quarters, N_firms)
- EWM demeaning operates on (T, N) arrays, returns (1, N) means
- R^2 is computed by pooling all actor-quarter (pred, actual) pairs per window
- Rolling update: after each test quarter, expand training, re-estimate everything

## What NOT to do

- Do NOT change existing SMIM or PCA code
- Do NOT modify existing results files
- Do NOT change the paper's structure or narrative beyond adding the new baseline rows
- Do NOT run nested CV for the new baselines -- fixed-config only (matching Table 3)
- Do NOT add the baselines to the abstract until results are verified
