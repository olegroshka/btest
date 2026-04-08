# Replication Guide: Every Number in the Paper

This document maps every numerical result in
*"Global Persistence, Local Residual Structure: Forecasting Heterogeneous Investment Panels"*
to the exact script, command, input data, and output file needed to reproduce it.

**Environment**: Python 3.11, `uv` package manager.  
**Setup** (run once):
```bash
uv sync --extra platform --extra dev --extra e2e
```

All commands are run from the project root (`C:\Users\olegr\PycharmProjects\btest`).  
All scripts use `PYTHONIOENCODING=utf-8` for Windows console compatibility.

---

## 1. Input Data

| File | Path | Description |
|------|------|-------------|
| **Primary panel** | `data/smim/intensities/experiment_a1_intensities.parquet` | 84Q x 93 actors, quarterly 2005Q1-2025Q4 |
| **Actor registry** | `data/smim/registries/experiment_a1_registry.json` | Actor metadata: sector, layer, actor_id |
| **Balance sheet** | `data/smim/processed/edgar_balance_sheet.parquet` | SEC EDGAR data for 146-firm and 270-actor panels |

The primary panel and registry are pre-built and committed to the repository.
No rebuild script is provided — the panel was constructed through a sequence of
data-pipeline scripts in `scripts/smim/` (FRED downloads, EDGAR processing, GDELT
consolidation) that depend on external API keys and rate limits. The committed
parquet files are the canonical inputs for all experiments.

---

## 2. Output Files

All experiment outputs are saved to `results/metrics/`. Key files:

| Output file | Produced by | Contains |
|-------------|-------------|----------|
| `iter6_4b.parquet` | `run_iter6_4b.py` | Table 5: all 8 architectures, 10 windows, per-block R² |
| `iter6_2_gate_a_models.parquet` | `run_iter6_2_gate_a.py` | Table 3: 9-model method comparison |
| `iter6_4b_placebo.parquet` | `run_iter6_4b_placebo.py` | Table 7: 1000 placebo permutations |
| `iter6_4b_xpanel.parquet` | `run_iter6_4b_xpanel.py` | Table 8: cross-panel validation (146-firm, 270-actor) |
| `iter6_3_gate_a_diagnostics.parquet` | `run_iter6_3_gate_a.py` | Geodesic distances, rotation predictability |
| `iter6_3_gate_a_models.parquet` | `run_iter6_3_gate_a.py` | 10 subspace prediction models |
| `iter6_4_gate_a.parquet` | `run_iter6_4_gate_a.py` | 7 target formulations |
| `iter6_4_gate_d.parquet` | `run_iter6_4_gate_d.py` | 5 gating policies |
| `iter6_4c_93actor.parquet` | `run_iter6_4c_parsimony.py` | Table 9: T x K_b parsimony grid |
| `iter6_4c_146firm.parquet` | `run_iter6_4c_parsimony.py` | 146-firm standalone spectral results |
| `iter6_4b_lowo.parquet` | `run_iter6_4b_lowo.py` | Section 5.3: leave-one-window-out block selection |
| `iter6_4b_referee_perwindow.parquet` | `run_iter6_4b_referee.py` | DM-HAC inference, block-specific AR(1) |
| `iter6_1_validation.parquet` | `run_iter6_1_validation.py` | Table 2: augmentation gain across panels |
| `iter6_4b_referee_round3.parquet` | `run_iter6_4b_referee_round3.py` | GBM+sector features, cross-sectional rank IC |
| `iter6_4b_referee_round3b.parquet` | `run_iter6_4b_referee_round3b.py` | Held-out decade, stratified placebo, rank-matched tech/health, SE(rho_b), ENS explanation |

**Note**: `run_iter6_4b_supplementary.py` (GBM, two-block, MAE) outputs to console only and does not save a parquet file. Re-run the script to reproduce those numbers.

---

## 3. Paper Tables — Script Mapping

### Table 1: Panel descriptive statistics (Section 2.1)
**Numbers**: N=93, 84 quarters, persistence ρ (macro≈0.88, firms≈0.60)  
**Source**: Descriptive statistics computed from `experiment_a1_intensities.parquet`  
**Reproduced by**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_1_validation.py
```
Persistence values are printed in the console output ("Panel descriptive statistics" section).

---

### Table 2: Two-stage augmentation vs baselines (Section 3.3)
**Numbers**: 93-actor Δ=+0.036; 146-firm Δ=+0.017; 270-actor Δ=+0.025  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_1_validation.py
```
**Output**: `results/metrics/iter6_1_validation.parquet`  
**Note**: The 93-actor AR(1) R²=0.594 is the fixed-parameter version. The rolling AR(1) R²=0.610 used as the Table 3 baseline is computed in `run_iter6_2_gate_a.py`.

---

### Table 3: Method comparison — 9 models (Section 3.4)
**Numbers**: 9 models across 3 complexity classes; forecast-error correlations ρ(DMD,PCA)=0.990, ρ(DMD,Ridge)=0.980, ρ(PCA,Ridge)=0.969  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_2_gate_a.py
```
**Output**: `results/metrics/iter6_2_gate_a_models.parquet`  
**Reproduces**: All 9 R² values, Δ vs rolling AR(1), t-statistics, p-values, CIs, and pairwise forecast-error correlations in the table footnote.  
**Holm-Bonferroni**: The adjustment is computed inline in the script output.

---

### Table 4: Block assignments (Section 3.5)
**Numbers**: Block sizes (Diversified=23, Macro/Inst=11, Tech/Health=25, Remainder=34)  
**Source**: Computed from `experiment_a1_registry.json` sector/layer metadata  
**Reproduced by**: Any script that calls `define_blocks()` — e.g. `run_iter6_4b.py` prints block sizes at startup.

---

### Table 5: Eight architectures (Section 4.1) — THE MAIN TABLE
**Numbers**: G0=0.591, BA=0.611, G1=0.630, ENS=0.639, S1=0.599, BA_M2=0.661, M1=0.669, M2=0.677  
All Δ vs G1, t-statistics, p-values, CIs, and W (windows positive)  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b.py
```
**Output**: `results/metrics/iter6_4b.parquet`  
**Runtime**: ~1.5 seconds  
**Reproduces**:
- All 8 architecture R² values (full-panel and per-block)
- Paired t-tests and bootstrap CIs for each vs G1
- Quality gates (G1≈0.630 ±0.005, G0≈0.591 ±0.005)
- M1 vs S1 diagnostic (+0.070, t=16.12)
- BA_M2 result (Δ=+0.031 vs G1, 10/10 — the A-1 referee experiment)
- ENS result (Δ=+0.009 vs G1, 9/10 — the C-3 referee experiment)

---

### Table 6: Per-block R² decomposition (Section 4.2)
**Numbers**: Diversified G0=0.415, G1=0.392; Tech/Health G0=0.554, G1=0.681, M2=0.808  
**Script**: Same as Table 5 — `run_iter6_4b.py`  
**Output**: Per-block columns in `iter6_4b.parquet` (`block_SEC_diversified`, `block_LAYER_macro_inst`, `block_MERGED_tech_health`, `block_REMAINDER`)

---

### Table 7: Placebo test (Section 5.1)
**Numbers**: Real Δ=+0.047, placebo mean=-0.004, placebo std=0.0065, z=7.82, p≤0.001 (0/1000)  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_placebo.py
```
**Output**: `results/metrics/iter6_4b_placebo.parquet` (1001 rows: 1 real + 1000 random partitions)  
**Runtime**: ~25 minutes (1000 permutations x 10 windows)  
**Exact z-score calculation**: `(0.04692 - (-0.00423)) / 0.006543 = 7.819`

---

### Table 8: Cross-panel validation (Section 5.2)
**Numbers**: 146-firm Δ=-0.003; 270-actor Δ=+0.001  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_xpanel.py
```
**Output**: `results/metrics/iter6_4b_xpanel.parquet`  
**Input data**: `data/smim/processed/edgar_balance_sheet.parquet`

---

### Table 9: Parsimony frontier T x K_b (Section 7.4)
**Numbers**: T=5yr/K_b=4 → Δ=+0.047; T=2yr/K_b=4 → Δ=+0.097  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4c_parsimony.py
```
**Output**: `results/metrics/iter6_4c_93actor.parquet`, `results/metrics/iter6_4c_146firm.parquet`

---

### Table 10: Falsification summary (Section 6)
**Numbers**: Method equivalence ρ≥0.97; rotation unpredictable; target-invariant ceiling; gating never helps  
**Reproduced by running all four gate scripts** (see Section 6 below).

---

### Table 11: Candidate block evaluation (Appendix G)
**Numbers**: 10 candidate blocks with per-block Δ R² and W/10  
**Script**:
```bash
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_paper_robustness.py
```
**Output**: Printed to console. The single-block diagnostics for all 10 candidates.

---

### Table 12: Hyperparameter inventory (Appendix J)
**Numbers**: All hyperparameter values and sources  
**Source**: Constants defined at the top of `scripts/smim/run_iter6_4b.py` (K_DEFAULT=8, Q_INIT_SCALE=0.5, LAMBDA_Q=0.3, etc.)  
**No script needed** — this table documents values set in code.

---

## 4. Paper Figures — Script Mapping

All figures generated by:
```bash
uv run python scripts/smim/paper_figures_v3.py
```
**Output directory**: `docs/smim/paper/figures/`

| Figure | Script output filename | LaTeX references | Data source |
|--------|----------------------|------------------|-------------|
| Figure 1 (hierarchy) | TikZ in LaTeX | — | No script — drawn in `smim_paper.tex` |
| Figure 2 (per-window R²) | `fig2_per_window_r2.pdf` | `figures/fig2_per_window.pdf` | `iter6_4b.parquet` |
| Figure 3 (placebo histogram) | `fig3_placebo.pdf` | `figures/fig3_placebo.pdf` | `iter6_4b_placebo.parquet` |
| Figure 4 (pipeline diagram) | TikZ in LaTeX | — | No script — drawn in `smim_paper.tex` |
| Figure 4b (geodesic) | `fig4_geodesic.pdf` | Not referenced in LaTeX | `iter6_3_gate_a_diagnostics.parquet` |
| Figure 5 (per-block R²) | `fig5_block_r2.pdf` | `figures/fig5_per_block.pdf` | `iter6_4b.parquet` |

**Note**: The LaTeX file references older filenames (`fig2_per_window.pdf`, `fig5_per_block.pdf`)
that are stale copies on disk. After running `paper_figures_v3.py`, either rename the
new outputs to match the LaTeX references, or update the `\includegraphics` paths in
`smim_paper.tex` to use the v3 filenames.

**Prerequisite**: Run `run_iter6_4b.py` and `run_iter6_4b_placebo.py` first to generate the parquet files.

---

## 5. Inline Numbers — Section by Section

### Abstract
| Number | Source |
|--------|--------|
| R² 0.630 → 0.677, Δ=+0.047 | `run_iter6_4b.py` → `iter6_4b.parquet` |
| CI [+0.036, +0.058] | `run_iter6_4b.py` bootstrap CI |
| 10/10 windows | `iter6_4b.parquet`, column `full_r2`, arch G1 vs M2 |
| placebo p≤0.001 (0/1000) | `run_iter6_4b_placebo.py` → `iter6_4b_placebo.parquet` |
| Held-out decade Δ=+0.050, 10/10 | `run_iter6_4b_referee_round3b.py` → Experiment 1 |
| 72% from tech/health | `run_paper_robustness.py` (drop-tech/health experiment) |
| Rank-matched super-additivity | `run_iter6_4b_referee_round3b.py` → Experiment 3 |
| +0.048 recursive macro | `run_fred_recursive_robustness.py` |
| +0.038 filing lag | `run_filing_lag_robustness.py` |
| Stratified placebo z=7.25, p≤0.001 | `run_iter6_4b_referee_round3b.py` → Experiment 2 |

### Section 1 (Introduction)
| Number | Source |
|--------|--------|
| +3.6 pp augmentation gain | `run_iter6_1_validation.py` → Table 2 |
| -2.3 pp diversified degradation | `run_iter6_4b.py` → per-block: G1(0.392) - G0(0.415) = -0.023 |
| Forecast-error correlations 0.969-0.990 | `run_iter6_2_gate_a.py` → Table 3 footnote |
| 49°/quarter rotation, ACF=-0.07, LB p=0.29 | `run_iter6_3_gate_a.py` |
| 146-firm Δ=-0.003, 270-actor Δ=+0.001 | `run_iter6_4b_xpanel.py` → Table 8 |
| 31.5°/quarter at K=4 | `run_iter6_3_gate_a.py` → `iter6_3_gate_a_diagnostics.parquet` |

### Section 3.4 (Method equivalence)
| Number | Source |
|--------|--------|
| R²≈0.630 ceiling | `run_iter6_2_gate_a.py` → Table 3 |
| GBM R²=0.661 | `run_iter6_4b_supplementary.py` → Experiment 2 |
| GBM+sector R²=0.592 | `run_iter6_4b_referee_round3.py` → Experiment 1 |
| Rolling AR(1) R²=0.610 | `run_iter6_2_gate_a.py` (rolling baseline) |

### Section 4.1 (Full-panel result)
| Number | Source |
|--------|--------|
| DM-HAC 6.84-7.38 | `run_iter6_4b_referee.py` → DM-HAC section |
| Moving-block bootstrap CIs [+0.035,+0.059] and [+0.036,+0.059] | `run_iter6_4b_referee.py` |
| Sign test 2^{-10}≈0.001 | Exact computation from 10/10 sign pattern |
| M1 vs S1: +0.070, t=16.12 | `run_iter6_4b.py` → diagnostic output |
| BA_M2: +0.031 vs G1, 10/10 | `run_iter6_4b.py` → architecture "BA_M2" |
| ENS: +0.009 vs G1, 9/10 | `run_iter6_4b.py` → architecture "ENS" |

### Section 4.3 (Geodesic)
| Number | Source |
|--------|--------|
| 31.5°/quarter global rotation (K=4) | `run_iter6_3_gate_a.py` → diagnostics |
| 14-28° within-block rotation | `run_iter6_3_gate_a.py` |
| p-values: TH=0.080, MI=0.145, Div=0.305 | `run_iter6_3_gate_a.py` → matched-size controls |

### Section 5.3 (LOWO)
| Number | Source |
|--------|--------|
| All 10 windows select same 3 blocks | `run_iter6_4b_lowo.py` |

### Section 5.4 (Held-out decade — NEW)
| Number | Source |
|--------|--------|
| Phase A: 7 blocks selected on 2010-2014 | `run_iter6_4b_referee_round3b.py` → Experiment 1 |
| Phase B: Δ=+0.050, t=9.11, CI [+0.040, +0.061], 10/10 | `run_iter6_4b_referee_round3b.py` → Experiment 1 |

### Section 5.5 (Stratified placebo — NEW)
| Number | Source |
|--------|--------|
| Stratified z=7.25, p≤0.001, 0/1000 | `run_iter6_4b_referee_round3b.py` → Experiment 2 |
| Placebo mean=+0.001, std=0.0063, max=+0.031 | `run_iter6_4b_referee_round3b.py` → Experiment 2 |

### Section 5.4 (Drop diversified)
| Number | Source |
|--------|--------|
| 70-actor panel Δ=+0.035, 10/10 | `run_paper_robustness.py` |
| Drop TH: Δ=+0.013, 7/10 (72% eliminated) | `run_paper_robustness.py` |
| Two-block (TH only): Δ=+0.031, 10/10, 66% retained | `run_iter6_4b_supplementary.py` → Experiment 1b |

### Section 5.5 (Block boundary sensitivity)
| Number | Source |
|--------|--------|
| Perturbation (a): Δ=+0.044; (b): +0.042; (c): +0.043 | `run_paper_robustness.py` |

### Section 5.6 (Remainder local treatment)
| Number | Source |
|--------|--------|
| Four-block: Δ=+0.043; Remainder-only: Δ=-0.004, 4/10 | `run_paper_robustness.py` |

### Section 5.7 (Pipeline robustness)
| Number | Source |
|--------|--------|
| Recursive FRED: Δ=+0.048 (10/10) | `run_fred_recursive_robustness.py` |
| Exclude macro/inst: Δ=+0.053 (10/10) | `run_fred_recursive_robustness.py` |
| Filing lag: Δ=+0.038 (10/10) | `run_filing_lag_robustness.py` |

### Section 5.8 (Economic content)
| Number | Source |
|--------|--------|
| 6.6% RMSE reduction | Computed: √(1-0.677)/√(1-0.630) = 0.934, i.e. 6.6% |
| RMSE 0.176 → 0.164 | Derived from R² values |
| MAE 0.129 → 0.120 | `run_iter6_4b_supplementary.py` → Experiment 3 |
| Rank IC: M2=0.822, G1=0.794, Δ=+0.029, t=6.30 | `run_iter6_4b_referee_round3.py` → Experiment 2 |
| Firm-only IC: M2=0.806, G1=0.773, Δ=+0.033, t=6.17 | `run_iter6_4b_referee_round3.py` → Experiment 2 |
| IC IR: 11.9→15.3 (all actors), 10.3→12.9 (firms) | `run_iter6_4b_referee_round3.py` → Experiment 2 |

### Section 6.1 (Method does not matter — locally)
| Number | Source |
|--------|--------|
| Local DMD vs PCA+ridge: mean Δ=-0.028, worst -0.059 | `run_iter6_4_gate_c.py` (all 9 blocks tested) |
| GBM R²=0.661 | `run_iter6_4b_supplementary.py` → Experiment 2 |

### Section 6.2 (Rotation not forecastable)
| Number | Source |
|--------|--------|
| 49.2°±16.8° at K=8; θ_1=45.8° | `run_iter6_3_gate_a.py` |
| ACF(1)=-0.07, LB p=0.29, direction cosine=0.047 | `run_iter6_3_gate_a.py` |
| All 10 models worse than persistence | `run_iter6_3_gate_a.py` → `iter6_3_gate_a_models.parquet` |

### Section 6.3 (Target not the problem)
| Number | Source |
|--------|--------|
| 7 target formulations, max |Δ gain|=0.018 | `run_iter6_4_gate_a.py` → `iter6_4_gate_a.parquet` |
| Split-half ρ=0.513, ceiling≈0.765 | `run_iter6_4_gate_a.py` |
| First differences: gain=+0.051, absolute R²=0.047 | `run_iter6_4_gate_a.py` |

### Section 6.4 (Gating does not help)
| Number | Source |
|--------|--------|
| 5 gating policies; best (dispersion) loses 0.014 | `run_iter6_4_gate_d.py` → `iter6_4_gate_d.parquet` |

### Section 7.5 (Hyperparameter insensitivity diagnostic P1-P5)
| Number | Source |
|--------|--------|
| P1: |Δ|=0.018 (predicted <0.005) | `run_iter6_4c_parsimony.py` |
| P2: Δ=-0.014 (predicted >+0.005) | `run_iter6_4c_parsimony.py` |
| P4: +0.008 (146-firm T=2yr K=2) | `run_iter6_4c_parsimony.py` → `iter6_4c_146firm.parquet` |
| P5: 0.010 (146-firm T=5yr K=8) | `run_iter6_4c_parsimony.py` → `iter6_4c_146firm.parquet` |

### Appendix A (DMD mathematics)
| Number | Source |
|--------|--------|
| Eigenvalue magnitudes |λ_k| ∈ [0.87, 0.97] | `run_iter6_3_gate_a.py` → spectral diagnostics |
| Modal R²=0.69 in-sample | `run_iter6_3_gate_a.py` |

### Appendix B (Standalone transition diagnostic)
| Number | Source |
|--------|--------|
| Standalone DMD R²=0.486 (full Ã), 0.483 (diag), 0.415 (near-identity) | `run_iter6_1_final.py` |
| Per-actor AR(1) R²=0.594 | `run_iter6_1_validation.py` |

### Appendix C (Structural spectral analysis)
| Number | Source |
|--------|--------|
| 8 stable modes, oscillatory components 44-91° | `run_iter6_3_gate_a.py` |

### Appendix D (Target sensitivity details)
| Number | Source |
|--------|--------|
| All 7 target R² values and gains | `run_iter6_4_gate_a.py` |

### Appendix E (Gating policy details)
| Number | Source |
|--------|--------|
| 5 gating policy Δ values | `run_iter6_4_gate_d.py` |

### Appendix F (Rotation diagnostics)
| Number | Source |
|--------|--------|
| 6 subspace prediction models and their Δ vs persistence | `run_iter6_3_gate_a.py` → `iter6_3_gate_a_models.parquet` |

### Appendix G (Candidate block evaluation)
| Number | Source |
|--------|--------|
| 10 candidate blocks, per-block Δ and W/10 | `run_paper_robustness.py` |
| Rank-matched: tech K3=+0.014, health K3=+0.001, merged K3=+0.028, merged K4=+0.031 | `run_iter6_4b_referee_round3b.py` → Experiment 3 |
| SE(rho_b) ≈ 0.04 per block, cross-block range ≈ 0.60 | `run_iter6_4b_referee_round3b.py` → Experiment 4 |

### Appendix H (FRED normalisation robustness)
| Number | Source |
|--------|--------|
| Recursive Δ=+0.048; exclude macro/inst Δ=+0.053 | `run_fred_recursive_robustness.py` |

### Appendix I (Filing-lag robustness)
| Number | Source |
|--------|--------|
| Lagged Δ=+0.038; G1 0.630→0.639; M2 0.677→0.677 | `run_filing_lag_robustness.py` |

---

## 6. Full Reproduction — Run Order

The scripts have dependencies: some scripts consume outputs from earlier scripts.
Run in this order to reproduce all paper results from scratch:

### Phase 1: Core pipeline (no dependencies)
```bash
# Table 2: augmentation baselines (93-actor, 146-firm, 270-actor)
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_1_validation.py

# Table 3: 9-model method comparison
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_2_gate_a.py

# Geodesic rotation diagnostics
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_3_gate_a.py
```

### Phase 2: Main architecture comparison
```bash
# Table 5: ALL 8 architectures (G0, BA, G1, S1, M1, M2, BA_M2, ENS)
# Also: Table 6 (per-block), M1 vs S1 diagnostic
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b.py
```

### Phase 3: Validation and robustness (depends on Phase 2 output)
```bash
# Table 7: placebo test (1000 permutations) — ~25 min
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_placebo.py

# Table 8: cross-panel validation (146-firm, 270-actor)
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_xpanel.py

# LOWO block selection
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_lowo.py

# DM-HAC inference, block-specific AR(1) baseline
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_referee.py

# Drop-sector, boundary sensitivity, remainder local, 10 candidates
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_paper_robustness.py

# FRED recursive normalisation
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_fred_recursive_robustness.py

# Filing-lag robustness
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_filing_lag_robustness.py
```

### Phase 4: Falsification programme (independent of Phase 2)
```bash
# Target sensitivity (7 formulations)
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4_gate_a.py

# Gating policies (5 policies)
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4_gate_d.py
```

### Phase 5: Supplementary experiments
```bash
# Table 9: T x K_b parsimony grid + pre-registered predictions
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4c_parsimony.py

# Two-block partition, GBM non-linear baseline, MAE robustness
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_supplementary.py
```

### Phase 5b: Referee round-3 experiments (depends on Phase 2 output)
```bash
# GBM+sector features, cross-sectional rank IC (~60s)
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_referee_round3.py

# Held-out decade, stratified placebo, rank-matched tech/health,
# SE(rho_b), ENS explanation (~4 min, dominated by stratified placebo)
PYTHONIOENCODING=utf-8 uv run python scripts/smim/run_iter6_4b_referee_round3b.py
```

### Phase 6: Figures (depends on Phase 2 and 3 outputs)
```bash
# All paper figures (requires iter6_4b.parquet and iter6_4b_placebo.parquet)
uv run python scripts/smim/paper_figures_v3.py
```

### Phase 7: LaTeX compilation
```bash
cd docs/smim/paper
pdflatex smim_paper.tex
bibtex smim_paper      # if using BibTeX; paper uses thebibliography
pdflatex smim_paper.tex
pdflatex smim_paper.tex
```

---

## 7. Expected Runtimes

| Script | Runtime | Notes |
|--------|---------|-------|
| `run_iter6_4b.py` | ~2 s | Main 8-architecture comparison |
| `run_iter6_4b_placebo.py` | ~25 min | 1000 permutations |
| `run_iter6_4b_xpanel.py` | ~5 s | Cross-panel |
| `run_iter6_2_gate_a.py` | ~10 s | 9-model comparison |
| `run_iter6_3_gate_a.py` | ~15 s | Geodesic + 10 prediction models |
| `run_iter6_4_gate_a.py` | ~10 s | 7 target formulations |
| `run_iter6_4_gate_d.py` | ~5 s | 5 gating policies |
| `run_iter6_4c_parsimony.py` | ~30 s | T x K_b grid |
| `run_iter6_4b_supplementary.py` | ~50 s | GBM (bottleneck: per-actor GBM fitting) |
| `run_paper_robustness.py` | ~10 s | Drop-sector, boundary, 10 candidates |
| `run_fred_recursive_robustness.py` | ~3 s | FRED recursive |
| `run_filing_lag_robustness.py` | ~3 s | Filing lag |
| `run_iter6_4b_lowo.py` | ~3 s | LOWO |
| `run_iter6_4b_referee.py` | ~3 s | DM-HAC, block AR(1) |
| `run_iter6_4b_referee_round3.py` | ~60 s | GBM+sector, rank IC |
| `run_iter6_4b_referee_round3b.py` | ~4 min | Held-out decade, stratified placebo, rank-matched, SE(rho_b), ENS |
| `paper_figures_v3.py` | ~2 s | Generate PDFs |
| **Total** | **~32 min** | Dominated by placebo permutations |

---

## 8. Verification Checksums

After running all scripts, verify these key values match the paper:

| Check | Expected | Script | How to verify |
|-------|----------|--------|---------------|
| G1 R² | 0.630 ±0.005 | `run_iter6_4b.py` | QG1 in output |
| G0 R² | 0.591 ±0.005 | `run_iter6_4b.py` | QG2 in output |
| M2 R² | 0.677 | `run_iter6_4b.py` | Full-panel table |
| M2-G1 Δ | +0.047 | `run_iter6_4b.py` | Full-panel table |
| M2 wins | 10/10 | `run_iter6_4b.py` | W column |
| Placebo z | 7.82 | `run_iter6_4b_placebo.py` | z-score output |
| Placebo std | 0.0065 | `run_iter6_4b_placebo.py` | std output |
| 146-firm Δ | ≈-0.003 | `run_iter6_4b_xpanel.py` | Cross-panel table |
| LOWO same partition | 10/10 same | `run_iter6_4b_lowo.py` | "All windows select..." |
| BA_M2 R² | 0.661 | `run_iter6_4b.py` | Full-panel table |
| ENS R² | 0.639 | `run_iter6_4b.py` | Full-panel table |
| GBM R² | 0.661 | `run_iter6_4b_supplementary.py` | Experiment 2 output |
| GBM+sector R² | 0.592 | `run_iter6_4b_referee_round3.py` | Experiment 1 output |
| MAE M2 | 0.120 | `run_iter6_4b_supplementary.py` | Experiment 3 output |
| Rank IC M2 | 0.822 | `run_iter6_4b_referee_round3.py` | Experiment 2 output |
| Rank IC G1 | 0.794 | `run_iter6_4b_referee_round3.py` | Experiment 2 output |
| Held-out decade Δ | +0.050 | `run_iter6_4b_referee_round3b.py` | Experiment 1 output |
| Stratified placebo z | 7.25 | `run_iter6_4b_referee_round3b.py` | Experiment 2 output |
| Merged K3 Δ | +0.028 | `run_iter6_4b_referee_round3b.py` | Experiment 3 output |

---

## 9. Complete Script Inventory

| Script | Paper section(s) | Purpose |
|--------|-----------------|---------|
| `run_iter6_1_validation.py` | Table 2, Appendix B | Augmentation baselines, standalone DMD |
| `run_iter6_1_final.py` | Appendix B | Standalone transition diagnostic |
| `run_iter6_2_gate_a.py` | Table 3, Section 3.4 | 9-model method comparison |
| `run_iter6_2_gate_b.py` | (Supporting) | Training-window / multi-horizon sweep |
| `run_iter6_3_gate_a.py` | Section 4.3, 6.2, Appendix C/F | Geodesic distances, rotation predictability |
| `run_iter6_4_gate_a.py` | Section 6.3, Appendix D | Target sensitivity (7 formulations) |
| `run_iter6_4_gate_b.py` | (Supporting) | Local coherence discovery, NCD diagnostics |
| `run_iter6_4_gate_c.py` | Section 6.1 | Local matched horse race (per-block DMD vs PCA+ridge); "mean Δ=-0.028" is in console output |
| `run_iter6_4_gate_d.py` | Section 6.4, Appendix E | 5 gating policies |
| **`run_iter6_4b.py`** | **Table 5, Table 6** | **Main 8-architecture comparison** |
| `run_iter6_4b_placebo.py` | Table 7 | 1000 placebo permutations |
| `run_iter6_4b_xpanel.py` | Table 8 | Cross-panel validation |
| `run_iter6_4b_lowo.py` | Section 5.3 | Leave-one-window-out block selection |
| `run_iter6_4b_referee.py` | Section 4.1 | DM-HAC inference |
| `run_iter6_4c_parsimony.py` | Table 9, Section 7.5 | T x K_b grid + predictions P1-P5 |
| `run_iter6_4b_supplementary.py` | Sections 3.4, 5.3, 5.8, 6.1 | Two-block, GBM, MAE (console output only — no parquet saved) |
| `run_iter6_4b_referee_round3.py` | Sections 3.4, 5.8 | GBM+sector features, cross-sectional rank IC |
| `run_iter6_4b_referee_round3b.py` | Sections 5.4, 5.5, App. G, Limitations | Held-out decade, stratified placebo, rank-matched tech/health, SE(rho_b), ENS explanation |
| `run_paper_robustness.py` | Sections 5.6-5.8, Appendix G | Drop-sector, boundary, remainder, 10 candidates |
| `run_fred_recursive_robustness.py` | Section 5.7, Appendix H | FRED recursive normalisation |
| `run_filing_lag_robustness.py` | Section 5.7, Appendix I | Filing-lag robustness |
| `paper_figures_v3.py` | Figures 2, 3, 5 | Generate all paper figures |
