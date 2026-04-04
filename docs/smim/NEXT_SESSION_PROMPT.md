# SMIM: Post-Iteration 5.2 — Paper Update

## What just happened (Iteration 5.2)

Parameter space exploration completed on 2026-04-04. Six phases tested ~170 configs.

**New best configuration:**
- K=2 modes, EWM=8Q, T=2yr training, no operator learning, DMD window=12Q
- Fixed config: SMIM R²=0.737 vs AR(1)=0.671, delta=+6.6pp, 10/10 wins
- Nested CV: SMIM R²=0.705 vs AR(1)=0.648, delta=+5.7pp, 8/8 wins, perm p=0.003
- Holdout (2023-2024): delta=+5.6pp

**Previous best (Iteration 5.1v2):**
- K=2, EWM=12Q, T=3yr → nested CV delta=+4.2pp, 8/8 wins

**Key findings:**
1. Dual-reg constants (F=0.99, Q₀=0.5, λ=0.3) are optimal — no improvement from retuning (Kalman redundant at K=2)
2. K=1 matches K=2 within 0.03pp — signal is "nearly 1-dimensional" but K=2 still wins
3. GOLD: Revenue/Assets (+1.0pp, 8/10 wins) and Multi-ratio (+2.8pp, 9/10 wins) also beat AR(1)
4. EWM=8 is the true halflife optimum (was 12 in coarse 5.1 grid)
5. T=2yr dominates — SMIM is more robust to short training than AR(1)
6. DMD window W=12Q slightly better than using all data (+0.4pp)

**Success criteria:**
- BRONZE: PASS (fixed Δ=+6.6pp > 4.0pp)
- SILVER: PASS (nested CV Δ=+5.7pp > 5.0pp)
- GOLD: PASS (2 alternative panels beat AR(1))
- PLATINUM: FAIL (K=1 close but doesn't beat K=2)

## Your task

Update the paper (`docs/smim/paper/smim_paper.tex`) with the 5.2 results:

1. **Table 1 (main results):** Update nested CV and fixed-config numbers
2. **Section on parameter sensitivity:** Add EWM fine-grid finding (EWM=8 optimal)
3. **Section on alternative signals:** Add Revenue/Assets and Multi-ratio results (GOLD)
4. **T-sweep discussion:** Update with T=2yr finding (SMIM robust, AR(1) degrades)
5. **K=1 finding:** Note that signal is nearly 1-dimensional (interesting for theory)
6. Regenerate any affected figures

## Key data files
- `results/metrics/iter5_2_phase_{a..f}.parquet` — phase results
- `results/metrics/iter5_2_nested_cv.parquet` — nested CV with new config space
- `scripts/smim/run_smim_iter5_2.py` — full experiment script

## Rules
- Only update numbers where the improvement is substantial AND passes nested CV
- The delta improvement from +4.2pp to +5.7pp is substantial (35% improvement, confirmed by nested CV)
- Keep the paper's structure and narrative intact
