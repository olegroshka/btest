# SMIM CLAUDE.md

## What This Is

SMIM (Spectral Multi-layer Investment Misallocation) estimates actor-specific investment
gaps by: building a directed multilayer graph → spectral decomposition → state-space
filtering with regime switching → emergence diagnostics → benchmark computation → gaps.

This is a **research framework**, not a trading strategy. The connection to the trading
backtester happens only through bridge signals in `smim/signals/`.

## Mathematical Notation → Python Mapping

| Math | Python | Shape | Module |
|------|--------|-------|--------|
| y_{i,t} | `intensities` | (N, T) | `data/` |
| A_t^{(r)} | `adj_channel` | sparse (N, N) | `graph/edges/` |
| A_t | `operator` | sparse (N, N) | `graph/operators.py` |
| U_t | `modal_frame.basis` | (N, K) | `spectral/` |
| α_t | `alpha_t` | (K,) or (T, K) | `dynamics/` |
| F^{(z)} | `transition_matrices[z]` | (K, K) | `dynamics/` |
| z_t | `regime_labels` | (T,) int | `dynamics/` |
| ψ_t | `order_param` | (d,) | `dynamics/phase_transition.py` |
| C_t | `criticality` | scalar | `dynamics/phase_transition.py` |
| S_{jk} | `synergy_matrix` | (K, K) | `emergence/pid.py` |
| Δ_{i,t} | `gap_result.gaps` | (N, T) | `gaps/` |
| y*_{i,t} | `gap_result.benchmarks` | (N, T) | `gaps/` |

## Standing Assumptions — NEVER VIOLATE

- **A1 (Point-in-time)**: Never use data with pub_date > backtest_date. Check via `pit_store`.
- **A2 (Typed comparability)**: Normalisation is per-ActorType via InvestmentIntensityMapper.
- **A3 (Sparse propagation)**: After sparsification, operator retains >80% spectral energy.
- **A4 (Stable modes)**: Eigenmode rank correlation >0.5 across ≥80% of rolling windows.
- **A5 (Regime persistence)**: Average regime duration >8 quarters.

## Mandatory Rule: Benchmark Labels

**Every GapResult must carry a BenchmarkClass.** The codebase enforces this via the
GapResult dataclass — the field is non-optional. Never report, log, or plot a gap
without its benchmark label.

## Implementation Pattern

Every SMIM component follows this pattern:

1. **Protocol** is defined in `interfaces.py` (already exists)
2. **Implementation** lives in the appropriate submodule
3. **Config** is a Pydantic model in `config.py` (section per component)
4. **Tests** mirror the source path under `tests/unit/smim/`
5. **Acceptance tests** live in `tests/acceptance/smim/`

```python
# Example: implementing a new edge estimator
# 1. Read the EdgeEstimator protocol in interfaces.py
# 2. Create smim/graph/edges/my_estimator.py
# 3. Implement the protocol
# 4. Add config to SmimConfig in config.py
# 5. Add tests in tests/unit/smim/graph/test_my_estimator.py
# 6. Register in the edge estimator factory
```

## Known Implementation Deviations from Spec

These are spec corrections discovered during acceptance test implementation.
Do not revert them — the tests encode the **correct** behaviour.

| Test | Original spec | Correct behaviour | Reason |
|------|--------------|-------------------|--------|
| I-MB-1 | "attr sums to gap[i,t]" | `attr_sum = gap_modal − gap_pred` | Spec had algebraic error |
| P-2 | M* = 1 for pure noise | BIC may select M > 1 | BIC penalty too small vs Kim filter LL gain; OOS R² is the definitive check. See ADR-001 in DECISIONS.md |
| R-TE-1 | Within 25% of IDTxl | Tolerance 50% | Kraskov Alg-1 vs Frenzel-Pompe ~37% divergence at T=2000 |
| I-TDA-1 | `d_B < ε` | `d_B < 2ε` | VR stability theorem: `d_B ≤ 2·d_H ≤ 2ε` |

## KimFilter Limitations (Important)

- **Symmetric initialisation**: `em_estimate` initialises all M regimes with identical
  `F = 0.9·I`. EM cannot break symmetry from a symmetric start → converges to M=1
  solution regardless of true M. When testing regime detection, provide known
  asymmetric parameters rather than relying on EM to discover them.
- **alpha_pred approximation**: Line 172 of `kim_filter.py` sets
  `alpha_pred[t] = alpha_filt[t]` (predicted = filtered). Predictive and modal
  benchmarks from KimFilter are therefore nearly identical.

## Acceptance Test Infrastructure

- `tests/acceptance/smim/conftest.py` — synthetic data generators (fixed seeds)
- `tests/acceptance/smim/conftest_report.py` — pytest plugin: auto-prints gate report
- `scripts/run_smim_acceptance.py` — standalone runner with `--section` support
- **119/119 tests pass** (as of 2026-03-19); skipped tests do not block the gate

### Running tests

```bash
# Unit tests (~4 s)
uv run pytest tests/unit/smim/ -q

# Acceptance suite with gate report (~60 s)
uv run python scripts/run_smim_acceptance.py

# Single section
uv run python scripts/run_smim_acceptance.py --section pipeline
```

### IDTxl dependency (R-TE-1)

```bash
# Not on PyPI — install once from GitHub
uv pip install "idtxl @ git+https://github.com/pwollstadt/IDTxl.git"
# Requires Java (JDK 11+): java -version
# JPype1 and setuptools are already in [dev] extras
```

## Current Status

Last updated: 2026-03-19

| WP | Gate | Status |
|----|------|--------|
| WP0 | G0 | ✅ Complete |
| WP1 | G1 | 🔨 Partial (ADAPTER_GUIDE done; adapters M1.2-T2..T7 not started) |
| WP2 | G2 | ✅ Complete |
| WP3 | G3 | ✅ Complete |
| WP4 | G4 | ✅ Complete |
| WP5 | G5 | ✅ Complete |
| WP6 | G6 | ✅ Complete |
| AT  | —  | ✅ 119/119 acceptance tests pass |

See `docs/smim/TASK_REGISTRY.md` for detailed per-task status.
