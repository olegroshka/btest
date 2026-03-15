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

```python
# Example: implementing a new edge estimator
# 1. Read the EdgeEstimator protocol in interfaces.py
# 2. Create smim/graph/edges/my_estimator.py
# 3. Implement the protocol
# 4. Add config to SmimConfig in config.py
# 5. Add tests in tests/unit/smim/graph/test_my_estimator.py
# 6. Register in the edge estimator factory
```

## Current Status

<!-- UPDATE THIS after each task completion -->
Last updated: [DATE]

| WP | Gate | Status |
|----|------|--------|
| WP0 | G0 | ☐ Not started |
| WP1 | G1 | ☐ Not started |
| WP2 | G2 | ☐ Not started |
| WP3 | G3 | ☐ Not started |
| WP4 | G4 | ☐ Not started |
| WP5 | G5 | ☐ Not started |
| WP6 | G6 | ☐ Not started |

See `docs/smim/TASK_REGISTRY.md` for detailed per-task status.
