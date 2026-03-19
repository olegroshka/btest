# SMIM — Spectral Multi-layer Investment Misallocation

SMIM estimates actor-specific investment gaps by building a directed multilayer
graph, applying spectral decomposition, running state-space filtering with
regime switching, computing emergence diagnostics, and finally computing
investment gap benchmarks.

This is a **research framework**, not a trading strategy. It lives as a
subpackage of `quantdsl_backtest` and shares its data infrastructure
(ArcticDB caching, FRED/parquet adapters). The connection to the backtesting
engine happens only through bridge signals in `smim/signals/`.

---

## Package Layout

```
src/quantdsl_backtest/smim/
├── interfaces.py        # All Protocols, dataclasses, enums (read first)
├── config.py            # Pydantic config models (SmimConfig)
├── data/                # Data adapters and actor registry
├── graph/               # Edge estimators, operators, sparsification, null models
│   └── edges/           # Granger, narrative, cosine-similarity estimators
├── spectral/            # Schur / polar / Hermitian decomposers, DMD, mode selection
├── dynamics/            # KalmanFilter, KimFilter, regime selection
├── emergence/           # PID, transfer entropy, criticality, TDA, phase transition
├── gaps/                # PredictiveBenchmark, ModalBenchmark, EmergenceAwareBenchmark
├── signals/             # Bridge signals: SMIM gaps → btest DSL
└── validation/          # Metrics, falsification tests
```

**Pipeline data flow:**

```
observations (N,T)
  └─ graph/     → directed multilayer adjacency A_t^(r)
  └─ graph/     → combined operator A_t (N,N sparse)
  └─ spectral/  → modal frame (U: N×K, eigenvalues: K)
  └─ spectral/  → mode selection K* via MDL
  └─ dynamics/  → KalmanFilter / KimFilter → FilteredState (alpha_t, regimes)
  └─ dynamics/  → regime count M* via BIC
  └─ emergence/ → PID synergy matrix S, criticality C_t, TDA complexity T_t
  └─ gaps/      → GapResult (gaps N×T, benchmarks N×T, BenchmarkClass)
  └─ signals/   → bridge to btest engine
```

---

## Mathematical Notation → Python

| Math | Python variable | Shape | Module |
|------|----------------|-------|--------|
| y_{i,t} | `intensities` | (N, T) | `data/` |
| A_t^{(r)} | `adj_channel` | sparse (N, N) | `graph/edges/` |
| A_t | `operator` | sparse (N, N) | `graph/operators.py` |
| U_t | `modal_frame.basis` | (N, K) | `spectral/` |
| α_t | `alpha_t` | (K,) or (T, K) | `dynamics/` |
| F^{(z)} | `transition_matrices[z]` | (K, K) | `dynamics/` |
| z_t | `regime_labels` | (T,) int | `dynamics/` |
| S_{jk} | `synergy_matrix` | (K, K) | `emergence/pid.py` |
| Δ_{i,t} | `gap_result.gaps` | (N, T) | `gaps/` |
| y*_{i,t} | `gap_result.benchmarks` | (N, T) | `gaps/` |

---

## Setup

```bash
# Standard dev install (all extras)
uv sync --extra dev --extra platform

# IDTxl — not on PyPI, install manually (needed for acceptance test R-TE-1)
uv pip install "idtxl @ git+https://github.com/pwollstadt/IDTxl.git"
```

**IDTxl requires Java.** Verify with `java -version` (JDK 11+ required).
JPype1 and setuptools (both in `[dev]` extras) are installed by `uv sync`.
JVM module-restriction warnings on Java 17+ are non-fatal and can be ignored.

---

## Running Tests

### Unit tests (fast, ~4 s)

```bash
# All SMIM unit tests
uv run pytest tests/unit/smim/ -q

# Single module
uv run pytest tests/unit/smim/dynamics/ -q
uv run pytest tests/unit/smim/test_config.py -q
```

### Acceptance tests (~60 s)

Use the dedicated runner that prints the gate report:

```bash
# Full suite
uv run python scripts/run_smim_acceptance.py

# Verbose (show each test name)
uv run python scripts/run_smim_acceptance.py -v

# Single section
uv run python scripts/run_smim_acceptance.py --section graph
uv run python scripts/run_smim_acceptance.py --section spectral
uv run python scripts/run_smim_acceptance.py --section kalman
uv run python scripts/run_smim_acceptance.py --section kim
uv run python scripts/run_smim_acceptance.py --section mode
uv run python scripts/run_smim_acceptance.py --section pid
uv run python scripts/run_smim_acceptance.py --section te
uv run python scripts/run_smim_acceptance.py --section tda
uv run python scripts/run_smim_acceptance.py --section benchmarks
uv run python scripts/run_smim_acceptance.py --section pipeline

# Stop on first failure
uv run python scripts/run_smim_acceptance.py -- -x

# Pass extra pytest args
uv run python scripts/run_smim_acceptance.py -- -k "P_1 or P_2" --tb=long
```

**Expected output (all installed):**

```
SMIM Acceptance Report — 2026-03-19
===========================================
Graph Construction:     20/20 passed ✅
Spectral Decomposition: 35/35 passed ✅
Mode Selection:          9/9  passed ✅
Kalman Filter + EM:     14/14 passed ✅
Observability:           3/3  passed ✅
Phase Transition:        8/8  passed ✅
PID:                     6/6  passed ✅
Transfer Entropy:        6/6  passed ✅
TDA:                     7/7  passed ✅
Benchmarks/Gaps:         7/7  passed ✅
Pipeline Sanity:         4/4  passed ✅
-------------------------------------------
TOTAL:                 119/119 passed ✅
STATUS: READY FOR EXPERIMENTS
```

> Without idtxl/Java: Transfer Entropy shows 5/6 (R-TE-1 skipped) and
> STATUS remains READY FOR EXPERIMENTS (skipped ≠ failed).

### Or run pytest directly

```bash
# Full acceptance suite with report
uv run pytest tests/acceptance/smim/ -v --tb=short

# By section (manual -k filters)
uv run pytest tests/acceptance/smim/ -v -k "A_GR or I_GR or D_GR"   # Graph
uv run pytest tests/acceptance/smim/ -v -k "A_SC or I_SC or R_SC"   # Schur
uv run pytest tests/acceptance/smim/ -v -k "A_KF or I_KF or A_EM"  # Kalman/EM
uv run pytest tests/acceptance/smim/ -v -k "A_TDA or I_TDA"         # TDA
uv run pytest tests/acceptance/smim/ -v -k "P_1 or P_2 or P_3 or P_4" # Pipeline
```

---

## Acceptance Gate

**Experiments MUST NOT start until all 119 tests pass.**

The gate report is generated automatically by
`tests/acceptance/smim/conftest_report.py` and printed at the end of every
pytest run targeting `tests/acceptance/smim/`.

Full acceptance test specification: `docs/smim/ACCEPTANCE_TESTS.md`.

---

## Key Reference Documents

| Document | Purpose |
|----------|---------|
| `docs/smim/CLAUDE.md` | Math notation, standing assumptions, implementation patterns |
| `docs/smim/ACCEPTANCE_TESTS.md` | Full test plan, pass criteria, known deviations |
| `docs/smim/IMPLEMENTATION_PLAN.md` | Milestones, quality gates, work packages |
| `docs/smim/TASK_REGISTRY.md` | Per-task status (Claude Code decomposition) |
| `docs/smim/PROPOSAL_SUMMARY.md` | Condensed research proposal |
| `docs/smim/notation.md` | Every mathematical symbol with Python mapping |
| `docs/smim/benchmark_specs.md` | Formal definitions for all 5 benchmark families |
| `smim/interfaces.py` | All Protocol definitions — read before implementing |
| `smim/config.py` | All tuneable parameters (Pydantic) |
| `experiments/mvp_energy_us_uk.yaml` | Sample experiment config |

---

## Standing Assumptions (NEVER VIOLATE)

| ID | Rule |
|----|------|
| A1 | Point-in-time: never use data with pub_date > backtest_date |
| A2 | Typed comparability: normalisation is per-ActorType |
| A3 | Sparse propagation: after sparsification, operator retains >80% spectral energy |
| A4 | Stable modes: eigenmode rank correlation >0.5 across ≥80% of rolling windows |
| A5 | Regime persistence: average regime duration >8 quarters |

Every `GapResult` must carry a `BenchmarkClass` — the field is non-optional.

---

## Implementation Pattern

Every SMIM component follows this pattern:

1. **Protocol** defined in `smim/interfaces.py`
2. **Implementation** in the appropriate submodule
3. **Config** as a Pydantic model section in `smim/config.py`
4. **Unit tests** in `tests/unit/smim/` mirroring the source path
5. **Acceptance tests** in `tests/acceptance/smim/`

```python
# Example: add a new edge estimator
# 1. Read EdgeEstimator protocol in interfaces.py
# 2. Create smim/graph/edges/my_estimator.py
# 3. Implement the protocol
# 4. Add config to SmimConfig in config.py
# 5. Add unit tests in tests/unit/smim/graph/test_my_estimator.py
# 6. Register in the edge estimator factory
```
