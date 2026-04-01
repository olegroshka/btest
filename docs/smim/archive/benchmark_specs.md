# SMIM Benchmark Specifications

This document formally defines the benchmark families used to compute
investment gaps $\Delta_{i,t} = y_{i,t} - y^*_{i,t}$.

**Rule**: every `GapResult` object MUST carry a `BenchmarkClass` label.
The codebase enforces this via the non-optional field in `GapResult`.

---

## 1 · Predictive Benchmark (`BenchmarkClass.PREDICTIVE`)

### Formula

$$y^*_{\text{pred},i,t} = \mathbb{E}\bigl[y_{i,t} \,\bigm|\, \mathcal{F}_{t-1}\bigr]$$

where $\mathcal{F}_{t-1}$ is the filtration of all information up to (and including)
period $t-1$, strictly respecting point-in-time discipline (Assumption A1).

In practice this is the **one-step-ahead prediction** from the state-space model:

$$y^*_{\text{pred},i,t} = \bigl(U_{K^*} \hat{\alpha}_{t|t-1}\bigr)_i + \bigl(B^{(z_{t-1})} x_{t-1}\bigr)_i$$

where:
- $U_{K^*}$ is the modal frame (`modal_frame.basis`, shape `(N, K*)`)
- $\hat{\alpha}_{t|t-1}$ is the Kim-filter one-step prediction (`filtered_state.alpha_predicted[t,:]`)
- $B^{(z)}$ is the local covariate loading matrix for regime $z$ (stored in `filtered_state.params`)
- $x_{t-1}$ are local covariates observed at $t-1$

### Inputs required

| Input | Shape | Source |
|---|---|---|
| `filtered_state.alpha_predicted` | `(T, K*)` | `smim/dynamics/` |
| `modal_frame.basis` | `(N, K*)` | `smim/spectral/` |
| `filtered_state.params["B"]` | `(M, N, d_x)` | `smim/dynamics/` |
| `filtered_state.regime_probs` | `(T, M)` | `smim/dynamics/` |
| `observations` | `(N, T)` | `smim/data/` |

### BenchmarkClass label

`BenchmarkClass.PREDICTIVE`

### Implementation target

`smim/gaps/predictive.py` → class `PredictiveBenchmark`

---

## 2 · Structural Benchmark (`BenchmarkClass.STRUCTURAL`)

### Formula

The structural benchmark removes the contribution of "distortionary" channels
(identified empirically as those with above-median deviation from the stable
long-run average):

$$y^*_{\text{str},i,t} = y^*_{\text{pred},i,t}
  - \sum_{r \in \mathcal{D}} \omega_{r,z_t}
    \bigl(A_t^{(r)} \hat{\alpha}_{t|t-1}\bigr)_i$$

where $\mathcal{D}$ is the set of channels classified as distortionary
(i.e., $\|A^{(r)}_t - \bar{A}^{(r)}\|_F > \text{threshold}$).

In Phase I the structural benchmark is approximated by constructing a
"stable-channel operator":

$$A_t^{\text{stable}} = \sum_{r \notin \mathcal{D}} \omega_{r,z_t} A_t^{(r)}$$

and re-running the prediction with $A_t^{\text{stable}}$ substituted for $A_t$.

### Inputs required

| Input | Shape | Source |
|---|---|---|
| All predictive benchmark inputs | — | see above |
| Per-channel adjacency matrices `adj_channel[r]` | sparse `(N, N)` each | `smim/graph/edges/` |
| Channel classification `channel_stable_mask` | `(R,)` bool | `smim/graph/operators.py` |

### BenchmarkClass label

`BenchmarkClass.STRUCTURAL`

### Implementation target

`smim/gaps/structural.py` → class `StructuralBenchmark`
(Phase I approximation; exact formula deferred to WP6 / M6.1)

---

## 3 · Modal Benchmark (`BenchmarkClass.MODAL`)

### Formula

The modal benchmark decomposes the gap into per-mode contributions:

$$\Delta^{\text{mode}}_{k,i,t} = u_{k,i} \cdot \alpha_{k,t} - u_{k,i} \cdot \hat{\alpha}_{k,t|t-1}$$

where $u_{k,i}$ is the $i$-th entry of the $k$-th eigenmode.

The aggregate modal benchmark is:

$$y^*_{\text{mode},i,t} = y_{i,t} - \sum_k \Delta^{\text{mode}}_{k,i,t}$$

and the `modal_attribution` field of `GapResult` carries the shape `(N, T, K*)` array.

### Inputs required

| Input | Shape | Source |
|---|---|---|
| `modal_frame.basis` | `(N, K*)` | `smim/spectral/` |
| `filtered_state.alpha_filtered` | `(T, K*)` | `smim/dynamics/` |
| `filtered_state.alpha_predicted` | `(T, K*)` | `smim/dynamics/` |

### BenchmarkClass label

`BenchmarkClass.MODAL`

### Implementation target

`smim/gaps/modal.py` → class `ModalBenchmark`

---

## 4 · Equilibrium Benchmark (`BenchmarkClass.EQUILIBRIUM`)

**Phase III only** (WP6+). Requires game-theoretic equilibrium computation.
Formally:

$$y^*_{\text{eq},i,t} = \arg\max_{y_i} \; U_i(y_i, y_{-i,t})$$

where $U_i$ is actor $i$'s utility function estimated from revealed preferences.

**Status**: deferred. Placeholder in `BenchmarkFactory` raises `NotImplementedError`.

---

## 5 · Emergence-Aware Benchmark (`BenchmarkClass.EMERGENCE_AWARE`)

### Formula

Adds synergistic corrections and the criticality index:

$$y^*_{\text{em},i,t} = y^*_{\text{pred},i,t}
  + \lambda_S \sum_{j,k} S_{jk} \cdot u_{j,i} \cdot u_{k,i} \cdot \alpha_{j,t} \alpha_{k,t}
  + \lambda_C \cdot C_t \cdot \sigma_i$$

where:
- $S_{jk}$ is the PID synergy matrix (`synergy_matrix`, shape `(K*, K*)`)
- $C_t$ is the criticality index (`criticality`, scalar)
- $\sigma_i$ is the historical std of $y_{i,t}$
- $\lambda_S, \lambda_C$ are calibrated shrinkage parameters from `EmergenceConfig`

### Inputs required

| Input | Shape | Source |
|---|---|---|
| All predictive benchmark inputs | — | see §1 |
| `synergy_matrix` | `(K*, K*)` | `smim/emergence/pid.py` |
| `criticality` | scalar per $t$ | `smim/dynamics/phase_transition.py` |
| `topological_complexity` | `(T,)` | `smim/emergence/` |

### BenchmarkClass label

`BenchmarkClass.EMERGENCE_AWARE`

### Implementation target

`smim/gaps/emergence_aware.py` → class `EmergenceAwareBenchmark`
(depends on M4.5, M4.6, M4.7 being complete)

---

## Implementation Notes

All benchmarks are registered in `smim/gaps/__init__.py` via `BenchmarkFactory`.
During WP0 and WP1 every class is a placeholder that raises `NotImplementedError`.
Implementations are added per milestone:

| Milestone | Benchmark |
|---|---|
| M4.4 | `PredictiveBenchmark`, `ModalBenchmark` |
| M6.1 | `StructuralBenchmark` (full) |
| M4.8 | `EmergenceAwareBenchmark` |
| Phase III | `EquilibriumBenchmark` |
