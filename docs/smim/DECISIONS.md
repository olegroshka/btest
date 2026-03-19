# SMIM Architectural Decision Log

This file records architectural decisions made during SMIM development.
Append a new entry after each gate or significant decision point.

Format: ADR-NNN, date, context → decision → consequences.

---

## ADR-001: BIC regime selection unreliable on short noise sequences

**Date**: 2026-03-19
**Gate**: G3 (state-space filtering complete) / acceptance test review

**Context**

Acceptance test P-2 (pure noise null) found that `select_regime_count` returns
M*=2 on pure iid Gaussian noise (K=1, T=150). The Kim filter gains thousands of
log-likelihood units by fitting heteroscedastic variance patterns in the noise,
far outweighing the BIC penalty (~25 units for 5 extra parameters × log(150)).
This is not a bug — BIC is correctly reporting that M=2 fits the data better
in-sample. The problem is that the fitted variance patterns are not generalisable.

**Decision**

BIC regime count M* is an **initial candidate**, not a final answer. The
experimental protocol requires:

> If regime switching improves in-sample BIC but does not improve OOS R² by
> ≥ 0.5 percentage points versus the M=1 baseline, regime switching is not
> justified for that condition.

The definitive null check is **OOS R² ≤ 0.1** (acceptance test P-2). All
experiments in Phase B that evaluate regime switching must report both BIC-M*
and OOS R² and only claim regime structure when OOS R² > 0.1.

**Consequences**

- Experiments B1 (component ablation) and B10 (regime sweep) must evaluate
  regime switching via OOS R², not BIC alone.
- The `select_regime_count` API is retained as-is; callers are responsible for
  validating M* with OOS metrics before committing to a regime count.
- P-2 acceptance test keeps OOS R² ≤ 0.1 as the primary criterion; the test
  does not assert M*=1 from BIC (see the existing note in the test docstring).
- Any future regime-selection improvement (e.g. a penalty schedule that scales
  with T) should be validated against P-2 before replacing the current BIC
  formulation.
