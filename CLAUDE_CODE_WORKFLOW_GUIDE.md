# Claude Code Workflow Guide for SMIM Development

## How to Provide Context to Claude Code

### Layer 1: Automatic (read every session)

Claude Code automatically reads `CLAUDE.md` files hierarchically:

```
~/.claude/CLAUDE.md          ← personal global preferences (optional)
btest/CLAUDE.md              ← project root: commands, style, architecture rules
btest/src/.../smim/CLAUDE.md ← SMIM-specific: notation, assumptions, status
```

**These files are your persistent memory.** Keep them lean (<150 lines each).
Don't inline specs — point to files.

### Layer 2: On-Demand Reference (Claude reads when needed)

These live in `docs/smim/` and Claude fetches them itself:

```
docs/smim/
├── PROPOSAL_SUMMARY.md     # Condensed math architecture (read for "why" questions)
├── IMPLEMENTATION_PLAN.md  # Milestones, gates, acceptance criteria
├── TASK_REGISTRY.md        # Per-task decomposition with status tracking
├── ADAPTER_GUIDE.md        # How to write a new data adapter (created in M1.2-T1)
└── DECISIONS.md            # Architectural decisions log (append after each gate)
```

**Claude reads these on demand.** Your CLAUDE.md tells it they exist and when to read them.
You never paste these into the prompt.

### Layer 3: Session Prompt (you provide at session start)

The prompt you type into Claude Code. This is task-specific and focused.
Structure it as: **what to do, where to do it, what tests to pass.**

### Layer 4: The Code Itself (Claude explores)

Claude Code reads source files, runs tests, checks types. Let it explore.
Don't paste code into prompts — tell Claude where to find it.

---

## File Placement in Your Repository

You have 7 files to place. Here is exactly what each one is:

| File you received | What it is | Where it goes |
|---|---|---|
| `CLAUDE.md` | Root project context — commands, style, architecture | `btest/CLAUDE.md` |
| `smim_CLAUDE.md` | SMIM-specific context — notation, assumptions, status | `btest/src/quantdsl_backtest/smim/CLAUDE.md` (rename to `CLAUDE.md`) |
| `smim_interfaces.py` | Protocol definitions — contracts for all components | `btest/src/quantdsl_backtest/smim/interfaces.py` |
| `smim_config.py` | Pydantic config models — parses experiment YAML | `btest/src/quantdsl_backtest/smim/config.py` |
| `docs_smim_PROPOSAL_SUMMARY.md` | Condensed research proposal — math architecture reference | `btest/docs/smim/PROPOSAL_SUMMARY.md` |
| `docs_smim_IMPLEMENTATION_PLAN.md` | Milestones, gates, acceptance criteria — operational plan | `btest/docs/smim/IMPLEMENTATION_PLAN.md` |
| `docs_smim_TASK_REGISTRY.md` | Claude Code task decomposition with status checkboxes | `btest/docs/smim/TASK_REGISTRY.md` |
| `docs_smim_DECISIONS.md` | Architectural decision log — append after each gate | `btest/docs/smim/DECISIONS.md` |
| `experiments_mvp_energy_us_uk.yaml` | Sample experiment config for the first build | `btest/experiments/mvp_energy_us_uk.yaml` |
| `CLAUDE_CODE_WORKFLOW_GUIDE.md` | This guide — for YOU, not for the repo | Keep for your own reference |

The original `implementation_plan.pdf` (29-page LaTeX) is the authoritative human-readable
plan. `IMPLEMENTATION_PLAN.md` is its markdown distillation — the same milestones, gates,
and acceptance criteria in a format Claude Code can read efficiently mid-session.

The `implementation_plan_adjustments.md` was an analysis document. Its recommendations are
already folded into the files above — you don't need to put it in the repo.

```
btest/                              ← project root
├── CLAUDE.md                       ← ROOT: project conventions (PROVIDED FILE)
├── docs/
│   └── smim/
│       ├── PROPOSAL_SUMMARY.md     ← condensed proposal (PROVIDED FILE)
│       ├── IMPLEMENTATION_PLAN.md  ← milestones, gates, criteria (PROVIDED FILE)
│       ├── TASK_REGISTRY.md        ← task decomposition (PROVIDED FILE)
│       ├── DECISIONS.md            ← architectural decision log (PROVIDED FILE)
│       └── ADAPTER_GUIDE.md        ← written during M1.2-T1 (not yet created)
├── experiments/
│   └── mvp_energy_us_uk.yaml      ← sample experiment config (PROVIDED FILE)
├── experiments/
│   └── mvp_energy_us_uk.yaml      ← experiment config (created in M0.5)
├── src/quantdsl_backtest/
│   └── smim/
│       ├── CLAUDE.md               ← SMIM-specific context (PROVIDED FILE)
│       ├── __init__.py
│       ├── interfaces.py           ← Protocol definitions (PROVIDED FILE)
│       ├── config.py               ← Pydantic config models
│       ├── data/ ...
│       ├── graph/ ...
│       ├── spectral/ ...
│       ├── dynamics/ ...
│       ├── emergence/ ...
│       ├── gaps/ ...
│       ├── validation/ ...
│       ├── signals/ ...
│       └── pipeline.py
├── tests/unit/smim/ ...            ← unit tests
└── tests_slow/smim/ ...            ← integration tests
```

---

## Session Workflow: Step by Step

### Before Your First Session Ever

1. Place the provided files in your repo (see file placement table above):
   - `CLAUDE.md` → `btest/CLAUDE.md`
   - `smim_CLAUDE.md` → `btest/src/quantdsl_backtest/smim/CLAUDE.md`
   - `smim_interfaces.py` → `btest/src/quantdsl_backtest/smim/interfaces.py`
   - `smim_config.py` → `btest/src/quantdsl_backtest/smim/config.py`
   - `docs_smim_PROPOSAL_SUMMARY.md` → `btest/docs/smim/PROPOSAL_SUMMARY.md`
   - `docs_smim_IMPLEMENTATION_PLAN.md` → `btest/docs/smim/IMPLEMENTATION_PLAN.md`
   - `docs_smim_TASK_REGISTRY.md` → `btest/docs/smim/TASK_REGISTRY.md`
   - `docs_smim_DECISIONS.md` → `btest/docs/smim/DECISIONS.md`
   - `experiments_mvp_energy_us_uk.yaml` → `btest/experiments/mvp_energy_us_uk.yaml`

2. Commit these files:
   ```bash
   git add -A
   git commit -m "[SMIM M0.0] Bootstrap SMIM package structure and context files"
   ```

3. Your very first Claude Code session should be the bootstrap (M0.0):
   ```
   Read the SMIM context in src/quantdsl_backtest/smim/CLAUDE.md and the interfaces
   in smim/interfaces.py. Then create the full package skeleton:
   - __init__.py for smim/ and every subpackage (data/, graph/, spectral/, etc.)
   - config.py with Pydantic models matching the experiment YAML schema
   - Stub test files in tests/unit/smim/ mirroring the source structure
   Make sure `uv run pytest -q` still passes after your changes.
   ```

### Before Each Session

1. **Pick one task** from `docs/smim/TASK_REGISTRY.md`
2. **Create a branch**: `git checkout -b smim/m2.1-t3-narrative-edges`
3. **Open Claude Code** in the project root

### During Each Session

1. **Start with the task prompt** (see templates below)
2. **Use Plan Mode first** (Shift+Tab twice) for any task touching >3 files
3. **Review the plan** before letting Claude execute
4. **Let Claude run tests** after implementation
5. **Check**: `uv run pytest -q` — all existing tests must still pass
6. **Check**: `uv run mypy src/quantdsl_backtest/smim/` — no type errors

### After Each Session

1. **Commit**: `git add -A && git commit -m "[SMIM M2.1-T3] Implement NarrativeEdgeEstimator"`
2. **Update status** in `docs/smim/TASK_REGISTRY.md` (mark task ✅)
3. **Update CLAUDE.md** in `smim/` if the current status section changed
4. **Merge** to main when all tasks in a milestone are done

---

## Session Prompt Templates

### Template A: Implement a Protocol

Use for: any task that implements an interface from `interfaces.py`.

```
Implement [ComponentName] following the [ProtocolName] protocol
in smim/interfaces.py.

File: src/quantdsl_backtest/smim/[path]/[filename].py
Tests: tests/unit/smim/[path]/test_[filename].py

Requirements:
- [Specific requirement 1]
- [Specific requirement 2]
- Config read from SmimConfig.[section] in smim/config.py

Acceptance:
- Tests pass
- mypy clean
- [Specific acceptance criterion]
```

### Template B: Add a Data Adapter

Use for: M1.2-T2 through M1.2-T7.

```
Create a new data adapter for [SOURCE_NAME] following btest's adapter pattern.

First read the existing FRED adapter to understand the interface. Then read
docs/smim/ADAPTER_GUIDE.md if it exists.

File: src/quantdsl_backtest/smim/data/adapters/[source].py
Tests: tests/unit/smim/data/test_[source]_adapter.py

The adapter must:
- Fetch [specific data types] from [API endpoint]
- Return pd.DataFrame with columns: [list columns]
- Support as_of parameter for point-in-time retrieval (A1)
- Use cached test fixtures in tests (never hit live API in tests)
- Handle rate limiting gracefully

Config section in smim/config.py: SmimConfig.data.[source]
```

### Template C: Mathematical/Numerical Implementation

Use for: spectral decomposition, filtering, PID, TDA tasks.

```
Implement [mathematical operation] in smim/[module]/[file].py.

Mathematical reference:
- Proposal equation: [equation number or brief formula]
- Read docs/smim/PROPOSAL_SUMMARY.md section "[section]" for context

File: src/quantdsl_backtest/smim/[path]/[filename].py
Tests: tests/unit/smim/[path]/test_[filename].py

Requirements:
- Input: [describe input arrays and shapes]
- Output: [describe output type, referencing dataclass from interfaces.py]
- Must handle: [edge cases — singular matrices, empty regimes, etc.]

Test strategy:
- Test against known analytical solution: [describe test case]
- Test shapes and types for random inputs
- Test edge case: [specific edge case]

Dependencies: [scipy.linalg.schur / ripser / dit / etc.]
```

### Template D: Validation/Testing Task

Use for: WP5 falsification tests, evaluation harnesses.

```
Implement [test/evaluation name] in smim/validation/[file].py.

This implements [falsification test name] from the research proposal
Section 8.8. Read docs/smim/PROPOSAL_SUMMARY.md "Falsification Tests"
for the full list.

The test must:
- Generate B ≥ 100 null instances by [describe null model]
- Compute [test statistic] for observed data and each null instance
- Return FalsificationResult (from interfaces.py) with empirical p-value
- passes = True iff observed statistic > 95th percentile of null distribution

Tests: tests/unit/smim/validation/test_[file].py
- Test with synthetic data where ground truth is known
- Verify p-value is <0.05 when structure is real, >0.05 when random
```

---

## Example: Complete Session for M2.1-T2 (Granger Edge Estimator)

Here is exactly what you would type into Claude Code:

```
Implement GrangerEdgeEstimator following the EdgeEstimator protocol
in smim/interfaces.py.

File: src/quantdsl_backtest/smim/graph/edges/granger.py
Tests: tests/unit/smim/graph/test_granger_edges.py

The estimator:
1. Takes actor-level time series (pd.DataFrame, columns=actor_ids, index=dates)
2. For each directed pair (i→j), fits a bivariate VAR with BIC-selected
   lag order (max_lag from config, default 4)
3. Runs Granger causality F-test for i→j
4. Sets A[j,i] = F-statistic if p < threshold (default 0.05), else 0
5. Returns scipy.sparse.csr_matrix of shape (N, N)

Point-in-time: only uses data with dates ≤ date_range.end.

Config: SmimConfig.edges.granger.max_lag, SmimConfig.edges.granger.p_threshold

Acceptance:
- tests pass (including on synthetic VAR(1) data where true edges are known)
- handles N=10 actors, T=80 periods in <5 seconds
- returns sparse matrix with density < 30% on random data
- channel property returns RelationChannel.FINANCIAL (default, configurable)

Use statsmodels.tsa.api for VAR estimation and Granger causality testing.
```

---

## Managing Context Window

Claude Code has ~200k tokens. A fresh session costs ~20k for CLAUDE.md + baseline.
That leaves ~180k for your work.

**Signs you're running low:**
- Claude starts forgetting earlier decisions
- Warning notification about context window
- Responses become less precise

**What to do:**
- Use `/compact` with instructions about what to preserve
- Or start a fresh session for the next task (preferred)
- One task per session is the safest approach

**Things that eat context fast:**
- Large file reads (each source file Claude reads costs tokens)
- Long test output
- Multiple failed attempts

**Things that are context-efficient:**
- Focused prompts that name exact files
- Letting Claude run targeted tests, not the full suite
- Using Plan Mode to align before coding

---

## Gate Reviews

At each quality gate (G0–G6), create a dedicated session:

```
We are reviewing Gate [GX] for WP[X]. Read the gate criteria in
docs/smim/IMPLEMENTATION_PLAN.md section [X].

Run all relevant tests and checks. Then produce a gate review report
covering:
1. Which artefacts exist (list files)
2. Quantitative thresholds (compute and report each metric)
3. Standing assumption verification (which assumptions were checked, results)
4. Pass/fail recommendation with evidence

Write the report to docs/smim/gate_reviews/GX_review.md
```

---

## Maintaining the Feedback Loop

After every few sessions:
1. **Update TASK_REGISTRY.md** — mark completed tasks ✅
2. **Update smim/CLAUDE.md** status table — reflect current WP progress
3. **Append to DECISIONS.md** — record any architectural choices made during sessions
   (e.g., "chose polar decomposition over Schur because condition numbers were 10x better")
4. **Refine CLAUDE.md** — if Claude consistently needed reminding about something,
   add it to the relevant CLAUDE.md. If something is never relevant, remove it.

The CLAUDE.md files are living documents. They get better as you use them.
