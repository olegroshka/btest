# Paper Review Prompt — Round 5 (Publication-Readiness)

Paste the full LaTeX source of `smim_paper.tex` after this prompt.

---

## Context

This manuscript has undergone four prior revision cycles with twelve
independent referee reports across JBES, JFE, and JoE. The most recent
round (Round 4) produced three verdicts of **Accept** or **Accept with
cosmetic revisions** from senior associate editors at Gemini, GPT, and
Opus. All numerical audits passed. All prior blocking issues have been
resolved.

This is the **final review before submission**. The purpose is not to
find new conceptual objections or request new experiments. The purpose
is to catch anything — however small — that would embarrass the author
or the journal after publication. Typos, orphaned references, prose
that contradicts a table, a hedge in one section that becomes an
assertion in another, a missing minus sign, a figure caption that
describes a different figure.

---

## Your role

You are three independent referees performing a final acceptance check.
Each referee reads the paper with the assumption that it WILL be
published, and asks: **"Is there anything in this paper that I would
be embarrassed to have in print under my editorial watch?"**

You are looking for:

**Category A — Errors of fact.** A number in the text that does not
match the corresponding table. A CI that is reported as excluding zero
but the bounds shown include it. A claim about "all 10 windows" when
the table shows 9. A K value cited without specification when two K
values are used. A figure caption describing content that doesn't
match the figure.

**Category B — Broken references.** A cross-reference to a table,
figure, appendix, or equation that does not exist or points to the
wrong object. An "Appendix X" that has no corresponding section. A
"Table Y" that was renumbered but the reference was not updated.

**Category C — Prose–evidence mismatches.** A sentence that asserts X
where the evidence shows "X with qualification." A hedge in Section 4
that becomes an unqualified assertion in the conclusion. A "robust
across three panels" when the result is positive on one and null on
two.

**Category D — Orphaned content.** A section that references material
that was deleted in revision. A footnote that no longer applies. A
limitation listed that has since been addressed. A "future work" item
that was actually done.

**Category E — Presentation.** Grammar, spelling, inconsistent
notation, unclear antecedents, sentences that are ambiguous about
which model is being discussed, paragraphs that could be misread.

You are NOT looking for:
- New experiments or baselines
- Fundamental reconceptualisation
- Scope expansion
- Economic significance (the paper explicitly positions as methods)
- Theory or asymptotics (the paper explicitly disclaims these)

---

## Report structure

Each referee writes a short report (max 1 page) with:

### Recommendation
Accept / Accept with corrections / Minor revision

### Findings
A numbered list of specific items found, categorised A–E. For each:
quote the exact text, state the problem, state the fix. If none
found, state "No issues found."

### Overall assessment
One paragraph: is this paper ready for publication in JBES?

---

## Referee assignments

### Referee 1 — The Proofreader
Focus on Categories A and B. Your job is to verify every number,
every cross-reference, every table/figure pointer. Read every table
caption and check it against the table contents. Read every CI and
check the bounds. Count table rows and match against captions.
Verify that every "\ref{}" resolves to the correct object.

### Referee 2 — The Consistency Checker
Focus on Categories C and D. Your job is to read the abstract, then
the conclusion, then the body, and check that no claim is stronger
in the abstract/conclusion than the evidence in the body supports.
Check that every hedge is propagated. Check that no deleted material
is still referenced. Check that limitations match what was actually
tested.

### Referee 3 — The Reader
Focus on Category E. Your job is to read the paper as a first-time
reader would, noting any sentence that is confusing, any paragraph
where the referent of "this" is unclear, any place where notation
switches without explanation, any figure that is hard to read. You
are the "fresh eyes" check.

---

## Synthesis

After all three reports, add:

### Publication-ready?
Yes / Yes with corrections / No

If corrections are needed, list them with estimated fix time
(minutes, not hours). If the total fix time exceeds 2 hours, the
paper needs another revision cycle. If it is under 30 minutes,
accept with corrections.
