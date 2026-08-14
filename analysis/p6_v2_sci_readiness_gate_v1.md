# P6-v2 SCI-readiness Gate v1

## Decision first

The present state is **`PROMISING_INCOMPLETE`**, and the execution decision is
**`GO_CONTINUE`**. This means that the design is strong enough to justify
finishing the already frozen experiment, not that an SCI-level empirical claim
has been demonstrated.

Only the 11 pending rows of the existing 24-row supplemental-v1 roster may be
resumed. This Gate does not authorize a new method, seed, case, retry,
threshold, endpoint, or held-out access. Immediately before launch, the
manager must recheck that no Python worker/validator or supplemental launcher
is live and that the frozen execution closure still matches its authority.

The machine-readable authority is
`analysis/p6_v2_sci_readiness_gate_v1.json`.

## Why this is a meta-gate

The primary analysis was frozen before held-out release. Therefore this Gate
must not redefine `core_primary.supportive`, create another confirmatory test
family, or convert a descriptive result into a confirmatory one. It is an
outcome-blind editorial and program-management layer that decides which claims
and manuscript path are justified after the registered analysis has run.

The Gate follows an outcome/driver/guardrail hierarchy:

1. Primary outcomes: central confirmatory evidence, mechanism/comparator
   sufficiency, and external/novelty/reproduction readiness.
2. Supporting drivers: the six-budget curve, registered ablations, per-OOD
   and tail behavior, source-seed sensitivity, and external compatibility.
3. Guardrails: complete populations, no leakage, physical/numerical safety,
   no post-hoc retry, and exact claim lineage.

No numerical publication probability is assigned. Journal acceptance depends
on venue and review and has no defensible denominator here; categorical states
are auditable and actionable.

## Gate A — permission to finish the frozen experiment

Gate A passes only when the protocol, statistical plans, run population, and
comparator families were frozen before held-out access; the target labels
remain sealed; any recovery is outcome-blind and preserves failed evidence;
and the exact campaign can resume under one writer.

At the assessment boundary based on commit
`32d2bb7292039d93b5fabf4536defbf703140576`, V7 recovery event
`2db63fa6bfddeb717f0ae9a713072b95750a66333f8fe9ce4896bf99cb5fc8fd`
left 13 rows complete, 11 pending, no active run, and no live Python process.
The recovery used no target test/OOD label and performed no scientific or GPU
re-execution. Gate A therefore returns `GO_CONTINUE`, subject to the immediate
pre-launch liveness and closure recheck.

## G0 — integrity and reproduction hard gate

Interpretation stops if any G0 item fails. G0 requires:

- registered terminal outcomes for all 89 core and 24 supplemental target
  rows, with the supplemental postflight passing;
- committed and tracked-clean Stage-1 and Stage-2 release receipts;
- the statistics execution gate passing for the exact 114-predictor,
  25,536-case-row released population;
- no dropped, imputed, replaced, or selectively retried primary cell;
- all required scientific and leakage tests passing; and
- a fresh clean-CPU rebuild reproducing every bound paper payload hash.

A G0 failure yields `BLOCKED_INVALID`; only integrity repair is allowed.

## G1 — central confirmatory evidence

The registered primary remains budget 32, five target seeds 100--104, 40
`combined_ood` cases, P (`cdcureno_full`) versus B7
(`generic_causal_transfer`), using complete-case composite-temperature
relative L2 in physical K.

`core_primary.supportive` must be true, which already requires all of:

- crossed-bootstrap 95% lower endpoint above zero;
- case-median relative improvement of at least 10%;
- P/B7 crossed-mean ratios no greater than 1.10 for peak-temperature absolute
  error, final-alpha MAE, and conservative energy residual; and
- zero nonfinite, alpha-bound, and alpha-monotonicity violations for P and B7.

For an SCI-readiness superiority statement, the separately registered exact
two-sided paired sign-flip p-value must also be below 0.05 and standardized
effect sizes must be finite. These are corroborating editorial requirements;
they do not change the frozen definition of `core_primary.supportive`.

A data-efficiency or label-saving title additionally requires P's frozen
six-budget log2-budget AUC to be lower than B7's. Without that result, the
paper must not use a label-saving headline even if the budget-32 primary is
supportive.

## G2 — mechanism, robustness, and comparator coverage

All registered B5/B6/B7 and supplemental B8/B9/B10 comparisons, plus A0--A6
and the A7=P alias, must be complete and reported. B8--B10 are clean-room
inspired controls and must not be described as exact reproductions.

Every retained component claim must have both its structural evidence and a
directionally supportive registered, multiplicity-controlled contrast. Exact
inflation/restriction, RVS/LIS, and CVS evidence governs restriction and
causality wording. Conservative-physics evidence must not break the frozen
field guardrails. Unsupported component claims are removed while the
full-system comparison may remain.

Per-split and tail results must be shown. The frozen 3-by-3 source-seed study
remains descriptive and cannot support a population-level seed-invariance
claim.

## G3 — external generalizability

- A compatible prelabel quantitative external bundle that passes its contract
  yields `PASS_VALIDATED`.
- Frozen context-only external evidence yields
  `PASS_COMPUTATIONAL_SCOPE_ONLY`; the paper is limited to a numerical or
  computational-methods scope and states the validation limitation.
- Missing external evidence or post-hoc compatibility reclassification fails
  G3.

External evidence cannot alter the core primary decision.

## G4 — novelty, claims, and submission package

Immediately before manuscript freeze/submission, the novelty matrix must be
refreshed against the closest current work. Novelty is limited to the combined
restriction-preserving, causal, thermochemically conservative transfer
framework. “First” claims remain forbidden.

Every manuscript number and panel must map to exact run IDs and lineage
sidecars; baseline fidelity, licenses, limitations, and negative results must
be explicit; and every claim must stay inside the permissions granted by
G1--G3.

## Branches that adjust the research goal

- G0 fails: stop analysis/manuscript work and repair integrity only.
- G0 passes but G1 fails: preserve the negative result; pivot to a supported
  physical-validity claim or register a new protocol on untouched data. No
  selective rerun is allowed.
- G0/G1 pass but G2 is partial: narrow to full-system performance and remove
  unsupported mechanism claims.
- G0/G1/G2/G4 pass with context-only external evidence: pursue a
  `CORE_COMPUTATIONAL_SCI_CANDIDATE` manuscript with explicit limits.
- G0--G4 pass with compatible quantitative external evidence: pursue a
  `STRONG_SCI_CANDIDATE` manuscript. This is an evidence classification, not
  a guarantee of journal acceptance.

## Current next action

Recheck the idle single-writer boundary and frozen closure, then resume the
exact supplemental roster at row index 13. After all 11 rows terminate, run
the strict supplemental postflight before any held-out release.
