# P6-v2 statistical analysis plan

## Status and inheritance

This plan is frozen after a validation-only P6-v1 design audit and before any
ID-test or OOD label release. It is therefore validation-informed, not
outcome-blind. The held-out populations and inferential procedures inherit
from `analysis/statistical_analysis_plan.md` at its frozen SHA-256, except for
the explicit v2 replacements below. When this file and the inherited plan
conflict, this file controls.

The inherited plan is bound to
`83db461e069bf43f41d5927ec3477053bf0b8ff806fc7f8605a6368413afe9f7`.
Changing that file requires a new, explicitly versioned v2 statistical plan;
an unrecorded edit is not inherited.

P6-v1 is excluded from every v2 analysis population. Its results may appear
only in the deviation/audit appendix and runtime planning. No v1 checkpoint is
reselected or evaluated on held-out labels, and v1 and v2 values are never
pooled.

## Analysis population and units

The primary comparison remains P (`cdcureno_full`) versus B7
(`generic_causal_transfer`) at target budget 32 on the 40-case frozen
`combined_ood` regime. The v2 target seeds are the ordered list
`[100,101,102,103,104]`. Complete simulation cases and target-training seeds
are the experimental units; cells and time samples are repeated measurements.
The design is crossed and paired by target seed and case.

Only run IDs beginning with `p6v2-` and present in the committed v2
confirmatory roster are eligible. The population filter rejects every `p6-`
v1 ID, development run, unregistered retry, and checkpoint whose protocol,
metric, schedule, or implementation hash differs from the v2 pre-label lock.
A7 is an alias and never contributes duplicate observations.

## Primary endpoint and estimands

The held-out primary endpoint remains the per-case composite temperature
relative L2 in physical kelvin with float64 time/cell quadrature, exactly as
defined in the inherited plan. This primary inferential endpoint is distinct
from the multi-field validation-only checkpoint objective in
`analysis/p6_v2_protocol.md`.

The primary point estimand remains:

```text
I_mean = 1 - mean_seed,case(error_P) / mean_seed,case(error_B7)
```

The robustness estimand remains the median over cases of the relative
improvement after averaging over seeds within case. The four registered
support conditions remain unchanged:

1. the 95% crossed hierarchical-bootstrap interval for `I_mean` has a
   strictly positive lower endpoint;
2. the case-median relative improvement is at least 0.10;
3. the P/B7 crossed-mean ratio is at most 1.10 for peak-temperature absolute
   error, final-alpha MAE, and dimensionless conservative residual; and
4. P and B7 have zero nonfinite, alpha-bound, and alpha-monotonicity
   violations in the primary prediction population.

Negative, null, or guardrail-failing outcomes are reported under the same
rules.

## Bootstrap and exact tests

Use exactly 10,000 crossed paired bootstrap replicates with NumPy PCG64 seed
`20260725`. RNG draw order, Cartesian seed/case resampling, linear percentile
intervals, stored draw matrices, and distribution outputs are unchanged from
the inherited plan, except that the seed dimension maps to the ordered v2
seed list.

The exact paired sign-flip family and Holm procedure are unchanged in form.
The 16-row confirmatory family is rebuilt exclusively from v2 run IDs and the
committed v2 frozen-results table. Supplemental B8--B10, staged-transfer, and
focused-ablation analyses are separate labeled families and cannot change the
primary decision.

## Selection and missingness

Checkpoint selection uses only the 32 validation cases and the canonical
multi-field float64 objective registered in the v2 protocol. Candidate step 0
is eligible. Every run has exactly 121 candidate opportunities, and exact
ties choose the smaller global step. Held-out metrics never participate.

A missing or nonfinite required seed-case primary cell fails the P6-v2
completeness gate. It is not dropped, imputed, winsorized, or replaced by a
favorable rerun. Infrastructure interruption may resume only from a
hash-consistent v2 checkpoint/journal state that proves no optimizer update or
candidate transaction was repeated. A numerical/scientific terminal failure
remains in the roster.

Truth-relative validation gradient diagnostics that are undefined on flat
truth remain null and never enter selection or missing-primary accounting.
The fixed-scale gradient endpoints are always expected to be finite.

## Data efficiency and compute

Report B7 and P at all six label budgets and the trapezoidal area under error
versus `log2(label budget)` on `combined_ood`. No budget or split is selected
after release.

Compute accounting uses global optimizer steps and device-synchronized epoch
durations. For a 120-epoch run with effective batch size 4, the required
optimizer-step total is `120 * label_budget / 4`. Candidate validation time is
reported separately because all common runs have exactly 121 selection
evaluations.

The fixed source cost remains conditional on the frozen P3 causal source
checkpoint. The source-amortization denominator for the 89-run core remains
81: the 60 B7/P runs, three B6 runs, and 18 A1--A6 runs use the source; B5 and
A0 are the eight source-free runs. A7 adds no compute. Supplemental rosters
use their own declared amortization denominators and are not folded into the
core denominator post hoc.

## Release ordering and supplemental families

All paper-relevant supplemental training protocols and rosters are frozen
before Stage 1. Their validation-only checkpoint selection may run before
release. Stage 1 evaluates the complete locked set on ID-test, then is
committed. Stage 2 evaluates the registered OOD regimes only after the Stage 1
commit.

B8--B10 comparisons, staged transfer, and focused secondary ablations are
reported as separate predeclared secondary families. They do not alter the
five-seed B7/P primary hypothesis. Any model or contrast designed after Stage
1 is explicitly post-label exploratory or is evaluated on a new untouched
manifest.

## Required v2 outputs

The atomic statistics bundle must contain, at minimum:

- the immutable v2 per-case frozen-results table and provenance graph;
- the machine-readable primary decision;
- full bootstrap distributions and exact seed/case draw matrices;
- the exact-test/Holm family;
- roster completeness and failure accounting;
- label-efficiency points and AUC;
- tail, violation, censoring/attainment, and OOD-regime summaries;
- one row per unique training run for compute accounting;
- raw and source-amortized time with candidate-validation time separated;
- exact input paths, bytes, and SHA-256 hashes;
- a population audit proving that no v1 ID was included; and
- a canonical bundle manifest plus an atomic current-version pointer.

All manuscript tables and numbers consume this frozen bundle rather than
recomputing metrics ad hoc.
