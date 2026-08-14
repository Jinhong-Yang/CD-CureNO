# P6-v2 supplemental statistical analysis plan

## Scope and inheritance

This plan is frozen with `p6v2-supplement-v1` before the first P6-v2
held-out release. It inherits metric definitions, case order, float64
quadrature, missingness rules, exact paired sign-flip implementation, and
two-sided testing conventions from `analysis/statistical_analysis_plan.md`
and the P6-v2 replacement plan. It creates secondary families only and cannot
change the five-seed B7-versus-P primary decision.

All comparisons use the 40-case frozen `combined_ood` regime unless stated
otherwise. The experimental units are complete simulation cases and target
training seeds. Cells and times are repeated measurements. Only exact frozen
roster members and checkpoints in the P6-v2 pre-label ledger are eligible.
P6-v1, development, retries outside a roster, and unregistered checkpoints
are rejected.

The tested endpoint is per-case physical-temperature relative L2. For each
method and case, first take the arithmetic mean over the common ordered target
seeds `[100,101,102]`. Every exact comparison then uses the inherited
case-wise seed-mean log-ratio difference and meet-in-the-middle two-sided
paired sign-flip test. The `1e-12` floor applies only to log inference;
activation counts are reported and raw errors remain unchanged.

## Family S1: paper-relevant B8--B10 baselines

The following three contrasts form one Holm family at familywise alpha 0.05:

1. P versus `b8_lp_fno_causal_coeff_v1`;
2. P versus `b9_pifno_dual_physics_only_v1`; and
3. P versus `b10_causality_deeponet_v1`.

All are at target budget 32 and seeds 100--102. Stored rows name the numerator,
denominator, common seeds, case order, and negative-log-ratio direction
explicitly. Apply Holm step-down correction to exactly these three raw
two-sided p-values and report all raw/adjusted p-values and effects regardless
of rejection.

For descriptive effect intervals, use 10,000 crossed paired bootstrap
replicates with NumPy PCG64 seed `20260727`. Generate the complete seed-index
matrix `[10000,3]` first, then the case-index matrix `[10000,40]`, both
row-major; use each Cartesian resample identically for both methods. Save the
draw matrices and their C-order byte hashes. Report raw-error mean ratios,
relative improvements, case-median relative improvements, and the inherited
guardrails. These intervals are secondary and do not inherit the primary
support threshold.

## Family S2: staged transfer

P versus `p_staged_t0t1t2_v1` is one separately labeled, two-sided exact
paired comparison at budget 32 and seeds 100--102. It has no multiplicity
adjustment because the family has one member. Descriptive crossed paired
bootstrap intervals use PCG64 seed `20260728` and the same draw order and
10,000-replicate contract.

The comparison concerns final predictive performance and compute. It does not
permit choosing a favorable stage or intermediate checkpoint after held-out
release. Stage-specific validation histories and trainable-parameter counts
are reported descriptively from the frozen training record.

## Family S3: focused physics/input ablations

The three pairwise contrasts among P, `p_no_raw_x_v1`, and
`p_naive_strong_residual_v1` form one Holm family:

1. P versus `p_no_raw_x_v1`;
2. P versus `p_naive_strong_residual_v1`; and
3. `p_no_raw_x_v1` versus `p_naive_strong_residual_v1`.

All use budget 32 and seeds 100--102. The first two are the component-focused
comparisons; the third is retained as a two-sided pairwise contrast without a
predeclared superiority direction. Apply Holm correction across exactly
three raw p-values at familywise alpha 0.05. Descriptive crossed paired
bootstrap intervals use PCG64 seed `20260729` under the same 10,000-replicate
and draw-order contract.

## Source-seed sensitivity

The full 3-by-3 source-seed/target-seed P grid is descriptive. Source seeds
are `[0,1,2]` and target seeds are `[100,101,102]`; the source-seed-0 cells are
the exact core P runs rather than retraining. Report every cell, marginal
means by source seed and target seed, the grand mean, ranges, and a
two-factor random-effects variance decomposition when estimable. Because
there are only three source initializations, no source-seed superiority test
or minimum p-value is reported.

If descriptive crossed resampling is shown, use PCG64 seed `20260730`, 10,000
replicates, and save all draw matrices. Such intervals are labeled
sensitivity intervals and cannot support a confirmatory source-seed claim.

## Coarse pseudo-predictor

The deterministic coarse predictor is descriptive only. Report the same
per-case endpoints, physical diagnostics, violations, and ID/OOD summaries as
for neural predictors, with parameter count and training time equal to zero.
Do not include it in a paired seed test, duplicate it three times, or use its
performance to redefine any family.

## Completeness, failures, and guardrails

Every registered predictor-case primary cell must be present and finite. It
is not dropped, imputed, winsorized, or replaced by a retry. A scientific
terminal failure remains visible and fails the relevant comparison's
completeness gate. A method not structurally designed for monotone alpha,
notably B9, is not retroactively excluded for monotonicity; its violations are
reported as outcomes. Nonfinite predictions and missing primary cells always
fail completeness.

For every comparison, report alpha relative L2, peak-temperature error,
final-alpha MAE, conservative residual, gradient errors, tail summaries,
alpha-bound/monotonicity counts, parameter count, optimizer updates,
candidate-evaluation time, device-synchronized target training time, and
source cost where applicable. These are descriptive unless explicitly named
as the tested temperature endpoint above. No favorable endpoint substitutes
for a failed temperature result.

## Pre-release waiver for historical source seed 0 timing

Source seed 0, run `p3-source-causal-v1-seed0-fix1`, predates the separated
source-timing instrumentation used for new source seeds 1 and 2. This
limitation was identified and registered before the first P6-v2 held-out
release. The historical run is not replayed, modified, or timing-backfilled.
Its provenance commit is
`5806d3cbf7248058bd85c317d2b99a6d0d1f22a1`.

The validated historical evidence reports exactly 163,270 parameters, 4,000
optimizer updates (100 epochs times 40 training batches), and 100 candidate
validations. Its `history.parquet` durations sum exactly to
112.71573159942636 seconds, and its one completed process session reports
127.30308979999973 seconds of wall time. The epoch duration starts before the
training batches and includes the optimizer work, scheduler step, complete
20-case validation pass (five batches), validation-based selection
bookkeeping, and a final CUDA device synchronization. It ends before
checkpoint and history serialization. The final held-out evaluations and
causality evaluation occur after the training loop and are also outside the
epoch-duration aggregate.

The 112.71573159942636-second value is therefore stored only as
`legacy_combined_epoch_seconds` with quality
`legacy_combined_only_no_separation`. It is never relabeled or inferred as
device-synchronized training time or candidate-validation time. For source
seed 0, `device_synchronized_training_seconds` and
`candidate_validation_seconds` remain NULL and `timing_complete` is false.
The 127.30308979999973-second session wall time remains a separate direct
observation and is not used to decompose the combined epoch value. Live
hashes of the historical metrics, history, resolved configuration, checksum
record, session log, completion marker, both checkpoints, declared
configuration, and split manifest bind the waiver evidence. The statistics
code does not open or hash the source data array or any P6-v2 held-out label
file while validating this evidence.

The historical combined epoch cost is amortized only in its own explicitly
labeled column: over the frozen core denominator of 81 for core seed-0
consumers and over the nine new supplemental seed-0 consumers for the
supplemental context. It is never added to the columns that combine exact
separated source training and candidate-validation time. New source seeds 1
and 2 must each provide their exact separated device-synchronized training
and candidate-validation evidence and remain amortized over three registered
target runs.

The compute-accounting gate may pass only when all target-run required fields
are exact, both new supplemental source rows have complete separated timing,
and the source-seed-0 historical evidence above validates exactly. A passing
gate must still report `all_fixed_source_costs_separated=false`,
`new_supplemental_source_costs_separated=true`, and
`legacy_seed0_combined_cost_documented=true`; it must not imply that all
three fixed source costs were separated.

## Required supplemental statistics artifacts

The atomic statistics bundle must contain:

- exact population and roster audits for 24 supplemental target runs, two
  source runs, nine source-sensitivity cells, and the one coarse predictor;
- predictor-case completeness tables for Stage 1 and Stage 2;
- the three-row S1 Holm table, one-row S2 table, and three-row S3 Holm table;
- exact sign-flip counts, denominators, input hashes, raw p-values, adjusted
  p-values, and implementation identity;
- all four registered PCG64 seeds, draw matrices, C-order hashes, and complete
  bootstrap distributions;
- descriptive source-seed and coarse-predictor outputs;
- one compute row per unique run with source costs kept separate from target
  and candidate-validation time;
- three fixed-source evidence rows. The seed-0 historical combined epoch cost
  is amortized over the nine new supplemental seed-0 consumers (staged,
  no-raw-x, and naive-strong-residual at three target seeds) only in the
  explicit legacy-combined column. The seed-1 and seed-2 source costs are each
  amortized over their three registered target runs. Their
  device-synchronized training and candidate-validation costs remain separate
  columns, are divided by the same registered denominator, and are combined
  only in an explicitly labeled exact total. B8--B10 are source-free. The
  three seed-0 source-sensitivity cells reuse existing core P runs, create no
  compute row, and retain the frozen core denominator of 81 rather than
  changing either denominator post hoc;
- explicit failure/nonfinite/violation accounting;
- exact input paths, byte counts, SHA-256 hashes, and a canonical bundle
  manifest; and
- an audit proving P6-v1 and unregistered retries contributed no value.

Manuscript tables and figures consume this frozen bundle and never recompute
supplemental metrics ad hoc.
