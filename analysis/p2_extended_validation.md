# P2 extended seed-0 validation

## Overall assessment: Share with caveats

The frozen seed-0 extended factorized and causal runs satisfy every P2 phase
gate and are reliable enough to unblock P3. They do not support a multi-seed
confirmatory superiority claim.

## Methodology review

- Question: can one joint 1+1-D model match the corrected 51-model temperature
  stack while predicting α and preserving or improving spatial coherence?
- Population: frozen 50/25/125 complete-case Case1 split.
- Selection: best validation weighted objective only.
- Temperature and spatial-gradient thresholds: corrected P1 three-seed means.
- Metric grain: one complete case; grid nodes are not independent samples.
- Artifacts: best/last checkpoints, complete histories, physical predictions,
  per-case metrics, provenance, and independent verification for both runs.

## Calculation spot checks

- Factorized field relative L2 `0.00119238` versus P1 `0.00189823`: 37.18%
  descriptive reduction.
- Factorized spatial-gradient relative L2 `0.0349349` versus P1 `0.0865327`:
  59.63% descriptive reduction.
- Causal field relative L2 `0.00131001`: 30.99% descriptive reduction.
- Causal spatial-gradient relative L2 `0.0408230`: 52.82% descriptive
  reduction.
- Parameter counts are 198,282 and 225,162 versus 15,076,467 independently
  trained legacy parameters, reductions of 98.68% and 98.51%.
- Every saved summary and per-case column was recomputed from frozen
  predictions within tolerance.
- α bound violations, α monotonicity violations, and tool-region cure were all
  zero.

## Causality validation

The trained causal best checkpoint was evaluated on fixed cases 75, 100, and
150 at time cutoffs 55, 111, and 167. Perturbing future raw-air and causal
baseline channels changed past temperature, residual, α, and cure rate by
exactly zero at every cutoff.

## Issues and caveats

1. **High:** the extended phase gate has one seed; do not report uncertainty,
   significance, or definitive superiority.
2. **Medium:** public Case1 is through-thickness plus time, not a true
   two-spatial-dimensional benchmark.
3. **Low:** the two CPU runs overlapped. Their epoch sums and peak RSS are
   auditable, but wall time is not an uncontended speed comparison.

## Decision

P2 passes and P3 may begin. The main P6 comparison remains blocked until at
least five seeds and the pre-registered statistical workflow are complete.
