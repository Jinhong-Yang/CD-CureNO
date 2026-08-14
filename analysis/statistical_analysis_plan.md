# P6 statistical analysis plan

## Status, population, and analysis unit

This plan is frozen before P6 confirmatory training and before ID-test or OOD
label release. The primary population is the 40-case frozen
`combined_ood` manifest. The experimental units are complete simulation
cases and target-training seeds; grid cells and time points are repeated
measurements within a case and are never treated as independent samples.

The primary estimand is conditional on the fixed P3 causal source seed-0
checkpoint. The crossed design has five target seeds and the same 40 cases
evaluated for both methods. Method comparisons are paired by target seed and
case. All metrics are calculated in float64 from saved physical-unit
predictions, even if model inference uses float32.

## Metric definitions

Let \(m\) denote method, \(s\) target seed, \(c\) case, \(T\) the true
temperature in kelvin, \(\widehat T\) its prediction, \(\alpha\) true degree of
cure, and \(M_c\) the static composite-cell mask. Sums over time and space use
the product quadrature \(w_{t,z,x}=w_t\,\Delta z_z\,\Delta x_x\). Time uses
endpoint trapezoidal weights: half the adjacent time gap at either endpoint
and half the sum of adjacent gaps at each interior sample. For each
cell-centred spatial coordinate, the first and last control-volume widths
equal the adjacent coordinate gap and every interior width is half the sum of
its adjacent gaps. These weights are formed independently in `z` and `x`.
The evaluator uses these physical float64 weights explicitly.

### Primary temperature error

The complete-case composite temperature relative L2 is

\[
e^{T}_{m,s,c}
=
\frac{
\left[\sum_{t,z,x}w_{t,z,x}M_c(z,x)
(\widehat T_{m,s,c}-T_c)^2\right]^{1/2}
}{
\left[\sum_{t,z,x}w_{t,z,x}M_c(z,x)T_c^2\right]^{1/2}
}.
\]

The kelvin denominator is strictly positive for the physical dataset. An
observed nonpositive or nonfinite denominator is a data-integrity failure, not
a value to impute. Metrics are first computed per complete case, never by
pooling all grid cells across cases.

### Field and manufacturing metrics

For temperature and alpha, compute per-case MAE, RMSE, relative L2,
\(L_\infty\), and range-normalized RMSE in the composite and, where defined,
the tool:

\[
\operatorname{NRMSE}_{\mathrm{range}}
=
\frac{\operatorname{RMSE}}
{\max(y)-\min(y)}.
\]

The denominator is the true field's within-case, within-region
spatiotemporal range. It is not a training-set normalization statistic. If
that range is at most `1e-12` in the field's physical unit, NRMSE is marked
structurally undefined and the event is reported; it is not replaced by a
different denominator post hoc. NRMSE is secondary and cannot make the
primary metric missing.

The three pre-registered safety guardrails are:

- **peak-temperature absolute error**, in kelvin:
  \(\left|\max_{t,z,x:M=1}\widehat T-\max_{t,z,x:M=1}T\right|\);
- **final-alpha MAE**: the spatial mean of
  \(|\widehat\alpha(t_{\mathrm{final}})-\alpha(t_{\mathrm{final}})|\)
  over composite cells; and
- **dimensionless conservative residual**, defined below.

P6 also reports time-to-peak error, time to alpha 0.8/0.9/0.95, and spatial
temperature/cure-gradient errors. Threshold comparison admits exactly the
immediate float64 predecessor of the registered decimal threshold so a
spatial mean that differs only by one downward ULP is not falsely censored.
Crossing time is otherwise the first grid crossing with linear interpolation
between adjacent output times. If a threshold is not reached by the true or
predicted trajectory, that observation is right-censored at the final time and
reported through separate truth/prediction attainment fractions, joint
attainment status, and a censoring-aware summary; it is not assigned an
invented crossing time.

Maximum thermal lag, maximum exotherm, and process-constraint satisfaction are
explicitly unevaluated in P6. The available project sources do not freeze an
independent ambient-reference aggregation or sourced manufacturing-window
thresholds, so defining them after held-out outcomes would be post hoc and
inventing thresholds would violate the project contract. P7 may add them only
with a dated, outcome-labeled exploratory analysis that sources the process
limits and fixes the aggregation before the external-validation labels are
opened. They are absent from every P6 gate rather than silently reported as
zero.

### Conservative residual and alpha violations

For each finite-volume cell and output interval, use the trapezoidal
interval-integrated energy residual in `J m^-3`,

\[
Q_{E,c}^{n}
=
\rho_c C_{p,c}
(\widehat T_c^{n+1}-\widehat T_c^n)
+\frac{\Delta t_n}{2}
\left(D_{q,c}^{n+1}+D_{q,c}^{n}\right)
-S_{\mathrm{cure},c}
(\widehat\alpha_c^{n+1}-\widehat\alpha_c^n),
\]

where \(D_q=V^{-1}\sum_f q_fA_f\) is outward heat-rate density.
Conductive and boundary heat rates use harmonic face conductivity, the same
boundary-flux convention as the P4 solver, zero reaction source in tool cells,
and one shared flux on each material interface. This is the trapezoidal
time integral of the rate-form finite-volume residual over the stored
interval. Define the strictly positive, prediction-independent reference
energy density in each cell as

\[
E_{\mathrm{ref},c}
=
\rho_c C_{p,c}(100\,\mathrm K)
+M_c\,S_{\mathrm{cure},c}\,r_{\mathrm{reaction},c},
\]

where \(S_{\mathrm{cure}}\) is the resolved base cure-source energy per unit
volume and unit alpha and \(r_{\mathrm{reaction}}\) is the frozen
reaction-enthalpy scale. The case-level evaluation guardrail is

\[
e^E_{m,s,c}
=
\frac{\sum_{n=0}^{N_{\mathrm{int}}-1}\sum_c
V_c\,|Q_{E,c}^{n}|}
{N_{\mathrm{int}}\sum_c V_c E_{\mathrm{ref},c}}.
\]

The implementation sums interval-integrated residuals and multiplies the
fixed reference by `N_int`. There is no prediction-dependent term-magnitude
denominator and no epsilon addition. A nonpositive or nonfinite fixed
reference is a data/configuration integrity failure. The same metric is
computed for the truth fields and stored as the snapshot-discretization
floor. P6 reports this global residual together with the cellwise
dimensionless energy-residual summaries.

Standalone kinetics, Robin, interface-temperature-jump, and
interface-flux-jump evaluation diagnostics are explicitly unevaluated in P6.
The current evaluator has no pre-frozen independent one-sided boundary or
interface reconstruction; reusing the shared finite-volume face flux would
make flux continuity tautological rather than diagnostic. P7 may add these
only under a dated exploratory protocol that fixes an independent one-sided
gradient/extrapolation estimator before external-validation label access.
Their absence cannot pass or fail a P6 scientific gate and must be shown as
`not_evaluated`, never imputed as zero.

This residual is a discrete consistency diagnostic on a surrogate prediction.
It is not proof that the prediction solves the continuous PDE and must not be
described as one. It is also distinct from the cellwise mean-squared
dimensionless residual used in the training physics loss.

An alpha-bound violation is any composite value below `-1e-7` or above
`1+1e-7`. A monotonicity violation is any consecutive composite increment
less than `-1e-7`. Report counts and the largest excursions, including
sub-tolerance excursions. A nonfinite violation is any NaN or positive or
negative infinity in a prediction or required metric; there is no tolerance
for nonfinite values.

## Primary hypothesis and effect estimands

The primary comparison is P (`cdcureno_full`) versus B7
(`generic_causal_transfer`) at target budget 32 on `combined_ood`, target
seeds 0--4.

The primary point estimand is the relative improvement in the crossed
seed-case arithmetic mean:

\[
I_{\mathrm{mean}}
=
1-
\frac{
\operatorname{mean}_{s,c}e^T_{P,s,c}
}{
\operatorname{mean}_{s,c}e^T_{B7,s,c}
}.
\]

Positive values favor P. The complementary robustness estimand first averages
over target seeds within each case,

\[
I_c
=
1-
\frac{\operatorname{mean}_{s}e^T_{P,s,c}}
{\operatorname{mean}_{s}e^T_{B7,s,c}},
\qquad
I_{\mathrm{case\ median}}=\operatorname{median}_c I_c.
\]

Raw metric values are never floored for these two effect estimates. Every
per-seed/per-case B7 primary error must be strictly positive; a zero value is
a degenerate primary estimand and a fail-closed analysis error rather than a
post hoc tie rule. Proposed errors may be zero. Guardrail metrics use the
separate zero-denominator rule below.

The scientific result is supportive only if all conditions hold:

1. the 95% hierarchical-bootstrap confidence interval for
   \(I_{\mathrm{mean}}\) has a strictly positive lower endpoint;
2. \(I_{\mathrm{case\ median}}\ge0.10\);
3. the P/B7 ratio of crossed seed-case arithmetic means is at most 1.10 for
   each of peak-temperature absolute error, final-alpha MAE, and dimensionless
   conservative residual; and
4. P and B7 each have zero nonfinite, alpha-bound, and alpha-monotonicity
   violations over every primary prediction.

Guardrail ratios use raw nonnegative metrics. If the B7 mean is zero, a P mean
of zero gives ratio 1 and a positive P mean gives infinity. Bootstrap
intervals for guardrails are reported descriptively, but the registered
guardrail decision uses the point ratios above.

## Crossed paired hierarchical bootstrap

Use exactly 10,000 bootstrap replicates generated by NumPy `default_rng` with
seed `20260725` (PCG64). RNG draw order is part of the lock: first generate
the complete `int64` seed-index matrix of shape `[10000,5]`, row-major, then
the complete `int64` case-index matrix of shape `[10000,40]`, row-major. Hash
the C-order bytes of those two matrices in that order. For each replicate:

1. sample five seed indices with replacement from the ordered list
   `[0,1,2,3,4]`;
2. independently sample 40 case indices with replacement from the manifest's
   stored case order;
3. take the Cartesian crossed submatrix of the sampled seed and case indices,
   including multiplicities;
4. use the identical sampled indices for P and B7; and
5. recompute \(I_{\mathrm{mean}}\), the case-median estimand, guardrail
   ratios, and requested descriptive effects.

This is a crossed, not nested, hierarchical bootstrap: target-seed and case
variation are independently resampled because every seed is evaluated on
every case. Pairing is never broken. The 95% interval is the percentile
interval at quantiles 0.025 and 0.975 using NumPy's linear quantile rule.
The point estimate is computed on the original matrix, not as the mean of
bootstrap replicates. Save the sampled seed-index and case-index matrices or
their deterministic hash so the interval can be independently regenerated.

For every registered bootstrap estimand, save the full float64 empirical
distribution or an immutable artifact containing it, plus its SHA-256,
arithmetic mean, median, standard deviation, interquartile range, and
percentile interval. The required endpoint distributions are
\(I_{\mathrm{mean}}\), \(I_{\mathrm{case\ median}}\), and the three P/B7
guardrail ratios. Also report the corresponding observed distribution
summaries by method. As
standardized paired effects, report Cohen's \(d_z\) on the 40 case-wise
seed-mean log ratios and Hedges
\(g_z=d_z[1-3/(4n-5)]\), with \(n=40\). The log ratio is
`log(P)-log(B7)`, so negative standardized effects favor P; tables must state
this explicitly.

## Confirmatory exact paired sign-flip test

For case \(c\), define

\[
d_c=
\log\!\left(\max(\bar e^T_{P,c},10^{-12})\right)
-
\log\!\left(\max(\bar e^T_{B7,c},10^{-12})\right),
\]

where bars are arithmetic means over the five paired target seeds. The
`1e-12` floor is used only for log-ratio inference and its activation count is
reported. It does not alter the primary raw-error estimand.

Test the sharp paired null of sign exchangeability with the two-sided
statistic \(|\sum_c d_c|\). A bitwise float64 zero difference is a deterministic
tie: it contributes zero, receives fixed positive sign, and is excluded from
the number \(q\) of flippable differences. The exact p-value is

\[
p=
\frac{
\#\left\{\boldsymbol\sigma\in\{-1,+1\}^{q}:
\left|\sum_c\sigma_cd_c\right|
\ge \left|\sum_cd_c\right|
\right\}
}{2^q}.
\]

The inclusive comparison makes boundary ties conservative. Compute the count
over all sign assignments, not by Monte Carlo. For up to 40 nonzero cases, a
deterministic meet-in-the-middle implementation enumerates two lists of at
most \(2^{20}\) signed partial sums, sorts one list, and counts qualifying
combinations. The implementation and input-value hash are saved.

Use two-sided alpha 0.05. A p-value below 0.05 is required for a
"confirmatory exact-test significant" statement. If the bootstrap gate and
exact test disagree, report both; do not replace either analysis.

## Secondary comparisons and multiplicity

The following 16 temperature relative-L2 contrasts form one pre-registered
secondary family. Every one is evaluated on the same frozen 40-case
`combined_ood` split; each contrast uses only common seeds and paired cases:

1. P versus B7 at budgets 8, 16, 64, 128, and 256 (five contrasts);
2. P versus B5 at budget 32, seeds 0--4;
3. P versus B6 at budget 32, seeds 0--2;
4. B7 versus B5 at budget 32, seeds 0--4;
5. B7 versus B6 at budget 32, seeds 0--2; and
6. A1--A0, A2--A1, A3--A2, A4--A3, A5--A4, A7--A5, and A7--A6 at budget
   32, seeds 0--2 (seven contrasts).

For every contrast, the first-named method in prose is the proposed lower
error direction; the stored contrast table must include explicit numerator,
denominator, and sign rather than relying on row order. Use the same case-wise
seed-mean log-ratio exact sign-flip test. Apply Holm's step-down correction to
all 16 raw two-sided p-values as one family at familywise alpha 0.05. Order
raw p-values ascending, compare the \(i\)-th to
`0.05 / (16 - i + 1)`, and enforce monotone Holm-adjusted p-values. Report raw
and adjusted values, all effects, and intervals regardless of rejection.

Secondary endpoint tests for alpha, gradients, physical diagnostics, and
manufacturing metrics are exploratory. Their raw p-values and effect
intervals may be shown but cannot be called multiplicity-confirmed unless a
separately labeled Holm family is declared before label release. No secondary
result overrides the primary decision.

For data efficiency, report each method's mean error at all six budgets and
the trapezoidal area under error versus `log2(label budget)` on
`combined_ood`. The curve, budget-32 primary point, and source-amortized
compute are shown together. There is no post hoc budget or split selection.

Compute accounting has one row per unique confirmatory run. Target time is
the exact sum of the 120 device-synchronized epoch durations in the terminal
training record, with optimizer updates checked as
`120 * label_budget / 4`. Peak accelerator memory is retained. The fixed
source cost is the summed epoch duration of
`p3-source-causal-v1-seed0-fix1`. The pre-registered amortization denominator
is 81: the 60 B7/P runs, three B6 runs, and 18 A1--A6 runs use the source.
B5 and A0 are the eight source-free runs and receive zero amortized source
cost. A7 is an alias with no extra compute. Thus a source-using row reports
`target_seconds + fixed_source_seconds / 81`, while a source-free row reports
target seconds; the undistributed fixed source cost is also displayed
separately.

The completed-epoch sum is the matched scientific-compute estimand. Each
pause, explicit infrastructure interruption, terminal failure, and completion
also records its own elapsed session wall time. Report the sum of all known
session durations, the number of known terminal sessions, and the number of
abrupt unclean interruptions whose partial duration cannot be recovered.
Unknown crash time is never imputed as zero. Failed-attempt and operational
session time remain in the audit/compute supplement even though source
amortization uses the completed-epoch measure.

## Missing values, failed runs, ties, and tails

Complete-case analysis means a case enters a paired contrast only if both
methods have the required finite metric for every seed used by that contrast.
For the primary analysis, however, any missing/nonfinite seed-case cell is
itself a P6 gate failure: it is not dropped, imputed, winsorized, or replaced
by a rerun chosen for favorable performance. Infrastructure-preempted runs may
resume only from an exact checkpoint containing consistent model, optimizer,
scheduler, epoch, history, sampler/order, resource-profile, and Python/NumPy/
Torch CPU/CUDA RNG state. If no such checkpoint exists, the same registered
run ID may restart from its locked initialization; every failed attempt
remains logged. A numerical or scientifically divergent terminal run remains
part of the roster and is not replaced. If it prevents a required finite
cell, the P6 completeness/execution gate fails while the observed failure is
still reported.

All ties use deterministic rules:

- exact validation-objective ties select the earlier epoch;
- exact zero paired log differences are handled as specified in the
  sign-flip test;
- equal case errors are ordered by manifest case ID in displays only; and
- quantile calculations use the fixed linear rule.

There is no outlier deletion or metric winsorization. Report every case, the
worst 5% as the two largest-error cases for a 40-case OOD regime, and the
worst 1% as the single largest-error case. Ties at a tail boundary are listed
in case-ID order and the tie is noted. Case studies for figures are either
preselected by ID or use a deterministic median-seed/worst-case rule labeled
as such; no unlabeled "best seed" figure is allowed.

## Reproducibility outputs

The statistical command must consume only immutable per-case metric tables
from `outputs/tables/p6_frozen_results.parquet` and emit:

- the machine-readable primary/execution decision and provenance at
  `outputs/tables/p6_primary_comparison.json`;
- all five 10,000-replicate float64 bootstrap distributions plus the exact
  int64 seed/case draw matrices at
  `outputs/tables/p6_primary_bootstrap_distributions.npz`;
- the 16-row exact-test/Holm family at
  `outputs/tables/p6_secondary_holm.csv`;
- a roster and completeness report for every required run at
  `outputs/tables/p6_roster_completeness.csv`;
- the label-efficiency points/AUC, tail distribution, and violation tables at
  `outputs/tables/p6_label_efficiency.csv`,
  `outputs/tables/p6_tail_distribution.csv`, and
  `outputs/tables/p6_violation_summary.csv`;
- the alpha 0.8/0.9/0.95 attainment, four-state joint censoring, and
  jointly-attained crossing-error summary at
  `outputs/tables/p6_alpha_attainment.csv`;
- the registered method-cell summaries for `cycle_ood`, `htc_ood`, and
  `pattern_ood` at `outputs/tables/p6_regime_summary.csv`;
- the exact 400 primary seed-case rows and 89-run compute accounting at
  `outputs/tables/p6_primary_per_case.csv` and
  `outputs/tables/p6_compute_accounting.csv`;
- exact input paths and SHA-256 hashes;
- per-method/per-seed/per-case primary and guardrail metrics;
- the 10,000-replicate crossed-bootstrap summary and replicate/index hash;
- exact sign-flip counts, denominators, raw p-values, and implementation hash;
- the 16-row Holm table;
- distribution and worst-tail tables;
- all alpha/nonfinite violation counts;
- label-efficiency and compute tables; and
- a machine-readable primary decision object with one Boolean per registered
  gate and no hand-entered result.

The execution gate may be true only when
`outputs/tables/p6_frozen_results_provenance.json` independently binds the
frozen result-table hash, the committed two-stage release record, the exact
89-run roster, every terminal metrics/best-checkpoint/DONE hash, and every
per-run/per-split prediction and metric artifact. Regenerating statistical
tables from a hand-built frame is insufficient. The statistical command
re-hashes those live artifacts and records the provenance-file hash before
setting its terminal-and-release gate.

Each terminal-run provenance closure must include
`preparation_launch_identity.json` and verify its resolved scientific
configuration, input checksums, Git/implementation identity, runtime/package
snapshot, and resource profile. Each `(run_id, split)` evaluation artifact is
valid only after an exact pending marker precedes the atomic prediction and
per-case-metric writes and an immutable pair-commit marker binds both hashes.
A matching pending partial pair may be fully recomputed on resume; markerless,
mismatched, or partially committed output fails closed.

The committed pre-label record must additionally bind one common training
runtime/package fingerprint across all 89 runs and one evaluator runtime
fingerprint. Held-out evaluation uses seed `20260726`, deterministic PyTorch
algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN with
benchmarking disabled, and CUDA/cuDNN TF32 disabled. Its fingerprint includes
the complete installed-distribution snapshot and the Python, NumPy, Pandas,
PyTorch, CUDA, cuDNN, driver, and accelerator identity. Stage 1 and Stage 2
must reproduce the same evaluator runtime hash exactly.

Stage 2 is prohibited until the complete 89-run Stage-1 record at
`outputs/tables/p6_evaluation_stage1_release_v1.json` is Git tracked,
committed, clean, and revalidated against its live prediction, metric, and
pair-commit artifacts.

After Stage 2, `outputs/tables/p6_evaluation_release_v1.json` and
`outputs/tables/p6_frozen_results_provenance.json` must be committed together
and Git tracked and HEAD-clean before `scripts/statistical_tests.py` runs.
The statistical provenance verifier independently rejects either untracked or
dirty file and revalidates their exact live hashes and contents before any
execution or scientific decision gate can pass.

All manuscript tables and figures must be generated from these outputs and
trace back to run IDs. Rounding occurs only for display; decisions use stored
float64 values. Failed runs, access violations, amendments, and superseded
results remain in the audit trail.
