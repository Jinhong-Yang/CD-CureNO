# P3 conservative solver and source-pretraining validation

## Decision

P3 passes its staged scientific gate. The conservative 1-D solver passes
pre-specified energy, public-case discrepancy, and mesh/time convergence
criteria. The selected source model passes the unchanged held-out-family
criteria. These results authorize P4 solver and dataset work; they are not a
confirmatory model-superiority claim.

The machine-readable decision is
`outputs/tables/p3_gate_summary.json`. It independently recomputes the primary
metrics from case-level CSV files, verifies the source-array checksum and split
partition, and checks that the three failed development runs remain preserved.

## Governing model and numerical contract

The solver advances the heat equation in conservative node-centred
finite-volume form with harmonic face conductance, Robin boundary fluxes, and
backward-Euler heat integration. Cure is integrated by RK4 inside a Picard
thermochemical coupling loop. Heat release is computed from the exact cure
increment over each step, so the discrete thermal source and cure update are
consistent.

All calculations use SI units. Material, kinetic, geometry, and boundary
parameters are frozen in `configs/material/as4_8552.yaml`; the source-by-source
transcription is in `references/as4_8552_parameter_table.md`. No public
temperature or cure label was used to fit a material parameter.

## Provenance conflicts and operational choices

Johnston (1997), Eq. 6.13, prints the diffusion-control denominator as
`1 + exp(C*(alpha - (C_0 + C_T*T)))`. Niaki et al. (2021), Eq. 31, prints the
exponential without the leading one. Replaying the immutable public temperature
histories gives approximately 1% global cure error with the Johnston form and
approximately 47% with the no-offset transcription. The solver therefore uses
the source-supported Johnston form and records the alternative as a provenance
conflict rather than silently tuning the law.

The public arrays and upstream mask use tool indices 0--20 and composite
indices 21--50 on a 0--50 mm grid, consistent with Niaki's 20 mm tool plus
30 mm composite benchmark. One passage in the final Chen et al. article reverses
those labels. Validation follows the immutable public mask.

## Public Case1 validation

The frozen benchmark contains 200 cases, 51 through-thickness nodes, and 223
one-minute outputs. The solver used a maximum internal step of 10 s.

| Metric | Result | Pre-specified limit | Pass |
|---|---:|---:|:---:|
| Temperature field relative L2, mean | 0.00114323 | 0.005 | yes |
| Temperature absolute error, maximum | 6.14000 K | 10 K | yes |
| Composite temperature relative L2, mean | 0.00127299 | descriptive | -- |
| Tool temperature relative L2, mean | 0.000908268 | descriptive | -- |
| Peak composite temperature error, MAE | 1.41191 K | descriptive | -- |
| Cure field relative L2, mean | 0.0132320 | 0.05 | yes |
| Cure absolute error, maximum | 0.0592220 | descriptive | -- |
| Cure bound violations | 0 | 0 | yes |
| Cure monotonicity violations | 0 | 0 | yes |
| Maximum local energy residual | 3.98636e-6 W m^-3 | 1e-4 W m^-3 | yes |
| Maximum relative global energy residual | 5.30376e-13 | 1e-8 | yes |

The agreement is scientifically acceptable for an independently implemented
open solver using literature-transcribed constants, rather than the unavailable
original proprietary COMSOL model. It does not establish bitwise reproduction
of that proprietary model.

## Mesh and time convergence

Convergence uses immutable public air schedule case 100 without reading its
temperature or cure labels. The reference uses 0.25 mm spacing and a 2.5 s
maximum step. Mesh and time temperature and cure errors strictly decrease, the
finest tested resolutions pass the frozen thresholds, and the reference
maximum energy residuals are 5.65092e-5 W m^-3 locally and 1.91452e-12
globally. Exact grids, errors, and checks are stored in
`outputs/tables/p3_solver_convergence.csv` and
`outputs/tables/p3_solver_convergence.json`.

## Parametric source data

The final source dataset `p3_source_1d_v3` contains 400 cases from a fixed
seed. `single_hold` and `two_hold` are the training families; all 100
`smart_cure` and all 100 `three_hold` cases are held out. Geometry, lower and
upper heat-transfer coefficients, composite conductivity scale, and heat of
reaction scale vary within the frozen bounds.

Fine labels use 1 mm spacing and a 20 s maximum step. Version 3 adds a
label-free low-fidelity thermochemical baseline using 16 spatial intervals and
a 60 s maximum step. It solves the same sourced equations but does not consume
fine labels. The array SHA-256 is
`9db82081f71e444265087531369a2293d4b7880db9b183943490862594d10ffa`.
All cases pass the source-generation energy checks.

## Source-pretraining development sequence

Acceptance thresholds were frozen in v1 and left unchanged. Failed variants
are negative results, not discarded trials.

| Run | Material change | Held-out T relative L2 mean | Held-out T Linf | Gate |
|---|---|---:|---:|---|
| v1 | 13-channel direct factorized FNO | 0.00657647 | 44.7876 K | fail |
| v2 | add inert causal thermal baseline | 0.00686316 | 38.3836 K | fail |
| v3 | higher capacity and normalized L4 term | 0.00272429 | 38.1166 K | fail |
| v4 | correct label-free coarse thermochemical baseline | 0.00128006 | 8.08954 K | pass |

The selected v4 model has 160,130 parameters and was selected at epoch 93 from
100 epochs using only the frozen validation set. Training used 160 cases and
validation used 20; held-out-family labels were excluded from training,
normalization, and checkpoint selection.

| Split | Cases | T relative L2 mean | T Linf | Cure relative L2 mean |
|---|---:|---:|---:|---:|
| In-family test | 20 | 0.00122718 | 6.73209 K | 0.0238063 |
| Smart-cure held out | 100 | 0.00133641 | 7.76144 K | 0.100918 |
| Three-hold held out | 100 | 0.00122371 | 8.08954 K | 0.0288409 |
| Combined held out | 200 | 0.00128006 | 8.08954 K | 0.0648793 |

All cure predictions obey bounds and time monotonicity.

## Interpretation and limitations

The v4 result demonstrates residual correction over a coarse physics solver,
not the performance of a pure black-box operator. P6 must report the
low-fidelity solver alone, the learned correction, scratch baselines, and
multi-seed uncertainty under matched budgets. The current source-model result
uses seed 0 and is sufficient only for the staged P3 gate.

The open solver was cross-checked against public arrays and internal
convergence/energy invariants. P4 still requires an independent selected-case
cross-check for the true-2-D solver; it cannot inherit this evidence by
assumption.
