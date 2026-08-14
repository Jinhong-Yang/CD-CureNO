# Known issues and threats to validity

## Confirmed legacy defects

- L1: the legacy field is assembled from independent position-wise models.
- L2: input and temperature normalizers are fitted before splitting.
- L3: a smoothness term is constructed but `l2.backward()` is called.
- L4: the alpha task reaches undefined `norm_y` state during evaluation.
- L5: maximum temperature error is overwritten for each test batch.
- L6: `Predict` requires `Ta`, but `Step1_main.py` omits it.
- L7: default training produces x=35 only; plotting loads all 51 positions.
- L8: most requirements are unpinned and non-core Google/visualization packages
  are mixed into the training environment.
- L9: `Step1_main.py` loads missing `data/Designed_2hold_200.mat`; the checkout
  contains `data/Case1.mat`.

## Open threats

- L10: upstream has no explicit license; redistribution remains blocked.
- L11: the original COMSOL models require proprietary software and have not yet
  been independently validated.
- L12: the upstream README does not provide an environment, seed protocol, or
  expected numerical reproduction tolerance.
- L13: saved checkpoints exist for only selected spatial positions.
- L16: corrected-vs-exact accuracy at x=35 is seed-sensitive and shows no
  material mean improvement over three matched seeds.
- L17: the exact full temperature field has one seed, so its same-seed
  comparison with corrected is descriptive rather than confirmatory.
- L18: corrected alpha execution is validated at one location in P1; full-field
  joint alpha prediction remains a P2 requirement.
- L19: the fixed P2 pilot trainer evaluated the test split. Extended
  configurations are frozen from validation histories only; any test-driven
  tuning would invalidate the confirmatory interpretation.
- L20: forced execution-window interruptions caused the old pilot
  `training_wall_seconds` field to represent only the final resume session.
  Pilot aggregation instead uses the sum of all epoch durations from
  `history.parquet`; new runs record explicit session events and both timings.
- L21: the P2 extended gate currently has one seed. It is sufficient for the
  staged P2 decision gate but not for uncertainty intervals, hypothesis tests,
  or a confirmatory superiority claim.
- L22: the two extended P2 jobs overlapped on 8 of 24 logical CPU threads.
  Per-run epoch time and peak process RSS are valid audit quantities, but their
  wall times must not be interpreted as an uncontended speed comparison.
- L23: Johnston (1997) Eq. 6.13 includes a leading one in the cure-law
  diffusion denominator, whereas Niaki et al. (2021) Eq. 31 omits it. Public
  replay strongly supports the Johnston form, but the publication discrepancy
  remains a provenance threat.
- L24: one Chen et al. article passage reverses the 20/30 mm tool/composite
  labels. The immutable public mask and Niaki benchmark support a 20 mm tool
  plus 30 mm composite, which is the frozen operational geometry.
- L25: P3 source-pretraining currently has one seed and v4 corrects a coarse
  thermochemical baseline. It cannot support a confirmatory superiority claim
  until P6 reports the baseline alone and matched multi-seed comparisons.
- L26: the independently implemented P3 solver agrees closely with the public
  arrays, but the proprietary original COMSOL model is unavailable. Exact
  implementation equivalence is therefore unverified.
- L27: the P4 SciPy-BDF cross-check independently assembles the flux operator
  and time integration, but shares the grid, material values, and cure law with
  the production solver. It is not a fully independent external FEM
  implementation and is reported only within that scope.
- L28: the canonical P4 benchmark freezes reference-temperature, unidirectional
  `k_x/k_z` conductivities and preserves their ratio under a common multiplier.
  Temperature-dependent conductivity, multidirectional laminates, and uncertain
  anisotropy are deferred generalization tests rather than implied coverage.
- L29: the preserved P4 debug and pilot arrays predate the sourced anisotropic
  conductivity contract. They are historical exploratory evidence only; the
  pre-label 512-case core is the first canonical P4 dataset.
- L31: the seed-0, budgets 8/16 noncausal RP-FFNO pilot is a validation-only
  stage gate. It cannot establish statistical transfer superiority or serve as
  final CD-CureNO evidence; P6 still requires the pre-registered multi-seed
  causal comparison.
- L32: P5 target arrays are shared files. Integrity verification checksum-reads
  them and preparation memory-maps them, including files that contain
  forbidden splits. ID-test/OOD label values are never indexed, materialized,
  interpreted, evaluated, or used for selection, but filesystem-level
  “unopened” language would be inaccurate.

- L33: P6 target-seed inference is conditional on one fixed P3 causal source
  seed. Five target seeds quantify target initialization/order variability but
  do not quantify source-pretraining variability.
- L34: the frozen P6 geometry-OOD manifest is intentionally empty and marked
  `deferred_until_F3`. P6 cannot support a geometry-generalization claim.
- L35: maximum thermal lag, maximum exotherm, and process-window satisfaction
  have no sourced, pre-frozen P6 aggregation/threshold contract. Independent
  one-sided kinetics/Robin/interface-jump diagnostics are also absent. These
  are reported as unevaluated rather than imputed; P7 requires a dated
  exploratory protocol before external-validation label access.
- L36: a hard host kill can leave the wall duration of that interrupted
  session unknowable. P6 reports completed-epoch compute and known session
  wall time separately, counts unknown interrupted sessions, and never
  zero-imputes their duration; wall-clock comparisons with incomplete session
  time remain limited.
- L39: the 21 completed P6-v1 confirmatory runs are ineligible for scientific
  inference because their registered validation objective is dominated by an
  ill-conditioned lateral-gradient relative error and omits step 0. They are
  preserved only as deviation evidence and cannot be repaired, pooled, or
  released to held-out labels.
- L40: P6-v2 is validation-informed by the P6-v1 failure mode. Even though no
  held-out label informed the redesign, the manuscript must not describe v2
  as outcome-blind preregistration.
- L41: P6-v2 has frozen its data, metric, schedule, config, roster, and journal
  foundations, but its trainer, managed launcher, CUDA resource preflight, and
  fresh development runs are not yet complete. No v2 performance claim is
  available.

## Resolved threats

- L38: the original masked relative-L2 implementation used an explicit
  square-root of a sum of squares. For the P initialization on a homogeneous
  training case, the x-gradient residual was exactly zero; the forward loss
  was correctly zero, but backward produced NaNs at `sqrt(0)`. The terminal
  failure and four earlier B7 completions are preserved and superseded.
  `torch.linalg.vector_norm` now provides a finite zero subgradient, with an
  exact-match regression test and a new `-rngfix1-gradfix2` cohort.
- L37: the original first P6 development run paused after epoch 1 but failed
  on resume because device-mapped checkpoint loading moved CPU-only RNG state
  tensors to CUDA. The terminal failure is preserved. Resume checkpoints now
  load on CPU, exact CPU-generator replay is regression-tested, and the
  scientifically unchanged eight-run development cohort uses new `-rngfix1`
  IDs under one repaired implementation lock.
- L30: the selected P3 v4 source retains its noncausal temporal FFT and is used
  only for the distinct RP-FFNO stage pilot. A separately trained causal source
  passed its source gate and was exactly inflated into a causal 2-D target with
  zero temporal FFT modules, closing the prerequisite for the P6 causal path.
- L15: whole-field P1 evidence now covers all 51 positions for exact seed 1 and
  corrected seeds 0–2; all 204 location runs were independently verified.
- L14: the local environment now uses official PyTorch `2.13.0+cu130`; an
  RTX 5080 CUDA matrix operation and the complete 111-test suite passed after
  installation.
