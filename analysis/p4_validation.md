# P4 validation

## Result

P4 passed every frozen gate check. The canonical benchmark contains
512 cases with field shape
`[512, 112, 50, 40]` and no failed, silent, or recovered
failure attempts.

## Solver evidence

The conservative 2-D finite-volume solver passed all
27 pre-specified checks. At the canonical
10-second internal step, the full-cycle stress case had temperature relative
L2 `0.000233475337`,
temperature-rise relative L2
`0.000692196883`, maximum
temperature error `0.429961 K`, and
cure relative L2 `0.00134358649` against the
1.25-second reference.

Exact lateral extrusion agreed with the converged 1-D solver, and the selected
heterogeneous case agreed with the independently assembled SciPy-BDF
method-of-lines calculation. The BDF check shares the grid, material values,
and cure law; it independently checks flux assembly and time integration, not
those shared inputs.

## Dataset integrity

- Pre-label plan SHA-256: `a97511e192450a11d3af3c966bae3fd26523b3eb12b8df5728ca49b25d55b0bb`
- Pre-label Git commit: `6d0b31f63428e00524a89445e9f4d7148f46d8f7`
- Core manifest SHA-256: `983ced5f4bcb8bfb06de88b6f17ad46dccd479932e1ae18b942e693957dbd445`
- Array bytes: `917819936`
- Maximum local energy residual:
  `1.83118814e-05 W m^-3`
- Maximum relative global energy residual:
  `1.34489851e-12`
- Temperature range: `293` to
  `527.277 K`
- Cure range: `0` to `0.927249`
- F0 maximum lateral temperature range:
  `0 K`
- Smallest non-F0 peak lateral range:
  `1.05407715 K`

The freezer independently reconstructed the Sobol plan, auxiliary arrays,
per-case hashes, physical invariants, diagnostic thresholds, split counts, OOD
semantics, and nested budgets. It then wrote immutable source, Phase-A, ID, and
OOD manifests. The complete test suite passed
111/111.

## Deterministic replay

Cases `0, 1, 2, 3, 5, 256, 288, 352, 392, 432, 472` were selected using plan metadata only: the first case in
each frozen split plus the first case in each difficulty family. Re-solving
every selected case reproduced the stored float32 temperature and cure slice
hashes exactly.

## Scope and limitations

- The benchmark freezes a unidirectional, reference-temperature conductivity
  model; temperature-dependent conductivity and multidirectional laminates are
  deferred.
- The old debug and pilot arrays are isotropic exploratory artifacts, not part
  of the canonical benchmark.
- Geometry OOD is explicitly deferred until F3; no empty placeholder is
  presented as validation evidence.
- The proprietary original COMSOL model remains unavailable, so exact
  implementation equivalence is not claimed.
