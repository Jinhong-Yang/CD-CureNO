# CD-CureNO

CD-CureNO is a Python research-software stack for auditable neural-operator
development in thermochemical composite curing. It provides conservative
one- and two-dimensional reference solvers, immutable data manifests,
joint-field and causal operators, restriction-preserving checkpoint inflation,
managed execution, and independent evaluation and reporting utilities.

## Release and scientific scope

Version `v0.0.3` adds a platform-independent canonical-path regression fix,
Linux CPU CI, independent continuous-reference solver verification, and a
small public two-dimensional training/reload/evaluation example. It also makes
the target lift's shared channels contiguous to preserve the existing `nx=1`
bitwise contract across CPU kernels. The model equations and validation
thresholds are unchanged; the target implementation hash is updated. The
production solvers and frozen P0--P5 scientific authorities are preserved.

The public software claim is limited to functionality validated in stages
P0--P5. The unfinished P6 held-out campaign is not used to claim predictive
superiority, broad generalization, or experimental validation. The included
SoftwareX manuscript records the numerical evidence and its claim boundaries.

## Quick installation

CD-CureNO supports Python 3.10--3.12. A CPU-only editable installation is
sufficient for the data-independent tests and documentation checks.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for the local shell.
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,revision]"
python -m pytest -q
```

CUDA is optional; the public demo trains and evaluates on a CPU. The exact
package versions used by this release are pinned in `pyproject.toml`.
On Windows systems that have legacy path-length handling enabled, create the
virtual environment at a short path (for example, `C:\venvs\cdcureno`) or
enable long paths; PyTorch's nested license tree can otherwise trigger
`WinError 206` during installation.

## Zero-data release QA

The following commands do not require the non-redistributed ResFNO arrays,
trained checkpoints, or generated P4 field arrays:

```bash
python -m pytest -q
python -m pytest -q tests/unit/test_p4_dataset_pipeline.py
python scripts/generate_p4_2d.py --tier core --plan-only \
  --plan tmp/p4_2d_core_v1_plan_release_review.json
python scripts/inflate_causal_1d_to_2d_checkpoint.py --help
python scripts/verify_p5_causal_restriction.py --help
```

Tests that require separately acquired upstream artifacts are explicitly
skipped when those artifacts are absent. The focused P4 tests validate the
freeze and resume contracts. The plan-only command emits a separate release
review copy and does not generate field labels or overwrite the canonical
pre-label plan.

Historical v0.0.2 QA on 16 August 2026 was a manual Windows run with
Python 3.11.9: `210 passed, 30 skipped`. The revision reproduced one failing
Windows-drive path rejection on clean Linux (`209 passed, 30 skipped, 1 failed`)
and fixes it without relying on host path semantics. A fresh Windows CPU
installation of the corrected revision passes `224 passed, 30 skipped`:
210 existing tests, eight new path cases, four solver-reference tests, and one
complete demo/tampering integration test, plus one strided-input bitwise
regression. The corrected commit also passes `224 passed, 30 skipped` on
[Ubuntu CPU CI](https://github.com/Jinhong-Yang/CD-CureNO/actions/runs/34214599882)
with Python 3.11.16 on an AMD EPYC 7763 runner. See
[platform QA](docs/revision_platform_qa.md) for tested commits, environments,
the preceding CI regression and exact skip scope.

## Public revision examples

No upstream data or pretrained model is needed for these examples:

```bash
python scripts/verify_revision_solver.py --output-dir outputs/revision_solver_verification/my_run --environment-label my_cpu_environment
python scripts/run_public_demo.py --config configs/demo/public_demo_v1.json --output outputs/public_demo/my_run
python scripts/verify_public_demo.py --input outputs/public_demo/my_run
```

The solver example compares the unchanged production solver with continuous
analytical/manufactured references across 17 configurations. The demo generates
its own labels, trains source and target production models, selects a validation
checkpoint, reloads it in a new process, evaluates two held-out cases, and
independently recomputes saved metrics with NumPy. See the [solver protocol](docs/revision_solver_verification.md)
and [demo guide](docs/public_demo.md) for definitions and limitations. These
examples do not reproduce the historical P3/P5 campaign or establish superiority.

Exact canonical P4 replay has a stricter contract than ordinary numerical
portability: the frozen plan records Python 3.11.9 and package versions and
requires bitwise design equality. The fresh Windows environment accepts the
plan; Linux Python 3.12.3 rejects tiny differences in 33 HTC values and the
Python provenance mismatch. Read the [canonical replay study](docs/revision_canonical_replay.md)
before attempting the original 512-case replay. No numerical-tolerance bypass
has been added.

## Repository layout

- `src/cdcureno/`: installable Python package.
- `scripts/`: auditable command-line workflows.
- `configs/`: frozen experiment, model, data, and solver configurations.
- `splits/`: immutable complete-case split and benchmark manifests.
- `tests/`: regression, contract, and scientific-integrity tests.
- `analysis/`: validation records and claim-boundary documents.
- `outputs/audit/` and `outputs/tables/`: compact P0--P5 release evidence.
- `manuscript/softwarex_overleaf/`: SoftwareX source and traceability files.

## Upstream ResFNO acquisition and checksums

The upstream ResFNO repository and its `.mat` arrays do not have a license
assertion recorded by this project and are therefore not redistributed.
Acquire them from the upstream project under its terms and use the pinned
revision recorded in `DATA_MANIFEST.csv`:

```bash
git clone https://github.com/gengxiangc/ResFNO external/ResFNO
git -C external/ResFNO checkout 78cc9f7851725f9b4183bd7adacba01f9d025c76
```

Expected inputs and SHA-256 values are:

| Input | Repository-relative path | SHA-256 |
|---|---|---|
| Case1 | `external/ResFNO/data/Case1.mat` | `5ec5d1a879bd4a398f856206008402c0a2f7abf92fb71f890fb753ff3e88f5a3` |
| double_hold_200_HL | `external/ResFNO/data/double_hold_200_HL.mat` | `3a55c5a9155b7d9c79064f645f01ce0b675933dc1c173250aa610f8c49f7975b` |

On systems with `sha256sum`, verify with:

```bash
sha256sum external/ResFNO/data/Case1.mat
sha256sum external/ResFNO/data/double_hold_200_HL.mat
```

On PowerShell, use `Get-FileHash -Algorithm SHA256 <path>`. Do not continue if
a hash differs from `DATA_MANIFEST.csv`.

## P0--P1 legacy audit

The audit is read-only with respect to the upstream checkout:

```bash
python scripts/audit_legacy_repo.py \
  --repo-root external/ResFNO \
  --output-dir outputs/audit \
  --dry-run
python scripts/audit_legacy_repo.py \
  --repo-root external/ResFNO \
  --output-dir outputs/audit
```

Expected products include `repo_inventory.json`, `data_shapes.json`,
`legacy_issues.md`, `license_report.md`, and `reproduction_plan.md` under
`outputs/audit/`. Historical P1 field reconstruction requires its separately
preserved run directories; compact release evidence remains available in
`outputs/tables/p1_common125_correction_v2.json`.

## P3 source prerequisites

P3 generation, training, and public-case validation are not zero-data
workflows. They require the upstream inputs above and, for saved-model
verification, the corresponding processed source dataset, split, and
checkpoint. Compact frozen results are provided in
`outputs/tables/p3_gate_summary.json`; large source arrays and checkpoints are
not distributed in Git.

## P4 plan, generation, validation, and gate

The canonical P4 sequence is deliberately staged. Review the immutable plan
before generating labels:

```bash
# 1. Validate plan integrity and immutable-generation contracts.
python -m pytest -q tests/unit/test_p4_dataset_pipeline.py

# 2. Optionally create a noncanonical review copy; no labels are generated.
python scripts/generate_p4_2d.py --tier core --plan-only \
  --plan tmp/p4_2d_core_v1_plan_release_review.json

# 3. Generate from the included immutable pre-label plan.
python scripts/generate_p4_2d.py --tier core \
  --plan splits/p4_2d_core_v1_plan.json

# 4. Validate against the separately acquired public Case1 input.
python scripts/validate_p4_solver.py \
  --input external/ResFNO/data/Case1.mat

# 5. Verify the frozen gate after the generated arrays are available.
python scripts/verify_p4_gate.py
```

The canonical plan is bound to the original pre-label commit and must not be
regenerated in place by a later documentation release. The optional review
copy records the current release commit; its case definitions and code hashes
must remain identical, while its provenance hash is expected to differ. The
production command loads the included canonical plan rather than replacing it.

The core field arrays occupy exactly `917,819,936` bytes (approximately
918 MB). The generator additionally enforces a 512 MiB free-disk margin and a
1 GiB free-memory preflight; allow at least about 1.5 GB of free disk. The
default is up to four CPU workers and two in-flight cases per worker. Runtime
depends strongly on CPU and storage performance and is not asserted by this
release.

Generation is resumable. Re-running the same hash-bound command reopens the
staging arrays and skips verified completion records. Failed completion
records are preserved. Use `--retry-failed` only as an explicit recovery
decision after diagnosing the recorded failure; it is never applied silently.

Expected final files include `temperature_K.npy`, `alpha.npy`,
`composite_mask.npy`, the remaining exogenous arrays, dataset metadata, and
the canonical manifest under `data/processed/p4_2d_core_v1/` and `splits/`.
The compact verified summary is `outputs/tables/p4_gate_summary.json`.

## P5 checkpoint inflation and restriction verification

The two P5 commands are not consecutive zero-data dry runs. First prepare the
P3 source checkpoint and P4 arrays, then persist the inflated target:

```bash
python scripts/inflate_causal_1d_to_2d_checkpoint.py \
  --source outputs/runs/p3-source-causal-v1-seed0-fix1/checkpoints/best.pt \
  --target-config configs/model/cdcureno_causal_2d.yaml \
  --output outputs/checkpoints/p5_causal_inflated_source.pt \
  --report outputs/tables/p5_causal_checkpoint_inflation.json
```

Verify it with every prerequisite stated explicitly:

```bash
python scripts/verify_p5_causal_restriction.py \
  --source-checkpoint outputs/runs/p3-source-causal-v1-seed0-fix1/checkpoints/best.pt \
  --target-checkpoint outputs/checkpoints/p5_causal_inflated_source.pt \
  --target-config configs/model/cdcureno_causal_2d.yaml \
  --source-data data/processed/p3_source_1d_v3.npz \
  --source-split splits/p3_source_1d_v3.json \
  --p4-plan splits/p4_2d_core_v1_plan.json \
  --p4-manifest splits/p4_2d_core_v1.json \
  --p4-id-split splits/2d_id_v1.json \
  --p4-array-root data/processed/p4_2d_core_v1 \
  --output outputs/tables/p5_causal_restriction_gate.json
```

Adding `--dry-run` to either command suppresses its output write; it does not
remove any input prerequisite. The release contains compact verified P5
summaries but not the source/target checkpoints or generated arrays.

## Reproducing manuscript numbers

`manuscript/softwarex_overleaf/SOURCE_TRACEABILITY.csv` maps every reported
number or bounded statement to a compact artifact, source locator, and scope
note. The principal release summaries are:

- `outputs/tables/p2_extended_seed0_summary.json`
- `outputs/tables/p3_gate_summary.json`
- `outputs/tables/p4_gate_summary.json`
- `outputs/tables/p5_causal_restriction_gate.json`
- `outputs/tables/p5_rp_ffno_pilot_gate.json`

P2 and the P5 transfer pilot are one-seed descriptive or validation-only
evidence. They are not confirmatory superiority results.

## Known limitations and redistribution boundary

- The canonical benchmark uses a rectangular grid, reference-temperature
  conductivity, and a unidirectional material model.
- Geometry OOD is deferred.
- The BDF cross-check shares physical inputs with the primary solver.
- Upstream ResFNO code/data and generated P4 arrays are not distributed here.
- P6 is unfinished and excluded from the SoftwareX evidence claim.

Generated arrays and third-party datasets are not implicitly relicensed by
this repository. Use only documented acquisition paths, checksums, and
upstream terms.

## Authors and support

- Hyojin Park, Gyeongnam Intelligence Innovation Center, Kyungnam University.
- Nam-Hyun Yoo, Computer Engineering, Kyungnam University.
- Jinhong Yang, Department of Medical Information Technology, Inje University.

Correspondence and support: `jinhong@inje.ac.kr`.

## Citation

Citation metadata are provided in `CITATION.cff`. Cite the exact tagged release
used in an analysis and, once published, the accompanying SoftwareX article.

## License

CD-CureNO is open-source software released under the
[Apache License 2.0](LICENSE.txt) (`Apache-2.0`). The license permits use,
modification, and redistribution, including commercial use, subject to its
terms. Third-party code and data retain their own licenses and are not covered
by the CD-CureNO license merely because they are referenced by this project.
