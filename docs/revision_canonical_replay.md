# Canonical P4 replay: measured environment compatibility

Date: 2026-09-08. This diagnosis calls the unmodified `build_case_plan` and
`_validate_plan` on `baseline_v002` at commit
`d9ee97f2d231d7aa2088139548a7352dfcb51efe`. No frozen plan, validator, code,
configuration, generated dataset, or environment was changed. The helper writes
only diagnostic reports. The full failed generation attempt remains separately
preserved by the main task in `runs/p4_core_linux_full_v1`.

## Finding

**The canonical plan validates unchanged in the clean Windows Python 3.11.9
environment. It fails the exact scientific-content comparison in the clean
Linux Python 3.12.3 environment because of last-bit differences in 33 top-boundary
heat-transfer coefficient values. Code/configuration hashes and CRLF are not
the cause in the inspected baseline checkout.**

| Comparison | Clean Windows | Clean Linux (WSL2) |
|---|---|---|
| Python | 3.11.9 | 3.12.3 |
| NumPy / SciPy / PyYAML | 2.2.6 / 1.15.3 / 6.0.2 | 2.2.6 / 1.15.3 / 6.0.2 |
| `_validate_plan` result | PASS | `Plan scientific design differs from the deterministic rebuilt plan.` |
| Scientific leaf differences | 0 | 60 |
| Breakdown | None | 33 `top_h_W_m2_K` values + 27 case-definition hashes |
| Maximum absolute numerical difference | 0 | 2.842170943040401e-14 W/(m2 K) |
| Additional provenance differences | Allowed Git commit difference | Git commit and Python version difference |

All 33 numerical differences are exactly **one ULP** (one adjacent float64
spacing at the stored value), as checked with `math.ulp`. All 60 scientific
leaf differences are within 27 of the 512 cases. The
remaining scientific fields, including raw Sobol coordinates, splits, cure-cycle
air temperatures, material parameters, resolved configuration, and code hashes,
are identical under recursive comparison. The 27 case hashes differ because
their JSON-serialized case definitions contain the 33 differing floating values.

For example, case 2, top boundary element 9 is `88.41953930340716` in the frozen
plan and `88.41953930340715` in the Linux rebuild. Their absolute difference is
1.4210854715202004e-14. Exact canonical JSON comparison intentionally rejects
this difference even though it is numerically tiny.

The affected array is formed at `scripts/generate_p4_2d.py` lines 631–635 using
`np.cos`, mean centering, and amplitude/offset operations. This pattern is
consistent with platform/runtime floating-point last-bit variation. This
diagnosis does not separately isolate the effects of Python version, NumPy
wheel/build, SIMD dispatch, or system mathematical library, and therefore does
not identify one of them as the proven individual cause.

## Source and configuration hashes

For each file below, **frozen-plan hash = working-tree hash = Git blob hash**.
All inspected working-tree files contain LF line endings and zero CRLF pairs.
The hashes of hypothetical CRLF conversions differ, but no such conversion is
present in this baseline checkout.

| File | SHA-256 |
|---|---|
| `scripts/generate_p4_2d.py` | `66be46e77f73501964f28eae2000700bef90cf7fff5fd935aad7d7253a300097` |
| `src/cdcureno/physics/as4_8552.py` | `2e1b1805c91b558d6f47831fd9de84f041dafd2ecbd249251bc7dd95c8795dab` |
| `src/cdcureno/solvers/conservative_2d.py` | `0c8d2433bc4c0b78949aa6b64fe9b1ea74ac5fde56d94d72088cce533c2a0a41` |
| `configs/data/p4_2d_benchmark_v1.yaml` | `c80f0d81dbb52a8747c13a6fe09519f0ff96b3d16f42af1f3863a5ea62f80d0f` |

The frozen plan records generating commit
`6d0b31f63428e00524a89445e9f4d7148f46d8f7`, while the release checkout has
`d9ee97f2d231d7aa2088139548a7352dfcb51efe`. The validator explicitly permits
this Git commit difference. Consequently Windows has only two full-plan
differences: Git provenance and the resulting overall plan hash. Both are
excluded from the scientific-content equality comparison.

## Separate Python provenance restriction

Even if the Linux numeric differences were absent, the frozen plan specifies
Python **3.11.9** and the current clean Linux interpreter is **3.12.3**.
The validator separately requires exact equality of Python/NumPy/SciPy/PyYAML
provenance after the scientific comparison. Therefore this particular Linux
environment is not a supported exact-replay environment for the existing
canonical plan. Fixing only a path check or line endings cannot resolve that
additional restriction. No validator bypass was used to demonstrate this;
the restriction is directly visible in the unmodified provenance loop.

## Implications for W4 and manuscript wording

The immediate compatible path for the requested full canonical regeneration
measurement is the clean Windows 3.11.9 environment with the unchanged frozen
plan and baseline code. A passing plan check alone does not establish that the
full generation succeeds, produces the recorded array hashes, or has any
particular runtime or memory footprint; the full run must supply those results.

These results do not support a claim of cross-platform byte-identical
canonical replay. Package-wide Linux install/CI compatibility should be distinguished from exact
replay of this historical canonical plan. Documentation should name the
validated replay environment explicitly and preserve the Linux failure as
evidence of the current boundary. A future portable-plan policy would require
an explicit design decision and separately versioned evidence; this diagnosis
does not recommend editing the frozen plan or silently relaxing equality.

## Diagnostic artifacts

- `outputs/revision_solver_verification/plan_diagnosis_windows.json`: complete recursive diff and file/hash evidence.
- `outputs/revision_solver_verification/plan_diagnosis_linux.json`: complete recursive diff and file/hash evidence.
- `outputs/revision_solver_verification/diagnose_frozen_plan.py`: read-only helper used for both reports, with CPU
  numerical-library threads limited to two.

Windows interpreter: `C:/cdcureno-r1-20260908/env_windows/Scripts/python.exe`.
Linux interpreter: `/home/jinhong/.venvs/cdcureno-softwarex-r1-20260908/bin/python`.
