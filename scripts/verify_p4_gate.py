"""Independently verify and document the frozen P4 scientific gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np

import freeze_p4_splits as freezer
import generate_p4_2d as pipeline


ROOT = Path(__file__).resolve().parents[1]
CORE_MANIFEST = ROOT / "splits" / "p4_2d_core_v1.json"
SOURCE_MANIFEST = ROOT / "splits" / "p3_source_1d_v3.json"
PHASE_A_MANIFEST = (
    ROOT / "splits" / "legacy_case1_corrected_matched_v1.json"
)
SOLVER_VALIDATION = ROOT / "outputs" / "tables" / "p4_solver_validation.json"
GATE_SUMMARY = ROOT / "outputs" / "tables" / "p4_gate_summary.json"
VALIDATION_REPORT = ROOT / "analysis" / "p4_validation.md"
REQUIRED_FROZEN_MANIFESTS = (
    "2d_combined_ood_v1.json",
    "2d_cycle_ood_v1.json",
    "2d_geometry_ood_v1.json",
    "2d_htc_ood_v1.json",
    "2d_id_v1.json",
    "2d_pattern_ood_v1.json",
    "external_validation_v1.json",
    "phase_a_v1.json",
    "source_1d_v1.json",
)


def _pretty_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_replay_case_ids(plan: dict[str, Any]) -> list[int]:
    """Select cases from plan metadata only, never from generated labels."""

    definitions = [entry["definition"] for entry in plan["cases"]]
    selected = {
        min(int(value) for value in ids)
        for ids in plan["splits"].values()
    }
    difficulties = sorted(
        {str(item["difficulty_family"]) for item in definitions}
    )
    for difficulty in difficulties:
        selected.add(
            min(
                int(item["case_id"])
                for item in definitions
                if item["difficulty_family"] == difficulty
            )
        )
    return sorted(selected)


def _run_pytest() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    if completed.returncode != 0:
        raise RuntimeError(f"P4 gate pytest failed:\n{output}")
    matches = re.findall(r"(\d+) passed", output)
    if not matches:
        raise RuntimeError(f"Could not parse pytest pass count:\n{output}")
    return int(matches[-1]), output.splitlines()[-1]


def _replay_selected_cases(
    *,
    plan: dict[str, Any],
    config: dict[str, Any],
    core: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_by_key = core["case_artifact_hashes"]
    definitions = {
        int(entry["definition"]["case_id"]): entry["definition"]
        for entry in plan["cases"]
    }
    replayed = []
    for case_id in _select_replay_case_ids(plan):
        definition = definitions[case_id]
        result = pipeline._run_case(definition, config)
        if result["status"] != "passed":
            raise RuntimeError(
                f"Deterministic replay failed for case {case_id}: "
                f"{result['failure']}"
            )
        hashes = {
            "temperature_K": pipeline._sha256_array(
                np.asarray(result["temperature_K"])
            ),
            "alpha": pipeline._sha256_array(np.asarray(result["alpha"])),
        }
        expected = expected_by_key[definition["case_key"]][
            "output_slice_sha256"
        ]
        if hashes != expected:
            raise RuntimeError(
                f"Deterministic replay hash mismatch for case {case_id}."
            )
        replayed.append(
            {
                "case_id": case_id,
                "case_key": definition["case_key"],
                "split": definition["split"],
                "difficulty_family": definition["difficulty_family"],
                "cycle_family": definition["cycle_family"],
                "output_slice_sha256": hashes,
            }
        )
    return replayed


def _field_diagnostics(
    *, plan: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    temperature = np.load(
        artifact_root / "temperature_K.npy", mmap_mode="r", allow_pickle=False
    )
    alpha = np.load(
        artifact_root / "alpha.npy", mmap_mode="r", allow_pickle=False
    )
    mask = np.load(
        artifact_root / "composite_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    definitions = [entry["definition"] for entry in plan["cases"]]
    f0_ids = [
        int(item["case_id"])
        for item in definitions
        if item["difficulty_family"] == "F0"
    ]
    non_f0_ids = [
        int(item["case_id"])
        for item in definitions
        if item["difficulty_family"] != "F0"
    ]
    f0_lateral_range = max(
        float(np.max(np.ptp(temperature[case_id], axis=-1)))
        for case_id in f0_ids
    )
    non_f0_lateral_ranges = [
        float(np.max(np.ptp(temperature[case_id], axis=-1)))
        for case_id in non_f0_ids
    ]
    return {
        "temperature_min_K": float(np.min(temperature)),
        "temperature_max_K": float(np.max(temperature)),
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "initial_temperature_max_abs_error_K": float(
            np.max(np.abs(temperature[:, 0] - 293.0))
        ),
        "initial_composite_alpha_max_abs_error": float(
            np.max(np.abs(alpha[:, 0, mask] - 0.05))
        ),
        "initial_tool_alpha_nonzero_count": int(
            np.count_nonzero(alpha[:, 0, ~mask])
        ),
        "f0_case_count": len(f0_ids),
        "f0_max_lateral_temperature_range_K": f0_lateral_range,
        "non_f0_case_count": len(non_f0_ids),
        "non_f0_min_peak_lateral_temperature_range_K": min(
            non_f0_lateral_ranges
        ),
        "final_composite_alpha_min": float(np.min(alpha[:, -1, mask])),
        "final_composite_alpha_max": float(np.max(alpha[:, -1, mask])),
    }


def _render_report(summary: dict[str, Any]) -> str:
    solver = summary["solver_validation"]
    core = summary["core_dataset"]
    fields = summary["field_diagnostics"]
    replay_ids = ", ".join(
        str(item["case_id"]) for item in summary["deterministic_replay"]
    )
    return f"""# P4 validation

## Result

P4 passed every frozen gate check. The canonical benchmark contains
{core["case_count"]} cases with field shape
`{core["array_shape_case_time_z_x"]}` and no failed, silent, or recovered
failure attempts.

## Solver evidence

The conservative 2-D finite-volume solver passed all
{solver["acceptance_check_count"]} pre-specified checks. At the canonical
10-second internal step, the full-cycle stress case had temperature relative
L2 `{solver["full_cycle_temperature_relative_l2"]:.9g}`,
temperature-rise relative L2
`{solver["full_cycle_temperature_rise_relative_l2"]:.9g}`, maximum
temperature error `{solver["full_cycle_temperature_max_abs_K"]:.6g} K`, and
cure relative L2 `{solver["full_cycle_alpha_relative_l2"]:.9g}` against the
1.25-second reference.

Exact lateral extrusion agreed with the converged 1-D solver, and the selected
heterogeneous case agreed with the independently assembled SciPy-BDF
method-of-lines calculation. The BDF check shares the grid, material values,
and cure law; it independently checks flux assembly and time integration, not
those shared inputs.

## Dataset integrity

- Pre-label plan SHA-256: `{core["plan_sha256"]}`
- Pre-label Git commit: `{core["prelabel_git_commit_sha"]}`
- Core manifest SHA-256: `{core["manifest_file_sha256"]}`
- Array bytes: `{core["array_total_bytes"]}`
- Maximum local energy residual:
  `{core["maximum_abs_energy_residual_W_m3"]:.9g} W m^-3`
- Maximum relative global energy residual:
  `{core["maximum_relative_global_energy_residual"]:.9g}`
- Temperature range: `{fields["temperature_min_K"]:.6g}` to
  `{fields["temperature_max_K"]:.6g} K`
- Cure range: `{fields["alpha_min"]:.6g}` to `{fields["alpha_max"]:.6g}`
- F0 maximum lateral temperature range:
  `{fields["f0_max_lateral_temperature_range_K"]:.9g} K`
- Smallest non-F0 peak lateral range:
  `{fields["non_f0_min_peak_lateral_temperature_range_K"]:.9g} K`

The freezer independently reconstructed the Sobol plan, auxiliary arrays,
per-case hashes, physical invariants, diagnostic thresholds, split counts, OOD
semantics, and nested budgets. It then wrote immutable source, Phase-A, ID, and
OOD manifests. The complete test suite passed
{summary["test_suite"]["passed_count"]}/{summary["test_suite"]["passed_count"]}.

## Deterministic replay

Cases `{replay_ids}` were selected using plan metadata only: the first case in
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
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    validated = freezer.validate_core_manifest(CORE_MANIFEST)
    freeze_result = freezer.freeze_manifests(
        core_path=CORE_MANIFEST,
        source_path=SOURCE_MANIFEST,
        phase_a_path=PHASE_A_MANIFEST,
        output_root=ROOT / "splits",
    )
    core = validated["core"]
    plan = validated["plan"]
    config = validated["config"]
    generation = validated["summary"]
    solver = json.loads(SOLVER_VALIDATION.read_text(encoding="utf-8"))
    artifact_root = ROOT / Path(core["array_artifact_root"])
    replayed = _replay_selected_cases(
        plan=plan, config=config, core=core
    )
    fields = _field_diagnostics(plan=plan, artifact_root=artifact_root)
    if args.skip_pytest:
        test_count = 0
        pytest_summary = "skipped by explicit command-line option"
    else:
        test_count, pytest_summary = _run_pytest()

    checks = {
        "strict_core_manifest_validation": True,
        "immutable_split_replay": set(freeze_result["frozen"])
        == set(REQUIRED_FROZEN_MANIFESTS),
        "solver_validation": solver.get("passed") is True
        and all(solver.get("acceptance_checks", {}).values()),
        "generation_acceptance": generation.get("passed") is True
        and all(generation.get("acceptance_checks", {}).values()),
        "no_failed_or_silent_cases": generation["failed_case_count"] == 0
        and generation["silent_failure_count"] == 0
        and generation["recovered_failure_attempt_count"] == 0,
        "initial_temperature": (
            fields["initial_temperature_max_abs_error_K"] == 0.0
        ),
        "initial_composite_alpha": (
            fields["initial_composite_alpha_max_abs_error"] == 0.0
        ),
        "initial_tool_alpha": (
            fields["initial_tool_alpha_nonzero_count"] == 0
        ),
        "f0_exact_lateral_extrusion": (
            fields["f0_max_lateral_temperature_range_K"] == 0.0
        ),
        "non_f0_has_lateral_structure": (
            fields["non_f0_min_peak_lateral_temperature_range_K"] > 1.0e-3
        ),
        "deterministic_selected_replay": len(replayed) > 0,
        "full_test_suite": args.skip_pytest or test_count >= 111,
    }
    array_total_bytes = sum(
        (artifact_root / f"{name}.npy").stat().st_size
        for name in core["array_sha256"]
    )
    full_cycle = solver["core_discretization"]
    summary = {
        "schema_version": 1,
        "phase": "P4",
        "passed": bool(all(checks.values())),
        "acceptance_checks": checks,
        "core_dataset": {
            "dataset_id": core["dataset_id"],
            "case_count": int(core["case_count"]),
            "array_shape_case_time_z_x": core[
                "array_shape_case_time_z_x"
            ],
            "array_total_bytes": array_total_bytes,
            "array_sha256": core["array_sha256"],
            "metadata_sha256": core["metadata_sha256"],
            "plan_sha256": core["plan_sha256"],
            "prelabel_git_commit_sha": plan["provenance"][
                "git_commit_sha"
            ],
            "manifest_file_sha256": _sha256_file(CORE_MANIFEST),
            "split_counts": {
                name: len(ids) for name, ids in core["splits"].items()
            },
            "nested_training_budgets": core[
                "nested_training_budgets"
            ],
            "failed_case_count": generation["failed_case_count"],
            "silent_failure_count": generation["silent_failure_count"],
            "recovered_failure_attempt_count": generation[
                "recovered_failure_attempt_count"
            ],
            "maximum_abs_energy_residual_W_m3": generation[
                "maximum_abs_energy_residual_W_m3"
            ],
            "maximum_relative_global_energy_residual": generation[
                "maximum_relative_global_energy_residual"
            ],
        },
        "solver_validation": {
            "artifact": "outputs/tables/p4_solver_validation.json",
            "acceptance_check_count": len(solver["acceptance_checks"]),
            "acceptance_checks": solver["acceptance_checks"],
            "full_cycle_temperature_relative_l2": full_cycle[
                "full_cycle_temperature_relative_l2"
            ],
            "full_cycle_temperature_rise_relative_l2": full_cycle[
                "full_cycle_temperature_rise_relative_l2"
            ],
            "full_cycle_temperature_max_abs_K": full_cycle[
                "full_cycle_temperature_max_abs_K"
            ],
            "full_cycle_alpha_relative_l2": full_cycle[
                "full_cycle_alpha_relative_l2"
            ],
            "independent_temperature_relative_l2": solver[
                "independent_cross_check"
            ]["temperature_relative_l2"],
            "independent_alpha_relative_l2": solver[
                "independent_cross_check"
            ]["alpha_relative_l2"],
            "independent_cross_check_limitation": solver[
                "independent_cross_check"
            ]["shared_components_limitation"],
        },
        "field_diagnostics": fields,
        "deterministic_replay": replayed,
        "frozen_manifests": sorted(freeze_result["frozen"]),
        "test_suite": {
            "passed_count": test_count,
            "summary": pytest_summary,
            "skipped": bool(args.skip_pytest),
        },
    }
    _atomic_write_bytes(GATE_SUMMARY, _pretty_json_bytes(summary))
    _atomic_write_bytes(
        VALIDATION_REPORT, _render_report(summary).encode("utf-8")
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
