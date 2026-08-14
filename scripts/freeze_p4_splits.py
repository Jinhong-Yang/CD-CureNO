"""Validate the canonical P4 core artifact and immutably freeze split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
from typing import Any

import numpy as np

import generate_p4_2d as p4_pipeline


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits"
REQUIRED_CORE_SPLITS = (
    "train",
    "validation",
    "id_test",
    "cycle_ood",
    "htc_ood",
    "pattern_ood",
    "combined_ood",
)
REQUIRED_ARRAYS = (
    "temperature_K",
    "alpha",
    "air_temperature_K",
    "top_h_W_m2_K",
    "time_s",
    "x_m",
    "z_m",
    "composite_mask",
)
EXPECTED_DTYPES = {
    "temperature_K": np.dtype("float32"),
    "alpha": np.dtype("float32"),
    "air_temperature_K": np.dtype("float32"),
    "top_h_W_m2_K": np.dtype("float32"),
    "time_s": np.dtype("float64"),
    "x_m": np.dtype("float64"),
    "z_m": np.dtype("float64"),
    "composite_mask": np.dtype("bool"),
}
DIAGNOSTIC_THRESHOLDS = (
    (
        "maximum_abs_energy_residual_W_m3",
        "maximum_abs_energy_residual_W_m3_max",
        "local_energy",
    ),
    (
        "maximum_relative_global_energy_residual",
        "maximum_relative_global_energy_residual_max",
        "global_energy",
    ),
    (
        "maximum_interface_flux_imbalance_W",
        "maximum_interface_flux_imbalance_W_max",
        "interface_flux_continuity",
    ),
    (
        "maximum_temperature_interface_jump_K",
        "maximum_temperature_interface_jump_K_max",
        "interface_temperature_continuity",
    ),
    (
        "maximum_robin_flux_imbalance_W",
        "maximum_robin_flux_imbalance_W_max",
        "robin_flux_continuity",
    ),
    (
        "maximum_converged_coupling_update_K",
        "maximum_converged_coupling_update_K_max",
        "maximum_coupling_update",
    ),
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json_bytes(list(contiguous.shape)))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


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


def _repo_relative_path(path: Path, project_root: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"{label} must be inside the repository.") from None


def _resolve_path(value: Any, project_root: Path, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty repository-relative path.")
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators.")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or Path(value).is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a normalized repository-relative path.")
    resolved = (project_root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        raise ValueError(f"{label} escapes the repository.") from None
    return resolved


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_split_partition(
    split_map: dict[str, Any],
    expected_case_ids: set[int],
    expected_counts: dict[str, Any],
) -> None:
    if set(split_map) != set(REQUIRED_CORE_SPLITS):
        raise ValueError(
            f"Core split keys must be {list(REQUIRED_CORE_SPLITS)}."
        )
    observed: set[int] = set()
    for name in REQUIRED_CORE_SPLITS:
        raw_ids = split_map[name]
        if not isinstance(raw_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_ids
        ):
            raise ValueError(f"Split {name} must contain integer case IDs.")
        ids = list(raw_ids)
        if len(ids) != int(expected_counts[name]):
            raise ValueError(
                f"Split {name} count mismatch: {len(ids)} != "
                f"{int(expected_counts[name])}."
            )
        if len(ids) != len(set(ids)):
            raise ValueError(f"Split {name} contains duplicate case IDs.")
        overlap = observed & set(ids)
        if overlap:
            raise ValueError(
                f"Split {name} overlaps preceding splits: {sorted(overlap)}"
            )
        observed.update(ids)
    if observed != expected_case_ids:
        raise ValueError(
            "Core split coverage mismatch; "
            f"missing={sorted(expected_case_ids - observed)}, "
            f"extra={sorted(observed - expected_case_ids)}"
        )


def _validate_semantics(
    plan: dict[str, Any],
    split_map: dict[str, list[int]],
) -> None:
    definitions = {
        int(entry["definition"]["case_id"]): entry["definition"]
        for entry in plan["cases"]
    }
    id_cycle_splits = {
        "train",
        "validation",
        "id_test",
        "htc_ood",
        "pattern_ood",
    }
    ranges = plan["resolved_config"]["sampling"]["ranges"]

    def in_half_open(value: float, bounds: list[float]) -> bool:
        return float(bounds[0]) <= value < float(bounds[1])

    for split, ids in split_map.items():
        for case_id in ids:
            definition = definitions[int(case_id)]
            cycle = definition["cycle_family"]
            if split in id_cycle_splits and cycle not in {
                "single_hold",
                "two_hold",
            }:
                raise ValueError(
                    f"Case {case_id} leaks OOD cycle {cycle} into {split}."
                )
            if split in {"cycle_ood", "combined_ood"} and cycle not in {
                "smart_cure",
                "three_hold",
            }:
                raise ValueError(
                    f"Case {case_id} lacks an OOD cycle in {split}."
                )
            difficulty = definition["difficulty_family"]
            if difficulty in {"F0", "F2_left", "F2_both"}:
                if (
                    float(definition["realized_top_h_amplitude_W_m2_K"]) != 0.0
                    or int(definition["realized_top_h_frequency_per_width"]) != 0
                ):
                    raise ValueError(
                        f"Uniform-top case {case_id} has nonzero realized pattern."
                    )
            if not np.isclose(
                float(definition["realized_top_h_mean_W_m2_K"]),
                float(definition["requested_top_h_center_W_m2_K"]),
                atol=1.0e-10,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Case {case_id} realized/requested top HTC center mismatch."
                )
            htc_ranges = (
                ranges["ood"]
                if split in {"htc_ood", "combined_ood"}
                else ranges["id"]
            )
            for field, range_name in (
                ("bottom_h_W_m2_K", "bottom_h_W_m2_K"),
                (
                    "requested_top_h_center_W_m2_K",
                    "top_h_center_W_m2_K",
                ),
                ("requested_edge_h_W_m2_K", "edge_h_W_m2_K"),
            ):
                if not in_half_open(
                    float(definition[field]), htc_ranges[range_name]
                ):
                    raise ValueError(
                        f"Case {case_id} {field} is outside its {split} range."
                    )
            amplitude_ranges = (
                ranges["ood"]
                if split in {"htc_ood", "combined_ood"}
                else ranges["id"]
            )
            if not in_half_open(
                float(definition["requested_top_h_amplitude_W_m2_K"]),
                amplitude_ranges["top_h_amplitude_W_m2_K"],
            ):
                raise ValueError(
                    f"Case {case_id} requested top amplitude is out of range."
                )
            frequency_ranges = (
                ranges["ood"]
                if split in {"pattern_ood", "combined_ood"}
                else ranges["id"]
            )
            if int(definition["requested_top_h_frequency_per_width"]) not in {
                int(value)
                for value in frequency_ranges[
                    "top_h_frequencies_per_width"
                ]
            }:
                raise ValueError(
                    f"Case {case_id} requested top frequency is out of range."
                )


def _expected_array_contract(
    plan: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, np.ndarray]]:
    settings = config["tiers"]["core"]
    count = int(settings["case_count"])
    duration = float(config["time"]["duration_s"])
    interval = float(config["time"]["output_interval_s"])
    nt = int(round(duration / interval)) + 1
    width = float(config["geometry"]["width_m"])
    total_z = float(config["geometry"]["tool_thickness_m"]) + float(
        config["geometry"]["composite_thickness_m"]
    )
    dx = float(settings["spacing_x_m"])
    dz = float(settings["spacing_z_m"])
    nx = int(round(width / dx))
    nz = int(round(total_z / dz))
    shapes = {
        "temperature_K": (count, nt, nz, nx),
        "alpha": (count, nt, nz, nx),
        "air_temperature_K": (count, nt),
        "top_h_W_m2_K": (count, nx),
        "time_s": (nt,),
        "x_m": (nx,),
        "z_m": (nz,),
        "composite_mask": (nz, nx),
    }
    definitions = [entry["definition"] for entry in plan["cases"]]
    time_s = np.arange(nt, dtype=np.float64) * interval
    x_m = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    z_m = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    composite_by_z = z_m >= float(config["geometry"]["tool_thickness_m"])
    mask = np.broadcast_to(composite_by_z[:, None], (nz, nx)).copy()
    expected = {
        "air_temperature_K": np.asarray(
            [item["air_temperature_K"] for item in definitions],
            dtype=np.float32,
        ),
        "top_h_W_m2_K": np.asarray(
            [item["top_h_W_m2_K"] for item in definitions],
            dtype=np.float32,
        ),
        "time_s": time_s,
        "x_m": x_m,
        "z_m": z_m,
        "composite_mask": mask.astype(np.bool_),
    }
    return shapes, expected


def _finite_nonnegative_metric(
    solver: dict[str, Any],
    name: str,
    *,
    case_id: int,
) -> float:
    raw_value = solver.get(name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(
            f"Case {case_id} solver diagnostic {name} is missing or invalid."
        )
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"Case {case_id} solver diagnostic {name} is not finite/nonnegative."
        )
    return value


def _validate_acceptance_evidence(
    *,
    metadata_by_id: dict[int, dict[str, Any]],
    arrays: dict[str, np.ndarray],
    plan: dict[str, Any],
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    acceptance = config["acceptance"]
    mask = np.asarray(arrays["composite_mask"], dtype=np.bool_)
    diagnostic_values = {
        diagnostic: [] for diagnostic, _, _ in DIAGNOSTIC_THRESHOLDS
    }
    bound_counts: list[int] = []
    monotonicity_counts: list[int] = []
    tool_counts: list[int] = []
    finite_flags: list[bool] = []
    coupling_flags: list[bool] = []

    for entry in plan["cases"]:
        definition = entry["definition"]
        case_id = int(definition["case_id"])
        row = metadata_by_id[case_id]
        expected_metadata = {
            key: value
            for key, value in definition.items()
            if key not in {"air_temperature_K", "top_h_W_m2_K"}
        }
        expected_metadata.update(
            {
                "air_temperature_min_K": float(
                    np.min(np.asarray(definition["air_temperature_K"]))
                ),
                "air_temperature_max_K": float(
                    np.max(np.asarray(definition["air_temperature_K"]))
                ),
            }
        )
        for key, expected_value in expected_metadata.items():
            if row.get(key) != expected_value:
                raise ValueError(
                    f"Core metadata definition field {key} differs for case "
                    f"{case_id}."
                )
        if row.get("status") != "passed" or row.get("failure") is not None:
            raise ValueError(f"Case {case_id} is not a clean passed case.")
        solver = row.get("solver")
        if not isinstance(solver, dict):
            raise ValueError(f"Case {case_id} solver metadata is missing.")

        temperature = np.asarray(arrays["temperature_K"][case_id])
        alpha = np.asarray(arrays["alpha"][case_id])
        finite = bool(
            np.all(np.isfinite(temperature)) and np.all(np.isfinite(alpha))
        )
        bound_count = int(
            np.count_nonzero(
                (alpha < -1.0e-12) | (alpha > 1.0 + 1.0e-12)
            )
        )
        monotonicity_count = int(
            np.count_nonzero(np.diff(alpha, axis=0) < -1.0e-12)
        )
        tool_count = int(np.count_nonzero(alpha[:, ~mask] != 0.0))
        if solver.get("finite_fields") is not finite:
            raise ValueError(f"Case {case_id} finite-field diagnostic mismatch.")
        for name, observed, expected in (
            (
                "alpha_bound_violation_count",
                solver.get("alpha_bound_violation_count"),
                bound_count,
            ),
            (
                "alpha_monotonicity_violation_count",
                solver.get("alpha_monotonicity_violation_count"),
                monotonicity_count,
            ),
            (
                "tool_alpha_nonzero_count",
                solver.get("tool_alpha_nonzero_count"),
                tool_count,
            ),
        ):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed != expected
            ):
                raise ValueError(f"Case {case_id} {name} diagnostic mismatch.")
        configured_step = solver.get("configured_maximum_step_s")
        if (
            isinstance(configured_step, bool)
            or not isinstance(configured_step, (int, float))
            or float(configured_step)
            != float(config["tiers"]["core"]["maximum_step_s"])
        ):
            raise ValueError(f"Case {case_id} maximum-step metadata mismatch.")
        configured_tolerance = solver.get("configured_coupling_tolerance_K")
        if (
            isinstance(configured_tolerance, bool)
            or not isinstance(configured_tolerance, (int, float))
            or float(configured_tolerance)
            != float(config["solver"]["coupling_tolerance_K"])
        ):
            raise ValueError(f"Case {case_id} coupling-tolerance mismatch.")
        configured_iterations = int(
            config["solver"]["maximum_coupling_iterations"]
        )
        if solver.get("configured_maximum_coupling_iterations") != (
            configured_iterations
        ):
            raise ValueError(f"Case {case_id} coupling-iteration config mismatch.")
        used_iterations = solver.get("maximum_coupling_iterations")
        if (
            isinstance(used_iterations, bool)
            or not isinstance(used_iterations, int)
            or not 1 <= used_iterations <= configured_iterations
        ):
            raise ValueError(f"Case {case_id} coupling-iteration diagnostic invalid.")
        substeps = solver.get("substeps")
        if (
            isinstance(substeps, bool)
            or not isinstance(substeps, int)
            or substeps < 1
        ):
            raise ValueError(f"Case {case_id} substep diagnostic invalid.")
        coupling = solver.get("all_coupling_steps_converged")
        if not isinstance(coupling, bool):
            raise ValueError(f"Case {case_id} coupling status is not boolean.")

        case_failed = (
            (bool(acceptance["finite_field_required"]) and not finite)
            or bound_count > int(acceptance["alpha_bound_violation_count_max"])
            or monotonicity_count
            > int(acceptance["alpha_monotonicity_violation_count_max"])
            or tool_count > int(acceptance["tool_alpha_nonzero_count_max"])
            or (
                bool(acceptance["all_coupling_steps_converged_required"])
                and not coupling
            )
        )
        for diagnostic, threshold, _ in DIAGNOSTIC_THRESHOLDS:
            value = _finite_nonnegative_metric(
                solver, diagnostic, case_id=case_id
            )
            diagnostic_values[diagnostic].append(value)
            case_failed = case_failed or value > float(acceptance[threshold])
        if case_failed:
            raise ValueError(f"Case {case_id} fails independently recomputed acceptance.")

        finite_flags.append(finite)
        bound_counts.append(bound_count)
        monotonicity_counts.append(monotonicity_count)
        tool_counts.append(tool_count)
        coupling_flags.append(coupling)

    aggregates: dict[str, Any] = {
        diagnostic: max(values)
        for diagnostic, values in diagnostic_values.items()
    }
    aggregates.update(
        {
            "maximum_alpha_bound_violation_count": max(bound_counts),
            "maximum_alpha_monotonicity_violation_count": max(
                monotonicity_counts
            ),
            "maximum_tool_alpha_nonzero_count": max(tool_counts),
            "all_finite_fields": all(finite_flags),
            "all_coupling_steps_converged": all(coupling_flags),
        }
    )
    checks = {
        "all_cases_accounted_for": len(metadata_by_id)
        == int(config["tiers"]["core"]["case_count"]),
        "no_failed_cases": True,
        "no_silent_failures": True,
        "finite_fields": (
            not bool(acceptance["finite_field_required"])
            or bool(aggregates["all_finite_fields"])
        ),
        "alpha_bounds": int(
            aggregates["maximum_alpha_bound_violation_count"]
        )
        <= int(acceptance["alpha_bound_violation_count_max"]),
        "alpha_monotonicity": int(
            aggregates["maximum_alpha_monotonicity_violation_count"]
        )
        <= int(acceptance["alpha_monotonicity_violation_count_max"]),
        "tool_alpha": int(aggregates["maximum_tool_alpha_nonzero_count"])
        <= int(acceptance["tool_alpha_nonzero_count_max"]),
        "coupling_convergence": (
            not bool(acceptance["all_coupling_steps_converged_required"])
            or bool(aggregates["all_coupling_steps_converged"])
        ),
    }
    for diagnostic, threshold, check_name in DIAGNOSTIC_THRESHOLDS:
        checks[check_name] = (
            float(aggregates[diagnostic]) <= float(acceptance[threshold])
        )
    if summary.get("acceptance_thresholds_prespecified") != acceptance:
        raise ValueError("Generation summary acceptance thresholds mismatch.")
    for key, expected_value in aggregates.items():
        actual_value = summary.get(key)
        if (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            raise ValueError(f"Generation summary aggregate {key} mismatch.")
    summary_checks = summary.get("acceptance_checks")
    if (
        not isinstance(summary_checks, dict)
        or any(type(value) is not bool for value in summary_checks.values())
        or summary_checks != checks
    ):
        raise ValueError("Generation summary acceptance checks mismatch.")
    if summary.get("passed") is not bool(all(checks.values())):
        raise ValueError("Generation summary passed flag mismatch.")


def validate_core_manifest(
    core_path: Path,
    *,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Fully validate a generated core artifact before any split is written."""

    _repo_relative_path(core_path, project_root, label="Core manifest")
    core = _read(core_path)
    if core.get("schema_version") != 2 or core.get("phase") != "P4":
        raise ValueError("Core manifest must use P4 schema_version 2.")
    if core.get("tier") != "core" or core.get("tier_role") != "canonical":
        raise ValueError("Only a canonical core tier can be frozen.")
    config_path = _resolve_path(
        core.get("config_path"),
        project_root,
        label="Core config_path",
    )
    if (
        not config_path.is_file()
        or _sha256_file(config_path) != core.get("config_sha256")
    ):
        raise ValueError("Core authoritative-config checksum mismatch.")
    config = p4_pipeline.load_config(config_path)
    count = int(config["tiers"]["core"]["case_count"])
    if (
        isinstance(core.get("case_count"), bool)
        or core.get("case_count") != count
    ):
        raise ValueError("Core case count differs from authoritative config.")
    expected_ids = set(range(count))
    expected_split_counts = config["tiers"]["core"]["split_counts"]
    _validate_split_partition(
        core["splits"], expected_ids, expected_split_counts
    )
    status = core["generation_status"]
    passed_ids = status.get("passed_case_ids")
    if (
        not isinstance(passed_ids, list)
        or any(type(value) is not int for value in passed_ids)
        or status.get("failed_case_ids") != []
        or status.get("silent_failure_case_ids") != []
        or passed_ids != list(range(count))
    ):
        raise ValueError("Core generation-status accounting is not exact.")

    plan_path = _resolve_path(
        core.get("plan_path"),
        project_root,
        label="Core plan_path",
    )
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing pre-label plan: {plan_path}")
    plan = p4_pipeline.load_and_validate_plan(
        plan_path,
        tier="core",
        config=config,
        config_path=config_path,
        project_root=project_root,
    )
    if plan["plan_sha256"] != core.get("plan_sha256"):
        raise ValueError("Core/pre-label plan checksum mismatch.")
    if plan["dataset_id"] != core.get("dataset_id"):
        raise ValueError("Core/pre-label plan identity mismatch.")
    if plan["splits"] != core["splits"]:
        raise ValueError("Core splits differ from the pre-label plan.")
    for field in ("config_path", "config_sha256", "code_sha256", "provenance"):
        if core.get(field) != plan.get(field):
            raise ValueError(f"Core/pre-label plan {field} mismatch.")
    resolved_config = plan["resolved_config"]
    if float(resolved_config["tiers"]["core"]["maximum_step_s"]) != 10.0:
        raise ValueError("Canonical core plan must use maximum_step_s=10.")
    if (
        resolved_config["material"]["layup"]
        != "unidirectional_fibres_along_x"
    ):
        raise ValueError("Canonical core plan has the wrong anisotropic layup.")

    summary_path = _resolve_path(
        core.get("generation_summary_path"),
        project_root,
        label="Core generation_summary_path",
    )
    if (
        not summary_path.is_file()
        or _sha256_file(summary_path)
        != core.get("generation_summary_sha256")
    ):
        raise ValueError("Core generation-summary checksum mismatch.")
    summary = _read(summary_path)
    if summary.get("passed") is not True:
        raise ValueError("Core generation summary is not passed.")
    if (
        isinstance(summary.get("maximum_step_s"), bool)
        or not isinstance(summary.get("maximum_step_s"), (int, float))
        or float(summary["maximum_step_s"]) != 10.0
    ):
        raise ValueError("Core generation summary must report maximum_step_s=10.")
    if (
        summary.get("schema_version") != 2
        or summary.get("phase") != "P4"
        or summary.get("dataset_id") != core["dataset_id"]
        or summary.get("tier") != "core"
        or summary.get("tier_role") != "canonical"
        or summary.get("case_count") != count
        or summary.get("array_shape_case_time_z_x")
        != core["array_shape_case_time_z_x"]
        or summary.get("plan_sha256") != core["plan_sha256"]
        or summary.get("array_sha256") != core["array_sha256"]
    ):
        raise ValueError("Core generation summary identity/checksum mismatch.")
    for field in ("config_path", "config_sha256", "code_sha256", "provenance"):
        if summary.get(field) != plan.get(field):
            raise ValueError(f"Generation summary/pre-label plan {field} mismatch.")
    if summary.get("manifest") != _repo_relative_path(
        core_path, project_root, label="Core manifest"
    ):
        raise ValueError("Generation summary manifest path mismatch.")
    if any(
        isinstance(summary.get(field), bool)
        or not isinstance(summary.get(field), int)
        for field in (
            "failed_case_count",
            "silent_failure_count",
            "passed_case_count",
        )
    ) or (
        summary["failed_case_count"] != 0
        or summary["silent_failure_count"] != 0
        or summary["passed_case_count"] != count
        or summary.get("failed_case_ids") != []
        or summary.get("silent_failure_case_ids") != []
    ):
        raise ValueError("Core generation summary case accounting is invalid.")

    metadata_path = _resolve_path(
        core.get("metadata_path"),
        project_root,
        label="Core metadata_path",
    )
    if (
        not metadata_path.is_file()
        or _sha256_file(metadata_path) != core["metadata_sha256"]
        or summary.get("metadata") != core["metadata_path"]
    ):
        raise ValueError("Core metadata checksum mismatch.")
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metadata_rows) != count:
        raise ValueError("Core metadata row count mismatch.")
    metadata_ids = [item.get("case_id") for item in metadata_rows]
    if (
        any(type(value) is not int for value in metadata_ids)
        or metadata_ids != list(range(count))
    ):
        raise ValueError("Core metadata IDs must be ordered, exact integers.")
    metadata_by_id = {
        int(item["case_id"]): item for item in metadata_rows
    }
    if set(metadata_by_id) != expected_ids or len(metadata_by_id) != count:
        raise ValueError("Core metadata IDs are not unique and exhaustive.")
    if any(item.get("status") != "passed" for item in metadata_rows):
        raise ValueError("Core metadata contains a non-passed case.")

    artifact_root = _resolve_path(
        core.get("array_artifact_root"),
        project_root,
        label="Core array_artifact_root",
    )
    recovered_failure_count = 0
    for entries in core.get("recovered_failure_history", {}).values():
        for entry in entries:
            history_path = _resolve_path(
                entry.get("path"),
                artifact_root,
                label="Recovered failure-history path",
            )
            if (
                not history_path.is_file()
                or _sha256_file(history_path) != entry["sha256"]
            ):
                raise ValueError("Recovered failure-history checksum mismatch.")
            history_record = _read(history_path)
            if history_record.get("status") == "passed":
                raise ValueError("Recovered failure history contains a passed record.")
            recovered_failure_count += 1
    if int(summary.get("recovered_failure_attempt_count", 0)) != recovered_failure_count:
        raise ValueError("Recovered failure-history accounting mismatch.")
    if set(core["array_sha256"]) != set(REQUIRED_ARRAYS):
        raise ValueError("Core array checksum inventory is incomplete.")
    arrays: dict[str, np.ndarray] = {}
    for name in REQUIRED_ARRAYS:
        path = artifact_root / f"{name}.npy"
        if not path.is_file() or _sha256_file(path) != core["array_sha256"][name]:
            raise ValueError(f"Core array checksum mismatch: {name}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shapes, expected_auxiliary = _expected_array_contract(plan, config)
    if core.get("array_shape_case_time_z_x") != list(
        expected_shapes["temperature_K"]
    ):
        raise ValueError("Core declared field shape differs from config/grid.")
    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(f"Core array shape mismatch: {name}")
        if array.dtype != EXPECTED_DTYPES[name]:
            raise ValueError(f"Core array dtype mismatch: {name}")
    for name, expected in expected_auxiliary.items():
        if not np.array_equal(np.asarray(arrays[name]), expected):
            raise ValueError(f"Core auxiliary array content mismatch: {name}")

    definitions_by_id: dict[int, dict[str, Any]] = {}
    for entry in plan["cases"]:
        definition = entry["definition"]
        case_id = int(definition["case_id"])
        actual_definition_hash = _sha256_bytes(_canonical_json_bytes(definition))
        if actual_definition_hash != entry["case_definition_sha256"]:
            raise ValueError(f"Plan case hash mismatch for case {case_id}.")
        if core["case_definition_hashes"].get(definition["case_key"]) != actual_definition_hash:
            raise ValueError(f"Core case hash mismatch for case {case_id}.")
        definitions_by_id[case_id] = definition
        expected_inputs = {
            "air_temperature_K": _sha256_array(
                np.asarray(arrays["air_temperature_K"][case_id])
            ),
            "top_h_W_m2_K": _sha256_array(
                np.asarray(arrays["top_h_W_m2_K"][case_id])
            ),
            "case_definition": actual_definition_hash,
        }
        expected_outputs = {
            "temperature_K": _sha256_array(
                np.asarray(arrays["temperature_K"][case_id])
            ),
            "alpha": _sha256_array(np.asarray(arrays["alpha"][case_id])),
        }
        hashes = core["case_artifact_hashes"].get(definition["case_key"], {})
        if hashes.get("input_slice_sha256") != expected_inputs:
            raise ValueError(f"Core input slice hash mismatch for case {case_id}.")
        if hashes.get("output_slice_sha256") != expected_outputs:
            raise ValueError(f"Core output slice hash mismatch for case {case_id}.")
        metadata = metadata_by_id[case_id]
        if (
            metadata.get("case_definition_sha256") != actual_definition_hash
            or metadata.get("input_slice_sha256") != expected_inputs
            or metadata.get("output_slice_sha256") != expected_outputs
        ):
            raise ValueError(f"Core metadata slice hash mismatch for case {case_id}.")

    _validate_acceptance_evidence(
        metadata_by_id=metadata_by_id,
        arrays=arrays,
        plan=plan,
        config=config,
        summary=summary,
    )
    _validate_semantics(plan, core["splits"])
    raw_budgets = core["nested_training_budgets"]
    if not isinstance(raw_budgets, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_budgets
    ):
        raise ValueError("Core nested budgets must contain integers.")
    budgets = list(raw_budgets)
    configured_budgets = [
        int(value)
        for value in config["tiers"]["core"]["nested_training_budgets"]
    ]
    if (
        not budgets
        or any(value <= 0 for value in budgets)
        or budgets != sorted(set(budgets))
        or budgets != configured_budgets
    ):
        raise ValueError(
            "Core nested budgets must be positive and exactly match config."
        )
    training = [int(value) for value in core["splits"]["train"]]
    if budgets[-1] != len(training):
        raise ValueError("Core final nested budget must equal the training pool.")
    if budgets != plan["nested_training_budgets"]:
        raise ValueError("Core nested budgets differ from the pre-label plan.")
    previous: list[int] = []
    for budget in budgets:
        current = training[:budget]
        if len(current) != budget or current[: len(previous)] != previous:
            raise ValueError("Core nested budget prefixes are not exactly nested.")
        previous = current
    return {
        "core": core,
        "plan": plan,
        "config": config,
        "summary": summary,
        "case_key_by_id": {
            case_id: definition["case_key"]
            for case_id, definition in definitions_by_id.items()
        },
    }


def _prepare_immutable_writes(payloads: dict[Path, bytes]) -> None:
    """Preflight all destinations, then atomically fill only missing files."""

    mismatches = [
        str(path)
        for path, payload in payloads.items()
        if path.exists() and path.read_bytes() != payload
    ]
    if mismatches:
        raise FileExistsError(
            "Refusing to replace non-identical frozen manifests: "
            + ", ".join(sorted(mismatches))
        )
    for path, payload in payloads.items():
        if not path.exists():
            _atomic_write_bytes(path, payload)


def freeze_manifests(
    *,
    core_path: Path,
    source_path: Path,
    phase_a_path: Path,
    output_root: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    _repo_relative_path(output_root, project_root, label="Freeze output root")
    source_manifest_path = _repo_relative_path(
        source_path, project_root, label="Source manifest"
    )
    phase_a_manifest_path = _repo_relative_path(
        phase_a_path, project_root, label="Phase-A manifest"
    )
    core_manifest_path = _repo_relative_path(
        core_path, project_root, label="Core manifest"
    )
    validated = validate_core_manifest(core_path, project_root=project_root)
    core = validated["core"]
    source = _read(source_path)
    phase_a = _read(phase_a_path)
    common = {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": core["dataset_id"],
        "dataset_array_sha256": core["array_sha256"],
        "dataset_metadata_sha256": core["metadata_sha256"],
        "dataset_plan_sha256": core["plan_sha256"],
        "source_manifest": core_manifest_path,
        "source_manifest_sha256": _sha256_file(core_path),
    }
    training = [int(value) for value in core["splits"]["train"]]
    nested = {
        str(budget): training[:budget]
        for budget in (int(value) for value in core["nested_training_budgets"])
    }
    case_key_by_id = validated["case_key_by_id"]

    def hashes_for(ids: list[int]) -> dict[str, str]:
        return {
            case_key_by_id[case_id]: core["case_definition_hashes"][
                case_key_by_id[case_id]
            ]
            for case_id in ids
        }

    outputs: dict[str, dict[str, Any]] = {
        "source_1d_v1.json": {
            "schema_version": 1,
            "phase": "P4",
            "role": "canonical_source_1d_alias",
            "source_manifest": source_manifest_path,
            "source_manifest_sha256": _sha256_file(source_path),
            "dataset_id": source["dataset_id"],
            "array_sha256": source["array_sha256"],
            "splits": source["splits"],
        },
        "phase_a_v1.json": {
            "schema_version": 1,
            "phase": "P4",
            "role": "canonical_phase_a_alias",
            "source_manifest": phase_a_manifest_path,
            "source_manifest_sha256": _sha256_file(phase_a_path),
            "dataset_id": phase_a.get("dataset_id", phase_a.get("version")),
            "splits": phase_a["splits"],
        },
        "2d_id_v1.json": {
            **common,
            "role": "target_id",
            "splits": {
                "train_pool": training,
                "validation": core["splits"]["validation"],
                "test": core["splits"]["id_test"],
            },
            "nested_training_budgets": nested,
            "case_definition_hashes": hashes_for(
                training
                + core["splits"]["validation"]
                + core["splits"]["id_test"]
            ),
        },
        "2d_geometry_ood_v1.json": {
            **common,
            "role": "geometry_ood",
            "status": "deferred_until_F3",
            "case_ids": [],
            "reason": (
                "P4 is pre-registered for F0--F2 only; geometry OOD begins "
                "after stable variable-geometry F3 generation."
            ),
        },
        "external_validation_v1.json": {
            "schema_version": 1,
            "phase": "P4",
            "role": "external_validation",
            "status": "deferred_until_P7_compatibility_audit",
            "case_ids": [],
            "reason": "No external field case is silently assigned before P7.",
        },
    }
    for filename, split_name in (
        ("2d_cycle_ood_v1.json", "cycle_ood"),
        ("2d_htc_ood_v1.json", "htc_ood"),
        ("2d_pattern_ood_v1.json", "pattern_ood"),
        ("2d_combined_ood_v1.json", "combined_ood"),
    ):
        ids = [int(value) for value in core["splits"][split_name]]
        outputs[filename] = {
            **common,
            "role": split_name,
            "case_ids": ids,
            "case_definition_hashes": hashes_for(ids),
        }
    payloads = {
        output_root / name: _pretty_json_bytes(payload)
        for name, payload in outputs.items()
    }
    _prepare_immutable_writes(payloads)
    return {
        "frozen": sorted(outputs),
        "nested_training_budgets": list(nested),
        "dataset_id": core["dataset_id"],
        "plan_sha256": core["plan_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-manifest",
        type=Path,
        default=SPLIT_ROOT / "p4_2d_core_v1.json",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=SPLIT_ROOT / "p3_source_1d_v3.json",
    )
    parser.add_argument(
        "--phase-a-manifest",
        type=Path,
        default=SPLIT_ROOT / "legacy_case1_corrected_matched_v1.json",
    )
    parser.add_argument("--output-root", type=Path, default=SPLIT_ROOT)
    args = parser.parse_args()
    result = freeze_manifests(
        core_path=args.core_manifest,
        source_path=args.source_manifest,
        phase_a_path=args.phase_a_manifest,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
