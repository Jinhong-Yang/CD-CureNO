"""Leakage-safe P6-v2 train/validation shard construction and loading.

The frozen P4 arrays contain training, validation, ID-test, and OOD labels in
one monolithic dataset.  This module creates an immutable physical shard whose
case axis is exactly cases 0--287.  The builder may mmap the P4 arrays, but it
copies only ``[:288]`` from every case-indexed array and never recomputes a
whole-file hash for either monolithic label array.

After construction, the training loader validates and opens only shard arrays
plus JSON pre-label metadata.  It has no code path that opens the monolithic
``temperature_K.npy`` or ``alpha.npy`` files.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import time
from typing import Any, Mapping

import numpy as np

from cdcureno.data.target_2d import (
    BaselineBuilder,
    PreparedTarget2DTraining,
    Target2DTrainingDataset,
    build_label_free_coarse_1d_baseline,
    load_source_normalization_contract,
)


TRAIN_CASE_IDS = tuple(range(256))
VALIDATION_CASE_IDS = tuple(range(256, 288))
INCLUDED_CASE_IDS = TRAIN_CASE_IDS + VALIDATION_CASE_IDS
INCLUDED_CASE_COUNT = len(INCLUDED_CASE_IDS)

CASE_INDEXED_ARRAYS = (
    "temperature_K",
    "alpha",
    "air_temperature_K",
    "top_h_W_m2_K",
)
SHARED_ARRAYS = ("time_s", "x_m", "z_m", "composite_mask")
SHARD_ARRAYS = CASE_INDEXED_ARRAYS + SHARED_ARRAYS
LABEL_ARRAYS = ("temperature_K", "alpha")
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


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
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolve_repo_path(
    project_root: Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    root = project_root.resolve()
    if isinstance(value, Path) and value.is_absolute():
        resolved = value.resolve()
    else:
        text = value.as_posix() if isinstance(value, Path) else value
        if not isinstance(text, str) or not text or "\\" in text:
            raise ValueError(
                f"{label} must be a nonempty repository-relative POSIX path"
            )
        pure = PurePosixPath(text)
        if (
            pure.is_absolute()
            or pure.as_posix() != text
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError(
                f"{label} must be a normalized repository-relative path"
            )
        resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root") from error
    return resolved


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Path is outside the project root: {path}") from error


def _artifact(
    path: Path,
    project_root: Path,
    *,
    declared_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "path": _relative(declared_path or path, project_root),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _validate_artifact_binding(
    record: Any,
    *,
    path: Path,
    project_root: Path,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} artifact binding must be an object")
    expected = _artifact(path, project_root)
    if record != expected:
        raise ValueError(f"{label} artifact binding does not match disk")


def _integer_ids(
    values: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{label} must contain only integer case IDs")
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate case IDs")
    if any(value < 0 for value in result):
        raise ValueError(f"{label} contains a negative case ID")
    return result


@dataclass(frozen=True)
class _P4PrelabelMetadata:
    project_root: Path
    core_path: Path
    id_path: Path
    plan_path: Path
    core_artifact: dict[str, Any]
    id_artifact: dict[str, Any]
    plan_artifact: dict[str, Any]
    core: dict[str, Any]
    id_manifest: dict[str, Any]
    plan: dict[str, Any]
    definitions: dict[int, dict[str, Any]]
    definition_hashes: dict[str, str]
    geometry: dict[str, float]
    source_array_paths: dict[str, Path]
    source_shapes: dict[str, tuple[int, ...]]
    nested_budgets: dict[str, list[int]]


@dataclass(frozen=True)
class TrainingShardContract:
    """Validated shard contract consumed by the existing target dataset."""

    project_root: Path
    artifact_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    definitions: dict[int, dict[str, Any]]
    geometry: dict[str, float]
    array_paths: dict[str, Path]
    checksums: dict[str, Any]
    train_case_ids: tuple[int, ...] = TRAIN_CASE_IDS
    validation_case_ids: tuple[int, ...] = VALIDATION_CASE_IDS


def _source_array_shapes(
    shape: list[int],
) -> dict[str, tuple[int, ...]]:
    case_count, nt, nz, nx = (int(value) for value in shape)
    return {
        "temperature_K": (case_count, nt, nz, nx),
        "alpha": (case_count, nt, nz, nx),
        "air_temperature_K": (case_count, nt),
        "top_h_W_m2_K": (case_count, nx),
        "time_s": (nt,),
        "x_m": (nx,),
        "z_m": (nz,),
        "composite_mask": (nz, nx),
    }


def _shard_shapes(
    source_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    return {
        name: (
            (INCLUDED_CASE_COUNT, *source_shapes[name][1:])
            if name in CASE_INDEXED_ARRAYS
            else source_shapes[name]
        )
        for name in SHARD_ARRAYS
    }


def _load_p4_prelabel_metadata(
    project_root: Path,
    *,
    core_manifest_path: Path,
    id_manifest_path: Path,
) -> _P4PrelabelMetadata:
    root = project_root.resolve()
    core_path = _resolve_repo_path(
        root, core_manifest_path, label="P4 core manifest"
    )
    id_path = _resolve_repo_path(
        root, id_manifest_path, label="P4 target-ID manifest"
    )
    core = _read_json(core_path, label="P4 core manifest")
    id_manifest = _read_json(id_path, label="P4 target-ID manifest")
    core_artifact = _artifact(core_path, root)
    id_artifact = _artifact(id_path, root)

    if core.get("schema_version") != 2 or core.get("phase") != "P4":
        raise ValueError("P4 core manifest must use schema_version=2")
    if (
        id_manifest.get("schema_version") != 2
        or id_manifest.get("role") != "target_id"
    ):
        raise ValueError("P4 target-ID manifest must have role=target_id")
    if id_manifest.get("source_manifest") != core_artifact["path"]:
        raise ValueError("Target-ID manifest points to a different core manifest")
    if id_manifest.get("source_manifest_sha256") != core_artifact["sha256"]:
        raise ValueError("Target-ID/core manifest SHA-256 binding differs")
    if id_manifest.get("dataset_id") != core.get("dataset_id"):
        raise ValueError("P4 core and target-ID dataset IDs differ")

    declared_hashes = core.get("array_sha256")
    if not isinstance(declared_hashes, dict) or set(declared_hashes) != set(
        SHARD_ARRAYS
    ):
        raise ValueError("P4 declared array hash inventory is incomplete")
    for name, value in declared_hashes.items():
        _validated_sha256(value, label=f"P4 array_sha256.{name}")
    if id_manifest.get("dataset_array_sha256") != declared_hashes:
        raise ValueError("Target-ID manifest has different declared array hashes")
    if id_manifest.get("dataset_metadata_sha256") != core.get(
        "metadata_sha256"
    ):
        raise ValueError("Target-ID manifest has different metadata binding")

    plan_path = _resolve_repo_path(
        root, core.get("plan_path"), label="P4 pre-label plan"
    )
    plan = _read_json(plan_path, label="P4 pre-label plan")
    plan_artifact = _artifact(plan_path, root)
    plan_sha = _validated_sha256(
        plan.get("plan_sha256"), label="P4 semantic plan SHA-256"
    )
    plan_without_sha = {
        key: value for key, value in plan.items() if key != "plan_sha256"
    }
    if _sha256_bytes(_canonical_json_bytes(plan_without_sha)) != plan_sha:
        raise ValueError("P4 pre-label semantic plan SHA-256 is invalid")
    if (
        core.get("plan_sha256") != plan_sha
        or id_manifest.get("dataset_plan_sha256") != plan_sha
    ):
        raise ValueError("P4 manifests do not bind the semantic pre-label plan")
    if (
        plan.get("plan_role") != "pre_label_case_plan"
        or plan.get("dataset_id") != core.get("dataset_id")
    ):
        raise ValueError("P4 plan is not the canonical pre-label case plan")

    source_splits = core.get("splits")
    plan_splits = plan.get("splits")
    id_splits = id_manifest.get("splits")
    if not all(
        isinstance(value, dict)
        for value in (source_splits, plan_splits, id_splits)
    ):
        raise ValueError("P4 manifests and plan need split mappings")
    if source_splits != plan_splits:
        raise ValueError("P4 core splits differ from the pre-label plan")
    train = _integer_ids(source_splits.get("train"), label="P4 train")
    validation = _integer_ids(
        source_splits.get("validation"), label="P4 validation"
    )
    if train != list(TRAIN_CASE_IDS):
        raise PermissionError("P6-v2 shard requires exact train cases 0--255")
    if validation != list(VALIDATION_CASE_IDS):
        raise PermissionError(
            "P6-v2 shard requires exact validation cases 256--287"
        )
    if id_splits.get("train_pool") != train:
        raise ValueError("Target-ID train_pool differs from the P4 train split")
    if id_splits.get("validation") != validation:
        raise ValueError(
            "Target-ID validation differs from the P4 validation split"
        )
    if id_splits.get("test") != source_splits.get("id_test"):
        raise ValueError("Target-ID test differs from the P4 ID-test split")
    for split_name, raw_ids in source_splits.items():
        ids = _integer_ids(
            raw_ids,
            label=f"P4 {split_name}",
            allow_empty=split_name not in {"train", "validation"},
        )
        if split_name not in {"train", "validation"} and any(
            case_id < INCLUDED_CASE_COUNT for case_id in ids
        ):
            raise PermissionError(
                f"P4 held-out split {split_name} overlaps the training shard"
            )

    raw_budgets = core.get("nested_training_budgets")
    if (
        not isinstance(raw_budgets, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in raw_budgets
        )
        or raw_budgets != sorted(set(raw_budgets))
        or raw_budgets[-1] != len(TRAIN_CASE_IDS)
    ):
        raise ValueError("P4 nested training budgets are invalid")
    id_budgets = id_manifest.get("nested_training_budgets")
    if not isinstance(id_budgets, dict):
        raise ValueError("Target-ID nested training budgets are missing")
    nested_budgets: dict[str, list[int]] = {}
    for budget in raw_budgets:
        key = str(budget)
        expected = list(TRAIN_CASE_IDS[:budget])
        if id_budgets.get(key) != expected:
            raise ValueError(f"Target-ID budget {budget} is not an exact prefix")
        nested_budgets[key] = expected
    if set(id_budgets) != set(nested_budgets):
        raise ValueError("Target-ID budget inventory differs from the P4 core")

    shape = core.get("array_shape_case_time_z_x")
    case_count = core.get("case_count")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in shape
        )
        or isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or shape[0] != case_count
        or case_count < INCLUDED_CASE_COUNT
    ):
        raise ValueError("P4 core array shape/case count is invalid")

    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != case_count:
        raise ValueError("P4 plan case inventory differs from the core")
    core_definition_hashes = core.get("case_definition_hashes")
    id_definition_hashes = id_manifest.get("case_definition_hashes")
    if not isinstance(core_definition_hashes, dict) or not isinstance(
        id_definition_hashes, dict
    ):
        raise ValueError("P4 case-definition hash mappings are missing")
    definitions: dict[int, dict[str, Any]] = {}
    definition_hashes: dict[str, str] = {}
    for position, entry in enumerate(cases):
        if not isinstance(entry, dict) or not isinstance(
            entry.get("definition"), dict
        ):
            raise ValueError("Every P4 plan case needs a definition")
        definition = entry["definition"]
        if definition.get("case_id") != position:
            raise ValueError("P4 plan case IDs must equal array row indices")
        if position >= INCLUDED_CASE_COUNT:
            continue
        case_key = definition.get("case_key")
        if not isinstance(case_key, str) or not case_key:
            raise ValueError("P4 plan definition needs a case_key")
        digest = _sha256_bytes(_canonical_json_bytes(definition))
        if entry.get("case_definition_sha256", digest) != digest:
            raise ValueError(f"P4 plan case hash differs for case {position}")
        if core_definition_hashes.get(case_key) != digest:
            raise ValueError(f"P4 core case hash differs for case {position}")
        if id_definition_hashes.get(case_key) != digest:
            raise ValueError(
                f"P4 target-ID case hash differs for case {position}"
            )
        definitions[position] = definition
        definition_hashes[case_key] = digest
    if tuple(definitions) != INCLUDED_CASE_IDS:
        raise ValueError("P4 pre-label definitions do not cover cases 0--287")

    resolved = plan.get("resolved_config")
    geometry_payload = (
        resolved.get("geometry") if isinstance(resolved, dict) else None
    )
    if not isinstance(geometry_payload, dict):
        raise ValueError("P4 pre-label plan has no resolved geometry")
    geometry = {
        name: float(geometry_payload[name])
        for name in (
            "width_m",
            "tool_thickness_m",
            "composite_thickness_m",
        )
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in geometry.values()):
        raise ValueError("P4 geometry must be finite and positive")

    source_root = _resolve_repo_path(
        root, core.get("array_artifact_root"), label="P4 array root"
    )
    source_array_paths = {
        name: source_root / f"{name}.npy" for name in SHARD_ARRAYS
    }
    source_shapes = _source_array_shapes(shape)
    return _P4PrelabelMetadata(
        project_root=root,
        core_path=core_path,
        id_path=id_path,
        plan_path=plan_path,
        core_artifact=core_artifact,
        id_artifact=id_artifact,
        plan_artifact=plan_artifact,
        core=core,
        id_manifest=id_manifest,
        plan=plan,
        definitions=definitions,
        definition_hashes=definition_hashes,
        geometry=geometry,
        source_array_paths=source_array_paths,
        source_shapes=source_shapes,
        nested_budgets=nested_budgets,
    )


def _write_npy_from_view(
    destination: Path,
    source_view: Any,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> None:
    if tuple(source_view.shape) != expected_shape:
        raise ValueError(
            f"Source view for {destination.name} has shape "
            f"{source_view.shape}, expected {expected_shape}"
        )
    if np.dtype(source_view.dtype) != expected_dtype:
        raise ValueError(
            f"Source view for {destination.name} has dtype "
            f"{source_view.dtype}, expected {expected_dtype}"
        )
    target = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=expected_dtype,
        shape=expected_shape,
    )
    try:
        target[...] = source_view
        target.flush()
    finally:
        del target


def _array_record(
    path: Path,
    *,
    final_path: Path,
    project_root: Path,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    return {
        **_artifact(path, project_root, declared_path=final_path),
        "shape": list(shape),
        "dtype": dtype.name,
    }


def _source_bindings_are_stable(metadata: _P4PrelabelMetadata) -> bool:
    return (
        _artifact(metadata.core_path, metadata.project_root)
        == metadata.core_artifact
        and _artifact(metadata.id_path, metadata.project_root)
        == metadata.id_artifact
        and _artifact(metadata.plan_path, metadata.project_root)
        == metadata.plan_artifact
    )


def _manifest_payload(
    metadata: _P4PrelabelMetadata,
    *,
    artifact_root: Path,
    arrays: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "P6V2",
        "role": "train_validation_label_shard",
        "shard_id": "p4_2d_train_validation_v1",
        "source_dataset_id": metadata.core["dataset_id"],
        "array_artifact_root": _relative(
            artifact_root, metadata.project_root
        ),
        "case_axis_policy": {
            "row_equals_case_id": True,
            "included_case_count": INCLUDED_CASE_COUNT,
            "included_case_ids": list(INCLUDED_CASE_IDS),
            "train_case_ids": list(TRAIN_CASE_IDS),
            "validation_case_ids": list(VALIDATION_CASE_IDS),
            "first_excluded_case_id": INCLUDED_CASE_COUNT,
            "id_test_or_ood_label_values_copied": False,
        },
        "nested_training_budgets": metadata.nested_budgets,
        "case_definition_hashes": metadata.definition_hashes,
        "source_binding": {
            "p4_core_manifest": metadata.core_artifact,
            "p4_target_id_manifest": metadata.id_artifact,
            "p4_pre_label_plan": {
                **metadata.plan_artifact,
                "semantic_sha256": metadata.plan["plan_sha256"],
            },
            "declared_p4_array_sha256": metadata.core["array_sha256"],
            "declared_p4_metadata_sha256": metadata.core["metadata_sha256"],
            "monolithic_label_file_sha256_recomputed": False,
        },
        "construction_contract": {
            "source_arrays_opened_with_mmap_mode": "r",
            "case_indexed_source_slice": "[:288]",
            "shared_arrays_have_no_case_axis": True,
            "source_binding_hashes_stable_before_and_after_copy": True,
            "atomic_staging_directory_promoted_without_overwrite": True,
        },
        "arrays": arrays,
        "shard_file_count": len(arrays),
        "all_shard_files_sha256_bound": True,
    }


def _safe_cleanup_staging(staging: Path, intended_parent: Path) -> None:
    if not staging.exists():
        return
    resolved = staging.resolve()
    if resolved.parent != intended_parent.resolve() or not resolved.name.startswith(
        ".p4_2d_train_validation_v1.staging-"
    ):
        raise RuntimeError(f"Refusing to clean unexpected staging path: {resolved}")
    shutil.rmtree(resolved)


def build_training_shard(
    *,
    project_root: Path,
    core_manifest_path: Path = Path("splits/p4_2d_core_v1.json"),
    id_manifest_path: Path = Path("splits/2d_id_v1.json"),
    output_manifest_path: Path = Path(
        "splits/p4_2d_train_validation_shard_v1.json"
    ),
    output_root: Path = Path(
        "data/processed/p4_2d_train_validation_v1"
    ),
    verify_existing: bool = False,
) -> dict[str, Any]:
    """Create the immutable shard, or validate an existing shard read-only."""

    root = project_root.resolve()
    manifest_path = _resolve_repo_path(
        root, output_manifest_path, label="training shard manifest"
    )
    artifact_root = _resolve_repo_path(
        root, output_root, label="training shard artifact root"
    )
    if verify_existing:
        contract = validate_training_shard(
            manifest_path,
            project_root=root,
            verify_shard_checksums=True,
        )
        return {
            "status": "verified_existing",
            "manifest": contract.checksums["training_shard_manifest"],
            "case_count": INCLUDED_CASE_COUNT,
            "arrays": contract.manifest["arrays"],
        }
    if manifest_path.exists() or artifact_root.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing training shard or manifest"
        )

    metadata = _load_p4_prelabel_metadata(
        root,
        core_manifest_path=core_manifest_path,
        id_manifest_path=id_manifest_path,
    )
    expected_shapes = _shard_shapes(metadata.source_shapes)
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging = artifact_root.parent / (
        f".p4_2d_train_validation_v1.staging-{os.getpid()}-"
        f"{time.time_ns()}"
    )
    staging.mkdir(exist_ok=False)
    manifest_temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    promoted = False
    try:
        for name in SHARD_ARRAYS:
            source_path = metadata.source_array_paths[name]
            try:
                source = np.load(
                    source_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Cannot mmap P4 source array {source_path}: {error}"
                ) from error
            if tuple(source.shape) != metadata.source_shapes[name]:
                raise ValueError(f"P4 source array shape differs: {name}")
            if np.dtype(source.dtype) != EXPECTED_DTYPES[name]:
                raise ValueError(f"P4 source array dtype differs: {name}")
            # This is the only case-indexed source access in the builder.
            # In particular, monolithic labels are never indexed at 288+.
            source_view = (
                source[:INCLUDED_CASE_COUNT]
                if name in CASE_INDEXED_ARRAYS
                else source[...]
            )
            _write_npy_from_view(
                staging / f"{name}.npy",
                source_view,
                expected_shape=expected_shapes[name],
                expected_dtype=EXPECTED_DTYPES[name],
            )
            del source_view, source

        arrays = {
            name: _array_record(
                staging / f"{name}.npy",
                final_path=artifact_root / f"{name}.npy",
                project_root=root,
                shape=expected_shapes[name],
                dtype=EXPECTED_DTYPES[name],
            )
            for name in SHARD_ARRAYS
        }
        if not _source_bindings_are_stable(metadata):
            raise RuntimeError(
                "P4 manifest or pre-label plan changed during shard creation"
            )
        payload = _manifest_payload(
            metadata,
            artifact_root=artifact_root,
            arrays=arrays,
        )
        manifest_bytes = _pretty_json_bytes(payload)
        with manifest_temporary.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if manifest_path.exists() or artifact_root.exists():
            raise FileExistsError(
                "Training shard destination appeared during construction"
            )
        staging.rename(artifact_root)
        promoted = True
        manifest_temporary.rename(manifest_path)
        return payload
    finally:
        if not promoted:
            _safe_cleanup_staging(staging, artifact_root.parent)
        if manifest_temporary.exists():
            manifest_temporary.unlink()


def _validate_case_policy(manifest: Mapping[str, Any]) -> None:
    policy = manifest.get("case_axis_policy")
    expected = {
        "row_equals_case_id": True,
        "included_case_count": INCLUDED_CASE_COUNT,
        "included_case_ids": list(INCLUDED_CASE_IDS),
        "train_case_ids": list(TRAIN_CASE_IDS),
        "validation_case_ids": list(VALIDATION_CASE_IDS),
        "first_excluded_case_id": INCLUDED_CASE_COUNT,
        "id_test_or_ood_label_values_copied": False,
    }
    if policy != expected:
        raise PermissionError(
            "Training shard case policy must be exactly cases 0--287"
        )


def _validate_construction_contract(manifest: Mapping[str, Any]) -> None:
    expected = {
        "source_arrays_opened_with_mmap_mode": "r",
        "case_indexed_source_slice": "[:288]",
        "shared_arrays_have_no_case_axis": True,
        "source_binding_hashes_stable_before_and_after_copy": True,
        "atomic_staging_directory_promoted_without_overwrite": True,
    }
    if manifest.get("construction_contract") != expected:
        raise PermissionError(
            "Training shard construction contract is missing or changed"
        )


def _validate_shard_array(
    *,
    name: str,
    record: Any,
    artifact_root: Path,
    project_root: Path,
    expected_shape: tuple[int, ...],
    verify_checksum: bool,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Shard array record is invalid: {name}")
    path = _resolve_repo_path(
        project_root, record.get("path"), label=f"shard array {name}"
    )
    expected_path = (artifact_root / f"{name}.npy").resolve()
    if path != expected_path:
        raise ValueError(f"Shard array {name} is outside its artifact root")
    if not path.is_file():
        raise FileNotFoundError(f"Missing shard array: {path}")
    if record.get("bytes") != path.stat().st_size:
        raise ValueError(f"Shard array byte count differs: {name}")
    _validated_sha256(
        record.get("sha256"), label=f"shard array {name} SHA-256"
    )
    if verify_checksum and _sha256_file(path) != record["sha256"]:
        raise ValueError(f"Shard array SHA-256 differs: {name}")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot mmap shard array {name}: {error}") from error
    if tuple(array.shape) != expected_shape:
        raise ValueError(f"Shard array shape differs: {name}")
    if np.dtype(array.dtype) != EXPECTED_DTYPES[name]:
        raise ValueError(f"Shard array dtype differs: {name}")
    if record.get("shape") != list(expected_shape):
        raise ValueError(f"Shard manifest shape differs: {name}")
    if record.get("dtype") != EXPECTED_DTYPES[name].name:
        raise ValueError(f"Shard manifest dtype differs: {name}")
    del array
    return path


def validate_training_shard(
    manifest_path: Path,
    *,
    project_root: Path | None = None,
    expected_manifest_sha256: str | None = None,
    verify_shard_checksums: bool = True,
) -> TrainingShardContract:
    """Validate only shard arrays and JSON metadata, never source labels."""

    path = manifest_path.resolve()
    root = (
        project_root.resolve()
        if project_root is not None
        else path.parent.parent.resolve()
    )
    path = _resolve_repo_path(root, path, label="training shard manifest")
    if expected_manifest_sha256 is not None:
        expected = _validated_sha256(
            expected_manifest_sha256,
            label="expected training shard manifest SHA-256",
        )
        if _sha256_file(path) != expected:
            raise ValueError("Training shard manifest SHA-256 differs")
    manifest = _read_json(path, label="training shard manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("phase") != "P6V2"
        or manifest.get("role") != "train_validation_label_shard"
        or manifest.get("shard_id") != "p4_2d_train_validation_v1"
    ):
        raise ValueError("Training shard identity or role is invalid")
    _validate_case_policy(manifest)
    _validate_construction_contract(manifest)

    binding = manifest.get("source_binding")
    if not isinstance(binding, dict):
        raise ValueError("Training shard source_binding is missing")
    core_record = binding.get("p4_core_manifest")
    id_record = binding.get("p4_target_id_manifest")
    plan_record = binding.get("p4_pre_label_plan")
    if not all(
        isinstance(record, dict)
        for record in (core_record, id_record, plan_record)
    ):
        raise ValueError("Training shard metadata bindings are incomplete")
    core_path = _resolve_repo_path(
        root, core_record.get("path"), label="bound P4 core manifest"
    )
    id_path = _resolve_repo_path(
        root, id_record.get("path"), label="bound P4 target-ID manifest"
    )
    plan_path = _resolve_repo_path(
        root, plan_record.get("path"), label="bound P4 pre-label plan"
    )
    _validate_artifact_binding(
        core_record,
        path=core_path,
        project_root=root,
        label="P4 core manifest",
    )
    _validate_artifact_binding(
        id_record,
        path=id_path,
        project_root=root,
        label="P4 target-ID manifest",
    )
    plan_file_record = {
        key: plan_record.get(key) for key in ("path", "sha256", "bytes")
    }
    _validate_artifact_binding(
        plan_file_record,
        path=plan_path,
        project_root=root,
        label="P4 pre-label plan",
    )
    metadata = _load_p4_prelabel_metadata(
        root,
        core_manifest_path=core_path,
        id_manifest_path=id_path,
    )
    if plan_path != metadata.plan_path:
        raise ValueError("Shard binds a different P4 pre-label plan")
    if plan_record.get("semantic_sha256") != metadata.plan["plan_sha256"]:
        raise ValueError("Shard semantic plan SHA-256 differs")
    if binding.get("declared_p4_array_sha256") != metadata.core.get(
        "array_sha256"
    ):
        raise ValueError("Shard declared P4 array hash binding differs")
    if binding.get("declared_p4_metadata_sha256") != metadata.core.get(
        "metadata_sha256"
    ):
        raise ValueError("Shard declared P4 metadata hash binding differs")
    if binding.get("monolithic_label_file_sha256_recomputed") is not False:
        raise PermissionError(
            "Shard must state that monolithic label hashes were not recomputed"
        )
    if manifest.get("nested_training_budgets") != metadata.nested_budgets:
        raise ValueError("Shard nested budgets differ from P4 pre-label metadata")
    if manifest.get("case_definition_hashes") != metadata.definition_hashes:
        raise ValueError("Shard case-definition hashes differ from P4 metadata")

    artifact_root = _resolve_repo_path(
        root,
        manifest.get("array_artifact_root"),
        label="training shard artifact root",
    )
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(SHARD_ARRAYS):
        raise ValueError("Training shard array inventory is incomplete")
    actual_files = {
        item.name for item in artifact_root.iterdir() if item.is_file()
    }
    expected_files = {f"{name}.npy" for name in SHARD_ARRAYS}
    if actual_files != expected_files:
        raise ValueError("Training shard contains missing or unhashed files")
    if (
        manifest.get("shard_file_count") != len(expected_files)
        or manifest.get("all_shard_files_sha256_bound") is not True
    ):
        raise ValueError("Training shard file-count/hash declaration differs")

    expected_shapes = _shard_shapes(metadata.source_shapes)
    array_paths = {
        name: _validate_shard_array(
            name=name,
            record=arrays[name],
            artifact_root=artifact_root,
            project_root=root,
            expected_shape=expected_shapes[name],
            verify_checksum=verify_shard_checksums,
        )
        for name in SHARD_ARRAYS
    }

    planned_air = np.asarray(
        [
            metadata.definitions[case_id]["air_temperature_K"]
            for case_id in INCLUDED_CASE_IDS
        ],
        dtype=np.float32,
    )
    planned_top = np.asarray(
        [
            metadata.definitions[case_id]["top_h_W_m2_K"]
            for case_id in INCLUDED_CASE_IDS
        ],
        dtype=np.float32,
    )
    shard_air = np.load(
        array_paths["air_temperature_K"],
        mmap_mode="r",
        allow_pickle=False,
    )
    shard_top = np.load(
        array_paths["top_h_W_m2_K"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.array_equal(shard_air, planned_air):
        raise ValueError("Shard air histories differ from the pre-label plan")
    if not np.array_equal(shard_top, planned_top):
        raise ValueError("Shard top-HTC fields differ from the pre-label plan")
    del shard_air, shard_top

    shard_time = np.load(
        array_paths["time_s"], mmap_mode="r", allow_pickle=False
    )
    if not np.array_equal(
        shard_time,
        np.asarray(metadata.plan.get("time_s"), dtype=np.float64),
    ):
        raise ValueError("Shard time grid differs from the pre-label plan")
    shard_x = np.load(array_paths["x_m"], mmap_mode="r", allow_pickle=False)
    shard_z = np.load(array_paths["z_m"], mmap_mode="r", allow_pickle=False)
    shard_mask = np.load(
        array_paths["composite_mask"],
        mmap_mode="r",
        allow_pickle=False,
    )
    total_z = (
        metadata.geometry["tool_thickness_m"]
        + metadata.geometry["composite_thickness_m"]
    )
    expected_mask = np.broadcast_to(
        (
            np.asarray(shard_z)
            > metadata.geometry["tool_thickness_m"]
        )[:, None],
        shard_mask.shape,
    )
    if (
        shard_x.min() < 0.0
        or shard_x.max() > metadata.geometry["width_m"]
        or shard_z.min() < 0.0
        or shard_z.max() > total_z
        or not np.array_equal(shard_mask, expected_mask)
    ):
        raise ValueError("Shard coordinates or material mask are invalid")
    del shard_time, shard_x, shard_z, shard_mask

    manifest_artifact = _artifact(path, root)
    checksums = {
        "training_shard_manifest": manifest_artifact,
        "training_shard_array_sha256": {
            name: arrays[name]["sha256"] for name in SHARD_ARRAYS
        },
        "p4_core_manifest_sha256": metadata.core_artifact["sha256"],
        "p4_target_id_manifest_sha256": metadata.id_artifact["sha256"],
        "p4_pre_label_plan_file_sha256": metadata.plan_artifact["sha256"],
        "p4_pre_label_plan_semantic_sha256": metadata.plan["plan_sha256"],
        "declared_p4_array_sha256": dict(metadata.core["array_sha256"]),
        "monolithic_label_files_opened_by_loader": False,
        "array_files_verified": bool(verify_shard_checksums),
    }
    return TrainingShardContract(
        project_root=root,
        artifact_root=artifact_root,
        manifest_path=path,
        manifest=manifest,
        definitions=metadata.definitions,
        geometry=metadata.geometry,
        array_paths=array_paths,
        checksums=checksums,
    )


def prepare_target_2d_training_shard(
    shard_manifest: Path,
    source_checkpoint: Path,
    *,
    label_budget: int,
    project_root: Path | None = None,
    expected_manifest_sha256: str | None = None,
    verify_shard_checksums: bool = True,
    baseline_builder: BaselineBuilder = build_label_free_coarse_1d_baseline,
) -> PreparedTarget2DTraining:
    """Prepare train/validation datasets backed only by the physical shard."""

    contract = validate_training_shard(
        shard_manifest,
        project_root=project_root,
        expected_manifest_sha256=expected_manifest_sha256,
        verify_shard_checksums=verify_shard_checksums,
    )
    if isinstance(label_budget, bool) or not isinstance(label_budget, int):
        raise ValueError("label_budget must be an integer")
    budget_key = str(label_budget)
    budgets = contract.manifest["nested_training_budgets"]
    if budget_key not in budgets:
        raise ValueError(
            f"label_budget={label_budget} is not one of "
            f"{sorted(int(value) for value in budgets)}"
        )
    train_case_ids = [int(value) for value in budgets[budget_key]]
    validation_case_ids = list(VALIDATION_CASE_IDS)
    source_normalization = load_source_normalization_contract(
        source_checkpoint
    )
    checksums = dict(contract.checksums)
    checksums.update(
        {
            "source_checkpoint_sha256": (
                source_normalization.checkpoint_sha256
            ),
            "source_normalization_sha256": (
                source_normalization.normalization_sha256
            ),
        }
    )
    train_dataset = Target2DTrainingDataset(
        split_name="train",
        case_ids=train_case_ids,
        contract=contract,
        normalization=source_normalization,
        baseline_builder=baseline_builder,
    )
    validation_dataset = Target2DTrainingDataset(
        split_name="validation",
        case_ids=validation_case_ids,
        contract=contract,
        normalization=source_normalization,
        baseline_builder=baseline_builder,
    )
    return PreparedTarget2DTraining(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        label_budget=label_budget,
        train_case_ids=tuple(train_case_ids),
        validation_case_ids=tuple(validation_case_ids),
        normalization=source_normalization.metadata,
        checksums=checksums,
    )
