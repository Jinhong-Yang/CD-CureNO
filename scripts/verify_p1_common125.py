"""Freeze the corrected P1 seed-1 comparison on common case IDs only.

The historical P1 exact and corrected seed-1 field summaries describe
different test populations (150 and 125 cases, respectively).  This verifier
loads only the saved field reconstruction artifacts, explicitly intersects
their case IDs, asserts identical ordering before every paired comparison, and
emits a versioned correction without modifying any v1 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "field_relative_l2",
    "composite_relative_l2",
    "tool_relative_l2",
)
EXPECTED_COMMON_CASE_COUNT = 125


@dataclass(frozen=True)
class SavedField:
    """Validated per-case metrics and matching saved field labels."""

    metrics: pd.DataFrame
    case_ids: np.ndarray
    input_air: np.ndarray
    target: np.ndarray


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload, raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(project_root: Path, path: Path) -> Path:
    resolved_root = project_root.resolve()
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (resolved_root / path).resolve()
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Artifact path escapes project root: {resolved}"
        ) from error
    return resolved


def _artifact(
    path: Path,
    project_root: Path,
    *,
    raw: bytes | None = None,
) -> dict[str, Any]:
    resolved = _project_path(project_root, path)
    size = len(raw) if raw is not None else resolved.stat().st_size
    checksum = _sha256_bytes(raw) if raw is not None else _sha256_file(resolved)
    return {
        "path": resolved.relative_to(project_root.resolve()).as_posix(),
        "sha256": checksum,
        "bytes": int(size),
    }


def _declared_artifact(
    project_root: Path,
    summary: Mapping[str, Any],
    key: str,
) -> Path:
    value = summary.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Summary field {key!r} must be a non-empty path")
    path = _project_path(project_root, Path(value))
    if not path.is_file():
        raise FileNotFoundError(f"Missing declared {key}: {path}")
    return path


def _validated_case_ids(values: pd.Series | np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} case IDs must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} case IDs must be numeric")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} case IDs must be finite integers")
    case_ids = numeric.astype(np.int64)
    if len(np.unique(case_ids)) != len(case_ids):
        raise ValueError(f"{label} contains duplicate case IDs")
    return case_ids


def _read_metrics(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = ("case_id", *METRIC_COLUMNS)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} per-case metrics missing columns: {missing}")
    frame = frame.loc[:, required].copy()
    frame["case_id"] = _validated_case_ids(frame["case_id"], label=label)
    for metric in METRIC_COLUMNS:
        values = frame[metric].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{label} metric {metric} must be numeric")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} metric {metric} contains nonfinite values")
        if np.any(values < 0):
            raise ValueError(f"{label} metric {metric} contains negative values")
    return frame


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = (prediction - target).reshape(len(target), -1)
    reference = target.reshape(len(target), -1)
    return np.linalg.norm(difference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-12
    )


def _read_saved_field(
    field_path: Path,
    metrics: pd.DataFrame,
    *,
    label: str,
) -> SavedField:
    with np.load(field_path, allow_pickle=False) as archive:
        required = {"case_ids", "input_air", "prediction", "target"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{label} field archive missing arrays: {missing}")
        case_ids = _validated_case_ids(archive["case_ids"], label=label)
        input_air = np.asarray(archive["input_air"])
        prediction = np.asarray(archive["prediction"])
        target = np.asarray(archive["target"])

    metric_ids = metrics["case_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(case_ids, metric_ids):
        raise ValueError(
            f"{label} field archive and per-case metrics have different case IDs"
        )
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(f"{label} prediction/target field shapes do not match")
    if prediction.shape[0] != len(case_ids) or prediction.shape[1] != 51:
        raise ValueError(f"{label} has an unexpected reconstructed field shape")
    if input_air.shape[0] != len(case_ids):
        raise ValueError(f"{label} input-air case count does not match")
    if not (
        np.isfinite(input_air).all()
        and np.isfinite(prediction).all()
        and np.isfinite(target).all()
    ):
        raise ValueError(f"{label} field archive contains nonfinite values")

    recomputed = {
        "field_relative_l2": _relative_l2(prediction, target),
        "tool_relative_l2": _relative_l2(
            prediction[:, :21], target[:, :21]
        ),
        "composite_relative_l2": _relative_l2(
            prediction[:, 21:], target[:, 21:]
        ),
    }
    for metric, values in recomputed.items():
        if not np.allclose(
            values,
            metrics[metric].to_numpy(),
            rtol=1e-7,
            atol=1e-12,
        ):
            raise ValueError(
                f"{label} saved {metric} does not match the field archive"
            )
    return SavedField(
        metrics=metrics,
        case_ids=case_ids,
        input_air=input_air,
        target=target,
    )


def _validate_summary(
    summary: Mapping[str, Any],
    field: SavedField,
    *,
    expected_experiment: str,
) -> None:
    if summary.get("experiment") != expected_experiment:
        raise ValueError(
            f"Expected {expected_experiment}, found {summary.get('experiment')}"
        )
    if summary.get("seed") != 1:
        raise ValueError("The common-case correction is restricted to seed 1")
    if summary.get("case_count") != len(field.case_ids):
        raise ValueError("Summary case_count does not match per-case artifacts")
    for metric in METRIC_COLUMNS:
        summary_key = f"{metric}_mean"
        expected = float(np.mean(field.metrics[metric].to_numpy()))
        if summary.get(summary_key) != expected:
            raise ValueError(
                f"Summary {summary_key} does not match per-case artifacts"
            )


def assert_identical_case_ids(
    exact: pd.DataFrame,
    corrected: pd.DataFrame,
) -> None:
    """Reject a paired comparison unless both frames are identically aligned."""

    exact_ids = exact["case_id"].to_numpy(dtype=np.int64)
    corrected_ids = corrected["case_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(exact_ids, corrected_ids):
        raise ValueError(
            "Direct paired comparison requires identical ordered case IDs; "
            "intersect and align populations first"
        )


def compare_aligned_metric(
    exact: pd.DataFrame,
    corrected: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    """Compare one metric after enforcing exact case-ID alignment."""

    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported P1 comparison metric: {metric}")
    assert_identical_case_ids(exact, corrected)
    exact_values = exact[metric].to_numpy()
    corrected_values = corrected[metric].to_numpy()
    exact_mean = float(np.mean(exact_values))
    corrected_mean = float(np.mean(corrected_values))
    if exact_mean <= 0:
        raise ValueError(f"Exact mean for {metric} must be positive")
    corrected_better = int(np.count_nonzero(corrected_values < exact_values))
    exact_better = int(np.count_nonzero(exact_values < corrected_values))
    ties = int(np.count_nonzero(exact_values == corrected_values))
    paired_count = int(len(exact_values))
    if corrected_better + exact_better + ties != paired_count:
        raise AssertionError("Paired comparison counts do not sum to population")
    return {
        "lower_is_better": True,
        "paired_case_count": paired_count,
        "exact_mean": exact_mean,
        "corrected_mean": corrected_mean,
        "corrected_minus_exact_mean": corrected_mean - exact_mean,
        "corrected_relative_improvement_percent": (
            (exact_mean - corrected_mean) / exact_mean * 100.0
        ),
        "corrected_better_case_count": corrected_better,
        "exact_better_case_count": exact_better,
        "tie_case_count": ties,
    }


def build_correction(
    project_root: Path,
    *,
    exact_summary_path: Path = Path(
        "outputs/tables/p1_exact_field_seed1_summary.json"
    ),
    corrected_summary_path: Path = Path(
        "outputs/tables/p1_corrected_field_seed1_summary.json"
    ),
    expected_common_case_count: int = EXPECTED_COMMON_CASE_COUNT,
) -> dict[str, Any]:
    """Build the deterministic v2 correction from frozen saved artifacts."""

    root = project_root.resolve()
    exact_summary_file = _project_path(root, exact_summary_path)
    corrected_summary_file = _project_path(root, corrected_summary_path)
    exact_summary, exact_summary_raw = _read_json_bytes(exact_summary_file)
    corrected_summary, corrected_summary_raw = _read_json_bytes(
        corrected_summary_file
    )

    exact_metrics_path = _declared_artifact(
        root, exact_summary, "per_case_metrics"
    )
    corrected_metrics_path = _declared_artifact(
        root, corrected_summary, "per_case_metrics"
    )
    exact_field_path = _declared_artifact(
        root, exact_summary, "field_artifact"
    )
    corrected_field_path = _declared_artifact(
        root, corrected_summary, "field_artifact"
    )
    source_paths = {
        "exact_summary": exact_summary_file,
        "exact_metrics_per_case": exact_metrics_path,
        "exact_temperature_field": exact_field_path,
        "corrected_summary": corrected_summary_file,
        "corrected_metrics_per_case": corrected_metrics_path,
        "corrected_temperature_field": corrected_field_path,
    }
    before = {
        name: _artifact(path, root) for name, path in source_paths.items()
    }
    if before["exact_summary"] != _artifact(
        exact_summary_file, root, raw=exact_summary_raw
    ):
        raise RuntimeError("The exact summary changed while it was read")
    if before["corrected_summary"] != _artifact(
        corrected_summary_file, root, raw=corrected_summary_raw
    ):
        raise RuntimeError("The corrected summary changed while it was read")

    exact_metrics = _read_metrics(exact_metrics_path, label="exact seed 1")
    corrected_metrics = _read_metrics(
        corrected_metrics_path, label="corrected seed 1"
    )
    exact_field = _read_saved_field(
        exact_field_path, exact_metrics, label="exact seed 1"
    )
    corrected_field = _read_saved_field(
        corrected_field_path,
        corrected_metrics,
        label="corrected seed 1",
    )
    _validate_summary(
        exact_summary,
        exact_field,
        expected_experiment="legacy_resfno_exact",
    )
    _validate_summary(
        corrected_summary,
        corrected_field,
        expected_experiment="legacy_resfno_corrected",
    )

    exact_ids = set(int(value) for value in exact_field.case_ids)
    corrected_ids = set(int(value) for value in corrected_field.case_ids)
    common_ids = sorted(exact_ids & corrected_ids)
    exact_only_ids = sorted(exact_ids - corrected_ids)
    corrected_only_ids = sorted(corrected_ids - exact_ids)
    if len(common_ids) != expected_common_case_count:
        raise ValueError(
            "Expected "
            f"{expected_common_case_count} common cases, found {len(common_ids)}"
        )

    exact_indexed = exact_metrics.set_index("case_id", drop=False)
    corrected_indexed = corrected_metrics.set_index("case_id", drop=False)
    exact_common = exact_indexed.loc[common_ids].reset_index(drop=True)
    corrected_common = corrected_indexed.loc[common_ids].reset_index(drop=True)
    assert_identical_case_ids(exact_common, corrected_common)

    exact_positions = {
        int(case_id): index
        for index, case_id in enumerate(exact_field.case_ids)
    }
    corrected_positions = {
        int(case_id): index
        for index, case_id in enumerate(corrected_field.case_ids)
    }
    exact_selection = [exact_positions[case_id] for case_id in common_ids]
    corrected_selection = [
        corrected_positions[case_id] for case_id in common_ids
    ]
    if not np.array_equal(
        exact_field.input_air[exact_selection],
        corrected_field.input_air[corrected_selection],
    ):
        raise ValueError("Common cases have different saved input-air histories")
    if not np.array_equal(
        exact_field.target[exact_selection],
        corrected_field.target[corrected_selection],
    ):
        raise ValueError("Common cases have different saved target fields")

    after = {
        name: _artifact(path, root) for name, path in source_paths.items()
    }
    if before != after:
        raise RuntimeError("A P1 source artifact changed during verification")

    source_artifacts = {
        "exact_seed1": {
            "summary": after["exact_summary"],
            "metrics_per_case": after["exact_metrics_per_case"],
            "temperature_field": after["exact_temperature_field"],
        },
        "corrected_seed1": {
            "summary": after["corrected_summary"],
            "metrics_per_case": after["corrected_metrics_per_case"],
            "temperature_field": after["corrected_temperature_field"],
        },
    }
    metrics = {
        metric: compare_aligned_metric(
            exact_common, corrected_common, metric
        )
        for metric in METRIC_COLUMNS
    }
    return {
        "schema_version": 2,
        "phase": "P1",
        "artifact_role": "seed1_common_case_descriptive_correction",
        "status": "passed",
        "generator": "scripts/verify_p1_common125.py",
        "seed": 1,
        "population_audit": {
            "exact_case_count": int(len(exact_ids)),
            "corrected_case_count": int(len(corrected_ids)),
            "expected_common_case_count": int(expected_common_case_count),
            "common_case_count": int(len(common_ids)),
            "common_case_ids": common_ids,
            "exact_only_case_count": int(len(exact_only_ids)),
            "exact_only_case_ids": exact_only_ids,
            "corrected_only_case_count": int(len(corrected_only_ids)),
            "corrected_only_case_ids": corrected_only_ids,
            "whole_population_case_ids_identical": exact_ids == corrected_ids,
            "whole_population_direct_comparison_performed": False,
            "common_input_air_histories_identical": True,
            "common_target_fields_identical": True,
            "paired_alignment_asserted_before_each_metric": True,
        },
        "metrics": metrics,
        "source_artifacts": source_artifacts,
        "source_hash_stability_checked_before_and_after_read": True,
        "claim_correction": {
            "superseded_claim": (
                "The v1 same-seed 2.04% field improvement compared an exact "
                "150-case mean with a corrected 125-case mean."
            ),
            "corrected_conclusion": (
                "On the 125 common seed-1 cases, corrected is not more "
                "accurate on mean field, composite, or tool relative L2."
            ),
            "scope": (
                "Descriptive seed-1 correction only; v1 summaries and run "
                "artifacts remain preserved."
            ),
        },
    }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def persist_correction(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    verify_existing: bool = False,
) -> None:
    """Create a correction exclusively, or verify an existing frozen copy."""

    encoded = _canonical_json(payload)
    if verify_existing:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Frozen correction does not exist: {output_path}"
            )
        if output_path.read_bytes() != encoded:
            raise ValueError(
                "Existing correction differs from recomputed canonical content"
            )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite frozen correction: {output_path}; "
            "use --verify-existing to verify it"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the P1 exact/corrected seed-1 comparison on their common "
            "125 saved cases."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--exact-summary",
        type=Path,
        default=Path("outputs/tables/p1_exact_field_seed1_summary.json"),
    )
    parser.add_argument(
        "--corrected-summary",
        type=Path,
        default=Path("outputs/tables/p1_corrected_field_seed1_summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tables/p1_common125_correction_v2.json"),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Recompute and compare byte-for-byte without writing.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    payload = build_correction(
        root,
        exact_summary_path=args.exact_summary,
        corrected_summary_path=args.corrected_summary,
    )
    output_path = _project_path(root, args.output)
    persist_correction(
        payload,
        output_path,
        verify_existing=args.verify_existing,
    )
    print(_canonical_json(payload).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
