"""Independent aggregation for the frozen P5 RP-FFNO pilot gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPERIMENT = "p5_rp_ffno_target_pilot_v1"
METHODS = ("scratch_ffno", "restriction_transfer_ffno")
BUDGETS = (8, 16)
SEED = 0
VALIDATION_CASE_IDS = tuple(range(256, 288))
FROZEN_EXPERIMENT_CONFIG_SHA256 = (
    "959a161372efcd8f0bd60d4116fe9d2990569f14eac3eadddcd1bbe05c344f40"
)
FROZEN_RESOURCE_PROFILE_SHA256 = (
    "c8da07384b6d39955fcc97d650c2bacd450598060b218546dcf236a8967c4650"
)
FROZEN_RESOURCE_CONTRACT_SHA256 = (
    "444a004cfdefaea62c493d0999078a39994eab33e06da710b79ab625d081eb83"
)
FROZEN_IMPLEMENTATION_SHA256 = (
    "65ed1e956118e0cdf6ab2dff892c7bd44224316f297ecc59a91538cf015df6a0"
)
FROZEN_INPUTS = {
    "target_id_manifest": (
        "splits/2d_id_v1.json",
        "0628acabce004df2d9d3bb1ac6928e2439dfdbc636575e52ffb1afd9103417f0",
    ),
    "source_checkpoint": (
        "outputs/runs/p3-source-factorized-v4-seed0/best.pt",
        "82d7a497f99a884f2b08eeac799fd7e7ab6391309d3e3037864ecddd8fa3f5f6",
    ),
    "inflated_checkpoint": (
        "outputs/checkpoints/p5_inflated_source.pt",
        "b66918640ddd80bbfccfd0d16c3557155ef26cb415bf280ba965c94d9f2264f0",
    ),
    "target_model_config": (
        "configs/model/cdcureno_2d.yaml",
        "6fb813d5d90e6868f132de1d4c6041b3e2d434bdc954ede4408ce30183dc8c7f",
    ),
    "source_virtual_input_data": (
        "data/processed/p3_source_1d_v3.npz",
        "9db82081f71e444265087531369a2293d4b7880db9b183943490862594d10ffa",
    ),
    "source_virtual_input_split": (
        "splits/p3_source_1d_v3.json",
        "9b06a09734ac04eb5b835948421b07b1db9244f5248841c39d8621b5a42f188c",
    ),
    "declared_experiment_config": (
        "configs/experiment/p5_rp_ffno_pilot_v1.yaml",
        FROZEN_EXPERIMENT_CONFIG_SHA256,
    ),
}
PRIMARY_COLUMN = "temperature_relative_l2_K_composite"
ALPHA_COLUMN = "alpha_relative_l2_composite"
PEAK_COLUMN = "peak_temperature_absolute_error_K"
PER_CASE_COLUMNS = (
    "temperature_relative_l2_K_composite",
    "temperature_mae_K_composite",
    "temperature_rmse_K_composite",
    "temperature_linf_K_composite",
    "peak_temperature_absolute_error_K",
    "alpha_relative_l2_composite",
    "alpha_mae_composite",
    "temperature_gradient_x_relative_l2",
    "temperature_gradient_z_relative_l2",
    "maximum_lateral_gradient_error_K_per_m",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    error = (prediction - target)[mask]
    reference = target[mask]
    return float(
        np.linalg.norm(error)
        / max(np.linalg.norm(reference), np.finfo(np.float64).eps)
    )


def recompute_validation_cases(
    predictions_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Recompute all published per-case metrics from saved validation fields."""

    with np.load(predictions_path, allow_pickle=False) as payload:
        required = {
            "case_ids",
            "temperature_prediction_K",
            "temperature_target_K",
            "alpha_prediction",
            "alpha_target",
            "composite_mask",
            "x_m",
            "z_m",
        }
        if set(payload.files) != required:
            raise ValueError(
                "Validation prediction artifact keys differ from the frozen "
                f"schema: {sorted(payload.files)}."
            )
        arrays = {name: np.asarray(payload[name]) for name in required}
    case_ids = arrays["case_ids"].astype(np.int64, copy=False)
    predicted_temperature = arrays["temperature_prediction_K"]
    target_temperature = arrays["temperature_target_K"]
    predicted_alpha = arrays["alpha_prediction"]
    target_alpha = arrays["alpha_target"]
    mask = arrays["composite_mask"].astype(bool, copy=False)
    x = arrays["x_m"].astype(np.float64, copy=False)
    z = arrays["z_m"].astype(np.float64, copy=False)
    if tuple(case_ids.tolist()) != VALIDATION_CASE_IDS:
        raise ValueError("Prediction artifact has the wrong validation IDs.")
    expected_shape = predicted_temperature.shape
    if (
        expected_shape != target_temperature.shape
        or expected_shape != predicted_alpha.shape
        or expected_shape != target_alpha.shape
        or expected_shape[0] != len(VALIDATION_CASE_IDS)
        or expected_shape[2] != len(z)
        or expected_shape[3] != len(x)
        or mask.shape != expected_shape[2:]
    ):
        raise ValueError("Validation prediction array shapes are inconsistent.")
    if (
        not np.isfinite(predicted_temperature).all()
        or not np.isfinite(target_temperature).all()
        or not np.isfinite(predicted_alpha).all()
        or not np.isfinite(target_alpha).all()
        or not np.isfinite(x).all()
        or not np.isfinite(z).all()
        or not np.all(np.diff(x) > 0.0)
        or not np.all(np.diff(z) > 0.0)
        or not np.any(mask)
    ):
        raise ValueError("Validation prediction artifact is non-finite or invalid.")
    mask_x_2d = mask[..., 1:] & mask[..., :-1]
    mask_z_2d = mask[1:, :] & mask[:-1, :]
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        temperature_prediction = predicted_temperature[index]
        temperature_truth = target_temperature[index]
        alpha_prediction = predicted_alpha[index]
        alpha_truth = target_alpha[index]
        composite_mask = np.broadcast_to(
            mask,
            temperature_prediction.shape,
        )
        mask_x = np.broadcast_to(
            mask_x_2d,
            (temperature_prediction.shape[0], *mask_x_2d.shape),
        )
        mask_z = np.broadcast_to(
            mask_z_2d,
            (temperature_prediction.shape[0], *mask_z_2d.shape),
        )
        prediction_gradient_x = np.diff(
            temperature_prediction, axis=2
        ) / np.diff(x)[None, None, :]
        target_gradient_x = np.diff(
            temperature_truth, axis=2
        ) / np.diff(x)[None, None, :]
        prediction_gradient_z = np.diff(
            temperature_prediction, axis=1
        ) / np.diff(z)[None, :, None]
        target_gradient_z = np.diff(
            temperature_truth, axis=1
        ) / np.diff(z)[None, :, None]
        temperature_error = temperature_prediction - temperature_truth
        alpha_error = alpha_prediction - alpha_truth
        rows.append(
            {
                "split": "validation",
                "case_id": int(case_id),
                PRIMARY_COLUMN: _relative_l2(
                    temperature_prediction,
                    temperature_truth,
                    composite_mask,
                ),
                "temperature_mae_K_composite": float(
                    np.mean(np.abs(temperature_error)[composite_mask])
                ),
                "temperature_rmse_K_composite": float(
                    np.sqrt(
                        np.mean(temperature_error[composite_mask] ** 2)
                    )
                ),
                "temperature_linf_K_composite": float(
                    np.max(np.abs(temperature_error)[composite_mask])
                ),
                PEAK_COLUMN: float(
                    abs(
                        np.max(temperature_prediction[composite_mask])
                        - np.max(temperature_truth[composite_mask])
                    )
                ),
                ALPHA_COLUMN: _relative_l2(
                    alpha_prediction,
                    alpha_truth,
                    composite_mask,
                ),
                "alpha_mae_composite": float(
                    np.mean(np.abs(alpha_error)[composite_mask])
                ),
                "temperature_gradient_x_relative_l2": _relative_l2(
                    prediction_gradient_x, target_gradient_x, mask_x
                ),
                "temperature_gradient_z_relative_l2": _relative_l2(
                    prediction_gradient_z, target_gradient_z, mask_z
                ),
                "maximum_lateral_gradient_error_K_per_m": float(
                    np.max(
                        np.abs(
                            prediction_gradient_x - target_gradient_x
                        )[mask_x]
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows)
    targets = {
        "temperature_target_K": target_temperature,
        "alpha_target": target_alpha,
        "composite_mask": mask,
        "x_m": x,
        "z_m": z,
    }
    return frame, targets


def summarize_cases(frame: pd.DataFrame) -> dict[str, float | int]:
    if tuple(frame["case_id"].astype(int)) != VALIDATION_CASE_IDS:
        raise ValueError("Per-case frame has the wrong validation order.")
    summary: dict[str, float | int] = {"case_count": len(frame)}
    for column in PER_CASE_COLUMNS:
        summary[f"{column}_mean"] = float(frame[column].mean())
        summary[f"{column}_median"] = float(frame[column].median())
        summary[f"{column}_max"] = float(frame[column].max())
    return summary


def _numeric_mapping_matches(
    recorded: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    if set(recorded) != set(expected):
        return False
    for name, expected_value in expected.items():
        actual = recorded[name]
        if isinstance(expected_value, int):
            if isinstance(actual, bool) or int(actual) != expected_value:
                return False
        elif not np.isclose(
            float(actual),
            float(expected_value),
            rtol=1.0e-7,
            atol=1.0e-12,
        ):
            return False
    return True


def _artifact_path(
    project_root: Path,
    run_dir: Path,
    record: Mapping[str, Any],
    expected_relative: str,
) -> Path:
    declared = Path(str(record.get("path", "")))
    path = (
        declared.resolve()
        if declared.is_absolute()
        else (project_root / declared).resolve()
    )
    expected = (run_dir / expected_relative).resolve()
    if path != expected or not path.is_file():
        raise ValueError(f"Run artifact path differs: {expected_relative}.")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"Run artifact hash differs: {expected_relative}.")
    return path


@dataclass(frozen=True)
class PilotRunEvidence:
    run_dir: Path
    metrics: dict[str, Any]
    resolved_config: dict[str, Any]
    cases: pd.DataFrame
    targets: dict[str, np.ndarray]
    recomputed_summary: dict[str, float | int]


def _expected_scientific_config(
    project_root: Path,
    run_dir: Path,
    *,
    method: str,
    budget: int,
) -> dict[str, Any]:
    root = project_root.resolve()
    absolute = lambda relative: (root / relative).resolve().as_posix()
    return {
        "alpha_weight": 0.5,
        "config_file": absolute(
            "configs/experiment/p5_rp_ffno_pilot_v1.yaml"
        ),
        "depth": 4,
        "device": "cuda",
        "early_stopping_patience": 20,
        "effective_batch_size": 4,
        "epochs": 120,
        "expected_inflated_checkpoint_sha256": FROZEN_INPUTS[
            "inflated_checkpoint"
        ][1],
        "expected_parameter_count": 172610,
        "expected_source_checkpoint_sha256": FROZEN_INPUTS[
            "source_checkpoint"
        ][1],
        "expected_source_data_sha256": FROZEN_INPUTS[
            "source_virtual_input_data"
        ][1],
        "expected_source_split_sha256": FROZEN_INPUTS[
            "source_virtual_input_split"
        ][1],
        "expected_target_model_config_sha256": FROZEN_INPUTS[
            "target_model_config"
        ][1],
        "expected_target_split_sha256": FROZEN_INPUTS[
            "target_id_manifest"
        ][1],
        "gradient_accumulation_steps": 2,
        "gradient_clip": 1.0,
        "gradient_x_weight": 0.05,
        "gradient_z_weight": 0.05,
        "inflated_checkpoint": absolute(
            FROZEN_INPUTS["inflated_checkpoint"][0]
        ),
        "label_budget": budget,
        "lateral_rank": 4,
        "learning_rate": 0.001,
        "method": method,
        "micro_batch_size": 2,
        "minimum_epochs": 40,
        "modes_time": 24,
        "modes_x": 12,
        "modes_z": 12,
        "num_threads": 8,
        "output_root": absolute("outputs/runs"),
        "preflight_candidate_batch_sizes": [2, 1],
        "project_root": root.as_posix(),
        "require_resource_profile": True,
        "resource_profile_path": absolute(
            "outputs/tables/p5_rp_ffno_resource_preflight.json"
        ),
        "restriction_lateral_invariance_max": 1.0e-6,
        "restriction_lateral_range_max": 1.0e-5,
        "restriction_validation_case_count": 4,
        "restriction_weight": 0.1,
        "run_id": run_dir.name,
        "seed": SEED,
        "source_checkpoint": absolute(
            FROZEN_INPUTS["source_checkpoint"][0]
        ),
        "source_data_path": absolute(
            FROZEN_INPUTS["source_virtual_input_data"][0]
        ),
        "source_split_manifest": absolute(
            FROZEN_INPUTS["source_virtual_input_split"][0]
        ),
        "target_model_config": absolute(
            FROZEN_INPUTS["target_model_config"][0]
        ),
        "target_split_manifest": absolute(
            FROZEN_INPUTS["target_id_manifest"][0]
        ),
        "temperature_weight": 1.0,
        "verify_array_checksums": True,
        "weight_decay": 0.0001,
        "width": 32,
    }


def _validate_frozen_run_contract(
    project_root: Path,
    run_dir: Path,
    metrics: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    method: str,
    budget: int,
) -> None:
    root = project_root.resolve()
    if resolved.get("scientific_config") != _expected_scientific_config(
        root,
        run_dir,
        method=method,
        budget=budget,
    ):
        raise ValueError("Pilot scientific config differs from the frozen YAML.")
    input_checksums = metrics.get("input_checksums")
    if not isinstance(input_checksums, Mapping) or set(
        input_checksums
    ) != set(FROZEN_INPUTS):
        raise ValueError("Pilot input-checksum schema differs.")
    for name, (relative, expected_sha) in FROZEN_INPUTS.items():
        record = input_checksums[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"Pilot input record is invalid: {name}.")
        expected_path = (root / relative).resolve()
        declared_path = Path(str(record.get("path", ""))).resolve()
        if (
            declared_path != expected_path
            or not expected_path.is_file()
            or record.get("sha256") != expected_sha
            or sha256_file(expected_path) != expected_sha
            or int(record.get("bytes", -1)) != expected_path.stat().st_size
        ):
            raise ValueError(f"Pilot frozen input differs: {name}.")
    profile_path = (
        root / "outputs" / "tables" / "p5_rp_ffno_resource_preflight.json"
    ).resolve()
    if (
        sha256_file(profile_path) != FROZEN_RESOURCE_PROFILE_SHA256
        or metrics.get("resource_profile") != _read_json(profile_path)
    ):
        raise ValueError("Pilot resource profile differs from the frozen file.")
    profile = metrics["resource_profile"]
    if (
        profile.get("passed") is not True
        or profile.get("resource_contract_sha256")
        != FROZEN_RESOURCE_CONTRACT_SHA256
        or profile.get("resource_contract", {}).get(
            "implementation_sha256"
        )
        != FROZEN_IMPLEMENTATION_SHA256
        or metrics.get("implementation_sha256")
        != FROZEN_IMPLEMENTATION_SHA256
    ):
        raise ValueError("Pilot implementation/resource contract differs.")
    expected_model = {
        **profile["resource_contract"]["architecture"],
        "parameter_count": 172610,
        "trainable_parameter_count": 172610,
        "transfer_stage": "T2",
    }
    if metrics.get("model") != expected_model:
        raise ValueError("Pilot model differs from the frozen architecture.")
    expected_training = {
        "configured_epochs": 120,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": False,
        "device": "cuda:0",
    }
    training = metrics.get("training", {})
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("Pilot training settings differ from the frozen run.")
    if metrics.get("loss_weights") != profile["resource_contract"][
        "loss_weights"
    ]:
        raise ValueError("Pilot loss weights differ from the frozen run.")
    access = metrics.get("target_label_access_audit", {})
    if access.get("train") != list(range(budget)):
        raise ValueError("Pilot did not use the exact frozen nested budget.")
    restriction = metrics.get("restriction_validation", {})
    thresholds = restriction.get("thresholds", {})
    if (
        thresholds
        != {
            "lateral_invariance_score_max": 1.0e-6,
            "maximum_lateral_range_max": 1.0e-5,
        }
        or restriction.get("passed") is not True
        or not all(restriction.get("contract_checks", {}).values())
    ):
        raise ValueError("Selected-model restriction thresholds failed.")
    initialization = metrics.get("initialization", {})
    if method == "scratch_ffno":
        expected_initialization = {
            "method": method,
            "deterministic_seed": 0,
            "source_checkpoint_weights_loaded": False,
            "inflated_checkpoint_weights_loaded": False,
            "architecture_specific_lateral_zero_residual_preserved": True,
            "initial_state_sha256": (
                "5359528e73024c36f7077f3fbad51d517b2d11a134c1354d8af58fb5afd1d758"
            ),
        }
    else:
        expected_initialization = {
            "method": method,
            "deterministic_seed": 20260726,
            "source_checkpoint_weights_loaded": True,
            "inflated_checkpoint_weights_loaded": True,
            "inflation_verification_passed": True,
            "inflated_checkpoint_sha256": FROZEN_INPUTS[
                "inflated_checkpoint"
            ][1],
            "initial_state_sha256": (
                "191de8efe232716872d65690ef5ac1f68c5c87854cd32487b3007099698b241d"
            ),
        }
    if any(
        initialization.get(key) != value
        for key, value in expected_initialization.items()
    ):
        raise ValueError("Pilot initialization differs from its frozen method.")


def load_pilot_run(
    project_root: Path,
    run_dir: Path,
    *,
    expected_method: str,
    expected_budget: int,
) -> PilotRunEvidence:
    root = project_root.resolve()
    directory = run_dir.resolve()
    required = (
        directory / "metrics.json",
        directory / "config_resolved.json",
        directory / "STATUS.json",
        directory / "DONE",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Pilot run is incomplete: {directory}")
    if (directory / "FAILED").exists():
        raise ValueError(f"Pilot run retains FAILED: {directory}")
    metrics = _read_json(directory / "metrics.json")
    status = _read_json(directory / "STATUS.json")
    if (
        metrics.get("status") != "completed"
        or status.get("status") != "completed"
        or metrics.get("experiment") != EXPERIMENT
        or metrics.get("phase") != "P5"
        or metrics.get("method") != expected_method
        or int(metrics.get("label_budget", -1)) != expected_budget
        or int(metrics.get("seed", -1)) != SEED
        or metrics.get("run_id") != directory.name
    ):
        raise ValueError(f"Pilot run identity/status differs: {directory}")
    if (
        metrics["selection"].get("split") != "validation"
        or metrics["selection"].get(
            "target_test_or_ood_labels_used"
        )
        is not False
        or metrics.get("pilot_gate_decision")
        != "requires_all_four_runs_and_paired_aggregator"
    ):
        raise ValueError("Pilot checkpoint selection contract differs.")
    access = metrics.get("target_label_access_audit", {})
    if (
        access.get("id_test") != []
        or access.get("ood") != []
        or tuple(access.get("validation", ())) != VALIDATION_CASE_IDS
        or len(access.get("train", ())) != expected_budget
    ):
        raise ValueError("Pilot target-label access audit failed.")
    restriction = metrics.get("restriction_validation")
    if (
        not isinstance(restriction, dict)
        or restriction.get("passed") is not True
        or restriction.get("provenance", {}).get("target_labels_used")
        is not False
        or restriction.get("provenance", {}).get(
            "source_teacher_predictions_used"
        )
        is not False
    ):
        raise ValueError("Selected-model restriction audit failed.")
    artifacts = metrics.get("auditable_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Pilot metrics have no auditable-artifact mapping.")
    history_path = _artifact_path(
        root, directory, artifacts["history"], "history.parquet"
    )
    case_path = _artifact_path(
        root,
        directory,
        artifacts["validation_metrics_per_case"],
        "metrics_per_case_validation.parquet",
    )
    prediction_path = _artifact_path(
        root,
        directory,
        artifacts["validation_predictions"],
        "predictions/validation_best.npz",
    )
    restriction_path = _artifact_path(
        root,
        directory,
        artifacts["restriction_validation"],
        "restriction_validation.json",
    )
    if _read_json(restriction_path) != restriction:
        raise ValueError("Restriction JSON differs from metrics.json.")
    history = pd.read_parquet(history_path)
    if (
        history.empty
        or history["epoch"].astype(int).tolist()
        != list(range(1, len(history) + 1))
        or int(metrics["selection"]["selected_epoch"])
        != int(
            history.loc[
                history["validation_weighted_objective"].idxmin(), "epoch"
            ]
        )
        or not np.isclose(
            float(metrics["selection"]["best_validation_objective"]),
            float(history["validation_weighted_objective"].min()),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Pilot validation-history selection is inconsistent.")
    recorded_cases = pd.read_parquet(case_path).sort_values(
        "case_id"
    ).reset_index(drop=True)
    recomputed_cases, targets = recompute_validation_cases(prediction_path)
    if list(recorded_cases.columns) != list(recomputed_cases.columns):
        raise ValueError("Published and recomputed per-case schemas differ.")
    for column in ("split", "case_id"):
        if not recorded_cases[column].equals(recomputed_cases[column]):
            raise ValueError(f"Per-case identity column differs: {column}.")
    for column in PER_CASE_COLUMNS:
        if not np.allclose(
            recorded_cases[column].to_numpy(dtype=np.float64),
            recomputed_cases[column].to_numpy(dtype=np.float64),
            rtol=1.0e-7,
            atol=1.0e-12,
        ):
            raise ValueError(f"Per-case metric differs: {column}.")
    summary = summarize_cases(recomputed_cases)
    if not _numeric_mapping_matches(metrics["validation_metrics"], summary):
        raise ValueError("Published validation summary was not reproduced.")
    for kind in ("best", "last"):
        record = metrics["checkpoints"][kind]
        expected = f"checkpoints/{kind}.pt"
        _artifact_path(root, directory, record, expected)
    resolved = _read_json(directory / "config_resolved.json")
    _validate_frozen_run_contract(
        root,
        directory,
        metrics,
        resolved,
        method=expected_method,
        budget=expected_budget,
    )
    return PilotRunEvidence(
        run_dir=directory,
        metrics=metrics,
        resolved_config=resolved,
        cases=recomputed_cases,
        targets=targets,
        recomputed_summary=summary,
    )


def _same_targets(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> bool:
    return set(left) == set(right) and all(
        np.array_equal(left[name], right[name]) for name in left
    )


def _method_pair_contract_matches(
    scratch: PilotRunEvidence,
    transfer: PilotRunEvidence,
) -> bool:
    scratch_scientific = dict(scratch.resolved_config["scientific_config"])
    transfer_scientific = dict(transfer.resolved_config["scientific_config"])
    for payload in (scratch_scientific, transfer_scientific):
        payload.pop("method", None)
        payload.pop("run_id", None)
        payload.pop("resume", None)
    scratch_metrics = scratch.metrics
    transfer_metrics = transfer.metrics
    training_keys = (
        "configured_epochs",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "optimizer",
        "scheduler",
        "mixed_precision",
        "device",
    )
    return bool(
        scratch_scientific == transfer_scientific
        and scratch_metrics["git_sha"] == transfer_metrics["git_sha"]
        and scratch_metrics["implementation_sha256"]
        == transfer_metrics["implementation_sha256"]
        and scratch_metrics["model"] == transfer_metrics["model"]
        and scratch_metrics["parameter_groups"]
        == transfer_metrics["parameter_groups"]
        and scratch_metrics["loss_weights"] == transfer_metrics["loss_weights"]
        and all(
            scratch_metrics["training"][key]
            == transfer_metrics["training"][key]
            for key in training_keys
        )
        and scratch_metrics["resource_profile"]
        == transfer_metrics["resource_profile"]
        and scratch_metrics.get("input_checksums")
        == transfer_metrics.get("input_checksums")
        and scratch_metrics.get("target_data_checksums")
        == transfer_metrics.get("target_data_checksums")
        and scratch_metrics.get("normalization")
        == transfer_metrics.get("normalization")
        and scratch_metrics["target_label_access_audit"]
        == transfer_metrics["target_label_access_audit"]
        and _same_targets(scratch.targets, transfer.targets)
        and scratch_metrics["initialization"]["method"] == "scratch_ffno"
        and transfer_metrics["initialization"]["method"]
        == "restriction_transfer_ffno"
        and scratch_metrics["initialization"][
            "source_checkpoint_weights_loaded"
        ]
        is False
        and transfer_metrics["initialization"][
            "source_checkpoint_weights_loaded"
        ]
        is True
    )


def _quartet_contract_matches(
    runs: Mapping[tuple[str, int], PilotRunEvidence],
) -> bool:
    evidence = list(runs.values())
    normalized_configs = []
    for run in evidence:
        scientific = dict(run.resolved_config["scientific_config"])
        for key in ("method", "label_budget", "run_id"):
            scientific.pop(key, None)
        normalized_configs.append(scientific)
    first = evidence[0]
    invariant_training_keys = (
        "configured_epochs",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "optimizer",
        "scheduler",
        "mixed_precision",
        "device",
    )
    return bool(
        all(config == normalized_configs[0] for config in normalized_configs)
        and all(
            run.metrics["git_sha"] == first.metrics["git_sha"]
            and run.metrics["implementation_sha256"]
            == first.metrics["implementation_sha256"]
            and run.metrics["model"] == first.metrics["model"]
            and run.metrics["parameter_groups"]
            == first.metrics["parameter_groups"]
            and run.metrics["loss_weights"] == first.metrics["loss_weights"]
            and run.metrics["resource_profile"]
            == first.metrics["resource_profile"]
            and run.metrics.get("input_checksums")
            == first.metrics.get("input_checksums")
            and all(
                run.metrics["training"][key]
                == first.metrics["training"][key]
                for key in invariant_training_keys
            )
            and _same_targets(run.targets, first.targets)
            for run in evidence[1:]
        )
        and all(
            len(run.metrics["git_sha"]) == 40
            and all(
                character in "0123456789abcdef"
                for character in run.metrics["git_sha"]
            )
            for run in evidence
        )
    )


def _restriction_inflation_gate_matches(
    restriction_gate: Mapping[str, Any],
    runs: Mapping[tuple[str, int], PilotRunEvidence],
) -> bool:
    integrity = restriction_gate.get("inflation_integrity", {})
    integrity_checks = integrity.get("checks", {})
    expected_source = FROZEN_INPUTS["source_checkpoint"][1]
    expected_target = FROZEN_INPUTS["inflated_checkpoint"][1]
    expected_config = FROZEN_INPUTS["target_model_config"][1]
    transfer_runs = [
        runs[("restriction_transfer_ffno", budget)] for budget in BUDGETS
    ]
    return bool(
        restriction_gate.get("schema_version") == 2
        and restriction_gate.get("phase") == "P5"
        and restriction_gate.get("gate")
        == "canonical_inflation_and_source_to_target_restriction"
        and restriction_gate.get("dry_run") is False
        and restriction_gate.get("passed") is True
        and restriction_gate.get("uses_target_temperature_or_cure_labels")
        is False
        and restriction_gate.get("source_checkpoint_sha256")
        == expected_source
        and restriction_gate.get("target_checkpoint_sha256")
        == expected_target
        and restriction_gate.get("target_config_sha256")
        == expected_config
        and restriction_gate.get("p4_id_split_sha256")
        == FROZEN_INPUTS["target_id_manifest"][1]
        and restriction_gate.get("actual_p3_validation", {}).get("passed")
        is True
        and restriction_gate.get("actual_p4_validation_f0", {}).get("passed")
        is True
        and integrity.get("schema_version") == 1
        and integrity.get("passed") is True
        and integrity.get("deterministic_seed") == 20260726
        and integrity.get("state_tensor_count") == 61
        and integrity.get("verification")
        == "deterministic_full_state_recreation"
        and integrity.get("source_checkpoint_sha256") == expected_source
        and integrity.get("target_config_sha256") == expected_config
        and bool(integrity_checks)
        and all(value is True for value in integrity_checks.values())
        and all(
            run.metrics["initialization"].get(
                "inflation_verification_passed"
            )
            is True
            and run.metrics["initialization"].get(
                "inflated_checkpoint_sha256"
            )
            == expected_target
            and run.metrics["input_checksums"]["source_checkpoint"][
                "sha256"
            ]
            == expected_source
            and run.metrics["input_checksums"]["inflated_checkpoint"][
                "sha256"
            ]
            == expected_target
            and run.metrics["input_checksums"]["target_model_config"][
                "sha256"
            ]
            == expected_config
            for run in transfer_runs
        )
    )


def decide_pilot_gate(
    runs: Mapping[tuple[str, int], PilotRunEvidence],
    *,
    restriction_inflation_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the pre-registered validation criteria to four pilot runs."""

    expected_keys = {
        (method, budget) for method in METHODS for budget in BUDGETS
    }
    if set(runs) != expected_keys:
        raise ValueError("Exactly four method/budget pilot runs are required.")
    comparisons: dict[str, Any] = {}
    method_contract_checks: dict[str, bool] = {}
    lower_temperature_checks: dict[str, bool] = {}
    alpha_guardrails: dict[str, bool] = {}
    peak_guardrails: dict[str, bool] = {}
    for budget in BUDGETS:
        scratch = runs[("scratch_ffno", budget)]
        transfer = runs[("restriction_transfer_ffno", budget)]
        scratch_cases = scratch.cases.set_index("case_id")
        transfer_cases = transfer.cases.set_index("case_id")
        scratch_temperature = float(scratch_cases[PRIMARY_COLUMN].mean())
        transfer_temperature = float(transfer_cases[PRIMARY_COLUMN].mean())
        relative_improvement = (
            scratch_temperature - transfer_temperature
        ) / max(scratch_temperature, np.finfo(np.float64).eps)
        paired = (
            scratch_cases[PRIMARY_COLUMN] - transfer_cases[PRIMARY_COLUMN]
        ) / scratch_cases[PRIMARY_COLUMN].clip(
            lower=np.finfo(np.float64).eps
        )
        scratch_alpha = float(scratch_cases[ALPHA_COLUMN].mean())
        transfer_alpha = float(transfer_cases[ALPHA_COLUMN].mean())
        scratch_peak = float(scratch_cases[PEAK_COLUMN].mean())
        transfer_peak = float(transfer_cases[PEAK_COLUMN].mean())
        key = str(budget)
        method_contract_checks[key] = _method_pair_contract_matches(
            scratch, transfer
        )
        lower_temperature_checks[key] = (
            transfer_temperature < scratch_temperature
        )
        alpha_guardrails[key] = transfer_alpha <= 1.10 * scratch_alpha
        peak_guardrails[key] = transfer_peak <= 1.10 * scratch_peak
        comparisons[key] = {
            "scratch_run_id": scratch.metrics["run_id"],
            "transfer_run_id": transfer.metrics["run_id"],
            "validation_case_count": len(scratch_cases),
            "scratch_temperature_relative_l2_mean": scratch_temperature,
            "transfer_temperature_relative_l2_mean": transfer_temperature,
            "temperature_relative_improvement": relative_improvement,
            "paired_case_relative_improvement_median": float(paired.median()),
            "paired_case_transfer_better_count": int((paired > 0.0).sum()),
            "scratch_alpha_relative_l2_mean": scratch_alpha,
            "transfer_alpha_relative_l2_mean": transfer_alpha,
            "alpha_transfer_to_scratch_ratio": (
                transfer_alpha
                / max(scratch_alpha, np.finfo(np.float64).eps)
            ),
            "scratch_peak_temperature_absolute_error_K_mean": scratch_peak,
            "transfer_peak_temperature_absolute_error_K_mean": transfer_peak,
            "peak_error_transfer_to_scratch_ratio": (
                transfer_peak
                / max(scratch_peak, np.finfo(np.float64).eps)
            ),
        }
    all_restriction_checks = bool(
        _restriction_inflation_gate_matches(
            restriction_inflation_gate,
            runs,
        )
        and all(
            run.metrics["restriction_validation"]["passed"] is True
            and all(
                run.metrics["restriction_validation"][
                    "contract_checks"
                ].values()
            )
            for run in runs.values()
        )
    )
    all_leakage_checks = all(
        run.metrics["selection"]["target_test_or_ood_labels_used"] is False
        and run.metrics["target_label_access_audit"]["id_test"] == []
        and run.metrics["target_label_access_audit"]["ood"] == []
        for run in runs.values()
    )
    checks = {
        "all_four_runs_share_frozen_contract": _quartet_contract_matches(
            runs
        ),
        "method_conditions_match_at_budget_8": method_contract_checks["8"],
        "method_conditions_match_at_budget_16": method_contract_checks["16"],
        "transfer_temperature_lower_at_budget_8": (
            lower_temperature_checks["8"]
        ),
        "transfer_temperature_lower_at_budget_16": (
            lower_temperature_checks["16"]
        ),
        "budget_16_relative_improvement_at_least_1_percent": (
            comparisons["16"]["temperature_relative_improvement"] >= 0.01
        ),
        "budget_16_paired_median_improvement_positive": (
            comparisons["16"][
                "paired_case_relative_improvement_median"
            ]
            > 0.0
        ),
        "alpha_guardrail_budget_8": alpha_guardrails["8"],
        "alpha_guardrail_budget_16": alpha_guardrails["16"],
        "peak_temperature_guardrail_budget_8": peak_guardrails["8"],
        "peak_temperature_guardrail_budget_16": peak_guardrails["16"],
        "all_restriction_checks_pass": all_restriction_checks,
        "all_leakage_checks_pass": all_leakage_checks,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "phase": "P5",
        "gate": "rp_ffno_transfer_initialization_pilot",
        "decision": "passed" if passed else "failed_preserved",
        "scope": "one_seed_validation_only_stage_gate",
        "uses_target_id_test_or_ood_labels": False,
        "criteria": {
            "temperature_lower_at_both_budgets": True,
            "budget_16_relative_improvement_minimum": 0.01,
            "budget_16_paired_median_improvement_positive": True,
            "alpha_relative_l2_maximum_worsening": 0.10,
            "peak_temperature_error_maximum_worsening": 0.10,
            "restriction_and_leakage_checks_required": True,
        },
        "comparisons": comparisons,
        "checks": checks,
        "passed": passed,
    }
