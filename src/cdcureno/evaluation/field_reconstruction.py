"""Reconstruct a legacy temperature field from 51 independent location runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = (prediction - target).reshape(len(target), -1)
    reference = target.reshape(len(target), -1)
    return np.linalg.norm(difference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-12
    )


def reconstruct_field(
    project_root: Path, sweep_state_path: Path, output_root: Path
) -> dict[str, Any]:
    state = json.loads(sweep_state_path.read_text(encoding="utf-8"))
    completed = [task for task in state["tasks"] if task["status"] == "completed"]
    locations = sorted(int(task["location"]) for task in completed)
    if locations != list(range(51)):
        missing = sorted(set(range(51)) - set(locations))
        raise ValueError(f"Field reconstruction requires locations 0-50; missing={missing}")
    if len(completed) != 51:
        raise ValueError(f"Expected 51 completed tasks, found {len(completed)}")

    by_location = {int(task["location"]): task for task in completed}
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    reference_case_ids: np.ndarray | None = None
    reference_air: np.ndarray | None = None
    total_parameters = 0
    total_epoch_seconds = 0.0
    selected_run_ids: list[str] = []

    for location in range(51):
        task = by_location[location]
        run_dir = project_root / "outputs" / "runs" / task["run_id"]
        if not (run_dir / "DONE").is_file():
            raise FileNotFoundError(f"Run is not complete: {run_dir}")
        archive = np.load(run_dir / "predictions" / "test_predictions.npz")
        case_ids = archive["case_ids"]
        input_air = archive["input_air"]
        prediction = archive["prediction"]
        target = archive["target"]
        if reference_case_ids is None:
            reference_case_ids = case_ids
            reference_air = input_air
        else:
            if not np.array_equal(reference_case_ids, case_ids):
                raise ValueError(f"Case ID mismatch at location {location}")
            if not np.array_equal(reference_air, input_air):
                raise ValueError(f"Input-air mismatch at location {location}")
        predictions.append(prediction)
        targets.append(target)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        total_parameters += int(metrics["parameter_count"])
        history = pd.read_parquet(run_dir / "history.parquet")
        total_epoch_seconds += float(history["duration_seconds"].sum())
        selected_run_ids.append(task["run_id"])

    assert reference_case_ids is not None and reference_air is not None
    prediction_field = np.stack(predictions, axis=1)
    target_field = np.stack(targets, axis=1)
    if prediction_field.shape != (len(reference_case_ids), 51, 223):
        raise ValueError(f"Unexpected reconstructed field shape: {prediction_field.shape}")

    difference = prediction_field - target_field
    field_relative = _relative_l2(prediction_field, target_field)
    tool_relative = _relative_l2(prediction_field[:, :21], target_field[:, :21])
    composite_relative = _relative_l2(
        prediction_field[:, 21:], target_field[:, 21:]
    )
    case_rows = pd.DataFrame(
        {
            "case_id": reference_case_ids,
            "field_relative_l2": field_relative,
            "tool_relative_l2": tool_relative,
            "composite_relative_l2": composite_relative,
            "field_mae": np.mean(np.abs(difference), axis=(1, 2)),
            "field_rmse": np.sqrt(np.mean(difference**2, axis=(1, 2))),
            "field_linf": np.max(np.abs(difference), axis=(1, 2)),
            "peak_composite_temperature_error": np.max(
                prediction_field[:, 21:], axis=(1, 2)
            )
            - np.max(target_field[:, 21:], axis=(1, 2)),
        }
    )

    field_dir = output_root / state["sweep_id"]
    field_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        field_dir / "temperature_field.npz",
        case_ids=reference_case_ids,
        input_air=reference_air,
        prediction=prediction_field,
        target=target_field,
    )
    case_rows.to_parquet(field_dir / "metrics_per_case.parquet", index=False)
    started = datetime.fromisoformat(state["started_at"])
    completed_at = datetime.fromisoformat(state["completed_at"])
    summary = {
        "sweep_id": state["sweep_id"],
        "experiment": state["experiment"],
        "seed": int(completed[0]["seed"]),
        "git_sha": state["git_sha"],
        "case_count": int(len(reference_case_ids)),
        "field_shape": list(prediction_field.shape),
        "tool_location_indices": [0, 20],
        "composite_location_indices": [21, 50],
        "field_relative_l2_mean": float(np.mean(field_relative)),
        "field_relative_l2_median": float(np.median(field_relative)),
        "tool_relative_l2_mean": float(np.mean(tool_relative)),
        "composite_relative_l2_mean": float(np.mean(composite_relative)),
        "field_mae_mean": float(case_rows["field_mae"].mean()),
        "field_rmse_mean": float(case_rows["field_rmse"].mean()),
        "field_linf_max": float(case_rows["field_linf"].max()),
        "peak_composite_temperature_error_mae": float(
            case_rows["peak_composite_temperature_error"].abs().mean()
        ),
        "independent_model_parameter_count": int(total_parameters),
        "single_model_parameter_count": int(total_parameters // 51),
        "summed_epoch_seconds": total_epoch_seconds,
        "sweep_wall_seconds": (completed_at - started).total_seconds(),
        "selected_run_ids": selected_run_ids,
        "field_artifact": str(
            (field_dir / "temperature_field.npz").relative_to(project_root)
        ).replace("\\", "/"),
        "per_case_metrics": str(
            (field_dir / "metrics_per_case.parquet").relative_to(project_root)
        ).replace("\\", "/"),
    }
    (field_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
