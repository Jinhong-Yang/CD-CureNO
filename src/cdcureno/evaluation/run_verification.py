"""Independent recomputation of saved run metrics from frozen predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _verify_joint_run(
    run_dir: Path,
    metrics: dict[str, Any],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    from cdcureno.training.joint import compute_joint_field_metrics

    saved_cases = (
        pd.read_parquet(run_dir / "metrics_per_case.parquet")
        .sort_values("case_id")
        .reset_index(drop=True)
    )
    archive = np.load(run_dir / "predictions" / "test_predictions.npz")
    required = (
        "case_ids",
        "temperature_prediction",
        "temperature_target",
        "alpha_prediction",
        "alpha_target",
    )
    missing = [name for name in required if name not in archive.files]
    if missing:
        raise ValueError(f"Joint prediction archive is missing arrays: {missing}")
    recomputed_summary, recomputed_cases = compute_joint_field_metrics(
        archive["case_ids"],
        archive["temperature_prediction"],
        archive["temperature_target"],
        archive["alpha_prediction"],
        archive["alpha_target"],
    )
    recomputed_cases = recomputed_cases.sort_values("case_id").reset_index(drop=True)
    columns = [
        column
        for column in recomputed_cases.columns
        if column != "case_id"
    ]
    column_checks = {
        column: bool(
            np.allclose(
                recomputed_cases[column].to_numpy(),
                saved_cases[column].to_numpy(),
                rtol=rtol,
                atol=atol,
            )
        )
        for column in columns
    }
    summary_checks = {
        key: bool(
            np.isclose(value, metrics["test"][key], rtol=rtol, atol=atol)
        )
        for key, value in recomputed_summary.items()
    }
    case_ids_match = bool(
        np.array_equal(archive["case_ids"], saved_cases["case_id"].to_numpy())
    )
    alpha = archive["alpha_prediction"]
    physical_constraints = {
        "alpha_in_bounds": bool(np.all((alpha >= -1e-7) & (alpha <= 1.0 + 1e-7))),
        "alpha_monotone_in_composite": bool(
            np.all(np.diff(alpha[:, :, 21:], axis=1) >= -1e-7)
        ),
        "alpha_zero_in_tool": bool(np.all(alpha[:, :, :21] == 0.0)),
    }
    passed = (
        case_ids_match
        and all(column_checks.values())
        and all(summary_checks.values())
        and all(physical_constraints.values())
    )
    return {
        "run_id": metrics["run_id"],
        "passed": passed,
        "prediction_shape": list(archive["temperature_prediction"].shape),
        "case_ids_match": case_ids_match,
        "per_case_columns_match": column_checks,
        "summary_metrics_match": summary_checks,
        "physical_constraints": physical_constraints,
        "recomputed_test": recomputed_summary,
    }


def verify_run(run_dir: Path, rtol: float = 1e-6, atol: float = 1e-8) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    cases_path = run_dir / "metrics_per_case.parquet"
    predictions_path = run_dir / "predictions" / "test_predictions.npz"
    for path in (metrics_path, cases_path, predictions_path, run_dir / "DONE"):
        if not path.exists():
            raise FileNotFoundError(f"Required completed-run artifact is missing: {path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("phase") == "P2":
        return _verify_joint_run(run_dir, metrics, rtol, atol)
    saved_cases = pd.read_parquet(cases_path).sort_values("case_id").reset_index(drop=True)
    archive = np.load(predictions_path)
    case_ids = archive["case_ids"]
    prediction = archive["prediction"]
    target = archive["target"]
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    difference = prediction - target
    flattened_difference = difference.reshape(len(target), -1)
    flattened_target = target.reshape(len(target), -1)
    relative_l2 = np.linalg.norm(flattened_difference, axis=1) / np.maximum(
        np.linalg.norm(flattened_target, axis=1), 1e-12
    )
    recomputed = pd.DataFrame(
        {
            "case_id": case_ids,
            "relative_l2": relative_l2,
            "mae": np.mean(np.abs(flattened_difference), axis=1),
            "rmse": np.sqrt(np.mean(flattened_difference**2, axis=1)),
            "linf": np.max(np.abs(flattened_difference), axis=1),
            "peak_value_error": np.max(prediction, axis=1) - np.max(target, axis=1),
            "time_to_peak_index_error": np.argmax(prediction, axis=1)
            - np.argmax(target, axis=1),
        }
    ).sort_values("case_id").reset_index(drop=True)

    columns = [
        "relative_l2",
        "mae",
        "rmse",
        "linf",
        "peak_value_error",
        "time_to_peak_index_error",
    ]
    column_checks = {
        column: bool(
            np.allclose(
                recomputed[column].to_numpy(),
                saved_cases[column].to_numpy(),
                rtol=rtol,
                atol=atol,
            )
        )
        for column in columns
    }
    summary_checks = {
        "relative_l2_mean": np.mean(relative_l2),
        "relative_l2_median": np.median(relative_l2),
        "mae_mean": np.mean(recomputed["mae"]),
        "rmse_mean": np.mean(recomputed["rmse"]),
        "linf_max": np.max(recomputed["linf"]),
        "peak_value_error_mae": np.mean(np.abs(recomputed["peak_value_error"])),
        "time_to_peak_index_error_mae": np.mean(
            np.abs(recomputed["time_to_peak_index_error"])
        ),
    }
    summary_matches = {
        key: bool(np.isclose(value, metrics["test"][key], rtol=rtol, atol=atol))
        for key, value in summary_checks.items()
    }
    case_ids_match = bool(np.array_equal(case_ids, saved_cases["case_id"].to_numpy()))
    passed = case_ids_match and all(column_checks.values()) and all(summary_matches.values())
    return {
        "run_id": metrics["run_id"],
        "passed": passed,
        "prediction_shape": list(prediction.shape),
        "case_ids_match": case_ids_match,
        "per_case_columns_match": column_checks,
        "summary_metrics_match": summary_matches,
        "recomputed_test": {key: float(value) for key, value in summary_checks.items()},
    }
