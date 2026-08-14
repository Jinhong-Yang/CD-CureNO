"""Validate and aggregate the frozen parameter-matched P2 pilot runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUN_IDS = (
    "p2pilot30-noncausal-w16d4-mt8mz6-seed0-1760fae",
    "p2pilot30-factorized-w40d4-mt16mz12-seed0-1760fae",
    "p2pilot30-causal-w40d8-mz12-seed0-1760fae",
)


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = (prediction - target).reshape(len(target), -1)
    flattened_target = target.reshape(len(target), -1)
    return np.linalg.norm(difference, axis=1) / np.maximum(
        np.linalg.norm(flattened_target, axis=1), 1e-12
    )


def _legacy_gradient_baseline(project_root: Path) -> dict:
    rows = []
    for seed in (0, 1, 2):
        summary_path = (
            project_root
            / "outputs"
            / "tables"
            / f"p1_corrected_field_seed{seed}_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        archive = np.load(project_root / summary["field_artifact"])
        prediction = archive["prediction"]
        target = archive["target"]
        if prediction.shape != (125, 51, 223) or target.shape != prediction.shape:
            raise ValueError(f"Unexpected P1 field shape for seed {seed}.")
        prediction_gradient = np.diff(prediction, axis=1)
        target_gradient = np.diff(target, axis=1)
        rows.append(
            {
                "seed": seed,
                "spatial_gradient_relative_l2_mean": float(
                    _relative_l2(prediction_gradient, target_gradient).mean()
                ),
                "spatial_gradient_mae_mean": float(
                    np.mean(np.abs(prediction_gradient - target_gradient), axis=(1, 2))
                    .mean()
                ),
            }
        )
    return {
        "per_seed": rows,
        "spatial_gradient_relative_l2_mean": float(
            np.mean(
                [row["spatial_gradient_relative_l2_mean"] for row in rows]
            )
        ),
        "spatial_gradient_mae_mean": float(
            np.mean([row["spatial_gradient_mae_mean"] for row in rows])
        ),
        "definition": (
            "Complete-case relative L2 and MAE of first differences between "
            "adjacent 1 mm through-thickness positions."
        ),
    }


def aggregate(project_root: Path, run_ids: tuple[str, ...]) -> dict:
    rows = []
    for run_id in run_ids:
        run_dir = project_root / "outputs" / "runs" / run_id
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        verification = json.loads(
            (run_dir / "verification.json").read_text(encoding="utf-8")
        )
        history = pd.read_parquet(run_dir / "history.parquet")
        best = history.loc[history["validation_weighted_objective"].idxmin()]
        rows.append(
            {
                "run_id": run_id,
                "model_family": metrics["model_family"],
                "seed": metrics["seed"],
                "epochs": len(history),
                "parameter_count": metrics["parameter_count"],
                "summed_epoch_seconds": float(history["duration_seconds"].sum()),
                "peak_process_rss_bytes": metrics["peak_process_rss_bytes"],
                "selected_epoch": metrics["selected_epoch"],
                "best_validation_temperature_relative_l2": float(
                    best["validation_temperature_relative_l2"]
                ),
                "best_validation_alpha_relative_l2": float(
                    best["validation_alpha_relative_l2"]
                ),
                "best_validation_spatial_gradient_relative_l2": float(
                    best["validation_spatial_gradient_relative_l2"]
                ),
                "best_validation_weighted_objective": float(
                    best["validation_weighted_objective"]
                ),
                "test_field_relative_l2_mean": metrics["test"][
                    "field_relative_l2_mean"
                ],
                "test_alpha_relative_l2_mean": metrics["test"][
                    "alpha_relative_l2_mean"
                ],
                "test_spatial_gradient_relative_l2_mean": metrics["test"][
                    "spatial_gradient_relative_l2_mean"
                ],
                "test_field_linf_max": metrics["test"]["field_linf_max"],
                "verification_passed": verification["passed"],
                "physical_constraints_passed": all(
                    verification["physical_constraints"].values()
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("model_family").reset_index(drop=True)
    output_dir = project_root / "outputs" / "tables"
    frame.to_csv(output_dir / "p2_pilot_runs.csv", index=False)

    p1 = json.loads(
        (output_dir / "p1_field_summary.json").read_text(encoding="utf-8")
    )
    baseline_gradient = _legacy_gradient_baseline(project_root)
    corrected_field = p1["corrected_three_seed"]["field_relative_l2"]["mean"]
    best_validation_row = frame.loc[
        frame["best_validation_weighted_objective"].idxmin()
    ]
    best_test_row = frame.loc[frame["test_field_relative_l2_mean"].idxmin()]
    parameter_ratio = float(
        frame["parameter_count"].max() / frame["parameter_count"].min()
    )
    gate_checks = {
        "all_runs_independently_verified": bool(frame["verification_passed"].all()),
        "all_physical_constraints_passed": bool(
            frame["physical_constraints_passed"].all()
        ),
        "parameter_count_ratio_at_most_1_15": parameter_ratio <= 1.15,
        "temperature_field_matches_corrected_baseline": bool(
            best_test_row["test_field_relative_l2_mean"] <= corrected_field
        ),
        "spatial_gradient_lower_or_unchanged": bool(
            best_test_row["test_spatial_gradient_relative_l2_mean"]
            <= baseline_gradient["spatial_gradient_relative_l2_mean"]
        ),
    }
    summary = {
        "status": "P2_pilot_only_not_gate",
        "validation_assessment": "Needs revision",
        "scope": (
            "Three seed-0, 30-epoch, approximately parameter-matched Case1 "
            "joint temperature/cure pilots."
        ),
        "data_and_metric_checks": {
            "test_case_count": 125,
            "canonical_joint_axis_order": ["case", "time", "position"],
            "legacy_axis_order_reoriented_for_gradient_check": [
                "case",
                "position",
                "time",
            ],
            "complete_case_aggregation": True,
            "verification_artifacts_checked": True,
        },
        "parameter_matching": {
            "minimum": int(frame["parameter_count"].min()),
            "maximum": int(frame["parameter_count"].max()),
            "maximum_to_minimum_ratio": parameter_ratio,
        },
        "p1_corrected_reference": {
            "field_relative_l2_three_seed_mean": corrected_field,
            "spatial_gradient": baseline_gradient,
        },
        "validation_selected_model": {
            "model_family": best_validation_row["model_family"],
            "weighted_objective": float(
                best_validation_row["best_validation_weighted_objective"]
            ),
            "selection_rule": (
                "Lowest frozen validation weighted objective; test metrics are "
                "not used for the next training configuration."
            ),
        },
        "descriptive_test_best": {
            "model_family": best_test_row["model_family"],
            "field_relative_l2_mean": float(
                best_test_row["test_field_relative_l2_mean"]
            ),
            "ratio_to_corrected_reference": float(
                best_test_row["test_field_relative_l2_mean"] / corrected_field
            ),
        },
        "pilot_gate_checks": gate_checks,
        "pilot_gate_passed": all(gate_checks.values()),
        "required_fixes": [
            "Continue factorized and causal training because their validation optima occur at or near the final pilot epoch.",
            "Freeze the extended-run configuration before reading its test metrics.",
            "Use summed history epoch time for resumed pilots; final-session wall time undercounts interrupted sessions.",
            "Do not advance to P3 unless the full P2 temperature and spatial-gradient checks pass.",
        ],
        "caveats": [
            "These are one-seed pilots and do not support uncertainty or superiority claims.",
            "The forced execution-window interruptions were resumed from atomic checkpoints.",
            "The fixed trainer evaluated the test split, but further configuration choices are based only on validation history.",
        ],
    }
    (output_dir / "p2_pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the frozen P2 parameter-matched pilots."
    )
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_ids = tuple(args.run_ids or DEFAULT_RUN_IDS)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "runs": [
                        {
                            "run_id": run_id,
                            "exists": (
                                project_root / "outputs" / "runs" / run_id
                            ).is_dir(),
                        }
                        for run_id in run_ids
                    ]
                },
                indent=2,
            )
        )
        return 0
    summary = aggregate(project_root, run_ids)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
