"""Aggregate frozen P2 extended runs and evaluate the seed-0 scientific gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RUN_IDS = (
    "p2extended200-factorized-w40d4-seed0-bbdc829",
    "p2extended200-causal-w40d8-seed0-8ae6702",
)


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
                "epochs_completed": len(history),
                "selected_epoch": metrics["selected_epoch"],
                "parameter_count": metrics["parameter_count"],
                "summed_epoch_seconds": metrics["summed_epoch_seconds"],
                "final_session_wall_seconds": metrics[
                    "final_session_wall_seconds"
                ],
                "recorded_session_count": metrics["recorded_session_count"],
                "peak_process_rss_bytes": metrics["peak_process_rss_bytes"],
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
                "test_composite_relative_l2_mean": metrics["test"][
                    "composite_relative_l2_mean"
                ],
                "test_alpha_relative_l2_mean": metrics["test"][
                    "alpha_relative_l2_mean"
                ],
                "test_spatial_gradient_relative_l2_mean": metrics["test"][
                    "spatial_gradient_relative_l2_mean"
                ],
                "test_spatial_gradient_mae_mean": metrics["test"][
                    "spatial_gradient_mae_mean"
                ],
                "test_field_linf_max": metrics["test"]["field_linf_max"],
                "alpha_monotonic_violation_count": metrics["test"][
                    "alpha_monotonic_violation_count"
                ],
                "alpha_bound_violation_count": metrics["test"][
                    "alpha_bound_violation_count"
                ],
                "verification_passed": verification["passed"],
                "physical_constraints_passed": all(
                    verification["physical_constraints"].values()
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("model_family").reset_index(drop=True)
    output_dir = project_root / "outputs" / "tables"
    frame.to_csv(output_dir / "p2_extended_seed0_runs.csv", index=False)

    pilot = json.loads(
        (output_dir / "p2_pilot_summary.json").read_text(encoding="utf-8")
    )
    corrected_field = pilot["p1_corrected_reference"][
        "field_relative_l2_three_seed_mean"
    ]
    corrected_gradient = pilot["p1_corrected_reference"]["spatial_gradient"][
        "spatial_gradient_relative_l2_mean"
    ]
    validation_selected = frame.loc[
        frame["best_validation_weighted_objective"].idxmin()
    ]
    qualifying = frame[
        (frame["test_field_relative_l2_mean"] <= corrected_field)
        & (
            frame["test_spatial_gradient_relative_l2_mean"]
            <= corrected_gradient
        )
    ]
    causal_row = frame[frame["model_family"] == "causal_factorized"]
    if len(causal_row) != 1:
        raise ValueError("Exactly one causal extended run is required.")
    causal_run_id = str(causal_row.iloc[0]["run_id"])
    causal_path = output_dir / f"{causal_run_id}_causality.json"
    causality = json.loads(causal_path.read_text(encoding="utf-8"))
    gate_checks = {
        "all_runs_independently_verified": bool(frame["verification_passed"].all()),
        "all_physical_constraints_passed": bool(
            frame["physical_constraints_passed"].all()
        ),
        "at_least_one_model_matches_temperature_and_gradient": not qualifying.empty,
        "causal_checkpoint_passes_future_perturbation": causality["passed"],
        "parameter_count_reported": bool((frame["parameter_count"] > 0).all()),
        "epoch_time_reported": bool((frame["summed_epoch_seconds"] > 0).all()),
        "peak_memory_reported": bool(
            frame["peak_process_rss_bytes"].notna().all()
        ),
    }
    passed = all(gate_checks.values())
    summary = {
        "status": "P2_seed0_gate_passed" if passed else "P2_seed0_gate_failed",
        "validation_assessment": (
            "Share with caveats" if passed else "Needs revision"
        ),
        "scope": (
            "Frozen seed-0 extended factorized and causal Case1 joint runs; "
            "P1 reference is the corrected three-seed mean."
        ),
        "selection_policy": {
            "validation_selected_model": str(
                validation_selected["model_family"]
            ),
            "validation_weighted_objective": float(
                validation_selected["best_validation_weighted_objective"]
            ),
            "test_used_for_configuration": False,
        },
        "p1_corrected_reference": {
            "field_relative_l2_mean": corrected_field,
            "spatial_gradient_relative_l2_mean": corrected_gradient,
        },
        "qualifying_models": qualifying["model_family"].tolist(),
        "gate_checks": gate_checks,
        "causality_artifact": str(causal_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "cost_note": (
            "The two extended CPU runs overlapped in wall-clock time on a "
            "24-logical-CPU host. Per-run summed epoch time and peak process RSS "
            "are reported; concurrent wall time is not used for a speed claim."
        ),
        "limitations": [
            "This is a seed-0 scientific gate, not a multi-seed confirmatory comparison.",
            "The public Case1 field is 1+1-D and cannot establish true two-spatial-dimensional performance.",
            "P3 may begin only if this gate passes; P6 still requires at least five seeds for main comparisons.",
        ],
    }
    (output_dir / "p2_extended_seed0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen P2 extended runs and evaluate the seed-0 gate."
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
                            "done": (
                                project_root
                                / "outputs"
                                / "runs"
                                / run_id
                                / "DONE"
                            ).is_file(),
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
    return 0 if summary["status"] == "P2_seed0_gate_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
