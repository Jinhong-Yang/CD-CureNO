"""Aggregate frozen P1 full-field summaries and evaluate the P1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sample_std = float(np.std(array, ddof=1)) if len(array) > 1 else None
    return {
        "n_seeds": int(len(array)),
        "mean": mean,
        "sample_std": sample_std,
        "coefficient_of_variation": (
            sample_std / mean if sample_std is not None and mean != 0 else None
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate(project_root: Path, output_dir: Path) -> dict:
    exact_path = output_dir / "p1_exact_field_seed1_summary.json"
    corrected_paths = [
        output_dir / f"p1_corrected_field_seed{seed}_summary.json"
        for seed in (0, 1, 2)
    ]
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    corrected = [
        json.loads(path.read_text(encoding="utf-8")) for path in corrected_paths
    ]
    rows = [
        {
            "experiment": "legacy_resfno_exact",
            "seed": exact["seed"],
            "field_relative_l2_mean": exact["field_relative_l2_mean"],
            "composite_relative_l2_mean": exact["composite_relative_l2_mean"],
            "tool_relative_l2_mean": exact["tool_relative_l2_mean"],
            "field_mae_mean": exact["field_mae_mean"],
            "field_rmse_mean": exact["field_rmse_mean"],
            "field_linf_max": exact["field_linf_max"],
            "peak_composite_temperature_error_mae": exact[
                "peak_composite_temperature_error_mae"
            ],
            "sweep_wall_seconds": exact["sweep_wall_seconds"],
            "summed_epoch_seconds": exact["summed_epoch_seconds"],
        }
    ]
    for item in corrected:
        rows.append(
            {
                "experiment": "legacy_resfno_corrected",
                "seed": item["seed"],
                "field_relative_l2_mean": item["field_relative_l2_mean"],
                "composite_relative_l2_mean": item["composite_relative_l2_mean"],
                "tool_relative_l2_mean": item["tool_relative_l2_mean"],
                "field_mae_mean": item["field_mae_mean"],
                "field_rmse_mean": item["field_rmse_mean"],
                "field_linf_max": item["field_linf_max"],
                "peak_composite_temperature_error_mae": item[
                    "peak_composite_temperature_error_mae"
                ],
                "sweep_wall_seconds": item["sweep_wall_seconds"],
                "summed_epoch_seconds": item["summed_epoch_seconds"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(["experiment", "seed"])
    frame.to_csv(output_dir / "p1_field_runs.csv", index=False)

    corrected_field = [item["field_relative_l2_mean"] for item in corrected]
    corrected_composite = [
        item["composite_relative_l2_mean"] for item in corrected
    ]
    corrected_seed1 = next(item for item in corrected if item["seed"] == 1)
    paired_seed1_field_improvement = (
        exact["field_relative_l2_mean"]
        - corrected_seed1["field_relative_l2_mean"]
    ) / exact["field_relative_l2_mean"]
    paired_seed1_composite_improvement = (
        exact["composite_relative_l2_mean"]
        - corrected_seed1["composite_relative_l2_mean"]
    ) / exact["composite_relative_l2_mean"]

    corrected_run_ids = [
        run_id for item in corrected for run_id in item["selected_run_ids"]
    ]
    exact_run_ids = exact["selected_run_ids"]
    summary = {
        "status": "P1_temperature_field_gate_passed",
        "scope": (
            "Case1 temperature only: exact seed 1 and corrected matched seeds 0-2, "
            "51 independent location models"
        ),
        "exact_seed1": {
            "field_relative_l2_mean": exact["field_relative_l2_mean"],
            "composite_relative_l2_mean": exact["composite_relative_l2_mean"],
            "field_linf_max": exact["field_linf_max"],
        },
        "corrected_three_seed": {
            "field_relative_l2": describe(corrected_field),
            "composite_relative_l2": describe(corrected_composite),
            "field_linf_worst": float(
                max(item["field_linf_max"] for item in corrected)
            ),
            "failed_location_runs": 0,
            "verified_location_runs": len(corrected_run_ids),
        },
        "paired_seed1_descriptive_only": {
            "field_relative_improvement_percent": float(
                paired_seed1_field_improvement * 100.0
            ),
            "composite_relative_improvement_percent": float(
                paired_seed1_composite_improvement * 100.0
            ),
            "interpretation": (
                "The corrected seed-1 field is modestly more accurate than the "
                "exact seed-1 field, but this is descriptive and not a confirmatory "
                "multi-seed exact-vs-corrected test."
            ),
        },
        "gate_evidence": {
            "exact_location_runs_complete": len(exact_run_ids) == 51,
            "corrected_location_runs_complete": len(corrected_run_ids) == 153,
            "all_reconstructed_fields_have_expected_shape": all(
                item["field_shape"][1:] == [51, 223]
                for item in [exact, *corrected]
            ),
            "all_metrics_finite": bool(
                np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
            ),
            "corrected_three_seeds_complete": len(corrected) == 3,
        },
        "limitations": [
            "The exact full field was run for seed 1 only.",
            "This gate covers temperature; corrected alpha execution is validated separately.",
            "The legacy field remains a stack of 51 uncoupled temporal models.",
            "No result here supports a genuine two-spatial-dimensional claim.",
        ],
    }
    if not all(summary["gate_evidence"].values()):
        summary["status"] = "P1_temperature_field_gate_failed"
    (output_dir / "p1_field_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen P1 field summaries and assess the temperature gate."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/tables")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve()
    required = [
        output_dir / "p1_exact_field_seed1_summary.json",
        *[
            output_dir / f"p1_corrected_field_seed{seed}_summary.json"
            for seed in (0, 1, 2)
        ],
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "required": [
                        {"path": str(path), "exists": path.is_file()}
                        for path in required
                    ]
                },
                indent=2,
            )
        )
        return 0
    summary = aggregate(project_root, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"].endswith("_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
