"""Aggregate completed P1 runs into small, tracked audit tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MATCHED_MANIFEST = "p1_matched_holdout_v1"


def load_run_row(project_root: Path, row: pd.Series) -> dict:
    run_dir = project_root / "outputs" / "runs" / row["run_id"]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    matched_path = run_dir / "evaluations" / MATCHED_MANIFEST / "metrics.json"
    matched = (
        json.loads(matched_path.read_text(encoding="utf-8"))["metrics"]
        if matched_path.is_file()
        else None
    )
    return {
        "run_id": row["run_id"],
        "experiment": row["experiment"],
        "training_split": row["split"],
        "seed": int(row["seed"]),
        "location": int(metrics["location"]),
        "selected_epoch": int(metrics["selected_epoch"]),
        "selection": metrics["selection"],
        "normalization_scope": metrics["normalization"]["normalization_scope"],
        "native_test_case_count": int(metrics["test"]["case_count"]),
        "native_relative_l2_mean": float(metrics["test"]["relative_l2_mean"]),
        "native_mae_mean": float(metrics["test"]["mae_mean"]),
        "native_linf_max": float(metrics["test"]["linf_max"]),
        "matched_case_count": int(matched["case_count"]) if matched else None,
        "matched_relative_l2_mean": (
            float(matched["relative_l2_mean"]) if matched else None
        ),
        "matched_mae_mean": float(matched["mae_mean"]) if matched else None,
        "matched_linf_max": float(matched["linf_max"]) if matched else None,
    }


def aggregate(project_root: Path, index_path: Path, output_dir: Path) -> dict:
    index = pd.read_csv(index_path)
    selected = index[
        (index["phase"] == "P1")
        & (index["status"] == "completed")
        & (index["model"].str.endswith("_x35"))
    ]
    runs = pd.DataFrame([load_run_row(project_root, row) for _, row in selected.iterrows()])
    runs = runs.sort_values(["experiment", "training_split", "seed"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "p1_one_location_runs.csv", index=False)

    exact = runs[
        (runs["experiment"] == "legacy_resfno_exact")
        & (runs["training_split"] == "legacy_case1_exact_v1")
    ][["seed", "matched_relative_l2_mean"]].rename(
        columns={"matched_relative_l2_mean": "exact_relative_l2"}
    )
    corrected = runs[
        (runs["experiment"] == "legacy_resfno_corrected")
        & (runs["training_split"] == "legacy_case1_corrected_matched_v1")
    ][["seed", "matched_relative_l2_mean"]].rename(
        columns={"matched_relative_l2_mean": "corrected_relative_l2"}
    )
    paired = exact.merge(corrected, on="seed", validate="one_to_one")
    paired["corrected_minus_exact"] = (
        paired["corrected_relative_l2"] - paired["exact_relative_l2"]
    )
    paired["corrected_relative_improvement_percent"] = (
        (paired["exact_relative_l2"] - paired["corrected_relative_l2"])
        / paired["exact_relative_l2"]
        * 100.0
    )
    paired.to_csv(output_dir / "p1_exact_vs_corrected_paired.csv", index=False)

    def describe(values: pd.Series) -> dict:
        array = values.dropna().to_numpy(dtype=float)
        return {
            "n_seeds": int(len(array)),
            "mean": float(np.mean(array)),
            "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else None,
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }

    upstream_path = project_root / "outputs" / "audit" / "upstream_checkpoint_x35_matched.json"
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    summary = {
        "status": "descriptive_P1_only_not_confirmatory",
        "evaluation_manifest": "splits/p1_matched_holdout_v1.json",
        "case_count": 125,
        "exact": describe(paired["exact_relative_l2"]),
        "corrected_matched": describe(paired["corrected_relative_l2"]),
        "paired_corrected_relative_improvement_percent": describe(
            paired["corrected_relative_improvement_percent"]
        ),
        "upstream_checkpoint_relative_l2_mean": upstream["metrics"][
            "relative_l2_mean"
        ],
        "interpretation": (
            "The corrected pipeline did not materially outperform exact legacy "
            "behavior at x=35 over three paired seeds. Preserve both baselines; "
            "do not generalize this one-location result to the full field."
        ),
    }
    (output_dir / "p1_one_location_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate verified P1 run metrics from RESULTS_INDEX.csv."
    )
    parser.add_argument("--index", type=Path, default=Path("RESULTS_INDEX.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    index = (project_root / args.index).resolve()
    output = (project_root / args.output_dir).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "index": str(index),
                    "index_exists": index.is_file(),
                    "output_dir": str(output),
                },
                indent=2,
            )
        )
        return 0
    summary = aggregate(project_root, index, output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
