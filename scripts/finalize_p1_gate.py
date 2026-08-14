"""Freeze corrected-alpha evidence and evaluate the complete P1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine the P1 temperature-field and corrected-alpha evidence."
    )
    parser.add_argument(
        "--alpha-run-id",
        default="legacy_resfno_corrected-A-x35-seed1-e300-20260723-164358",
    )
    parser.add_argument(
        "--field-summary",
        type=Path,
        default=Path("outputs/tables/p1_field_summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/tables")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve()
    field_path = (project_root / args.field_summary).resolve()
    alpha_run_dir = project_root / "outputs" / "runs" / args.alpha_run_id
    metrics_path = alpha_run_dir / "metrics.json"
    verification_path = alpha_run_dir / "verification.json"
    required = [field_path, metrics_path, verification_path]
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

    field = json.loads(field_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    test_values = np.asarray(list(metrics["test"].values()), dtype=float)
    alpha_checks = {
        "corrected_experiment": metrics["experiment"] == "legacy_resfno_corrected",
        "alpha_task": metrics["task"] == "A",
        "complete_case_test_count": metrics["test"]["case_count"] == 125,
        "train_only_input_normalization": (
            metrics["normalization"]["normalization_scope"] == "train_cases_only"
        ),
        "no_temperature_output_normalizer": metrics["normalization"]["y"] is None,
        "finite_test_metrics": bool(np.isfinite(test_values).all()),
        "independent_verification_passed": verification["passed"] is True,
    }
    alpha_summary = {
        "status": (
            "P1_corrected_alpha_path_passed"
            if all(alpha_checks.values())
            else "P1_corrected_alpha_path_failed"
        ),
        "run_id": args.alpha_run_id,
        "location": metrics["location"],
        "seed": metrics["seed"],
        "selected_epoch": metrics["selected_epoch"],
        "parameter_count": metrics["parameter_count"],
        "normalization": metrics["normalization"],
        "test": metrics["test"],
        "verification": verification,
        "gate_evidence": alpha_checks,
        "limitations": [
            "This is a one-location execution check, not a full alpha-field benchmark.",
            "The exact legacy alpha path remains a documented crashing behavior.",
        ],
        "source_metrics": str(metrics_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "source_verification": str(
            verification_path.relative_to(project_root)
        ).replace("\\", "/"),
    }
    alpha_output = output_dir / "p1_corrected_alpha_x35_summary.json"
    alpha_output.write_text(
        json.dumps(alpha_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gate_checks = {
        "temperature_field_gate_passed": (
            field["status"] == "P1_temperature_field_gate_passed"
        ),
        "corrected_alpha_path_passed": (
            alpha_summary["status"] == "P1_corrected_alpha_path_passed"
        ),
    }
    gate = {
        "status": "P1_passed" if all(gate_checks.values()) else "P1_failed",
        "gate_evidence": gate_checks,
        "temperature_field_summary": str(
            field_path.relative_to(project_root)
        ).replace("\\", "/"),
        "corrected_alpha_summary": str(alpha_output.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "scope_boundary": (
            "P1 establishes auditable exact/corrected legacy baselines. "
            "It does not establish a spatially coupled or genuine 2-D operator."
        ),
    }
    gate_output = output_dir / "p1_gate_summary.json"
    gate_output.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["status"] == "P1_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
