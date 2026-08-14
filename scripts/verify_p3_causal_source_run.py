"""Independently verify and publish a compact causal-source run summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cdcureno.models.joint_operators import CausalFactorizedOperator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT / "outputs" / "runs" / "p3-source-causal-v1-seed0-fix1"
)
DEFAULT_SUMMARY = (
    ROOT / "outputs" / "tables" / "p3_causal_source_v1_seed0.json"
)
DEFAULT_HISTORY = (
    ROOT / "outputs" / "tables" / "p3_causal_source_v1_seed0_history.parquet"
)
DEFAULT_CASES = (
    ROOT / "outputs" / "tables" / "p3_causal_source_v1_seed0_cases.parquet"
)
EXPECTED_ACCEPTANCE = {
    "in_family_temperature_relative_l2_mean_max": 0.005,
    "held_out_temperature_relative_l2_mean_max": 0.01,
    "held_out_each_family_temperature_relative_l2_mean_max": 0.0125,
    "held_out_alpha_relative_l2_mean_max": 0.12,
    "held_out_temperature_linf_K_max": 20.0,
    "alpha_bound_violation_count_max": 0,
    "alpha_monotonicity_violation_count_max": 0,
}
EXPECTED_SPLIT_COUNTS = {
    "in_family_test": 20,
    "smart_cure": 100,
    "three_hold": 100,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _case_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "case_count": int(len(frame)),
        "temperature_relative_l2_mean": float(
            frame["temperature_relative_l2"].mean()
        ),
        "temperature_relative_l2_median": float(
            frame["temperature_relative_l2"].median()
        ),
        "temperature_mae_K_mean": float(frame["temperature_mae_K"].mean()),
        "temperature_linf_K_max": float(frame["temperature_linf_K"].max()),
        "alpha_relative_l2_mean": float(
            frame["alpha_relative_l2"].mean()
        ),
        "alpha_mae_mean": float(frame["alpha_mae"].mean()),
        "alpha_bound_violation_count": int(
            frame["alpha_bound_violation_count"].sum()
        ),
        "alpha_monotonicity_violation_count": int(
            frame["alpha_monotonicity_violation_count"].sum()
        ),
    }


def _case_evidence(
    cases: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, float | int],
    dict[str, bool],
]:
    required_columns = {
        "split",
        "case_id",
        "family_id",
        "temperature_relative_l2",
        "temperature_mae_K",
        "temperature_linf_K",
        "alpha_relative_l2",
        "alpha_mae",
        "alpha_bound_violation_count",
        "alpha_monotonicity_violation_count",
    }
    if set(cases.columns) != required_columns:
        raise ValueError("Per-case metric columns differ from the frozen schema.")
    if cases[["split", "case_id"]].duplicated().any():
        raise ValueError("Per-case metrics contain duplicate split/case rows.")
    numeric = cases.drop(columns=["split"])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Per-case metrics contain non-finite values.")
    if (
        (cases["temperature_relative_l2"] < 0.0).any()
        or (cases["temperature_mae_K"] < 0.0).any()
        or (cases["temperature_linf_K"] < 0.0).any()
        or (cases["alpha_relative_l2"] < 0.0).any()
        or (cases["alpha_mae"] < 0.0).any()
        or (cases["alpha_bound_violation_count"] < 0).any()
        or (cases["alpha_monotonicity_violation_count"] < 0).any()
    ):
        raise ValueError("Per-case metrics contain impossible negative values.")
    split_names = set(str(value) for value in cases["split"].unique())
    if split_names != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError("Per-case metrics contain unexpected split names.")
    summaries = {
        split: _case_summary(cases.loc[cases["split"] == split])
        for split in EXPECTED_SPLIT_COUNTS
    }
    held_out_names = ("smart_cure", "three_hold")
    held_out_count = sum(
        int(summaries[name]["case_count"]) for name in held_out_names
    )
    held_out = {
        "case_count": held_out_count,
        "temperature_relative_l2_mean": sum(
            float(summaries[name]["temperature_relative_l2_mean"])
            * int(summaries[name]["case_count"])
            for name in held_out_names
        )
        / held_out_count,
        "temperature_linf_K_max": max(
            float(summaries[name]["temperature_linf_K_max"])
            for name in held_out_names
        ),
        "alpha_relative_l2_mean": sum(
            float(summaries[name]["alpha_relative_l2_mean"])
            * int(summaries[name]["case_count"])
            for name in held_out_names
        )
        / held_out_count,
        "alpha_bound_violation_count": sum(
            int(summaries[name]["alpha_bound_violation_count"])
            for name in held_out_names
        ),
        "alpha_monotonicity_violation_count": sum(
            int(summaries[name]["alpha_monotonicity_violation_count"])
            for name in held_out_names
        ),
    }
    checks = {
        "in_family_temperature": (
            float(
                summaries["in_family_test"][
                    "temperature_relative_l2_mean"
                ]
            )
            <= EXPECTED_ACCEPTANCE[
                "in_family_temperature_relative_l2_mean_max"
            ]
        ),
        "held_out_temperature": (
            float(held_out["temperature_relative_l2_mean"])
            <= EXPECTED_ACCEPTANCE[
                "held_out_temperature_relative_l2_mean_max"
            ]
        ),
        "held_out_each_family_temperature": all(
            float(summaries[name]["temperature_relative_l2_mean"])
            <= EXPECTED_ACCEPTANCE[
                "held_out_each_family_temperature_relative_l2_mean_max"
            ]
            for name in held_out_names
        ),
        "held_out_alpha": (
            float(held_out["alpha_relative_l2_mean"])
            <= EXPECTED_ACCEPTANCE[
                "held_out_alpha_relative_l2_mean_max"
            ]
        ),
        "held_out_linf": (
            float(held_out["temperature_linf_K_max"])
            <= EXPECTED_ACCEPTANCE["held_out_temperature_linf_K_max"]
        ),
        "alpha_bounds": (
            int(held_out["alpha_bound_violation_count"])
            <= EXPECTED_ACCEPTANCE["alpha_bound_violation_count_max"]
        ),
        "alpha_monotonicity": (
            int(held_out["alpha_monotonicity_violation_count"])
            <= EXPECTED_ACCEPTANCE[
                "alpha_monotonicity_violation_count_max"
            ]
        ),
    }
    return summaries, held_out, checks


def _mapping_matches(
    recorded: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    if set(recorded) != set(expected):
        return False
    for name, expected_value in expected.items():
        recorded_value = recorded[name]
        if isinstance(expected_value, dict):
            if not isinstance(recorded_value, dict) or not _mapping_matches(
                recorded_value, expected_value
            ):
                return False
        elif isinstance(expected_value, bool):
            if recorded_value is not expected_value:
                return False
        elif isinstance(expected_value, int):
            if (
                isinstance(recorded_value, bool)
                or int(recorded_value) != expected_value
            ):
                return False
        elif not np.isclose(
            float(recorded_value),
            float(expected_value),
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            return False
    return True


def _causality_evidence(causality: dict[str, Any]) -> dict[str, Any]:
    rows = causality.get("rows")
    if not isinstance(rows, list) or len(rows) != 4 * 3 * 4:
        raise ValueError("Causality evidence has incomplete case/cutoff/output rows.")
    tolerance = float(causality["tolerance"])
    outputs = {"temperature", "temperature_residual", "alpha", "cure_rate"}
    identities = {
        (
            int(row["case_id"]),
            int(row["cutoff_index"]),
            str(row["output"]),
        )
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("Causality evidence contains duplicate rows.")
    if {str(row["output"]) for row in rows} != outputs:
        raise ValueError("Causality evidence has unexpected output names.")
    row_checks = [
        (
            int(row["future_start_index"]) == int(row["cutoff_index"]) + 1
            and float(row["maximum_prefix_abs_difference"]) <= tolerance
            and bool(row["passed"])
        )
        for row in rows
    ]
    maximum_abs = max(
        float(row["maximum_prefix_abs_difference"]) for row in rows
    )
    maximum_relative = max(float(row["prefix_relative_l2"]) for row in rows)
    return {
        "row_count": len(rows),
        "maximum_prefix_abs_difference": maximum_abs,
        "maximum_prefix_relative_l2": maximum_relative,
        "rows_pass": bool(all(row_checks)),
        "aggregate_matches": bool(
            np.isclose(
                maximum_abs,
                float(causality["maximum_prefix_abs_difference"]),
                rtol=0.0,
                atol=0.0,
            )
            and np.isclose(
                maximum_relative,
                float(causality["maximum_prefix_relative_l2"]),
                rtol=0.0,
                atol=0.0,
            )
        ),
    }


def verify_run(
    run_dir: Path,
    *,
    summary_path: Path,
    history_path: Path,
    cases_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    required = {
        "metrics": run_dir / "metrics.json",
        "history": run_dir / "history.parquet",
        "cases": run_dir / "metrics_per_case.parquet",
        "causality": run_dir / "causality.json",
        "best": run_dir / "checkpoints" / "best.pt",
        "last": run_dir / "checkpoints" / "last.pt",
        "done": run_dir / "DONE",
        "status": run_dir / "STATUS.json",
        "config": run_dir / "config_resolved.json",
        "environment": run_dir / "environment.json",
        "checksums": run_dir / "data_checksums.json",
        "git_state": run_dir / "git_state.txt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Causal-source run is incomplete: {missing}")
    if (run_dir / "FAILED").exists():
        raise ValueError("Completed causal-source run retains a FAILED marker.")

    metrics = _read_json(required["metrics"])
    status = _read_json(required["status"])
    causality = _read_json(required["causality"])
    history = pd.read_parquet(required["history"])
    cases = pd.read_parquet(required["cases"])
    if history.empty or cases.empty:
        raise ValueError("Causal-source history or per-case metrics are empty.")
    split_summaries, held_out, acceptance_checks = _case_evidence(cases)
    causality_evidence = _causality_evidence(causality)

    epochs = history["epoch"].astype(int).tolist()
    selected_index = int(history["validation_weighted_objective"].idxmin())
    independently_selected_epoch = int(history.loc[selected_index, "epoch"])
    independently_best_validation = float(
        history.loc[selected_index, "validation_weighted_objective"]
    )
    recorded_selected_epoch = int(metrics["selection"]["selected_epoch"])
    recorded_best_validation = float(
        metrics["selection"]["best_validation_objective"]
    )

    best = torch.load(required["best"], map_location="cpu", weights_only=False)
    last = torch.load(required["last"], map_location="cpu", weights_only=False)
    state_parameter_count = sum(
        int(tensor.numel()) for tensor in best["model"].values()
    )
    model_spec = metrics["model"]
    reconstructed = CausalFactorizedOperator(
        input_channels=int(model_spec["input_channels"]),
        width=int(model_spec["width"]),
        depth=int(model_spec["depth"]),
        modes_space=int(model_spec["modes_space"]),
    )
    reconstructed.load_state_dict(best["model"], strict=True)
    model_tensors_finite = all(
        bool(torch.isfinite(tensor).all()) for tensor in best["model"].values()
    )
    split_counts = {
        str(name): int(count)
        for name, count in cases.groupby("split").size().items()
    }
    checksums = _read_json(required["checksums"])
    actual_input_hashes = {
        name: _sha256(Path(record["path"]))
        for name, record in checksums.items()
    }
    declared_input_hashes = {
        name: str(record["sha256"]) for name, record in checksums.items()
    }
    checks = {
        "metrics_status_completed": (
            metrics.get("status") == "completed"
            and status.get("status") == "completed"
        ),
        "run_passed": metrics.get("passed") is True,
        "acceptance_contract_matches": (
            metrics.get("acceptance_thresholds_prespecified")
            == EXPECTED_ACCEPTANCE
        ),
        "history_contiguous": epochs == list(range(1, len(epochs) + 1)),
        "selected_epoch_is_validation_minimum": (
            independently_selected_epoch == recorded_selected_epoch
            and np.isclose(
                independently_best_validation,
                recorded_best_validation,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "best_checkpoint_epoch_matches": (
            int(best["epoch"]) == recorded_selected_epoch
            and int(best["epoch"]) <= int(history["epoch"].max())
            and np.isclose(
                float(best["best_validation"]),
                independently_best_validation,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "last_checkpoint_epoch_matches": (
            int(last["epoch"]) == int(history["epoch"].max())
        ),
        "checkpoint_selection_is_validation_only": (
            best.get("selection_split") == "validation"
            and best.get("held_out_labels_used_for_selection") is False
            and metrics["selection"]["held_out_labels_used"] is False
        ),
        "checkpoint_hashes_match_metrics": (
            _sha256(required["best"])
            == metrics["checkpoints"]["best"]["sha256"]
            and _sha256(required["last"])
            == metrics["checkpoints"]["last"]["sha256"]
        ),
        "parameter_count_matches": (
            state_parameter_count == int(metrics["model"]["parameter_count"])
        ),
        "best_model_is_strict_and_finite": model_tensors_finite,
        "case_counts_match": (
            split_counts == EXPECTED_SPLIT_COUNTS
        ),
        "split_metrics_recomputed": _mapping_matches(
            metrics["split_metrics"], split_summaries
        ),
        "held_out_metrics_recomputed": _mapping_matches(
            metrics["held_out_combined"], held_out
        ),
        "acceptance_checks_recomputed": (
            metrics["acceptance_checks"] == acceptance_checks
            and all(acceptance_checks.values())
        ),
        "causality_rows_complete": causality_evidence["rows_pass"],
        "causality_aggregates_recomputed": causality_evidence[
            "aggregate_matches"
        ],
        "causality_exact": (
            causality.get("passed") is True
            and float(causality["maximum_prefix_abs_difference"]) == 0.0
            and float(causality["maximum_prefix_relative_l2"]) == 0.0
            and metrics["causality"] == {
                name: value
                for name, value in causality.items()
                if name != "rows"
            }
        ),
        "status_flags_derived": (
            metrics.get("field_acceptance_passed")
            is all(acceptance_checks.values())
            and metrics.get("causality_passed") is bool(causality["passed"])
            and metrics.get("passed")
            is bool(all(acceptance_checks.values()) and causality["passed"])
        ),
        "input_checksums_match_files": (
            checksums == metrics["input_checksums"]
            and actual_input_hashes == declared_input_hashes
        ),
        "git_sha_bound": (
            best.get("git_sha") == metrics.get("git_sha")
            and str(metrics.get("git_sha")) in required[
                "git_state"
            ].read_text(encoding="utf-8")
        ),
    }
    passed = bool(all(checks.values()))
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise ValueError(f"Causal-source verification failed: {failed}")

    summary = {
        "schema_version": 1,
        "phase": "P5",
        "gate": "structurally_causal_source_pretraining",
        "run_id": metrics["run_id"],
        "source_phase": metrics["phase"],
        "experiment": metrics["experiment"],
        "git_sha": metrics["git_sha"],
        "model": metrics["model"],
        "selection": metrics["selection"],
        "training": metrics["training"],
        "split_metrics": metrics["split_metrics"],
        "held_out_combined": metrics["held_out_combined"],
        "acceptance_thresholds_prespecified": metrics[
            "acceptance_thresholds_prespecified"
        ],
        "acceptance_checks": metrics["acceptance_checks"],
        "causality": metrics["causality"],
        "checkpoints": metrics["checkpoints"],
        "input_checksums": metrics["input_checksums"],
        "normalization": metrics["normalization"],
        "resource": {
            "peak_accelerator_memory_bytes": metrics[
                "peak_accelerator_memory_bytes"
            ],
            "peak_python_tracemalloc_bytes": metrics[
                "peak_python_tracemalloc_bytes"
            ],
        },
        "independent_verification": {
            "checks": checks,
            "selected_epoch": independently_selected_epoch,
            "best_validation_objective": independently_best_validation,
            "state_parameter_count": state_parameter_count,
            "history_row_count": len(history),
            "per_case_row_count": len(cases),
            "per_case_split_counts": split_counts,
            "split_metrics_recomputed": split_summaries,
            "held_out_combined_recomputed": held_out,
            "acceptance_checks_recomputed": acceptance_checks,
            "causality_recomputed": causality_evidence,
        },
        "published_tables": {
            "history": history_path.relative_to(ROOT).as_posix(),
            "cases": cases_path.relative_to(ROOT).as_posix(),
        },
        "dry_run": bool(dry_run),
        "passed": passed,
    }
    summary["independent_verification"]["checks"] = {
        name: bool(value) for name, value in checks.items()
    }
    if dry_run:
        return summary
    _atomic_parquet(history_path, history)
    _atomic_parquet(cases_path, cases)
    summary["published_tables"]["history_sha256"] = _sha256(history_path)
    summary["published_tables"]["cases_sha256"] = _sha256(cases_path)
    _atomic_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a completed causal-source run and publish compact, "
            "Git-trackable evidence."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = verify_run(
        args.run_dir.resolve(),
        summary_path=args.summary.resolve(),
        history_path=args.history.resolve(),
        cases_path=args.cases.resolve(),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
