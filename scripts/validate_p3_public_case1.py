"""Validate the conservative P3 solver against immutable public Case1 arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from cdcureno.solvers import simulate_cure_1d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "resfno" / "Case1.mat"
DEFAULT_SUMMARY = ROOT / "outputs" / "tables" / "p3_public_case1_validation.json"
DEFAULT_CASES = ROOT / "outputs" / "tables" / "p3_public_case1_cases.csv"
ACCEPTANCE = {
    "temperature_field_relative_l2_mean_max": 5.0e-3,
    "temperature_field_absolute_error_K_max": 10.0,
    "alpha_field_relative_l2_mean_max": 5.0e-2,
    "alpha_bound_violation_count_max": 0,
    "alpha_monotonicity_violation_count_max": 0,
    "maximum_abs_energy_residual_W_per_m3_max": 1.0e-4,
    "maximum_relative_global_energy_residual_max": 1.0e-8,
}


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def _run_case(
    task: tuple[int, np.ndarray, np.ndarray, np.ndarray, float],
) -> dict[str, float | int]:
    case_id, air, target_temperature, target_alpha, maximum_step_s = task
    times_s = np.arange(air.size, dtype=np.float64) * 60.0
    started = time.perf_counter()
    result = simulate_cure_1d(
        times_s,
        air,
        initial_alpha=float(target_alpha[0, 21]),
        maximum_step_s=maximum_step_s,
        maximum_coupling_iterations=16,
    )
    elapsed = time.perf_counter() - started
    prediction_temperature = result.temperature_K
    prediction_alpha = result.alpha
    error_temperature = prediction_temperature - target_temperature
    error_alpha = prediction_alpha - target_alpha
    alpha_differences = np.diff(prediction_alpha[:, 21:], axis=0)
    return {
        "case_id": case_id,
        "temperature_relative_l2": _relative_l2(
            prediction_temperature, target_temperature
        ),
        "tool_temperature_relative_l2": _relative_l2(
            prediction_temperature[:, :21], target_temperature[:, :21]
        ),
        "composite_temperature_relative_l2": _relative_l2(
            prediction_temperature[:, 21:], target_temperature[:, 21:]
        ),
        "temperature_mae_K": float(np.mean(np.abs(error_temperature))),
        "temperature_max_abs_K": float(np.max(np.abs(error_temperature))),
        "peak_composite_temperature_error_K": float(
            np.max(prediction_temperature[:, 21:])
            - np.max(target_temperature[:, 21:])
        ),
        "alpha_relative_l2": _relative_l2(
            prediction_alpha[:, 21:], target_alpha[:, 21:]
        ),
        "alpha_mae": float(np.mean(np.abs(error_alpha[:, 21:]))),
        "alpha_max_abs": float(np.max(np.abs(error_alpha[:, 21:]))),
        "alpha_bound_violation_count": int(
            np.count_nonzero(
                (prediction_alpha[:, 21:] < 0.0)
                | (prediction_alpha[:, 21:] > 1.0)
            )
        ),
        "alpha_monotonicity_violation_count": int(
            np.count_nonzero(alpha_differences < -1.0e-12)
        ),
        "tool_alpha_nonzero_count": int(
            np.count_nonzero(prediction_alpha[:, :21] != 0.0)
        ),
        "maximum_abs_energy_residual_W_m3": (
            result.diagnostics.maximum_abs_energy_residual_W_m3
        ),
        "maximum_relative_global_energy_residual": (
            result.diagnostics.maximum_relative_global_energy_residual
        ),
        "wall_seconds": elapsed,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "case_count": int(len(frame)),
        "temperature_field_relative_l2_mean": float(
            frame["temperature_relative_l2"].mean()
        ),
        "temperature_field_relative_l2_median": float(
            frame["temperature_relative_l2"].median()
        ),
        "temperature_field_relative_l2_p95": float(
            frame["temperature_relative_l2"].quantile(0.95)
        ),
        "tool_temperature_relative_l2_mean": float(
            frame["tool_temperature_relative_l2"].mean()
        ),
        "composite_temperature_relative_l2_mean": float(
            frame["composite_temperature_relative_l2"].mean()
        ),
        "temperature_mae_K_mean": float(frame["temperature_mae_K"].mean()),
        "temperature_field_absolute_error_K_max": float(
            frame["temperature_max_abs_K"].max()
        ),
        "peak_composite_temperature_error_K_mae": float(
            frame["peak_composite_temperature_error_K"].abs().mean()
        ),
        "alpha_field_relative_l2_mean": float(
            frame["alpha_relative_l2"].mean()
        ),
        "alpha_mae_mean": float(frame["alpha_mae"].mean()),
        "alpha_field_absolute_error_max": float(
            frame["alpha_max_abs"].max()
        ),
        "alpha_bound_violation_count": int(
            frame["alpha_bound_violation_count"].sum()
        ),
        "alpha_monotonicity_violation_count": int(
            frame["alpha_monotonicity_violation_count"].sum()
        ),
        "tool_alpha_nonzero_count": int(frame["tool_alpha_nonzero_count"].sum()),
        "maximum_abs_energy_residual_W_per_m3": float(
            frame["maximum_abs_energy_residual_W_m3"].max()
        ),
        "maximum_relative_global_energy_residual": float(
            frame["maximum_relative_global_energy_residual"].max()
        ),
        "summed_case_wall_seconds": float(frame["wall_seconds"].sum()),
    }


def _acceptance_checks(
    aggregate: dict[str, float | int],
) -> dict[str, bool]:
    return {
        "temperature_field_relative_l2_mean": (
            aggregate["temperature_field_relative_l2_mean"]
            <= ACCEPTANCE["temperature_field_relative_l2_mean_max"]
        ),
        "temperature_field_absolute_error": (
            aggregate["temperature_field_absolute_error_K_max"]
            <= ACCEPTANCE["temperature_field_absolute_error_K_max"]
        ),
        "alpha_field_relative_l2_mean": (
            aggregate["alpha_field_relative_l2_mean"]
            <= ACCEPTANCE["alpha_field_relative_l2_mean_max"]
        ),
        "alpha_bounds": (
            aggregate["alpha_bound_violation_count"]
            <= ACCEPTANCE["alpha_bound_violation_count_max"]
        ),
        "alpha_monotonicity": (
            aggregate["alpha_monotonicity_violation_count"]
            <= ACCEPTANCE["alpha_monotonicity_violation_count_max"]
        ),
        "local_energy_residual": (
            aggregate["maximum_abs_energy_residual_W_per_m3"]
            <= ACCEPTANCE["maximum_abs_energy_residual_W_per_m3_max"]
        ),
        "global_energy_residual": (
            aggregate["maximum_relative_global_energy_residual"]
            <= ACCEPTANCE["maximum_relative_global_energy_residual_max"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--maximum-step-s", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()
    dataset = loadmat(args.input)
    temperature = np.asarray(dataset["dataT"], dtype=np.float64)
    alpha = np.asarray(dataset["dataA"], dtype=np.float64)
    air = np.asarray(dataset["dataTair"], dtype=np.float64)
    count = temperature.shape[0]
    if args.case_limit is not None:
        count = min(count, args.case_limit)
    tasks = [
        (
            case_id,
            air[case_id],
            temperature[case_id].T,
            alpha[case_id].T,
            args.maximum_step_s,
        )
        for case_id in range(count)
    ]
    wall_started = time.perf_counter()
    if args.workers == 1:
        rows = [_run_case(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with context.Pool(args.workers) as pool:
            rows = list(pool.imap_unordered(_run_case, tasks, chunksize=1))
        rows.sort(key=lambda row: int(row["case_id"]))
    wall_seconds = time.perf_counter() - wall_started
    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    aggregate = _aggregate(frame)
    checks = _acceptance_checks(aggregate)
    summary = {
        "schema_version": 1,
        "phase": "P3",
        "benchmark": "immutable_public_ResFNO_Case1",
        "solver": "conservative_node_centred_finite_volume_1d",
        "input": {
            "path": str(args.input.relative_to(ROOT)),
            "sha256": _sha256(args.input),
            "shape_temperature": list(temperature[:count].shape),
            "shape_alpha": list(alpha[:count].shape),
            "shape_air": list(air[:count].shape),
        },
        "numerics": {
            "output_spacing_m": 0.001,
            "output_interval_s": 60.0,
            "maximum_step_s": args.maximum_step_s,
            "workers": args.workers,
            "wall_seconds": wall_seconds,
        },
        "provenance": {
            "parameter_config": "configs/material/as4_8552.yaml",
            "parameter_table": "references/as4_8552_parameter_table.md",
            "kinetic_denominator": "1 + exp(C*(alpha-C_T*T-C_0))",
            "public_geometry": "20 mm tool + 30 mm composite",
            "geometry_conflict_recorded": True,
            "kinetic_printing_conflict_recorded": True,
            "labels_used_for_parameter_fitting": False,
        },
        "acceptance_thresholds_prespecified": ACCEPTANCE,
        "aggregate": aggregate,
        "acceptance_checks": checks,
        "passed": bool(all(checks.values())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.cases, index=False)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
