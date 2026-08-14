"""Run prespecified mesh/time refinement and conservation checks for P3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from cdcureno.solvers import public_case1_grid, simulate_cure_1d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "resfno" / "Case1.mat"
DEFAULT_SUMMARY = ROOT / "outputs" / "tables" / "p3_solver_convergence.json"
DEFAULT_TABLE = ROOT / "outputs" / "tables" / "p3_solver_convergence.csv"
FIXED_CASE_ID = 100
REFERENCE_SPACING_M = 0.00025
REFERENCE_STEP_S = 2.5
MESH_SPACINGS_M = (0.002, 0.001, 0.0005)
TIME_STEPS_S = (20.0, 10.0, 5.0)
ACCEPTANCE = {
    "mesh_errors_strictly_decrease": True,
    "time_errors_strictly_decrease": True,
    "finest_mesh_temperature_relative_l2_max": 5.0e-4,
    "finest_mesh_alpha_relative_l2_max": 5.0e-3,
    "finest_time_temperature_relative_l2_max": 1.0e-4,
    "finest_time_alpha_relative_l2_max": 1.0e-3,
    "maximum_abs_energy_residual_W_m3_max": 1.0e-4,
    "maximum_relative_global_energy_residual_max": 1.0e-8,
}


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def _reference_on_grid(
    values: np.ndarray, reference_z: np.ndarray, target_z: np.ndarray
) -> np.ndarray:
    return np.stack(
        [np.interp(target_z, reference_z, row) for row in values],
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    data = loadmat(args.input)
    air = np.asarray(data["dataTair"][FIXED_CASE_ID], dtype=np.float64)
    times = np.arange(air.size, dtype=np.float64) * 60.0
    cache: dict[tuple[float, float], object] = {}

    def run(spacing_m: float, maximum_step_s: float):
        key = (spacing_m, maximum_step_s)
        if key not in cache:
            started = time.perf_counter()
            result = simulate_cure_1d(
                times,
                air,
                grid=public_case1_grid(spacing_m),
                maximum_step_s=maximum_step_s,
                maximum_coupling_iterations=16,
            )
            cache[key] = (result, time.perf_counter() - started)
        return cache[key]

    reference, reference_wall = run(
        REFERENCE_SPACING_M, REFERENCE_STEP_S
    )
    rows: list[dict[str, float | str]] = []
    for refinement, values in (
        ("mesh", [(spacing, REFERENCE_STEP_S) for spacing in MESH_SPACINGS_M]),
        (
            "time",
            [(REFERENCE_SPACING_M, maximum_step) for maximum_step in TIME_STEPS_S],
        ),
    ):
        for spacing_m, maximum_step_s in values:
            result, wall_seconds = run(spacing_m, maximum_step_s)
            target_temperature = _reference_on_grid(
                reference.temperature_K, reference.z_m, result.z_m
            )
            target_alpha = _reference_on_grid(
                reference.alpha, reference.z_m, result.z_m
            )
            rows.append(
                {
                    "refinement": refinement,
                    "spacing_m": spacing_m,
                    "maximum_step_s": maximum_step_s,
                    "temperature_relative_l2_to_reference": _relative_l2(
                        result.temperature_K, target_temperature
                    ),
                    "temperature_max_abs_K_to_reference": float(
                        np.max(
                            np.abs(result.temperature_K - target_temperature)
                        )
                    ),
                    "alpha_relative_l2_to_reference": _relative_l2(
                        result.alpha, target_alpha
                    ),
                    "alpha_max_abs_to_reference": float(
                        np.max(np.abs(result.alpha - target_alpha))
                    ),
                    "maximum_abs_energy_residual_W_m3": (
                        result.diagnostics.maximum_abs_energy_residual_W_m3
                    ),
                    "maximum_relative_global_energy_residual": (
                        result.diagnostics.maximum_relative_global_energy_residual
                    ),
                    "wall_seconds": wall_seconds,
                }
            )
    frame = pd.DataFrame(rows)
    mesh = frame[frame["refinement"] == "mesh"].sort_values(
        "spacing_m", ascending=False
    )
    time_frame = frame[frame["refinement"] == "time"].sort_values(
        "maximum_step_s", ascending=False
    )

    def decreasing(series: pd.Series) -> bool:
        return bool(np.all(np.diff(series.to_numpy()) < 0.0))

    checks = {
        "mesh_temperature_errors_decrease": decreasing(
            mesh["temperature_relative_l2_to_reference"]
        ),
        "mesh_alpha_errors_decrease": decreasing(
            mesh["alpha_relative_l2_to_reference"]
        ),
        "time_temperature_errors_decrease": decreasing(
            time_frame["temperature_relative_l2_to_reference"]
        ),
        "time_alpha_errors_decrease": decreasing(
            time_frame["alpha_relative_l2_to_reference"]
        ),
        "finest_mesh_temperature": (
            float(mesh.iloc[-1]["temperature_relative_l2_to_reference"])
            <= ACCEPTANCE["finest_mesh_temperature_relative_l2_max"]
        ),
        "finest_mesh_alpha": (
            float(mesh.iloc[-1]["alpha_relative_l2_to_reference"])
            <= ACCEPTANCE["finest_mesh_alpha_relative_l2_max"]
        ),
        "finest_time_temperature": (
            float(
                time_frame.iloc[-1]["temperature_relative_l2_to_reference"]
            )
            <= ACCEPTANCE["finest_time_temperature_relative_l2_max"]
        ),
        "finest_time_alpha": (
            float(time_frame.iloc[-1]["alpha_relative_l2_to_reference"])
            <= ACCEPTANCE["finest_time_alpha_relative_l2_max"]
        ),
        "local_energy_residual": (
            float(frame["maximum_abs_energy_residual_W_m3"].max())
            <= ACCEPTANCE["maximum_abs_energy_residual_W_m3_max"]
        ),
        "global_energy_residual": (
            float(frame["maximum_relative_global_energy_residual"].max())
            <= ACCEPTANCE["maximum_relative_global_energy_residual_max"]
        ),
    }
    summary = {
        "schema_version": 1,
        "phase": "P3",
        "fixed_public_air_schedule_case_id": FIXED_CASE_ID,
        "public_temperature_or_alpha_labels_used": False,
        "reference": {
            "spacing_m": REFERENCE_SPACING_M,
            "maximum_step_s": REFERENCE_STEP_S,
            "wall_seconds": reference_wall,
            "maximum_abs_energy_residual_W_m3": (
                reference.diagnostics.maximum_abs_energy_residual_W_m3
            ),
            "maximum_relative_global_energy_residual": (
                reference.diagnostics.maximum_relative_global_energy_residual
            ),
        },
        "acceptance_thresholds_prespecified": ACCEPTANCE,
        "acceptance_checks": checks,
        "passed": bool(all(checks.values())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
