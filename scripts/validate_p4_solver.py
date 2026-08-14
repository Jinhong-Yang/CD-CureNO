"""Validate the P4 cell-centred 2-D thermochemical solver."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from cdcureno.physics import cure_rate_per_s
from cdcureno.solvers import (
    RobinBoundaries2D,
    public_case1_grid,
    rectangular_tool_composite_grid,
    simulate_cure_1d,
    simulate_cure_2d,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "resfno" / "Case1.mat"
DEFAULT_SUMMARY = ROOT / "outputs" / "tables" / "p4_solver_validation.json"
DEFAULT_TABLE = ROOT / "outputs" / "tables" / "p4_solver_convergence.csv"
DEFAULT_RUNTIME = ROOT / "outputs" / "tables" / "p4_solver_runtime.json"
P4_CONFIG = ROOT / "configs" / "data" / "p4_2d_benchmark_v1.yaml"
FIXED_PUBLIC_CASE_ID = 100
CORE_MAXIMUM_STEP_S = float(
    yaml.safe_load(P4_CONFIG.read_text(encoding="utf-8"))["tiers"]["core"][
        "maximum_step_s"
    ]
)
ACCEPTANCE = {
    "extrusion_temperature_relative_l2_max": 5.0e-4,
    "extrusion_alpha_relative_l2_max": 5.0e-3,
    "extrusion_temperature_max_abs_K_max": 2.0,
    "lateral_invariance_score_max": 1.0e-10,
    "maximum_abs_energy_residual_W_m3_max": 1.0e-4,
    "maximum_relative_global_energy_residual_max": 1.0e-8,
    "maximum_interface_flux_imbalance_W_max": 1.0e-9,
    "maximum_temperature_interface_jump_K_max": 1.0e-10,
    "maximum_robin_flux_imbalance_W_max": 1.0e-9,
    "maximum_converged_coupling_update_K_max": 1.0e-9,
    "independent_temperature_relative_l2_max": 5.0e-4,
    "independent_alpha_relative_l2_max": 5.0e-4,
    "core_time_temperature_relative_l2_max": 1.0e-3,
    "core_time_alpha_relative_l2_max": 1.0e-5,
    "full_cycle_core_temperature_relative_l2_max": 3.0e-4,
    "full_cycle_core_temperature_rise_relative_l2_max": 1.0e-3,
    "full_cycle_core_temperature_max_abs_K_max": 0.5,
    "full_cycle_core_alpha_relative_l2_max": 2.0e-3,
    "full_cycle_minimum_observed_time_order": 0.8,
    "mesh_errors_strictly_decrease": True,
    "time_errors_strictly_decrease": True,
}


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - target) / np.linalg.norm(target))


def _invariant_fields(result: Any, composite_mask: np.ndarray) -> dict[str, Any]:
    alpha = np.asarray(result.alpha)
    return {
        "finite_fields": bool(
            np.all(np.isfinite(result.temperature_K))
            and np.all(np.isfinite(alpha))
        ),
        "alpha_bound_violation_count": int(
            np.count_nonzero((alpha < -1.0e-12) | (alpha > 1.0 + 1.0e-12))
        ),
        "alpha_monotonicity_violation_count": int(
            np.count_nonzero(np.diff(alpha, axis=0) < -1.0e-12)
        ),
        "tool_alpha_nonzero_count": int(
            np.count_nonzero(alpha[:, ~composite_mask] != 0.0)
        ),
    }


def _interpolate_z(
    values: np.ndarray, source_z: np.ndarray, target_z: np.ndarray
) -> np.ndarray:
    return np.stack(
        [np.interp(target_z, source_z, field) for field in values], axis=0
    )


def _interpolate_2d(
    values: np.ndarray,
    source_z: np.ndarray,
    source_x: np.ndarray,
    target_z: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    target_mesh = np.stack(
        np.meshgrid(target_z, target_x, indexing="ij"), axis=-1
    )
    outputs = []
    for field in values:
        interpolator = RegularGridInterpolator(
            (source_z, source_x), field, bounds_error=True
        )
        outputs.append(interpolator(target_mesh))
    return np.asarray(outputs)


def _robin_conductance(
    h: np.ndarray,
    conductivity: np.ndarray,
    half_distance: float,
    face_measure: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(h)
    active = h > 0.0
    output[active] = face_measure[active] / (
        1.0 / h[active] + half_distance / conductivity[active]
    )
    return output


def _independent_method_of_lines(
    times_s: np.ndarray,
    air_temperature_K: np.ndarray,
    *,
    width_m: float,
    spacing_x_m: float,
    spacing_z_m: float,
    boundaries: RobinBoundaries2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive BDF cross-check with independently assembled face fluxes."""

    grid = rectangular_tool_composite_grid(
        width_m=width_m,
        spacing_x_m=spacing_x_m,
        spacing_z_m=spacing_z_m,
    )
    nz, nx = grid.shape
    volume = grid.control_volume_m2
    dx = spacing_x_m
    dz = spacing_z_m
    x_conductance = (
        dz
        / (
            0.5 * dx / grid.conductivity_x_W_m_K[:, :-1]
            + 0.5 * dx / grid.conductivity_x_W_m_K[:, 1:]
        )
    )
    z_conductance = (
        dx
        / (
            0.5 * dz / grid.conductivity_z_W_m_K[:-1, :]
            + 0.5 * dz / grid.conductivity_z_W_m_K[1:, :]
        )
    )

    def boundary_array(value: float | np.ndarray, size: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        return (
            np.full(size, float(array), dtype=np.float64)
            if array.ndim == 0
            else array.copy()
        )

    lower_h = boundary_array(boundaries.lower_h_W_m2_K, nx)
    upper_h = boundary_array(boundaries.upper_h_W_m2_K, nx)
    left_h = boundary_array(boundaries.left_h_W_m2_K, nz)
    right_h = boundary_array(boundaries.right_h_W_m2_K, nz)
    lower_g = _robin_conductance(
        lower_h,
        grid.conductivity_z_W_m_K[0],
        0.5 * dz,
        np.full(nx, dx),
    )
    upper_g = _robin_conductance(
        upper_h,
        grid.conductivity_z_W_m_K[-1],
        0.5 * dz,
        np.full(nx, dx),
    )
    left_g = _robin_conductance(
        left_h,
        grid.conductivity_x_W_m_K[:, 0],
        0.5 * dx,
        np.full(nz, dz),
    )
    right_g = _robin_conductance(
        right_h,
        grid.conductivity_x_W_m_K[:, -1],
        0.5 * dx,
        np.full(nz, dz),
    )

    initial_temperature = np.full(grid.shape, 293.0)
    initial_alpha = np.zeros(grid.shape)
    initial_alpha[grid.composite_mask] = 0.05
    initial = np.concatenate(
        [initial_temperature.ravel(), initial_alpha.ravel()]
    )

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature = state[: nz * nx].reshape(grid.shape)
        alpha = state[nz * nx :].reshape(grid.shape)
        air = float(np.interp(time_s, times_s, air_temperature_K))
        rate = np.zeros(grid.shape)
        rate[grid.composite_mask] = cure_rate_per_s(
            temperature[grid.composite_mask],
            alpha[grid.composite_mask],
        )
        outward = np.zeros(grid.shape)
        x_flux = x_conductance * (
            temperature[:, :-1] - temperature[:, 1:]
        )
        outward[:, :-1] += x_flux
        outward[:, 1:] -= x_flux
        z_flux = z_conductance * (
            temperature[:-1] - temperature[1:]
        )
        outward[:-1] += z_flux
        outward[1:] -= z_flux
        outward[0] += lower_g * (temperature[0] - air)
        outward[-1] += upper_g * (temperature[-1] - air)
        outward[:, 0] += left_g * (temperature[:, 0] - air)
        outward[:, -1] += right_g * (temperature[:, -1] - air)
        temperature_rate = (
            grid.cure_source_J_m3_per_alpha * rate - outward / volume
        ) / (grid.density_kg_m3 * grid.specific_heat_J_kg_K)
        return np.concatenate([temperature_rate.ravel(), rate.ravel()])

    solution = solve_ivp(
        rhs,
        (float(times_s[0]), float(times_s[-1])),
        initial,
        method="BDF",
        t_eval=times_s,
        rtol=2.0e-9,
        atol=1.0e-10,
        max_step=2.0,
    )
    if not solution.success:
        raise RuntimeError(f"Independent BDF integration failed: {solution.message}")
    states = solution.y.T
    temperature = states[:, : nz * nx].reshape((-1, nz, nx))
    alpha = states[:, nz * nx :].reshape((-1, nz, nx))
    return temperature, alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()

    public_air = np.asarray(
        loadmat(args.input)["dataTair"][FIXED_PUBLIC_CASE_ID], dtype=np.float64
    )
    public_times = np.arange(public_air.size, dtype=np.float64) * 60.0
    reference_1d = simulate_cure_1d(
        public_times,
        public_air,
        grid=public_case1_grid(0.00025),
        maximum_step_s=2.5,
        maximum_coupling_iterations=16,
    )
    extrusion_grid = rectangular_tool_composite_grid(
        width_m=0.04, spacing_x_m=0.01, spacing_z_m=0.001
    )
    extrusion_2d = simulate_cure_2d(
        public_times,
        public_air,
        grid=extrusion_grid,
        boundaries=RobinBoundaries2D(70.0, 120.0),
        maximum_step_s=2.5,
        maximum_coupling_iterations=16,
    )
    target_temperature = _interpolate_z(
        reference_1d.temperature_K, reference_1d.z_m, extrusion_2d.z_m
    )
    target_alpha = _interpolate_z(
        reference_1d.alpha, reference_1d.z_m, extrusion_2d.z_m
    )
    extrusion_temperature_relative = _relative_l2(
        extrusion_2d.temperature_K[:, :, 0], target_temperature
    )
    extrusion_alpha_relative = _relative_l2(
        extrusion_2d.alpha[:, :, 0], target_alpha
    )
    extrusion_temperature_max = float(
        np.max(
            np.abs(extrusion_2d.temperature_K[:, :, 0] - target_temperature)
        )
    )
    lateral_std = float(
        np.sqrt(np.mean(np.var(extrusion_2d.temperature_K, axis=-1)))
    )
    lateral_scale = float(np.std(extrusion_2d.temperature_K))
    lateral_invariance = lateral_std / max(lateral_scale, 1.0e-12)

    check_times = np.arange(0.0, 601.0, 60.0)
    check_air = 293.0 + 130.0 * check_times / check_times[-1]
    check_grid = rectangular_tool_composite_grid(
        width_m=0.06, spacing_x_m=0.02, spacing_z_m=0.005
    )
    check_top_h = 120.0 + 30.0 * np.cos(
        2.0 * np.pi * check_grid.x_m / 0.06
    )
    check_boundaries = RobinBoundaries2D(
        70.0, check_top_h, left_h_W_m2_K=20.0, right_h_W_m2_K=60.0
    )
    primary_check = simulate_cure_2d(
        check_times,
        check_air,
        grid=check_grid,
        boundaries=check_boundaries,
        maximum_step_s=0.5,
        maximum_coupling_iterations=16,
    )
    independent_temperature, independent_alpha = _independent_method_of_lines(
        check_times,
        check_air,
        width_m=0.06,
        spacing_x_m=0.02,
        spacing_z_m=0.005,
        boundaries=check_boundaries,
    )
    independent_temperature_relative = _relative_l2(
        primary_check.temperature_K, independent_temperature
    )
    independent_alpha_relative = _relative_l2(
        primary_check.alpha, independent_alpha
    )

    convergence_times = np.arange(0.0, 301.0, 30.0)
    convergence_air = 293.0 + 120.0 * convergence_times / 300.0
    reference_grid = rectangular_tool_composite_grid(
        width_m=0.20, spacing_x_m=0.0025, spacing_z_m=0.0005
    )
    reference_top_h = 120.0 + 35.0 * np.cos(
        4.0 * np.pi * reference_grid.x_m / 0.20
    )
    convergence_reference = simulate_cure_2d(
        convergence_times,
        convergence_air,
        grid=reference_grid,
        boundaries=RobinBoundaries2D(70.0, reference_top_h),
        maximum_step_s=1.25,
        maximum_coupling_iterations=16,
    )
    rows: list[dict[str, float | str | bool]] = []
    runtime_rows: list[dict[str, float | str]] = []
    for spacing_x, spacing_z in (
        (0.02, 0.005),
        (0.01, 0.0025),
        (0.005, 0.001),
    ):
        grid = rectangular_tool_composite_grid(
            width_m=0.20,
            spacing_x_m=spacing_x,
            spacing_z_m=spacing_z,
        )
        top_h = 120.0 + 35.0 * np.cos(
            4.0 * np.pi * grid.x_m / 0.20
        )
        result = simulate_cure_2d(
            convergence_times,
            convergence_air,
            grid=grid,
            boundaries=RobinBoundaries2D(70.0, top_h),
            maximum_step_s=1.25,
            maximum_coupling_iterations=16,
        )
        target_t = _interpolate_2d(
            convergence_reference.temperature_K,
            convergence_reference.z_m,
            convergence_reference.x_m,
            result.z_m,
            result.x_m,
        )
        target_a = _interpolate_2d(
            convergence_reference.alpha,
            convergence_reference.z_m,
            convergence_reference.x_m,
            result.z_m,
            result.x_m,
        )
        row: dict[str, float | str | bool] = {
                "refinement": "mesh",
                "spacing_x_m": spacing_x,
                "spacing_z_m": spacing_z,
                "maximum_step_s": 1.25,
                "temperature_relative_l2": _relative_l2(
                    result.temperature_K, target_t
                ),
                "alpha_relative_l2": _relative_l2(result.alpha, target_a),
                "maximum_abs_energy_residual_W_m3": (
                    result.diagnostics.maximum_abs_energy_residual_W_m3
                ),
                "maximum_relative_global_energy_residual": (
                    result.diagnostics.maximum_relative_global_energy_residual
                ),
                "maximum_interface_flux_imbalance_W": (
                    result.diagnostics.maximum_interface_flux_imbalance_W
                ),
                "maximum_temperature_interface_jump_K": (
                    result.diagnostics.maximum_temperature_interface_jump_K
                ),
                "maximum_robin_flux_imbalance_W": (
                    result.diagnostics.maximum_robin_flux_imbalance_W
                ),
                "all_coupling_steps_converged": (
                    result.diagnostics.all_coupling_steps_converged
                ),
                "maximum_converged_coupling_update_K": (
                    result.diagnostics.maximum_converged_coupling_update_K
                ),
                **_invariant_fields(result, grid.composite_mask),
            }
        rows.append(row)
        runtime_rows.append(
            {
                "refinement": "mesh",
                "spacing_x_m": spacing_x,
                "spacing_z_m": spacing_z,
                "maximum_step_s": 1.25,
                "wall_seconds": result.diagnostics.wall_seconds,
            }
        )
    time_grid = rectangular_tool_composite_grid(
        width_m=0.20, spacing_x_m=0.005, spacing_z_m=0.001
    )
    time_top_h = 120.0 + 35.0 * np.cos(
        4.0 * np.pi * time_grid.x_m / 0.20
    )
    time_reference = simulate_cure_2d(
        convergence_times,
        convergence_air,
        grid=time_grid,
        boundaries=RobinBoundaries2D(70.0, time_top_h),
        maximum_step_s=1.25,
        maximum_coupling_iterations=16,
    )
    for maximum_step in (10.0, 5.0, 2.5):
        result = simulate_cure_2d(
            convergence_times,
            convergence_air,
            grid=time_grid,
            boundaries=RobinBoundaries2D(70.0, time_top_h),
            maximum_step_s=maximum_step,
            maximum_coupling_iterations=16,
        )
        row = {
                "refinement": "time",
                "spacing_x_m": 0.005,
                "spacing_z_m": 0.001,
                "maximum_step_s": maximum_step,
                "temperature_relative_l2": _relative_l2(
                    result.temperature_K, time_reference.temperature_K
                ),
                "alpha_relative_l2": _relative_l2(
                    result.alpha, time_reference.alpha
                ),
                "maximum_abs_energy_residual_W_m3": (
                    result.diagnostics.maximum_abs_energy_residual_W_m3
                ),
                "maximum_relative_global_energy_residual": (
                    result.diagnostics.maximum_relative_global_energy_residual
                ),
                "maximum_interface_flux_imbalance_W": (
                    result.diagnostics.maximum_interface_flux_imbalance_W
                ),
                "maximum_temperature_interface_jump_K": (
                    result.diagnostics.maximum_temperature_interface_jump_K
                ),
                "maximum_robin_flux_imbalance_W": (
                    result.diagnostics.maximum_robin_flux_imbalance_W
                ),
                "all_coupling_steps_converged": (
                    result.diagnostics.all_coupling_steps_converged
                ),
                "maximum_converged_coupling_update_K": (
                    result.diagnostics.maximum_converged_coupling_update_K
                ),
                **_invariant_fields(result, time_grid.composite_mask),
            }
        rows.append(row)
        runtime_rows.append(
            {
                "refinement": "time",
                "spacing_x_m": 0.005,
                "spacing_z_m": 0.001,
                "maximum_step_s": maximum_step,
                "wall_seconds": result.diagnostics.wall_seconds,
            }
        )

    full_cycle_times = np.arange(0.0, 13320.0 + 1.0, 120.0)
    full_cycle_air = np.interp(
        full_cycle_times,
        [0.0, 1800.0, 3600.0, 5400.0, 10800.0, 13320.0],
        [293.0, 455.0, 370.0, 475.0, 475.0, 293.0],
    )
    full_cycle_grid = rectangular_tool_composite_grid(
        width_m=0.20,
        spacing_x_m=0.005,
        spacing_z_m=0.001,
        composite_conductivity_x_scale=0.85,
        composite_conductivity_z_scale=0.85,
    )
    full_cycle_source = (
        full_cycle_grid.cure_source_J_m3_per_alpha.copy()
    )
    full_cycle_source[full_cycle_grid.composite_mask] *= 1.10
    full_cycle_grid = replace(
        full_cycle_grid,
        cure_source_J_m3_per_alpha=full_cycle_source,
    )
    full_cycle_top_h = 175.0 + 40.0 * np.cos(
        8.0 * np.pi * full_cycle_grid.x_m / 0.20
    )
    full_cycle_boundaries = RobinBoundaries2D(
        115.0,
        full_cycle_top_h,
        left_h_W_m2_K=100.0,
        right_h_W_m2_K=75.0,
    )
    full_cycle_reference = simulate_cure_2d(
        full_cycle_times,
        full_cycle_air,
        grid=full_cycle_grid,
        boundaries=full_cycle_boundaries,
        maximum_step_s=1.25,
        maximum_coupling_iterations=16,
    )
    for maximum_step in (20.0, 10.0, 5.0, 2.5):
        result = simulate_cure_2d(
            full_cycle_times,
            full_cycle_air,
            grid=full_cycle_grid,
            boundaries=full_cycle_boundaries,
            maximum_step_s=maximum_step,
            maximum_coupling_iterations=16,
        )
        rows.append(
            {
                "refinement": "full_cycle_time",
                "spacing_x_m": 0.005,
                "spacing_z_m": 0.001,
                "maximum_step_s": maximum_step,
                "temperature_relative_l2": _relative_l2(
                    result.temperature_K,
                    full_cycle_reference.temperature_K,
                ),
                "temperature_rise_relative_l2": float(
                    np.linalg.norm(
                        result.temperature_K
                        - full_cycle_reference.temperature_K
                    )
                    / np.linalg.norm(
                        full_cycle_reference.temperature_K - 293.0
                    )
                ),
                "temperature_max_abs_K": float(
                    np.max(
                        np.abs(
                            result.temperature_K
                            - full_cycle_reference.temperature_K
                        )
                    )
                ),
                "alpha_relative_l2": _relative_l2(
                    result.alpha, full_cycle_reference.alpha
                ),
                "maximum_abs_energy_residual_W_m3": (
                    result.diagnostics.maximum_abs_energy_residual_W_m3
                ),
                "maximum_relative_global_energy_residual": (
                    result.diagnostics.maximum_relative_global_energy_residual
                ),
                "maximum_interface_flux_imbalance_W": (
                    result.diagnostics.maximum_interface_flux_imbalance_W
                ),
                "maximum_temperature_interface_jump_K": (
                    result.diagnostics.maximum_temperature_interface_jump_K
                ),
                "maximum_robin_flux_imbalance_W": (
                    result.diagnostics.maximum_robin_flux_imbalance_W
                ),
                "all_coupling_steps_converged": (
                    result.diagnostics.all_coupling_steps_converged
                ),
                "maximum_converged_coupling_update_K": (
                    result.diagnostics.maximum_converged_coupling_update_K
                ),
                **_invariant_fields(
                    result, full_cycle_grid.composite_mask
                ),
            }
        )
        runtime_rows.append(
            {
                "refinement": "full_cycle_time",
                "spacing_x_m": 0.005,
                "spacing_z_m": 0.001,
                "maximum_step_s": maximum_step,
                "wall_seconds": result.diagnostics.wall_seconds,
            }
        )
    table = pd.DataFrame(rows)
    mesh_rows = table[table["refinement"] == "mesh"]
    time_rows = table[table["refinement"] == "time"]
    full_cycle_rows = table[table["refinement"] == "full_cycle_time"]
    core_time_rows = time_rows[
        np.isclose(time_rows["maximum_step_s"], CORE_MAXIMUM_STEP_S)
    ]
    if len(core_time_rows) != 1:
        raise RuntimeError("The actual core time step must have one validation row.")
    full_cycle_core_rows = full_cycle_rows[
        np.isclose(full_cycle_rows["maximum_step_s"], CORE_MAXIMUM_STEP_S)
    ]
    if len(full_cycle_core_rows) != 1:
        raise RuntimeError(
            "The actual core time step must have one full-cycle validation row."
        )
    full_cycle_temperature_orders = np.log2(
        full_cycle_rows["temperature_relative_l2"].to_numpy()[:-1]
        / full_cycle_rows["temperature_relative_l2"].to_numpy()[1:]
    )
    full_cycle_alpha_orders = np.log2(
        full_cycle_rows["alpha_relative_l2"].to_numpy()[:-1]
        / full_cycle_rows["alpha_relative_l2"].to_numpy()[1:]
    )
    diagnostic_results = (
        extrusion_2d,
        primary_check,
        convergence_reference,
        full_cycle_reference,
    )
    direct_invariants = (
        _invariant_fields(extrusion_2d, extrusion_grid.composite_mask),
        _invariant_fields(primary_check, check_grid.composite_mask),
        _invariant_fields(
            convergence_reference, reference_grid.composite_mask
        ),
        _invariant_fields(
            full_cycle_reference, full_cycle_grid.composite_mask
        ),
    )
    checks = {
        "extrusion_temperature": (
            extrusion_temperature_relative
            <= ACCEPTANCE["extrusion_temperature_relative_l2_max"]
        ),
        "extrusion_alpha": (
            extrusion_alpha_relative
            <= ACCEPTANCE["extrusion_alpha_relative_l2_max"]
        ),
        "extrusion_temperature_max": (
            extrusion_temperature_max
            <= ACCEPTANCE["extrusion_temperature_max_abs_K_max"]
        ),
        "lateral_invariance": (
            lateral_invariance <= ACCEPTANCE["lateral_invariance_score_max"]
        ),
        "energy_local": (
            max(
                extrusion_2d.diagnostics.maximum_abs_energy_residual_W_m3,
                primary_check.diagnostics.maximum_abs_energy_residual_W_m3,
                float(table["maximum_abs_energy_residual_W_m3"].max()),
            )
            <= ACCEPTANCE["maximum_abs_energy_residual_W_m3_max"]
        ),
        "energy_global": (
            max(
                extrusion_2d.diagnostics.maximum_relative_global_energy_residual,
                primary_check.diagnostics.maximum_relative_global_energy_residual,
                float(table["maximum_relative_global_energy_residual"].max()),
            )
            <= ACCEPTANCE["maximum_relative_global_energy_residual_max"]
        ),
        "interface_flux_continuity": (
            max(
                result.diagnostics.maximum_interface_flux_imbalance_W
                for result in diagnostic_results
            )
            <= ACCEPTANCE["maximum_interface_flux_imbalance_W_max"]
            and float(table["maximum_interface_flux_imbalance_W"].max())
            <= ACCEPTANCE["maximum_interface_flux_imbalance_W_max"]
        ),
        "interface_temperature_continuity": (
            max(
                result.diagnostics.maximum_temperature_interface_jump_K
                for result in diagnostic_results
            )
            <= ACCEPTANCE["maximum_temperature_interface_jump_K_max"]
            and float(table["maximum_temperature_interface_jump_K"].max())
            <= ACCEPTANCE["maximum_temperature_interface_jump_K_max"]
        ),
        "robin_flux_continuity": (
            max(
                result.diagnostics.maximum_robin_flux_imbalance_W
                for result in diagnostic_results
            )
            <= ACCEPTANCE["maximum_robin_flux_imbalance_W_max"]
            and float(table["maximum_robin_flux_imbalance_W"].max())
            <= ACCEPTANCE["maximum_robin_flux_imbalance_W_max"]
        ),
        "coupling_convergence": (
            all(
                result.diagnostics.all_coupling_steps_converged
                for result in diagnostic_results
            )
            and bool(table["all_coupling_steps_converged"].all())
            and max(
                result.diagnostics.maximum_converged_coupling_update_K
                for result in diagnostic_results
            )
            <= ACCEPTANCE["maximum_converged_coupling_update_K_max"]
            and float(
                table["maximum_converged_coupling_update_K"].max()
            )
            <= ACCEPTANCE["maximum_converged_coupling_update_K_max"]
        ),
        "finite_and_cure_invariants": (
            all(
                invariant["finite_fields"]
                and invariant["alpha_bound_violation_count"] == 0
                and invariant["alpha_monotonicity_violation_count"] == 0
                and invariant["tool_alpha_nonzero_count"] == 0
                for invariant in direct_invariants
            )
            and bool(table["finite_fields"].all())
            and int(table["alpha_bound_violation_count"].sum()) == 0
            and int(table["alpha_monotonicity_violation_count"].sum()) == 0
            and int(table["tool_alpha_nonzero_count"].sum()) == 0
        ),
        "independent_temperature": (
            independent_temperature_relative
            <= ACCEPTANCE["independent_temperature_relative_l2_max"]
        ),
        "independent_alpha": (
            independent_alpha_relative
            <= ACCEPTANCE["independent_alpha_relative_l2_max"]
        ),
        "mesh_temperature_errors_decrease": bool(
            np.all(np.diff(mesh_rows["temperature_relative_l2"]) < 0.0)
        ),
        "mesh_alpha_errors_decrease": bool(
            np.all(np.diff(mesh_rows["alpha_relative_l2"]) < 0.0)
        ),
        "time_temperature_errors_decrease": bool(
            np.all(np.diff(time_rows["temperature_relative_l2"]) < 0.0)
        ),
        "time_alpha_errors_decrease": bool(
            np.all(np.diff(time_rows["alpha_relative_l2"]) < 0.0)
        ),
        "core_time_temperature_accuracy": (
            float(core_time_rows.iloc[0]["temperature_relative_l2"])
            <= ACCEPTANCE["core_time_temperature_relative_l2_max"]
        ),
        "core_time_alpha_accuracy": (
            float(core_time_rows.iloc[0]["alpha_relative_l2"])
            <= ACCEPTANCE["core_time_alpha_relative_l2_max"]
        ),
        "full_cycle_time_temperature_errors_decrease": bool(
            np.all(
                np.diff(full_cycle_rows["temperature_relative_l2"]) < 0.0
            )
        ),
        "full_cycle_time_alpha_errors_decrease": bool(
            np.all(np.diff(full_cycle_rows["alpha_relative_l2"]) < 0.0)
        ),
        "full_cycle_core_temperature_accuracy": (
            float(
                full_cycle_core_rows.iloc[0]["temperature_relative_l2"]
            )
            <= ACCEPTANCE["full_cycle_core_temperature_relative_l2_max"]
        ),
        "full_cycle_core_temperature_rise_accuracy": (
            float(
                full_cycle_core_rows.iloc[0][
                    "temperature_rise_relative_l2"
                ]
            )
            <= ACCEPTANCE[
                "full_cycle_core_temperature_rise_relative_l2_max"
            ]
        ),
        "full_cycle_core_temperature_max_accuracy": (
            float(full_cycle_core_rows.iloc[0]["temperature_max_abs_K"])
            <= ACCEPTANCE["full_cycle_core_temperature_max_abs_K_max"]
        ),
        "full_cycle_core_alpha_accuracy": (
            float(full_cycle_core_rows.iloc[0]["alpha_relative_l2"])
            <= ACCEPTANCE["full_cycle_core_alpha_relative_l2_max"]
        ),
        "full_cycle_temperature_observed_order": (
            float(np.min(full_cycle_temperature_orders))
            >= ACCEPTANCE["full_cycle_minimum_observed_time_order"]
        ),
        "full_cycle_alpha_observed_order": (
            float(np.min(full_cycle_alpha_orders))
            >= ACCEPTANCE["full_cycle_minimum_observed_time_order"]
        ),
    }
    payload = {
        "schema_version": 1,
        "phase": "P4",
        "solver": "structured_cell_centred_conservative_finite_volume_2d",
        "acceptance_thresholds_prespecified": ACCEPTANCE,
        "acceptance_checks": checks,
        "extrusion_anchor": {
            "fixed_public_air_schedule_case_id": FIXED_PUBLIC_CASE_ID,
            "public_temperature_or_alpha_labels_used": False,
            "reference_1d_spacing_m": 0.00025,
            "reference_maximum_step_s": 2.5,
            "solver_2d_spacing_x_m": 0.01,
            "solver_2d_spacing_z_m": 0.001,
            "temperature_relative_l2": extrusion_temperature_relative,
            "temperature_max_abs_K": extrusion_temperature_max,
            "alpha_relative_l2": extrusion_alpha_relative,
            "lateral_invariance_score": lateral_invariance,
            "maximum_abs_energy_residual_W_m3": (
                extrusion_2d.diagnostics.maximum_abs_energy_residual_W_m3
            ),
            "maximum_relative_global_energy_residual": (
                extrusion_2d.diagnostics.maximum_relative_global_energy_residual
            ),
            "maximum_interface_flux_imbalance_W": (
                extrusion_2d.diagnostics.maximum_interface_flux_imbalance_W
            ),
            "maximum_temperature_interface_jump_K": (
                extrusion_2d.diagnostics.maximum_temperature_interface_jump_K
            ),
            "maximum_robin_flux_imbalance_W": (
                extrusion_2d.diagnostics.maximum_robin_flux_imbalance_W
            ),
            "all_coupling_steps_converged": (
                extrusion_2d.diagnostics.all_coupling_steps_converged
            ),
            "maximum_converged_coupling_update_K": (
                extrusion_2d.diagnostics.maximum_converged_coupling_update_K
            ),
            "physical_invariants": direct_invariants[0],
        },
        "independent_cross_check": {
            "method": "independently_assembled_method_of_lines_scipy_BDF",
            "case": "smooth_top_htc_plus_asymmetric_edge_cooling",
            "temperature_relative_l2": independent_temperature_relative,
            "alpha_relative_l2": independent_alpha_relative,
            "primary_maximum_step_s": 0.5,
            "reference_rtol": 2.0e-9,
            "reference_atol": 1.0e-10,
            "primary_maximum_abs_energy_residual_W_m3": (
                primary_check.diagnostics.maximum_abs_energy_residual_W_m3
            ),
            "primary_maximum_relative_global_energy_residual": (
                primary_check.diagnostics.maximum_relative_global_energy_residual
            ),
            "primary_maximum_interface_flux_imbalance_W": (
                primary_check.diagnostics.maximum_interface_flux_imbalance_W
            ),
            "primary_maximum_temperature_interface_jump_K": (
                primary_check.diagnostics.maximum_temperature_interface_jump_K
            ),
            "primary_maximum_robin_flux_imbalance_W": (
                primary_check.diagnostics.maximum_robin_flux_imbalance_W
            ),
            "primary_maximum_converged_coupling_update_K": (
                primary_check.diagnostics.maximum_converged_coupling_update_K
            ),
            "shared_components_limitation": (
                "The independent assembly uses the same grid, material values, "
                "and cure law; it independently checks flux assembly and time "
                "integration, not those shared inputs."
            ),
            "primary_physical_invariants": direct_invariants[1],
        },
        "convergence_reference": {
            "spacing_x_m": 0.0025,
            "spacing_z_m": 0.0005,
            "maximum_step_s": 1.25,
        },
        "core_discretization": {
            "spacing_x_m": 0.005,
            "spacing_z_m": 0.001,
            "maximum_step_s": CORE_MAXIMUM_STEP_S,
            "temperature_relative_l2": float(
                core_time_rows.iloc[0]["temperature_relative_l2"]
            ),
            "alpha_relative_l2": float(
                core_time_rows.iloc[0]["alpha_relative_l2"]
            ),
            "full_cycle_temperature_relative_l2": float(
                full_cycle_core_rows.iloc[0]["temperature_relative_l2"]
            ),
            "full_cycle_temperature_rise_relative_l2": float(
                full_cycle_core_rows.iloc[0][
                    "temperature_rise_relative_l2"
                ]
            ),
            "full_cycle_temperature_max_abs_K": float(
                full_cycle_core_rows.iloc[0]["temperature_max_abs_K"]
            ),
            "full_cycle_alpha_relative_l2": float(
                full_cycle_core_rows.iloc[0]["alpha_relative_l2"]
            ),
            "full_cycle_temperature_observed_orders": (
                full_cycle_temperature_orders.tolist()
            ),
            "full_cycle_alpha_observed_orders": (
                full_cycle_alpha_orders.tolist()
            ),
            "full_cycle_stress_case": {
                "cycle": (
                    "455 K ramp/hold, 370 K dip, 475 K ramp/hold, cooldown"
                ),
                "bottom_h_W_m2_K": 115.0,
                "top_h_mean_W_m2_K": 175.0,
                "top_h_amplitude_W_m2_K": 40.0,
                "top_h_frequency_cycles_per_width": 4,
                "left_h_W_m2_K": 100.0,
                "right_h_W_m2_K": 75.0,
                "conductivity_scale": 0.85,
                "reaction_enthalpy_scale": 1.10,
                "reference_maximum_step_s": 1.25,
            },
        },
        "passed": bool(all(checks.values())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table.to_csv(args.table, index=False)
    args.runtime.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "P4",
                "role": "noncanonical_runtime_telemetry",
                "extrusion_wall_seconds": (
                    extrusion_2d.diagnostics.wall_seconds
                ),
                "independent_primary_wall_seconds": (
                    primary_check.diagnostics.wall_seconds
                ),
                "convergence_reference_wall_seconds": (
                    convergence_reference.diagnostics.wall_seconds
                ),
                "full_cycle_reference_wall_seconds": (
                    full_cycle_reference.diagnostics.wall_seconds
                ),
                "rows": runtime_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
