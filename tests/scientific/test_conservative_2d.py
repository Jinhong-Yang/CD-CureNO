from dataclasses import replace

import numpy as np
import pytest

from cdcureno.physics import PublicCase1Material
from cdcureno.solvers import (
    LayeredGrid2D,
    RobinBoundaries,
    RobinBoundaries2D,
    public_case1_grid,
    rectangular_tool_composite_grid,
    simulate_cure_1d,
    simulate_cure_2d,
)
from cdcureno.solvers.conservative_2d import (
    _assemble_operator,
    _interface_diagnostics_2d,
    _robin_flux_imbalance_2d,
)


def _homogeneous_grid() -> LayeredGrid2D:
    x_width = np.full(50, 0.002)
    z_width = np.full(2, 0.005)
    shape = (z_width.size, x_width.size)
    return LayeredGrid2D(
        x_m=(np.arange(50) + 0.5) * 0.002,
        z_m=(np.arange(2) + 0.5) * 0.005,
        control_volume_width_x_m=x_width,
        control_volume_width_z_m=z_width,
        composite_mask=np.zeros(shape, dtype=bool),
        density_kg_m3=np.full(shape, 1000.0),
        specific_heat_J_kg_K=np.full(shape, 1000.0),
        conductivity_x_W_m_K=np.ones(shape),
        conductivity_z_W_m_K=np.ones(shape),
        cure_source_J_m3_per_alpha=np.zeros(shape),
    )


def test_pure_diffusion_matches_insulated_cosine_solution() -> None:
    grid = _homogeneous_grid()
    initial = 300.0 + 10.0 * np.cos(np.pi * grid.x_m / 0.1)[None, :]
    initial = np.broadcast_to(initial, grid.shape).copy()
    times = np.array([0.0, 250.0, 500.0])
    result = simulate_cure_2d(
        times,
        np.full_like(times, 300.0),
        grid=grid,
        boundaries=RobinBoundaries2D(0.0, 0.0),
        initial_temperature_K=initial,
        initial_alpha=0.0,
        maximum_step_s=0.5,
    )
    diffusivity = 1.0e-6
    exact = np.empty_like(result.temperature_K)
    for index, time_s in enumerate(times):
        amplitude = np.exp(-diffusivity * (np.pi / 0.1) ** 2 * time_s)
        exact[index] = 300.0 + amplitude * (initial - 300.0)
    relative = np.linalg.norm(result.temperature_K - exact) / np.linalg.norm(
        exact - 300.0
    )
    assert relative < 1.0e-3


def test_uniform_manufactured_source_matches_energy_solution() -> None:
    grid = _homogeneous_grid()
    times = np.array([0.0, 10.0, 20.0])
    source = 2.5e4
    result = simulate_cure_2d(
        times,
        np.full_like(times, 300.0),
        grid=grid,
        boundaries=RobinBoundaries2D(0.0, 0.0),
        initial_temperature_K=300.0,
        initial_alpha=0.0,
        external_heat_source_W_m3=source,
        maximum_step_s=2.0,
    )
    expected = 300.0 + source * times / 1.0e6
    assert np.max(
        np.abs(result.temperature_K - expected[:, None, None])
    ) < 2e-11
    assert result.diagnostics.maximum_abs_energy_residual_W_m3 < 1e-5


def test_2d_uniform_equilibrium_is_preserved() -> None:
    grid = rectangular_tool_composite_grid(
        width_m=0.04, spacing_x_m=0.01, spacing_z_m=0.005
    )
    times = np.array([0.0, 20.0, 40.0])
    result = simulate_cure_2d(
        times,
        np.full_like(times, 333.0),
        grid=grid,
        initial_temperature_K=333.0,
        initial_alpha=0.0,
        maximum_step_s=10.0,
    )
    assert np.max(np.abs(result.temperature_K - 333.0)) < 2.0e-11
    assert result.diagnostics.maximum_relative_global_energy_residual < 1e-11


def test_grid_uses_sourced_longitudinal_transverse_anisotropy() -> None:
    material = PublicCase1Material()
    grid = rectangular_tool_composite_grid(
        width_m=0.04, spacing_x_m=0.01, spacing_z_m=0.005
    )
    assert np.allclose(
        grid.conductivity_x_W_m_K[grid.composite_mask],
        material.composite_longitudinal_k_W_m_K,
    )
    assert np.allclose(
        grid.conductivity_z_W_m_K[grid.composite_mask],
        material.composite_k_W_m_K,
    )
    assert material.composite_longitudinal_k_W_m_K > (
        7.0 * material.composite_k_W_m_K
    )


def test_nonfinite_boundary_input_is_rejected() -> None:
    grid = rectangular_tool_composite_grid(
        width_m=0.04, spacing_x_m=0.01, spacing_z_m=0.005
    )
    with pytest.raises(ValueError, match="finite"):
        simulate_cure_2d(
            np.array([0.0, 10.0]),
            np.array([293.0, 303.0]),
            grid=grid,
            boundaries=RobinBoundaries2D(70.0, np.nan),
        )


def test_interface_and_robin_series_resistances_are_reconstructed() -> None:
    grid = LayeredGrid2D(
        x_m=np.array([0.5]),
        z_m=np.array([0.05, 0.15]),
        control_volume_width_x_m=np.array([1.0]),
        control_volume_width_z_m=np.array([0.1, 0.1]),
        composite_mask=np.array([[False], [True]]),
        density_kg_m3=np.full((2, 1), 1000.0),
        specific_heat_J_kg_K=np.full((2, 1), 1000.0),
        conductivity_x_W_m_K=np.ones((2, 1)),
        conductivity_z_W_m_K=np.array([[2.0], [8.0]]),
        cure_source_J_m3_per_alpha=np.zeros((2, 1)),
    )
    boundaries = RobinBoundaries2D(10.0, 0.0)
    operator = _assemble_operator(grid, boundaries)
    expected_interface_conductance = 1.0 / (0.05 / 2.0 + 0.05 / 8.0)
    expected_lower_robin_conductance = 1.0 / (1.0 / 10.0 + 0.05 / 2.0)
    assert operator.z_face_conductance_W_K[0, 0] == pytest.approx(
        expected_interface_conductance
    )
    assert operator.robin_rhs_coefficient[0, 0] == pytest.approx(
        expected_lower_robin_conductance
    )
    assert operator.lower_robin_conductance_W_K[0] == pytest.approx(
        expected_lower_robin_conductance
    )
    temperature = np.array([[400.0], [300.0]])
    flux_imbalance, temperature_jump = _interface_diagnostics_2d(
        grid, operator, temperature
    )
    assert flux_imbalance < 1.0e-10
    assert temperature_jump < 1.0e-12
    assert (
        _robin_flux_imbalance_2d(
            grid, operator, boundaries, temperature, 293.0
        )
        < 1.0e-12
    )

    bad_interface = replace(
        operator,
        z_face_conductance_W_K=(
            1.1 * operator.z_face_conductance_W_K
        ),
    )
    bad_flux, bad_jump = _interface_diagnostics_2d(
        grid, bad_interface, temperature
    )
    assert bad_flux > 1.0
    assert bad_jump > 1.0

    bad_robin = replace(
        operator,
        lower_robin_conductance_W_K=(
            1.1 * operator.lower_robin_conductance_W_K
        ),
    )
    assert (
        _robin_flux_imbalance_2d(
            grid, bad_robin, boundaries, temperature, 293.0
        )
        > 1.0
    )


def test_extrusion_matches_converged_1d_and_is_laterally_invariant() -> None:
    times = np.arange(0.0, 601.0, 60.0)
    air = np.linspace(293.0, 413.0, times.size)
    grid_1d = public_case1_grid(spacing_m=0.0005)
    result_1d = simulate_cure_1d(
        times,
        air,
        grid=grid_1d,
        boundaries=RobinBoundaries(70.0, 120.0),
        maximum_step_s=2.5,
        maximum_coupling_iterations=12,
    )
    grid_2d = rectangular_tool_composite_grid(
        width_m=0.04, spacing_x_m=0.01, spacing_z_m=0.001
    )
    result_2d = simulate_cure_2d(
        times,
        air,
        grid=grid_2d,
        boundaries=RobinBoundaries2D(70.0, 120.0),
        maximum_step_s=2.5,
        maximum_coupling_iterations=12,
    )
    reference_temperature = np.empty_like(result_2d.temperature_K[:, :, 0])
    reference_alpha = np.empty_like(result_2d.alpha[:, :, 0])
    for index in range(times.size):
        reference_temperature[index] = np.interp(
            result_2d.z_m, result_1d.z_m, result_1d.temperature_K[index]
        )
        reference_alpha[index] = np.interp(
            result_2d.z_m, result_1d.z_m, result_1d.alpha[index]
        )
    temperature_relative_l2 = np.linalg.norm(
        result_2d.temperature_K[:, :, 0] - reference_temperature
    ) / np.linalg.norm(reference_temperature)
    alpha_relative_l2 = np.linalg.norm(
        result_2d.alpha[:, :, 0] - reference_alpha
    ) / np.linalg.norm(reference_alpha)
    assert temperature_relative_l2 < 2e-4
    assert alpha_relative_l2 < 2e-3
    lateral_variance = np.max(np.var(result_2d.temperature_K, axis=-1))
    assert lateral_variance < 1e-20


def test_spatially_varying_htc_creates_finite_lateral_structure() -> None:
    grid = rectangular_tool_composite_grid(
        width_m=0.08, spacing_x_m=0.01, spacing_z_m=0.005
    )
    top_h = 120.0 + 40.0 * np.cos(2.0 * np.pi * grid.x_m / 0.08)
    times = np.arange(0.0, 301.0, 60.0)
    result = simulate_cure_2d(
        times,
        np.linspace(293.0, 433.0, times.size),
        grid=grid,
        boundaries=RobinBoundaries2D(70.0, top_h),
        maximum_step_s=5.0,
        maximum_coupling_iterations=12,
    )
    assert np.all(np.isfinite(result.temperature_K))
    assert np.max(np.ptp(result.temperature_K[-1], axis=-1)) > 1e-3
    assert result.diagnostics.maximum_abs_energy_residual_W_m3 < 1e-4
    assert result.diagnostics.maximum_relative_global_energy_residual < 1e-10
    assert result.diagnostics.maximum_interface_flux_imbalance_W < 1e-9
    assert result.diagnostics.maximum_temperature_interface_jump_K < 1e-10
    assert result.diagnostics.maximum_robin_flux_imbalance_W < 1e-9
    assert result.diagnostics.all_coupling_steps_converged
    assert np.all(np.diff(result.alpha, axis=0) >= -1e-14)
    assert np.all(result.alpha[:, ~grid.composite_mask] == 0.0)


def test_2d_mesh_refinement_decreases_extrusion_error() -> None:
    times = np.arange(0.0, 301.0, 30.0)
    air = np.linspace(293.0, 393.0, times.size)
    reference = simulate_cure_1d(
        times,
        air,
        grid=public_case1_grid(spacing_m=0.0005),
        maximum_step_s=2.5,
        maximum_coupling_iterations=12,
    )
    errors = []
    for spacing in (0.005, 0.0025, 0.001):
        result = simulate_cure_2d(
            times,
            air,
            grid=rectangular_tool_composite_grid(
                width_m=0.02,
                spacing_x_m=0.01,
                spacing_z_m=spacing,
            ),
            maximum_step_s=2.5,
            maximum_coupling_iterations=12,
        )
        reference_temperature = np.empty_like(result.temperature_K[:, :, 0])
        for index, field in enumerate(reference.temperature_K):
            reference_temperature[index] = np.interp(
                result.z_m, reference.z_m, field
            )
        error = np.linalg.norm(
            result.temperature_K[:, :, 0] - reference_temperature
        ) / np.linalg.norm(reference_temperature)
        errors.append(error)
    assert errors[0] > errors[1] > errors[2]
