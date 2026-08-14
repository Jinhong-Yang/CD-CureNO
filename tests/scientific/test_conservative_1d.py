import numpy as np

from cdcureno.solvers import (
    LayeredGrid1D,
    RobinBoundaries,
    public_case1_grid,
    simulate_cure_1d,
)


def test_public_grid_has_harmonic_interface_and_expected_masks() -> None:
    grid = public_case1_grid()
    assert grid.z_m.shape == (51,)
    assert np.array_equal(np.flatnonzero(~grid.composite_mask), np.arange(21))
    assert np.array_equal(np.flatnonzero(grid.composite_mask), np.arange(21, 51))
    assert np.isclose(np.sum(grid.control_volume_width_m), 0.05)


def test_uniform_equilibrium_is_preserved_without_reaction() -> None:
    base = public_case1_grid()
    no_source = LayeredGrid1D(
        z_m=base.z_m,
        control_volume_width_m=base.control_volume_width_m,
        composite_mask=np.zeros_like(base.composite_mask),
        density_kg_m3=base.density_kg_m3,
        specific_heat_J_kg_K=base.specific_heat_J_kg_K,
        conductivity_W_m_K=base.conductivity_W_m_K,
        cure_source_J_m3_per_alpha=np.zeros_like(
            base.cure_source_J_m3_per_alpha
        ),
    )
    times = np.array([0.0, 60.0, 120.0])
    result = simulate_cure_1d(
        times,
        np.full_like(times, 333.0),
        grid=no_source,
        initial_temperature_K=333.0,
        maximum_step_s=10.0,
    )
    assert np.max(np.abs(result.temperature_K - 333.0)) < 1.0e-11
    assert np.all(result.alpha == 0.0)
    assert result.diagnostics.maximum_relative_global_energy_residual < 1e-7


def test_conservative_energy_residual_is_at_roundoff_scale() -> None:
    times = np.arange(0.0, 601.0, 60.0)
    air = np.linspace(293.0, 393.0, times.size)
    result = simulate_cure_1d(
        times,
        air,
        maximum_step_s=5.0,
        maximum_coupling_iterations=12,
    )
    assert result.diagnostics.maximum_abs_energy_residual_W_m3 < 2.0e-5
    assert result.diagnostics.maximum_relative_global_energy_residual < 5e-11
    assert np.all(np.diff(result.alpha[:, 21:], axis=0) >= -1.0e-14)
    assert np.all(result.alpha[:, :21] == 0.0)


def test_insulated_boundary_support_via_zero_robin_coefficient() -> None:
    times = np.array([0.0, 60.0])
    result = simulate_cure_1d(
        times,
        np.array([293.0, 293.0]),
        boundaries=RobinBoundaries(0.0, 0.0),
        maximum_step_s=5.0,
    )
    assert np.all(np.isfinite(result.temperature_K))
