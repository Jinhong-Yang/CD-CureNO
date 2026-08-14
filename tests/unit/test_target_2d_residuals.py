from __future__ import annotations

import pytest
import torch

from cdcureno.physics.as4_8552 import cure_rate_per_s
from cdcureno.physics.target_2d_residuals import (
    Target2DPhysicsContext,
    cure_rate_per_s_torch,
    cure_kinetics_interval_residual,
    target_2d_physics_residuals,
    trapezoidal_energy_residual,
)


DTYPE = torch.float64


def _coordinates(
    time_count: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.arange(time_count, dtype=DTYPE) * 120.0,
        torch.tensor([0.0005, 0.0015], dtype=DTYPE),
        torch.tensor([0.0005, 0.0015], dtype=DTYPE),
    )


def _context(
    *,
    air_temperature_K: float | torch.Tensor = 300.0,
    reaction_scale: float = 1.0,
    initial_temperature_K: float = 300.0,
    boundary_h: float = 0.0,
) -> Target2DPhysicsContext:
    return Target2DPhysicsContext(
        density_kg_m3=2.0,
        specific_heat_J_kg_K=5.0,
        conductivity_x_W_m_K=2.0,
        conductivity_z_W_m_K=3.0,
        base_cure_source_J_m3_per_alpha=100.0,
        reaction_enthalpy_scale=reaction_scale,
        air_temperature_K=air_temperature_K,
        lower_h_W_m2_K=boundary_h,
        upper_h_W_m2_K=boundary_h,
        left_h_W_m2_K=boundary_h,
        right_h_W_m2_K=boundary_h,
        initial_temperature_K=initial_temperature_K,
    )


def test_uniform_equilibrium_has_zero_energy_kinetics_and_initial_residual() -> None:
    time, z, x = _coordinates()
    temperature = torch.full((1, 2, 2, 2), 300.0, dtype=DTYPE)
    alpha = torch.zeros_like(temperature)
    mask = torch.tensor([[0, 0], [1, 1]], dtype=torch.bool)
    result = target_2d_physics_residuals(
        temperature,
        alpha,
        time,
        z,
        x,
        mask,
        _context(boundary_h=25.0),
    )

    assert torch.count_nonzero(result.energy_raw_J_m3) == 0
    assert torch.count_nonzero(result.energy_dimensionless) == 0
    assert torch.count_nonzero(result.cure_kinetics_dimensionless) == 0
    assert torch.count_nonzero(
        result.temperature_initial_dimensionless
    ) == 0
    assert result.energy_mean_square.item() == 0.0
    assert result.cure_kinetics_mean_square.item() == 0.0
    assert result.temperature_initial_mean_square.item() == 0.0


def test_torch_cure_law_matches_sourced_numpy_relation() -> None:
    temperature = torch.tensor(
        [293.0, 350.0, 425.0, 475.0], dtype=DTYPE
    )
    alpha = torch.tensor([0.0, 0.05, 0.4, 1.0], dtype=DTYPE)
    actual = cure_rate_per_s_torch(temperature, alpha)
    expected = torch.from_numpy(
        cure_rate_per_s(temperature.numpy(), alpha.numpy())
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-15, rtol=1.0e-12)


def test_reaction_scale_balances_storage_in_interval_integral() -> None:
    time, z, x = _coordinates()
    temperature = torch.empty((1, 2, 2, 2), dtype=DTYPE)
    temperature[:, 0] = 300.0
    temperature[:, 1] = 302.0
    alpha = torch.empty_like(temperature)
    alpha[:, 0] = 0.0
    alpha[:, 1] = 0.1
    mask = torch.ones((2, 2), dtype=torch.bool)

    raw, normalized = trapezoidal_energy_residual(
        temperature,
        alpha,
        time,
        z,
        x,
        mask,
        _context(reaction_scale=2.0),
    )
    # rho*Cp*dT = 2*5*2 = 20 J/m3 and
    # q_base*scale*dalpha = 100*2*0.1 = 20 J/m3.
    torch.testing.assert_close(raw, torch.zeros_like(raw), atol=1.0e-12, rtol=0)
    torch.testing.assert_close(
        normalized, torch.zeros_like(normalized), atol=1.0e-12, rtol=0
    )

    unbalanced_raw, _ = trapezoidal_energy_residual(
        temperature,
        alpha,
        time,
        z,
        x,
        mask,
        _context(reaction_scale=1.0),
    )
    torch.testing.assert_close(
        unbalanced_raw,
        torch.full_like(unbalanced_raw, 10.0),
        atol=1.0e-12,
        rtol=0,
    )


def test_local_temperature_perturbation_worsens_energy_consistency() -> None:
    time, z, x = _coordinates()
    temperature = torch.empty((1, 2, 2, 2), dtype=DTYPE)
    temperature[:, 0] = 300.0
    temperature[:, 1] = 302.0
    alpha = torch.empty_like(temperature)
    alpha[:, 0] = 0.0
    alpha[:, 1] = 0.1
    mask = torch.ones((2, 2), dtype=torch.bool)
    context = _context(reaction_scale=2.0)
    balanced = target_2d_physics_residuals(
        temperature, alpha, time, z, x, mask, context
    )

    perturbed = temperature.clone()
    perturbed[0, 1, 0, 0] += 1.0
    changed = target_2d_physics_residuals(
        perturbed, alpha, time, z, x, mask, context
    )

    assert balanced.energy_mean_square.item() <= 1.0e-28
    assert changed.energy_mean_square > balanced.energy_mean_square
    assert changed.energy_mean_square.item() > 0.0


def test_robin_series_resistance_sets_boundary_heat_rate() -> None:
    time, z, x = _coordinates()
    temperature = torch.full((1, 2, 2, 2), 310.0, dtype=DTYPE)
    alpha = torch.zeros_like(temperature)
    mask = torch.ones((2, 2), dtype=torch.bool)
    h = 20.0
    raw, _ = trapezoidal_energy_residual(
        temperature,
        alpha,
        time,
        z,
        x,
        mask,
        _context(
            air_temperature_K=300.0,
            initial_temperature_K=310.0,
            boundary_h=h,
        ),
    )
    dx = dz = 0.001
    volume = dx * dz
    conductance_z = dx / (1.0 / h + 0.5 * dz / 3.0)
    conductance_x = dz / (1.0 / h + 0.5 * dx / 2.0)
    expected = (
        120.0
        * (conductance_z + conductance_x)
        * (310.0 - 300.0)
        / volume
    )
    torch.testing.assert_close(
        raw,
        torch.full_like(raw, expected),
        atol=1.0e-8,
        rtol=1.0e-12,
    )


def test_harmonic_internal_face_rates_are_globally_conservative() -> None:
    time, z, x = _coordinates()
    first = torch.tensor(
        [[[300.0, 315.0], [305.0, 310.0]]], dtype=DTYPE
    )
    temperature = torch.stack((first, first), dim=1)
    alpha = torch.zeros_like(temperature)
    mask = torch.ones((2, 2), dtype=torch.bool)
    context = Target2DPhysicsContext(
        density_kg_m3=2.0,
        specific_heat_J_kg_K=5.0,
        conductivity_x_W_m_K=torch.tensor(
            [[1.0, 4.0], [2.0, 8.0]], dtype=DTYPE
        ),
        conductivity_z_W_m_K=torch.tensor(
            [[3.0, 6.0], [9.0, 12.0]], dtype=DTYPE
        ),
        base_cure_source_J_m3_per_alpha=100.0,
        air_temperature_K=300.0,
        initial_temperature_K=first[0],
    )
    raw, _ = trapezoidal_energy_residual(
        temperature, alpha, time, z, x, mask, context
    )

    # Every interior face contributes equal and opposite heat rates to its two
    # adjacent cells. The uniform cell volumes make their raw-density sum zero.
    torch.testing.assert_close(
        torch.sum(raw),
        torch.zeros((), dtype=DTYPE),
        atol=1.0e-7,
        rtol=0,
    )
    assert torch.count_nonzero(raw) > 0


def test_all_losses_backpropagate_finite_gradients() -> None:
    time, z, x = _coordinates(time_count=3)
    temperature_values = torch.tensor(
        [
            [
                [[300.0, 300.2], [300.4, 300.6]],
                [[302.0, 302.1], [302.4, 302.7]],
                [[304.0, 304.3], [304.5, 304.9]],
            ]
        ],
        dtype=DTYPE,
    )
    alpha_values = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.050, 0.052]],
                [[0.0, 0.0], [0.055, 0.058]],
                [[0.0, 0.0], [0.061, 0.064]],
            ]
        ],
        dtype=DTYPE,
    )
    temperature = temperature_values.clone().requires_grad_()
    alpha = alpha_values.clone().requires_grad_()
    mask = torch.tensor([[0, 0], [1, 1]], dtype=torch.bool)
    context = _context(
        air_temperature_K=torch.tensor([300.0, 302.0, 304.0], dtype=DTYPE),
        reaction_scale=1.1,
        initial_temperature_K=300.0,
        boundary_h=20.0,
    )
    residuals = target_2d_physics_residuals(
        temperature,
        alpha,
        time,
        z,
        x,
        mask,
        context,
    )
    loss = (
        residuals.energy_mean_square
        + residuals.cure_kinetics_mean_square
        + residuals.temperature_initial_mean_square
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert temperature.grad is not None
    assert alpha.grad is not None
    assert torch.all(torch.isfinite(temperature.grad))
    assert torch.all(torch.isfinite(alpha.grad))
    assert torch.count_nonzero(temperature.grad) > 0
    assert torch.count_nonzero(alpha.grad) > 0


def test_shape_coordinate_and_finite_validation_fail_closed() -> None:
    time, z, x = _coordinates()
    temperature = torch.full((1, 2, 2, 2), 300.0, dtype=DTYPE)
    alpha = torch.zeros_like(temperature)
    mask = torch.ones((2, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match=r"\[B,T,Z,X\]"):
        trapezoidal_energy_residual(
            temperature[:, :, :, 0],
            alpha[:, :, :, 0],
            time,
            z,
            x,
            mask,
            _context(),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        cure_kinetics_interval_residual(
            temperature,
            alpha,
            torch.tensor([0.0, 0.0], dtype=DTYPE),
            mask,
        )
    bad_temperature = temperature.clone()
    bad_temperature[0, 1, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        trapezoidal_energy_residual(
            bad_temperature,
            alpha,
            time,
            z,
            x,
            mask,
            _context(),
        )
