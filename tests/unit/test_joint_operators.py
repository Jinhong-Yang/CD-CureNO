import pytest
import torch

from cdcureno.models.joint_operators import (
    CausalFactorizedOperator,
    CausalTemporalConv1d,
    SpatialSpectralConv1d,
    build_joint_operator,
    parameter_count,
)


def _input(batch: int = 2, time: int = 17, space: int = 11) -> torch.Tensor:
    torch.manual_seed(20260723)
    values = torch.rand(batch, time, space, 7)
    values[..., 4] = 1.0
    values[:, :, :3, 4] = 0.0
    values[..., 6] = 0.05 * values[..., 4]
    return values


@pytest.mark.parametrize(
    "name", ["noncausal_fno2d", "factorized_fno", "causal_factorized"]
)
def test_joint_operator_families_have_canonical_outputs_and_physical_alpha(
    name: str,
) -> None:
    model = build_joint_operator(
        name,  # type: ignore[arg-type]
        width=8,
        depth=2,
        modes_time=5,
        modes_space=4,
    )
    inputs = _input()
    outputs = model(inputs)
    assert outputs["temperature"].shape == (2, 17, 11)
    assert outputs["temperature_residual"].shape == (2, 17, 11)
    assert outputs["alpha"].shape == (2, 17, 11)
    assert outputs["cure_rate"].shape == (2, 17, 11)
    assert torch.all(outputs["cure_rate"] >= 0)
    assert torch.all(outputs["alpha"] >= 0)
    assert torch.all(outputs["alpha"] <= 1)
    assert torch.all(outputs["alpha"][:, :, :3] == 0)
    assert torch.all(outputs["alpha"][:, 1:, 3:] >= outputs["alpha"][:, :-1, 3:])
    assert parameter_count(model) > 0


def test_spatial_spectral_layer_preserves_axis_order_and_truncates_modes() -> None:
    layer = SpatialSpectralConv1d(channels=4, modes=64)
    values = torch.randn(2, 13, 9, 4)
    output = layer(values)
    assert output.shape == values.shape
    values_with_other_time_changed = values.clone()
    values_with_other_time_changed[:, 7] += 10.0
    changed_output = layer(values_with_other_time_changed)
    torch.testing.assert_close(output[:, 6], changed_output[:, 6])


def test_causal_temporal_layer_passes_future_perturbation_test() -> None:
    torch.manual_seed(9)
    layer = CausalTemporalConv1d(channels=5, kernel_size=3, dilation=4)
    values = torch.randn(2, 19, 7, 5)
    perturbed = values.clone()
    perturbed[:, 11:] += torch.randn_like(perturbed[:, 11:]) * 20.0
    with torch.no_grad():
        original = layer(values)
        changed = layer(perturbed)
    torch.testing.assert_close(original[:, :11], changed[:, :11], rtol=0, atol=0)


def test_complete_causal_operator_passes_future_perturbation_test() -> None:
    torch.manual_seed(11)
    model = CausalFactorizedOperator(
        input_channels=7,
        width=8,
        depth=3,
        modes_space=4,
    ).eval()
    values = _input(batch=1, time=23, space=9)
    perturbed = values.clone()
    perturbed[:, 13:, :, :2] += 5.0
    with torch.no_grad():
        original = model(values)
        changed = model(perturbed)
    for field in ("temperature", "temperature_residual", "alpha", "cure_rate"):
        torch.testing.assert_close(
            original[field][:, :13],
            changed[field][:, :13],
            rtol=0,
            atol=2e-6,
        )


def test_default_causal_receptive_field_covers_the_full_public_trajectory() -> None:
    model = CausalFactorizedOperator(width=4, modes_space=3)
    receptive_field = 1 + sum(
        block.temporal.left_padding for block in model.blocks
    )
    assert receptive_field >= 223
