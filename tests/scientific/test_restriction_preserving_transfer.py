from __future__ import annotations

import torch

from cdcureno.data.source_1d import INPUT_CHANNELS
from cdcureno.models.checkpoint_inflation import inflate_checkpoint_payload
from cdcureno.models.joint_operators import FactorizedFNO
from cdcureno.models.target_operators import (
    LateralSpectralAdapter,
    P5_TARGET_ONLY_CHANNEL_NAMES,
)


NEW_CHANNELS = P5_TARGET_ONLY_CHANNEL_NAMES


def _models() -> tuple[FactorizedFNO, torch.nn.Module]:
    torch.manual_seed(4102)
    source = FactorizedFNO(
        input_channels=len(INPUT_CHANNELS),
        width=8,
        depth=2,
        modes_time=5,
        modes_space=4,
    ).eval()
    checkpoint = {
        "model": source.state_dict(),
        "channel_names": INPUT_CHANNELS,
    }
    config = {
        "schema_version": 1,
        "family": "axis_factorized_2d",
        "source_family": "factorized_fno",
        "temporal_family": "spectral_noncausal",
        "causal": False,
        "axis_order": ["batch", "time", "z", "x", "channel"],
        "source_channel_names": list(INPUT_CHANNELS),
        "new_channel_names": list(NEW_CHANNELS),
        "width": 8,
        "depth": 2,
        "modes_time": 5,
        "modes_z": 4,
        "modes_x": 4,
        "lateral_rank": 2,
    }
    target, _, _ = inflate_checkpoint_payload(
        checkpoint, config, seed=20260726
    )
    return source, target.eval()


def _source_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1789)
    values = torch.rand(2, 19, 11, len(INPUT_CHANNELS), generator=generator)
    names = list(INPUT_CHANNELS)
    values[..., names.index("time_normalized")] = torch.linspace(0, 1, 19)[
        None, :, None
    ]
    values[
        ..., names.index("through_thickness_position_normalized")
    ] = torch.linspace(0, 1, 11)[None, None, :]
    mask = torch.ones(2, 11)
    mask[:, :3] = 0
    values[..., names.index("composite_mask")] = mask[:, None]
    initial_alpha = 0.04 * torch.rand(2, 11, generator=generator) * mask
    values[..., names.index("initial_degree_of_cure")] = initial_alpha[:, None]
    return values


def test_inflated_target_exactly_preserves_nx_one_source_mapping() -> None:
    source, target = _models()
    inputs = _source_input()
    geometry = torch.randn(
        *inputs.shape[:-1], 1, len(NEW_CHANNELS)
    )
    target_inputs = torch.cat((inputs[:, :, :, None, :], geometry), dim=-1)
    with torch.no_grad():
        expected = source(inputs)
        actual = target(target_inputs)
    for field in ("temperature", "temperature_residual", "alpha", "cure_rate"):
        assert torch.equal(actual[field].squeeze(3), expected[field])
    assert actual["field"].shape == (2, 19, 11, 1, 2)


def test_inflated_target_preserves_source_on_multiple_lateral_resolutions() -> None:
    source, target = _models()
    inputs = _source_input()
    with torch.no_grad():
        expected = source(inputs)
        for nx in (2, 7, 40):
            shared = inputs[:, :, :, None, :].expand(-1, -1, -1, nx, -1)
            # Arbitrary target-only features must have zero contribution at
            # inflation. F0 data later use exact zeros for these six channels.
            generator = torch.Generator().manual_seed(9000 + nx)
            geometry = torch.randn(
                *shared.shape[:-1],
                len(NEW_CHANNELS),
                generator=generator,
            )
            actual = target(torch.cat((shared, geometry), dim=-1))
            tolerances = {
                "temperature": 1.0e-6,
                "temperature_residual": 1.0e-6,
                "alpha": 1.0e-6,
                "cure_rate": 1.0e-5,
            }
            for field, tolerance in tolerances.items():
                reference = expected[field][:, :, :, None].expand(
                    -1, -1, -1, nx
                )
                torch.testing.assert_close(
                    actual[field], reference, rtol=0, atol=tolerance
                )
                lateral_range = torch.amax(actual[field], dim=3) - torch.amin(
                    actual[field], dim=3
                )
                assert float(torch.max(lateral_range)) <= tolerance


def test_strided_target_input_preserves_nx_one_source_mapping_bitwise() -> None:
    source, target = _models()
    inputs = _source_input()
    geometry = torch.randn(*inputs.shape[:-1], 1, len(NEW_CHANNELS))
    packed = torch.cat((inputs.unsqueeze(3), geometry), dim=-1)
    storage = torch.empty(*packed.shape[:-1], 2 * packed.shape[-1])
    strided = storage[..., ::2]
    strided.copy_(packed)
    assert not strided.is_contiguous()
    assert inputs.is_contiguous()
    with torch.no_grad():
        expected = source(inputs)
        actual = target(strided)
    for field in ("temperature", "temperature_residual", "alpha", "cure_rate"):
        assert torch.equal(actual[field].squeeze(3), expected[field])


def test_lateral_adapter_is_high_pass_and_has_a_live_zero_residual_init() -> None:
    torch.manual_seed(92)
    adapter = LateralSpectralAdapter(
        channels=5,
        modes=4,
        rank=2,
        initialization_seed=12,
    )
    assert torch.count_nonzero(adapter.input_factor) > 0
    assert torch.count_nonzero(adapter.output_factor) == 0
    varying = torch.randn(1, 3, 4, 9, 5)
    target = torch.randn_like(varying)
    loss = torch.mean((adapter(varying) - target) ** 2)
    loss.backward()
    assert adapter.output_factor.grad is not None
    assert torch.count_nonzero(adapter.output_factor.grad) > 0
    assert adapter.input_factor.grad is not None
    assert torch.count_nonzero(adapter.input_factor.grad) == 0
    with torch.no_grad():
        adapter.output_factor.normal_()
        constant_x = torch.randn(1, 3, 4, 1, 5).expand(-1, -1, -1, 9, -1)
        high_pass_output = adapter(constant_x)
    torch.testing.assert_close(
        high_pass_output,
        torch.zeros_like(high_pass_output),
        rtol=0,
        atol=2.0e-6,
    )


def test_lateral_adapter_preserves_time_z_x_axis_order() -> None:
    adapter = LateralSpectralAdapter(
        channels=3,
        modes=3,
        rank=2,
        initialization_seed=5,
    )
    with torch.no_grad():
        adapter.output_factor.normal_()
    values = torch.zeros(1, 3, 4, 7, 3)
    values[:, 1, 2, 3, :] = 1.0
    output = adapter(values)
    assert torch.count_nonzero(output[:, 1, 2]) > 0
    assert torch.count_nonzero(output[:, 0]) == 0
    assert torch.count_nonzero(output[:, 2]) == 0
    assert torch.count_nonzero(output[:, 1, :2]) == 0
    assert torch.count_nonzero(output[:, 1, 3:]) == 0
