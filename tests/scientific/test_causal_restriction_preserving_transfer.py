from __future__ import annotations

import torch

from cdcureno.data.source_1d import INPUT_CHANNELS
from cdcureno.models.causal_checkpoint_inflation import (
    inflate_causal_checkpoint_payload,
    inspect_causal_target_architecture,
    verify_causal_models_on_input,
    verify_target_future_invariance,
)
from cdcureno.models.joint_operators import CausalFactorizedOperator
from cdcureno.models.target_operators import P5_TARGET_ONLY_CHANNEL_NAMES


def _inflated_models():
    torch.manual_seed(1827)
    source = CausalFactorizedOperator(
        input_channels=len(INPUT_CHANNELS),
        width=5,
        depth=3,
        modes_space=4,
    )
    checkpoint = {
        "model_family": "causal_factorized",
        "model_spec": {
            "family": "causal_factorized",
            "input_channels": len(INPUT_CHANNELS),
            "channel_names": list(INPUT_CHANNELS),
            "width": 5,
            "depth": 3,
            "modes_space": 4,
            "temporal_kernel_size": 3,
            "temporal_dilations": [1, 2, 4],
            "temporal_receptive_field": 15,
            "structurally_causal": True,
        },
        "model": source.state_dict(),
        "channel_names": INPUT_CHANNELS,
    }
    config = {
        "schema_version": 1,
        "family": "causal_axis_factorized_2d",
        "source_family": "causal_factorized",
        "expected_source_checkpoint_sha256": "a" * 64,
        "inflation_seed": 20260726,
        "temporal_family": "causal_dilated_convolution",
        "causal": True,
        "structurally_causal": True,
        "axis_order": ["batch", "time", "z", "x", "channel"],
        "source_channel_names": list(INPUT_CHANNELS),
        "new_channel_names": list(P5_TARGET_ONLY_CHANNEL_NAMES),
        "width": 5,
        "depth": 3,
        "modes_z": 4,
        "modes_x": 4,
        "lateral_rank": 2,
        "temporal_kernel_size": 3,
        "temporal_dilations": [1, 2, 4],
    }
    target, _, _ = inflate_causal_checkpoint_payload(
        checkpoint,
        config,
        seed=20260726,
        source_checkpoint_sha256="a" * 64,
        target_config_sha256="b" * 64,
    )
    return source.eval(), target.eval()


def _source_inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7001)
    values = torch.rand(1, 17, 9, len(INPUT_CHANNELS), generator=generator)
    values[..., INPUT_CHANNELS.index("time_normalized")] = torch.linspace(
        0.0, 1.0, 17
    )[None, :, None]
    values[
        ..., INPUT_CHANNELS.index("through_thickness_position_normalized")
    ] = torch.linspace(0.0, 1.0, 9)[None, None, :]
    mask = torch.ones(1, 9)
    mask[:, :2] = 0.0
    values[..., INPUT_CHANNELS.index("composite_mask")] = mask[:, None, :]
    values[..., INPUT_CHANNELS.index("initial_degree_of_cure")] = (
        0.03 * mask[:, None, :]
    )
    return values


def test_trained_lateral_branch_cannot_change_the_homogeneous_kx_zero_anchor() -> None:
    source, target = _inflated_models()
    with torch.no_grad():
        for block in target.blocks:
            block.lateral.output_factor.normal_(mean=0.0, std=0.05)
    result = verify_causal_models_on_input(
        source,
        target,
        _source_inputs(),
        seed=91,
        nx_values=(1, 2, 7, 40),
    )

    assert result["passed"] is True
    for nx in ("2", "7", "40"):
        for metrics in result["per_nx"][nx]["fields"].values():
            assert metrics["maximum_lateral_range"] <= metrics["tolerance"]


def test_causal_target_with_live_geometry_and_lateral_paths_has_no_future_leakage() -> None:
    _, target = _inflated_models()
    with torch.no_grad():
        target.lift.geometry.weight.normal_(mean=0.0, std=0.1)
        for block in target.blocks:
            block.lateral.output_factor.normal_(mean=0.0, std=0.1)
    architecture = inspect_causal_target_architecture(target)
    causality = verify_target_future_invariance(
        target,
        _source_inputs(),
        seed=112,
        nx=9,
        tolerance=1.0e-7,
    )

    assert architecture["passed"] is True
    assert architecture["temporal_fft_module_count"] == 0
    assert architecture["causal_temporal_module_count"] == 3
    assert causality["passed"] is True
    assert causality["maximum_prefix_abs_difference"] == 0.0
