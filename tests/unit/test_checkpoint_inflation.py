from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from cdcureno.data.source_1d import INPUT_CHANNELS
from cdcureno.models.checkpoint_inflation import (
    SHARED_LIFT_MAPPING,
    inflate_checkpoint_payload,
    load_inflated_target,
    verify_inflated_checkpoint_integrity,
)
from cdcureno.models.joint_operators import FactorizedFNO
from cdcureno.models.target_operators import (
    P5_SOURCE_CHANNEL_NAMES,
    P5_TARGET_CHANNEL_NAMES,
    P5_TARGET_ONLY_CHANNEL_NAMES,
)


NEW_CHANNELS = P5_TARGET_ONLY_CHANNEL_NAMES


def _source_checkpoint() -> dict[str, object]:
    torch.manual_seed(3101)
    model = FactorizedFNO(
        input_channels=len(INPUT_CHANNELS),
        width=8,
        depth=2,
        modes_time=5,
        modes_space=4,
    )
    return {
        "model": model.state_dict(),
        "epoch": 17,
        "validation_objective": 0.0123,
        "normalization": {"scope": "training_cases_only"},
        "channel_names": INPUT_CHANNELS,
    }


def _target_config() -> dict[str, object]:
    return {
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
        "modes_x": 3,
        "lateral_rank": 2,
    }


def test_model_channel_contract_matches_source_and_target_data_contracts() -> None:
    from cdcureno.data.target_2d import (
        TARGET_ADAPTER_CHANNELS,
        TARGET_INPUT_CHANNELS,
    )

    assert P5_SOURCE_CHANNEL_NAMES == INPUT_CHANNELS
    assert P5_TARGET_ONLY_CHANNEL_NAMES == TARGET_ADAPTER_CHANNELS
    assert P5_TARGET_CHANNEL_NAMES == TARGET_INPUT_CHANNELS


def test_checkpoint_inflation_maps_every_source_tensor_exactly() -> None:
    checkpoint = _source_checkpoint()
    target, payload, report = inflate_checkpoint_payload(
        checkpoint, _target_config(), seed=44
    )
    source_state = checkpoint["model"]
    assert isinstance(source_state, dict)
    target_state = target.state_dict()
    assert report["tensor_mapping"]["copied_count"] == len(source_state)
    assert report["tensor_mapping"]["all_copied_tensors_bitwise_equal"]
    for source_key, source_tensor in source_state.items():
        target_key = SHARED_LIFT_MAPPING.get(source_key, source_key)
        assert torch.equal(source_tensor, target_state[target_key])
    assert torch.equal(
        target.lift.combined_weight()[:, : len(INPUT_CHANNELS)],
        source_state["lift.0.weight"],
    )
    assert torch.count_nonzero(target.lift.geometry.weight) == 0
    for block in target.blocks:
        assert torch.count_nonzero(block.lateral.input_factor) > 0
        assert torch.count_nonzero(block.lateral.output_factor) == 0
        assert torch.count_nonzero(block.lateral.effective_weight()) == 0
    assert payload["channel_names"] == (*INPUT_CHANNELS, *NEW_CHANNELS)
    assert report["verification"]["uses_labels"] is False
    assert report["verification"]["passed"]


def test_checkpoint_inflation_new_tensor_initialization_is_deterministic() -> None:
    first, _, first_report = inflate_checkpoint_payload(
        _source_checkpoint(), _target_config(), seed=991
    )
    torch.manual_seed(999_999)
    second, _, second_report = inflate_checkpoint_payload(
        _source_checkpoint(), _target_config(), seed=991
    )
    for first_block, second_block in zip(
        first.blocks, second.blocks, strict=True
    ):
        assert torch.equal(
            first_block.lateral.input_factor,
            second_block.lateral.input_factor,
        )
        assert torch.equal(
            first_block.lateral.output_factor,
            second_block.lateral.output_factor,
        )
    first_initialized = first_report["tensor_mapping"]["initialized"]
    second_initialized = second_report["tensor_mapping"]["initialized"]
    assert [
        (item["target"], item["sha256"]) for item in first_initialized
    ] == [
        (item["target"], item["sha256"]) for item in second_initialized
    ]


def test_checkpoint_inflation_rejects_temporal_family_or_causality_mismatch() -> None:
    causal = deepcopy(_target_config())
    causal["causal"] = True
    with pytest.raises(ValueError, match="cannot be advertised"):
        inflate_checkpoint_payload(_source_checkpoint(), causal)
    other_family = deepcopy(_target_config())
    other_family["temporal_family"] = "causal_convolution"
    with pytest.raises(ValueError, match="temporal families differ"):
        inflate_checkpoint_payload(_source_checkpoint(), other_family)


def test_checkpoint_inflation_enforces_a_pinned_source_file_digest() -> None:
    config = deepcopy(_target_config())
    config["expected_source_checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="must provide"):
        inflate_checkpoint_payload(_source_checkpoint(), config)
    with pytest.raises(ValueError, match="differs"):
        inflate_checkpoint_payload(
            _source_checkpoint(),
            config,
            source_checkpoint_sha256="b" * 64,
        )
    target, _, report = inflate_checkpoint_payload(
        _source_checkpoint(),
        config,
        source_checkpoint_sha256="a" * 64,
    )
    assert target is not None
    assert report["target"]["expected_source_checkpoint_sha256"] == "a" * 64


def test_checkpoint_inflation_enforces_a_pinned_seed() -> None:
    config = deepcopy(_target_config())
    config["inflation_seed"] = 44
    inflate_checkpoint_payload(_source_checkpoint(), config, seed=44)
    with pytest.raises(ValueError, match="differs from the target contract"):
        inflate_checkpoint_payload(_source_checkpoint(), config, seed=45)


def test_inflation_integrity_rejects_a_mutated_lateral_output_factor() -> None:
    source_sha = "a" * 64
    config_sha = "b" * 64
    config = deepcopy(_target_config())
    config["expected_source_checkpoint_sha256"] = source_sha
    config["inflation_seed"] = 44
    source = _source_checkpoint()
    _, payload, report = inflate_checkpoint_payload(
        source,
        config,
        seed=44,
        source_checkpoint_sha256=source_sha,
    )
    report["source"]["checkpoint_sha256"] = source_sha
    report["target"]["config_sha256"] = config_sha
    payload["inflation_report"] = report
    verified = verify_inflated_checkpoint_integrity(
        source,
        payload,
        config,
        source_checkpoint_sha256=source_sha,
        target_config_sha256=config_sha,
    )
    assert verified["passed"]
    payload["model"]["blocks.0.lateral.output_factor"].fill_(0.25)
    with pytest.raises(
        ValueError, match="all_state_tensors_bitwise_match_recreation"
    ):
        verify_inflated_checkpoint_integrity(
            source,
            payload,
            config,
            source_checkpoint_sha256=source_sha,
            target_config_sha256=config_sha,
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("width", 9, "differs from source"),
        ("depth", 3, "differs from source"),
        ("modes_time", 6, "differs from source"),
        ("modes_z", 5, "differs from source"),
    ],
)
def test_checkpoint_inflation_rejects_incompatible_copied_shapes(
    key: str, value: int, message: str
) -> None:
    config = deepcopy(_target_config())
    config[key] = value
    with pytest.raises(ValueError, match=message):
        inflate_checkpoint_payload(_source_checkpoint(), config)


def test_transfer_stages_never_modify_shared_lift_during_t0() -> None:
    target, _, _ = inflate_checkpoint_payload(
        _source_checkpoint(), _target_config()
    )
    summary = target.set_transfer_stage("T0")
    assert summary["trainable_parameters"] > 0
    assert not target.lift.shared.weight.requires_grad
    assert target.lift.geometry.weight.requires_grad
    assert all(
        block.lateral.output_factor.requires_grad for block in target.blocks
    )
    target.set_transfer_stage("T1")
    assert target.head.temperature_residual[0].weight.requires_grad
    assert not target.blocks[0].spatial.weight.requires_grad
    target.set_transfer_stage("T2")
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_saved_reloaded_inflated_checkpoint_is_equivalent(
    tmp_path: Path,
) -> None:
    target, payload, _ = inflate_checkpoint_payload(
        _source_checkpoint(), _target_config(), seed=81
    )
    checkpoint_path = tmp_path / "inflated.pt"
    torch.save(payload, checkpoint_path)
    reloaded, reloaded_payload = load_inflated_target(checkpoint_path)
    for key, tensor in target.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[key])
    generator = torch.Generator().manual_seed(787)
    values = torch.rand(
        1,
        13,
        9,
        7,
        len(INPUT_CHANNELS) + len(NEW_CHANNELS),
        generator=generator,
    )
    values[..., INPUT_CHANNELS.index("composite_mask")] = 1.0
    with torch.no_grad():
        expected = target.eval()(values)
        actual = reloaded.eval()(values)
    for field in ("field", "temperature_residual", "cure_rate"):
        assert torch.equal(expected[field], actual[field])
    assert reloaded_payload["inflation_report"]["passed"]
