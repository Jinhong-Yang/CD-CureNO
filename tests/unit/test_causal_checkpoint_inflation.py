from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdcureno.data.source_1d import INPUT_CHANNELS
from cdcureno.models.causal_checkpoint_inflation import (
    CAUSAL_SHARED_LIFT_MAPPING,
    inflate_causal_checkpoint_payload,
    load_inflated_causal_target,
    tensor_sha256,
    verify_causal_checkpoint_integrity,
)
from cdcureno.models.joint_operators import CausalFactorizedOperator
from cdcureno.models.target_operators import P5_TARGET_ONLY_CHANNEL_NAMES


SOURCE_SHA = "a" * 64
CONFIG_SHA = "b" * 64


def _source_checkpoint() -> dict[str, object]:
    torch.manual_seed(9021)
    model = CausalFactorizedOperator(
        input_channels=len(INPUT_CHANNELS),
        width=6,
        depth=2,
        modes_space=4,
    )
    return {
        "schema_version": 1,
        "model_family": "causal_factorized",
        "model_spec": {
            "family": "causal_factorized",
            "input_channels": len(INPUT_CHANNELS),
            "channel_names": list(INPUT_CHANNELS),
            "width": 6,
            "depth": 2,
            "modes_space": 4,
            "temporal_kernel_size": 3,
            "temporal_dilations": [1, 2],
            "temporal_receptive_field": 7,
            "structurally_causal": True,
        },
        "model": model.state_dict(),
        "epoch": 11,
        "best_validation": 0.018,
        "normalization": {"scope": "training_cases_only"},
        "channel_names": INPUT_CHANNELS,
    }


def _target_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "family": "causal_axis_factorized_2d",
        "source_family": "causal_factorized",
        "expected_source_checkpoint_sha256": SOURCE_SHA,
        "inflation_seed": 57,
        "temporal_family": "causal_dilated_convolution",
        "causal": True,
        "structurally_causal": True,
        "axis_order": ["batch", "time", "z", "x", "channel"],
        "source_channel_names": list(INPUT_CHANNELS),
        "new_channel_names": list(P5_TARGET_ONLY_CHANNEL_NAMES),
        "width": 6,
        "depth": 2,
        "modes_z": 4,
        "modes_x": 3,
        "lateral_rank": 2,
        "temporal_kernel_size": 3,
        "temporal_dilations": [1, 2],
    }


def _inflate():
    return inflate_causal_checkpoint_payload(
        _source_checkpoint(),
        _target_config(),
        seed=57,
        source_checkpoint_sha256=SOURCE_SHA,
        target_config_sha256=CONFIG_SHA,
    )


def test_causal_inflation_copies_every_source_tensor_and_separates_family() -> None:
    target, payload, report = _inflate()
    source_state = _source_checkpoint()["model"]
    assert isinstance(source_state, dict)
    target_state = target.state_dict()

    assert report["source"]["family"] == "causal_factorized"
    assert report["target"]["family"] == "causal_axis_factorized_2d"
    assert report["target"]["temporal_family"] == (
        "causal_dilated_convolution"
    )
    assert report["pilot_separation"]["shares_noncausal_pilot_weights"] is False
    assert report["tensor_mapping"]["copied_count"] == len(source_state)
    for source_key, source_tensor in source_state.items():
        target_key = CAUSAL_SHARED_LIFT_MAPPING.get(source_key, source_key)
        assert torch.equal(source_tensor, target_state[target_key])
    assert torch.count_nonzero(target.lift.geometry.weight) == 0
    for block in target.blocks:
        assert torch.count_nonzero(block.lateral.input_factor) > 0
        assert torch.count_nonzero(block.lateral.output_factor) == 0
    assert report["verification"]["restriction"]["tested_nx"] == [1, 2, 7, 40]
    assert report["verification"]["restriction"]["passed"] is True
    assert report["verification"]["future_invariance"]["passed"] is True
    assert report["verification"]["architecture"]["temporal_fft_module_count"] == 0
    assert report["integrity"]["all_tensor_hashes_match_embedded_report"]
    assert payload["model_family"] == "causal_axis_factorized_2d"


def test_causal_inflation_fails_closed_on_source_or_config_binding() -> None:
    with pytest.raises(ValueError, match="differs"):
        inflate_causal_checkpoint_payload(
            _source_checkpoint(),
            _target_config(),
            source_checkpoint_sha256="c" * 64,
            target_config_sha256=CONFIG_SHA,
        )
    config = deepcopy(_target_config())
    config["_config_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="binding is inconsistent"):
        inflate_causal_checkpoint_payload(
            _source_checkpoint(),
            config,
            seed=57,
            source_checkpoint_sha256=SOURCE_SHA,
            target_config_sha256=CONFIG_SHA,
        )


def test_causal_integrity_rejects_a_mutated_zero_residual_adapter() -> None:
    _, payload, _ = _inflate()
    mutated = deepcopy(payload)
    output_key = "blocks.0.lateral.output_factor"
    with torch.no_grad():
        mutated["model"][output_key].fill_(0.25)
    with pytest.raises(ValueError, match="deterministic"):
        verify_causal_checkpoint_integrity(
            _source_checkpoint(),
            mutated,
            _target_config(),
            source_checkpoint_sha256=SOURCE_SHA,
            target_config_sha256=CONFIG_SHA,
        )


def test_causal_integrity_rejects_mutated_input_factor_even_with_updated_hash() -> None:
    _, payload, _ = _inflate()
    mutated = deepcopy(payload)
    input_key = "blocks.0.lateral.input_factor"
    with torch.no_grad():
        mutated["model"][input_key][0, 0, 0] += 0.125
    for item in mutated["inflation_report"]["tensor_mapping"]["initialized"]:
        if item["target"] == input_key:
            item["sha256"] = tensor_sha256(mutated["model"][input_key])
            break
    else:
        raise AssertionError("Input factor was absent from the inflation report.")
    with pytest.raises(ValueError, match="deterministic"):
        verify_causal_checkpoint_integrity(
            _source_checkpoint(),
            mutated,
            _target_config(),
            source_checkpoint_sha256=SOURCE_SHA,
            target_config_sha256=CONFIG_SHA,
        )


def test_causal_integrity_rejects_mutated_source_metadata() -> None:
    _, payload, _ = _inflate()
    mutated = deepcopy(payload)
    mutated["normalization"] = {"scope": "mutated"}
    with pytest.raises(ValueError, match="source metadata differs"):
        verify_causal_checkpoint_integrity(
            _source_checkpoint(),
            mutated,
            _target_config(),
            source_checkpoint_sha256=SOURCE_SHA,
            target_config_sha256=CONFIG_SHA,
        )


def test_causal_integrity_rejects_mutated_embedded_verification() -> None:
    _, payload, _ = _inflate()
    mutated = deepcopy(payload)
    mutated["inflation_report"]["verification"]["future_invariance"][
        "maximum_prefix_abs_difference"
    ] = 1.0
    with pytest.raises(ValueError, match="verification evidence differs"):
        verify_causal_checkpoint_integrity(
            _source_checkpoint(),
            mutated,
            _target_config(),
            source_checkpoint_sha256=SOURCE_SHA,
            target_config_sha256=CONFIG_SHA,
        )


def test_causal_inflation_rejects_a_seed_outside_the_frozen_contract() -> None:
    with pytest.raises(ValueError, match="seed differs"):
        inflate_causal_checkpoint_payload(
            _source_checkpoint(),
            _target_config(),
            seed=58,
            source_checkpoint_sha256=SOURCE_SHA,
            target_config_sha256=CONFIG_SHA,
        )


def test_causal_inflation_is_deterministic_and_reloadable(
    tmp_path: Path,
) -> None:
    first, payload, first_report = _inflate()
    torch.manual_seed(999_999)
    second, _, second_report = _inflate()
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
    assert [
        (item["target"], item["sha256"])
        for item in first_report["tensor_mapping"]["initialized"]
    ] == [
        (item["target"], item["sha256"])
        for item in second_report["tensor_mapping"]["initialized"]
    ]
    path = tmp_path / "causal-target.pt"
    torch.save(payload, path)
    reloaded, reloaded_payload = load_inflated_causal_target(path)
    for key, tensor in first.state_dict().items():
        assert torch.equal(tensor, reloaded.state_dict()[key])
    assert reloaded_payload["inflation_report"]["integrity"]["passed"]


def test_causal_transfer_stage_keeps_copied_source_frozen_at_t0() -> None:
    target, _, _ = _inflate()
    summary = target.set_transfer_stage("T0")
    assert summary["trainable_parameters"] > 0
    assert not target.lift.shared.weight.requires_grad
    assert target.lift.geometry.weight.requires_grad
    assert all(
        block.lateral.output_factor.requires_grad for block in target.blocks
    )
    target.set_transfer_stage("T2")
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_causal_gate_dry_run_executes_all_gates_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import verify_p5_causal_restriction as gate

    files = {
        name: tmp_path / name
        for name in (
            "source.pt",
            "target.pt",
            "target.yaml",
            "source.npz",
            "source.json",
            "plan.json",
            "manifest.json",
            "id.json",
        )
    }
    for path in files.values():
        path.touch()
    array_root = tmp_path / "arrays"
    array_root.mkdir()
    output = tmp_path / "must-not-be-written.json"
    embedded = {"inflation_report": {"verification": {"passed": True}}}
    target_model = SimpleNamespace(
        family="causal_axis_factorized_2d",
        temporal_family="causal_dilated_convolution",
    )
    prepared = SimpleNamespace(
        dataset=lambda split: [
            (
                torch.zeros(5, 4, len(INPUT_CHANNELS)),
                None,
                None,
                80,
                0,
            )
        ]
    )
    monkeypatch.setattr(gate, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(gate.torch, "load", lambda *args, **kwargs: embedded)
    monkeypatch.setattr(gate, "load_causal_target_config", lambda path: {})
    monkeypatch.setattr(
        gate,
        "verify_causal_checkpoint_integrity",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        gate,
        "load_causal_source_checkpoint",
        lambda checkpoint: (
            object(),
            {"family": "causal_factorized"},
            {},
        ),
    )
    monkeypatch.setattr(
        gate,
        "load_inflated_causal_target",
        lambda path: (target_model, embedded),
    )
    monkeypatch.setattr(gate, "prepare_source_1d", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        gate,
        "verify_causal_models_on_input",
        lambda *args, **kwargs: {"passed": True, "tested_nx": [1, 2, 7, 40]},
    )
    monkeypatch.setattr(
        gate,
        "verify_target_future_invariance",
        lambda *args, **kwargs: {
            "passed": True,
            "maximum_prefix_abs_difference": 0.0,
        },
    )
    monkeypatch.setattr(
        gate,
        "_verify_p4_f0",
        lambda *args, **kwargs: {
            "case_id": 267,
            "target_only_nonzero_count": 0,
            "passed": True,
        },
    )
    args = SimpleNamespace(
        source_checkpoint=files["source.pt"],
        target_checkpoint=files["target.pt"],
        target_config=files["target.yaml"],
        source_data=files["source.npz"],
        source_split=files["source.json"],
        p4_plan=files["plan.json"],
        p4_manifest=files["manifest.json"],
        p4_id_split=files["id.json"],
        p4_array_root=array_root,
        output=output,
        dry_run=True,
    )

    report = gate.verify(args)

    assert report["passed"] is True
    assert report["dry_run"] is True
    assert report["checkpoint_integrity"]["passed"] is True
    assert report["actual_validation_restriction"]["passed"] is True
    assert report["actual_validation_future_invariance"]["passed"] is True
    assert report["actual_p4_validation_f0"]["passed"] is True
    assert not output.exists()
