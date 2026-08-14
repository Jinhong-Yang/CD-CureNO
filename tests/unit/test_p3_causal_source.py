from pathlib import Path

import torch

from cdcureno.data.source_1d import INPUT_CHANNELS
from cdcureno.models.joint_operators import (
    CausalFactorizedOperator,
    parameter_count,
)
from cdcureno.training.causal_source_1d import (
    CausalSourceTrainConfig,
    _resolve_device,
    causal_receptive_field,
    check_future_invariance,
    load_causal_source_config,
)


def _config(tmp_path: Path) -> CausalSourceTrainConfig:
    return CausalSourceTrainConfig(
        data_path=tmp_path / "source.npz",
        split_manifest=tmp_path / "split.json",
        output_root=tmp_path / "runs",
        project_root=tmp_path,
    )


def test_canonical_causal_source_architecture_matches_frozen_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path).validated()
    model = CausalFactorizedOperator(
        input_channels=config.input_channels,
        width=config.width,
        depth=config.depth,
        modes_space=config.modes_space,
    )

    assert tuple(INPUT_CHANNELS) == (
        "air_temperature_normalized",
        "causal_physics_temperature_baseline_normalized",
        "time_normalized",
        "through_thickness_position_normalized",
        "composite_mask",
        "signed_distance_to_interface_normalized",
        "initial_degree_of_cure",
        "tool_thickness_normalized",
        "composite_thickness_normalized",
        "lower_htc_normalized",
        "upper_htc_normalized",
        "composite_conductivity_scale_normalized",
        "heat_of_reaction_scale_normalized",
        "low_fidelity_degree_of_cure",
    )
    assert parameter_count(model) == 163_270
    assert causal_receptive_field(config.depth) == 511
    assert causal_receptive_field(config.depth) >= 112


def test_future_suffix_cannot_change_any_causal_source_prefix() -> None:
    torch.manual_seed(20260726)
    model = CausalFactorizedOperator(
        input_channels=14,
        width=4,
        depth=2,
        modes_space=3,
    ).eval()
    inputs = torch.rand(2, 7, 6, 14)
    inputs[..., 4] = 1.0
    inputs[:, :, :2, 4] = 0.0
    inputs[..., 6] = 0.05 * inputs[..., 4]

    result = check_future_invariance(
        model,
        inputs,
        [101, 102],
        tolerance=1.0e-7,
        cutoff_fractions=(0.25, 0.5, 0.75),
    )

    assert result["passed"] is True
    assert result["maximum_prefix_abs_difference"] == 0.0
    assert {row["output"] for row in result["rows"]} == {
        "temperature",
        "temperature_residual",
        "alpha",
        "cure_rate",
    }


def test_canonical_yaml_loads_the_frozen_cuda_experiment() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = load_causal_source_config(
        project_root
        / "configs"
        / "experiment"
        / "p3_causal_source_pretrain_v1.yaml",
        project_root=project_root,
    )

    assert config.device == "cuda"
    assert config.width == 34
    assert config.depth == 8
    assert config.modes_space == 12
    assert config.expected_parameter_count == 163_270
    assert config.expected_data_sha256 == (
        "9db82081f71e444265087531369a2293d4b7880db9b183943490862594d10ffa"
    )


def test_unindexed_cuda_request_resolves_to_explicit_device_zero(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert _resolve_device("cuda") == torch.device("cuda:0")
    assert _resolve_device("auto") == torch.device("cuda:0")
